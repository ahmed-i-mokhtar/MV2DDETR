import os
import json
import time
from operator import itemgetter
import copy

import numpy as np
from PIL import Image
import kornia
from torchvision.datasets import VisionDataset
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from multiview_detector.utils.projection import *
from multiview_detector.utils.image_utils import draw_umich_gaussian, random_affine
import matplotlib.pyplot as plt


def get_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    reg_mask = np.zeros([top_k], dtype=np.bool)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, reID=False, world_reduce=4, img_reduce=12,
                 world_kernel_size=10, img_kernel_size=10,
                 train_ratio=0.9, top_k=100, force_download=True,
                 semi_supervised=0.0, dropout=0.0, augmentation=False):
        super().__init__(base.root)

        self.base = base
        self.num_cam, self.num_frame = base.num_cam, base.num_frame
        self.img_reduce = img_reduce
        self.img_shape = base.img_shape # H,W; N_row,N_col 
        self.img_kernel_size = img_kernel_size 
        self.semi_supervised = semi_supervised * train
        self.dropout = dropout
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    T.Resize((np.array(self.img_shape) * 8 // self.img_reduce).tolist())])
        self.augmentation = augmentation
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()
        self.top_k = top_k
        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)


        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.imgs_gt = {}
        self.pid_dict = {}
        self.keeps = {}
        num_frame, num_imgs_bbox = 0, 0
        num_keep, num_all = 0, 0
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                keep = np.mean(np.array(frame_range) < frame) < self.semi_supervised if self.semi_supervised else 1
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_objects = json.load(json_file)
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                if keep:
                    for obj in all_objects:
                        if obj['objectID'] not in self.pid_dict:
                            self.pid_dict[obj['objectID']] = len(self.pid_dict)
                        num_all += 1
                        num_keep += keep
                        for cam in range(self.num_cam):
                            if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(obj['views'][cam]) != (-1, -1, -1, -1):
                                img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                      (obj['views'][cam]))
                                img_pids[cam].append(self.pid_dict[obj['objectID']])
                                num_imgs_bbox += 1
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))
                self.keeps[frame] = keep

        print(f'all: pid: {len(self.pid_dict)}, frame: {num_frame}, keep ratio: {num_keep / num_all:.3f}\n'
              f'imgs bbox per cam: {num_imgs_bbox / num_frame / self.num_cam:.1f}')
        # # gt in mot format for evaluation
        # self.gt_fpath = os.path.join(self.root, 'gt.txt')
        # if not os.path.exists(self.gt_fpath) or force_download:
        #     self.prepare_gt()



    # def prepare_gt(self):
    #     og_gt = []
    #     for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
    #         frame = int(fname.split('.')[0])
    #         with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
    #             all_objects = json.load(json_file)
    #         for single_pedestrian in all_objects:
    #             def is_in_cam(cam):
    #                 return not (single_pedestrian['views'][cam]['xmin'] == -1 and
    #                             single_pedestrian['views'][cam]['xmax'] == -1 and
    #                             single_pedestrian['views'][cam]['ymin'] == -1 and
    #                             single_pedestrian['views'][cam]['ymax'] == -1)

    #             in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
    #             if not in_cam_range:
    #                 continue
    #             grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID']).squeeze()
    #             og_gt.append(np.array([frame, grid_x, grid_y]))
    #     og_gt = np.stack(og_gt, axis=0)
    #     os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
    #     np.savetxt(self.gt_fpath, og_gt, '%d')

    def __getitem__(self, index, visualize=False):
        def plt_visualize():
            import cv2
            from matplotlib.patches import Circle
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for i in range(len(img_x_s)):
                x, y = img_x_s[i], img_y_s[i]
                if x > 0 and y > 0:
                    ax.add_patch(Circle((x, y), 10))
            plt.show()
            img0 = img.copy()
            for bbox in img_bboxs:
                bbox = tuple(int(pt) for pt in bbox)
                cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            plt.imshow(img0)
            plt.show()

        frame = list(self.imgs_gt.keys())[index]
        # imgs
        imgs, imgs_gt, affine_mats, masks = [], [], [], []
        for cam in range(self.num_cam):
            img = np.array(Image.open(self.img_fpaths[cam][frame]).convert('RGB'))
            img_bboxs, img_pids = self.imgs_gt[frame][cam]
            if self.augmentation:
                img, img_bboxs, img_pids, M = random_affine(img, img_bboxs, img_pids)
            else:
                M = np.eye(3)
            imgs.append(self.transform(img))
            affine_mats.append(torch.from_numpy(M).float())
            img_x_s, img_y_s = (img_bboxs[:, 0] + img_bboxs[:, 2]) / 2, img_bboxs[:, 3]
            img_w_s, img_h_s = (img_bboxs[:, 2] - img_bboxs[:, 0]), (img_bboxs[:, 3] - img_bboxs[:, 1])

            img_gt = get_gt(self.Rimg_shape, img_x_s, img_y_s, img_w_s, img_h_s, v_s=img_pids,
                            reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            imgs_gt.append(img_gt)
            if visualize:
                plt_visualize()

        imgs = torch.stack(imgs)
        affine_mats = torch.stack(affine_mats)
        # inverse_M = torch.inverse(
        #     torch.cat([affine_mats, torch.tensor([0, 0, 1]).view(1, 1, 3).repeat(self.num_cam, 1, 1)], dim=1))[:, :2]
        imgs_gt = {key: torch.stack([img_gt[key] for img_gt in imgs_gt]) for key in imgs_gt[0]}
        # imgs_gt['heatmap_mask'] = self.imgs_region if self.keeps[frame] else torch.zeros_like(self.imgs_region)
        # imgs_gt['heatmap_mask'] = kornia.warp_perspective(imgs_gt['heatmap_mask'], affine_mats, self.img_shape,
        #                                                   align_corners=False)
        # imgs_gt['heatmap_mask'] = F.interpolate(imgs_gt['heatmap_mask'], self.Rimg_shape, mode='bilinear',
        #                                         align_corners=False).bool().float()
        drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
        if drop:
            drop_cam = np.random.randint(0, self.num_cam)
            keep_cams[drop_cam] = 0
            for key in imgs_gt:
                imgs_gt[key][drop_cam] = 0

        return imgs, imgs_gt, affine_mats, frame

    def __len__(self):
        return len(self.imgs_gt.keys())


