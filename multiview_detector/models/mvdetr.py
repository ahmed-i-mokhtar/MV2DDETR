import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg11
import torchvision.transforms as T
import kornia
from multiview_detector.models.resnet import *
from multiview_detector.utils.image_utils import img_color_denormalize, array2heatmap
# from multiview_detector.utils.projection import get_worldcoord_from_imgcoord_mat, project_2d_points
# from multiview_detector.models.conv_world_feat import ConvWorldFeat, DeformConvWorldFeat
# from multiview_detector.models.trans_world_feat import TransformerWorldFeat, DeformTransWorldFeat, \
#     DeformTransWorldFeat_aio
import matplotlib.pyplot as plt


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def output_head(in_dim, feat_dim, out_dim):
    if feat_dim:
        fc = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(feat_dim, out_dim, 1))
    else:
        fc = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
    return fc





class MVDeTr(nn.Module):
    def __init__(self, dataset, arch='resnet18', z=0,
                 bottleneck_dim=128, outfeat_dim=64, droupout=0.5):
        super().__init__()
        self.Rimg_shape = dataset.Rimg_shape
        self.img_reduce = dataset.img_reduce

        if arch == 'vgg11':
            self.base = vgg11(pretrained=True).features
            self.base[-1] = nn.Identity()
            self.base[-4] = nn.Identity()
            base_dim = 512
        elif arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[:-2])
            base_dim = 512
        elif arch == 'resnet50':
            self.base = nn.Sequential(*list(resnet50(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[:-2])
            base_dim = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        if bottleneck_dim:
            self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, bottleneck_dim, 1), nn.Dropout2d(droupout))
            base_dim = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()

        # img heads
        self.img_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.img_offset = output_head(base_dim, outfeat_dim, 2)
        self.img_wh = output_head(base_dim, outfeat_dim, 2)
        # self.img_id = output_head(base_dim, outfeat_dim, len(dataset.pid_dict))

        # init
        self.img_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.img_offset)
        fill_fc_weights(self.img_wh)
        

    def forward(self, imgs, M, visualize=True):
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)

        inverse_affine_mats = torch.inverse(M.view([B * N, 3, 3]))
        imgcoord_from_Rimggrid_mat = inverse_affine_mats @ \
                                     torch.from_numpy(np.diag([self.img_reduce, self.img_reduce, 1])
                                                      ).view(1, 3, 3).repeat(B * N, 1, 1).float()

        imgs_feat = self.base(imgs)
        imgs_feat = self.bottleneck(imgs_feat)
        if visualize:
            for cam in range(N):
                visualize_img = array2heatmap(torch.norm(imgs_feat[cam * B].detach(), dim=0).cpu())
                visualize_img.save(f'/home/amokhtar/Research/Multiview/MVDeTr/imgs/augimgfeat{cam + 1}.png')
                #plt.imshow(visualize_img)
                #plt.show()

        # img heads
        _, C, H, W = imgs_feat.shape
        imgs_heatmap = self.img_heatmap(imgs_feat)
        imgs_offset = self.img_offset(imgs_feat)
        imgs_wh = self.img_wh(imgs_feat)

   
        return (imgs_heatmap, imgs_offset, imgs_wh)


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from multiview_detector.utils.decode import ctdet_decode

    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=False, augmentation=False)
    # create_reference_map(dataset, 4)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    model = MVDeTr(dataset, world_feat_arch='deform_trans').cuda()
    # model.load_state_dict(torch.load(
    #     '../../logs/wildtrack/augFCS_deform_trans_lr0.001_baseR0.1_neck128_out64_alpha1.0_id0_drop0.5_dropcam0.0_worldRK4_10_imgRK12_10_2021-04-09_22-39-28/MultiviewDetector.pth'))
    imgs, world_gt, imgs_gt, affine_mats, frame = next(iter(dataloader))
    imgs = imgs.cuda()
    (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(imgs, affine_mats)
    xysc = ctdet_decode(world_heatmap, world_offset)
    pass


if __name__ == '__main__':
    test()
