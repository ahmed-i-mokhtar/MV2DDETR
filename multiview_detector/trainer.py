import time
import os
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from PIL import Image
from multiview_detector.loss import *
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image, img_color_denormalize


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, logdir, cls_thres=0.4, alpha=1.0, use_mse=True, id_ratio=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss()
        self.regress_loss = RegL1Loss()
        self.ce_loss = RegCELoss()
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.alpha = alpha
        self.use_mse = use_mse
        self.id_ratio = id_ratio

    def train(self, epoch, dataloader, optimizer, scaler, scheduler=None, log_interval=100):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, imgs_gt, affine_mats, frame) in enumerate(dataloader): 
            B, N = imgs_gt['heatmap'].shape[:2]
            data = data.cuda()
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
            # with autocast():
            # supervised
            (imgs_heatmap, imgs_offset, imgs_wh) = self.model(data, affine_mats)
            loss_img_hm = self.focal_loss(imgs_heatmap, imgs_gt['heatmap'])
            loss_img_off = self.regress_loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
            loss_img_wh = self.regress_loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
            # loss_img_id = self.ce_loss(imgs_id, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['pid'])
            # multiview regularization

          
            img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1  # + self.id_ratio * loss_img_id
            loss = img_loss / N * self.alpha 
            if self.use_mse:
                loss = self.alpha * self.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device)) 

            t_f = time.time()
            t_forward += t_f - t_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            losses += loss.item()

            t_b = time.time()
            t_backward += t_b - t_f

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))
            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.6f}, '
                      f'Time: {t_epoch:.1f}, maxima: {imgs_heatmap.max():.3f}')
                pass
        return losses / len(dataloader)

    def test(self, epoch, dataloader, res_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        res_list = []
        t0 = time.time()
        for batch_idx, (data, imgs_gt, affine_mats, frame) in enumerate(dataloader):
            B, N = imgs_gt['heatmap'].shape[:2]
            data = data.cuda()
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
            # with autocast():
            with torch.no_grad():
                (imgs_heatmap, imgs_offset, imgs_wh) = self.model(data, affine_mats)
                loss_img_hm = self.focal_loss(imgs_heatmap, imgs_gt['heatmap'])
                loss_img_off = self.regress_loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
                loss_img_wh = self.regress_loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
                # loss_img_id = self.ce_loss(imgs_id, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['pid'])
                # multiview regularization

            
                img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1  # + self.id_ratio * loss_img_id
                loss = img_loss / N * self.alpha 
                if self.use_mse:
                    loss = self.alpha * self.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device))

            losses += loss.item()


        t1 = time.time()
        t_epoch = t1 - t0

        moda = 0

        print(f'Test, loss: {losses / len(dataloader):.6f}, Time: {t_epoch:.3f}')

        return losses / len(dataloader), moda
