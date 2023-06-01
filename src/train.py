"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
from collections import OrderedDict

from .models import compile_model
from .tools import SimpleLoss, get_batch_iou, get_val_info

from .dataloader import get_loader
from .radar_loader import radar_preprocessing
from .gan import Discriminator


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets.long())


def MyCrossEntropyLoss2d(output, target, mask):
    '''
    output: bsz, channels, height, width
    target: bsz, height, width
    mask: bsz, height, width
    '''
    assert target.shape == mask.shape, ''
    log_softmax = torch.nn.LogSoftmax(dim=1)(output)
    bsz, h, w = target.shape
    loss = 0
    for b in range(bsz):
        ind = target[b, :, :].type(torch.int64).unsqueeze(0)
        pred = log_softmax[b, :, :, :]
        pvalue = -pred.gather(0, ind)
        msk = (mask[b:b + 1, :, :] > 0).detach()
        if pvalue[msk].shape[0] > 0:
            loss = loss + torch.mean(pvalue[msk])
        else:
            bsz = bsz - 1
    if bsz == 0:
        return torch.mean(pvalue) * 0
    return loss / bsz


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, prediction, gt):
        err = prediction - gt
        mask = (gt > 0).detach()
        mse_loss = torch.mean((err[mask]) ** 2)
        return mse_loss


class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, ypred, ytgt, mask):
        loss = self.loss_fn(ypred * mask, ytgt * mask)
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(gpuid=0,

          H=512, W=1024,
          resize_lim=(0.193, 0.225),
          final_dim=(512, 1024),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=1,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',

          xbound=[-75.0, 75.0, 0.25],
          ybound=[0.0, 75.0, 0.25],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 75.0, 1.0],

          bsz=4,
          nworkers=10,
          lr=0.00005,
          weight_decay=1e-7,
          ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    # INIT datasetd
    args = {
        # img -> (600, 1200), others -> (301, 601)
        'crop_h': 600,
        'crop_w': 300,
        'no_aug': None,
        'data_path': './dataset_direct',
        'rotate': False,
        'flip': None,  # if 'hflip', the inverse-projection will ?
        'batch_size': bsz,
        'nworkers': 4,
        'val_batch_size': bsz,
        'nworkers_val': 4,

    }

    torch.backends.cudnn.benchmark = True
    multi_gpu = True
    pre_train = False
    pre_train_path = 'exp/20230215_datax2000/model-80_0.001.pth'
    resume = False
    resume_path = 'exp/20230306_datax2000_height/model-120.pth'
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    model = compile_model(grid_conf, data_aug_conf, outC=2).cuda()  # confidence + height


    dataset = radar_preprocessing(args['data_path'])
    dataset.prepare_dataset()
    trainloader, validloader = get_loader(args, dataset)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #
    # lr_policy = 'plateau'
    # if lr_policy == 'plateau':
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
    #                                                            factor=0.7,
    #                                                            threshold=0.000001,
    #                                                            patience=13)
    #
    # if lr_policy == 'step':
    #     scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    epoch = 0
    if pre_train:
        checkpoint = torch.load(pre_train_path)
        state_dict = checkpoint['model_state_dict']
        if multi_gpu:
            model.load_state_dict(state_dict)
            model = torch.nn.DataParallel(model).cuda(gpuid)
        else:
            model.load_state_dict(state_dict)
            model = model.cuda(gpuid)
    elif resume:
        checkpoint = torch.load(resume_path)
        state_dict = checkpoint['model_state_dict']
        opt.load_state_dict(checkpoint['opt_state_dict'])
        epoch = checkpoint['epoch']
        if multi_gpu:
            model.load_state_dict(state_dict)
            model = torch.nn.DataParallel(model).cuda(gpuid)
        else:
            model.load_state_dict(state_dict)
            model = model.cuda(gpuid)
    else:
        if multi_gpu:
            model = torch.nn.DataParallel(model).cuda(gpuid)
        else:
            model = model.cuda(gpuid)

    lr_policy = 'plateau'
    if lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                               factor=0.7,
                                                               threshold=0.000001,
                                                               patience=30)

    if lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    loss_bce = torch.nn.BCEWithLogitsLoss().cuda(gpuid)
    l_mse = MSELoss().cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)

    model.train()
    counter = 0
    while True:
        np.random.seed()

        # AverageMeter()
        losses = AverageMeter()
        bg_loss = AverageMeter()
        fg_loss = AverageMeter()
        ht_loss = AverageMeter()
        dp_loss = AverageMeter()

        for batchi, (imgs, radars, lidars, lidHts, depths, fovs, objs, calibs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds, dis = model(imgs.to(device),
                               radars.to(device),
                               calibs.to(device)
                               )
            lidars = lidars.to(device)
            lidHts = lidHts.to(device)
            depths = depths.to(device)
            fovs = fovs.to(device)
            objs = objs.to(device)

            l_bg = MyCrossEntropyLoss2d(preds[:, 0:2], lidars[:, 0], fovs[:, 0])
            l_fg = MyCrossEntropyLoss2d(preds[:, 0:2], lidars[:, 0], objs[:, 0])
            l_ht = l_mse(preds[:, 2:3] * fovs, lidHts * fovs) * 20
            l_dp = l_mse(dis[:, 0:1] / 75, depths)
            # argmax does not have a grad_fn
            # -> softmax
            pred_bg = torch.nn.functional.softmax(preds[:, 0:2], dim=1)
            l_bg2 = l_mse((pred_bg[:, 1:2] - pred_bg[:, 0:1]) * fovs, lidars * fovs)
            # loss = l_bg + l_fg + l_ht + l_dp
            loss = l_bg + l_fg + l_ht
            # AverageMeter()
            bg_loss.update(l_bg.item(), imgs.size(0))
            fg_loss.update(l_fg.item(), imgs.size(0))
            ht_loss.update(l_ht.item(), imgs.size(0))
            dp_loss.update(l_dp.item(), imgs.size(0))
            losses.update(loss.item(), imgs.size(0))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(f'epoch-{epoch:3d}|{counter}, '
                      f'l_bg:{bg_loss.avg:.4f}, l_fg:{fg_loss.avg:.4f}, '
                      f'height:{ht_loss.avg:.4f}, depth:{dp_loss.avg:.4f}, '
                      f'total:{losses.avg:.4f}')
                writer.add_scalar('train/loss', loss, counter)
                writer.add_scalar('train/l_bg', l_bg, counter)
                writer.add_scalar('train/l_fg', l_fg, counter)
                writer.add_scalar('train/l_dp', l_dp, counter)
                writer.add_scalar('train/l_ht', l_ht, counter)
                writer.add_scalar('train/lr', opt.param_groups[0]['lr'], counter)

        # LR plateaued
        if lr_policy == 'plateau':
            scheduler.step(losses.avg)
            print('LR plateaued, hence is set to {}'.format(opt.param_groups[0]['lr']))
        else:
            scheduler.step()
            print('LR plateaued, hence is set to {}'.format(opt.param_groups[0]['lr']))

        _, _, iou = get_batch_iou(preds[:, 0:2].argmax(dim=1).unsqueeze(1).long(), lidars, fovs)
        writer.add_scalar('train/iou', iou, counter)
        writer.add_scalar('train/epoch', epoch, counter)
        writer.add_scalar('train/step_time', t1 - t0, counter)

        if epoch % 50 == 0 and epoch > 0:
            model.eval()
            print(opt)
            mname = os.path.join(logdir, "model-{}.pth".format(epoch))
            print('saving', mname)
            if multi_gpu:
                checkpoint = {'model_state_dict': model.module.state_dict(),
                              'opt_state_dict': opt.state_dict(),
                              'epoch': epoch}
            else:
                checkpoint = {'model_state_dict': model.state_dict(),
                              'opt_state_dict': opt.state_dict(),
                              'epoch': epoch}
            torch.save(checkpoint, mname)
            model.train()

        epoch = epoch + 1
