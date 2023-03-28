"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import cv2
from PIL import Image
from .models import compile_model

from .tools import SimpleLoss, get_batch_iou, get_val_info

from .dataloader import get_loader
from .radar_loader import radar_preprocessing
from .train import MyCrossEntropyLoss2d


def model_test(gpuid=0,

               H=600, W=1200,
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

               bsz=1,
               nworkers=10,
               lr=1e-3,
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
        'data_path': './mini1',
        'rotate': False,
        'flip': None,  # if 'hflip', the inverse-projection will ?
        'batch_size': bsz,
        'nworkers': 4,
        'val_batch_size': bsz,
        'nworkers_val': 4,

    }

    torch.backends.cudnn.benchmark = True
    multi_gpu = False

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    model = compile_model(grid_conf, data_aug_conf, outC=2)  # confidence + height
    #######################################
    model_path = './exp/test/model-330.pth'
    # model_path = './exp/20230209_datax1000/model-600_1.4699999999999998e-05.pth'
    if multi_gpu:
        state_dict = torch.load(model_path)['model_state_dict']
        model.load_state_dict(state_dict)
        model = torch.nn.DataParallel(model).cuda(gpuid)
    else:
        state_dict = torch.load(model_path)['model_state_dict']
        model.load_state_dict(state_dict)
        print(model)
        model = model.cuda(gpuid)

    dataset = radar_preprocessing(args['data_path'])
    dataset.prepare_dataset()
    trainloader, validloader = get_loader(args, dataset)

    model.train()
    with torch.no_grad():
        for batchi, (imgs, radars, lidars, lidHts, depths, fovs, objs, calibs) in enumerate(trainloader):
            print(f'#####{batchi:4d}')
            preds, _ = model(imgs.to(device),
                             radars.to(device),
                             calibs.to(device)
                             )
            lidars = lidars.to(device)
            masks = fovs.to(device)
            l_bg = MyCrossEntropyLoss2d(preds[:, 0:2], lidars[:, 0], fovs[:, 0])
            l_fg = MyCrossEntropyLoss2d(preds[:, 0:2], lidars[:, 0], objs[:, 0])
            print(f'background: {l_bg:.4f}, foreground: {l_fg:.4f}')
            # print(torch.max(preds), torch.min(preds))
            p = preds[0, 0:2, :, :].argmax(dim=0).detach().cpu().numpy()
            d = preds[0, 2, :, :].detach().cpu().numpy() * 40
            l = lidars[0, 0, :, :].detach().cpu().numpy()
            ld = lidHts[0, 0, :, :].detach().cpu().numpy() * 40
            m = masks[0, 0, :, :].detach().cpu().numpy()
            t = lidars[0, 0, :, :].detach().cpu().numpy()
            p = p * m
            d = d * m
            t = t * m
            l = l * m
            ld = ld * m

            # PointCloud Visualization
            # print(np.max(d), np.min(d))
            # raise NotImplemented
            print(p.shape)  # 600, 300
            h_list = []
            for i in range(d.shape[0]):
                for j in range(d.shape[1]):
                    h_list.append(d[i, j] - 10)
            # print(set(h_list))
            lh_list = []
            for i in range(ld.shape[0]):
                for j in range(ld.shape[1]):
                    lh_list.append(ld[i, j] - 10)
            print(set(lh_list))
            # raise NotImplemented
            import mayavi.mlab as mlab
            pc_list = []
            # Use GT
            # p = l
            d = ld

            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    if p[i, j] == 1:
                        pc_list.append([j / 4, i / 4 - 75, d[i, j] - 10])
            pc_array = np.asarray(pc_list)
            print(pc_array.shape)
            mlab.points3d(pc_array[:, 0], pc_array[:, 1], pc_array[:, 2],
                          mode='point',
                          colormap='gnuplot', scale_factor=1)

            # draw origin
            mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

            # draw axis
            axes = np.array([
                [2., 0., 0., 0.],
                [0., 2., 0., 0.],
                [0., 0., 2., 0.],
            ], dtype=np.float64)
            mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None,
                        )
            mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None,
                        )
            mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None,
                        )

            # draw fov (todo: update to real sensor spec.)
            fov = np.array([  # 45 degree
                [20., 20., 0., 0.],
                [20., -20., 0., 0.],
            ], dtype=np.float64)

            mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                        )
            mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                        )

            # draw square region
            TOP_Y_MIN = -75
            TOP_Y_MAX = 75
            TOP_X_MIN = 0
            TOP_X_MAX = 75
            TOP_Z_MIN = -2.0
            TOP_Z_MAX = 0.4

            x1 = TOP_X_MIN
            x2 = TOP_X_MAX
            y1 = TOP_Y_MIN
            y2 = TOP_Y_MAX
            mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, )
            mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, )
            mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, )
            mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, )

            mlab.show()
            # raise NotImplemented

            p1 = np.uint8((1 - p) * 255).T[::-1, :]
            t1 = np.uint8((1 - t) * 255).T[::-1, :]
            im = np.vstack([p1, t1])
            # Image.fromarray(im).save(f'results/20230306_datax2000/val/{batchi}.png')
            Image.fromarray(im).show()
            print('height range: ', np.min(d), np.max(d))
            # break
