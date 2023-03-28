"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
import re
import warnings
import json

import torch
import numpy as np
import mayavi.mlab as mlab
from PIL import Image
from scipy.linalg import inv

from .gdc import GDC
from .models import compile_model
from .dataloader import get_loader
from .radar_loader import radar_preprocessing
from .train import MyCrossEntropyLoss2d

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_header(lines):
    '''Parse header of PCD files'''
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn(f'warning: cannot understand line: {ln}')
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()

    if 'count' not in metadata:
        metadata['count'] = [1] * len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def _build_dtype(metadata):
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type] * c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_ascii_pc_data(f, dtype, metadata):
    # for radar point
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    # for lidar point
    rowstep = metadata['points'] * dtype.itemsize
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    raise NotImplemented


def read_pcd(pcd_path):
    f = open(pcd_path, 'rb')
    header = []
    while True:
        ln = f.readline().strip()  # ln is bytes
        ln = str(ln, encoding='utf-8')
        header.append(ln)
        # print(type(ln), ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or "binary_compressed"')

    points = np.concatenate([pc_data[metadata['fields'][0]][:, None],
                             pc_data[metadata['fields'][1]][:, None],
                             pc_data[metadata['fields'][2]][:, None],
                             pc_data[metadata['fields'][3]][:, None]], axis=-1)

    return points


def pts2camera(pts, matrix):
    n = pts.shape[0]
    pts = np.hstack((pts, np.ones((n, 1))))
    # print(pts.shape, matrix.shape)
    pts_2d = np.dot(pts, np.transpose(matrix))
    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2]
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2]
    mask = (pts_2d[:, 0] < 3517) & (pts_2d[:, 1] < 1700) & \
           (pts_2d[:, 0] > 0) & (pts_2d[:, 1] > 0) & \
           (pts_2d[:, 2] > 0) & (pts_2d[:, 2] < 75)
    pts_2d = pts_2d[mask, :]
    # print(pts_2d[:, 0].shape)
    x = np.asarray(pts_2d[:, 0] / 3517 * 512, dtype=np.int32)
    y = np.asarray(pts_2d[:, 1] / 1700 * 256, dtype=np.int32)
    v = np.asarray(pts_2d[:, 2], dtype=np.uint8)
    im = np.zeros([256, 512], dtype=np.uint8)
    im[y, x] = v

    return im


def knn_test(gpuid=0,

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

            # TI Radar Points
            pts = read_pcd('/media/personal_data/lizc/2023/radar-bev/ti.pcd')
            pts = pts[np.argsort(pts[:, 1])[::-1], :]
            calibs = calibs[0, 0, :, :].detach().cpu().numpy()
            # print(calibs, inv(calibs))
            # raise NotImplemented
            ti_depth = pts2camera(pts[:, :3], inv(calibs)[:3, :])
            # Image.fromarray(np.uint8(ti_depth*255)).show()
            # PointCloud Visualization
            pc_list = []
            # Use GT
            # p = l
            d = ld

            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    if p[i, j] == 1:
                        # x, y, z
                        pc_list.append([i / 4 - 75, j / 4, d[i, j] - 10])
            pred_pts = np.asarray(pc_list)
            pred_pts = pred_pts[np.argsort(pred_pts[:, 1])[::-1], :]
            pred_depth = pts2camera(pred_pts[:, :3], inv(calibs)[:3, :])


            # KNN Graph
            knn_pts = GDC(pred_depth, ti_depth, calibs)


            # Draw KNN-Points
            # mlab.points3d(knn_pts[:, 0], knn_pts[:, 1], knn_pts[:, 2],
            #               # np.sqrt(pred_pts[:, 0] ** 2 + pred_pts[:, 1] ** 2),
            #               mode='point',
            #               colormap='gnuplot', scale_factor=1)

            # Draw Points
            mlab.points3d(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2],
                          np.sqrt(pred_pts[:, 0] ** 2 + pred_pts[:, 1] ** 2),
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
                [-20., 20., 0., 0.],
            ], dtype=np.float64)

            mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                        )
            mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                        )

            # draw square region
            TOP_Y_MIN = 0
            TOP_Y_MAX = 75
            TOP_X_MIN = -75
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

            p1 = np.uint8((1 - p) * 255).T[::-1, :]
            t1 = np.uint8((1 - t) * 255).T[::-1, :]
            im = np.vstack([p1, t1])
            Image.fromarray(im).show()
            print('height range: ', np.min(d), np.max(d))
            # break
