import numpy as np
import os
from PIL import Image

files = os.listdir('../rad/train/radbev')
files = sorted(files)

for file in files:
    name = file[:-4]
    fov = Image.open(f'../dataset_direct/train/fov_mask/20211025_1_group0005_frame0000_1635145120_915.png')
    bev = Image.open(f'../rad/train/radbev/{name}.png')
    height = Image.open(f'../rad/train/radHt/{name}.tiff')

    fov = np.asarray(fov) / 255
    bev = np.asarray(bev) / 255
    height = np.asarray(height) / 40
    print(np.min(fov), np.max(fov))
    print(np.min(bev), np.max(bev))
    print(np.min(height), np.max(height))

    p = bev * fov
    d = height * fov * 40

    pc_list = []
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i, j] == 1:
                # x, y, z
                if d[i, j] == 0:
                    continue
                pc_list.append([i / 4 - 75, j / 4, d[i, j] - 10])
    pred_pts = np.asarray(pc_list)
    print(pred_pts.shape)
    pred_pts = pred_pts[np.argsort(pred_pts[:, 1])[::-1], :]
    np.save(f'pts/radar/{name}.npy', pred_pts)
