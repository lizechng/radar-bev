import os

import cv2
import numpy as np

from PIL import Image
from scipy.signal import convolve
from scipy.stats import multivariate_normal


def guassian_blur(rin):
    hm_h, hm_w = rin.shape

    coord = np.stack((
        np.linspace(0, hm_w - 1, hm_w, dtype=np.int64).reshape(1, hm_w).repeat(hm_h, 0),
        np.linspace(0, hm_h - 1, hm_h, dtype=np.int64).reshape(hm_h, 1).repeat(hm_w, 1)
    ), -1)

    hm = rin
    # Gaussian Blur for Image
    border = 2
    kernel = 5
    origin_max = np.max(hm)
    # print(origin_max)
    dr = np.zeros((hm_h + 2 * border, hm_w + 2 * border))
    dr[border:-border, border:-border] = hm.copy()
    dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
    hm = dr[border:-border, border:-border].copy()
    hm *= origin_max / np.max(hm)
    return hm


def taylor(rin):
    # is Peak?
    hm_h, hm_w = rin.shape

    coord = np.stack((
        np.linspace(0, hm_w - 1, hm_w, dtype=np.int64).reshape(1, hm_w).repeat(hm_h, 0),
        np.linspace(0, hm_h - 1, hm_h, dtype=np.int64).reshape(hm_h, 1).repeat(hm_w, 1)
    ), -1)

    hm = rin

    l = 8
    r = 9
    mean = [0, 0]
    cov = [[l, 0], [0, r]]
    x_, y_ = np.mgrid[-l:r:1, -l:r:1]
    pos = np.dstack((x_, y_))
    rv = multivariate_normal(mean, cov)
    kernel = rv.pdf(pos)
    # kernel = np.where(kernel < 0.5, 1, kernel)
    # 5x5 grids
    for py in np.arange(l, hm_h - r, (l + r) // 2):
        for px in np.arange(l, hm_w - r, (l + r) // 2):
            # Peak at Center
            if hm[py, px] == np.max(hm[py - 2:py + 3, px - 2:px + 3]) and hm[py, px] >= 40:
                dx = 0.5 * (hm[py, px + 1] - hm[py, px - 1])
                dy = 0.5 * (hm[py + 1, px] - hm[py - 1, px])
                dxx = 0.25 * (hm[py, px + 2] - 2 * hm[py, px] + hm[py, px - 2])
                dxy = 0.25 * (hm[py + 1, px + 1] - hm[py - 1, px + 1] - hm[py + 1, px - 1] \
                              + hm[py - 1, px - 1])
                dyy = 0.25 * (hm[py + 2 * 1, px] - 2 * hm[py, px] + hm[py - 2 * 1, px])

                hessian_value = abs(dxx + dyy + 2 * dxy) / 4

                if hessian_value > 255 / 220:
                    hessian_value = 255 / 220
                # query = np.zeros((l + r, l + r))
                template = hm[py - l:py + r, px - l:px + r]
                query = (template * kernel) / np.max(template * kernel) * 220 * hessian_value
                hm[py - l:py + r, px - l:px + r] = np.where(query < np.max(template), template, query)

    return hm

for type in ['train', 'val']:
    pth = f'./dataset_log/{type}/radar'
    files = os.listdir(pth)
    files = sorted(files)
    for file in files:
        print(file)
        rin = Image.open(os.path.join(pth, file))
        rin = np.asarray(rin, dtype=np.int64)
        rin = guassian_blur(rin)
        res = taylor(rin)
        res = res / np.max(res) * 255
        Image.fromarray(np.uint8(res)).save(f'./dataset_log/{type}/hessian/{file}')

