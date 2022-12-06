"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import numpy as np
from scipy.linalg import pinv, inv
from tools import gen_dx_bx, cumsum_trick, QuickCumsum
from PIL import Image


class L(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(L, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        # print(self.grid_conf)
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16  # The downsample rate we use should be ?
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

        # inverse matrix
        matrix = np.asarray([[2019.613635, 1745.881668, -111.4337968, -419.9388818],
                             [26.01936737, 870.7969811, -2038.300785, -120.9971104],
                             [0.02443084799, 0.997614078, -0.06457000164, -0.006415358346]])
        m = np.vstack([matrix, np.asarray([0, 0, 0, 1])])
        self.inv = torch.Tensor(inv(m))

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 128, 352
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 8, 22
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH,
                                                                                      fW)  # from 0 to 351, 22 bins
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH,
                                                                                      fW)  # from 0 to 128, 8 bins
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, x):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _, h_, w_ = x.shape
        # rots: 3x3 ,trans: 1x3, intrins: 3x3, post_rots:3x3, post_trans: 1x1
        # undo post-transformation
        # B x N x D x H x W x 3
        # self.frustum is 41x8x22x3
        D, H, W, _ = self.frustum.shape
        # points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)  # points' shape will be 1x1x41x8x22x3
        points = self.frustum.repeat(B, N, 1, 1, 1, 1)
        # points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(
        #     points.unsqueeze(-1))  # insert dimension at position '-1'
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :1] * points[:, :, :, :, :, 2:3] * 1700 / 512,
                            points[:, :, :, :, :, 1:2] * points[:, :, :, :, :, 2:3] * 3517 / 1024,
                            points[:, :, :, :, :, 2:3],
                            torch.ones_like(points[:, :, :, :, :, 2:3])
                            ), 5)
        combine = self.inv.to(x.device)
        # calibs = calibs.unsqueeze(2).unsqueeze(2).transpose(-1, -2)
        combine = combine.repeat(B, N, 1, 1, 1, 1).transpose(-1, -2)
        # points = combine.repeat(B, N, 1, 1, 1, 1, 1).matmul(points).squeeze(-1)
        points = points.matmul(combine)
        # points += trans.view(B, N, 1, 1, 1, 3)  # points: batch, ncams, 41, 8, 22, 3, under XYZ coordinates
        return points[..., :3]

    # def get_cam_feats(self, x):
    #     """Return B x N x D x H/downsample x W/downsample x C
    #     """
    #     B, N, C, imH, imW = x.shape
    #
    #     x = x.view(B * N, C, imH, imW)
    #     x = self.camencode(x)  # shape: 512/16 1024/16
    #     # print(x.shape) # 4 64, 71, 32, 64
    #     x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
    #     x = x.permute(0, 1, 3, 4, 5, 2)  # camera features: batch, ncams, 41, 8, 22, 64
    #
    #     return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape  # D is distance, from 4m to 45m

        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        # geom_feats is positions under XYZ coordinate
        # print(torch.min(geom_feats[..., 0]), torch.max(geom_feats[..., 0])) # from 5.67 to 45.89
        # print(torch.min(geom_feats[..., 1]), torch.max(geom_feats[..., 1]))  # from -26.76 to 28.65
        # print(torch.min(geom_feats[..., 2]), torch.max(geom_feats[..., 2]))  # from -11.99 to 9.00
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        # print(geom_feats.shape)
        # print(torch.min(geom_feats[..., 0]), torch.max(geom_feats[..., 0])) # from 111 to 192
        # print(torch.min(geom_feats[..., 1]), torch.max(geom_feats[..., 1])) # from 46 to 159
        # print(torch.min(geom_feats[..., 2]), torch.max(geom_feats[..., 2])) # from 0 to 0
        # print(self.bx) # tensor([-49.7500, -49.7500,   0.0000], device='cuda:1')
        # print(self.dx) # tensor([ 0.5000,  0.5000, 20.0000], device='cuda:1')
        # i think, frustum is under image coordinate, thus, HxWxD
        # geom_feats is under XYZ coordinate, thus, 200x200 in BEV
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # print(self.nx) # tensor([200, 200,   1], device='cuda:1')
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()  # from small to big
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        # cumsum trick
        # camera points decrease rapidly
        # if not self.use_quickcumsum:
        #     x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        # else:
        #     x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        # cumsum trick: |----point A----| the feature is the sum of the feature of points whose position is A
        # print(geom_feats.shape) # shape: 1266, 4
        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x):
        geom = self.get_geometry(x)

        x = x.permute(0, 1, 3, 4, 2) # 1, 1, 32, 64, 3
        x = x.unsqueeze(0).repeat(1, 1, 71, 1, 1, 1)

        # dis = torch.zeros((1, 1, 71, 512 // self.downsample, 1024 // self.downsample)).to(x.device)
        # for i in range(512 // self.downsample):
        #     for j in range(1024 // self.downsample):
        #         idx = int(np.random.rand()*71)
        #         dis[0, 0, idx, i, j] = 1
        # x = x.permute(0, 2, 1, 3, 4)
        # x = dis * x
        # x = x.view(1, 1, 3, 71, 512 // self.downsample, 1024 // self.downsample)
        # x = x.permute(0, 1, 3, 4, 5, 2)  # camera features: batch, ncams, 41, 8, 22, 64

        x = self.voxel_pooling(geom, x)

        return x

    def taylor(self, hm, rinf):
        mask = rinf[:, 0:1]
        peak = rinf[:, 1:3].unsqueeze(-1).permute(0, 4, 2, 3, 1)
        hm = hm * mask
        B, C, hm_h, hm_w = hm.shape

        px = torch.linspace(2, hm_w - 3, hm_w - 4, dtype=torch.long).view(1, hm_w - 4).expand(hm_h - 4, hm_w - 4)
        py = torch.linspace(2, hm_h - 3, hm_h - 4, dtype=torch.long).view(hm_h - 4, 1).expand(hm_h - 4, hm_w - 4)

        coord = torch.stack((
            torch.linspace(0, hm_w - 1, hm_w, dtype=torch.long).view(1, hm_w).expand(hm_h, hm_w),
            torch.linspace(0, hm_h - 1, hm_h, dtype=torch.long).view(hm_h, 1).expand(hm_h, hm_w)
        ), -1).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1, 1).to(hm.device)

        dx = 0.5 * (hm[:, :, py, px + 1] - hm[:, :, py, px - 1])
        dy = 0.5 * (hm[:, :, py + 1, px] - hm[:, :, py - 1, px])
        dxx = 0.25 * (hm[:, :, py, px + 2] - 2 * hm[:, :, py, px] + hm[:, :, py, px - 2])
        dxy = 0.25 * (hm[:, :, py + 1, px + 1] - hm[:, :, py - 1, px + 1] - hm[:, :, py + 1, px - 1] \
                      + hm[:, :, py - 1, px - 1])
        dyy = 0.25 * (hm[:, :, py + 2 * 1, px] - 2 * hm[:, :, py, px] + hm[:, :, py - 2 * 1, px])

        derivative = torch.stack([dx, dy], -1).view(B, C, hm_h - 4, hm_w - 4, 2, 1)
        h1 = torch.stack([dxx, dxy], -1).view(B, C, hm_h - 4, hm_w - 4, 2, 1)
        h2 = torch.stack([dxy, dyy], -1).view(B, C, hm_h - 4, hm_w - 4, 2, 1)
        hessian = torch.cat([h1, h2], -1)
        # the inversion could not be completed because the matrix is singular
        # each pixel or max-pixel
        # if hessian.det() != 0:
        det = dxx * dyy - dxy * dxy
        cond = det.view(B, C, hm_h - 4, hm_w - 4, 1, 1).repeat(1, 1, 1, 1, 2, 2)
        diagm = torch.eye(2, 2).view(1, 1, 1, 1, 2, 2).repeat(B, C, hm_h - 4, hm_w - 4, 1, 1).to(det.device)
        hessian = torch.where(cond != 0, hessian, diagm)
        hessianinv = hessian.inverse()
        oft = (-hessianinv).matmul(derivative).squeeze(-1)
        # coord + offset | maxval + offset
        oft = torch.nn.functional.pad(oft, (0, 0, 2, 2, 2, 2), value=0)
        cnd = torch.nn.functional.pad(cond[..., 0], (0, 0, 2, 2, 2, 2), value=0)
        crd = (coord + peak - oft)
        # singular matrix means the pixel is far from peak value
        # the coord of pixels (hessian matrix is sigular) is w/o rectification
        coord = torch.where(cnd == 0, coord.long(), crd.long())
        coord = coord.view(-1, 2)
        hm = hm.view(-1)
        # Check whether the coordinates match the values
        kept = (coord[:, 0] >= 0) & (coord[:, 0] < hm_h) \
               & (coord[:, 1] >= 0) & (coord[:, 1] < hm_w)
        coord = coord[kept]
        hm = hm[kept]
        f = torch.zeros((B, C, hm_h, hm_w), device=hm.device)
        f[:, :, coord[:, 0], coord[:, 1]] = hm
        return f

    def forward(self, x):
        cout = self.get_voxels(x)
        return cout

H = 512,
W = 1024,
resize_lim = (0.193, 0.225),
final_dim = [512, 1024],
bot_pct_lim = (0.0, 0.22),
rot_lim = (-5.4, 5.4),
rand_flip = True,
ncams = 1,
max_grad_norm = 5.0,
pos_weight = 2.13,

xbound = [-75.0, 75.0, 0.25],
ybound = [0.0, 75.0, 0.25],
zbound = [-10.0, 10.0, 20.0],
dbound = [4.0, 75.0, 1.0],
grid_conf = {
    'xbound': [-75.0, 75.0, 0.25],
    'ybound': [0.0, 75.0, 0.25],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 75.0, 1.0],
}
data_aug_conf = {
    'resize_lim': resize_lim,
    'final_dim': [512, 1024],
    'rot_lim': rot_lim,
    'H': H, 'W': W,
    'rand_flip': rand_flip,
    'bot_pct_lim': bot_pct_lim,
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': ncams,
}

from torchvision import transforms

img = '../1635318024.379.png'
im = Image.open(img).resize((1024 // 16, 512 // 16), resample=0)
im = np.asarray(im)
# im.show()
device = 'cuda:0'
m = L(grid_conf, data_aug_conf, 2)
m.to(device)
x = transforms.ToTensor()(im).float()
# x = x.permute(1, 2, 0)
x = x.unsqueeze(0).unsqueeze(0)#.unsqueeze(0).repeat(1, 1, 71, 1, 1, 1)
y = m(x.to(device))

print(y.shape)
bev = y[0, :, :, :].permute(2, 1, 0).detach().cpu().numpy()
print(bev.shape)
print(np.max(bev), np.min(bev))
# bev = bev.T
proj = bev[::-1, :]
Image.fromarray(np.uint8(proj*255)).show()