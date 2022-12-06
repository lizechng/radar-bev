"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
import torch
from PIL import Image
import random
import torchvision.transforms.functional as F


def get_loader(args, dataset):
    """
    Define the different dataloaders for training and validation
    """
    crop_size = (args['crop_h'], args['crop_w'])
    perform_transformation = not args['no_aug']

    train_dataset = Dataset_loader(
        args['data_path'], dataset.train_paths,
        rotate=args['rotate'], crop=crop_size,
        flip=args['flip'], train=perform_transformation)
    val_dataset = Dataset_loader(
        args['data_path'], dataset.val_paths,
        rotate=args['rotate'], crop=crop_size,
        flip=args['flip'], train=False)

    train_sampler = None
    val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], sampler=train_sampler,
        shuffle=False, num_workers=args['nworkers'],
        pin_memory=True, drop_last=True)
    # shuffle = val_sampler is None
    val_loader = DataLoader(
        val_dataset, batch_size=int(args['val_batch_size']), sampler=val_sampler,
        shuffle=False, num_workers=args['nworkers_val'],
        pin_memory=True, drop_last=True)

    return train_loader, val_loader


class Dataset_loader(Dataset):
    """Dataset with labeled lanes"""

    def __init__(self, data_path, dataset_type,
                 rotate, crop, flip, train=False):

        # Constants
        self.datapath = data_path
        self.dataset_type = dataset_type
        self.train = train
        self.flip = flip
        self.crop = crop
        self.rotate = rotate

        # Transformations
        self.totensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(size=crop)
        # Names
        self.img_name = 'img'
        self.radar_name = 'radar'
        self.lidar_name = 'lidar'
        self.lidHt_name = 'lidHt'
        self.fov_name = 'fov_mask'
        self.obj_name = 'obj_mask'
        self.depth_name = 'depth'
        self.calib_name = 'calib'

    def __len__(self):
        """
        Conventional len method
        """
        return len(self.dataset_type['radar'])

    def value_read(self, img):
        png = np.array(img, dtype=int)
        png = np.expand_dims(png, axis=2)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(png) <= 255)
        value = png.astype(np.float) / 255.
        return value

    def depth_read(self, img):
        png = np.array(img, dtype=int)
        png = np.expand_dims(png, axis=2)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(png) <= 75)
        value = png.astype(np.float) / 75.
        return value

    def lidHt_read(self, img):
        png = np.array(img, dtype=int)
        png = np.expand_dims(png, axis=2)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(png) <= 40)
        value = png.astype(np.float) / 40.
        return value

    def define_transforms(self, img_radar, img_lidar, img_lidHt, img_depth, fov_mask, obj_mask, img=None):
        # Define random variabels
        hflip_input = np.random.uniform(0.0, 1.0) > 0.5 and self.flip == 'hflip'

        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(img_radar, output_size=self.crop)
            img_radar = F.crop(img_radar, i, j, h, w)
            img_lidar = F.crop(img_lidar, i, j, h, w)
            img_lidHt = F.crop(img_lidHt, i, j, h, w)
            fov_mask = F.crop(fov_mask, i, j, h, w)
            obj_mask = F.crop(obj_mask, i, j, h, w)
            # img = F.crop(img, transforms.RandomCrop.get_params(img_radar, output_size=(600, 1200)))
            if hflip_input:
                img_radar, img_lidar, img_lidHt, fov_mask, obj_mask = F.hflip(img_radar), F.hflip(img_lidar), F.hflip(
                    img_lidHt), F.hflip(fov_mask), F.hflip(obj_mask)
                img_depth = F.hflip(img_depth)
                img = F.hflip(img)
            img_radar_np, img_lidar_np, img_lidHt_np, fov_mask_np, obj_mask_np = self.value_read(img_radar), \
                                                                                 self.value_read(img_lidar), \
                                                                                 self.lidHt_read(img_lidHt), \
                                                                                 self.value_read(fov_mask), \
                                                                                 self.value_read(obj_mask)
            img_depth_np = self.depth_read(img_depth)
        else:
            img_radar, img_lidar, img_lidHt, fov_mask, obj_mask = self.center_crop(img_radar), self.center_crop(
                img_lidar), self.center_crop(img_lidHt), self.center_crop(fov_mask), self.center_crop(obj_mask)
            img = transforms.CenterCrop(size=(img.size[1], img.size[0]))(img)  # Image crop func
            img_radar_np, img_lidar_np, img_lidHt_np, fov_mask_np, obj_mask_np = self.value_read(img_radar), \
                                                                                 self.value_read(img_lidar), \
                                                                                 self.lidHt_read(img_lidHt), \
                                                                                 self.value_read(fov_mask), \
                                                                                 self.value_read(obj_mask)
            img_depth_np = self.depth_read(img_depth)
        return img_radar_np, img_lidar_np, img_lidHt_np, img_depth_np, fov_mask_np, obj_mask_np, img

    def __getitem__(self, idx):
        radar_filename = os.path.join(self.dataset_type[self.radar_name][idx])
        lidar_filename = os.path.join(self.dataset_type[self.lidar_name][idx])
        lidHt_filename = os.path.join(self.dataset_type[self.lidHt_name][idx])
        fov_filename = os.path.join(self.dataset_type[self.fov_name][idx])
        obj_filename = os.path.join(self.dataset_type[self.obj_name][idx])
        calib_filename = os.path.join(self.dataset_type[self.calib_name][idx])
        depth_filename = os.path.join(self.dataset_type[self.depth_name][idx])
        img_filename = self.dataset_type[self.img_name][idx]

        with open(calib_filename, 'r') as f:
            m = np.asarray(json.load(f)['Cam2TI'])

        with open(depth_filename, 'rb') as f:
            img_depth = Image.open(f)
            w, h = img_depth.size
            img_depth = F.crop(img_depth, 0, 0, h, w)
        with open(radar_filename, 'rb') as f:
            img_radar = Image.open(f)
            w, h = img_radar.size
            # crop[0] should be h
            img_radar = F.crop(img_radar, h - self.crop[0], 0, self.crop[0], w)
        with open(lidar_filename, 'rb') as f:
            img_lidar = Image.open(f)
            w, h = img_lidar.size
            img_lidar = F.crop(img_lidar, h - self.crop[0], 0, self.crop[0], w)
        with open(lidHt_filename, 'rb') as f:
            img_lidHt = Image.open(f)
            w, h = img_lidHt.size
            img_lidHt = F.crop(img_lidHt, h - self.crop[0], 0, self.crop[0], w)
        with open(fov_filename, 'rb') as f:
            fov_mask = Image.open(f)
            fov_mask = F.crop(fov_mask, h - self.crop[0], 0, self.crop[0], w)
        with open(obj_filename, 'rb') as f:
            obj_mask = Image.open(f)
            obj_mask = F.crop(obj_mask, h - self.crop[0], 0, self.crop[0], w)
        with open(img_filename, 'rb') as f:
            img = (Image.open(f).convert('RGB'))
        img = F.crop(img, 0, 0, img.size[1], img.size[0])

        img_radar_np, img_lidar_np, img_lidHt_np, img_depth_np, \
        fov_mask_np, obj_mask_np, img_pil = self.define_transforms(img_radar,
                                                                   img_lidar,
                                                                   img_lidHt,
                                                                   img_depth,
                                                                   fov_mask,
                                                                   obj_mask,
                                                                   img)
        tensor_m = self.totensor(m).float()
        tensor_radar = self.totensor(img_radar_np).float()  # 1, C, H, W
        tensor_lidar = self.totensor(img_lidar_np).float()  # .unsqueeze(0) # 1, C, H, W
        tensor_lidHt = self.totensor(img_lidHt_np).float()  # .unsqueeze(0) # 1, C, H, W
        tensor_depth = self.totensor(img_depth_np).float()  # .unsqueeze(0) # 1, C, H, W
        tensor_fov = self.totensor(fov_mask_np).float()  # .unsqueeze(0) # 1, C, H, W
        tensor_obj = self.totensor(obj_mask_np).float()  # .unsqueeze(0) # 1, C, H, W
        tensor_img = self.totensor(img_pil).float().unsqueeze(0)  # 1, C, H, W  # in transform.ToTensor(), img.div(255)

        return tensor_img, tensor_radar, tensor_lidar, tensor_lidHt, tensor_depth, tensor_fov, tensor_obj, tensor_m
