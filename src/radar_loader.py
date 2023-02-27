"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
import re

sys.path.insert(1, os.path.join(sys.path[0], '..'))


class radar_preprocessing(object):
    def __init__(self, dataset_path):
        self.train_paths = {'img': [], 'lidar': [], 'lidHt': [], 'radar': [],
                            'fov_mask': [], 'obj_mask': [], 'calib': [], 'depth': []}
        self.val_paths = {'img': [], 'lidar': [], 'lidHt': [], 'radar': [],
                          'fov_mask': [], 'obj_mask': [], 'calib': [], 'depth': []}
        self.dataset_path = dataset_path

    def get_paths(self):
        # train and validation dirs
        for type_set in os.listdir(self.dataset_path):
            for root, dirs, files in os.walk(os.path.join(self.dataset_path, type_set)):
                self.train_paths['radar'].extend(sorted([os.path.join(root, file) for file in files
                                                         if re.search('radar', root)
                                                         and re.search('train', root)
                                                         and re.search('png', file)]))
                self.val_paths['radar'].extend(sorted([os.path.join(root, file) for file in files
                                                       if re.search('radar', root)
                                                       and re.search('val', root)
                                                       and re.search('png', file)]))
                self.train_paths['lidar'].extend(sorted([os.path.join(root, file) for file in files
                                                         if re.search('lidar', root)
                                                         and re.search('train', root)
                                                         and re.search('png', file)]))
                self.val_paths['lidar'].extend(sorted([os.path.join(root, file) for file in files
                                                       if re.search('lidar', root)
                                                       and re.search('val', root)
                                                       and re.search('png', file)]))
                self.train_paths['lidHt'].extend(sorted([os.path.join(root, file) for file in files
                                                         if re.search('lidHt', root)
                                                         and re.search('train', root)
                                                         and re.search('png', file)]))
                self.val_paths['lidHt'].extend(sorted([os.path.join(root, file) for file in files
                                                       if re.search('lidHt', root)
                                                       and re.search('val', root)
                                                       and re.search('png', file)]))
                self.train_paths['fov_mask'].extend(sorted([os.path.join(root, file) for file in files
                                                            if re.search('fov_mask', root)
                                                            and re.search('train', root)
                                                            and re.search('png', file)]))
                self.val_paths['fov_mask'].extend(sorted([os.path.join(root, file) for file in files
                                                          if re.search('fov_mask', root)
                                                          and re.search('val', root)
                                                          and re.search('png', file)]))
                self.train_paths['depth'].extend(sorted([os.path.join(root, file) for file in files
                                                         if re.search('depth', root)
                                                         and re.search('train', root)
                                                         and re.search('png', file)]))
                self.val_paths['depth'].extend(sorted([os.path.join(root, file) for file in files
                                                       if re.search('depth', root)
                                                       and re.search('val', root)
                                                       and re.search('png', file)]))
                self.train_paths['obj_mask'].extend(sorted([os.path.join(root, file) for file in files
                                                            if re.search('obj_mask', root)
                                                            and re.search('train', root)
                                                            and re.search('png', file)]))
                self.val_paths['obj_mask'].extend(sorted([os.path.join(root, file) for file in files
                                                          if re.search('obj_mask', root)
                                                          and re.search('val', root)
                                                          and re.search('png', file)]))
                self.train_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                       if re.search('img', root)
                                                       and re.search('train', root)
                                                       and re.search('png', file)]))
                self.val_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                     if re.search('img', root)
                                                     and re.search('val', root)
                                                     and re.search('png', file)]))
                self.train_paths['calib'].extend(sorted([os.path.join(root, file) for file in files
                                                         if re.search('calib', root)
                                                         and re.search('train', root)
                                                         and re.search('json', file)]))
                self.val_paths['calib'].extend(sorted([os.path.join(root, file) for file in files
                                                       if re.search('calib', root)
                                                       and re.search('val', root)
                                                       and re.search('json', file)]))

    def prepare_dataset(self):
        self.get_paths()
        # print(self.val_paths['img'], len(self.val_paths['img']))
        print('===============================')
        print('img   in training dataset: ', len(self.train_paths['img']))
        print('radar in training dataset: ', len(self.train_paths['radar']))
        print('lidar in training dataset: ', len(self.train_paths['lidar']))
        print('lidHt in training dataset: ', len(self.train_paths['lidHt']))
        print('fovmk in training dataset: ', len(self.train_paths['fov_mask']))
        print('objmk in training dataset: ', len(self.train_paths['obj_mask']))
        print('depth in training dataset: ', len(self.train_paths['depth']))
        print('calib in training dataset: ', len(self.train_paths['calib']))
        print('===============================')
        print('img   in val/test dataset: ', len(self.val_paths['img']))
        print('radar in val/test dataset: ', len(self.val_paths['radar']))
        print('lidar in val/test dataset: ', len(self.val_paths['lidar']))
        print('lidHt in val/test dataset: ', len(self.val_paths['lidHt']))
        print('fovmk in val/test dataset: ', len(self.val_paths['fov_mask']))
        print('objmk in val/test dataset: ', len(self.val_paths['obj_mask']))
        print('depth in val/test dataset: ', len(self.val_paths['depth']))
        print('calib in val/test dataset: ', len(self.val_paths['calib']))
        print('===============================')


if __name__ == '__main__':
    # Imports
    import os
    import argparse

    # arguments
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--num_samples', default=0, type=int, help='number of samples')
    parser.add_argument('--datapath', default='../dataset')
    parser.add_argument('--dest', default='/usr/data/tmp/')
    args = parser.parse_args()

    dataset = radar_preprocessing(args.datapath)
    dataset.prepare_dataset()
