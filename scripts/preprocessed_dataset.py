#!/usr/bin/env python

from chainer import dataset

import random
import numpy as np
import cv2
import os

def calc_quaternion(rot):
    quat = np.zeros(4)
    quat[0] = (rot[0,0] + rot[1,1] + rot[2,2] + 1.0) / 4.0
    quat[1] = np.sign(rot[2,1] - rot[1,2]) * (rot[0,0] - rot[1,1] - rot[2,2] + 1.0) / 4.0
    quat[2] = np.sign(rot[0,2] - rot[2,0]) * (-rot[0,0] + rot[1,1] - rot[2,2] + 1.0) / 4.0
    quat[3] = np.sign(rot[1,0] - rot[0,1]) * (-rot[0,0] - rot[1,1] + rot[2,2] + 1.0) / 4.0
    return quat

class PreprocessedDataset(dataset.DatasetMixin):

    def __init__(self, path, num_class, num_view, img_size=(256, 192), random=True):
        self.base = path
        self.n_class = num_class
        self.n_view = num_view
        self.img_size = img_size
        # self.mean = mean.astype('f')
        # self.random = random

    def __len__(self):
        return (self.n_class - 1) * self.n_view

    def load_orig_data(self, c_idx, v_idx):
        v_idx_format = '{0:08d}'.format(v_idx)
        c_idx_format = 'object_' + '{0:02d}'.format(c_idx + 1)
        c_path = os.path.join(self.base, c_idx_format)
        rgb = cv2.imread(os.path.join(c_path, 'rgb_' + v_idx_format +'.png'))
        depth = cv2.imread(os.path.join(c_path, 'depth_' + v_idx_format +'.png'), 0)
        mask = cv2.imread(os.path.join(c_path, 'mask_' + v_idx_format +'.png'))
        dist = np.load(os.path.join(c_path, 'dist_' + v_idx_format +'.npy'))
        rot = np.load(os.path.join(c_path, 'rot_' + v_idx_format +'.npy'))
        quat = calc_quaternion(rot)
        return rgb, depth, mask, dist, quat

    def get_example(self, i):
        img_size = self.img_size
        c_i = i // self.n_view
        v_i = i % self.n_view
        img_rgb, img_depth, mask, dist, quat = self.load_orig_data(c_i, v_i)

        # image, label = self.base[i]
        # _, h, w = image.shape

        # TODO image preprocessing
        #     - Cropping (random or center rectangular)
        #     - Random flip

        # if self.random:
        #     # Randomly crop a region and flip the image
        #     top = random.randint(0, h - crop_size - 1)
        #     left = random.randint(0, w - crop_size - 1)
        #     if random.randint(0, 1):
        #         image = image[:, :, ::-1]
        # else:
        #     # Crop the center
        #     top = (h - crop_size) // 2
        #     left = (w - crop_size) // 2
        # bottom = top + crop_size
        # right = left + crop_size

        # image = image[:, top:bottom, left:right]
        # image -= self.mean[:, top:bottom, left:right]
        # image *= (1.0 / 255.0)  # Scale to [0, 1];

        ## temporary crop
        img_rgb = img_rgb[48:432,34:576]
        img_depth = img_depth[48:432,34:576]
        mask = mask[48:432,34:576]
        dist = dist[48:432,34:576]

        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = cv2.resize(img_rgb, img_size)
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)
        img_depth = img_depth / 255.0  # Scale to [0, 1];
        img_depth =  cv2.resize(img_depth, img_size)
        img_depth = img_depth.reshape(1, img_size[1], img_size[0]).astype(np.float32)

        mask = mask / 255.0  # Scale to [0, 1];
        dist = dist / 255.0  # Scale to [0, 1];

        dist = cv2.resize(mask * dist, img_size)
        dist[dist!=dist] = -1.0 # non-nan

        pose = dist.transpose(2,0,1).astype(np.float32)

        mask = cv2.resize(mask, img_size)
        label = mask.transpose(2,0,1)[0] * (c_i + 1)

        mask_one = mask.transpose(2,0,1)[0]
        orientation = np.array([mask_one * quat[0], mask_one * quat[1],
                                mask_one * quat[2], mask_one * quat[3]], dtype=np.float32)

        return img_rgb, img_depth, label.astype(np.int32), pose, orientation


