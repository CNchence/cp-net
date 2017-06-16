#!/usr/bin/env python

from chainer import dataset

import random
import numpy as np
import cv2
import os

import cp_net.utils.preprocess_utils as preprocess_utils


class DepthInvariantNetDataset(dataset.DatasetMixin):

    def __init__(self, path, class_indices, view_indices, img_size=(256, 192),
                 random=True, random_flip=True, random_ratio=False):
        self.base = path
        self.n_class = len(class_indices)
        self.n_view = len(view_indices)
        self.class_indices = class_indices
        self.view_indices = view_indices
        self.img_size = img_size
        # self.mean = mean.astype('f')
        self.random = random
        self.random_flip = random_flip

    def __len__(self):
        return self.n_class * self.n_view

    def load_orig_data(self, c_idx, v_idx):
        v_idx_format = '{0:08d}'.format(v_idx)
        c_idx_format = 'object_' + '{0:02d}'.format(c_idx)
        c_path = os.path.join(self.base, c_idx_format)
        rgb = cv2.imread(os.path.join(c_path, 'rgb_' + v_idx_format +'.png'))
        mask = cv2.imread(os.path.join(c_path, 'mask_' + v_idx_format +'.png'))
        pc = np.load(os.path.join(c_path, 'pc_' + v_idx_format +'.npy'))
        return rgb, mask, pc

    def get_example(self, i):
        img_size = self.img_size
        c_i = self.class_indices[i // self.n_view]
        v_i = self.view_indices[i % self.n_view]
        img_rgb, mask, pc = self.load_orig_data(c_i, v_i)

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
        # img_rgb = img_rgb[48:432,34:576]
        # img_depth = img_depth[48:432,34:576]
        # mask = mask[48:432,34:576]
        # pc =  pc[48:432,34:576]

        if self.random:
            img_rgb = preprocess_utils.add_noise(img_rgb)
            rand_h = random.randint(0,40)
            rand_w = random.randint(0,40)
            img_rgb = img_rgb[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
            img_depth = pc[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
            mask = mask[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
            pc = pc[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
        else:
            img_rgb = img_rgb[140:332,160:416]
            img_depth = pc[140:332,160:416]
            mask = mask[140:332,160:416]
            pc =  pc[140:332,160:416]

        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = cv2.resize(img_rgb, img_size)
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)

        # simple inpaint depth (using opencv function only considering depth, not using rgb)
        img_depth = np.sqrt(np.square(img_depth).sum(axis=2))
        img_depth = preprocess_utils.depth_inpainting(img_depth)

        ksizes = preprocess_utils.roi_kernel_size(img_depth).reshape(1,img_depth.shape[0],
                                                                     img_depth.shape[1])

        # only consider range 0.5 ~ 2.5[m]
        img_depth = (img_depth - 0.5) / 2.0
        img_depth[img_depth > 1.0] = 1.0
        img_depth[img_depth < 0.0] = 0.0

        img_depth =  cv2.resize(img_depth, img_size)
        img_depth = img_depth.reshape(1, img_size[1], img_size[0]).astype(np.float32)

        mask = mask.transpose(2,0,1)[0] / 255.0  # Scale to [0, 1];
        mask = cv2.resize(mask, img_size)
        label = mask * c_i

        ## random flip images
        if self.random_flip:
            rand_flip = random.randint(0,1)
            if rand_flip:
                img_rgb = img_rgb[:,:,::-1]
                ksizes = ksizes[:,:,::-1]
                img_depth =  img_depth[:,:,::-1]
                label = label[:,::-1]

        return img_rgb, ksizes.astype(np.float32), label.astype(np.int32)
