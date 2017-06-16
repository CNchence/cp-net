#!/usr/bin/env python

from chainer import dataset

import random
import numpy as np
import cv2
import os

import cp_net.utils.preprocess_utils as preprocess_utils


class DepthInvariantNetDataset(dataset.DatasetMixin):

    def __init__(self, path, class_indices, view_indices, img_size=(256, 192),
                 random=True, random_flip=True, random_resize=False, out_size=0.25):
        self.base = path
        self.n_class = len(class_indices)
        self.n_view = len(view_indices)
        self.class_indices = class_indices
        self.view_indices = view_indices
        self.img_size = img_size
        # self.mean = mean.astype('f')
        self.random = random
        self.random_flip = random_flip
        self.random_resize = random_resize
        self.out_size = 0.25

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

        # simple inpaint depth (using opencv function only considering depth, not using rgb)
        img_depth = np.sqrt(np.square(img_depth).sum(axis=2))
        img_depth = preprocess_utils.depth_inpainting(img_depth)

        ksizes = preprocess_utils.roi_kernel_size(img_depth)
        ksizes =  cv2.resize(ksizes, img_size)

        ## still we do not use depth
        img_depth = img_depth.reshape(1, img_size[1], img_size[0]).astype(np.float32)

        # only consider range 0.5 ~ 2.5[m]
        img_depth = (img_depth - 0.5) / 2.0
        img_depth[img_depth > 1.0] = 1.0
        img_depth[img_depth < 0.0] = 0.0

        img_depth =  cv2.resize(img_depth, img_size)

        # 1 ch mask
        mask = mask.transpose(2,0,1)[0] / 255.0  # Scale to [0, 1];

        ## random flip images
        if self.random_flip:
            rand_flip = random.randint(0,1)
            if rand_flip:
                img_rgb = img_rgb[:, ::-1, :]
                ksizes = ksizes[:,::-1]
                img_depth =  img_depth[:,:,::-1]
                mask = mask[:,::-1]

        # random resizing
        if self.random_resize:
            resize_ratio = random.uniform(0.5, 1.5)
            resized_imsize = (int(img_size[0] * resize_ratio),
                              int(img_size[1] * resize_ratio))

            if resize_ratio < 1.0:
                clop_h = random.randint(0, img_size[1] - resized_imsize[1])
                clop_w = random.randint(0, img_size[0] - resized_imsize[0])

                img_rgb = img_rgb[clop_h:(clop_h + resized_imsize[1]),
                                  clop_w:(clop_w + resized_imsize[0]), :]
                img_rgb = cv2.resize(img_rgb, img_size)

                ksizes = ksizes[clop_h:(clop_h + resized_imsize[1]),
                                clop_w:(clop_w + resized_imsize[0])]
                ksizes = cv2.resize(ksizes, img_size)

                mask = mask[clop_h:(clop_h + resized_imsize[1]),
                                  clop_w:(clop_w + resized_imsize[0])]
                mask = cv2.resize(mask, img_size)

            elif resize_ratio > 1.0:
                clop_h = random.randint(0, resized_imsize[1] - img_size[1])
                clop_w = random.randint(0, resized_imsize[0] - img_size[0])

                img_rgb = cv2.resize(img_rgb, resized_imsize)
                img_rgb = img_rgb[clop_h:(clop_h + img_size[1]),
                                  clop_w:(clop_w + img_size[0]), :]

                ksizes = cv2.resize(ksizes, resized_imsize)
                ksizes = ksizes[clop_h:(clop_h + img_size[1]),
                                clop_w:(clop_w + img_size[0])]

                mask = cv2.resize(mask, resized_imsize)
                mask = mask[clop_h:(clop_h + img_size[1]),
                            clop_w:(clop_w + img_size[0])]

            ksizes = ksizes * resize_ratio

        # create 1 / 4 label mask
        imsize_resizeh = int(img_size[0] * self.out_size)
        imsize_resizew = int(img_size[1] * self.out_size)
        mask = cv2.resize(mask,  (imsize_resizeh, imsize_resizew))
        label = mask * c_i

        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)
        ksizes = ksizes.reshape(1, img_size[0], img_size[1])

        return img_rgb, ksizes.astype(np.float32), label.astype(np.int32)
