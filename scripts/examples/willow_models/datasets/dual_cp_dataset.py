#!/usr/bin/env python

from chainer import dataset

import random
import numpy as np
import cv2
import os

import cp_net.utils.preprocess_utils as preprocess_utils


class DualCPNetDataset(dataset.DatasetMixin):

    def __init__(self, path, class_indices, view_indices, img_size=(256, 192),
                 random=True, random_flip=False, random_resize=False):
        self.base = path
        self.n_class = len(class_indices)
        self.n_view = len(view_indices)
        self.class_indices = class_indices
        self.view_indices = view_indices
        self.img_size = img_size
        self.random = random
        self.random_flip = random_flip
        self.random_resize = random_resize

    def __len__(self):
        return self.n_class * self.n_view

    def load_orig_data(self, c_idx, v_idx):
        v_idx_format = '{0:08d}'.format(v_idx)
        c_idx_format = 'object_' + '{0:02d}'.format(c_idx)
        c_path = os.path.join(self.base, c_idx_format)
        rgb = cv2.imread(os.path.join(c_path, 'rgb_' + v_idx_format +'.png'))
        mask = cv2.imread(os.path.join(c_path, 'mask_' + v_idx_format +'.png'))
        pc = np.load(os.path.join(c_path, 'pc_' + v_idx_format +'.npy'))
        pos = np.load(os.path.join(c_path, 'pos_' + v_idx_format +'.npy'))
        rot = np.load(os.path.join(c_path, 'rot_' + v_idx_format +'.npy'))
        return rgb, mask, pc, pos, rot

    def get_example(self, i):
        img_size = self.img_size
        c_i = self.class_indices[i // self.n_view]
        v_i = self.view_indices[i % self.n_view]
        img_rgb, mask, pc, pos, rot = self.load_orig_data(c_i, v_i)

        if self.random:
            img_rgb = preprocess_utils.add_noise(img_rgb)
            rand_h = random.randint(0,40)
            rand_w = random.randint(0,40)
            img_rgb = img_rgb[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
            mask = mask[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
            pc = pc[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
        else:
            img_rgb = img_rgb[140:332,160:416]
            img_depth = pc[140:332,160:416]
            mask = mask[140:332,160:416]
            pc =  pc[140:332,160:416]

        if self.random_flip:
            rand_flip = random.randint(0,1)
        else:
            rand_flip = False

        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = cv2.resize(img_rgb, img_size)
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)

        mask = mask.transpose(2,0,1)[0] / 255.0  # Scale to [0, 1];
        mask = cv2.resize(mask, img_size)
        label = mask * c_i

        ## random flip train data
        if rand_flip:
            img_rgb = img_rgb[:,:,::-1]
            label = label[:,::-1]
            pc[:,:,::-1]
            pc[0] *= -1.0
            pos[0] *= -1.0


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

                pc = pc[clop_h:(clop_h + resized_imsize[1]),
                        clop_w:(clop_w + resized_imsize[0]), :]
                pc = cv2.resize(pc, img_size)
                pc[:,:,3] *= 1.0 / resize_ratio

                mask = mask[clop_h:(clop_h + resized_imsize[1]),
                            clop_w:(clop_w + resized_imsize[0])]
                mask = cv2.resize(mask, img_size)

            elif resize_ratio > 1.0:
                clop_h = random.randint(0, resized_imsize[1] - img_size[1])
                clop_w = random.randint(0, resized_imsize[0] - img_size[0])

                img_rgb = cv2.resize(img_rgb, resized_imsize)
                img_rgb = img_rgb[clop_h:(clop_h + img_size[1]),
                                  clop_w:(clop_w + img_size[0]), :]

                pc = cv2.resize(pc, resized_imsize)
                pc = pc[clop_h:(clop_h + img_size[1]),
                        clop_w:(clop_w + img_size[0]), :]
                pc[:,:,3] *= 1.0 / resize_ratio

                mask = cv2.resize(mask, resized_imsize)
                mask = mask[clop_h:(clop_h + img_size[1]),
                            clop_w:(clop_w + img_size[0])]
        # print "-----"
        # print rot
        inv_rot = np.linalg.inv(rot)

        pc = cv2.resize(pc, img_size).transpose(2,0,1)
        img_cp = pos[:, np.newaxis, np.newaxis] - pc
        img_cp[img_cp != img_cp] = 0

        img_ocp = np.dot(inv_rot, - img_cp.reshape(3,-1)).reshape(img_cp.shape)

        img_cp = (img_cp * mask).astype(np.float32)
        img_ocp = (img_ocp * mask).astype(np.float32)

        ## nonnan mask
        nonnan_mask = np.invert(np.isnan(pc[0])).astype(np.float32)

        pos_arr = np.zeros((self.n_class + 1, 3))
        pos_arr[c_i] = pos

        # print "============"
        # print rot3
        # print inv_rot
        # print np.ma3x(((img_cp + pc_nonnan) * nonnan_mask).reshape(3,-1), axis=1)
        # print np.min(((img_cp + pc_nonnan) * nonnan_mask).reshape(3,-1), axis=1)
        # print np.max(pc_nonnan.reshape(3,-1), axis=1)
        # print np.min(pc_nonnan.reshape(3,-1), axis=1)
        # mask = mask.reshape(1, mask.shape[0], mask.shape[1])

        return img_rgb, label.astype(np.int32), img_cp, img_ocp, pos_arr, pc, mask.astype(np.int32), nonnan_mask
