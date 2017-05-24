#!/usr/bin/env python

from chainer import dataset

import random
import numpy as np
import cv2
import os

def add_noise(src):
    row,col,ch = src.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss_img = src + gauss
    return gauss_img

def calc_quaternion(rot):
    quat = np.zeros(4)
    quat[0] = (rot[0,0] + rot[1,1] + rot[2,2] + 1.0) / 4.0
    quat[1] = np.sign(rot[2,1] - rot[1,2]) * (rot[0,0] - rot[1,1] - rot[2,2] + 1.0) / 4.0
    quat[2] = np.sign(rot[0,2] - rot[2,0]) * (-rot[0,0] + rot[1,1] - rot[2,2] + 1.0) / 4.0
    quat[3] = np.sign(rot[1,0] - rot[0,1]) * (-rot[0,0] - rot[1,1] + rot[2,2] + 1.0) / 4.0
    return quat

def rpy_param(rot):
    return np.array([rot[0,0], rot[1,0], rot[2,0], rot[2,1], rot[2,2]])

def calc_rpy(rot, eps=10e-5):
    # r-p-y eular angle
    # return array : [tan_x, sin_y, tan_z]
    #
    if rot[2,2] != 0 :
        tan_x = rot[2,1] / rot[2,2]
    else:
        tan_x = rot[2,1] / (rot[2,2] + eps)
    sin_y = - rot[2,0]
    if rot[0,0] != 0 :
        tan_z = rot[1,0] / rot[0,0]
    else:
        tan_z = rot[1,0] / (rot[0,0] + eps)
    return np.array([tan_x, sin_y, tan_z])

class PreprocessedDataset(dataset.DatasetMixin):

    def __init__(self, path, class_indices, view_indices, img_size=(256, 192), random=True):
        self.base = path
        self.n_class = len(class_indices)
        self.n_view = len(view_indices)
        self.class_indices = class_indices
        self.view_indices = view_indices
        self.img_size = img_size
        # self.mean = mean.astype('f')
        self.random = random

    def __len__(self):
        return self.n_class * self.n_view

    def load_orig_data(self, c_idx, v_idx):
        v_idx_format = '{0:08d}'.format(v_idx)
        c_idx_format = 'object_' + '{0:02d}'.format(c_idx)
        c_path = os.path.join(self.base, c_idx_format)
        rgb = cv2.imread(os.path.join(c_path, 'rgb_' + v_idx_format +'.png'))
        depth = cv2.imread(os.path.join(c_path, 'depth_' + v_idx_format +'.png'), 0)
        mask = cv2.imread(os.path.join(c_path, 'mask_' + v_idx_format +'.png'))
        pos = np.load(os.path.join(c_path, 'pos_' + v_idx_format +'.npy'))
        rot = np.load(os.path.join(c_path, 'rot_' + v_idx_format +'.npy'))
        # rot = rot.flatten()
        # quat = calc_quaternion(rot)
        # rpy = rpy_param(rot)
        pc = np.load(os.path.join(c_path, 'pc_' + v_idx_format +'.npy'))
        return rgb, depth, mask, pos, rot, pc

    def get_example(self, i):
        img_size = self.img_size
        c_i = self.class_indices[i // self.n_view]
        v_i = self.view_indices[i % self.n_view]
        img_rgb, img_depth, mask, pos, rot, pc = self.load_orig_data(c_i, v_i)

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
            img_rgb = add_noise(img_rgb)
            rand_h = random.randint(0,40)
            rand_w = random.randint(0,40)
            img_rgb = img_rgb[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
            img_depth = img_depth[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
            mask = mask[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
            pc = pc[(120+rand_h):(120+192+rand_h), (140+rand_w):(140+256+rand_w)]
        else:
            img_rgb = img_rgb[140:332,160:416]
            img_depth = img_depth[140:332,160:416]
            mask = mask[140:332,160:416]
            pc =  pc[140:332,160:416]

        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = cv2.resize(img_rgb, img_size)
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)
        img_depth = img_depth / 255.0  # Scale to [0, 1];
        img_depth =  cv2.resize(img_depth, img_size)
        img_depth = img_depth.reshape(1, img_size[1], img_size[0]).astype(np.float32)

        mask = mask.transpose(2,0,1)[0] / 255.0  # Scale to [0, 1];
        mask = cv2.resize(mask, img_size)
        label = mask * c_i

        pc = cv2.resize(pc, img_size).transpose(2,0,1)
        rot_param = rpy_param(rot)

        mask5 = np.tile(mask.flatten(), 5).reshape(5, mask.shape[0], mask.shape[1])
        rot_map = mask5 * rot_param[:,np.newaxis, np.newaxis]

        dist_map = pc
        dist_map = pos[:,np.newaxis,np.newaxis] - dist_map
        dist_map[dist_map!=dist_map] = 0

        return img_rgb, img_depth, label.astype(np.int32), dist_map.astype(np.float32), pos, rot_param, rot_map.astype(np.float32), pc.astype(np.float32)
