#!/usr/bin/env python

import glob
import os

from chainer import dataset

import random
import numpy as np
import cv2
import six

from obj_pose_eval import inout

import cp_net.utils.preprocess_utils as preprocess_utils



class DualCPNetDataset(dataset.DatasetMixin):

    def __init__(self, path, data_indices, img_height = 480, img_width = 640,
                 random=False, random_crop=False, random_flip=False, random_resize=False):
        self.base_path = path
        self.n_class = 9
        self.data_indices = data_indices
        self.img_height = img_height
        self.img_width = img_width
        self.random = random
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_resize = random_resize
        self.objs = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']

        # get GT poses
        self.gt_poses = []
        gt_poses_mask = os.path.join(self.base_path, 'poses', '{0}', '*.txt')
        for obj in self.objs:
            gt_fpaths = sorted(glob.glob(gt_poses_mask.format(obj)))
            gt_poses_obj = []
            for gt_fpath in gt_fpaths:
                gt_poses_obj.append(
                    inout.load_gt_pose_dresden(gt_fpath))
                self.gt_poses.append(gt_poses_obj)

    def __len__(self):
        return len(self.data_indices)

    def _get_pose(self, idx):
        ret_pos = np.zeros((self.n_class, 3))
        ret_rot = np.zeros((self.n_class, 3, 3))
        for obj_id, obj in enumerate(self.objs):
            pose = self.gt_poses[obj_id + 1][int(idx)]
            if pose['R'].size != 0 and pose['t'].size != 0:
                ret_pos[obj_id + 1] = pose['t'].ravel()
                ret_rot[obj_id + 1] = pose['R']
        return ret_pos, ret_rot


    def _get_mask(self, idx, path):
        ret = []
        for obj in self.objs:
            mask = cv2.imread(
                os.path.join(path, "mask", 'mask_{0:0>5}_{1}.png'.format(idx, obj)))
            ret.append(mask)
        return ret

    def load_orig_data(self, idx):
        rgbd_path = os.path.join(self.base_path, "RGB-D")
        rgb = cv2.imread(os.path.join(rgbd_path, "rgb_noseg", 'color_{0:0>5}.png'.format(idx)))
        pc = np.load(os.path.join(rgbd_path, "xyz", 'xyz_{0:0>5}.npy'.format(idx)))
        masks = self._get_mask(idx, rgbd_path)
        pos, rot = self._get_pose(idx)
        return rgb, masks, pc, pos, rot

    def get_example(self, i):
        ii = self.data_indices[i]
        img_rgb, masks, pc, pos, rot = self.load_orig_data(ii)
        #
        ## todo random crop
        #

        if self.random:
            img_rgb = preprocess_utils.add_noise(img_rgb)

        if self.random_crop:
            rand_h = random.randint(0, 24)
            rand_w = random.randint(0, 32)
            crop_h = 480 - 24
            crop_w = 640 - 32
            img_rgb = img_rgb[rand_h:(crop_h + rand_h), rand_w:(crop_w + rand_w)]
            for i in six.moves.range(len(masks)):
                if masks[i] is not None:
                    masks[i] = masks[i][rand_h:(crop_h + rand_h), rand_w:(crop_w + rand_w)]
            pc = pc[rand_h:(crop_h + rand_h), rand_w:(crop_w + rand_w)]

        if self.random_flip:
            rand_flip = random.randint(0,1)
        else:
            rand_flip = False

        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = cv2.resize(img_rgb, (self.img_width, self.img_height))
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)

        imagenet_mean = np.array(
            [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]
        img_rgb -= imagenet_mean

        label = np.zeros((self.img_height, self.img_width))
        masks_tmp = np.zeros((self.n_class, self.img_height, self.img_width))
        obj_mask = np.zeros((self.img_height, self.img_width))

        for obj_id in six.moves.range(1, self.n_class):
            mask = masks[obj_id - 1]
            if mask is not None:
                mask = mask.transpose(2,0,1)[0] / 255.0  # Scale to [0, 1];
                mask = cv2.resize(mask, (self.img_width, self.img_height))
                label[mask.astype(np.bool)] = obj_id
                masks_tmp[obj_id] = mask
                obj_mask = np.logical_or(obj_mask, mask)
        masks = masks_tmp.astype(np.bool)

        ## random flip train data
        if rand_flip:
            img_rgb = img_rgb[:,:,::-1]
            label = label[:,::-1]
            pc[:,:,::-1]
            pc[0] *= -1.0
            pos[0] *= -1.0

        pc = cv2.resize(pc, (self.img_width, self.img_height)).transpose(2,0,1)
        img_cp = np.zeros_like(pc)
        img_ocp = np.zeros_like(pc)

        for idx in six.moves.range(1, self.n_class):
            if np.linalg.norm(pos[idx]) != 0:
                inv_rot = np.linalg.inv(rot[idx])
                img_cp_tmp = pos[idx][:, np.newaxis, np.newaxis] - pc
                img_cp_tmp[img_cp_tmp != img_cp_tmp] = 0

                img_ocp_tmp = np.dot(inv_rot, - img_cp_tmp.reshape(3,-1)).reshape(img_cp.shape)
                img_cp[:, masks[idx]] = img_cp_tmp[:, masks[idx]]
                img_ocp[:, masks[idx]] = img_ocp_tmp[:, masks[idx]]

        img_cp = img_cp.astype(np.float32)
        img_ocp = img_ocp.astype(np.float32)

        img_cp[img_cp != img_cp] = 0
        img_ocp[img_ocp != img_ocp] = 0

        ## nonnan mask
        nonnan_mask = np.invert(np.isnan(pc[0])).astype(np.float32)
        ## ignore nan
        label[np.isnan(pc[0])] = -1

        return img_rgb, label.astype(np.int32), img_cp, img_ocp, pos, pc, obj_mask.astype(np.int32), nonnan_mask
