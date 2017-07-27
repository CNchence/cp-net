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
                 n_class=9,
                 random=False, random_crop=False, random_flip=False, random_resize=False,
                 ver2=False):
        self.base_path = path
        self.n_class = n_class
        self.data_indices = data_indices
        self.img_height = img_height
        self.img_width = img_width
        self.random = random
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.random_resize = random_resize
        self.objs = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']

        self.ver2 = ver2

    def __len__(self):
        return len(self.data_indices)

    def _get_pose(self, idx):
        ret_pos = np.zeros((self.n_class - 1, 3))
        ret_rot = np.zeros((self.n_class - 1, 3, 3))
        fpath = os.path.join(self.base_path, "poses", '{0}', 'info_{1:0>5}.txt')
        flip_z = np.diag((1, 1, -1))
        for obj_id, obj in enumerate(self.objs):
            pose = inout.load_gt_pose_dresden(fpath.format(obj, idx))
            if pose['R'].size != 0 and pose['t'].size != 0:
                # convert obj->camera coordinates to camera->obj coordinates
                ret_pos[obj_id] = pose['t'].ravel() * np.array([-1, -1, 1]) # because original data use negative z-axis
                ret_rot[obj_id] = np.linalg.inv(flip_z.dot(pose['R']))
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

        # imagenet_mean = np.array(
        #     [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]
        # img_rgb -= imagenet_mean

        label = np.zeros((self.img_height, self.img_width))
        masks_tmp = np.zeros((self.n_class - 1, self.img_height, self.img_width))
        obj_mask = np.zeros((self.img_height, self.img_width))

        pc = cv2.resize(pc, (self.img_width, self.img_height)).transpose(2,0,1)

        ## nonnan mask
        nonnan_mask = np.invert(np.isnan(pc[0])).astype(np.float32)

        img_cp = np.zeros_like(pc)
        img_ocp = np.zeros_like(pc)

        if self.ver2:
            # ver2
            img_cp = np.zeros((self.n_class - 1, 3, self.img_height, self.img_width))
            img_cp = pos[:, :, np.newaxis, np.newaxis] - pc[np.newaxis, :, :, :]
            img_ocp = np.zeros_like(img_cp)
            for idx in six.moves.range(self.n_class - 1):
                mask = masks[idx]
                if mask is None:
                    mask = np.zeros((self.img_height, self.img_width, 1))
                mask = mask.transpose(2,0,1)[0] / 255.0  # Scale to [0, 1];
                mask = cv2.resize(mask, (self.img_width, self.img_height))
                masks_tmp[idx] = mask
                label[mask.astype(np.bool)] = idx + 1

                if np.linalg.norm(pos[idx]) != 0:
                    inv_rot = np.linalg.inv(rot[idx])
                    img_ocp[idx] = np.dot(inv_rot, - img_cp[idx].reshape(3,-1)).reshape(img_cp[idx].shape)

            masks = masks_tmp.astype(np.bool)
            obj_mask = masks_tmp

            img_cp = img_cp * masks[:, np.newaxis, :, :]
            img_ocp = img_ocp * masks[:, np.newaxis, :, :]
        else:
            # ver1
            for idx in six.moves.range(self.n_class - 1):
                mask = masks[idx]
                if mask is None:
                    mask = np.zeros((self.img_height, self.img_width, 1))
                mask = mask.transpose(2,0,1)[0] / 255.0  # Scale to [0, 1];
                mask = cv2.resize(mask, (self.img_width, self.img_height)).astype(np.bool)
                label[mask] = idx + 1
                masks_tmp[idx] = mask
                obj_mask = np.logical_or(obj_mask, mask)

                if np.linalg.norm(pos[idx]) != 0:
                    inv_rot = np.linalg.inv(rot[idx])
                    img_cp_tmp = pos[idx][:, np.newaxis, np.newaxis] - pc
                    img_cp_tmp[img_cp_tmp != img_cp_tmp] = 0
                    img_ocp_tmp = np.dot(inv_rot, - img_cp_tmp.reshape(3,-1)).reshape(img_cp.shape)
                    img_cp[:, mask] = img_cp_tmp[:, mask]
                    img_ocp[:, mask] = img_ocp_tmp[:, mask]

            obj_mask = obj_mask * nonnan_mask

        img_cp = img_cp.astype(np.float32)
        img_ocp = img_ocp.astype(np.float32)

        img_cp[img_cp != img_cp] = 0
        img_ocp[img_ocp != img_ocp] = 0

        ## ignore nan
        label[np.isnan(pc[0])] = -1

        return img_rgb, label.astype(np.int32), img_cp, img_ocp, pos, rot, pc, obj_mask.astype(np.int32), nonnan_mask
