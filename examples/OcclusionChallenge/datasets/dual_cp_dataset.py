#!/usr/bin/env python

import glob
import os

from chainer import dataset

import random
import numpy as np
import quaternion
import cv2
import six

from cp_net.utils import inout
import cp_net.utils.preprocess_utils as preprocess_utils


class DualCPNetDataset(dataset.DatasetMixin):

    def __init__(self, path, data_indices, img_height = 480, img_width = 640,
                 n_class=9,
                 random_crop=False,
                 dummy_crop=False,
                 random_resize=False,
                 train_output_scale=1.0,
                 gaussian_noise=False,
                 gamma_augmentation=False,
                 avaraging=False,
                 salt_pepper_noise=False,
                 contrast=False,
                 ver2=False):
        self.base_path = path
        self.n_class = n_class
        self.data_indices = data_indices
        self.img_height = img_height
        self.img_width = img_width

        ## augmentation option
        self.gaussian_noise = gaussian_noise
        self.gamma_augmentation = gamma_augmentation
        self.avaraging = avaraging
        self.salt_pepper_noise =salt_pepper_noise
        self.contrast = contrast
        if self.contrast:
            self.contrast_server = preprocess_utils.ContrastAugmentation()

        self.random_crop = random_crop
        self.crop_sizeh = 24
        self.crop_sizew = 32
        self.dummy_crop = dummy_crop
        self.random_resize = random_resize
        self.objs = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']

        self.original_im_size = (480, 640)
        self.K = np.array([[572.41140, 0, 325.26110],
                           [0, 573.57043, 242.04899],
                           [0, 0, 0]])

        self.ver2 = ver2

    def __len__(self):
        return len(self.data_indices)

    def _get_pose(self, idx):
        ret_pos = np.zeros((self.n_class - 1, 3))
        ret_rot = np.zeros((self.n_class - 1, 3, 3))
        fpath = os.path.join(self.base_path, "poses", '{0}', 'info_{1:0>5}.txt')
        for obj_id, obj in enumerate(self.objs):
            pose = inout.load_gt_pose_dresden(fpath.format(obj, idx))
            if pose['R'].size != 0 and pose['t'].size != 0:
                # convert obj->camera coordinates to camera->obj coordinates
                ret_pos[obj_id] = pose['t'].ravel() ##* np.array([-1, -1, 1]) # because original data use negative z-axis
                # u, s, v = np.linalg.svd(pose['R'])
                # ret_rot[obj_id] = np.dot(u, v).T
                # quat = quaternion.from_rotation_matrix(pose['R'])
                # quat.z *= -1.0
                # ret_rot[obj_id] = quaternion.as_rotation_matrix(quat)
                ret_rot[obj_id] = pose['R']
        return ret_pos, ret_rot

    def _get_mask(self, idx, path):
        ret = []
        for obj in self.objs:
            mask = cv2.imread(
                os.path.join(path, "mask_inpaint", 'mask_{0:0>5}_{1}.png'.format(idx, obj)))
            ret.append(mask)
        return ret

    def load_orig_data(self, idx):
        rgbd_path = os.path.join(self.base_path, "RGB-D")
        rgb = cv2.imread(os.path.join(rgbd_path, "rgb_noseg", 'color_{0:0>5}.png'.format(idx)))
        depth = cv2.imread(os.path.join(rgbd_path, "depth_noseg", 'depth_{0:0>5}.png'.format(idx)),
                           cv2.IMREAD_UNCHANGED) / 1000.0
        # data augmentation
        if self.gaussian_noise and np.random.randint(0,2):
            rgb = preprocess_utils.add_noise(rgb)
        if self.avaraging and np.random.randint(0,2):
            rgb = preprocess_utils.avaraging(rgb)
        rand_gamma = np.random.randint(0, 3)
        if self.gamma_augmentation and rand_gamma:
            if rand_gamma - 1:
                rgb = preprocess_utils.gamma_augmentation(rgb)
            else:
                rgb = preprocess_utils.gamma_augmentation(rgb, gamma=1.5)
        if self.salt_pepper_noise and np.random.randint(0,2):
            rgb = preprocess_utils.salt_pepper_augmentation(rgb)
        rand_contrast = np.random.randint(0, 3)
        if self.contrast and rand_contrast:
            if rand_contrast - 1:
                rgb = self.contrast_server.high_contrast(rgb)
            else:
                rgb = self.contrast_server.low_contrast(rgb)
        imagenet_mean = np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
        rgb = rgb - imagenet_mean

        pc = np.load(os.path.join(rgbd_path, "xyz", 'xyz_{0:0>5}.npy'.format(idx)))
        masks = self._get_mask(idx, rgbd_path)
        pos, rot = self._get_pose(idx)
        return rgb, depth, masks, pc, pos, rot

    def get_example(self, i):
        ii = self.data_indices[i]
        img_rgb, img_depth, masks, pc, pos, rot = self.load_orig_data(ii)
        K = self.K.copy()

        resize_rate = 1.0
        if self.ver2:
            resize_rate = 0.5
        out_height = int(self.img_height * resize_rate)
        out_width = int(self.img_width * resize_rate)

        # cropping
        if self.random_crop:
            rand_h = random.randint(0, self.crop_sizeh)
            rand_w = random.randint(0, self.crop_sizew)
            crop_h = 480 - self.crop_sizeh
            crop_w = 640 - self.crop_sizew
            img_rgb = img_rgb[rand_h:(crop_h + rand_h), rand_w:(crop_w + rand_w)]
            img_depth = img_depth[rand_h:(crop_h + rand_h), rand_w:(crop_w + rand_w)]
            pc = pc[rand_h:(crop_h + rand_h), rand_w:(crop_w + rand_w)]
            for i in six.moves.range(len(masks)):
                if masks[i] is not None:
                    masks[i] = masks[i][rand_h:(crop_h + rand_h), rand_w:(crop_w + rand_w)]
            K[0, 2] -= rand_w
            K[1, 2] -= rand_h


        img_rgb = cv2.resize(img_rgb, (self.img_width, self.img_height))
        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)

        label = np.zeros((out_height, out_width))
        masks_tmp = np.empty((self.n_class - 1, out_height, out_width))
        obj_mask = np.zeros((out_height, out_width))

        ## depth
        K = 1.0 * out_height / img_depth.shape[0] * K
        img_depth = cv2.resize(img_depth, (out_width, out_height))

        ## point cloud
        pc = cv2.resize(pc, (out_width, out_height)).transpose(2,0,1)

        ## nonnan mask
        nonnan_mask = np.invert(np.isnan(pc[0])).astype(np.float32)

        if self.ver2:
            # ver2
            img_cp = pos[:, :, np.newaxis, np.newaxis] - pc[np.newaxis, :, :, :]
            img_ocp = np.empty_like(img_cp)
            for idx in six.moves.range(self.n_class - 1):
                mask = masks[idx]
                if mask is None:
                    mask = np.zeros((out_height, out_width, 1))
                mask = mask.transpose(2,0,1)[0] / 255.0  # Scale to [0, 1];
                mask = cv2.resize(mask, (out_width, out_height))
                masks_tmp[idx] = mask
                label[mask.astype(np.bool)] = idx + 1
                img_ocp[idx] = np.dot(rot[idx].T, - img_cp[idx].reshape(3, -1)).reshape(img_cp[idx].shape)
            masks = masks_tmp.astype(np.bool)
            obj_mask = masks_tmp

            img_cp = img_cp * masks[:, np.newaxis, :, :]
            img_ocp = img_ocp * masks[:, np.newaxis, :, :]

        else:
            # ver1
            img_cp = np.empty_like(pc)
            img_ocp = np.empty_like(pc)
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
        label[np.isnan(pc[0]) * (label == 0)] = -1


        return img_rgb, label.astype(np.int32), img_depth, img_cp, img_ocp, pos, rot, pc, obj_mask.astype(np.int32), nonnan_mask, K
