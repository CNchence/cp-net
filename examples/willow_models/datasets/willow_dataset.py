#!/usr/bin/env python

nimport glob
import os

from chainer import dataset

import random
import numpy as np
import quaternion
import cv2
import six

from cp_net.utils import inout
import cp_net.utils.preprocess_utils as preprocess_utils



class WillowDataset(dataset.DatasetMixin):
    """
    dataset class for willow dataset
    original willow dataset images have one class object per image
    """
    def __init__(self, path, data_indices, img_height = 480, img_width = 640,
                 n_class=9,
                 n_view=35,
                 random_crop=False,
                 train_output_scale=1.0,
                 gaussian_noise=False,
                 gamma_augmentation=False,
                 avaraging=False,
                 salt_pepper_noise=False,
                 contrast=False,
                 metric_filter=1.0):
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

        self.original_im_size = (480, 640)
        self.K = np.array([[572.41140, 0, 325.26110],
                           [0, 573.57043, 242.04899],
                           [0, 0, 0]])

        self.metric_filter = metric_filter

    def __len__(self):
        return len(self.data_indices)

    def _get_pose(self, idx):
        return ret_pos, ret_rot

    def _get_pointcloud(self, depth_im, K):
        xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
        ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T
        Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
        Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])
        xyz_im = np.dstack((Xs, Ys, depth_im))
        xyz_im[xyz_im == 0] = np.nan
        return xyz_im

    def _get_mask(self, idx, path):
        ret = []
        for obj in self.objs:
            mask = cv2.imread(
                os.path.join(path, "mask", 'mask_{0:0>5}_{1}.png'.format(idx, obj)))
            ret.append(mask)
        return ret

    def load_orig_data(self, c_idx, v_idx):
        c_path = os.path.join(self.base, 'object_{0:0>2}'.format(c_idx))
        rgb = cv2.imread(os.path.join(c_path, 'rgb_{0;0>8}.png').format(v_idx))
        depth = cv2.imread(os.path.join(c_path, 'depth_{0;0>8}.png').format(v_idx))
        mask = cv2.imread(os.path.join(c_path, 'mask_{0;0>8}.png').format(v_idx))
        pos = np.load(os.path.join(c_path, 'pos_{0;0>8}.npy').format(v_idx))
        rot = np.load(os.path.join(os.path.join(c_path, 'rot_{0;0>8}.png').format(v_idx)))
        return rgb, depth, mask, pos, rot

    def get_example(self, i):
        img_size = self.img_size
        c_i = self.class_indices[i // self.n_view]
        v_i = self.view_indices[i % self.n_view]
        img_rgb, depth, mask, pc, pos, rot = self.load_orig_data(c_i, v_i)

        return img_rgb, label.astype(np.int32), img_depth, img_cp, img_ocp, pos, rot, pc, obj_mask.astype(np.int32), nonnan_mask, K


class WillowRandomDataset(WillowDataset):
    """
    dataset class for willow dataset
    For support multi calss object image,
    this class does paste object image randomly
    """
    def __init__(self, path, data_indices, img_height = 480, img_width = 640,
                 n_class=9,
                 n_view=35,
                 random_crop=False,
                 train_output_scale=1.0,
                 gaussian_noise=False,
                 gamma_augmentation=False,
                 avaraging=False,
                 salt_pepper_noise=False,
                 contrast=False,
                 metric_filter=1.0):
        super.__init__(self, path, data_indices, img_height = img_height, img_width = img_width,
                 n_class=n_class,
                 random_crop=random_crop,
                 train_output_scale=train_output_scale,
                 gaussian_noise=gaussian_noise,
                 gamma_augmentation=gamma_augmentation,
                 avaraging=avaraging,
                 salt_pepper_noise=salt_pepper_noise,
                 contrast=contrast,
                 metric_filter=metric_filter)

    def pointcloud_to_depth(pc, K, img_size):
        xs = np.round(pc[:, 0] * K[0, 0] / pc[:, 2] + K[0, 2])
        ys = np.round(pc[:, 1] * K[1, 1] / pc[:, 2] + K[1, 2])

        inimage_mask = (xs > 0) * (xs < img_size[0]) * \
                       (ys > 0) * (ys < img_size[1])

        xs = xs[inimage_mask].astype(np.int32)
        ys = ys[inimage_mask].astype(np.int32)
        zs = pc[:, 2][inimage_mask]
        idx = np.argsort(zs)
        # render depth
        ren_depth = np.zeros(img_size[::-1])
        ren_depth[ys[idx], xs[idx]] = zs[idx]
        return ren_depth

    def _generate_paste_data(translate_xy_max=3.0, translate_z_max=1.0, rotation_max = 45):
        rand_cls = np.random.randint(0, self.n_class)
        rand_view = np.random.randint(0, self.n_view)
        rgb, depth, mask, pos, rot = self.load_orig_data(self, rand_cls, rand_view)
        ## translate center
        trans_to_center = np.array([pos[0] * K[0,0] / pos[2], pos[0] * K[1,1] / pos[2]])
        M = np.float32([[1,0, trans_to_center[0]],[0,1, trans_to_center[1]]])
        rgb_center = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]))
        mask_center = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        depth_center = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        ## translate randomly
        t = np.empty(3)
        t[:2] = (np.random.rand(2) - 0.5) * 2 * translate_xy_max
        t[2] = (np.random.rand(1) - 0.5) * 2 * translate_z_max
        ## rotate randomly
        angle = (np.rand(1) - 0.5) * 2 * rotation_max
        rad = np.math.radians(angle)
        ## translate z
        rgb_trans = cv2.resize(rgb_center, (rgb_center.shape[1] / (pos[2] + t[2]) / pos[2],
                                            rgb_center.shape[0] / (pos[2] + t[2]) / pos[2]))
        mask_trans = cv2.resize(mask_center, (mask_center.shape[1] / (pos[2] + t[2]) / pos[2],
                                              mask_center.shape[0] / (pos[2] + t[2]) / pos[2]))
        depth_trans = cv2.resize(depth_center, (depth_center.shape[1] / (pos[2] + t[2]) / pos[2],
                                                depth_center.shape[0] / (pos[2] + t[2]) / pos[2]))
        depth_trans = depth_trans + t[2]


        return rgb, mask, pos, rot

    def load_orig_data(self, idx):
        return img_rgb, label.astype(np.int32), img_depth, img_cp, img_ocp, pos, rot, pc, obj_mask.astype(np.int32), nonnan_mask, K


