#!/usr/bin/env python

import numpy as np
import cv2
import six
from PIL import Image

from linemod_sixd2017 import LinemodSIXDDataset


class LinemodSIXDSingleInstanceDataset(LinemodSIXDDataset):
    '''
    dataset for evaluation
    we cannot use this dataset for training ...
    '''
    def __init__(self, path, scene_index, objs_indices=np.arange(15) + 1,
                 img_height=480, img_width=640,
                 mode='test',
                 interval=1,
                 resize_rate = 0.5,
                 metric_filter=1.0):
        scene_indices = np.array([scene_index])
        super(LinemodSIXDSingleInstanceDataset, self).__init__(path, scene_indices, objs_indices=objs_indices,
                                                               img_height=img_height, img_width=img_width,
                                                               mode=mode,
                                                               dataset='test',
                                                               interval=interval,
                                                               resize_rate = resize_rate,
                                                               load_poses=True,
                                                               metric_filter=metric_filter)

        self.scene_id = scene_index

    def _load_poses(self, scene_id, im_id):
        scene_order = np.where(self.scenes==scene_id)[0][0]
        poses = self.gt_poses[scene_order][im_id]
        pos = np.zeros((self.n_class, 3))
        rot = np.zeros((self.n_class, 3, 3))
        for pose in poses:
            obj_id = pose['obj_id']
            if obj_id in self.objs:
                idx = np.where(self.objs==obj_id)[0][0]
                p = pose['cam_t_m2c']
                r = pose['cam_R_m2c']
                pos[idx] = p / 1000.0
                rot[idx] = r
        return pos, rot

    def _load_mask(self, scene_id, im_id):
        scene_order = np.where(self.scenes==scene_id)[0][0]
        poses = self.gt_poses[scene_order][im_id]
        mask = np.zeros((self.img_width, self.n_class))
        for pose in poses:
            obj_id = pose['obj_id']
            if obj_id in self.objs:
                idx = np.where(self.objs==obj_id)[0][0]
                f = Image.open(self.mask_fpath_mask.format(scene_id, im_id, obj_id))
                mask = np.asarray(f.convert('P'))
                break
        return mask

    def get_example(self, i):
        cv2.setNumThreads(0)
        scene_id, im_id = self.idx_dict[:, i]
        img_rgb, img_depth = self._load_images(scene_id, im_id)
        mask = self._load_mask(scene_id, im_id)
        pos, rot = self._load_poses(scene_id, im_id)
        K = self._load_k(scene_id, im_id)
        # rgb
        imagenet_mean = np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
        img_rgb = (img_rgb - imagenet_mean) / 255.0
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)
        ## depth
        K = 1.0 * self.out_height / img_depth.shape[0] * K
        img_depth = cv2.resize(img_depth, (self.out_width, self.out_height))

        return img_rgb, img_depth, pos, rot, K


