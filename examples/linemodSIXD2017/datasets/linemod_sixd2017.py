#!/usr/bin/env python

import glob
import os

from chainer import dataset

import random
import numpy as np
import quaternion
import cv2
import six
import yaml

from cp_net.utils import preprocess_utils
from cp_net.utils import inout
from cp_net.utils import multi_object_renderer

from multiprocessing import Process, Pool, cpu_count
import itertools

def tomap(args):
    return getattr(args[0], args[1])(args[2:])


## super class of LinemodSIXD datasets
class LinemodSIXDDataset(dataset.DatasetMixin):
    def __init__(self, path, scene_indices, objs_indices=None,
                 img_height=480, img_width=640,
                 gaussian_noise=False,
                 gamma_augmentation=False,
                 avaraging=False,
                 salt_pepper_noise=False,
                 contrast=False,
                 mode='test',
                 interval=1,
                 resize_rate = 0.5,
                 load_poses=True,
                 metric_filter=1.0):

        if objs_indices is None:
            objs_indices = scene_indices
        self.base_path = path
        self.img_height = img_height
        self.img_width = img_width

        self.out_height = int(img_height * resize_rate)
        self.out_width = int(img_width * resize_rate)

        self.objs = objs_indices
        self.scenes = scene_indices
        self.n_class = len(objs_indices)
        # augmentation option
        self.gaussian_noise = gaussian_noise
        self.gamma_augmentation = gamma_augmentation
        self.avaraging = avaraging
        self.salt_pepper_noise =salt_pepper_noise
        self.contrast = contrast
        if self.contrast:
            self.contrast_server = preprocess_utils.ContrastAugmentation()

        self.rgb_fpath_mask = os.path.join(path, 'test', '{0:0>2}', 'rgb', '{1:0>4}.png')
        self.depth_fpath_mask = os.path.join(path, 'test', '{0:0>2}', 'depth', '{1:0>4}.png')
        self.mask_fpath_mask = os.path.join(path, 'test', '{0:0>2}', 'mask', '{1:0>4}_{2:0>2}.png')
        self.gt_poses_mask = os.path.join(path, 'test', '{0:0>2}','gt.yml')
        self.info_mask = os.path.join(path, 'test', '{0:0>2}','info.yml')

        self.metric_filter = metric_filter

        self.gt_poses = []
        self.infos = []
        self.idx_dict = np.array([[], []])

        if not load_poses:
            return

        gts_dict = {}
        jobs = []
        p = Pool(cpu_count())
        args = itertools.izip(itertools.repeat(self), itertools.repeat('_load_gt_yaml'), scene_indices)
        yaml_list = p.map(tomap, args)
        p.close()
        for i in six.moves.range(len(scene_indices)):
            gts = yaml_list[i][0]
            info = yaml_list[i][1]
            self.gt_poses.append(gts)
            self.infos.append(info)
            n_frames = len(gts)
            self.idx_dict = np.hstack((self.idx_dict,
                                       np.vstack((np.ones(n_frames) * scene_indices[i],
                                                  np.arange(n_frames)))))
            self.idx_dict = self.idx_dict.astype(np.int32)

        if interval > 1:
            if mode == 'train':
                self.idx_dict = self.idx_dict[:, 0::interval]
            elif mode == 'test':
                self.idx_dict = np.delete(self.idx_dict, self.idx_dict[:, 0::interval], axis=1)

    def __len__(self):
        return len(self.idx_dict[0])

    def _load_gt_yaml(self, args):
        sidx = args[0]
        gts = yaml.load(open(self.gt_poses_mask.format(sidx)))
        info = yaml.load(open(self.info_mask.format(sidx)))
        print "load gt poses : scene_{0:0>2}".format(sidx)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape(3)
        return gts, info

    def _get_pointcloud(self, depth_im, K, fill_nan=False):
        xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
        ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T
        Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
        Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])
        xyz_im = np.dstack((Xs, Ys, depth_im))
        if fill_nan:
            xyz_im[xyz_im == 0] = np.nan
        return xyz_im.astype(np.float32)

    def _load_pose(self, scene_id, im_id, obj_id):
        scene_order = np.where(self.scenes==scene_id)[0][0]
        pose = self.gt_poses[scene_order][im_id][obj_id]
        pos = np.asarray(pose['cam_t_m2c'])
        rot = np.asarray(pose['cam_R_m2c']).reshape(3, 3)
        return pos, rot

    def _load_k(self, scene_id, im_id):
        scene_order = np.where(self.scenes==scene_id)[0][0]
        info = self.infos[scene_order]
        k = np.asarray(info[im_id]['cam_K']).reshape(3,3)
        return k

    def _load_images(self, scene_id, im_id):
        rgb = cv2.imread(self.rgb_fpath_mask.format(scene_id, im_id))
        depth = cv2.imread(self.depth_fpath_mask.format(scene_id, im_id),
                           cv2.IMREAD_UNCHANGED) / 1000.0
        return rgb, depth

    def rgb_augmentation(self, img):
        ret = img.copy()
        if self.gaussian_noise and np.random.randint(0,2):
            ret = preprocess_utils.add_noise(ret)
        if self.avaraging and np.random.randint(0,2):
            ret = preprocess_utils.avaraging(ret)
        rand_gamma = np.random.randint(0, 3)
        if self.gamma_augmentation and rand_gamma:
            if rand_gamma - 1:
                ret = preprocess_utils.gamma_augmentation(ret)
            else:
                ret = preprocess_utils.gamma_augmentation(ret, gamma=1.5)
        if self.salt_pepper_noise and np.random.randint(0,2):
            ret = preprocess_utils.salt_pepper_augmentation(ret)
        rand_contrast = np.random.randint(0, 3)
        if self.contrast and rand_contrast:
            if rand_contrast - 1:
                ret = self.contrast_server.high_contrast(ret)
            else:
                ret = self.contrast_server.low_contrast(ret)
        return ret


class LinemodSIXDAutoContextDataset(LinemodSIXDDataset):
    def __init__(self, path, objs_indices, background_path, img_height = 480, img_width = 640,
                 gaussian_noise=False,
                 gamma_augmentation=False,
                 avaraging=False,
                 salt_pepper_noise=False,
                 contrast=False,
                 bg_flip=True,
                 channel_swap = True,
                 random_iteration=False,
                 mode='test',
                 interval=1,
                 resize_rate = 0.5,
                 iteration_per_epoch=1000,
                 load_poses=True,
                 metric_filter=1.0):

        super(LinemodSIXDAutoContextDataset, self).__init__(path, objs_indices,
                                                            img_height=img_height, img_width=img_width,
                                                            gaussian_noise=gaussian_noise,
                                                            gamma_augmentation=gamma_augmentation,
                                                            avaraging=False,
                                                            salt_pepper_noise=salt_pepper_noise,
                                                            contrast=contrast,
                                                            mode=mode,
                                                            interval=interval,
                                                            resize_rate = resize_rate,
                                                            load_poses=load_poses,
                                                            metric_filter=metric_filter)

        self.bg_fpaths = glob.glob(os.path.join(background_path, '*.jpg'))
        self.bg_flip = bg_flip
        self.random_iteration = random_iteration
        self.channel_swap = channel_swap
        self.iteration_per_epoch = iteration_per_epoch
        self.resize_rate = resize_rate

    def __len__(self):
        return min(self.iteration_per_epoch, len(self.idx_dict[0]))

    def _load_pose(self, scene_id, im_id):
        scene_order = np.where(self.scenes==scene_id)[0][0]
        poses = self.gt_poses[scene_order][im_id]
        pos = np.zeros(3)
        rot = np.zeros((3, 3))
        for pose in poses:
            obj_id = pose['obj_id']
            if obj_id == scene_id:
                idx = np.where(self.objs==obj_id)[0][0]
                p = pose['cam_t_m2c']
                r = pose['cam_R_m2c']
                pos = p / 1000.0
                rot = r
        return pos, rot

    def _load_mask(self, scene_id, im_id):
        scene_order = np.where(self.scenes==scene_id)[0][0]
        poses = self.gt_poses[scene_order][im_id]
        masks = np.zeros((self.img_height, self.img_width))
        for pose in poses:
            obj_id = pose['obj_id']
            if obj_id == scene_id:
                idx = np.where(self.objs==obj_id)[0][0]
                mask = cv2.imread(self.mask_fpath_mask.format(scene_id, im_id, obj_id), 0)
        return mask

    def _load_bg_data(self, idx):
        bg = cv2.imread(self.bg_fpaths[idx])
        # random crop
        height, width, ch = bg.shape
        resize_height = int((np.random.rand() * 0.5 + 0.5) * height)
        resize_width = int((np.random.rand() * 0.5 + 0.5) * width)
        crop_h = np.floor((height - resize_height) * np.random.rand()).astype(np.int64)
        crop_w = np.floor((width - resize_width) * np.random.rand()).astype(np.int64)

        bg = bg[crop_h:(crop_h + resize_height), crop_w:(crop_w + resize_width)]
        bg = cv2.resize(bg, (self.img_width, self.img_height))
        bg = self.rgb_augmentation(bg)
        if self.bg_flip and np.random.randint(0,2):
            bg = bg[:,::-1, :]
        if self.channel_swap:
            bg = bg[:, :, np.random.choice(np.arange(3), 3, replace=False)]
        return bg

    def estimate_visib_region(self, depth, depth_ref):
        visib_mask = np.logical_or((depth < depth_ref),
                                   np.logical_and((depth_ref == 0), (depth != 0)))
        return visib_mask

    def get_example(self, i):
        scene_id, im_id = self.idx_dict[:, i]
        K = self._load_k(scene_id, im_id)

        min_obj_num = min(10, len(self.objs)) # tmp
        obj_num = np.random.randint(min_obj_num, len(self.objs) + 1)
        obj_ind = np.random.choice(self.objs, obj_num, replace=False)

        img_rgb = np.zeros((self.img_height, self.img_width, 3))
        img_depth = np.zeros((self.out_height, self.out_width))
        label = np.zeros((self.out_height, self.out_width))
        img_cp = np.zeros((self.n_class, 3, self.out_height, self.out_width))
        img_ocp = np.zeros_like(img_cp)
        obj_mask = np.zeros((self.n_class, self.out_height, self.out_width))
        nonnan_mask = np.ones((self.out_height, self.out_width)).astype(np.float32)
        rgb_mask = np.zeros(((self.img_height, self.img_width)))
        positions = np.zeros((self.n_class, 3))
        rotations = np.zeros((self.n_class, 3, 3))

        min_z = 0.5
        max_z = 1.5
        edge_offset = 5

        for ii in six.moves.range(min_obj_num):
            target_obj = obj_ind[ii]
            obj_order = np.where(self.objs==target_obj)[0][0]
            im_id = np.random.choice(self.idx_dict[1][self.idx_dict[0] == target_obj], 1)[0]
            rgb, depth = self._load_images(target_obj, im_id)
            pos, rot = self._load_pose(target_obj, im_id)
            mask = self._load_mask(target_obj, im_id)
            points = self._get_pointcloud(depth, K, fill_nan=False)
            cp = pos[np.newaxis, np.newaxis, :] - points
            # translate and scaling
            x = np.random.randint(edge_offset, self.img_width - edge_offset) - self.img_width / 2
            y = np.random.randint(edge_offset, self.img_height - edge_offset)- self.img_height / 2
            z = min_z + np.random.rand() * (max_z - min_z)
            depth = depth + (z - pos[2])
            # Affine transform(scaling, translate, resize)
            g_x = K[0,2] + pos[0] * K[0,0] / pos[2]
            g_y = K[1,2] + pos[1] * K[1,1] / pos[2]
            M0 = np.float32([[pos[2] / z, 0, x + K[0, 2] - g_x * pos[2]/z],
                             [0, pos[2] / z, y + K[1, 2] - g_y * pos[2]/z]])
            M = M0.copy() * self.resize_rate
            rgb = cv2.warpAffine(rgb, M0, (self.img_width, self.img_height))
            depth = cv2.warpAffine(depth, M, (self.out_width, self.out_height))
            mask = cv2.warpAffine(mask, M, (self.out_width, self.out_height))
            cp = cv2.warpAffine(cp, M, (self.out_width, self.out_height))
            if np.sum(mask) == 0:
                continue
            # visib mask
            visib_mask = self.estimate_visib_region(depth, img_depth)
            visib_mask = visib_mask * mask
            visib_mask_resize = cv2.resize(visib_mask, (self.img_width, self.img_height))
            visib_mask_resize = visib_mask_resize.astype(np.bool)
            rgb_mask = np.logical_or(rgb_mask, visib_mask_resize)
            visib_mask = visib_mask.astype(np.bool)
            trans_pos = np.array([x * z / K[0, 0], y * z / K[1, 1], z])

            # masking
            img_rgb[visib_mask_resize, :] = rgb[visib_mask_resize, :]
            img_depth[visib_mask] = depth[visib_mask]
            cp = (cp * visib_mask[:,:, np.newaxis]).transpose(2,0,1)
            ocp = np.dot(rot.T, - cp.reshape(3, -1)).reshape(cp.shape)
            img_cp[obj_order] = cp
            img_ocp[obj_order] = ocp
            obj_mask[obj_order] = visib_mask
            label[visib_mask] = target_obj
            # pose
            positions[obj_order] = trans_pos
            rotations[obj_order] = rot

        img_rgb = self.rgb_augmentation(img_rgb)
        bg_id = np.random.randint(0, len(self.bg_fpaths))
        img_bg = self._load_bg_data(bg_id)
        ## random light color
        img_rgb = (img_rgb * (np.random.rand(3) * 0.4 + 0.8)[np.newaxis, np.newaxis, :])
        img_rgb = img_rgb * rgb_mask[:, :, np.newaxis] + img_bg * np.invert(rgb_mask[:, :, np.newaxis].astype(np.bool))
        img_rgb = preprocess_utils.gaussian_blur(img_rgb)
        imagenet_mean = np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
        img_rgb = img_rgb - imagenet_mean
        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)

        pc = self._get_pointcloud(img_depth, K, fill_nan=True).transpose(2,0,1)
        obj_mask = (obj_mask * nonnan_mask).astype(np.float32)

        img_cp = img_cp.astype(np.float32)
        img_ocp = img_ocp.astype(np.float32)

        img_cp[img_cp != img_cp] = 0
        img_ocp[img_ocp != img_ocp] = 0

        img_cp[np.abs(img_cp) > self.metric_filter] = 0
        img_ocp[np.abs(img_ocp) > self.metric_filter] = 0

        ## ignore nan
        # label[np.isnan(pc[0]) * (label == 0)] = -1

        return img_rgb, label.astype(np.int32), img_depth, img_cp, img_ocp, positions, rotations, pc, obj_mask.astype(np.int32), nonnan_mask, K


class LinemodSIXDRenderingDataset(LinemodSIXDAutoContextDataset):
    def __init__(self, path, objs_indices, background_path, models_path=None,
                 img_height = 480, img_width = 640,
                 gaussian_noise=False,
                 gamma_augmentation=False,
                 avaraging=False,
                 salt_pepper_noise=False,
                 contrast=False,
                 bg_flip=True,
                 channel_swap = True,
                 random_iteration=False,
                 mode='test',
                 interval=1,
                 resize_rate = 0.5,
                 iteration_per_epoch=1000,
                 metric_filter=1.0):

        super(LinemodSIXDRenderingDataset, self).__init__(path, objs_indices, background_path,
                                                          img_height=img_height, img_width=img_width,
                                                          gaussian_noise=gaussian_noise,
                                                          gamma_augmentation=gamma_augmentation,
                                                          avaraging=False,
                                                          salt_pepper_noise=salt_pepper_noise,
                                                          contrast=contrast,
                                                          mode=mode,
                                                          interval=interval,
                                                          resize_rate = resize_rate,
                                                          random_iteration=random_iteration,
                                                          iteration_per_epoch=iteration_per_epoch,
                                                          bg_flip=bg_flip,
                                                          channel_swap = channel_swap,
                                                          load_poses=False,
                                                          metric_filter=metric_filter)
        if models_path is None:
            models_path = os.path.join(path, 'models')
        models_fpath_mask = os.path.join(models_path, 'obj_{0:0>2}.ply')
        self.models = []
        for obj in objs_indices:
            print 'Loading data: obj_{0:0>2}'.format(obj)
            model_fpath = models_fpath_mask.format(obj)
            self.models.append(inout.load_ply(model_fpath))

        self.K = np.array([[572.41140, 0, 325.26110],
                           [0, 573.57043, 242.04899],
                           [0, 0, 0]])

        self.iteration_per_epoch=iteration_per_epoch

    def __len__(self):
        return self.iteration_per_epoch

    def generate_pose(self, min_z=0.3, max_z=1.5, edge_offset=3):
        z = min_z + np.random.rand() * (max_z - min_z)
        ## determine x and y by z value
        min_x = (edge_offset - self.K[0, 2]) * z / self.K[0, 0]
        max_x = ((self.img_width - edge_offset) - self.K[0, 2]) * z / self.K[0, 0]
        min_y = (edge_offset - self.K[1, 2]) * z / self.K[1, 1]
        max_y = ((self.img_height - edge_offset) - self.K[1, 2]) * z / self.K[1, 1]
        x =  min_x + np.random.rand() * (max_x - min_x)
        y =  min_y + np.random.rand() * (max_y - min_y)
        pos = np.array([x, y, z])
        quat = quaternion.from_euler_angles(np.random.rand()*360, np.random.rand()*360, np.random.rand()*360)
        rot = quaternion.as_rotation_matrix(quat)
        return pos, rot


    def render_objects(self, K, min_obj_num=5):
        obj_num = np.random.randint(min_obj_num, len(self.objs))
        obj_ind = np.random.choice(np.arange(len(self.objs)), obj_num, replace=False)
        pos_list = []
        rot_list = []
        model_list = []
        ret_pos = np.zeros((len(self.objs), 3))
        ret_rot = np.zeros((len(self.objs), 3, 3))
        for i, obj_idx in enumerate(obj_ind):
            pos, rot = self.generate_pose()
            pos_list.append(pos.T * 1000)
            rot_list.append(rot)
            model_list.append(self.models[obj_idx])
            ret_pos[obj_idx] = pos
            ret_rot[obj_idx] = rot
        labels_idx = obj_ind + 1
        min_light = 0.8
        ren_rgb, ren_depth, ren_label = multi_object_renderer.render(
            model_list, (self.img_width, self.img_height), K, rot_list, pos_list, 100, 4000,
            ambient_weight=np.random.rand(),
            light_color= np.random.rand(3) * (1.0 - min_light) + min_light,
            labels=labels_idx, mode='rgb+depth+label')

        ## rgb->bgr
        ren_depth = ren_depth / 1000.0

        return ren_rgb[:,:,::-1], ren_depth, ren_label, ret_pos.astype(np.float32), ret_rot.astype(np.float32)


    def get_example(self, i):
        K = self.K.copy()
        ren_rgb, img_depth, label_large, pos, rot = self.render_objects(K)
        bg_rgb = self._load_bg_data(np.random.randint(0, len(self.bg_fpaths)))
        img_rgb = ren_rgb * (label_large != 0)[:, :, np.newaxis] + bg_rgb * (label_large == 0)[:, :, np.newaxis]
        label_large[label_large < 0] = 0
        img_rgb = preprocess_utils.gaussian_blur(img_rgb, ksize=3)
        # rgb
        imagenet_mean = np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
        img_rgb = img_rgb - imagenet_mean
        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)

        obj_mask = np.empty((self.n_class - 1, self.out_height, self.out_width))

        ## depth
        K = 1.0 * self.out_height / img_depth.shape[0] * K
        img_depth = cv2.resize(img_depth, (self.out_width, self.out_height))
        ## point cloud
        pc = self._get_pointcloud(img_depth, K, fill_nan=True).transpose(2,0,1)

        ## mask
        label = np.zeros((self.out_height, self.out_width))
        obj_mask = np.empty((self.n_class, self.out_height, self.out_width))
        nonnan_mask = np.invert(np.isnan(pc[0])).astype(np.float32)

        img_cp = pos[:, :, np.newaxis, np.newaxis] - pc[np.newaxis, :, :, :]
        img_ocp = np.empty_like(img_cp)
        for idx in six.moves.range(self.n_class - 1):
            mask = (label_large == idx + 1).astype(np.uint8)
            mask = cv2.resize(mask, (self.out_width, self.out_height)).astype(np.bool)
            obj_mask[idx] = mask
            label[mask] = idx + 1
            img_ocp[idx] = np.dot(rot[idx].T, - img_cp[idx].reshape(3, -1)).reshape(img_cp[idx].shape)

        img_cp = img_cp * obj_mask[:, np.newaxis, :, :].astype(np.bool)
        img_ocp = img_ocp * obj_mask[:, np.newaxis, :, :].astype(np.bool)

        obj_mask = obj_mask * nonnan_mask

        img_cp = img_cp.astype(np.float32)
        img_ocp = img_ocp.astype(np.float32)

        img_cp[img_cp != img_cp] = 0
        img_ocp[img_ocp != img_ocp] = 0

        img_cp[np.abs(img_cp) > self.metric_filter] = 0
        img_ocp[np.abs(img_ocp) > self.metric_filter] = 0

        ## ignore nan
        label[np.isnan(pc[0]) * (label == 0)] = -1

        return img_rgb, label.astype(np.int32), img_depth, img_cp, img_ocp, pos, rot, pc, obj_mask.astype(np.int32), nonnan_mask, K


class LinemodSIXDCombinedDataset(dataset.DatasetMixin):
    def __init__(self, path, objs_indices, background_path, models_path=None,
                 img_height = 480, img_width = 640,
                 gaussian_noise=False,
                 gamma_augmentation=False,
                 avaraging=False,
                 salt_pepper_noise=False,
                 contrast=False,
                 bg_flip=True,
                 channel_swap = True,
                 random_iteration=False,
                 mode='test',
                 interval=1,
                 resize_rate = 0.5,
                 iteration_per_epoch=1000,
                 metric_filter=1.0):

        self.auto_context_dataset = LinemodSIXDAutoContextDataset(path, objs_indices, background_path,
                                                                  img_height=img_height, img_width=img_width,
                                                                  gaussian_noise=gaussian_noise,
                                                                  gamma_augmentation=gamma_augmentation,
                                                                  avaraging=False,
                                                                  salt_pepper_noise=salt_pepper_noise,
                                                                  contrast=contrast,
                                                                  mode=mode,
                                                                  interval=interval,
                                                                  resize_rate = resize_rate,
                                                                  random_iteration=random_iteration,
                                                                  iteration_per_epoch=iteration_per_epoch,
                                                                  bg_flip=bg_flip,
                                                                  channel_swap = channel_swap,
                                                                  metric_filter=metric_filter)

        self.rendering_dataset = LinemodSIXDRenderingDataset(path, objs_indices, background_path, models_path,
                                                             img_height=img_height, img_width=img_width,
                                                             gaussian_noise=gaussian_noise,
                                                             gamma_augmentation=gamma_augmentation,
                                                             avaraging=False,
                                                             salt_pepper_noise=salt_pepper_noise,
                                                             contrast=contrast,
                                                             resize_rate = resize_rate,
                                                             random_iteration=random_iteration,
                                                             iteration_per_epoch=iteration_per_epoch,
                                                             bg_flip=bg_flip,
                                                             channel_swap = channel_swap,
                                                             metric_filter=metric_filter)
        self.iteration_per_epoch = iteration_per_epoch


    def __len__(self):
        return self.iteration_per_epoch

    def get_example(self, idx):
        if np.random.randint(0, 2):
            return self.auto_context_dataset.get_example(idx)
        else:
            return self.rendering_dataset.get_example(idx)


class LinemodSIXDExtendedDataset(LinemodSIXDDataset):
    def __init__(self, path, objs_indices, img_height = 480, img_width = 640,
                 gaussian_noise=False,
                 gamma_augmentation=False,
                 avaraging=False,
                 salt_pepper_noise=False,
                 contrast=False,
                 mode='test',
                 interval=1,
                 resize_rate = 0.5,
                 metric_filter=1.0):
        scene_indices = np.array([2]) ## benchvise scene id
        super(LinemodSIXDExtendedDataset, self).__init__(path, scene_indices, objs_indices=objs_indices,
                                                         img_height=img_height, img_width=img_width,
                                                         gaussian_noise=gaussian_noise,
                                                         gamma_augmentation=gamma_augmentation,
                                                         avaraging=avaraging,
                                                         salt_pepper_noise=salt_pepper_noise,
                                                         contrast=contrast,
                                                         mode=mode,
                                                         interval=interval,
                                                         resize_rate = resize_rate,
                                                         metric_filter=metric_filter)

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

    def _load_masks(self, scene_id, im_id):
        scene_order = np.where(self.scenes==scene_id)[0][0]
        poses = self.gt_poses[scene_order][im_id]
        masks = np.zeros((self.n_class, self.img_height, self.img_width))
        for pose in poses:
            obj_id = pose['obj_id']
            if obj_id in self.objs:
                idx = np.where(self.objs==obj_id)[0][0]
                mask = cv2.imread(self.mask_fpath_mask.format(scene_id, im_id, obj_id), 0)
                masks[idx] = (mask > 0) * 1  # Scale to [0, 1]
        return masks

    def get_example(self, i):
        scene_id, im_id = self.idx_dict[:, i]
        img_rgb, img_depth = self._load_images(scene_id, im_id)
        masks = self._load_masks(scene_id, im_id)
        pos, rot = self._load_poses(scene_id, im_id)
        K = self._load_k(scene_id, im_id)
        # rgb
        imagenet_mean = np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
        img_rgb = img_rgb - imagenet_mean
        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)
        ## depth
        K = 1.0 * self.out_height / img_depth.shape[0] * K
        img_depth = cv2.resize(img_depth, (self.out_width, self.out_height))
        ## point cloud
        pc = self._get_pointcloud(img_depth, K, fill_nan=True).transpose(2,0,1)
        ## mask
        label = np.zeros((self.out_height, self.out_width))
        obj_mask = np.empty((self.n_class, self.out_height, self.out_width))
        nonnan_mask = np.invert(np.isnan(pc[0])).astype(np.float32)

        img_cp = pos[:, :, np.newaxis, np.newaxis] - pc[np.newaxis, :, :, :]
        img_ocp = np.empty_like(img_cp)

        for idx in six.moves.range(self.n_class):
            mask = masks[idx]
            mask = cv2.resize(mask, (self.out_width, self.out_height))
            obj_mask[idx] = mask
            label[mask.astype(np.bool)] = idx + 1
            img_ocp[idx] = np.dot(rot[idx].T, - img_cp[idx].reshape(3, -1)).reshape(img_cp[idx].shape)

        img_cp = img_cp * obj_mask[:, np.newaxis, :, :].astype(np.bool)
        img_ocp = img_ocp * obj_mask[:, np.newaxis, :, :].astype(np.bool)

        obj_mask = obj_mask * nonnan_mask

        img_cp = img_cp.astype(np.float32)
        img_ocp = img_ocp.astype(np.float32)

        img_cp[img_cp != img_cp] = 0
        img_ocp[img_ocp != img_ocp] = 0

        img_cp[np.abs(img_cp) > self.metric_filter] = 0
        img_ocp[np.abs(img_ocp) > self.metric_filter] = 0

        ## ignore nan
        label[np.isnan(pc[0]) * (label == 0)] = -1

        return img_rgb, label.astype(np.int32), img_depth, img_cp, img_ocp, pos, rot, pc, obj_mask.astype(np.int32), nonnan_mask, K



# visualize dataset demonstration
if __name__== '__main__':
    import matplotlib.pyplot as plt
    from skimage.color import color_dict
    from skimage.color import rgb2gray
    from skimage.color import gray2rgb
    from skimage.util import img_as_float
    alpha=0.3
    image_alpha=1
    root = '../../..'
    train_path = os.path.join(os.getcwd(), root, 'train_data/linemodSIXD2017')
    bg_path = os.path.join(os.getcwd(), root, 'train_data/VOCdevkit/VOC2012/JPEGImages')
    obj_list = np.arange(15) + 1
    # obj_list = np.arange(2) + 6

    visualize_extended_data = True
    visualize_auto_context_data = True
    visualize_rendering_data = True

    if visualize_extended_data:
        extended_dataset = LinemodSIXDExtendedDataset(train_path, obj_list,
                                                      gaussian_noise=False,
                                                      gamma_augmentation=True,
                                                      avaraging=True,
                                                      salt_pepper_noise=False,
                                                      resize_rate=1.0,
                                                      contrast=False)

    if visualize_auto_context_data:
        auto_context_dataset = LinemodSIXDAutoContextDataset(train_path, obj_list, bg_path,
                                                             gaussian_noise=False,
                                                             gamma_augmentation=False,
                                                             avaraging=True,
                                                             salt_pepper_noise=False,
                                                             resize_rate=1.0,
                                                             contrast=False)
    if visualize_rendering_data:
        rendering_dataset = LinemodSIXDRenderingDataset(train_path, obj_list, bg_path,
                                                        gaussian_noise=False,
                                                        gamma_augmentation=True,
                                                        avaraging=True,
                                                        salt_pepper_noise=False,
                                                        resize_rate=1.0,
                                                        contrast=False)

    colors= ('grey', 'red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink',
             'yellowgreen', 'blueviolet', 'brown', 'darkmagenta', 'aqua', 'crimson')
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))

    imsize = (640, 480)
    imagenet_mean = np.array(
        [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]

    for im_id in range(100):
        rgb = np.zeros((imsize[1], imsize[0], 3))
        rgb_ac = np.zeros_like(rgb)
        rgb_ren = np.zeros_like(rgb)
        label = np.zeros((imsize[1], imsize[0]))
        label_ac = np.zeros_like(label)
        label_ren = np.zeros_like(label)

        if visualize_extended_data:
            rgb, label, depth, cp, ocp, pos, rot, _, obj_mask, _, _ = extended_dataset.get_example(im_id)
            rgb = rgb.transpose(1,2,0)* 255 + imagenet_mean
            rgb = rgb.astype(np.uint8)
            rgb[rgb > 255] = 255
            rgb[rgb < 0] = 0

        if visualize_auto_context_data:
            rgb_ac, label_ac, depth_ac, cp_ac, ocp_ac, pos, rot, _, obj_mask_ac, _, _ = auto_context_dataset.get_example(im_id)
            rgb_ac = rgb_ac.transpose(1,2,0)* 255 + imagenet_mean
            rgb_ac = rgb_ac.astype(np.uint8)
            rgb_ac[rgb_ac > 255] = 255
            rgb_ac[rgb_ac < 0] = 0

        if visualize_rendering_data:
            rgb_ren, label_ren, depth_ren, cp_ren, ocp_ren, pos_ren, rot_ren, _, obj_mask_ren, _, _ = rendering_dataset.get_example(im_id)
            rgb_ren = rgb_ren.transpose(1,2,0)* 255 + imagenet_mean
            rgb_ren = rgb_ren.astype(np.uint8)
            rgb_ren[rgb_ren > 255] = 255
            rgb_ren[rgb_ren < 0] = 0

        label_img = np.zeros((label.shape[0], label.shape[1], 3))
        label_img_ac = np.zeros((label_ac.shape[0], label_ac.shape[1], 3))
        label_img_ren = np.zeros((label_ren.shape[0], label_ren.shape[1], 3))
        for lbl_id in range(16):
            if lbl_id > 0:
                color = color_dict[colors[lbl_id]]
                label_img[(label == lbl_id), :] = color
                label_img_ac[(label_ac == lbl_id), :] = color
                label_img_ren[(label_ren == lbl_id), :] = color

        rgb_resize = cv2.resize(rgb, (rgb.shape[1], rgb.shape[0]))
        gray = img_as_float(rgb2gray(rgb_resize))
        gray = gray2rgb(gray) * image_alpha + (1 - image_alpha)
        cls_vis = label_img * alpha + gray * (1 - alpha)
        cls_vis = (cls_vis * 255).astype(np.uint8)

        rgb_resize_ac = cv2.resize(rgb_ac, (rgb_ac.shape[1], rgb_ac.shape[0]))
        gray_ac = img_as_float(rgb2gray(rgb_resize_ac))
        gray_ac = gray2rgb(gray_ac) * image_alpha + (1 - image_alpha)
        cls_vis_ac = label_img_ac * alpha + gray_ac * (1 - alpha)
        cls_vis_ac = (cls_vis_ac * 255).astype(np.uint8)

        rgb_resize_ren = cv2.resize(rgb_ren, (rgb_ren.shape[1], rgb_ren.shape[0]))
        gray_ren = img_as_float(rgb2gray(rgb_resize_ren))
        gray_ren = gray2rgb(gray_ren) * image_alpha + (1 - image_alpha)
        cls_vis_ren = label_img_ren * alpha + gray_ren * (1 - alpha)
        cls_vis_ren = (cls_vis_ren * 255).astype(np.uint8)

        # Clear axes
        for ax in axes.flatten():
            ax.clear()
        if visualize_extended_data:
            axes[0, 0].imshow(rgb[:,:,::-1])
            axes[0, 0].set_title('RGB')
            axes[0, 1].imshow(depth)
            axes[0, 1].set_title('Depth')
            axes[0, 2].imshow(label_img)
            axes[0, 2].set_title('Label')
            axes[0, 3].imshow(cls_vis)
            axes[0, 3].set_title('RGB + Label')
        if visualize_auto_context_data:
            axes[1, 0].imshow(rgb_ac[:,:,::-1].astype(np.uint8))
            axes[1, 0].set_title('RGB')
            axes[1, 1].imshow(depth_ac)
            axes[1, 1].set_title('Depth')
            axes[1, 2].imshow(label_img_ac)
            axes[1, 2].set_title('Label')
            axes[1, 3].imshow(cls_vis_ac)
            axes[1, 3].set_title('RGB + Label')
        if visualize_rendering_data:
            axes[2, 0].imshow(rgb_ren[:,:,::-1])
            axes[2, 0].set_title('RGB')
            axes[2, 1].imshow(depth_ren)
            axes[2, 1].set_title('Depth')
            axes[2, 2].imshow(label_img_ren)
            axes[2, 2].set_title('Label')
            axes[2, 3].imshow(cls_vis_ren)
            axes[2, 3].set_title('RGB + Label')

        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            hspace=0.15, wspace=0.15)
        plt.draw()
        plt.waitforbuttonpress()