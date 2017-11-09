#!/usr/bin/env python

import numpy as np
import quaternion

import glob
import os
import cv2
import six

from chainer import dataset

from cp_net.utils import inout
from cp_net.utils import multi_object_renderer
import cp_net.utils.preprocess_utils as preprocess_utils


class DualCPNetAutoGenerateDataset(dataset.DatasetMixin):
    def __init__(self, path, background_path, data_indices, img_height = 480, img_width = 640,
                 n_class=9,
                 gaussian_noise=False,
                 gamma_augmentation=False,
                 avaraging=False,
                 salt_pepper_noise=False,
                 contrast=False,
                 random_iteration=False,
                 bg_flip = True,
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

        self.bg_flip = bg_flip

        self.bg_fpaths = glob.glob(os.path.join(background_path, '*.jpg'))

        self.random_iteration = random_iteration

        ## load models
        model_fpath_mask = os.path.join(path, 'models_ply', '{0}.ply')
        dummy_model_fpath_mask = os.path.join(path, 'dummy_models_ply', '{0}.ply')
        self.objs = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
        self.dummy_objs = ['benchviseblue', 'cam', 'iron', 'lamp', 'phone']
        self.models = []
        self.dummy_models = []
        for obj in self.objs:
            print 'Loading data:', obj
            model_fpath = model_fpath_mask.format(obj)
            self.models.append(inout.load_ply(model_fpath))
        for obj in self.dummy_objs:
            print 'Loading data:', obj
            model_fpath = dummy_model_fpath_mask.format(obj)
            self.dummy_models.append(inout.load_ply(model_fpath))

        self.K = np.array([[572.41140, 0, 325.26110],
                           [0, 573.57043, 242.04899],
                           [0, 0, 0]])

        self.metric_filter = metric_filter

    def __len__(self):
        return len(self.data_indices)

    def _get_pointcloud(self, depth_im, K, fill_nan=False):
        xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
        ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T
        Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
        Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])
        xyz_im = np.dstack((Xs, Ys, depth_im))
        if fill_nan:
            xyz_im[xyz_im == 0] = np.nan
        return xyz_im

    def pose_generator(self, min_z=0.3, max_z=1.5, edge_offset=3):
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

    def render_objects(self, min_obj_num=5):
        obj_num = np.random.randint(min_obj_num, len(self.objs))
        dummy_num = np.random.randint(2, len(self.dummy_objs))
        # obj_ind = np.random.randint(0, len(self.objs), obj_num)
        # dummy_ind = np.random.randint(0, len(self.dummy_objs), dummy_num)
        obj_ind = np.random.choice(np.arange(len(self.objs)), obj_num, replace=False)
        dummy_ind = np.random.choice(np.arange(len(self.dummy_objs)), dummy_num, replace=False)
        pos_list = []
        rot_list = []
        model_list = []
        ret_pos = np.zeros((len(self.objs), 3))
        ret_rot = np.zeros((len(self.objs), 3, 3))
        for i, obj_idx in enumerate(obj_ind):
            pos, rot = self.pose_generator()
            pos_list.append(pos.T)
            rot_list.append(rot)
            model_list.append(self.models[obj_idx])
            ret_pos[obj_idx] = pos
            ret_rot[obj_idx] = rot
        for j, dummy_idx in enumerate(dummy_ind):
            pos, rot = self.pose_generator()
            pos_list.append(pos.T)
            rot_list.append(rot)
            model_list.append(self.dummy_models[dummy_idx])
        labels_idx = np.hstack((obj_ind + 1, np.ones(len(dummy_ind)) * -1)).astype(np.int64)
        # labels_idx = obj_ind + 1
        ren_rgb, ren_depth, ren_label  = multi_object_renderer.render(
            model_list, (self.img_width, self.img_height), self.K, rot_list, pos_list, 0.1, 4.0,
            ambient_weight=np.random.rand(),
            labels=labels_idx, mode='rgb+depth+label')
        # data augmentation for RGB
        if self.gaussian_noise and np.random.randint(0,2):
            ren_rgb = preprocess_utils.add_noise(ren_rgb)
        if self.avaraging and np.random.randint(0,2):
            ren_rgb = preprocess_utils.avaraging(ren_rgb)
        rand_gamma = np.random.randint(0, 3)
        if self.gamma_augmentation and rand_gamma:
            if rand_gamma - 1:
                ren_rgb = preprocess_utils.gamma_augmentation(ren_rgb)
            else:
                ren_rgb = preprocess_utils.gamma_augmentation(ren_rgb, gamma=1.5)
        if self.salt_pepper_noise and np.random.randint(0,2):
            ren_rgb = preprocess_utils.salt_pepper_augmentation(ren_rgb)
        rand_contrast = np.random.randint(0, 3)
        if self.contrast and rand_contrast:
            if rand_contrast - 1:
                ren_rgb = self.contrast_server.high_contrast(ren_rgb)
            else:
                ren_rgb = self.contrast_server.low_contrast(ren_rgb)

        return ren_rgb[:,:,::-1], ren_depth, ren_label, ret_pos.astype(np.float32), ret_rot.astype(np.float32)

    def load_bg_data(self, idx):
        bg = cv2.imread(self.bg_fpaths[idx])
        # random crop
        height, width, ch = bg.shape
        resize_height = int((np.random.rand() * 0.5 + 0.5) * height)
        resize_width = int((np.random.rand() * 0.5 + 0.5) * width)
        crop_h = np.floor((height - resize_height) * np.random.rand()).astype(np.int64)
        crop_w = np.floor((width - resize_width) * np.random.rand()).astype(np.int64)

        bg = bg[crop_h:(crop_h + resize_height), crop_w:(crop_w + resize_width)]
        bg = cv2.resize(bg, (self.img_width, self.img_height))

        # data augmentation
        if self.gaussian_noise and np.random.randint(0,2):
            bg = preprocess_utils.add_noise(bg)
        if self.avaraging and np.random.randint(0,2):
            bg = preprocess_utils.avaraging(bg)
        rand_gamma = np.random.randint(0, 3)
        if self.gamma_augmentation and rand_gamma:
            if rand_gamma - 1:
                bg = preprocess_utils.gamma_augmentation(bg)
            else:
                bg = preprocess_utils.gamma_augmentation(bg, gamma=1.5)
        if self.salt_pepper_noise and np.random.randint(0,2):
            bg = preprocess_utils.salt_pepper_augmentation(bg)
        # rand_contrast = np.random.randint(0, 3)
        # if self.contrast and rand_contrast:
        #     if rand_contrast - 1:
        #         bg = self.contrast_server.high_contrast(bg)
        #     else:
        #         bg = self.contrast_server.low_contrast(bg)
        if self.contrast and np.random.randint(0,2):
            bg = self.contrast_server.low_contrast(bg)

        if self.bg_flip and np.random.randint(0,2):
            bg = bg[:,::-1, :]
        return bg

    def get_example(self, i):
        if self.random_iteration:
            ii = np.random.randint(0, len(self.bg_fpaths))
        else:
            ii = self.data_indices[i]
        ren_rgb, img_depth, label_large, pos, rot = self.render_objects()
        # cv2.imwrite("test/test_rgb_{}.png".format(i), ren_rgb)
        bg_rgb = self.load_bg_data(ii)
        K = self.K.copy()
        resize_rate = 0.5

        out_height = int(self.img_height * resize_rate)
        out_width = int(self.img_width * resize_rate)

        img_rgb = ren_rgb + bg_rgb * (label_large == 0)[:, :, np.newaxis]
        label_large[label_large < 0] = 0
        imagenet_mean = np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
        img_rgb = img_rgb - imagenet_mean
        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)

        obj_mask = np.empty((self.n_class - 1, out_height, out_width))

        ## depth
        K = 1.0 * out_height / img_depth.shape[0] * K
        img_depth = cv2.resize(img_depth, (out_width, out_height)).astype(np.float32)
        ## point cloud
        pc = self._get_pointcloud(img_depth, K, fill_nan=False).transpose(2,0,1).astype(np.float32)

        img_cp = pos[:, :, np.newaxis, np.newaxis] - pc[np.newaxis, :, :, :]
        img_ocp = np.empty_like(img_cp)
        label = np.zeros((out_height, out_width))

        for idx in six.moves.range(self.n_class - 1):
            mask = (label_large == idx + 1).astype(np.uint8)
            mask = cv2.resize(mask, (out_width, out_height)).astype(np.bool)
            obj_mask[idx] = mask
            label[mask] = idx + 1
            img_ocp[idx] = np.dot(rot[idx].T, - img_cp[idx].reshape(3, -1)).reshape(img_cp[idx].shape)

        img_cp = img_cp * obj_mask[:, np.newaxis, :, :].astype(np.bool)
        img_ocp = img_ocp * obj_mask[:, np.newaxis, :, :].astype(np.bool)

        # print "---"
        # print pos
        # tmp = obj_mask[:, np.newaxis, : , :] * pc[np.newaxis, : , :, :]
        # print np.sum(tmp, axis=(2,3)) / np.sum(tmp.astype(np.bool), axis=(2,3))
        # print pos - np.sum(tmp, axis=(2,3)) / np.sum(tmp.astype(np.bool), axis=(2,3))

        img_cp[np.abs(img_cp) > self.metric_filter] = 0
        img_ocp[np.abs(img_ocp) > self.metric_filter] = 0
        img_cp = img_cp.astype(np.float32)
        img_ocp = img_ocp.astype(np.float32)
        ## nonnan mask
        nonnan_mask = np.ones((out_height, out_width)).astype(np.float32)

        return img_rgb, label.astype(np.int32), img_depth, img_cp, img_ocp, pos, rot, pc, obj_mask.astype(np.int32), nonnan_mask, K


# visualize dataset demonstration
if __name__== '__main__':
    import matplotlib.pyplot as plt
    from skimage.color.colorlabel import DEFAULT_COLORS
    from skimage.color.colorlabel import color_dict
    root = '../../..'
    train_path = os.path.join(os.getcwd(), root, 'train_data/OcclusionChallengeICCV2015')
    bg_path = os.path.join(os.getcwd(), root, 'train_data/VOCdevkit/VOC2012/JPEGImages')
    im_ids = range(100)
    dataset = DualCPNetAutoGenerateDataset(train_path, bg_path, im_ids,
                                           gaussian_noise=False,
                                           gamma_augmentation=True,
                                           avaraging=True,
                                           salt_pepper_noise=False,
                                           contrast=False)
    argmentation_flg = False
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    for im_id in im_ids:
        img_rgb, label, img_depth, img_cp, img_ocp, pos, rot, pc, obj_mask, nonnan_mask, K = dataset.get_example(im_id)
        imagenet_mean = np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
        img_rgb = img_rgb.transpose(1,2,0)* 255 + imagenet_mean
        label_img = np.zeros((label.shape[0], label.shape[1], 3))
        obj_label = np.zeros((label.shape[0], label.shape[1], 3))
        n_colors = len(DEFAULT_COLORS)
        for lbl_id in range(np.max(label) + 1):
            if lbl_id > 0:
                color = color_dict[DEFAULT_COLORS[lbl_id % n_colors]]
                label_img[(label == lbl_id), :] = color
                obj_label[obj_mask[lbl_id - 1] == True] = color
        # Clear axes
        for ax in axes.flatten():
            ax.clear()
        axes[0, 0].imshow(img_rgb[:,:,::-1].astype(np.uint8))
        axes[0, 0].set_title('RGB image')
        axes[0, 1].imshow(img_depth)
        axes[0, 1].set_title('Depth image')
        axes[1, 0].imshow(label_img)
        axes[1, 0].set_title('Label image')
        axes[1, 1].imshow(obj_label)
        axes[1, 1].set_title('Label image(experimental)')
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            hspace=0.15, wspace=0.15)
        plt.draw()
        plt.waitforbuttonpress()


