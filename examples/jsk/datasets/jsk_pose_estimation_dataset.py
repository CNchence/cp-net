
import glob
import os

from chainer import dataset

import random
import numpy as np
import quaternion
import cv2
from PIL import Image
import six
import yaml

from cp_net.utils import preprocess_utils
from cp_net.utils import inout
from cp_net.utils import multi_object_renderer
from cp_net.utils.imgaug_utils import ImageAugmenter



class JSKPoseEstimationDatasetMixin(dataset.DatasetMixin):
    imagenet_mean = np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
    K = np.array([[570.3422241210938, 0.0, 319.5],
                  [0.0, 570.3422241210938, 239.5],
                  [0.0, 0.0, 1.0]])

    def __init__(self, path, scene_indices, objs_indices=None,
                 img_height=480, img_width=640,
                 mode='test',
                 interval=1,
                 resize_rate = 0.5,
                 metric_filter=1.0):

        if objs_indices is None:
            objs_indices = scene_indices

        self.img_height = img_height
        self.img_width = img_width
        self.resize_rate = resize_rate
        self.out_height = int(img_height * resize_rate)
        self.out_width = int(img_width * resize_rate)

        self.rgb_fpath_mask = os.path.join(path, '{0:0>2}', 'rgb', 'rgb{1:0>4}.png')
        self.depth_fpath_mask = os.path.join(path, '{0:0>2}', 'depth', 'depth{1:0>4}.png')
        self.mask_fpath_mask = os.path.join(path, '{0:0>2}', 'mask', 'mask{1:0>4}.png')
        self.pose_fpath_mask = os.path.join(path, '{0:0>2}', 'pose', 'pose{1:0>4}.yaml')

        self.objs = objs_indices
        self.scenes = scene_indices
        if isinstance(self.objs, int):
            self.n_class = 1
        else:
            self.n_class = len(objs_indices)

        # count number of data
        self.idx_dict = np.array([[], []])
        for i in six.moves.range(len(self.scenes)):
            n_d = len(glob.glob(os.path.join(path, '{0:0>2}'.format(i + 1), 'rgb', '*.png')))
            self.idx_dict = np.hstack((self.idx_dict,
                                       np.vstack((np.ones(n_d) * self.scenes[i],
                                                  np.arange(n_d)))))
        self.idx_dict = self.idx_dict.astype(np.int32)
        print "idx_dict : ", len(self.idx_dict[0])
        if interval > 1:
            if mode == 'train':
                self.idx_dict = self.idx_dict[:, 0::interval]
            elif mode == 'test':
                self.idx_dict = np.delete(self.idx_dict, self.idx_dict[:, 0::interval], axis=1)

        self.metric_filter = metric_filter
        self.imgaug = ImageAugmenter()

    def __len__(self):
        return len(self.idx_dict[0])

    def _load_pose(self, scene_id, im_id):
        pos = np.zeros(3)
        rot = np.zeros((3, 3))
        gt = yaml.load(open(self.pose_fpath_mask.format(scene_id, im_id)))
        pos = np.asarray(gt['position'])
        quat = quaternion.quaternion()
        quat.x = gt['orientation'][0]
        quat.y = gt['orientation'][1]
        quat.z = gt['orientation'][2]
        quat.w = gt['orientation'][3]
        rot = quaternion.as_rotation_matrix(quat)
        return pos, rot

    def _get_pointcloud(self, depth_im, K, fill_nan=False):
        xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
        ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T
        Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
        Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])
        xyz_im = np.dstack((Xs, Ys, depth_im))
        if fill_nan:
            xyz_im[xyz_im == 0] = np.nan
        return xyz_im.astype(np.float32)

    def _load_images(self, scene_id, im_id):
        f_rgb = Image.open(self.rgb_fpath_mask.format(scene_id, im_id))
        f_depth = Image.open(self.depth_fpath_mask.format(scene_id, im_id))
        rgb = f_rgb.convert('RGB')
        depth = f_depth.convert('I')
        rgb = np.asarray(rgb)
        depth = np.asarray(depth) / 1000.0
        #rgb->bgr
        return rgb[:,:,::-1], depth

    def _load_mask(self, scene_id, im_id):
        f_mask = Image.open(self.mask_fpath_mask.format(scene_id, im_id))
        mask = f_mask.convert('I')
        mask = np.asarray(mask) / 255 * scene_id
        return mask


class JSKPoseEstimationAutoContextDataset(JSKPoseEstimationDatasetMixin):
    def __init__(self, path, objs_indices, background_path,
                 img_height=480, img_width=640,
                 mode='train',
                 interval=1,
                 resize_rate = 0.5,
                 metric_filter=1.0,
                 iteration_per_epoch=1000,
                 bg_flip=True,
                 channel_swap=True):
        self.bg_fpaths = glob.glob(os.path.join(background_path, '*.jpg'))
        self.bg_flip = bg_flip
        self.channel_swap = channel_swap
        self.iteration_per_epoch = iteration_per_epoch
        super(JSKPoseEstimationAutoContextDataset, self).__init__(path, objs_indices,
                                                                  mode=mode,
                                                                  img_height=img_height,
                                                                  img_width=img_width,
                                                                  interval=interval,
                                                                  resize_rate=resize_rate,
                                                                  metric_filter=metric_filter)

    def __len__(self):
        return min(self.iteration_per_epoch, len(self.idx_dict[0]))

    def _load_bg_data(self, idx):
        f = Image.open(self.bg_fpaths[idx])
        img = f.convert('RGB')
        width, height = img.size
        resize_height = int((np.random.rand() * 0.5 + 0.5) * height)
        resize_width = int((np.random.rand() * 0.5 + 0.5) * width)
        crop_h = np.floor((height - resize_height) * np.random.rand()).astype(np.int64)
        crop_w = np.floor((width - resize_width) * np.random.rand()).astype(np.int64)
        box = (crop_w, crop_h, crop_w + resize_width, crop_h + resize_height)
        img = img.crop(box)
        img = img.resize((self.img_width, self.img_height))
        bg = np.asarray(img, dtype=np.uint8)
        return bg

    def estimate_visib_region(self, depth, depth_ref):
        visib_mask = np.logical_or((depth < depth_ref),
                                   np.logical_and((depth_ref == 0), (depth != 0)))
        return visib_mask

    def get_example(self, i):
        cv2.setNumThreads(0)
        scene_id, im_id = self.idx_dict[:, i]
        K = self.K.copy()

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
        max_z = 4.0
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
            trans_pos = np.array([x * z / K[0, 0], y * z / K[1, 1], z])
            depth = depth + (z - pos[2])
            # Affine transform(scaling, translate, resize)
            g_x = K[0,2] + pos[0] * K[0,0] / pos[2]
            g_y = K[1,2] + pos[1] * K[1,1] / pos[2]
            M0 = np.float32([[pos[2] / z, 0, x + K[0, 2] - g_x * pos[2]/z],
                             [0, pos[2] / z, y + K[1, 2] - g_y * pos[2]/z]])
            M = M0.copy() * self.resize_rate
            rgb = cv2.warpAffine(rgb, M0, (self.img_width, self.img_height))
            depth = cv2.warpAffine(depth, M, (self.out_width, self.out_height))
            mask = cv2.warpAffine(mask.astype(np.float64), M, (self.out_width, self.out_height))
            cp = cv2.warpAffine(cp, M, (self.out_width, self.out_height))
            # visib mask
            visib_mask = self.estimate_visib_region(depth, img_depth)
            visib_mask = visib_mask * mask
            pilimg = Image.fromarray(np.uint8(visib_mask))
            pilimg = pilimg.resize((self.img_width, self.img_height))
            visib_mask_resize = np.asarray(pilimg).astype(np.bool)
            rgb_mask = np.logical_or(rgb_mask, visib_mask_resize)
            visib_mask = visib_mask.astype(np.bool)

            # masking
            img_rgb[visib_mask_resize, :] = rgb[visib_mask_resize, :]
            img_depth[visib_mask] = depth[visib_mask]
            cp = (cp * visib_mask[:,:, np.newaxis]).transpose(2,0,1)
            cp[np.abs(cp) > self.metric_filter] = 0
            cp[cp != cp] = 0
            img_ocp[obj_order] = np.dot(rot.T, - cp.reshape(3, -1)).reshape(cp.shape)
            img_cp[obj_order] = cp
            obj_mask[obj_order] = visib_mask
            label[visib_mask] = target_obj
            # pose
            positions[obj_order] = trans_pos
            rotations[obj_order] = rot
        bg_id = np.random.randint(0, len(self.bg_fpaths))
        img_bg = self._load_bg_data(bg_id)[:, :, ::-1]
        if self.bg_flip and np.random.randint(0,2):
            img_bg = img_bg[:,::-1, :]
        if self.channel_swap:
            img_bg = img_bg[:, :, np.random.choice(np.arange(3), 3, replace=False)]

        ## random light color
        img_rgb = (img_rgb * (np.random.rand(3) * 0.4 + 0.8)[np.newaxis, np.newaxis, :])
        img_rgb = img_rgb * rgb_mask[:, :, np.newaxis] + img_bg * np.invert(rgb_mask[:, :, np.newaxis].astype(np.bool))
        img_rgb = self.imgaug.augment(img_rgb)
        img_rgb = img_rgb - self.imagenet_mean
        img_rgb = img_rgb / 255.0  # Scale to [0, 1];
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)

        pc = self._get_pointcloud(img_depth, K, fill_nan=True).transpose(2,0,1)
        obj_mask = (obj_mask * nonnan_mask).astype(np.float32)

        img_cp = img_cp.astype(np.float32)
        img_ocp = img_ocp.astype(np.float32)

        return img_rgb, label.astype(np.int32), img_depth, img_cp, img_ocp, positions, rotations, pc, obj_mask.astype(np.int32), nonnan_mask, K



class JSKPoseEstimationDataset(JSKPoseEstimationDatasetMixin):
    def __init__(self, path, scene_indices,
                 img_height=480, img_width=640,
                 mode='test',
                 interval=1,
                 resize_rate = 0.5,
                 metric_filter=1.0):
        super(JSKPoseEstimationDataset, self).__init__(path, scene_indices,
                                                       img_height=img_height,
                                                       img_width=img_width,
                                                       mode=mode,
                                                       interval=interval,
                                                       resize_rate=resize_rate,
                                                       metric_filter=metric_filter)

    def get_example(self, i):
        cv2.setNumThreads(0)
        scene_id, im_id = self.idx_dict[:, i]
        img_rgb, img_depth = self._load_images(scene_id, im_id)
        mask = self._load_mask(scene_id, im_id)
        pos, rot = self._load_pose(scene_id, im_id)
        K = self.K.copy()
        # rgb
        img_rgb = (img_rgb - self.imagenet_mean) / 255.0
        img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)
        ## depth
        K = 1.0 * self.out_height / img_depth.shape[0] * K
        img_depth = cv2.resize(img_depth, (self.out_width, self.out_height))
        ## point cloud
        pc = self._get_pointcloud(img_depth, K, fill_nan=True).transpose(2,0,1)
        ## mask
        nonnan_mask = np.invert(np.isnan(pc[0])).astype(np.float32)
        mask = cv2.resize(mask.astype(np.float64), (self.out_width, self.out_height),
                          interpolation=cv2.INTER_NEAREST)

        img_cp = pos[:, np.newaxis, np.newaxis] - pc[np.newaxis, :, :, :]
        img_cp[img_cp != img_cp] = 0
        img_cp[np.abs(img_cp) > self.metric_filter] = 0
        img_cp = img_cp * mask.astype(np.bool)
        img_ocp = np.empty_like(img_cp)

        img_ocp = np.dot(rot.T, - img_cp.reshape(3, -1)).reshape(img_cp.shape)

        ret_cp = np.zeros((self.n_class, 3, self.out_height, self.out_width))
        ret_ocp = np.zeros((self.n_class, 3, self.out_height, self.out_width))
        ret_cp[scene_id - 1] = img_cp
        ret_ocp[scene_id - 1] = img_ocp
        ret_cp = ret_cp.astype(np.float32)
        ret_ocp = ret_ocp.astype(np.float32)

        ## label
        label = mask * scene_id
        label[np.isnan(pc[0]) * (label == 0)] = -1

        ret_pos = np.zeros((self.n_class, 3))
        ret_rot = np.zeros((self.n_class, 3, 3))
        ret_pos[scene_id - 1] = pos
        ret_rot[scene_id - 1] = rot

        obj_mask = label.copy()
        obj_mask[obj_mask > 0] = 1

        return img_rgb, label.astype(np.int32), img_depth, ret_cp, ret_ocp, pos, rot, pc, obj_mask.astype(np.int32), nonnan_mask, K



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

    train_path = os.path.join(os.getcwd(), root, 'train_data/JSK_Objects')
    bg_path = os.path.join(os.getcwd(), root, 'train_data/VOCdevkit/VOC2012/JPEGImages')
    obj_list = np.arange(1) + 1
    visualize_test_data = True
    visualize_auto_context_data = True

    if visualize_auto_context_data:
        ac_dataset = JSKPoseEstimationAutoContextDataset(train_path, obj_list, bg_path,
                                                         resize_rate=1.0)

    if visualize_test_data:
        test_dataset = JSKPoseEstimationDataset(train_path, obj_list, resize_rate=1.0)

    colors= ('grey', 'red', 'blue', 'yellow', 'magenta',
             'green', 'indigo', 'darkorange', 'cyan', 'pink')
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 10))

    imsize = (640, 480)
    imagenet_mean = np.array(
        [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
    for im_id in range(100):
        rgb = np.zeros((imsize[1], imsize[0], 3))
        rgb_ac = np.zeros_like(rgb)
        if visualize_test_data:
            rgb, label, depth, cp, ocp, pos, rot, _, obj_mask, _, _ = test_dataset.get_example(im_id)
            rgb = rgb.transpose(1,2,0)* 255 + imagenet_mean
            rgb = rgb.astype(np.uint8)
            rgb[rgb > 255] = 255
            rgb[rgb < 0] = 0

        if visualize_auto_context_data:
            rgb_ac, label_ac, depth_ac, cp_ac, ocp_ac, pos, rot, _, obj_mask_ac, _, _ = ac_dataset.get_example(im_id)
            rgb_ac = rgb_ac.transpose(1,2,0)* 255 + imagenet_mean
            rgb_ac = rgb_ac.astype(np.uint8)
            rgb_ac[rgb_ac > 255] = 255
            rgb_ac[rgb_ac < 0] = 0

        label_img = np.zeros((label.shape[0], label.shape[1], 3))
        label_img_ac = np.zeros((label_ac.shape[0], label_ac.shape[1], 3))

        for lbl_id in range(5):
            if lbl_id > 0:
                color = color_dict[colors[lbl_id]]
                label_img[(label == lbl_id), :] = color
                label_img_ac[(label_ac == lbl_id), :] = color

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

        # Clear axes
        for ax in axes.flatten():
            ax.clear()
        if visualize_test_data:
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

        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            hspace=0.15, wspace=0.15)
        plt.draw()
        plt.waitforbuttonpress()
