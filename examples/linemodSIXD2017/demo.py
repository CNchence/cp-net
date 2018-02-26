#!/usr/bin/env python

import argparse
import os
import glob
import six

import chainer
from chainer import cuda
import numpy as np
import quaternion
import cv2
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.util import img_as_float
from skimage.color.colorlabel import color_dict
from sklearn.cross_validation import train_test_split

from cp_net.models.dual_cp_network_ver2 import DualCenterProposalNetworkRes50_predict7
from cp_net.classifiers.dual_cp_classifier import DualCPNetClassifier
from datasets.linemod_sixd2017 import LinemodSIXDAutoContextDataset, LinemodSIXDExtendedDataset

from cp_net.pose_estimation_interface import SimplePoseEstimationInterface, PoseEstimationInterface
from cp_net.utils import renderer
from cp_net.utils import inout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('--auto-data', dest='auto_data', action='store_true')
    parser.set_defaults(auto_data=False)
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help='if -1, use cpu only (default: 0)')
    args = parser.parse_args()
    alpha=0.3
    image_alpha=1
    render_eps = 0.015

    data_path = '../../train_data/linemodSIXD2017'
    bg_path = '../../train_data/VOCdevkit/VOC2012/JPEGImages'

    ## delta for visibility correspondance
    delta = 0.015 # [m]
    # objs =['background', 'Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
    objs = np.arange(15) + 1
    n_class = len(objs) + 1
    distance_sanity = 0.05
    min_distance= 0.005
    output_scale = 0.14
    prob_eps = 0.4
    eps = 0.05
    im_size=(640, 480)
    interval = 15

    K_orig = np.array([[572.41140000, 0.00000000, 325.26110000],
                       [0.00000000, 573.57043000, 242.04899000],
                       [0.00000000, 0.00000000, 1.00000000]])

    ## load object models
    obj_model_fpath_mask = os.path.join(data_path, 'models', 'obj_{0:0>2}.ply')
    obj_models = []
    for obj in objs:
        if obj != 'background':
            print 'Loading data: obj_{0}'.format(obj)
            obj_model_fpath = obj_model_fpath_mask.format(obj)
            obj_models.append(inout.load_ply(obj_model_fpath))

    ## load network model
    model = DualCenterProposalNetworkRes50_predict7(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    test = LinemodSIXDExtendedDataset(data_path, objs,
                                      mode='test',
                                      interval=interval,
                                      metric_filter=output_scale + eps)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 10))
    im_ids = range(test.__len__())

    # pose estimator instance
    # pei = SimplePoseEstimationInterface(distance_sanity=distance_sanity,
    #                                     base_path=data_path,
    #                                     min_distance=min_distance, eps=eps, im_size=im_size)

    pei = PoseEstimationInterface(objs= ['obj_{0:0>2}'.format(i) for i in objs],
                                  base_path=data_path,
                                  distance_sanity=distance_sanity,
                                  model_scale = 1000.0,
                                  model_partial= 1,
                                  min_distance=min_distance, eps=prob_eps, im_size=im_size)

    imagenet_mean = np.array(
        [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
    colors= ('grey', 'red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink',
             'yellowgreen', 'blueviolet', 'brown', 'darkmagenta', 'aqua', 'crimson')

    for im_id in im_ids:
        print "executing {0} / {1}".format(im_id, test.__len__())
        img_rgb, label, img_depth, img_cp, img_ocp, pos, rot, pc, obj_mask, nonnan_mask, K = test.get_example(im_id)
        x_data = np.expand_dims(img_rgb, axis=0)
        with chainer.no_backprop_mode():
            if args.gpu >= 0:
                x_data = cuda.to_gpu(x_data)
            x = chainer.Variable(x_data)
            with chainer.using_config('train', False):
                y_cls_d, y_cp_d, y_ocp_d = model(x)
                cls_pred = chainer.functions.argmax(y_cls_d, axis=1)[0]
                cls_prob = chainer.functions.max(y_cls_d, axis=1)[0]
                cls_pred = chainer.cuda.to_cpu(cls_pred.data)
                cls_prob = chainer.cuda.to_cpu(cls_prob.data)
                y_cls = chainer.cuda.to_cpu(y_cls_d.data)[0]
                y_cp = chainer.cuda.to_cpu(y_cp_d.data)[0]
                y_ocp = chainer.cuda.to_cpu(y_ocp_d.data)[0]

        cls_pred[cls_prob < prob_eps] = 0

        y_pos, y_rot = pei.execute(y_cls, y_cp * output_scale, y_ocp * output_scale, img_depth, K)

        img_rgb = (img_rgb.transpose(1,2,0) * 255.0 + imagenet_mean).astype(np.uint8)
        img_rgb_resize = cv2.resize(img_rgb, (img_rgb.shape[1] / 2, img_rgb.shape[0] / 2))
        # gray = img_as_float(rgb2gray(img_rgb_resize))
        gray = img_as_float(rgb2gray(img_rgb))
        gray = gray2rgb(gray) * image_alpha + (1 - image_alpha)

        cls_gt = np.zeros_like(img_rgb_resize)
        cls_mask = np.zeros_like(img_rgb_resize)
        # cls ground truth
        for cls_id, cls in enumerate(objs):
            if cls_id != 0:
                color = color_dict[colors[cls_id]]
                acls = (cls_pred == cls_id)[:, :, np.newaxis] * color
                acls_gt = (label == cls_id)[:, :, np.newaxis] * color
                cls_mask = cls_mask + acls
                cls_gt = cls_gt + acls_gt

        cls_mask = cv2.resize((cls_mask * 255).astype(np.uint8), im_size, interpolation=cv2.INTER_NEAREST)
        cls_gt = cv2.resize((cls_gt * 255).astype(np.uint8), im_size, interpolation=cv2.INTER_NEAREST)

        cls_vis_gt = cls_gt * alpha + gray * (1 - alpha)
        cls_vis_gt = (cls_vis_gt * 255).astype(np.uint8)
        cls_vis = cls_mask * alpha + gray * (1 - alpha)
        cls_vis = (cls_vis * 255).astype(np.uint8)

        pose_vis_gt = (gray.copy() * 255).astype(np.uint8)
        pose_vis_pred = (gray.copy() * 255).astype(np.uint8)
        vis_render = True

        for i in six.moves.range(n_class - 1):
            if np.sum(pos[i]) != 0 and  np.sum(rot[i]) != 0:
                pos_diff = np.linalg.norm(y_pos[i] - pos[i])
                quat = quaternion.from_rotation_matrix(np.dot(y_rot[i].T, rot[i]))
                quat_w = min(1, abs(quat.w))
                diff_angle = np.rad2deg(np.arccos(quat_w)) * 2
                print "obj_{0:0>2} : position_diff = {1}, rotation_diff = {2}".format(i + 1, pos_diff, diff_angle)

            if not vis_render:
                continue
            # Render the object model
            if np.sum(rot[i]) != 0:
                ren_gt = renderer.render(
                    obj_models[i], (im_size[0] / 2, im_size[1] / 2),  K,
                    rot[i], (pos[i].T * 1000), 100, 4000, mode='rgb+depth')
                mask = np.logical_and((ren_gt[1] != 0), np.logical_or((ren_gt[1] / 1000.0 < img_depth + render_eps), (img_depth == 0)))
                pose_vis_gt[mask, :] = ren_gt[0][mask]
            if np.sum(y_rot[i]) != 0:
                ren_pred = renderer.render(
                    obj_models[i], (im_size[0] / 2, im_size[1] / 2), K, y_rot[i],
                    (y_pos[i].T * 1000), 100, 4000, mode='rgb+depth')
                mask = np.logical_and((ren_pred[1] != 0), np.logical_or((ren_pred[1] / 1000.0 < img_depth + render_eps), (img_depth == 0)))
                pose_vis_pred[mask, :]  = ren_pred[0][mask]

        # Clear axes
        for ax in axes.flatten():
            ax.clear()
        axes[0, 0].imshow(img_rgb[:,:,::-1].astype(np.uint8))
        axes[0, 0].set_title('RGB image')
        axes[0, 1].imshow(img_depth)
        axes[0, 1].set_title('Depth image')
        axes[1, 0].imshow(cls_vis_gt)
        axes[1, 0].set_title('class gt')
        axes[1, 1].imshow(cls_vis)
        axes[1, 1].set_title('class pred')
        axes[2, 0].imshow(pose_vis_gt)
        axes[2, 0].set_title('pose gt vis')
        axes[2, 1].imshow(pose_vis_pred)
        axes[2, 1].set_title('pose pred vis')

        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.15, wspace=0.15)
        plt.draw()
        plt.pause(0.01)
        # plt.waitforbuttonpress()


if __name__ == '__main__':
    main()