#!/usr/bin/env python

import argparse
import os
import glob

import chainer
from chainer import cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.util import img_as_float
from skimage.color.colorlabel import DEFAULT_COLORS
from skimage.color.colorlabel import color_dict
from sklearn.cross_validation import train_test_split

from cp_net.models.dual_cp_network_ver2 import DualCenterProposalNetworkRes50_predict7
from cp_net.classifiers.dual_cp_classifier import DualCPNetClassifier
from datasets.dual_cp_dataset import DualCPNetDataset
from cp_net.pose_estimation_interface import SimplePoseEstimationInterface
from cp_net.utils import renderer
from cp_net.utils import inout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('data_path')
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help='if -1, use cpu only (default: 0)')
    args = parser.parse_args()
    alpha=0.3
    image_alpha=1
    render_eps = 0.015

    ## delta for visibility correspondance
    delta = 0.015 # [m]

    objs =['background', 'Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
    n_class = len(objs)
    distance_sanity = 0.05
    min_distance= 0.005
    output_scale = 0.12
    eps = 0.2
    im_size=(512, 384)

    ## load object models
    obj_model_fpath_mask = os.path.join(args.data_path, 'models_ply' , '{0}.ply')
    obj_models = []
    for obj in objs:
        if obj != 'background':
            print 'Loading data:', obj
            obj_model_fpath = obj_model_fpath_mask.format(obj)
            obj_models.append(inout.load_ply(obj_model_fpath))

    ## load network model
    model = DualCenterProposalNetworkRes50_predict7(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    train_range, test_range = train_test_split(np.arange(1213), test_size=100, random_state=1234)
    test = DualCPNetDataset(args.data_path, test_range, img_height = 384, img_width = 512)

    # pose estimator instance
    pei = SimplePoseEstimationInterface(distance_sanity=distance_sanity,
                                        base_path=args.data_path,
                                        min_distance=min_distance, eps=eps, im_size=im_size)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 10))
    n_colors = len(DEFAULT_COLORS)
    im_ids = range(test.__len__())

    imagenet_mean = np.array(
        [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]

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
                cls_pred = chainer.cuda.to_cpu(cls_pred.data)
                y_cls = chainer.cuda.to_cpu(y_cls_d.data)[0]
                y_cp = chainer.cuda.to_cpu(y_cp_d.data)[0]
                y_ocp = chainer.cuda.to_cpu(y_ocp_d.data)[0]

        estimated_ocp, estimated_R = pei.execute(y_cls, y_cp * output_scale, y_ocp * output_scale, img_depth, K)

        img_rgb = (img_rgb.transpose(1,2,0) * 255.0 + imagenet_mean).astype(np.uint8)
        img_rgb_resize = cv2.resize(img_rgb, (img_rgb.shape[1] / 2, img_rgb.shape[0] / 2))
        gray = img_as_float(rgb2gray(img_rgb_resize))
        gray = gray2rgb(gray) * image_alpha + (1 - image_alpha)

        cls_gt = np.zeros_like(img_rgb_resize)
        cls_mask = np.zeros_like(img_rgb_resize)
        # cls ground truth
        for cls_id, cls in enumerate(objs):
            if cls_id != 0:
                color = color_dict[DEFAULT_COLORS[cls_id % n_colors]]
                acls = (cls_pred == cls_id)[:, :, np.newaxis] * color
                acls_gt = (label == cls_id)[:, :, np.newaxis] * color
                cls_mask = cls_mask + acls
                cls_gt = cls_gt + acls_gt
        cls_vis_gt = cls_gt * alpha + gray * (1 - alpha)
        cls_vis_gt = (cls_vis_gt * 255).astype(np.uint8)
        cls_vis = cls_mask * alpha + gray * (1 - alpha)
        cls_vis = (cls_vis * 255).astype(np.uint8)

        pose_mask_gt = np.zeros_like(img_rgb_resize)
        pose_mask_pred = np.zeros_like(img_rgb_resize)
        pose_vis_gt = (gray.copy() * 255).astype(np.uint8)
        pose_vis_pred = (gray.copy() * 255).astype(np.uint8)
        for obj_id, obj_name in enumerate(objs):
            if obj == 'background':
                continue
            # Render the object model
            if np.sum(rot[obj_id - 1]) != 0:
                ren_gt = renderer.render(
                    obj_models[obj_id - 1], (im_size[0] / 2, im_size[1] / 2),  K,
                    rot[obj_id - 1], pos[obj_id - 1].T, 0.1, 4.0, mode='rgb+depth')
                mask = np.logical_and((ren_gt[1] != 0), np.logical_or((ren_gt[1] < img_depth + render_eps), (img_depth == 0)))
                pose_vis_gt[mask, :] = ren_gt[0][mask]
            if np.sum(estimated_R[obj_id - 1]) != 0:
                ren_pred = renderer.render(
                    obj_models[obj_id - 1], (im_size[0] / 2, im_size[1] / 2), K, estimated_R[obj_id - 1],
                    estimated_ocp[obj_id - 1].T, 0.1, 4.0, mode='rgb+depth')
                mask = np.logical_and((ren_pred[1] != 0), np.logical_or((ren_pred[1] < img_depth + render_eps), (img_depth == 0)))
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
        # # plt.pause(0.01)
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()