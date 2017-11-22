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
from skimage.color.colorlabel import color_dict
from sklearn.cross_validation import train_test_split

from cp_net.models.dual_cp_network_ver2 import DualCenterProposalNetworkRes50_predict7
from cp_net.classifiers.dual_cp_classifier import DualCPNetClassifier
from datasets.linemod_sixd2017 import LinemodSIXDAutoContextDataset, LinemodSIXDExtendedDataset

from cp_net.pose_estimation_interface import SimplePoseEstimationInterface
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

    ## load object models
    obj_model_fpath_mask = os.path.join(data_path, 'models', 'obj_{0:0>2}.ply')
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

    test = LinemodSIXDExtendedDataset(data_path, objs,
                                      mode='test',
                                      interval=interval,
                                      metric_filter=output_scale + eps)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    im_ids = range(test.__len__())

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

        cls_pred[cls_prob < prob_eps] = 0

        img_rgb = (img_rgb.transpose(1,2,0) * 255.0 + imagenet_mean).astype(np.uint8)
        img_rgb_resize = cv2.resize(img_rgb, (img_rgb.shape[1] / 2, img_rgb.shape[0] / 2))
        gray = img_as_float(rgb2gray(img_rgb_resize))
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
        cls_vis_gt = cls_gt * alpha + gray * (1 - alpha)
        cls_vis_gt = (cls_vis_gt * 255).astype(np.uint8)
        cls_vis = cls_mask * alpha + gray * (1 - alpha)
        cls_vis = (cls_vis * 255).astype(np.uint8)

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

        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.15, wspace=0.15)
        plt.draw()
        # # plt.pause(0.01)
        plt.waitforbuttonpress()


if __name__ == '__main__':
    main()