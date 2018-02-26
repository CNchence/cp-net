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

from cp_net.models.dual_cp_network_ver2 import DualCenterProposalNetworkRes50_predict7
from cp_net.pose_estimation_interface import SimplePoseEstimationInterface, PoseEstimationInterface
from cp_net.utils import renderer
from cp_net.utils import inout

from datasets.jsk_pose_estimation_dataset import JSKPoseEstimationDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help='if -1, use cpu only (default: 0)')
    args = parser.parse_args()
    alpha=0.3
    image_alpha=1
    render_eps = 0.015

    data_path = '../../train_data/JSK_Objects'

    ## delta for visibility correspondance
    delta = 0.015 # [m]
    objs = np.arange(3) + 1
    n_class = len(objs) + 1
    distance_sanity = 0.05
    min_distance= 0.005
    output_scale = 0.5
    prob_eps = 0.4
    eps = 0.05
    interval = 15

    ## load object models
    # obj_model_fpath_mask = os.path.join(data_path, 'models', 'obj_{0:0>2}.ply')
    # obj_models = []
    # for obj in objs:
    #     if obj != 'background':
    #         print 'Loading data: obj_{0}'.format(obj)
    #         obj_model_fpath = obj_model_fpath_mask.format(obj)
    #         obj_models.append(inout.load_ply(obj_model_fpath))

    ## load network model
    model = DualCenterProposalNetworkRes50_predict7(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    test = JSKPoseEstimationDataset(os.path.join(data_path, 'train'),
                                    objs,
                                    mode='train',
                                    interval=interval,
                                    resize_rate=0.5,
                                    metric_filter=output_scale + eps)

    # fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 10))
    im_ids = range(test.__len__())

    # pose estimator instance
    pei = SimplePoseEstimationInterface(distance_sanity=distance_sanity,
                                        min_distance=min_distance, eps=eps)

    # obj_name = []
    # for i in xrange(len(objs)):
    #     obj_name.append("{0:0>2}".format(objs[i]))
    # pei = PoseEstimationInterface(objs=obj_name,
    #                               distance_sanity=distance_sanity,
    #                               base_path=data_path,
    #                               min_distance=min_distance, eps=eps, im_size=im_size)


    imagenet_mean = np.array(
        [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]

    colors= ('grey', 'red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink')

    scores_pos = []
    scores_rot = []
    ids_arr = []
    scores_5cm5deg = []
    scores_10cm10deg = []
    scores_15cm15deg = []
    cnt_ims = np.zeros(n_class -1)
    cnt_5cm5deg = np.zeros(n_class -1)
    cnt_10cm10deg = np.zeros(n_class -1)
    cnt_15cm15deg = np.zeros(n_class -1)
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
        y_pos, y_rot = pei.execute(y_cls, y_cp * output_scale, y_ocp * output_scale, img_depth, K)
        for i in six.moves.range(n_class - 1):
            if np.sum(pos[i]) != 0 and  np.sum(rot[i]) != 0:
                pos_diff = np.linalg.norm(y_pos[i] - pos[i])
                quat = quaternion.from_rotation_matrix(np.dot(y_rot[i].T, rot[i]))
                quat_w = min(1, abs(quat.w))
                diff_angle = np.rad2deg(np.arccos(quat_w)) * 2
                print "obj_{0:0>2} : position_diff = {1}, rotation_diff = {2}".format(i + 1, pos_diff, diff_angle)
                cnt_ims[i] += 1
                if  pos_diff < 0.05 and diff_angle < 5:
                    cnt_5cm5deg[i] +=1
                if  pos_diff < 0.10 and diff_angle < 10:
                    cnt_10cm10deg[i] +=1
                if  pos_diff < 0.15 and diff_angle < 15:
                    cnt_15cm15deg[i] +=1

    print "5cm 5deg metric per object: {}".format(cnt_5cm5deg / cnt_ims)
    print "10cm 10deg metric per object: {}".format(cnt_10cm10deg / cnt_ims)
    print "15cm 15deg metric per object: {}".format(cnt_15cm15deg / cnt_ims)

    print "5cm 5deg metric : {}".format(np.sum(cnt_5cm5deg) / np.sum(cnt_ims))
    print "10cm 10deg metric : {}".format(np.sum(cnt_10cm10deg) / np.sum(cnt_ims))
    print "15cm 15deg metric : {}".format(np.sum(cnt_15cm15deg) / np.sum(cnt_ims))



if __name__ == '__main__':
    main()