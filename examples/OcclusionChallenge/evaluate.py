#!/usr/bin/env python

import argparse
import os
import glob
import six

import chainer
from chainer import cuda

import numpy as np
import quaternion

import matplotlib.pyplot as plt

from cp_net.models.dual_cp_network_ver2 import DualCenterProposalNetworkRes50_predict7
from cp_net.classifiers.dual_cp_classifier import DualCPNetClassifier
from datasets.dual_cp_dataset import DualCPNetDataset
from cp_net.pose_estimation_interface import SimplePoseEstimationInterface

import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('data_path')
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help='if -1, use cpu only (default: 0)')
    parser.add_argument('-o', '--output', default='False',
                        help='if -1, use cpu only (default: 0)')
    args = parser.parse_args()

    ## delta for visibility correspondance
    delta = 0.015 # [m]

    objs =['background', 'Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
    n_class = len(objs)
    distance_sanity = 0.05
    min_distance= 0.005
    output_scale = 0.12
    eps = 0.2
    im_size=(640, 480)

    ## load network model
    model = DualCenterProposalNetworkRes50_predict7(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # pose estimator instance
    pei = SimplePoseEstimationInterface(distance_sanity=distance_sanity,
                                        base_path=args.data_path,
                                        min_distance=min_distance, eps=eps, im_size=im_size)

    # data_size = 1213
    data_size = 5
    im_ids = range(data_size)
    test = DualCPNetDataset(args.data_path, im_ids, img_height = 480, img_width = 640)

    hist = np.zeros((n_class, n_class))
    pos_diffs = []
    rot_diffs = []

    for im_id in tqdm.trange(data_size):
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

        y_pos, y_rot = pei.execute(y_cls, y_cp * output_scale, y_ocp * output_scale, img_depth, K)

        for i in six.moves.range(n_class - 1):
            if np.sum(pos[i]) != 0 and  np.sum(rot[i]) != 0:
                pos_diff = np.linalg.norm(y_pos[i] - pos[i])
                quat = quaternion.from_rotation_matrix(np.dot(y_rot[i].T, rot[i]))
                quat_w = min(1, abs(quat.w))
                diff_angle = np.rad2deg(np.arccos(quat_w)) * 2
                pos_diffs.append(pos_diff)
                rot_diffs.append(diff_angle)

    ## label accuracy
    mask = (label.ravel() >= 0) & (cls_pred.ravel() < n_class)
    ahist = np.bincount(
        n_class * label.ravel()[mask].astype(int) + cls_pred.ravel()[mask],
        minlength=n_class ** 2).reshape(n_class, n_class)
    hist += ahist

    pos_diffs = np.asarray(pos_diffs)
    rot_diffs = np.asarray(rot_diffs)
    ## todo : write code for calculating accuracy
    print "--- results ---"
    print pos_diffs
    print rot_diffs
    acc = np.sum(hist * np.diag(np.ones(n_class))) / np.sum(hist)
    print 'classification accuracy : {}'.format(acc)

    if args.output:
        np.save('host.npy', hist)
        np.save('pos_diff.npy', pos_diffs)
        np.save('rot_diff.npy', rot_diffs)

if __name__ == '__main__':
    main()