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
from datasets.linemod_sixd2017 import LinemodSIXDExtendedDataset

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

    objs = np.arange(15) + 1
    n_class = len(objs) + 1
    distance_sanity = 0.05
    output_scale = 0.14
    eps = 0.4
    im_size=(640, 480)

    ## load network model
    model = DualCenterProposalNetworkRes50_predict7(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # data_size = 1213
    data_size = 50
    im_ids = range(data_size)
    test = LinemodSIXDExtendedDataset(args.data_path, objs,
                                      gaussian_noise=False,
                                      gamma_augmentation=True,
                                      avaraging=True,
                                      salt_pepper_noise=False,
                                      resize_rate=0.5,
                                      contrast=False)

    hist = np.zeros((n_class, n_class))

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

        ## label accuracy
        mask = (label.ravel() >= 0) & (cls_pred.ravel() < n_class)
        ahist = np.bincount(
            n_class * label.ravel()[mask].astype(int) + cls_pred.ravel()[mask],
            minlength=n_class ** 2).reshape(n_class, n_class)
        hist += ahist

    ## todo : write code for calculating accuracy
    print "--- results ---"
    acc = np.sum(hist * np.diag(np.ones(n_class))) / np.sum(hist)
    print 'classification accuracy : {}'.format(acc)
    cls_acc = np.diag(hist) / (np.sum(hist, axis=1) + 1e-15)
    for i in range(n_class):
        print 'class accuracy obj_{0:0>2}: {1}'.format(i, cls_acc[i])

    print "-- confusion matrix --"
    np.set_printoptions(threshold=np.inf,linewidth=np.inf)
    print hist / np.sum(hist, axis=1, keepdims=True)

    if args.output:
        np.save('host.npy', hist)

if __name__ == '__main__':
    main()