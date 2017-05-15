#!/usr/bin/env python

import chainer
from chainer import training

from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable

import chainer.links as L
import chainer.functions as F
from chainer.training import extensions
from chainer.datasets import tuple_dataset

from cp_network import CenterProposalNetworkRes50FCN
from cp_network import CenterProposalNetworkRes50FCNtest

import argparse
import cv2
import os

import math
import numpy as np
from chainer.functions.array import concat

def class_pose_multi_loss(y, t):
    cls, pos = y
    t_cls_tmp, t_pos_tmp = F.separate(t, 1)
    t_cls = Variable(np.array([]).astype(np.float32))
    t_pos = Variable(np.array([]).astype(np.float32))
    for i in range(len(t_cls_tmp)):
        t_cls = concat.concat((t_cls, F.flatten(t_cls_tmp[i].data.astype(np.float32))), 0)
        t_pos = concat.concat((t_pos, F.flatten(t_pos_tmp[i].data.astype(np.float32))), 0)
    t_cls = F.reshape(F.flatten(t_cls), (cls.shape[0], cls.shape[2], cls.shape[3]))
    t_pos = F.reshape(F.flatten(t_pos), pos.shape)
    t_cls = Variable(t_cls.data.astype(np.int32))
    l_pos = F.mean_squared_error(pos, t_pos)
    l_cls = Variable(np.array([],dtype=np.float32))
    l_cls = F.softmax_cross_entropy(cls, t_cls)
    return  l_cls + l_pos

def load_train_data(path, num_class, num_view, img_size= (480, 640)):

    rgb = np.zeros(3 * img_size[0] * img_size[1] * (num_class - 1)* num_view)
    rgb = rgb.reshape((num_class - 1)*num_view, 3, img_size[1], img_size[0])

    depth = np.zeros(img_size[0] * img_size[1] * (num_class - 1) * num_view)
    depth = depth.reshape((num_class - 1)*num_view, 1, img_size[1], img_size[0])

    t_cls = np.zeros( img_size[0] * img_size[1] * (num_class - 1) * num_view)
    t_cls = t_cls.reshape((num_class - 1)*num_view, 1, img_size[1], img_size[0])

    t_pose = np.zeros(3 * img_size[0] * img_size[1] * (num_class - 1) * num_view)
    t_pose = t_pose.reshape((num_class - 1) *num_view, 3, img_size[1], img_size[0])

    for i in range(num_class -1):
        c_idx =  '{0:02d}'.format(i+1)
        c_path = os.path.join(path, 'object_' + c_idx)
        for j in range(num_view):
            v_idx = '{0:08d}'.format(j)
            ## rgb
            img = cv2.imread(os.path.join(c_path, 'rgb_' + v_idx +'.png'))
            img = cv2.resize(img, img_size)
            rgb[i * num_view + j] = img.transpose(2, 0, 1)

            ## depth
            d_img = cv2.imread(os.path.join(c_path, 'depth_' + v_idx +'.png'), 1)
            d_img = cv2.resize(img, img_size)
            depth[i * num_view + j] = d_img.transpose(2, 0, 1)[0]

            ##  mask
            mask = cv2.imread(os.path.join(c_path, 'mask_' + v_idx +'.png'), 1) / 255.0

            ## center pose with mask
            dist = np.load(os.path.join(c_path, 'dist_' + v_idx +'.npy'))
            dist = cv2.resize(mask * dist, img_size)
            # dist[dist != dist] = -1.0 ## non-nan
            for ii in range(img_size[1]):
                for jj in range(img_size[0]):
                    if math.isnan(dist[ii][jj][0]) or math.isnan(dist[ii][jj][0]) or math.isnan(dist[ii][jj][0]):
                        dist[ii][jj] = [-1, -1, -1]
            t_pose[i * num_view + j] = dist.transpose(2, 0, 1)

            ## class prob
            mask = cv2.resize(mask, img_size)
            t_cls[i * num_view + j][0] = mask.transpose(2,0,1)[0] * (i + 1)


    return rgb.astype(np.float32), depth.astype(np.float32), t_cls.astype(np.int32), t_pose.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Fully Convolutional Center Pose Proposal Network for Pose Estimation')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    n_class = 10
    n_view = 3
    #  n_view = 37
    train_path = os.path.join(os.getcwd(), '../train_data/willow_models')

    model = L.Classifier(CenterProposalNetworkRes50FCN(n_class=n_class), lossfun=class_pose_multi_loss)
    model.compute_accuracy = False
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)


    # load pre-train Network
    #
    # TODO
    #

    print('load train data')
    # load train data
    train_rgb, train_depth, train_cls, train_pose = load_train_data(train_path, n_class, n_view)
    # load test data
    # test_rgb, test_depth, test_cls, test_pose = load_test_data(test_path)

    t_append = []
    for i in range(len(train_cls)):
        t_append.append([train_cls[i], train_pose[i]])
    # t_append = np.array(t_append).astype(np.float32)
    train = tuple_dataset.TupleDataset(train_rgb, train_depth, t_append)
    # train = tuple_dataset.TupleDataset(train_rgb, train_depth, train_pose)
    # test = tuple_dataset.TupleDataset((test_rgb, test_depth), (test_cls, test_pose))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    # test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
    #                                              repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


    # Evaluate the model with the test dataset for each epoch
    # trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss'],
                                  'epoch', file_name='loss.png'))
        # trainer.extend(
        #     extensions.PlotReport(
        #         ['main/accuracy'],
        #         'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'elapsed_time']))


    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
    else:
        # load pre-train model
        pass
    
    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()

