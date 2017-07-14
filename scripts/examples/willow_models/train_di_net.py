#!/usr/bin/env python

import chainer
from chainer import training

from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer import link

from chainer.dataset import download

import chainer.links as L
import chainer.functions as F
import chainer.links.model.vision.resnet as R

from chainer.training import extensions
from chainer.links.caffe import CaffeFunction

from cp_net.models.depth_invariant_network import DepthInvariantNetworkRes50FCN
from cp_net.models.depth_invariant_network_v2 import DepthInvariantNetworkRes50FCNVer2
from cp_net.di_net_dataset import DepthInvariantNetDataset

import argparse
import os
import numpy as np

def _transfer_pretrain_resnet50(src, dst, use_res5=True):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    R._transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    R._transfer_block(src, dst.res3, ['3a', '3b', '3c', '3d'])
    R._transfer_block(src, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    if use_res5:
        R._transfer_block(src, dst.res5, ['5a', '5b', '5c'])

def _make_chainermodel_npz(path_npz, path_caffemodel, model, num_class, v2=True):
    print('Now loading caffemodel (usually it may take few minutes)')
    if not os.path.exists(path_caffemodel):
        raise IOError('The pre-trained caffemodel does not exist.')
    caffemodel = CaffeFunction(path_caffemodel)
    if v2:
        chainermodel = DepthInvariantNetworkRes50FCNVer2(n_class=num_class)
        _transfer_pretrain_resnet50(caffemodel, chainermodel, use_res5=False)
    else:
        chainermodel = DepthInvariantNetworkRes50FCN(n_class=num_class)
        _transfer_pretrain_resnet50(caffemodel, chainermodel, use_res5=True)
    classifier_model = L.Classifier(chainermodel)
    serializers.save_npz(path_npz, classifier_model, compression=False)
    print('model npz is saved')
    serializers.load_npz(path_npz, model)
    return model


def main():
    parser = argparse.ArgumentParser(description='Fully Convolutional Depth Invariant Network')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='di_net_result',
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
    parser.add_argument('--train_resnet', type=bool, default=False,
                        help='train resnet')
    parser.add_argument('--ver2', type=bool, default=True,
                        help='di net version 2')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    n_class = 20
    # n_class = 36
    n_view = 37
    train_path = os.path.join(os.getcwd(), '../../../train_data/willow_models')
    caffe_model = 'ResNet-50-model.caffemodel'

    chainer.using_config('cudnn_deterministic', True)

    if args.ver2:
        model = L.Classifier(
            DepthInvariantNetworkRes50FCNVer2(n_class=n_class,
                                              pretrained_model= not args.train_resnet))
    else:
        model = L.Classifier(
            DepthInvariantNetworkRes50FCN(n_class=n_class,
                                          pretrained_model= not args.train_resnet))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # load train data
    train = DepthInvariantNetDataset(train_path, range(1,n_class), range(0, n_view - 2),
                                     random_resize=True, resize_train=True,
                                     force_resize=True)
    # load test data
    test = DepthInvariantNetDataset(train_path, range(1,n_class), range(n_view - 2, n_view),
                                    img_size=(256, 192), random=False, random_flip=False)

    test_resized = DepthInvariantNetDataset(train_path, range(1,n_class),
                                            range(n_view - 2, n_view),
                                            img_size=(256, 192), random=False,
                                            random_flip=False, random_resize=True,
                                            force_resize=True)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    test_resized_iter = chainer.iterators.SerialIterator(test_resized, args.batchsize,
                                                         repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=args.gpu)
    evaluator.default_name = 'val'
    trainer.extend(evaluator)

    evaluator_resized = extensions.Evaluator(test_resized_iter, model, device=args.gpu)
    evaluator_resized.default_name = 'resized_val'
    trainer.extend(evaluator_resized)

    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    # if extensions.PlotReport.available():
        # trainer.extend(
        #     extensions.PlotReport(['main/loss'],
        #                           'epoch', file_name='loss.png'))
        # trainer.extend(
        #     extensions.PlotReport(
        #         ['main/accuracy'],
        #         'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch',  'main/loss', 'main/accuracy',
         'val/main/loss','val/main/accuracy',
         'resized_val/main/loss','resized_val/main/accuracy',
         'elapsed_time']))


    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
    else:
        root = '../../../'
        if args.ver2:
            npz_name = 'DepthInvariantNetworkRes50FCNVer2.npz'
        else:
            npz_name = 'DepthInvariantNetworkRes50FCN.npz'
        caffemodel_name = 'ResNet-50-model.caffemodel'
        path = os.path.join(root, 'trained_data/', npz_name)
        path_caffemodel = os.path.join(root, 'trained_data/', caffemodel_name)
        print 'npz model path : ' + path
        print 'caffe model path : ' + path_caffemodel
        download.cache_or_load_file(
            path,
            lambda path: _make_chainermodel_npz(path, path_caffemodel,
                                                model, n_class, v2=args.ver2),
            lambda path: serializers.load_npz(path, model))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
