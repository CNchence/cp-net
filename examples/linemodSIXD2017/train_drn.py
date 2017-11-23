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

from chainer.training import extensions

from cp_net.models.dual_cp_psp import DualCPNetworkPSPNetBase
from cp_net.models.dual_cp_drn import DualCPDRN
from cp_net.classifiers.dual_cp_classifier import DualCPNetClassifier
from datasets.linemod_sixd2017 import LinemodSIXDAutoContextDataset, LinemodSIXDExtendedDataset, LinemodSIXDCombinedDataset

import convert_pspnet

from sklearn.cross_validation import train_test_split

import argparse
import os
import numpy as np


root = '../..'

def _make_chainermodel_npz(model, path_npz, param_fn, proto_fn, num_class, im_size=(640, 480), psp=False):
    print('Now loading caffemodel (usually it may take few minutes)')
    if not os.path.exists(param_fn):
        raise IOError('The pre-trained caffemodel does not exist.')
    if not os.path.exists(proto_fn):
        raise IOError('prototxt does not exist.')
    param, net = convert_pspnet.get_param_net(param_fn, proto_fn)
    if psp:
        with chainer.using_config('aux_train', True):
            chainermodel = DualCPNetworkPSPNetBase(n_class=num_class, input_size=(640, 480))
    else:
        chainermodel = DualCPDRN(n_class=num_class, input_size=(640, 480))
    chainermodel(np.random.rand(1, 3, im_size[1], im_size[0]).astype(np.float32))
    chainermodel = convert_pspnet.transfer(chainermodel, param, net)
    classifier_model = L.Classifier(chainermodel)
    serializers.save_npz(path_npz, classifier_model, compression=False)
    print('model npz is saved')
    serializers.load_npz(path_npz, model)
    return model


def main():
    parser = argparse.ArgumentParser(description='Fully Convolutional Dual Center Pose Proposal Network for Pose Estimation')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='results/dual_cp',
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
    parser.add_argument('--train-resnet', dest='train_resnet', action='store_true')
    parser.set_defaults(train_resnet=False)
    parser.add_argument('--psp', dest='train_resnet', action='store_true')
    parser.set_defaults(use_psp=False)
    parser.add_argument('--no-accuracy', dest='compute_acc', action='store_false')
    parser.set_defaults(compute_acc=True)
    parser.add_argument('--no-pose-accuracy', dest='compute_pose_acc', action='store_false')
    parser.set_defaults(compute_pose_acc=True)

    args = parser.parse_args()
    use_psp = args.use_psp

    compute_class_accuracy = args.compute_acc
    compute_pose_accuracy = args.compute_pose_acc and args.compute_acc

    chainer.config.aux_train = False

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# compute class accuracy: {}'.format(compute_class_accuracy))
    print('# compute pose accuracy: {}'.format(compute_pose_accuracy))
    print('')

    im_size = (640, 480)
    objs = np.arange(15) + 1
    n_class = len(objs) + 1
    train_path = os.path.join(os.getcwd(), root, 'train_data/linemodSIXD2017')
    # bg_path = os.path.join(os.getcwd(), root, 'train_data/VOCdevkit/VOC2012/JPEGImages')
    bg_path = os.path.join(os.getcwd(), root, 'train_data/MS_COCO/train2017')

    distance_sanity = 0.05
    output_scale = 0.14
    eps = 0.05
    interval = 15

    chainer.using_config('cudnn_deterministic', True)
    if use_psp:
        model = DualCPNetworkPSPNetBase(n_class=n_class)
    else:
        model = DualCPDRN(n_class=n_class)
    model = DualCPNetClassifier(
        model,
        basepath=train_path,
        im_size=im_size,
        distance_sanity=distance_sanity,
        compute_class_accuracy=compute_class_accuracy,
        compute_pose_accuracy=compute_pose_accuracy,
        output_scale=output_scale)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # # load train data
    # train = LinemodSIXDAutoContextDataset(train_path, objs, bg_path,
    #                                       gaussian_noise=True,
    #                                       gamma_augmentation=True,
    #                                       avaraging=True,
    #                                       salt_pepper_noise=True,
    #                                       contrast=False,
    #                                       mode='train',
    #                                       interval=interval,
    #                                       resize_rate = 1.0,
    #                                       metric_filter=output_scale + eps)

    # train = LinemodSIXDExtendedDataset(train_path, objs,
    #                                   mode='train',
    #                                   interval=interval,
    #                                   resize_rate = 1.0,
    #                                   metric_filter=output_scale + eps)

    train = LinemodSIXDCombinedDataset(train_path, objs, bg_path,
                                       mode='train',
                                       interval=interval,
                                       resize_rate=1.0,
                                       metric_filter=output_scale + eps)

    # load test data
    test = LinemodSIXDExtendedDataset(train_path, objs,
                                      mode='test',
                                      interval=interval,
                                      resize_rate = 1.0,
                                      metric_filter=output_scale + eps)

    # train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    # test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
    #                                              repeat=False, shuffle=False)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    test_iter = chainer.iterators.MultiprocessIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=args.gpu)
    evaluator.default_name = 'val'
    trainer.extend(evaluator)

    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot and snapshot object for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='model_iteration-{.updater.iteration}'),
                   trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch',  'main/l_cls',  'main/l_cp', 'main/l_ocp',
         'main/cls_acc', 'main/ocp_acc', 'main/rot_acc', 'main/5cm5deg',
         'val/main/l_cls',  'val/main/l_cp', 'val/main/l_ocp',
         'val/main/cls_acc', 'val/main/ocp_acc', 'val/main/rot_acc', 'val/main/5cm5deg',
         'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)
    else:
        if use_psp:
            npz_name = 'DualCPNetworkPSPNetBase_LinemodSIXD.npz'
        else:
            npz_name = 'DualCPDRN_LinemodSIXD.npz'
        caffemodel_name = 'pspnet101_VOC2012.caffemodel'
        path = os.path.join(root, 'trained_data/', npz_name)
        path_caffemodel = os.path.join(root, 'trained_data/', caffemodel_name)
        path_prototxt = os.path.join(root, 'trained_data/prototxt', 'pspnet101_VOC2012_473.prototxt')
        print 'npz model path : ' + path
        print 'caffe model path : ' + path_caffemodel
        download.cache_or_load_file(
            path,
            lambda path: _make_chainermodel_npz(model, path, path_caffemodel, path_prototxt, n_class, psp=use_psp),
            lambda path: serializers.load_npz(path, model))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()