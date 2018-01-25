#!/usr/bin/env python

import chainer
from chainer import training

from chainer import cuda
from chainer import optimizers
from chainer import serializers

from chainer.dataset import download

import chainer.links as L
import chainer.functions as F
import chainer.links.model.vision.resnet as R

from chainer.training import extensions
from chainer.links.caffe import CaffeFunction

from cp_net.models.dual_cp_network_ver2 import DualCenterProposalNetworkRes50_predict7
from cp_net.classifiers.dual_cp_classifier import DualCPNetClassifier

from datasets.jsk_pose_estimation_dataset import JSKPoseEstimationAutoContextDataset
from datasets.jsk_pose_estimation_dataset import JSKPoseEstimationDataset

import argparse
import os
import numpy as np


root = '../..'


def _transfer_pretrain_resnet50(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    R._transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    R._transfer_block(src, dst.res3, ['3a', '3b', '3c', '3d'])
    R._transfer_block(src, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    R._transfer_block(src, dst.res5, ['5a', '5b', '5c'])

def _make_chainermodel_npz(path_npz, path_caffemodel, model, num_class):
    print('Now loading caffemodel (usually it may take few minutes)')
    if not os.path.exists(path_caffemodel):
        raise IOError('The pre-trained caffemodel does not exist.')
    caffemodel = CaffeFunction(path_caffemodel)
    chainermodel = DualCenterProposalNetworkRes50_predict7(n_class=num_class)
    _transfer_pretrain_resnet50(caffemodel, chainermodel)
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
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--train_resnet', type=bool, default=False,
                        help='train resnet')
    parser.add_argument('--train-resnet', dest='train_resnet', action='store_true')
    parser.set_defaults(train_resnet=False)
    parser.add_argument('--no-accuracy', dest='compute_acc', action='store_false')
    parser.set_defaults(compute_acc=True)
    parser.add_argument('--no-pose-accuracy', dest='compute_pose_acc', action='store_false')
    parser.set_defaults(compute_pose_acc=True)

    args = parser.parse_args()

    compute_class_accuracy = args.compute_acc
    compute_pose_accuracy = args.compute_pose_acc and args.compute_acc

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# compute class accuracy: {}'.format(compute_class_accuracy))
    print('# compute pose accuracy: {}'.format(compute_pose_accuracy))
    print('')

    im_size = (640, 480)
    objs = np.arange(1) + 1
    n_class = len(objs) + 1

    train_path = os.path.join(os.getcwd(), root, 'train_data/JSK_Objects')
    bg_path = os.path.join(os.getcwd(), root, 'train_data/MS_COCO/train2017')
    # bg_path = os.path.join(os.getcwd(), root, 'train_data/VOCdevkit/VOC2012/JPEGImages')

    caffe_model = 'ResNet-50-model.caffemodel'

    distance_sanity = 0.05
    output_scale = 0.5
    eps = 0.05
    interval = 15

    chainer.using_config('cudnn_deterministic', True)
    model = DualCPNetClassifier(
        DualCenterProposalNetworkRes50_predict7(n_class=n_class, pretrained_model= not args.train_resnet),
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
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # load train data
    train = JSKPoseEstimationAutoContextDataset(train_path, objs, bg_path,
                                                interval=interval,
                                                mode='test',
                                                resize_rate=0.5)

    # load test data
    test = JSKPoseEstimationDataset(train_path, objs,
                                    mode='train',
                                    interval=interval,
                                    resize_rate=0.5,
                                    metric_filter=output_scale + eps)

    print "number of train data : ", train.__len__()
    print "number of test data : ", test.__len__()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    # test_iter = chainer.iterators.MultiprocessIterator(test, args.batchsize,
    #                                                    repeat=False, shuffle=False)


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
        npz_name = 'DualCenterProposalNetworkRes50_jsk_class{}.npz'
        caffemodel_name = 'ResNet-50-model.caffemodel'
        path = os.path.join(root, 'trained_data/', npz_name.format(n_class))
        path_caffemodel = os.path.join(root, 'trained_data/', caffemodel_name)
        print 'npz model path : ' + path
        print 'caffe model path : ' + path_caffemodel
        download.cache_or_load_file(
            path,
            lambda path: _make_chainermodel_npz(path, path_caffemodel, model, n_class),
            lambda path: serializers.load_npz(path, model))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
