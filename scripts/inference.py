#!/usr/bin/env python


from cp_classifier import CPNetClassifier

import argparse
import os
import numpy as np


class CPNetInference(object):
    def __init__(model):
        self.gpu = gpu
        self.model = CenterProposalNetworkRes50FCN(n_class=n_class, pretrained_model=True)

        if self.gpu != -1:
            self.model.to_gpu(self.gpu)

    def inference(rgb, depth):
        prob_map, position_map, rot_map = self.model(rgb, depth)
        return prob_map, position_map, rot_map


    def pose_estimation(rgb, depth, min_size=10, prob_thre=0.20):
        prob, pos, rot = self.inference(rgb, depth)
        # 0. calculate number of object
        # 1. clastering pos by k-means (use object prob mask)
        #  ransacでもいい??
        # 2. vote pos and rot
        # 3. return object label and pose
        return obj_label, pose

def main():
    parser = argparse.ArgumentParser(description='Fully Convolutional Center Pose Proposal Network for Pose Estimation')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')

    args = parser.parse_args()

    model = hoge
    cpn_infer = CPNetInference(model)


if __name__ == '__main__':
    main()
