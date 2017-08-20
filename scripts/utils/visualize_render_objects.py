#!/usr/bin/env python

import os

from matplotlib import pylab as plt
import cv2
import numpy as np

from cp_net.utils import renderer
from obj_pose_eval import inout, transform

import argparse

def main():
    parser = argparse.ArgumentParser(description='Data Augmentation Samples')
    parser.add_argument('--path', '-p', default="../../train_data/OcclusionChallengeICCV2015/models_ply",
                        help='Path to ply data')
    parser.add_argument('--out', '-o', default="output.png",
                        help='output image name')
    parser.add_argument('--objs', default=['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher'],)
    args = parser.parse_args()


    # load object ply
    models = []
    for obj_name in args.objs:
        print 'load ply : ', obj_name
        models.append(inout.load_ply(os.path.join(args.path, '{}.ply'.format(obj_name))))

    # Camera parameters
    K = np.eye(3)
    K[0, 0] = 500.0 # fx
    K[1, 1] = 500.0 # fy
    K[0, 2] = 250.0 # cx
    K[1, 2] = 250.0 # cy
    im_size = (500, 500)

    # pose
    R = np.zeros((3,3,3))
    R[0] = transform.rotation_matrix(np.pi, (1, 0, 0))[:3, :3]
    R[1] = transform.rotation_matrix(np.pi, (1, 0.5, 0.5))[:3, :3]
    R[2] = transform.rotation_matrix(np.pi, (1, 0, 1))[:3, :3]

    t = np.array([[0, 0, 0.3]]).T

    # output fig
    fig = plt.figure(figsize=(len(models), 3))

    for i in range(len(models)):
        for j in range(3):
            depth = renderer.render(
                models[i], im_size, K, R[j], t, 0.1, 2.0, mode='depth')
            ax = fig.add_subplot(3, len(models), i + len(models) * j + 1)
            ax.imshow(depth)

    plt.show()

if __name__ == '__main__':
    main()