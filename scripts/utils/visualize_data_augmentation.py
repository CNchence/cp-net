#!/usr/bin/env python

# import os, sys,mat

from matplotlib import pylab as plt
import cv2
import numpy as np

from cp_net.utils import preprocess_utils

import argparse

def main():
    parser = argparse.ArgumentParser(description='Data Augmentation Samples')
    parser.add_argument('--path', '-p', default="Lenna.png",
                        help='Path to sample image')
    parser.add_argument('--out', '-o', default="output.png",
                        help='output image name')
    args = parser.parse_args()

    # load sample image
    src = cv2.imread(args.path, 1)
    # bgr -> rgb
    src = src[:,:,::-1]

    # output fig
    fig = plt.figure(figsize=(5, 2))

    # original image
    ax = fig.add_subplot(2, 5, 1)
    ax.imshow(src)

    ## data augmentation
    noised_src = preprocess_utils.add_noise(src)
    ax = fig.add_subplot(2, 5, 2)
    ax.imshow(noised_src)
    # noise image is not visualized correctly, please check png image
    # cv2.imwrite('noise.png', noised_src[:,:,::-1])

    avarage_src = preprocess_utils.avaraging(src)
    ax = fig.add_subplot(2, 5, 3)
    ax.imshow(avarage_src)

    avarage_src2 = preprocess_utils.avaraging(src, 10)
    ax = fig.add_subplot(2, 5, 4)
    ax.imshow(avarage_src2)

    gamma_src1 = preprocess_utils.gamma_augmentation(src)
    ax = fig.add_subplot(2, 5, 5)
    ax.imshow(gamma_src1)

    gamma_src2 = preprocess_utils.gamma_augmentation(src, gamma=1.5)
    ax = fig.add_subplot(2, 5, 6)
    ax.imshow(gamma_src2)

    sp_src = preprocess_utils.salt_pepper_augmentation(src)
    ax = fig.add_subplot(2, 5, 7)
    ax.imshow(sp_src)

    sp_src2 = preprocess_utils.salt_pepper_augmentation(src, sp_rate=0.0)
    ax = fig.add_subplot(2, 5, 8)
    ax.imshow(sp_src2)

    ca = preprocess_utils.ContrastAugmentation()
    hc_src = ca.high_contrast(src)
    ax = fig.add_subplot(2, 5, 9)
    ax.imshow(hc_src)

    lc_src = ca.low_contrast(src)
    ax = fig.add_subplot(2, 5, 0)
    ax.imshow(lc_src)

    plt.show()


if __name__ == '__main__':
    main()

