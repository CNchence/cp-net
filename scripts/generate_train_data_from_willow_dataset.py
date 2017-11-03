#!/usr/bin/env python

import warnings
import os
import glob
import argparse
import cv2
import numpy as np

img_width = 640
img_height = 480

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_data_path')
    parser.add_argument('generated_data_path')
    args = parser.parse_args()
    orig_datapath = args.original_data_path
    gen_datapath = args.generated_data_path

    if not os.path.exists(orig_datapath):
        warnings.warn('Please install willow dataset')
        return False
    if not os.path.exists(gen_datapath):
        os.makedirs(gen_datapath, mode=0755)

    cls_view_path_mask = os.path.join(orig_datapath, 'object_{0:02d}', 'views')
    cls_out_mask = os.path.join(gen_datapath, 'object_{0:02d}')
    cloud_name = 'cloud_{0:0>8}.pcd'
    indices_name = 'object_indices_{0:0>8}.txt'
    rgb_name = 'rgb_{0:0>8}.png'
    depth_name = 'depth_{0:0>8}.png'
    mask_name = 'mask_{0:0>8}.png'
    pos_name = 'pos_{0:0>8}.npy'
    rot_name = 'rot_{0:0>8}.npy'

    n_class = len(glob.glob(os.path.join(orig_datapath, 'object_*')))
    for i in range(n_class):
        in_fpath = cls_view_path_mask.format(i + 1)
        out_fpath = cls_out_mask.format(i + 1)
        if not os.path.exists(out_fpath):
            os.makedirs(out_fpath, mode=0755)
        num_view = len(glob.glob(os.path.join(in_fpath, "cloud_*.pcd")))
        for j in range(num_view):
            print "genrating data class:{} view:{}".format(i + 1, j)
            # 1. generate RGB image
            cmd1 ='pcl_pcd2png {} {}'.format(os.path.join(in_fpath, cloud_name.format(j)),
                                             os.path.join(out_fpath, rgb_name.format(j)))
            os.system(cmd1)
            # 2. generate depth image
            cmd2 ='pcl_pcd2png --field z --scale no {} {}'.format(os.path.join(in_fpath, cloud_name.format(j)),
                                                                  os.path.join(out_fpath, depth_name.format(j)))
            os.system(cmd2)
            # 3 . generate mask image
            f = open(os.path.join(in_fpath, indices_name.format(j)), 'r')
            mask_arr = np.zeros(img_width * img_height)
            for line in f:
                mask_arr[int(line)] = 255
            mask = mask_arr.reshape(img_height, img_width)
            cv2.imwrite(os.path.join(out_fpath, mask_name.format(j)), mask)
            # 4. save pos and rot
            f = open(os.path.join(in_fpath, 'pose_' + v_idx + '.txt'), 'r')
            pose_arr = []
            for line in f:
                pose_arr = list(map(float,line.split(" ")))
            pose_mat = np.asarray(pose_arr).reshape(4,4)
            pos = np.array([pose_mat[0][3], pose_mat[1][3], pose_mat[2][3]])
            rot = pose_mat[0:3,0:3]
            np.save(os.path.join(out_fpath, pos_name.format(j)), pos)
            np.save(os.path.join(out_fpath, rot_name.format(j)), rot)

if __name__ == '__main__':
    main()
