#!/usr/bin/env python

import warnings
import pypcd
import os

import cv2
import numpy as np

num_class = 35
num_view = 37

img_width = 640
img_height = 480


orig_data_path = os.path.join(os.getcwd(), '../models/willow_models')
dataset_path = os.path.join(os.getcwd(), '../train_data/willow_models')

# 1 . generate RGB image
# 2 . generate depth image
# 3 . generate mask image
# 4 . generate cp-dist-map

def main():
    if not os.path.exists(orig_data_path):
        warnings.warn('Please install willow dataset')
        return false
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, mode=0755)

    for i in range(num_class):
        c_idx =  '{0:02d}'.format(i+1)
        c_name = 'object_' + c_idx
        orig_c_path = os.path.join(orig_data_path, c_name, 'views')
        c_path = os.path.join(dataset_path, c_name)
        if not os.path.exists(c_path):
            os.makedirs(c_path, mode=0755)
        for j in range(num_view):
            v_idx = '{0:08d}'.format(j)
            # 1. generate RGB image
            cmd1 ='pcl_pcd2png ' + os.path.join(orig_c_path, 'cloud_' + v_idx + '.pcd') + ' ' + os.path.join(c_path, 'rgb_' + v_idx + '.png')
            os.system(cmd1)
            # 2. generate RGB image
            cmd2 ='pcl_pcd2png --field z ' + os.path.join(orig_c_path, 'cloud_' + v_idx + '.pcd') + ' ' + os.path.join(c_path, 'depth_' + v_idx + '.png')
            os.system(cmd2)
            # 3 . generate mask image
            f = open(os.path.join(orig_c_path, 'object_indices_' + v_idx + '.txt'), 'r')
            mask_arr = np.zeros(img_width * img_height)
            for line in f:
                mask_arr[int(line)] = 255
            mask = mask_arr.reshape(img_height, img_width)
            cv2.imwrite(os.path.join(c_path, 'mask_' + v_idx + '.png'), mask)

            # generate center point (cp) distance map image
            # load point cloud
            pc = pypcd.PointCloud.from_path(os.path.join(orig_c_path,
                                                         'cloud_' + v_idx + '.pcd'))
            # load pose text
            f = open(os.path.join(orig_c_path, 'pose_' + v_idx + '.txt'), 'r')
            pose_arr = []
            for line in f:
                pose_arr = list(map(float,line.split(" ")))
            pose_mat = np.array(pose_arr).reshape(4,4)
            pos = np.array([pose_mat[0][3], pose_mat[1][3], pose_mat[2][3]])
            rot = pose_mat[0:3,0:3]
            dist_map = np.zeros(3 * img_height * img_width).reshape(img_height, img_width, 3)
            for i in range(img_height):
                for j in range(img_width):
                    for k in range(3):
                        dist_map[i][j][k] = pos[k] - pc.pc_data[i * img_width + j][k]
            # save dist map
            np.save(os.path.join(c_path, 'dist_' + v_idx + '.npy'), dist_map)
            # save center point position
            np.save(os.path.join(c_path, 'pos_' + v_idx + '.npy'), pos)
            # save rotation_matrix numpy
            np.save(os.path.join(c_path, 'rot_' + v_idx + '.npy'), rot)

            print pose_mat
            print pos
            print rot
            
if __name__ == '__main__':
    main()
