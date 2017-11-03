#!/usr/bin/env python

import warnings
import os
import glob
import argparse
import cv2
import numpy as np

img_width = 640
img_height = 480
n_class = 35

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_data_path')
    parser.add_argument('annotation_data_path')
    parser.add_argument('generated_data_path')
    args = parser.parse_args()
    orig_datapath = args.original_data_path
    anno_datapath = args.annotation_data_path
    gen_datapath = args.generated_data_path

    if not os.path.exists(orig_datapath):
        warnings.warn('Please install willow dataset')
        return False
    if not os.path.exists(gen_datapath):
        os.makedirs(gen_datapath, mode=0755)


    scene_pathes = sorted(glob.glob(os.path.join(orig_datapath, 'T_*_willow_dataset')))
    out_path_mask = os.path.join(gen_datapath, '{0}')
    anno_fpath_mask = os.path.join(anno_datapath, 'willow', '{0}')
    cloud_name = 'cloud_{0:0>10}.pcd'
    anno_name = 'cloud_{0:0>10}.anno'
    rgb_name = 'rgb_{0:0>10}.png'
    depth_name = 'depth_{0:0>10}.png'
    pos_name = 'pos_{0}_{1:0>2}_{2:0>10}.npy'
    rot_name = 'rot_{0}_{1:0>2}_{2:0>10}.npy'

    n_class = len(glob.glob(os.path.join(orig_datapath, 'object_*')))

    for i, in_path in enumerate(scene_pathes):
        in_fpath = scene_pathes[i]
        out_fpath = out_path_mask.format(os.path.basename(in_fpath))
        anno_fpath = anno_fpath_mask.format(os.path.basename(in_fpath))
        if not os.path.exists(out_fpath):
            os.makedirs(out_fpath, mode=0755)
        n_view = len(glob.glob(os.path.join(in_fpath, 'cloud_*.pcd')))
        for j in range(n_view):
            print "genrating data scene:{} view:{}".format(i + 1, j)
            # generate RGB image
            cmd1 ='pcl_pcd2png {} {}'.format(os.path.join(in_fpath, cloud_name.format(j)),
                                             os.path.join(out_fpath, rgb_name.format(j)))
            os.system(cmd1)
            # generate depth image
            cmd2 ='pcl_pcd2png --field z --scale no {} {}'.format(os.path.join(in_fpath, cloud_name.format(j)),
                                                                  os.path.join(out_fpath, depth_name.format(j)))
            os.system(cmd2)
            # save pos and rot
            anno_path = os.path.join(anno_fpath, anno_name.format(j))
            if not os.path.exists(anno_path):
                continue
            f = open(anno_path, 'r')
            text_list = []
            for line in f:
                pose_txt = list(line.split(" "))
                text_list.append(pose_txt)
            for i in range(len(text_list)):
                obj_name = text_list[i][0]
                pose_mat = np.asarray(list(map(float, text_list[i][2:2+16])), dtype=np.float64).reshape(4,4)
                pos = np.array([pose_mat[0][3], pose_mat[1][3], pose_mat[2][3]])
                rot = pose_mat[0:3,0:3]
                np.save(os.path.join(out_fpath, pos_name.format(i, obj_name, j)), pos)
                np.save(os.path.join(out_fpath, rot_name.format(i, obj_name, j)), rot)

if __name__ == '__main__':
    main()