#!/usr/bin/env python

# this script is inspirated by sample code in https://github.com/thodan/obj_pose_eval.git
#
# A script for generation of mask image and point cloud like array for training
# from the ICCV2015 Occluded Object Challenge dataset[1,2]
# Besides the dataset [2], please download also the object
# models in PLY format from [3] and put them into subfolder "models_ply" of
# the main dataset folder "OcclusionChallengeICCV2015". You will also need to
# set data_basepath below.
#
# [1] http://cvlab-dresden.de/iccv2015-occlusion-challenge
# [2] https://cloudstore.zih.tu-dresden.de/public.php?service=files&t=a65ec05fedd4890ae8ced82dfcf92ad8
# [3] http://cmp.felk.cvut.cz/~hodanto2/store/OcclusionChallengeICCV2015_models_ply.zip


import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from obj_pose_eval import inout, misc, renderer, visibility


# Full Path to the OcclusionChallengeICCV2015 folder:
#-------------------------------------------------------------------------------
# data_basepath = 'path/to/OcclusionChallengeICCV2015'
data_basepath = '/Users/yusuke/Desktop/OcclusionChallengeICCV2015'
#-------------------------------------------------------------------------------

def depth_im_to_xyz(depth_im, K):
    xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
    ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T

    Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
    Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])
    xyz_im = np.dstack((Xs, Ys, depth_im))
    return xyz_im

def main():
    objs = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']

    calib_fpath = os.path.join(data_basepath, 'calib.yml')
    model_fpath_mask = os.path.join(data_basepath, 'models_ply', '{0}.ply')
    rgb_fpath_mask = os.path.join(data_basepath, 'RGB-D', 'rgb_noseg', 'color_{0}.png')
    depth_fpath_mask = os.path.join(data_basepath, 'RGB-D', 'depth_noseg', 'depth_{0}.png')
    gt_poses_mask = os.path.join(data_basepath, 'poses', '{0}', '*.txt')

    ## mask image save dir
    mask_fpath = os.path.join(data_basepath, 'RGB-D', 'mask')
    if not os.path.exists(mask_fpath):
        os.makedirs(mask_fpath, mode=0755)

    ## point cloud npy save dir
    xyz_fpath = os.path.join(data_basepath, 'RGB-D', 'xyz')
    if not os.path.exists(xyz_fpath):
        os.makedirs(xyz_fpath, mode=0755)
    mask_fpath_mask = os.path.join(data_basepath, 'RGB-D', 'mask', 'mask_{0:0>5}_{1}.png')
    xyz_fpath_mask = os.path.join(data_basepath, 'RGB-D', 'xyz', 'xyz_{0:0>5}.npy')

    # Camera parameters
    im_size = (640, 480)
    K = np.array([[572.41140, 0, 325.26110],
                  [0, 573.57043, 242.04899],
                  [0, 0, 0]])


    ## delta for visibility correspondance
    delta = 15 # [mm]

    # Load models and ground truth poses
    models = []
    gt_poses = []

    for obj in objs:
        print 'Loading data:', obj
        model_fpath = model_fpath_mask.format(obj)
        models.append(inout.load_ply(model_fpath))

        gt_fpaths = sorted(glob.glob(gt_poses_mask.format(obj)))
        gt_poses_obj = []
        for gt_fpath in gt_fpaths:
            gt_poses_obj.append(
                inout.load_gt_pose_dresden(gt_fpath))
        gt_poses.append(gt_poses_obj)

    # Loop over images
    # im_ids = range(len(gt_poses[0])) #range(100)
    im_ids = range(10)
    for im_id in im_ids:
        im_id_str = str(im_id).zfill(5)
        print 'Processing image:', im_id

        # Load the RGB and the depth image
        rgb_fpath = rgb_fpath_mask.format(im_id_str)
        rgb = cv2.imread(rgb_fpath, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth_fpath = depth_fpath_mask.format(im_id_str)
        depth = cv2.imread(depth_fpath, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Convert the input depth image to a distance image
        dist = misc.depth_im_to_dist_im(depth, K)

        # Convert the input depth image to a xyz array
        xyz = depth_im_to_xyz(depth, K)
        np.save(xyz_fpath_mask.format(im_id), xyz)

        for obj_id, obj_name in enumerate(objs):
            pose = gt_poses[obj_id][int(im_id)]
            if pose['R'].size != 0 and pose['t'].size != 0:

                # Render the object model
                depth_ren_gt = renderer.render(
                    models[obj_id], im_size, K, pose['R'], pose['t'], 0.1, 2.0,
                    surf_color=(0.0, 1.0, 0.0), mode='depth')
                depth_ren_gt *= 1000 # Convert the rendered depth map to [mm]

                # Convert the input depth image to a distance image
                dist_ren_gt = misc.depth_im_to_dist_im(depth_ren_gt, K)

                # Estimate the visibility mask
                visib_mask = visibility.estimate_visib_mask(dist, dist_ren_gt, delta)
                cv2.imwrite(mask_fpath_mask.format(im_id, obj_name), visib_mask * 255)

                # Difference between the test and the rendered distance image
                dist_diff = dist_ren_gt.astype(np.float32) - dist.astype(np.float32)
                dist_diff *= np.logical_and(dist > 0, dist_ren_gt > 0).astype(np.float32)

if __name__ == '__main__':
    main()

