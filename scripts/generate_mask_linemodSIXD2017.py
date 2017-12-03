import glob
import os
import sys
import argparse
import six

import cv2
import numpy as np

import yaml

from cp_net.utils import renderer
from cp_net.utils import inout
from cp_net.utils import preprocess_utils

def depth_im_to_dist_im(depth_im, K):
    xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
    ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T

    Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
    Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])

    dist_im = np.linalg.norm(np.dstack((Xs, Ys, depth_im)), axis=2)
    return dist_im

def estimate_visib_mask(d_test, d_model, delta):
    assert(d_test.shape == d_model.shape)
    mask_valid = np.logical_and(d_test > 0, d_model > 0)

    d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
    visib_mask = np.logical_and(d_diff <= delta, mask_valid)

    return visib_mask

def main():
    parser = argparse.ArgumentParser(description='generate mask image of LinemodSIXD dataset')
    parser.add_argument('--mode',  default='train',
                        help='select train or test')
    args = parser.parse_args()
    mode = args.mode
    if not mode in ['train', 'test']:
        sys.stderr.write("Error: mode should be 'train' or 'test', but mode is {}\n".format(mode))
        sys.exit()

    objs = np.arange(15) + 1
    im_size = (640, 480)
    data_basepath = '../train_data/linemodSIXD2017'

    model_fpath_mask = os.path.join(data_basepath, 'models', 'obj_{0:0>2}.ply')
    rgb_fpath_mask = os.path.join(data_basepath, mode, '{0:0>2}', 'rgb', '{1:0>4}.png')
    depth_fpath_mask = os.path.join(data_basepath, mode, '{0:0>2}', 'depth', '{1:0>4}.png')
    gt_poses_mask = os.path.join(data_basepath, mode, '{0:0>2}','gt.yml')
    info_mask = os.path.join(data_basepath, mode, '{0:0>2}','info.yml')
    gt_mask_dir = os.path.join(data_basepath, mode, '{0:0>2}', 'mask')
    mask_fpath_mask = os.path.join(data_basepath, mode, '{0:0>2}', 'mask', '{1:0>4}_{2:0>2}.png')

    # load object models
    models = []
    if mode == 'test':
        for obj in objs:
            print 'Loading data: {0:0>2}'.format(obj)
            model_fpath = model_fpath_mask.format(obj)
            models.append(inout.load_ply(model_fpath))

    ## delta for visibility correspondance
    delta = 15 # [mm]

    for scene_id in six.moves.range(len(objs)):
        gt_poses = yaml.load(open(gt_poses_mask.format(objs[scene_id])))
        info = yaml.load(open(info_mask.format(objs[scene_id])))
        if not os.path.exists(gt_mask_dir.format(objs[scene_id])):
            os.makedirs(gt_mask_dir.format(objs[scene_id]))
        n_imgs = len(glob.glob(rgb_fpath_mask.format(objs[scene_id], '****')))
        for im_id in six.moves.range(n_imgs):
            print "scene : obj_{0:0>2}, image id : {1}".format(objs[scene_id], im_id)
            if mode == 'test':
                depth_fpath = depth_fpath_mask.format(objs[scene_id], im_id)
                depth = cv2.imread(depth_fpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                depth = preprocess_utils.depth_inpainting(depth)
                K = np.asarray(info[im_id]['cam_K']).reshape(3,3)
                # Convert the input depth image to a distance image
                dist = depth_im_to_dist_im(depth, K)
                poses = gt_poses[im_id]
                for pose in poses:
                    obj_id = pose['obj_id']
                    rot = np.asarray(pose['cam_R_m2c']).reshape(3, 3)
                    trans = np.asarray(pose['cam_t_m2c']).T
                    # Render the object model
                    depth_ren_gt = renderer.render(
                        models[obj_id - 1], im_size, K, rot, trans, 100, 2500, mode='depth')
                    depth_ren_gt = depth_ren_gt
                    dist_ren_gt = depth_im_to_dist_im(depth_ren_gt, K)
                    visib_mask = estimate_visib_mask(dist, dist_ren_gt, delta)

                    cv2.imwrite(mask_fpath_mask.format(objs[scene_id], im_id, obj_id), visib_mask * 255)

            elif mode == 'train':
                depth_fpath = depth_fpath_mask.format(objs[scene_id], im_id)
                depth = cv2.imread(depth_fpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                mask = (depth > 0) * 255
                cv2.imwrite(mask_fpath_mask.format(objs[scene_id], im_id, objs[scene_id]), mask)

if __name__ == '__main__':
    main()
