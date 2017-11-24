import numpy as np
import quaternion
import six
import os

import pypcd
import pyflann

from chainer import cuda
from chainer import function
import chainer.functions as F
from chainer.utils import type_check

import cv2

import pose_estimation


class SimplePoseEstimationInterface(object):
    ## Support multi-class, one instance per one class
    def __init__(self, eps=0.2,
                 distance_sanity=0.1, min_distance=0.005,
                 base_path = 'OcclusionChallengeICCV2015',
                 objs =['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher'],
                 model_partial= 1,
                 K =  np.array([[572.41140, 0, 325.26110],
                                [0, 573.57043, 242.04899],
                                [0, 0, 0]]),
                 im_size = (640, 480)):
        self.eps = eps
        self.distance_sanity = distance_sanity
        self.min_distance = min_distance
        self.im_size = im_size

    def _get_pointcloud(self, depth_im, K):
        xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
        ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T
        Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
        Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])
        xyz_im = np.dstack((Xs, Ys, depth_im))
        xyz_im[xyz_im == 0] = np.nan
        return xyz_im

    def pre_processing(self, y_cls, y_cp, y_ocp, depth, K):
        n_class, img_h, img_w = y_cls.shape
        ## softmax
        y_cls = y_cls - np.max(y_cls, axis=0, keepdims=True)
        y_cls = np.exp(y_cls) / np.sum(np.exp(y_cls), axis=0, keepdims=True)
        prob = np.max(y_cls, axis=0)
        pred = np.argmax(y_cls, axis=0)
        # probability threshold
        pred[prob < self.eps] = 0

        masks = np.zeros((n_class - 1, img_h, img_w))
        for i_c in six.moves.range(n_class - 1):
            masks[i_c] = (pred == i_c + 1)

        t_pc = self._get_pointcloud(depth, K).transpose(2, 0, 1)

        # with nonnan_mask
        pred_mask = np.invert(np.isnan(t_pc[0, :, :]))

        t_pc[t_pc != t_pc] = 0

        y_cp = masks[:, np.newaxis, :, :] * y_cp.reshape(n_class - 1, 3, img_h, img_w)
        y_ocp = masks[:, np.newaxis, :, :] * y_ocp.reshape(n_class - 1, 3, img_h, img_w)
        y_cp = np.sum(y_cp, axis=0)
        y_ocp = np.sum(y_ocp, axis=0)

        ## check dual cp distance sanity
        if self.distance_sanity:
            dist_cp = np.linalg.norm(y_cp, axis=0)
            dist_ocp = np.linalg.norm(y_ocp, axis=0)
            dist_mask = (np.abs(dist_cp - dist_ocp) < self.distance_sanity)
            pred_mask = pred_mask * dist_mask

        ## minimum distance threshold
        if self.min_distance:
            dist_cp = np.linalg.norm(y_cp, axis=0)
            dist_ocp = np.linalg.norm(y_ocp, axis=0)
            min_cp_mask = (dist_cp > self.min_distance)
            min_ocp_mask = (dist_ocp > self.min_distance)
            pred_mask = pred_mask * min_cp_mask * min_ocp_mask

        pred_mask = masks * pred_mask[np.newaxis, :, :]
        y_cp_reshape = (y_cp + t_pc).reshape(3, -1)
        t_pc_reshape = t_pc.reshape(3, -1)
        y_ocp_reshape = y_ocp.reshape(3, -1)
        return pred_mask, y_cp_reshape, t_pc_reshape, y_ocp_reshape

    def execute(self, y_cls, y_cp, y_ocp, depth, K):
        n_class = y_cls.shape[0]
        pred_mask, y_cp_reshape, t_pc_reshape, y_ocp_reshape = self.pre_processing(y_cls, y_cp, y_ocp, depth, K)

        estimated_ocp = np.ones((n_class - 1, 3)) * 10
        estimated_R = np.zeros((n_class - 1, 3, 3))

        for i_c in six.moves.range(n_class - 1):
            if np.sum(pred_mask[i_c]) < 10:
                continue
            pmask = pred_mask[i_c].ravel().astype(np.bool)
            y_cp_nonzero = y_cp_reshape[:, pmask]
            y_cp_mean = np.mean(y_cp_nonzero, axis=1, keepdims=True)
            ## Remove outlier direct Center Point
            cp_mask3d = (y_cp_nonzero - y_cp_mean < 0.1)
            cp_mask = (np.sum(cp_mask3d, axis=0) == 3)
            if np.sum(cp_mask) < 10:
                continue
            t_pc_nonzero = t_pc_reshape[:, pmask][:, cp_mask]
            y_ocp_nonzero = y_ocp_reshape[:, pmask][:, cp_mask]
            ret_ocp, ret_R = pose_estimation.simple_ransac_estimation_cpp(t_pc_nonzero, y_ocp_nonzero)

            estimated_ocp[i_c] =  ret_ocp
            estimated_R[i_c] = ret_R

        return estimated_ocp, estimated_R


class PoseEstimationInterface(object):
    ## Support multi-class, one instance per one class
    def __init__(self, eps=0.2,
                 distance_sanity=0.1, min_distance=0.005,
                 base_path = 'OcclusionChallengeICCV2015',
                 mseh_basepath = None,
                 objs =['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher'],
                 model_partial= 1,
                 K =  np.array([[572.41140, 0, 325.26110],
                               [0, 573.57043, 242.04899],
                               [0, 0, 0]]),
                 n_ransac=100,
                 im_size = (640, 480)):
        super.__init__(eps, distance_sanity, min_distance, base_path,
                 objs, model_partial, K, im_size)
        # ## for flann
        # self.flann_search_idx = []
        # self.models_pc = []
        # self.objs = objs
        # pyflann.set_distance_type('euclidean')
        # for obj_name in objs:
        #     pc = pypcd.PointCloud.from_path(
        #         os.path.join(base_path, 'models_pcd', obj_name + '.pcd'))
        #     pc = np.asarray(pc.pc_data.tolist())[:,:3] ## only use xyz

        #     search_idx = pyflann.FLANN()
        #     search_idx.build_index(pc[0::model_partial, :], algorithm='kmeans',
        #                            centers_init='kmeanspp', random_seed=1234)
        #     self.flann_search_idx.append(search_idx)
        #     self.models_pc.append(pc)

        # for Cypose_estimator
        self.mesh_pathes = []
        for obj_name in objs:
            if mesh_basepath is None:
                mesh_path = os.path.join(base_path, 'models_ply', obj_name + '.ply')
            else:
                mesh_path = os.path.join(mesh_basepath,  obj_name + '.ply')
            self.mesh_pathes.append(mesh_path)

        self.pose_estimator = pose_estimation.CyPoseEstimator(self.mesh_pathes,
                                                              im_size[1], im_size[0])
        self.pose_estimator.set_ransac_count(n_ransac)

    def execute(self, y_cls, y_cp, y_ocp, depth, K):
        n_class = y_cls.shape[0]
        pred_mask, y_cp_reshape, t_pc_reshape, y_ocp_reshape = self.pre_processing(y_cls, y_cp, y_ocp, depth, K)

        self.pose_estimator.set_depth(depth)
        self.pose_estimator.set_k(K)
        for i_c in six.moves.range(n_class - 1):
            if np.sum(pred_mask[i_c]) < 10:
                continue
            pmask = pred_mask[i_c].ravel().astype(np.bool)
            y_cp_nonzero = y_cp_reshape[:, pmask]
            y_cp_mean = np.mean(y_cp_nonzero, axis=1, keepdims=True)
            ## Remove outlier direct Center Point
            cp_mask3d = (y_cp_nonzero - y_cp_mean < 0.1)
            cp_mask = (np.sum(cp_mask3d, axis=0) == 3)
            ## flann refinement
            num_nn = 2
            flann_ret, flann_dist = self.flann_search_idx[i_c].nn_index(y_ocp_reshape[:, pmask].transpose(1, 0), num_nn)
            flann_ret_unique = np.unique(flann_ret.ravel())
            flann_mask = (flann_dist[:, 0] < 5e-3)
            refine_mask = cp_mask * flann_mask
            if np.sum(refine_mask) < 10:
                continue
            t_pc_nonzero = t_pc_reshape[:, pmask][:, refine_mask]
            y_ocp_nonzero = y_ocp_reshape[:, pmask][:, refine_mask]

            self.pose_estimator.set_mask(pred_mask[i_c])
            self.pose_estimator.set_object_id(i_c)
            ret_ocp, ret_R = self.pose_estimator.ransac_estimation(t_pc_nonzero, y_ocp_nonzero)

            estimated_ocp[i_c] =  ret_ocp
            estimated_R[i_c] = ret_R

        return estimated_ocp, estimated_R