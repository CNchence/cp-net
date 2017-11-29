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


def calc_rot_by_svd(Y, X):
    U, S, V = np.linalg.svd(np.dot(Y, X.T))
    VU_det = np.linalg.det(np.dot(V, U))
    H = np.diag(np.array([1, 1, VU_det], dtype=np.float64))
    R = np.dot(np.dot(U, H), V)
    return R


def icp(a, b, dst_search_idx=None, n_iter=100, thre_precent=95):
    # input src shape = (3, num)
    # input dst shape = (3, num)

    ## pyflann search index build
    pyflann.set_distance_type('euclidean')
    if dst_search_idx is None:
        search_idx = pyflann.FLANN()
        search_idx.build_index(b.transpose(1, 0), algorithm='kmeans',
                               centers_init='kmeanspp', random_seed=1234)
    else:
        search_idx = dst_search_idx

    ## intialize for iteration
    _t = np.zeros(3)
    _R = np.diag((1, 1, 1))
    old_a = a.copy()

    for i in six.moves.range(n_iter):
        indices, distances = search_idx.nn_index(old_a.transpose(1, 0),  1)
        # percentile outlier removal
        percentile_thre = np.percentile(distances, thre_precent)
        inlier_mask = (distances <= percentile_thre)

        b_mean = np.mean(b[:, indices[inlier_mask]], axis=1)
        b_demean = b[:, indices[inlier_mask]] - b_mean[:, np.newaxis]

        a_mean = np.mean(a[:, inlier_mask], axis=1)
        a_demean = a[:, inlier_mask] - a_mean[:, np.newaxis]

        _R = calc_rot_by_svd(b_demean, a_demean)
        _t = b_mean - np.dot(_R, a_mean)
        new_a = np.dot(_R, a) + _t[:, np.newaxis]

        if np.mean(np.abs(new_a - old_a)) < 1e-12:
            break
        old_a = new_a
    return _t, _R


class SimplePoseEstimationInterface(object):
    ## Support multi-class, one instance per one class
    def __init__(self, eps=0.2,
                 distance_sanity=0.1, min_distance=0.005,
                 base_path = 'OcclusionChallengeICCV2015',
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


class PoseEstimationInterface(SimplePoseEstimationInterface):
    ## Support multi-class, one instance per one class
    def __init__(self, eps=0.2,
                 distance_sanity=0.1, min_distance=0.005,
                 base_path = 'OcclusionChallengeICCV2015',
                 mseh_basepath = None,
                 objs =['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher'],
                 model_partial= 1,
                 model_scale= 1.0,
                 n_ransac=100,
                 im_size = (640, 480)):
        super(PoseEstimationInterface, self).__init__(eps, distance_sanity, min_distance, base_path, im_size)
        ## for flann
        self.flann_search_idx = []
        self.models_pc = []
        self.objs = objs
        pyflann.set_distance_type('euclidean')
        for obj_name in objs:
            pc = pypcd.PointCloud.from_path(
                os.path.join(base_path, 'models_pcd', '{}.pcd'.format(obj_name)))
            pc = np.asarray(pc.pc_data.tolist())[:, [0, 1, 2]] / model_scale ## only use xyz

            search_idx = pyflann.FLANN()
            search_idx.build_index(pc[0::model_partial, :], algorithm='kmeans',
                                   centers_init='kmeanspp', random_seed=1234)
            self.flann_search_idx.append(search_idx)
            self.models_pc.append(pc)

        # for Cypose_estimator
        # self.mesh_pathes = []
        # for obj_name in objs:
        #     if mesh_basepath is None:
        #         mesh_path = os.path.join(base_path, 'models_ply', obj_name + '.ply')
        #     else:
        #         mesh_path = os.path.join(mesh_basepath,  obj_name + '.ply')
        #     self.mesh_pathes.append(mesh_path)

        # self.pose_estimator = pose_estimation.CyPoseEstimator(self.mesh_pathes,
        #                                                       im_size[1], im_size[0])
        # self.pose_estimator.set_ransac_count(n_ransac)

    def execute(self, y_cls, y_cp, y_ocp, depth, K):
        n_class = y_cls.shape[0]
        pred_mask, y_cp_reshape, t_pc_reshape, y_ocp_reshape = self.pre_processing(y_cls, y_cp, y_ocp, depth, K)

        estimated_ocp = np.ones((n_class - 1, 3)) * 10
        estimated_R = np.zeros((n_class - 1, 3, 3))

        # self.pose_estimator.set_depth(depth)
        # self.pose_estimator.set_k(K)
        for i_c in six.moves.range(n_class - 1):
            if np.sum(pred_mask[i_c]) < 10:
                continue
            pmask = pred_mask[i_c].ravel().astype(np.bool)
            y_cp_nonzero = y_cp_reshape[:, pmask]
            y_cp_mean = np.mean(y_cp_nonzero, axis=1, keepdims=True)
            ## Remove outlier direct Center Point
            cp_mask3d = (y_cp_nonzero - y_cp_mean < 0.05)
            cp_mask = (np.sum(cp_mask3d, axis=0) == 3)
            if np.sum(cp_mask) < 10:
                continue
            t_pc_nonzero = t_pc_reshape[:, pmask][:, cp_mask]
            y_ocp_nonzero = y_ocp_reshape[:, pmask][:, cp_mask]

            # self.pose_estimator.set_mask(pred_mask[i_c])
            # self.pose_estimator.set_object_id(i_c)
            # ret_ocp, ret_R = self.pose_estimator.ransac_estimation(t_pc_nonzero, y_ocp_nonzero)

            ret_ocp, ret_R = pose_estimation.simple_ransac_estimation_cpp(t_pc_nonzero, y_ocp_nonzero)
            icp_ocp, icp_R = icp(np.dot(ret_R.T, t_pc_nonzero - ret_ocp[:, np.newaxis]),
                                 self.models_pc[i_c].transpose(1,0),
                                 dst_search_idx=self.flann_search_idx[i_c])
            icp_R = np.dot(ret_R, icp_R.T)
            icp_ocp = ret_ocp - np.dot(ret_R, icp_ocp)
            # quat = quaternion.from_rotation_matrix(icp_R)
            # quat_w = min(1, abs(quat.w))
            # diff_angle = np.rad2deg(np.arccos(quat_w)) * 2
            # print "obj_{0} icp refinement : {1}".format(i_c + 1, diff_angle)
            # print icp_ocp, icp_R
            # ret_R = np.dot(ret_R, icp_R.T)
            # ret_ocp -= np.dot(ret_R, icp_ocp)
            # if i_c == 6:
            #     np.save("t_pc.npy", np.dot(ret_R.T, t_pc_nonzero - ret_ocp[:, np.newaxis]))
            #     np.save("t_icp.npy", np.dot(icp_R.T, t_pc_nonzero - icp_ocp[:, np.newaxis]))
            #     np.save("model.npy", self.models_pc[i_c].transpose(1,0))
            # estimated_ocp[i_c] =  ret_ocp
            # estimated_R[i_c] = ret_R
            estimated_ocp[i_c] =  icp_ocp
            estimated_R[i_c] = icp_R

        return estimated_ocp, estimated_R