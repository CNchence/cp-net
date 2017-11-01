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

# tmp
import time
import cv2

import pose_estimation

def icp(src, dst, dst_search_idx=None, n_iter=50, thre_precent=95):
    # input src shape = (3, num)
    # input dst shape = (3, num)
    if len(src) <= len(dst):
        a = src
        b = dst
    else:
        a = dst
        b = src

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
    old_a = a
    a_mean = np.mean(a, axis=1)
    a_demean = a - a_mean[:, np.newaxis]

    for i in six.moves.range(n_iter):
        indices, distances = search_idx.nn_index(old_a.transpose(1, 0),  1)
        # percentile outlier removal
        percentile_thre = np.percentile(distances, thre_precent)
        inlier_mask = (distances <= percentile_thre)

        b_mean = np.mean(b[:, indices[inlier_mask]], axis=1)
        b_demean = b[:, indices[inlier_mask]] - b_mean[:, np.newaxis]

        _R = calc_rot_by_svd(b_demean, a_demean[:, inlier_mask])
        _t = b_mean - np.dot(_R, a_mean)
        new_a = np.dot(_R, a) + _t[:, np.newaxis]

        if np.mean(np.abs(new_a - old_a)) < 1e-12:
            break
        old_a = new_a

    if len(src) > len(dst):
        _t = - _t
        _R = _R.T

    return _t, _R


def calc_rot_by_svd(Y, X):
    U, S, V = np.linalg.svd(np.dot(Y, X.T))
    VU_det = np.linalg.det(np.dot(V, U))
    H = np.diag(np.array([1, 1, VU_det], dtype=np.float64))
    R = np.dot(np.dot(U, H), V)
    return R



def calc_accuracy_impl(estimated_cp, estimated_ocp, estimated_R,
                       t_cp, t_rot, batch_size, n_class,
                       debug=False):
    match_cnt = 0.0
    cp_acc = 0.0
    ocp_acc = 0.0
    rot_acc = 0.0
    penalty = np.array([10, 10, 10])
    eval_rate = 0.0
    non_obj_cnt = 0.0
    for i_b in six.moves.range(batch_size):
        if debug:
            print "---"
        for i_c in six.moves.range(n_class - 1):
            if np.linalg.norm(estimated_ocp[i_b, i_c] - penalty) > 0.001 \
               and np.linalg.norm(t_cp[i_b, i_c]) != 0 \
               and np.sum(estimated_R[i_b, i_c]) != 0:
                match_cnt += 1.0
                cp_acc += np.linalg.norm(estimated_cp[i_b, i_c] - t_cp[i_b, i_c])
                diff_ocp = np.linalg.norm(estimated_ocp[i_b, i_c] - t_cp[i_b, i_c])
                ocp_acc += diff_ocp
                quat = quaternion.from_rotation_matrix(
                    np.dot(estimated_R[i_b, i_c].T, t_rot[i_b, i_c]))
                quat_w = min(1, abs(quat.w))
                diff_angle = np.rad2deg(np.arccos(quat_w)) * 2
                rot_acc += diff_angle
                ## for debug
                if debug:
                    # print estimated_R[i_b, i_c]
                    # print t_rot[i_b, i_c]
                    # print np.dot(estimated_R[i_b, i_c], np.linalg.inv(t_rot[i_b, i_c]))
                    # print np.dot(estimated_R[i_b, i_c].T, t_rot[i_b, i_c])
                    # print quaternion.from_rotation_matrix(estimated_R[i_b, i_c])
                    # print quaternion.from_rotation_matrix(t_rot[i_b, i_c])
                    # print quat
                    print "diff angle : " + str(diff_angle) + \
                        ",  diff dist : " + str(np.linalg.norm(estimated_ocp[i_b, i_c] - t_cp[i_b, i_c]))

                if diff_angle < 5 and diff_ocp < 0.05:
                    eval_rate += 1.0
            elif np.linalg.norm(t_cp[i_b, i_c]) == 0:
                non_obj_cnt += 1.0

    if match_cnt > 0:
        cp_acc /= match_cnt
        ocp_acc /= match_cnt
        rot_acc /= match_cnt
        eval_rate /= (batch_size * n_class - non_obj_cnt)
    else:
        cp_acc = np.linalg.norm(penalty)
        ocp_acc = np.linalg.norm(penalty)
        rot_acc = 180

    return cp_acc, ocp_acc, rot_acc, eval_rate

class ModelBaseConsensusAccuracy(function.Function):

    ## Support multi-class, one instance per one class

    def __init__(self, eps=0.2,
                 distance_sanity=0.1, min_distance=0.005,
                 base_path = 'OcclusionChallengeICCV2015',
                 objs =['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher'],
                 model_partial= 1,
                 K = None,
                 im_size = (640, 480),
                 method="SVD", ver2=False):
        self.eps = eps
        self.distance_sanity = distance_sanity
        self.min_distance = min_distance
        self.method = method
        self.ver2 = ver2
        self.im_size = im_size
        ## camera parm
        if K is not None:
            self.K = K
        else:
            self.K = np.array([[572.41140, 0, 325.26110],
                               [0, 573.57043, 242.04899],
                               [0, 0, 0]])

        ## for flann
        self.flann_search_idx = []
        self.models_pc = []
        self.mesh_pathes = []
        self.objs = objs
        pyflann.set_distance_type('euclidean')
        for obj_name in objs:
            pc = pypcd.PointCloud.from_path(
                os.path.join(base_path, 'models_pcd', obj_name + '.pcd'))
            pc = np.asarray(pc.pc_data.tolist())[:,:3] ## only use xyz
            # pc = pc * np.array([1, -1, -1])[np.newaxis, :] ## negative z
            # trans = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            # pc = np.dot(trans, pc.transpose(1, 0)).transpose(1, 0)

            search_idx = pyflann.FLANN()
            search_idx.build_index(pc[0::model_partial, :], algorithm='kmeans',
                                   centers_init='kmeanspp', random_seed=1234)
            self.flann_search_idx.append(search_idx)
            self.models_pc.append(pc)

            # for Cypose_estimator
            mesh_path = os.path.join(base_path, 'models_ply', obj_name + '.ply')
            self.mesh_pathes.append(mesh_path)

        self.pose_estimator = pose_estimation.CyPoseEstimator(self.mesh_pathes,
                                                              im_size[1], im_size[0])
        self.pose_estimator.set_ransac_count(100)

    def forward(self, inputs):
        y_cls, y_cp, y_ocp, t_ocp, t_cp, t_rot, t_pc, depth, K, rgb  = inputs
        ## gpu to cpu
        if cuda.get_array_module(*inputs) != np:
            y_cls = cuda.to_cpu(y_cls)
            y_cp = cuda.to_cpu(y_cp)
            y_ocp = cuda.to_cpu(y_ocp)
            t_ocp = cuda.to_cpu(t_ocp)
            t_cp = cuda.to_cpu(t_cp)
            t_rot = cuda.to_cpu(t_rot)
            t_pc = cuda.to_cpu(t_pc)
            depth = cuda.to_cpu(depth)
            K = cuda.to_cpu(K)
            rgb = cuda.to_cpu(rgb)

        batch_size, n_class, img_h, img_w = y_cls.shape

        ## softmax
        y_cls = y_cls - np.max(y_cls, axis=1, keepdims=True)
        y_cls = np.exp(y_cls) / np.sum(np.exp(y_cls), axis=1, keepdims=True)

        prob = np.max(y_cls, axis=1)
        pred = np.argmax(y_cls, axis=1)

        # probability threshold
        pred[prob < self.eps] = 0

        masks = np.zeros((batch_size, n_class - 1, img_h, img_w))
        for i_b in six.moves.range(batch_size):
            for i_c in six.moves.range(n_class - 1):
                masks[i_b, i_c] = (pred[i_b] == i_c + 1)

        # with nonnan_mask
        pred_mask = np.invert(np.isnan(t_pc[:,0,:]))

        t_pc[t_pc != t_pc] = 0
        estimated_cp = np.ones((batch_size, n_class - 1, 3)) * 10
        estimated_ocp = np.ones((batch_size, n_class - 1, 3)) * 10
        estimated_R = np.zeros((batch_size, n_class - 1, 3, 3))
        penalty = np.array([10, 10, 10])

        y_cp = masks[:, :, np.newaxis, :, :] * y_cp.reshape(batch_size, n_class - 1, 3, img_h, img_w)
        y_ocp = masks[:, :, np.newaxis, :, :] * y_ocp.reshape(batch_size, n_class - 1, 3, img_h, img_w)
        y_cp = np.sum(y_cp, axis=1)
        y_ocp = np.sum(y_ocp, axis=1)
        t_ocp = masks[:, :, np.newaxis, :, :] * t_ocp.reshape(batch_size, n_class - 1, 3, img_h, img_w)
        t_ocp = np.sum(t_ocp, axis=1)

        ## check dual cp distance sanity
        if self.distance_sanity:
            dist_cp = np.linalg.norm(y_cp, axis=1)
            dist_ocp = np.linalg.norm(y_ocp, axis=1)
            dist_mask = (np.abs(dist_cp - dist_ocp) < self.distance_sanity)
            pred_mask = pred_mask * dist_mask

        ## minimum distance threshold
        if self.min_distance:
            dist_cp = np.linalg.norm(y_cp, axis=1)
            dist_ocp = np.linalg.norm(y_ocp, axis=1)
            min_cp_mask = (dist_cp > self.min_distance)
            min_ocp_mask = (dist_ocp > self.min_distance)
            pred_mask = pred_mask * min_cp_mask * min_ocp_mask

        pred_mask = masks * pred_mask[:, np.newaxis, :, :]

        ## First of All, Calculate Center point directly
        prob_weight = (pred_mask * y_cls[:, 1:])
        prob_weight = prob_weight / (1e-15 + np.sum(
            prob_weight.reshape(batch_size, n_class - 1, -1), axis=2)[:, :, np.newaxis, np.newaxis])
        estimated_cp = np.sum(
            (prob_weight[:, :, np.newaxis, :, :] * \
             (y_cp + t_pc)[:, np.newaxis, :, :, :]).reshape(batch_size, n_class - 1, 3, -1), axis=3)

        y_cp_reshape = (y_cp + t_pc).reshape(batch_size, 3, -1)
        t_pc_reshape = t_pc.reshape(batch_size, 3, -1)
        y_ocp_reshape = y_ocp.reshape(batch_size, 3, -1)
        t_ocp_reshape = t_ocp.reshape(batch_size, 3, -1)

        for i_b in six.moves.range(batch_size):
            self.pose_estimator.set_depth(depth[i_b])
            self.pose_estimator.set_k(K[i_b])
            for i_c in six.moves.range(n_class - 1):
                if np.sum(pred_mask[i_b, i_c]) < 10:
                    continue
                pmask = pred_mask[i_b, i_c].ravel().astype(np.bool)
                y_cp_nonzero = y_cp_reshape[i_b][:, pmask]
                y_cp_mean = np.mean(y_cp_nonzero, axis=1, keepdims=True)
                ## Remove outlier direct Center Point
                cp_mask3d = (y_cp_nonzero - y_cp_mean < 0.1)
                cp_mask = (np.sum(cp_mask3d, axis=0) == 3)

                ## flann refinement
                num_nn = 2
                # pyflann.set_distance_type('manhattan')
                pyflann.set_distance_type('euclidean')
                flann_ret, flann_dist = self.flann_search_idx[i_c].nn_index(y_ocp_reshape[i_b][:, pmask].transpose(1, 0), num_nn)
                flann_ret_unique = np.unique(flann_ret.ravel())
                flann_mask = (flann_dist[:, 0] < 5e-3)  # threshold is tmp
                refine_mask = cp_mask * flann_mask
                if np.sum(refine_mask) < 10:
                    continue

                t_pc_nonzero = t_pc_reshape[i_b][:, pmask][:, refine_mask]
                y_ocp_nonzero = y_ocp_reshape[i_b][:, pmask][:, refine_mask]
                self.pose_estimator.set_mask(pred_mask[i_b, i_c])
                self.pose_estimator.set_object_id(i_c)
                ret_ocp, ret_R = self.pose_estimator.ransac_estimation(t_pc_nonzero, y_ocp_nonzero)
                # ret_ocp, ret_R = pose_estimation.model_base_ransac_estimation_cy(t_pc_nonzero, y_ocp_nonzero, self.models_pc[i_c],
                #                                                                  depth[i_b], K[i_b], pred_mask[i_b, i_c],
                #                                                                  self.im_size)             
                # print "---"
                # print ret_ocp
                # print ret_R
                ## icp refinement
                icp_ocp, icp_R = icp(np.dot(ret_R.T, t_pc_nonzero - ret_ocp[:, np.newaxis]),
                                     self.models_pc[i_c].transpose(1,0),
                                     dst_search_idx=self.flann_search_idx[i_c])
                ret_R = np.dot(ret_R, icp_R.T)
                ret_ocp -= np.dot(ret_R, icp_ocp)

                if False:
                    imagenet_mean = np.array(
                        [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
                    rgb_out = rgb[i_b].transpose(1,2,0) * 255.0 + imagenet_mean
                    rgb_out = cv2.resize(rgb_out, (depth[i_b].shape[::-1]))
                    cv2.imwrite("depth_rgb.jpg", rgb_out)
                # print "--"
                # print self.objs[i_c]
                # pc_trans = np.dot(estimated_R[i_b, i_c].T, (t_pc_nonzero - estimated_ocp[i_b, i_c][:, np.newaxis]))
                # _t, _R = icp_by_search_idx(pc_trans, self.models_pc[i_c].transpose(1, 0), self.flann_search_idx[i_c])
                # estimated_R[i_b, i_c] = np.dot(estimated_R[i_b, i_c], _R.T)
                # estimated_ocp[i_b, i_c] = estimated_ocp[i_b, i_c] - _t

                # print np.mean(t_ocp_nonzero - y_ocp_nonzero, axis=1)
                # np.save("test.npy", y_ocp_nonzero)
                # np.save("test_t.npy", t_ocp_nonzero)
                # np.save("test_model.npy", self.models_pc[i_c])
                # np.save("test_rot.npy", np.dot(t_rot[i_b,i_c], self.models_pc[i_c].transpose(1,0)) + (t_cp[i_b, i_c] - estimated_cp[i_b, i_c])[:, np.newaxis])
                # np.save("test_t_rot.npy", t_rot[i_b,i_c])
                # np.save("test_pc_all.npy", (t_pc_reshape[i_b][:, pmask] - estimated_cp[i_b, i_c][:, np.newaxis]))
                # np.save("test_pc.npy", t_pc_demean)

                # time.sleep(1)
                estimated_ocp[i_b, i_c] =  ret_ocp
                estimated_R[i_b, i_c] = ret_R


        ret_cp, ret_ocp, ret_rot, ret_rate = calc_accuracy_impl(estimated_cp, estimated_ocp,
                                                                estimated_R, t_cp, t_rot, batch_size,
                                                                n_class, debug=True)

        return np.asarray(ret_cp, dtype=y_cp.dtype), np.asarray(ret_ocp, dtype=y_ocp.dtype), np.asarray(ret_rot, dtype=y_ocp.dtype), np.asarray(ret_rate, dtype=y_ocp.dtype)


def model_base_consensus_accuracy(y_cls, y_cp, y_ocp, t_ocp, t_cp, t_rot, t_pc, depth, K,
                                  eps=0.2, distance_sanity=0.1, method="SVD", ver2=False):
    return ModelBaseConsensusAccuracy(eps=eps,
                                      distance_sanity=distance_sanity,
                                      method=method, ver2=ver2)(y_cls, y_cp, y_ocp, t_ocp, t_cp, t_rot, t_pc, depth, K)
