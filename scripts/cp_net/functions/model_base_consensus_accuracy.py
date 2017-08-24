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

def icp(src, dst, n_iter=100):
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
    search_idx = pyflann.FLANN()
    search_idx.build_index(b.transpose(1, 0), algorithm='kmeans',
                           centers_init='kmeanspp', random_seed=1234)

    ## intialize for iteration
    _t = np.zeros(3)
    _R = np.diag((1, 1, 1))
    old_a = a
    a_mean = np.mean(a, axis=1)
    a_demean = a - a_mean[:, np.newaxis]

    for i in six.moves.range(n_iter):
        indices, distances = search_idx.nn_index(old_a.transpose(1, 0),  1)

        b_mean = np.mean(b[:, indices], axis=1)
        b_demean = b[:, indices] - b_mean[:, np.newaxis]

        _R = calc_rot_by_svd(b_demean, a_demean)
        _t = b_mean - np.dot(_R, a_mean)
        new_a = np.dot(_R, a) + _t[:, np.newaxis]

        # print "-"
        # print indices - np.arange(len(indices))
        # print np.mean(distances)
        # print np.mean(np.abs(new_a - b[:, indices]))
        # print np.mean(np.abs(new_a - b[:, indices]))

        if np.mean(np.abs(new_a - old_a)) < 1e-12:
            break
        old_a = new_a

    if len(src) > len(dst):
        _t = - t
        _R = _R.T

    return _t, _R


def icp_rotation(src, dst, n_iter=50):
    # input src shape = (3, num)
    # input dst shape = (3, num)
    # src = np.array(src, copy=True).astype(np.float32)
    # dst = np.array(dst, copy=True).astype(np.float32)

    if len(src) < len(dst):
        a = src
        b = dst
    else:
        a = dst
        b = src

    pyflann.set_distance_type('euclidean')
    search_idx = pyflann.FLANN()
    search_idx.build_index(b.transpose(1, 0).astype(np.float64), algorithm='kmeans',
                           centers_init='kmeanspp', random_seed=1234)
    _R = np.diag((1, 1, 1))

    for i in six.moves.range(n_iter):
        indices, distances = search_idx.nn_index(np.dot(_R, a).transpose(1, 0).astype(np.float64), 1)
        _R = calc_rot_by_svd(b[:, indices], a)
        # tmp_R = calc_rot_by_svd(b, np.dot(tmp_R, src[:, indices]))
        # _R = np.dot(_R, tmp_R)

    if len(src) > len(dst):
        _t = - _t
        _R = _R.T

    return _R


def calc_rot_by_svd(Y, X):
    U, S, V = np.linalg.svd(np.dot(Y, X.T))
    VU_det = np.linalg.det(np.dot(V, U))
    H = np.diag(np.array([1, 1, VU_det], dtype=np.float64))
    R = np.dot(np.dot(U, H), V)
    return R

def ransac_estimation(x_arr, y_arr, n_ransac=50, thre = 0.025):
    max_cnt = -1
    max_inlier_mask = np.empty(y_arr.shape[1])
    rand_sample = np.array(
        np.random.randint(0, len(max_inlier_mask), (n_ransac, 3)))

    _R = np.empty((3,3))
    _t = np.empty(3)

    random_x = x_arr[:,rand_sample]
    random_x_mean = np.mean(random_x, axis=2)

    random_y = y_arr[:, rand_sample]
    random_y_mean = np.mean(random_y, axis=2)

    for i_ransac in six.moves.range(n_ransac):
        random_x_demean = random_x[:, i_ransac] - random_x_mean[:, i_ransac]
        random_y_demean = random_y[:, i_ransac] - random_y_mean[:, i_ransac]

        _R = calc_rot_by_svd(random_y_demean, random_x_demean)
        _t = random_y_mean[:, i_ransac] - np.dot(_R, random_x_mean[:, i_ransac])
        ## count inliers
        # thre = np.std(
        #     y_demean - np.dot(_R, x_demean), axis=1)[:, np.newaxis]
        inlier_mask3d = (
            np.abs(y_arr - np.dot(_R, x_arr) \
                   - _t[:, np.newaxis]) < thre)
        inlier_mask = np.sum(inlier_mask3d, axis=0)
        inlier_mask = (inlier_mask == 3)
        cnt = np.sum(inlier_mask)
        if cnt > max_cnt:
            max_cnt = cnt
            max_inlier_mask = inlier_mask

    x_mean = np.mean(x_arr[:, max_inlier_mask], axis=1)
    y_mean = np.mean(y_arr[:, max_inlier_mask], axis=1)
    ret_R = calc_rot_by_svd(y_arr[:, max_inlier_mask] - y_mean[:, np.newaxis],
                            x_arr[:, max_inlier_mask] - x_mean[:, np.newaxis])
    ret_cp = y_mean - np.dot(ret_R, x_mean)
    return ret_cp, ret_R


def rotation_ransac(y_arr, x_arr, n_ransac=50, thre=0.05):
    max_cnt = -1
    max_inlier_mask = np.empty(y_arr.shape[1])
    rand_sample = np.array(
        np.random.randint(0, len(max_inlier_mask), (n_ransac, 3)))

    _R = np.empty((3,3))

    random_x = x_arr[:, rand_sample]
    random_y = y_arr[:, rand_sample]

    for i_ransac in six.moves.range(n_ransac):
        _R = calc_rot_by_svd(random_y[:, i_ransac], random_x[:, i_ransac])
        ## count inliers
        inlier_mask3d = (np.abs(y_arr - np.dot(_R, x_arr)) < thre)
        inlier_mask = np.sum(inlier_mask3d, axis=0)
        inlier_mask = (inlier_mask == 3)
        cnt = np.sum(inlier_mask)
        if cnt > max_cnt:
            max_cnt = cnt
            max_inlier_mask = inlier_mask
    if max_cnt < 3:
        # print np.min(np.abs(y_arr - np.dot(_R, x_arr)))
        # print max_cnt
        ret_R = np.zeros((3,3))
    else:
        ret_R = calc_rot_by_svd(y_arr[:, max_inlier_mask],  x_arr[:, max_inlier_mask])
    return ret_R, max_inlier_mask


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

    # print "---"
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
                 base_path = '/home/oshiroy/workspace/cp-net/train_data/OcclusionChallengeICCV2015/models_pcd',
                 objs =['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher'],
                 model_partial= 1,
                 K = None,
                 original_im_size = (640, 480),
                 method="SVD", ver2=False):
        self.eps = eps
        self.distance_sanity = distance_sanity
        self.min_distance = min_distance
        self.method = method
        self.ver2 = ver2
        self.original_im_size = original_im_size

        ## camera parm
        if K is not None:
            self.K = K
        else:
            self.K = np.array([[572.41140, 0, 325.26110],
                               [0, 573.57043, 242.04899],
                               [0, 0, 0]])

        ## for flann
        self.flann_search_idx = []
        self.flann_search_idx_sphere = []
        self.models_pc = []
        self.objs = objs
        pyflann.set_distance_type('euclidean')
        for obj_name in objs:
            pc = pypcd.PointCloud.from_path(
                os.path.join(base_path, obj_name + '.pcd'))
            pc = np.asarray(pc.pc_data.tolist())[:,:3] ## only use xyz
            # pc = pc * np.array([1, -1, -1])[np.newaxis, :] ## negative z
            # trans = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            # pc = np.dot(trans, pc.transpose(1, 0)).transpose(1, 0)

            search_idx = pyflann.FLANN()
            search_idx.build_index(pc[0::model_partial, :], algorithm='kmeans',
                                   centers_init='kmeanspp', random_seed=1234)
            self.flann_search_idx.append(search_idx)
            self.models_pc.append(pc)

            # sphere model index
            norm_pc = np.linalg.norm(pc, axis=1, keepdims=True)
            sphere_pc = pc / norm_pc
            sphere_pc = np.hstack((sphere_pc, norm_pc))

            search_idx_sphere = pyflann.FLANN()
            search_idx_sphere.build_index(sphere_pc, algorithm='kmeans',
                                          centers_init='kmeanspp', random_seed=1234)
            self.flann_search_idx_sphere.append(search_idx_sphere)


    def _sphere_nn_with_norm(self, x, model_idx, dist_algo='euclidean', nn=3):
        norm_x = np.linalg.norm(x, axis=0, keepdims=True)
        sphere_x = x / norm_x
        sphere_x = np.vstack((sphere_x, norm_x))

        ## pyflann search
        pyflann.set_distance_type(dist_algo)
        indices, distances = self.flann_search_idx_sphere[model_idx].nn_index(sphere_x.transpose(1, 0), nn)

        return np.mean(self.models_pc[model_idx][indices], axis=2).transpose(1, 0)


    def _refine_points_triplet(self, x, x2, x3, pc, iteration = 5):
        ret_x = x
        dist_map_pc = np.linalg.norm(pc[:, np.newaxis, :] - pc[:, :, np.newaxis], axis=0)
        mask = 1 - np.diag(np.ones_like(dist_map_pc[0]))

        for i in six.moves.range(iteration):
            dist_map_x = np.linalg.norm(ret_x[:, np.newaxis, :] - x[:, :, np.newaxis], axis=0)
            dist_map_x2 = np.linalg.norm(ret_x[:, np.newaxis, :] - x2[:, :, np.newaxis], axis=0)
            dist_map_x3 = np.linalg.norm(ret_x[:, np.newaxis, :] - x3[:, :, np.newaxis], axis=0)

            dist1 = np.sum(np.abs((dist_map_pc - dist_map_x) * mask), axis=0)
            dist2 = np.sum(np.abs((dist_map_pc - dist_map_x2) * mask), axis=0)
            dist3 = np.sum(np.abs((dist_map_pc - dist_map_x3) * mask), axis=0)

            for j in six.moves.range(len(x)):
                if dist2[j] < dist1[j] and dist2[j] <= dist3[j]:
                    ret_x[:, j] = x2[:, j]
                if dist3[j] < dist1[j] and dist3[j] < dist2[j]:
                    ret_x[:, j] = x3[:, j]
        return ret_x

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

                # flann_ocp = self.models_pc[i_c][flann_ret_unique, :] # shape (n_ocp, num_nn, 3)
                # flann_ocp = flann_ocp.transpose(1, 0)

                refine_mask = cp_mask * flann_mask
                if np.sum(refine_mask) < 10:
                    continue

                t_pc_nonzero = t_pc_reshape[i_b][:, pmask][:, refine_mask]
                y_ocp_nonzero = y_ocp_reshape[i_b][:, pmask][:, refine_mask]
                # y_cp_nonzero = y_cp_reshape[i_b][:, pmask][:, refine_mask]
                # y_cp_nonzero2 = (y_cp_reshape - t_pc_reshape)[i_b][:, pmask][:, refine_mask]

                # y_ocp_flann = self.models_pc[i_c][flann_ret[:,0], :].transpose(1, 0)[:, refine_mask]

                # sphere norm
                # y_ocp_sphere = self._sphere_nn_with_norm(y_ocp_nonzero, i_c)

                # t_pc_demean = (t_pc_nonzero - estimated_cp[i_b, i_c][:, np.newaxis])
                masked_pc_nonzero = t_pc_reshape[i_b]

                # y_concat = y_ocp_nonzero.reshape(3,1,-1)
                # y_ocp_tmp = self._refine_points_triplet(y_ocp_nonzero, y_ocp_flann, y_ocp_sphere, t_pc_nonzero)
                # print "--"
                # print np.mean(np.linalg.norm(y_ocp_nonzero - t_ocp_nonzero, axis=0))

                # print "--"
                # print self.objs[i_c]
                ret_ocp, ret_R = model_base_ransac_estimatation(t_pc_nonzero, y_ocp_nonzero, t_pc_nonzero,
                                                                self.models_pc[i_c].transpose(1,0),
                                                                depth[i_b], K[i_b], t_cp[i_b, i_c],
                                                                pred_mask[i_b, i_c],
                                                                t_rot = t_rot[i_b, i_c],
                                                                y_repeat=1)
                ## icp refinement
                icp_ocp, icp_R = icp(np.dot(ret_R.T, t_pc_nonzero - ret_ocp[:, np.newaxis]),
                                     self.models_pc[i_c].transpose(1,0),
                                     dst_search_idx=self.flann_search_idx[i_c])
                ret_R = np.dot(ret_R, icp_R.T)
                ret_ocp -= np.dot(ret_R, icp_ocp)

                ## icp refinement only rotation
                # icp_R = icp_rotation(np.dot(ret_R.T, t_pc_nonzero - ret_ocp[:, np.newaxis]),
                #                      self.models_pc[i_c].transpose(1,0))
                # ret_R = np.dot(ret_R, icp_R.T)


                if False:
                    imagenet_mean = np.array(
                        [103.939, 116.779, 123.68], dtype=np.float32)[np.newaxis, np.newaxis, :]
                    rgb_out = rgb[i_b].transpose(1,2,0) * 255.0 + imagenet_mean
                    rgb_out = cv2.resize(rgb_out, (depth[i_b].shape[::-1]))
                    cv2.imwrite("depth_rgb.jpg", rgb_out)

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
                                                                n_class, debug=False)

        return np.asarray(ret_cp, dtype=y_cp.dtype), np.asarray(ret_ocp, dtype=y_ocp.dtype), np.asarray(ret_rot, dtype=y_ocp.dtype), np.asarray(ret_rate, dtype=y_ocp.dtype)


def model_base_consensus_accuracy(y_cls, y_cp, y_ocp, t_ocp, t_cp, t_rot, t_pc, depth, K,
                                  eps=0.2, distance_sanity=0.1, method="SVD", ver2=False):
    return ModelBaseConsensusAccuracy(eps=eps,
                                      distance_sanity=distance_sanity,
                                      method=method, ver2=ver2)(y_cls, y_cp, y_ocp, t_ocp, t_cp, t_rot, t_pc, depth, K)
