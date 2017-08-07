import numpy as np
import quaternion
import six

from chainer import cuda
from chainer import function
import chainer.functions as F
from chainer.utils import type_check


def calc_rot_by_svd(Y, X):
    U, S, V = np.linalg.svd(np.dot(Y, X.T))
    VU_det = np.linalg.det(np.dot(V, U))
    H = np.diag(np.array([1, 1, VU_det], dtype=np.float64))
    R = np.dot(np.dot(U, H), V)
    return R

def svd_estimation(y_ocp_nonzero, t_pc_nonzero, n_iter=2):
    ## iterative SVD
    diff_mask = np.ones(t_pc_nonzero.shape[1]).astype(np.bool)
    for j in six.moves.range(n_iter):
        if np.sum(diff_mask) >= 3:
            y_ocp_nonzero = y_ocp_nonzero[:, diff_mask]
            y_ocp_mean = np.mean(y_ocp_nonzero, axis=1, keepdims=True)
            y_ocp_demean = y_ocp_nonzero - y_ocp_mean

            t_pc_nonzero = t_pc_nonzero[:, diff_mask]
            t_pc_mean = np.mean(t_pc_nonzero, axis=1, keepdims=True)
            t_pc_demean = t_pc_nonzero - t_pc_mean

            ret_R = calc_rot_by_svd(t_pc_demean, y_ocp_demean)
            ret_cp = t_pc_mean - np.dot(ret_R, y_ocp_mean)

            diff = t_pc_demean - np.dot(ret_R, y_ocp_demean)
            diff_var = np.sqrt(np.var(diff))
            diff_mask = (np.linalg.norm(diff, axis=0) < diff_var * 1.96).astype(np.bool)

    return ret_cp.ravel(), ret_R


def ransac_estimation(x_arr, y_arr, n_ransac=50, thre = 0.05):
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


def rotation_ransac(y_arr, x_arr, n_ransac=50, thre=0.025):
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


def triplet_ransac_estimation(x_arr, y_arr, pc_arr, n_ransac=50, thre = 0.05):
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



def calc_accuracy_impl(estimated_cp, estimated_ocp, estimated_R,
                       t_cp, t_rot, batch_size, n_class):
    match_cnt = 0.0
    cp_acc = 0.0
    ocp_acc = 0.0
    rot_acc = 0.0
    penalty = np.array([10, 10, 10])
    eval_rate = 0.0
    non_obj_cnt = 0.0

    # print "---"
    for i_b in six.moves.range(batch_size):
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
                # print estimated_R[i_b, i_c]
                # print t_rot[i_b, i_c]
                # print np.dot(estimated_R[i_b, i_c], np.linalg.inv(t_rot[i_b, i_c]))
                # print np.dot(estimated_R[i_b, i_c].T, t_rot[i_b, i_c])
                # print quaternion.from_rotation_matrix(estimated_R[i_b, i_c])
                # print quaternion.from_rotation_matrix(t_rot[i_b, i_c])
                # print quat
                # print "diff angle : " + str(diff_angle) + \
                #     ",  diff dist : " + str(np.linalg.norm(estimated_ocp[i_b, i_c] - t_cp[i_b, i_c]))

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

class DualCenterProposalAccuracy(function.Function):

    ## Support multi-class, one instance per one class

    def __init__(self, eps=0.2,
                 distance_sanity=0.1, min_distance=0.015,
                 method="SVD", ver2=False):
        self.eps = eps
        self.distance_sanity = distance_sanity
        self.min_distance = min_distance
        self.method = method
        self.ver2 = ver2

    def forward(self, inputs):
        y_cls, y_cp, y_ocp, t_ocp, t_cp, t_rot, t_pc = inputs
        ## gpu to cpu
        if cuda.get_array_module(*inputs) != np:
            y_cls = cuda.to_cpu(y_cls)
            y_cp = cuda.to_cpu(y_cp)
            y_ocp = cuda.to_cpu(y_ocp)
            t_ocp = cuda.to_cpu(t_ocp)
            t_cp = cuda.to_cpu(t_cp)
            t_rot = cuda.to_cpu(t_rot)
            t_pc = cuda.to_cpu(t_pc)

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

        if self.ver2:
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

        # print "---"
        # print np.max(np.abs(y_ocp_reshape - t_ocp_reshape), axis=(0,2))
        # print np.min(np.abs(y_ocp_reshape - t_ocp_reshape), axis=(0,2))
        # print np.mean(np.abs(y_ocp_reshape - t_ocp_reshape), axis=(0,2))
        # print np.median(np.abs(y_ocp_reshape - t_ocp_reshape), axis=(0,2))
        # print np.std(np.abs(y_ocp_reshape - t_ocp_reshape), axis=(0,2))

        # print np.sum(pred_mask.reshape(batch_size, n_class -1, -1), axis=2)
        for i_b in six.moves.range(batch_size):
            for i_c in six.moves.range(n_class - 1):
                if np.sum(pred_mask[i_b, i_c]) < 10:
                    continue
                pmask = pred_mask[i_b, i_c].ravel().astype(np.bool)
                y_cp_nonzero = (y_cp[i_b] + t_pc[i_b]).reshape(3, -1)[:, pmask]
                y_cp_mean = np.mean(y_cp_nonzero, axis=1, keepdims=True)
                ## Remove outlier direct Center Point
                cp_mask3d = (y_cp_nonzero - y_cp_mean < 0.1)
                cp_mask = (np.sum(cp_mask3d, axis=0) == 3)

                if np.sum(cp_mask) < 10:
                    continue
                t_pc_nonzero = t_pc_reshape[i_b][:, pmask][:, cp_mask]
                y_ocp_nonzero = y_ocp_reshape[i_b][:, pmask][:, cp_mask]
                t_ocp_nonzero = t_ocp_reshape[i_b][:, pmask][:, cp_mask]
                # print "max : " + str(np.max(np.abs(y_ocp_nonzero - t_ocp_nonzero), axis=1)) + \
                    #     ", mean : " + str(np.mean(y_ocp_nonzero - t_ocp_nonzero, axis=1)) + \
                    #     ", median : " + str(np.median(y_ocp_nonzero - t_ocp_nonzero, axis=1)) + \
                    #     ", min : " + str(np.min(np.abs(y_ocp_nonzero - t_ocp_nonzero), axis=1))

                if self.method == 'SVD':
                    estimated_ocp[i_b, i_c], estimated_R[i_b, i_c] = svd_estimation(y_ocp_nonzero, t_pc_nonzero)

                elif self.method == "RANSAC":
                    estimated_ocp[i_b, i_c], estimated_R[i_b, i_c] = ransac_estimation(y_ocp_nonzero,
                                                                                           t_pc_nonzero, n_ransac=50)
                elif self.method == "DUAL":
                    y_cp_nonzero = y_cp_reshape[i_b][:, pmask][:, cp_mask]
                    y_cp_nonzero2 = (y_cp_reshape - t_pc_reshape)[i_b][:, pmask][:, cp_mask]
                    t_pc_nonzero =  t_pc_reshape[i_b][:, pmask][:, cp_mask]
                    diff_norm = np.abs(np.linalg.norm(y_cp_nonzero2, axis=0) - np.linalg.norm(y_ocp_nonzero, axis=0))
                    weight = np.exp(- diff_norm) / np.sum(np.exp(- diff_norm))
                    # estimated_R[i_b, i_c] = calc_rot_by_svd(
                    #     - y_cp_nonzero2 / np.linalg.norm(y_cp_nonzero2, axis=0),
                    #     y_ocp_nonzero / np.linalg.norm(y_ocp_nonzero, axis=0))
                    tmp_R, inlier_mask = rotation_ransac(- y_cp_nonzero2, y_ocp_nonzero)
                    estimated_ocp[i_b, i_c] = np.mean(
                        (t_pc_nonzero + np.dot(tmp_R, - y_ocp_nonzero)), axis=1)
                    # print "-"
                    # print np.mean(y_ocp_nonzero - t_ocp_nonzero, axis=1)
                    # print np.max(np.abs(y_ocp_nonzero - t_ocp_nonzero), axis=1)
                    # print np.max(np.linalg.norm(y_ocp_nonzero - t_ocp_nonzero, axis=0))
                    # print np.std(y_ocp_nonzero - t_ocp_nonzero,axis=1)

                    # estimated_R[i_b, i_c] = calc_rot_by_svd(- y_cp_nonzero2 * weight[np.newaxis, :],
                    #                                         y_ocp_nonzero * weight[np.newaxis, :])

                    # diff_norm2 = np.abs(np.linalg.norm(t_pc_nonzero - estimated_ocp[i_b, i_c][:, np.newaxis], axis=0)
                    #                     - np.linalg.norm(y_ocp_nonzero, axis=0))
                    # weight2 = np.exp(- diff_norm2) / np.sum(np.exp(- diff_norm2))

                    t_pc_demean = (t_pc_nonzero - estimated_cp[i_b, i_c][:, np.newaxis])

                    # estimated_R[i_b, i_c] = calc_rot_by_svd(t_pc_demean / np.linalg.norm(t_pc_demean, axis=0),
                    #                                         y_ocp_nonzero / np.linalg.norm(y_ocp_nonzero, axis=0))

                    # print (y_ocp_nonzero/  np.linalg.norm(y_ocp_nonzero, axis=0)).shape
                    estimated_R[i_b, i_c], _ = rotation_ransac(t_pc_demean, y_ocp_nonzero)
                    # print np.mean(t_pc_demean - np.dot(tmp_R, y_ocp_nonzero), axis=1)
                    # print np.max(np.abs(t_pc_demean - np.dot(tmp_R, y_ocp_nonzero)), axis=1)
                    # print np.mean(t_pc_demean[:, inlier_mask] - np.dot(tmp_R, y_ocp_nonzero[:, inlier_mask]), axis=1)
                    # print np.max(np.abs(t_pc_demean[:, inlier_mask] - np.dot(tmp_R, y_ocp_nonzero[:, inlier_mask])), axis=1)
                    # estimated_R[i_b, i_c] = np.dot(estimated_R[i_b, i_c], tmp_R)

                    # estimated_ocp[i_b, i_c], estimated_R[i_b, i_c] = triplet_ransac_estimation(y_ocp_nonzero, y_cp_nonzero2, t_pc_nonzero)

        ret_cp, ret_ocp, ret_rot, ret_rate= calc_accuracy_impl(estimated_cp, estimated_ocp, estimated_R, t_cp, t_rot, batch_size, n_class)

        return np.asarray(ret_cp, dtype=y_cp.dtype), np.asarray(ret_ocp, dtype=y_ocp.dtype), np.asarray(ret_rot, dtype=y_ocp.dtype), np.asarray(ret_rate, dtype=y_ocp.dtype)


def dual_cp_accuracy(y_cls, y_cp, y_ocp, t_ocp, t_cp, t_rot, t_pc,
                     eps=0.2, distance_sanity=0.1, method="SVD", ver2=False):
    return DualCenterProposalAccuracy(eps=eps,
                                      distance_sanity=distance_sanity,
                                      method=method, ver2=ver2)(y_cls, y_cp, y_ocp, t_ocp, t_cp, t_rot, t_pc)
