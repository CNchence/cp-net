import numpy as np
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


def ransac_estimation(y_ocp_nonzero, t_pc_nonzero, n_ransac=50):
    # thre = np.sqrt(np.sum(np.var(y_cp_nonzero - y_cp_mean[:, np.newaxis], axis=1)))
    max_cnt = -1
    max_inlier_mask = np.empty(t_pc_nonzero.shape[1])
    rand_sample = np.array(
        np.random.randint(0, len(max_inlier_mask), (n_ransac, 3)))

    _R = np.empty((3,3))
    _t = np.empty(3)

    random_ocp = y_ocp_nonzero[:,rand_sample]
    random_ocp_mean = np.mean(random_ocp, axis=2)

    random_pc = t_pc_nonzero[:, rand_sample]
    random_pc_mean = np.mean(random_pc, axis=2)

    for i_ransac in six.moves.range(n_ransac):
        random_ocp_demean = random_ocp[:, i_ransac] - random_ocp_mean[:, i_ransac]
        random_pc_demean = random_pc[:, i_ransac] - random_pc_mean[:, i_ransac]

        _R = calc_rot_by_svd(random_pc_demean, random_ocp_demean)
        _t = random_pc_mean[:, i_ransac] - np.dot(_R, random_ocp_mean[:, i_ransac])
        ## count inliers
        # thre = np.std(
        #     t_pc_demean - np.dot(_R, y_ocp_demean), axis=1)[:, np.newaxis]
        thre = 0.025
        inlier_mask3d = (
            np.abs(t_pc_nonzero - np.dot(_R, y_ocp_nonzero) \
                   - _t[:, np.newaxis]) < thre * 2)
        inlier_mask = np.sum(inlier_mask3d, axis=0)
        inlier_mask = (inlier_mask == 3)
        cnt = np.sum(inlier_mask)
        if cnt > max_cnt:
            max_cnt = cnt
            max_inlier_mask = inlier_mask

    y_ocp_mean = np.mean(y_ocp_nonzero[:, max_inlier_mask], axis=1)
    t_pc_mean = np.mean(t_pc_nonzero[:, max_inlier_mask], axis=1)
    ret_R = calc_rot_by_svd(t_pc_nonzero[:, max_inlier_mask] - t_pc_mean[:, np.newaxis],
                            y_ocp_nonzero[:, max_inlier_mask] - y_ocp_mean[:, np.newaxis])
    ret_cp = t_pc_mean - np.dot(ret_R, y_ocp_mean)
    return ret_cp, ret_R


def calc_accuracy_impl(estimated_cp, estimated_ocp, t_cp, batch_size, n_class):
    match_cnt = 0.0
    cp_acc = 0.0
    ocp_acc = 0.0
    rot_acc = 0.0
    penalty = np.array([20, 20, 20])

    for i_b in six.moves.range(batch_size):
        for i_c in six.moves.range(1, n_class):
            if np.linalg.norm(estimated_cp[i_b, i_c] - penalty) > 0.001 \
               and np.linalg.norm(t_cp[i_b, i_c]) != 0:
                match_cnt += 1.0
                cp_acc += np.linalg.norm(estimated_cp[i_b, i_c] - t_cp[i_b, i_c])
                ocp_acc += np.linalg.norm(estimated_ocp[i_b, i_c] - t_cp[i_b, i_c])

    if match_cnt > 0:
        cp_acc /= match_cnt
        ocp_acc /= match_cnt
    else:
        cp_acc = np.linalg.norm(penalty)
        ocp_acc = np.linalg.norm(penalty)
        rot_acc = 180

    return cp_acc, ocp_acc

class DualCenterProposalAccuracy(function.Function):

    ## Support multi-class, one instance per one class

    def __init__(self, eps=0.2, cp_eps=0.7,
                 distance_sanity=0.1, method="SVD", ver2=False):
        self.eps = eps
        self.cp_eps = cp_eps
        self.distance_sanity = distance_sanity
        self.method = method
        self.ver2 = ver2

    def forward(self, inputs):
        y_cls, y_cp, y_ocp, t_cp, t_rot, t_pc = inputs

        if cuda.get_array_module(*inputs) != np:
            y_cls = cuda.to_cpu(y_cls)
            y_cp = cuda.to_cpu(y_cp)
            y_ocp = cuda.to_cpu(y_ocp)
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

        masks = np.zeros_like(y_cls)
        for i_b in six.moves.range(batch_size):
            for i_c in six.moves.range(1, n_class):
                masks[i_b, i_c] = (pred[i_b] == i_c)

        # with nonnan_mask
        pred_mask = np.invert(np.isnan(t_pc[:,0,:]))

        t_pc[t_pc != t_pc] = 0
        estimated_cp = np.empty((batch_size, n_class, 3))
        estimated_ocp = np.empty((batch_size, n_class, 3))
        estimated_R = np.empty((batch_size, n_class, 3, 3))
        penalty = np.array([20, 20, 20])

        if self.ver2:
            y_cp = masks[:, :, np.newaxis, :, :] * y_cp.reshape(batch_size, n_class, 3, img_h, img_w)
            y_ocp = masks[:, :, np.newaxis, :, :] * y_ocp.reshape(batch_size, n_class, 3, img_h, img_w)
            y_cp = np.sum(y_cp, axis=1)
            y_ocp = np.sum(y_ocp, axis=1)

        ## check dual cp distance sanity
        if self.distance_sanity:
            dist_cp = np.linalg.norm(y_cp, axis=1)
            dist_ocp = np.linalg.norm(y_ocp, axis=1)
            dist_mask = (np.abs(dist_cp - dist_ocp) < self.distance_sanity)
            pred_mask = pred_mask * dist_mask

        pred_mask = masks * pred_mask[:, np.newaxis, :, :]

        ## First of All, Calculate Center point directly
        prob_weight = (pred_mask * y_cls) / (1e-15 + np.sum(
            (pred_mask * y_cls).reshape(batch_size, n_class, -1), axis=2)[:, :, np.newaxis, np.newaxis])
        estimated_cp = np.sum(
            (prob_weight[:, :, np.newaxis, :, :] * \
             (y_cp + t_pc)[:, np.newaxis, :, :, :]).reshape(batch_size, n_class, 3, -1), axis=3)

        y_cp_reshape = (y_cp + t_pc).reshape(batch_size, 3, -1)
        t_pc_reshape = t_pc.reshape(batch_size, 3,-1)
        y_ocp_reshape = y_ocp.reshape(batch_size, 3,-1)

        for i_b in six.moves.range(batch_size):
            for i_c in six.moves.range(1, n_class):
                if np.sum(pred_mask[i_b, i_c]) < 50:
                    estimated_cp[i_b, i_c] = penalty
                    estimated_ocp[i_b, i_c] = penalty
                    estimated_R[i_b, i_c] = np.diag((1,1,1))
                else:
                    pmask = pred_mask[i_b, i_c].ravel().astype(np.bool)
                    y_cp_nonzero = (y_cp[i_b] + t_pc[i_b]).reshape(3, -1)[:, pmask]
                    y_cp_mean = np.mean(y_cp_nonzero, axis=1, keepdims=True)

                    ## Remove outlier direct Center point
                    cp_demean = y_cp_nonzero - y_cp_mean
                    cp_mask = (np.linalg.norm(cp_demean, axis=0) < np.std(cp_demean) * 3)

                    t_pc_nonzero = t_pc_reshape[i_b][:, pmask][:, cp_mask]
                    y_ocp_nonzero = y_ocp_reshape[i_b][:, pmask][:, cp_mask]

                    if self.method == 'SVD':
                        estimated_ocp[i_b, i_c], R = svd_estimation(y_ocp_nonzero, t_pc_nonzero)

                    elif self.method == "RANSAC":
                        estimated_ocp[i_b, i_c], R = ransac_estimation(y_ocp_nonzero,
                                                                       t_pc_nonzero, n_ransac=50)
                    elif self.method == "DUAL":
                        y_cp_nonzero = y_cp_reshape[i_b][:, pmask][:, cp_mask]
                        y_cp_nonzero2 = y_cp_reshape - t_pc_reshape
                        diff_norm = np.linalg.norm(y_cp_nonzero2, axis=0) - np.linalg.norm(y_ocp_nonzero, axis=0)
                        weight = np.exp(- diff_norm) / np.sum(np.exp(- diff_norm))
                        estimated_ocp[i_b, i_c] = np.sum(weight[np.newaxis, :] * y_cp_nonzero, axis=1)
                        R = calc_rot_by_svd(y_cp_nonzero2, y_ocp_nonzero)

        ret_cp, ret_ocp = calc_accuracy_impl(estimated_cp, estimated_ocp, t_cp, batch_size, n_class)

        return np.asarray(ret_cp, dtype=y_cp.dtype), np.asarray(ret_ocp, dtype=y_ocp.dtype),


def dual_cp_accuracy(y_cls, y_cp, y_ocp, t_cp, t_rot, t_pc,
                     eps=0.2, distance_sanity=0.1, method="SVD", ver2=False):
    return DualCenterProposalAccuracy(eps=eps,
                                      distance_sanity=distance_sanity,
                                      method=method, ver2=ver2)(y_cls, y_cp, y_ocp, t_cp, t_rot, t_pc)
