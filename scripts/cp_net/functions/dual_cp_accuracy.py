import numpy
import six

from chainer import cuda
from chainer import function
import chainer.functions as F
from chainer.utils import type_check

def softmax(x, xp):
    return xp.exp(x) / xp.sum(xp.exp(x), axis=1)[:, numpy.newaxis, :, :]

def det3x3(mat):
    ret = mat[0,0] * mat[1,1] * mat[2,2] + mat[0,1] * mat[1,2] * mat[2,0] +\
          mat[0,2] * mat[1,0] * mat[2,1] - mat[0,2] * mat[1,1] * mat[2,0] -\
          mat[0,1] * mat[1,0] * mat[2,2] - mat[0,0] * mat[1,2] * mat[2,1]
    return ret

def calc_rot_by_svd(Y, X, xp):
    U, S, V = xp.linalg.svd(xp.dot(Y, X.T))
    VU_det = det3x3(xp.dot(V, U))
    H = xp.diag(xp.array([1, 1, VU_det], dtype=xp.float64))
    R = xp.dot(xp.dot(U, H), V)
    return R

def svd_estimation(y_ocp_nonzero, t_pc_nonzero, xp, n_iter=2):
    ## iterative SVD
    diff_mask = xp.ones(t_pc_nonzero.shape[1]).astype(numpy.bool)
    for j in six.moves.range(n_iter):
        if xp.sum(diff_mask) >= 3:
            y_ocp_nonzero = y_ocp_nonzero[:, diff_mask]
            y_ocp_mean = xp.mean(y_ocp_nonzero, axis=1)
            y_ocp_demean = y_ocp_nonzero - y_ocp_mean[:,numpy.newaxis]

            t_pc_nonzero = t_pc_nonzero[:, diff_mask]
            t_pc_mean = xp.mean(t_pc_nonzero, axis=1)
            t_pc_demean = t_pc_nonzero - t_pc_mean[:,numpy.newaxis]

            ret_R = calc_rot_by_svd(t_pc_demean, y_ocp_demean, xp)
            ret_cp = t_pc_mean - xp.dot(ret_R, y_ocp_mean)

            diff = t_pc_demean - xp.dot(ret_R, y_ocp_demean)
            diff_var = xp.sqrt(xp.var(diff))
            diff_mask = (xp.linalg.norm(diff, axis=0) < diff_var * 1.96).astype(numpy.bool)

    return ret_cp, ret_R


def ransac_estimation(y_ocp_nonzero, t_pc_nonzero, xp, n_ransac=50):
    # thre = xp.sqrt(xp.sum(xp.var(y_cp_nonzero - y_cp_mean[:, numpy.newaxis], axis=1)))
    max_cnt = -1
    max_inlier_mask = xp.empty(t_pc_nonzero.shape[1])
    rand_sample = xp.array(
        numpy.random.randint(0, len(max_inlier_mask), (n_ransac, 3)))

    _R = xp.empty((3,3))
    _t = xp.empty(3)

    random_ocp = y_ocp_nonzero[:,rand_sample]
    random_ocp_mean = xp.mean(random_ocp, axis=2)

    random_pc = t_pc_nonzero[:, rand_sample]
    random_pc_mean = xp.mean(random_pc, axis=2)

    for i_ransac in six.moves.range(n_ransac):
        random_ocp_demean = random_ocp[:, i_ransac] - random_ocp_mean[:, i_ransac]
        random_pc_demean = random_pc[:, i_ransac] - random_pc_mean[:, i_ransac]

        _R = calc_rot_by_svd(random_pc_demean, random_ocp_demean, xp)
        _t = random_pc_mean[:, i_ransac] - xp.dot(_R, random_ocp_mean[:, i_ransac])
        ## count inliers
        # thre = xp.std(
        #     t_pc_demean - xp.dot(_R, y_ocp_demean), axis=1)[:, numpy.newaxis]
        thre = 0.025
        inlier_mask3d = (
            xp.abs(t_pc_nonzero - xp.dot(_R, y_ocp_nonzero) \
                   - _t[:, numpy.newaxis]) < thre * 2)
        inlier_mask = xp.sum(inlier_mask3d, axis=0)
        inlier_mask = (inlier_mask == 3)
        cnt = xp.sum(inlier_mask)
        if cnt > max_cnt:
            max_cnt = cnt
            max_inlier_mask = inlier_mask

    y_ocp_mean = xp.mean(y_ocp_nonzero[:, max_inlier_mask], axis=1)
    t_pc_mean = xp.mean(t_pc_nonzero[:, max_inlier_mask], axis=1)
    ret_R = calc_rot_by_svd(t_pc_nonzero[:, max_inlier_mask] - t_pc_mean[:,numpy.newaxis],
                            y_ocp_nonzero[:, max_inlier_mask] - y_ocp_mean[:,numpy.newaxis], xp)
    ret_cp = t_pc_mean - xp.dot(ret_R, y_ocp_mean)
    return ret_cp, ret_R


def calc_accuracy_impl(estimated_cp, estimated_ocp, t_cp, batch_size, n_class, xp):
    match_cnt = 0.0
    cp_acc = 0.0
    ocp_acc = 0.0
    rot_acc = 0.0
    penalty = xp.array([20, 20, 20])

    for i_b in six.moves.range(batch_size):
        for i_c in six.moves.range(1, n_class):
            if xp.linalg.norm(estimated_cp[i_b, i_c] - penalty) > 0.001 \
               and xp.linalg.norm(t_cp[i_b, i_c]) != 0:
                match_cnt += 1.0
                cp_acc += xp.linalg.norm(estimated_cp[i_b, i_c] - t_cp[i_b, i_c])
                ocp_acc += xp.linalg.norm(estimated_ocp[i_b, i_c] - t_cp[i_b, i_c])

    if match_cnt > 0:
        cp_acc /= match_cnt
        ocp_acc /= match_cnt
    else:
        cp_acc = xp.linalg.norm(penalty)
        ocp_acc = xp.linalg.norm(penalty)
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
        xp = cuda.get_array_module(*inputs)
        y_cls, y_cp, y_ocp, t_cp, t_rot, t_pc = inputs
        batch_size, n_class = y_cls.shape[:2]

        y_cls = softmax(y_cls, xp)

        prob = xp.max(y_cls, axis=1)
        pred = xp.argmax(y_cls, axis=1)

        # threshold
        pred[prob < self.eps] = 0

        # with nonnan_mask
        pred_mask = xp.invert(xp.isnan(t_pc[:,0,:]))

        t_pc[t_pc != t_pc] = 0
        estimated_cp = xp.empty((batch_size, n_class, 3))
        estimated_ocp = xp.empty((batch_size, n_class, 3))
        estimated_R = xp.empty((batch_size, n_class, 3, 3))
        penalty = xp.array([20, 20, 20])
        pred_mask_tmp = pred_mask

        ## check dual cp distance sanity
        if self.distance_sanity:
            dist_cp = xp.linalg.norm(y_cp.reshape(batch_size, 3, -1), axis=1)
            dist_cp_std =  xp.std(dist_cp)
            dist_ocp = xp.linalg.norm(y_ocp.reshape(batch_size, 3, -1), axis=1)
            dist_mask = (xp.abs(dist_cp - dist_ocp) < self.distance_sanity).reshape(pred_mask.shape)
            pred_mask = pred_mask * dist_mask

        # pred_mask = pred_mask * cp_pred_mask * ocp_pred_mask
        for i_b in six.moves.range(batch_size):
            for i_c in six.moves.range(1, n_class):
                pmask = pred_mask[i_b] * (pred[i_b] == i_c)
                pmask_tmp = pred_mask_tmp[i_b] * (pred[i_b] == i_c)
                # print str(xp.sum(pmask)) + "  :  " + str(xp.sum(pmask_tmp))
                if xp.sum(pmask) < 50:
                    estimated_cp[i_b, i_c] = penalty
                    estimated_ocp[i_b, i_c] = penalty
                else:
                    ## First of All, Calculate Center point directly
                    prob_weight = (pmask * prob[i_b]) / xp.sum(pmask * prob[i_b])
                    estimated_cp[i_b, i_c] = xp.sum(
                        (prob_weight * (y_cp[i_b] + t_pc[i_b])).reshape(3, -1), axis=1)

                    y_cp_nonzero = (y_cp[i_b] + t_pc[i_b]).reshape(3, -1)[:, pmask.ravel()]
                    y_cp_mean = xp.mean(y_cp_nonzero, axis=1)

                    ## Remove outlier direct Center point
                    cp_demean = y_cp_nonzero - y_cp_mean[:, numpy.newaxis]
                    cp_std = xp.std(cp_demean)
                    cp_mask = (xp.linalg.norm(cp_demean, axis=0) < cp_std * 3)

                    y_cp_nonzero = (y_cp[i_b] + t_pc[i_b]).reshape(3, -1)[:, pmask.ravel()][:, cp_mask]
                    t_pc_nonzero = t_pc[i_b].reshape(3,-1)[:, pmask.ravel()][:, cp_mask]
                    y_ocp_nonzero = y_ocp[i_b].reshape(3,-1)[:, pmask.ravel()][:, cp_mask]

                    y_cp_nonzero2 = y_cp[i_b].reshape(3,-1)[:, pmask.ravel()][:,cp_mask]

                    if self.method == 'SVD':
                        estimated_ocp[i_b, i_c], R = svd_estimation(y_ocp_nonzero, t_pc_nonzero, xp)

                    elif self.method == "RANSAC":
                        estimated_ocp[i_b, i_c], R = ransac_estimation(y_ocp_nonzero,
                                                                       t_pc_nonzero, xp, n_ransac=50)
                    elif self.method == "DUAL":
                        # print y_cp_nonzero.shape
                        diff_norm = xp.linalg.norm(y_cp_nonzero2, axis=0) - xp.linalg.norm(y_ocp_nonzero, axis=0)
                        weight = xp.exp(- diff_norm) / xp.sum(xp.exp(- diff_norm))
                        estimated_ocp[i_b, i_c] = xp.sum(weight[numpy.newaxis, :] * y_cp_nonzero, axis=1)
                        R = calc_rot_by_svd(y_cp_nonzero2, y_ocp_nonzero, xp)

        ret_cp, ret_ocp = calc_accuracy_impl(estimated_cp, estimated_ocp, t_cp, batch_size, n_class, xp)

        return xp.asarray(ret_cp, dtype=y_cp.dtype), xp.asarray(ret_ocp, dtype=y_ocp.dtype),


def dual_cp_accuracy(y_cls, y_cp, y_ocp, t_cp, t_rot, t_pc,
                     eps=0.2, distance_sanity=0.1, method="SVD", ver2=False):
    return DualCenterProposalAccuracy(eps=eps,
                                      distance_sanity=distance_sanity,
                                      method=method, ver2=ver2)(y_cls, y_cp, y_ocp, t_cp, t_rot, t_pc)
