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

class DualCenterProposalAccuracy(function.Function):

    ## Support multi-class, one instance per one class

    def __init__(self, eps=0.2, cp_eps=0.7,
                 distance_sanity=0.1, method="SVD"):
        self.eps = eps
        self.cp_eps = cp_eps
        self.distance_sanity = distance_sanity
        self.method = method

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y_cls, y_cp, t_ocp, y_ocp, t_cp, t_pc, cp_mask, ocp_mask = inputs
        batch_size, n_class = y_cls.shape[:2]

        y_cls = softmax(y_cls, xp)
        cp_mask = softmax(cp_mask, xp)
        ocp_mask = softmax(ocp_mask, xp)

        prob = xp.max(y_cls, axis=1)
        pred = xp.argmax(y_cls, axis=1)
        cp_prob = xp.max(cp_mask, axis=1)
        ocp_prob = xp.max(ocp_mask, axis=1)
        cp_pred = xp.argmax(cp_mask, axis=1)
        ocp_pred = xp.argmax(ocp_mask, axis=1)

        cp_pred_mask = ((cp_prob > self.cp_eps) * cp_pred).astype(numpy.bool)
        ocp_pred_mask = ((ocp_prob > self.cp_eps) * ocp_pred).astype(numpy.bool)

        # threshold
        pred[prob < self.eps] = 0

        # with nonnan_mask
        pred_mask = xp.invert(xp.isnan(t_pc[:,0,:]))

        t_pc[t_pc != t_pc] = 0
        estimated_cp = xp.empty((batch_size, n_class, 3))
        estimated_ocp = xp.empty((batch_size, n_class, 3))
        penalty = xp.array([20, 20, 20])

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
                if xp.sum(pmask) < 50:
                    estimated_cp[i_b, i_c] = penalty
                    estimated_ocp[i_b, i_c] = penalty
                else:
                    ## First of All, Calculate Center point directly
                    prob_weight = (pmask * prob[i_b]) / xp.sum(pmask * prob[i_b])
                    estimated_cp[i_b, i_c] = xp.sum((prob_weight * (y_cp[i_b] + t_pc[i_b])).reshape(3, -1), axis=1)

                    y_cp_nonzero = (y_cp[i_b] + t_pc[i_b]).reshape(3, -1)[:, pmask.ravel()]
                    y_cp_mean = xp.mean(y_cp_nonzero, axis=1)

                    ## Remove outlier direct Center point
                    cp_demean = y_cp_nonzero - y_cp_mean[:, numpy.newaxis]
                    cp_std = xp.std(cp_demean)
                    cp_mask = (xp.linalg.norm(cp_demean, axis=0) < cp_std * 3)

                    t_pc_nonzero = t_pc[i_b].reshape(3,-1)[:, pmask.ravel()][:, cp_mask]
                    t_pc_mean = xp.mean(t_pc_nonzero, axis=1)
                    t_pc_demean = t_pc_nonzero - t_pc_mean[:,numpy.newaxis]

                    y_ocp_nonzero = y_ocp[i_b].reshape(3,-1)[:, pmask.ravel()][:, cp_mask]
                    y_ocp_mean = xp.mean(y_ocp_nonzero, axis=1)
                    y_ocp_demean = y_ocp_nonzero - y_ocp_mean[:,numpy.newaxis]

                    t_ocp_nonzero = t_ocp[i_b].reshape(3,-1)[:, pmask.ravel()][:,cp_mask]

                    if self.method == 'SVD':
                        ## iterative SVD
                        diff_mask = xp.ones(t_pc_nonzero.shape[1]).astype(numpy.bool)
                        R = xp.empty((3,3))
                        for j in six.moves.range(2):
                            if xp.sum(diff_mask) >= 3:
                                y_ocp_nonzero = y_ocp_nonzero[:, diff_mask]
                                y_ocp_mean = xp.mean(y_ocp_nonzero, axis=1)
                                y_ocp_demean = y_ocp_nonzero - y_ocp_mean[:,numpy.newaxis]

                                t_pc_nonzero = t_pc_nonzero[:, diff_mask]
                                t_pc_mean = xp.mean(t_pc_nonzero, axis=1)
                                t_pc_demean = t_pc_nonzero - t_pc_mean[:,numpy.newaxis]

                                ## rotation matrix estimation using SVD method
                                U, S, V = xp.linalg.svd(xp.dot(t_pc_demean, y_ocp_demean.T))

                                # det(U, V)
                                VU_det = det3x3(xp.dot(V, U))
                                H = xp.diag(xp.array([1,1,VU_det], dtype=xp.float64))
                                R = xp.dot(xp.dot(U, H), V)

                                estimated_ocp[i_b, i_c] = t_pc_mean - xp.dot(R, y_ocp_mean)

                                diff = t_pc_demean - xp.dot(R, y_ocp_demean)
                                diff_var = xp.sqrt(xp.var(diff))
                                diff_mask = (xp.linalg.norm(diff, axis=0) < diff_var * 1.96).astype(numpy.bool)

                        # print R
                        # print "--"
                        # print xp.sum(diff_mask)
                        # print xp.sum(test_mask)
                        # print xp.sum(diff_mask * test_mask)

                        # print xp.mean(t_pc_demean - xp.dot(R, y_ocp_demean), axis = 1)
                        # print xp.max(xp.abs(t_pc_demean - xp.dot(R, y_ocp_demean)), axis=1)
                        # print xp.sqrt(xp.var(t_pc_demean - xp.dot(R, y_ocp_demean), axis=1))

                    elif self.method == "RANSAC":
                        # thre = xp.sqrt(xp.sum(xp.var(y_cp_nonzero - y_cp_mean[:, numpy.newaxis], axis=1)))

                        n_ransac = 50
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
                            random_ocp_demean = random_ocp[:, i_ransac] \
                                                - random_ocp_mean[:, i_ransac]
                            random_pc_demean = random_pc[:, i_ransac] \
                                               - random_pc_mean[:, i_ransac]
                            U, S, V = xp.linalg.svd(xp.dot(random_pc_demean, random_ocp_demean.T))
                            # det(U, V)
                            VU_det = det3x3(xp.dot(V, U))
                            H = xp.diag(xp.array([1,1,VU_det], dtype=xp.float64))
                            _R = xp.dot(xp.dot(U, H), V)
                            _t = random_pc_mean[:, i_ransac] \
                                 - xp.dot(_R, random_ocp_mean[:, i_ransac])

                            ## count inliers
                            thre = xp.std(
                                t_pc_demean - xp.dot(_R, y_ocp_demean), axis=1)[:, numpy.newaxis]
                            inlier_mask3d = (
                                xp.abs(t_pc_nonzero - xp.dot(_R, y_ocp_nonzero) \
                                       - _t[:, numpy.newaxis]) < thre * 2)
                            inlier_mask = xp.sum(inlier_mask3d, axis=0)
                            inlier_mask = (inlier_mask == 3)
                            cnt = xp.sum(inlier_mask)
                            if cnt > max_cnt:
                                max_cnt = cnt
                                max_inlier_mask = inlier_mask

                        y_ocp_nonzero = y_ocp_nonzero[:, max_inlier_mask]
                        y_ocp_mean = xp.mean(y_ocp_nonzero, axis=1)
                        y_ocp_demean = y_ocp_nonzero - y_ocp_mean[:,numpy.newaxis]

                        t_pc_nonzero = t_pc_nonzero[:, max_inlier_mask]
                        t_pc_mean = xp.mean(t_pc_nonzero, axis=1)
                        t_pc_demean = t_pc_nonzero - t_pc_mean[:,numpy.newaxis]

                        ## rotation matrix estimation using SVD method
                        U, S, V = xp.linalg.svd(xp.dot(t_pc_demean, y_ocp_demean.T))
                        VU_det = det3x3(xp.dot(V, U))
                        H = xp.diag(xp.array([1,1,VU_det], dtype=xp.float64))
                        R = xp.dot(xp.dot(U, H), V)
                        estimated_ocp[i_b, i_c] = t_pc_mean - xp.dot(R, y_ocp_mean)

        match_cnt = 0.0
        ret_cp = 0.0
        ret_ocp = 0.0

        for i_b in six.moves.range(batch_size):
            for i_c in six.moves.range(1, n_class):
                if xp.linalg.norm(estimated_cp[i_b, i_c] - penalty) > 0.001 \
                   or xp.linalg.norm(t_cp[i_b, i_c]) != 0:
                    match_cnt += 1.0
                    ret_cp += xp.linalg.norm(estimated_cp[i_b, i_c] - t_cp[i_b, i_c])
                    ret_ocp += xp.linalg.norm(estimated_ocp[i_b, i_c] - t_cp[i_b, i_c])
        if match_cnt > 0:
            ret_cp /= match_cnt
            ret_ocp /= match_cnt
        else:
            ret_cp = xp.linalg.norm(penalty)
            ret_ocp = xp.linalg.norm(penalty)

        return xp.asarray(ret_cp, dtype=y_cp.dtype), xp.asarray(ret_ocp, dtype=y_ocp.dtype),


def dual_cp_accuracy(y_cls, y_cp, t_ocp, y_ocp, t_cp, t_pc, cp_mask, ocp_mask,
                     eps=0.2, distance_sanity=0.1):
    return DualCenterProposalAccuracy(eps=eps,
                                      distance_sanity=distance_sanity,
                                      method="SVD")(y_cls, y_cp, t_ocp, y_ocp, t_cp,
                                                    t_pc, cp_mask, ocp_mask)
