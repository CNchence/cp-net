import numpy
import six

from chainer import cuda
from chainer import function
import chainer.functions as F
from chainer.utils import type_check

def softmax(x, xp):
    return xp.exp(x) / xp.sum(xp.exp(x), axis=1)[:, numpy.newaxis, :, :]


class DualCenterProposalAccuracy(function.Function):

    ## Support one class per flame

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

        # class count
        cls_count = xp.empty(y_cls.shape[:2])
        for i in range(batch_size):
            cls_count[i] = xp.bincount(pred[i].ravel(), minlength= n_class)

        obj_cls = numpy.argmax(cls_count[:,1:], axis=1) + 1

        # with nonnan_mask
        pred_mask = xp.invert(xp.isnan(t_pc[:,0,:])) * (pred == obj_cls[:, numpy.newaxis, numpy.newaxis])

        t_pc[t_pc != t_pc] = 0
        pred_mask_tmp = pred_mask
        estimated_cp = xp.empty((batch_size, 3))
        estimated_ocp = xp.empty((batch_size, 3))
        penalty = xp.array([20, 20, 20])

        ## check dual cp distance sanity
        if self.distance_sanity:
            dist_cp = xp.linalg.norm(y_cp.reshape(batch_size, 3, -1), axis=1)
            dist_cp_std =  xp.std(dist_cp)
            dist_ocp = xp.linalg.norm(y_ocp.reshape(batch_size, 3, -1), axis=1)
            dist_mask = (xp.abs(dist_cp - dist_ocp) < self.distance_sanity).reshape(pred_mask.shape)
            pred_mask = pred_mask * dist_mask

        # pred_mask = pred_mask * cp_pred_mask * ocp_pred_mask

        for i in range(batch_size):
            if pred_mask[i].sum() < 3:
                estimated_cp[i] = penalty
                estimated_ocp[i] = penalty
            else:
                ## First of All, Calculate Center point directly
                prob_weight = (pred_mask[i] * prob[i]) / xp.sum(pred_mask[i] * prob[i])
                estimated_cp[i] = xp.sum((prob_weight * (y_cp[i] + t_pc[i])).reshape(3, -1), axis=1)

                y_cp_nonzero = (y_cp[i] + t_pc[i]).reshape(3, -1)[:, pred_mask[i].ravel()]
                y_cp_mean = xp.mean(y_cp_nonzero, axis=1)

                ## Remove outlier direct Center point
                cp_demean = y_cp_nonzero - y_cp_mean[:, numpy.newaxis]
                cp_std = xp.std(cp_demean)
                cp_mask = (xp.linalg.norm(cp_demean, axis=0) < cp_std * 3)

                # print "---"
                # print estimated_cp[i] - t_cp[i]
                # print xp.mean((y_cp[i] + t_pc[i]).reshape(3,-1)[:, pred_mask[i].ravel()][:, cp_mask], axis=1) - t_cp[i]
                # print numpy.median(cuda.to_cpu((y_cp[i] + t_pc[i]).reshape(3,-1)[:, pred_mask[i].ravel()][:, cp_mask]), axis=1) - cuda.to_cpu(t_cp[i])
                t_pc_nonzero = t_pc[i].reshape(3,-1)[:, pred_mask[i].ravel()][:, cp_mask]
                t_pc_mean = xp.mean(t_pc_nonzero, axis=1)
                t_pc_demean = t_pc_nonzero - t_pc_mean[:,numpy.newaxis]

                y_ocp_nonzero = y_ocp[i].reshape(3,-1)[:, pred_mask[i].ravel()][:, cp_mask]
                y_ocp_mean = xp.mean(y_ocp_nonzero, axis=1)
                y_ocp_demean = y_ocp_nonzero - y_ocp_mean[:,numpy.newaxis]

                t_ocp_nonzero = t_ocp[i].reshape(3,-1)[:, pred_mask[i].ravel()][:,cp_mask]


                if self.method == 'SVD':
                    ## iterative SVD
                    diff_mask = xp.ones(t_pc_nonzero.shape[1]).astype(numpy.bool)
                    R = xp.empty((3,3))
                    for j in range(2):
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
                            VU = xp.dot(V, U)
                            VU_det = VU[0,0] * VU[1,1] * VU[2,2] + VU[0,1] * VU[1,2] * VU[2,0] +\
                                     VU[0,2] * VU[1,0] * VU[2,1] - VU[0,2] * VU[1,1] * VU[2,0] -\
                                     VU[0,1] * VU[1,0] * VU[2,2] - VU[0,0] * VU[1,2] * VU[2,1]
                            H = xp.diag(xp.array([1,1,VU_det], dtype=xp.float64))
                            R = xp.dot(xp.dot(U, H), V)

                            estimated_ocp[i] = t_pc_mean - xp.dot(R, y_ocp_mean)

                            diff = t_pc_demean - xp.dot(R, y_ocp_demean)
                            diff_var = xp.sqrt(xp.var(diff))
                            diff_mask = (xp.linalg.norm(diff, axis=0) < diff_var * 1.96).astype(numpy.bool)

                            # print "loop : "  + str(j)
                            # print xp.mean(t_pc_demean - xp.dot(R, y_ocp_demean), axis = 1)
                            # print xp.max(xp.abs(t_pc_demean - xp.dot(R, y_ocp_demean)), axis=1)
                            # print xp.sqrt(xp.var(t_pc_demean - xp.dot(R, y_ocp_demean), axis=1))

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
                    max_cnt = -1
                    max_inlier_mask = xp.empty(t_pc_nonzero.shape[1])

                    n_ransac = 50
                    rand_sample = xp.array(numpy.random.randint(0, len(max_inlier_mask), (n_ransac, 3)))

                    _R = xp.empty((3,3))
                    _t = xp.empty(3)

                    random_ocp = y_ocp_nonzero[:,rand_sample]
                    random_ocp_mean = xp.mean(random_ocp, axis=2)

                    random_pc = t_pc_nonzero[:, rand_sample]
                    random_pc_mean = xp.mean(random_pc, axis=2)

                    for i_ransac in range(n_ransac):
                        random_ocp_demean = random_ocp[:, i_ransac] - random_ocp_mean[:, i_ransac]
                        random_pc_demean = random_pc[:, i_ransac] - random_pc_mean[:, i_ransac]

                        U, S, V = xp.linalg.svd(xp.dot(random_pc_demean, random_ocp_demean.T))
                        # det(U, V)
                        VU = xp.dot(V, U)
                        VU_det = VU[0,0] * VU[1,1] * VU[2,2] + VU[0,1] * VU[1,2] * VU[2,0] +\
                                 VU[0,2] * VU[1,0] * VU[2,1] - VU[0,2] * VU[1,1] * VU[2,0] -\
                                 VU[0,1] * VU[1,0] * VU[2,2] - VU[0,0] * VU[1,2] * VU[2,1]
                        H = xp.diag(xp.array([1,1,VU_det], dtype=xp.float64))

                        _R = xp.dot(xp.dot(U, H), V)
                        # _t = (random_pc_mean[:, i_ransac] - xp.dot(_R, random_ocp_mean[:, i_ransac])

                        ## count inliers
                        thre = xp.std(t_pc_demean - xp.dot(_R, y_ocp_demean), axis=1)[:, numpy.newaxis]
                        inlier_mask3d = (
                            xp.abs(t_pc_nonzero - xp.dot(_R, y_ocp_nonzero) - _t[:, numpy.newaxis]) < thre * 2)
                        inlier_mask = xp.sum(inlier_mask3d, axis=0)
                        inlier_mask = (inlier_mask == 3)
                        cnt = xp.sum(inlier_mask)
                        if cnt > max_cnt:
                            max_cnt = cnt
                            max_inlier_mask = inlier_mask
                            # print "-"
                            # print xp.sum(max_inlier_mask)
                            # print len(max_inlier_mask)
                    # test_mask = (xp.linalg.norm(t_ocp_nonzero - y_ocp_nonzero, axis=0) < thre / 2.0)
                    # print xp.sum(1 - test_mask)
                    # print xp.sum(1 - max_inlier_mask)
                    # print xp.sum(max_inlier_mask)
                    # print xp.sum(max_inlier_mask * (1 - test_mask))

                    # print (t_ocp_nonzero - y_ocp_nonzero)[:, max_inlier_mask].shape
                    # print xp.max(xp.abs((t_ocp_nonzero - y_ocp_nonzero)[:, max_inlier_mask]), axis=1)
                    # print xp.mean((t_ocp_nonzero - y_ocp_nonzero)[:, max_inlier_mask], axis=1)
                    # print xp.std((t_ocp_nonzero - y_ocp_nonzero)[:, max_inlier_mask], axis=1)

                    y_ocp_nonzero = y_ocp_nonzero[:, max_inlier_mask]
                    y_ocp_mean = xp.mean(y_ocp_nonzero, axis=1)
                    y_ocp_demean = y_ocp_nonzero - y_ocp_mean[:,numpy.newaxis]

                    t_pc_nonzero = t_pc_nonzero[:, max_inlier_mask]
                    t_pc_mean = xp.mean(t_pc_nonzero, axis=1)
                    t_pc_demean = t_pc_nonzero - t_pc_mean[:,numpy.newaxis]

                    ## rotation matrix estimation using SVD method
                    U, S, V = xp.linalg.svd(xp.dot(t_pc_demean, y_ocp_demean.T))

                    # det(U, V)
                    VU = xp.dot(V, U)
                    VU_det = VU[0,0] * VU[1,1] * VU[2,2] + VU[0,1] * VU[1,2] * VU[2,0] +\
                             VU[0,2] * VU[1,0] * VU[2,1] - VU[0,2] * VU[1,1] * VU[2,0] -\
                             VU[0,1] * VU[1,0] * VU[2,2] - VU[0,0] * VU[1,2] * VU[2,1]
                    H = xp.diag(xp.array([1,1,VU_det], dtype=xp.float64))
                    R = xp.dot(xp.dot(U, H), V)
                # print y_cp_mean - t_cp[i]
                # # numpy.save("test_cp.npy", cuda.to_cpu(y_cp_nonzero - t_cp[i][:, numpy.newaxis]))
                # print numpy.median(cuda.to_cpu(y_cp_nonzero - t_cp[i][:, numpy.newaxis]), axis=1)
                # print xp.max(xp.abs((y_cp_nonzero - t_cp[i][:, numpy.newaxis])), axis=1)
                # print xp.sqrt(xp.var(y_cp_nonzero, axis=1))

                # t_ocp_nonzero = t_ocp[i].reshape(3,-1)[:, pred_mask[i].ravel()]
                # # # print len(y_ocp_nonzero[0])
                # ver1 = xp.sqrt(xp.var(y_ocp_nonzero -  t_ocp_nonzero, axis=1))
                # mean1 = xp.mean(y_ocp_nonzero -  t_ocp_nonzero, axis=1)
                # max1 = xp.max(xp.abs(y_ocp_nonzero -  t_ocp_nonzero), axis=1)
                # print "---"
                # print mean1
                # print numpy.median(cuda.to_cpu(y_ocp_nonzero -  t_ocp_nonzero), axis=1)
                # print ver1
                # print max1

        ret_cp = xp.sqrt(xp.sum(xp.square(estimated_cp - t_cp)) / batch_size)
        ret_ocp = xp.sqrt(xp.sum(xp.square(estimated_ocp - t_cp)) / batch_size)

        return xp.asarray(ret_cp, dtype=y_cp.dtype), xp.asarray(ret_ocp, dtype=y_ocp.dtype),


def dual_cp_accuracy(y_cls, y_cp, t_ocp, y_ocp, t_cp, t_pc, cp_mask, ocp_mask, eps=0.2):
    return DualCenterProposalAccuracy(eps=eps, method="SVD")(y_cls, y_cp, t_ocp, y_ocp, t_cp,
                                                             t_pc, cp_mask, ocp_mask)
