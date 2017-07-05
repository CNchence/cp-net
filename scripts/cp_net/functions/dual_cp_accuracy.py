import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class DualCenterProposalAccuracy(function.Function):

    ## Support one class per flame

    def __init__(self, eps=0.2):
        self.eps = eps

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y_cls, y_cp, y_ocp, t_cp, t_pc, nonnan_mask = inputs
        batch_size, n_class = y_cls.shape[:2]

        prob = xp.max(y_cls, axis=1)
        pred = xp.argmax(y_cls, axis=1)

        # threshold
        pred[prob < self.eps] = 0

        cls_count = xp.empty(y_cls.shape[:2])
        for i in range(batch_size):
            cls_count[i] = xp.bincount(pred[i].ravel(), minlength= n_class)

        obj_cls = numpy.argmax(cls_count[:,1:], axis=1) + 1

        pred_mask = xp.invert(xp.isnan(t_pc[:,0,:])) * (pred == obj_cls[:, numpy.newaxis, numpy.newaxis])
        # pred_mask = nonnan_mask.astype(numpy.bool)

        t_pc[t_pc != t_pc] = 0

        estimated_cp = xp.empty((batch_size, 3))
        estimated_ocp = xp.empty((batch_size, 3))
        penalty = xp.array([20, 20, 20])

        for i in range(batch_size):
            if pred_mask[i].sum() == 0 :
                estimated_cp[i] = penalty
                estimated_ocp[i] = penalty
            else:
                prob_weight = (pred_mask[i] * prob[i]) / xp.sum(pred_mask[i] * prob[i])
                estimated_cp[i] = xp.sum((prob_weight * (y_cp[i] + t_pc[i])).reshape(3, -1), axis=1)

                y_ocp_nonzero = y_ocp[i].reshape(3,-1)[:, pred_mask[i].ravel()]
                y_ocp_mean = xp.mean(y_ocp_nonzero, axis=1)
                # y_ocp_mean = xp.mean(y_ocp[i].reshape(3,-1)[:, pred_mask.ravel()], axis=1)
                y_ocp_demean = y_ocp_nonzero - y_ocp_mean[:,numpy.newaxis]

                t_pc_nonzero = t_pc[i].reshape(3,-1)[:, pred_mask[i].ravel()]
                t_pc_mean = xp.mean(t_pc_nonzero, axis=1)
                t_pc_demean = t_pc_nonzero - t_pc_mean[:,numpy.newaxis]

                U, S, V = xp.linalg.svd(xp.dot(t_pc_demean, y_ocp_demean.T))

                # det(U, V)
                VU = xp.dot(V, U)
                VU_det = VU[0,0] * VU[1,1] * VU[2,2] + VU[0,1] * VU[1,2] * VU[2,0] +\
                         VU[0,2] * VU[1,0] * VU[2,1] - VU[0,2] * VU[1,1] * VU[2,0] -\
                         VU[0,1] * VU[1,0] * VU[2,2] - VU[0,0] * VU[1,2] * VU[2,1]
                H = xp.diag(xp.array([1,1,VU_det], dtype=xp.float64))
                R = xp.dot(xp.dot(U, H), V)

                estimated_ocp[i] = t_pc_mean - xp.dot(R, y_ocp_mean.T)


        ret_cp = xp.sqrt(xp.sum(xp.square(estimated_cp - t_cp)) / batch_size)
        ret_ocp = xp.sqrt(xp.sum(xp.square(estimated_ocp - t_cp)) / batch_size)

        return xp.asarray(ret_cp, dtype=y_cp.dtype), xp.asarray(ret_ocp, dtype=y_ocp.dtype),


def dual_cp_accuracy(y_cls, y_cp, y_ocp, t_cp, t_pc, nonnan_mask, eps=0.2):
    return DualCenterProposalAccuracy(eps=eps)(y_cls, y_cp, y_ocp, t_cp, t_pc, nonnan_mask)
