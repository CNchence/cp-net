import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check
from scipy import stats


class DualCenterProposalAccuracy(function.Function):

    ## Support one class per flame

    def __init__(self, eps=0.2):
        self.eps = eps

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y_cls, y_cp, y_ocp, t_cp, t_ocp, t_pc= inputs

        prob = xp.max(y_cls[0], axis=0)
        pred = xp.argmax(y_cls[0], axis=0)
        pred[prob < self.eps] = 0

        # todo use bincount
        cls_count = numpy.array([(pred == i).sum() for i in range(len(y_cls[0]))])
        # print cls_count
        obj_cls = numpy.argmax(cls_count[1:]) + 1

        pred_mask = xp.invert(xp.isnan(t_pc[0]))[0] * (pred == obj_cls)


        if pred_mask.sum() == 0 :
            return xp.asarray(4, dtype=y_cp.dtype), xp.asarray(4, dtype=y_ocp.dtype),
        else:
            prob_weight = (pred_mask * prob) / (pred_mask * prob).sum()
            estimated_cp = xp.sum(xp.sum(prob_weight * (y_cp[0] + t_pc[0]), axis=1),axis=1)
            estimated_ocp = xp.sum(xp.sum(prob_weight * (y_ocp[0] + t_pc[0]), axis=1), axis=1)

        ret_cp = xp.sqrt(xp.sum(xp.square(estimated_cp - t_cp[0])))
        ret_ocp = xp.sqrt(xp.sum(xp.square(estimated_ocp - t_ocp[0])))
        return xp.asarray(ret_pos, dtype=y_pos.dtype), xp.asarray(ret_rot, dtype=y_rot.dtype),


def dual_cp_accuracy(y_cls, y_cp, y_ocp, t_cp, t_ocp, t_pc, eps=0.2):
    return DualCenterProposalAccuracy(eps=eps)(y_cls, y_cp, y_ocp, t_cp, t_ocp, t_pc)
