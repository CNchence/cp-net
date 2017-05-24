import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class SingleClassPoseAccuracy(function.Function):
    def __init__(self, eps=0.2):
        self.eps = eps

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y_cls, y_pos,  t_pos, t_pc = inputs

        y_cls = y_cls[0]
        y_pos = y_pos[0]
        t_pos = t_pos[0]
        t_pc = t_pc[0]
        prob = xp.max(y_cls, axis=0)
        pred = xp.argmax(y_cls, axis=0)

        pred[prob < self.eps] = 0
        prob[prob < self.eps] = 0

        cls_count = xp.array([len(xp.where(pred == i)[0]) for i in range(len(y_cls))])
        # print cls_count
        obj_cls = xp.argmax(cls_count[:1]) + 1

        pc_mask = xp.invert(xp.isnan(t_pc))[0]
        t_pc[xp.isnan(t_pc)] = 0

        pred_mask = (pred == obj_cls) * pc_mask
        pred_cnt = pred_mask.sum()

        if pred_cnt == 0 :
            return xp.asarray(xp.nan, dtype=y_pos.dtype),
        else:

            prob_weight = (pred_mask * prob) / (pred_mask * prob).sum()
            estimated_pos =  xp.sum(xp.sum(prob_weight * (y_pos + t_pc), axis=1),axis=1)

            # print estimated_pos
            # print t_pos

            ret = xp.sqrt(xp.square(estimated_pos - t_pos).sum())
            return xp.asarray(ret, dtype=y_pos.dtype),

def single_class_pose_accuracy(y_cls, y_pos, t_pos, t_pc, eps=0.2):
    return SingleClassPoseAccuracy(eps=eps)(y_cls, y_pos, t_pos, t_pc)
