import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check
from scipy import stats


def trim2d_idx(arr,xp,rate=0.2):
    if xp.__name__ == 'numpy':
        np_arr = arr
    else:
        np_arr = cuda.to_cpu(arr)
    trim_idx = numpy.tile(True, len(np_arr))
    n_trim = int(len(np_arr) * rate)
    mse_arr = numpy.square(np_arr - numpy.median(np_arr, axis=0)).sum(axis=1)
    mse_order = mse_arr.argsort()[::1][:n_trim]
    trim_idx[mse_order] = False
    return trim_idx

class SingleClassPoseAccuracy(function.Function):
    def __init__(self, eps=0.2, use_trim=True):
        self.eps = eps
        self.use_trim = use_trim
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y_cls, y_pos, y_rot, t_pos, t_rot, t_pc = inputs

        prob = xp.max(y_cls[0], axis=0)
        pred = xp.argmax(y_cls[0], axis=0)
        pred[prob < self.eps] = 0

        cls_count = numpy.array([(pred == i).sum() for i in range(len(y_cls[0]))])
        # print cls_count
        obj_cls = numpy.argmax(cls_count[1:]) + 1

        pred_mask = xp.invert(xp.isnan(t_pc[0]))[0] * (pred == obj_cls)

        t_pc[0][xp.isnan(t_pc[0])] = 0
   
        if pred_mask.sum() == 0 :
            return xp.asarray(4, dtype=y_pos.dtype), xp.asarray(4, dtype=y_rot.dtype),
        else:
            if self.use_trim:
                pred_mask = pred_mask.flatten()
                pos_list = (y_pos[0] + t_pc[0]).transpose(1,2,0).reshape(y_pos[0].size/3, 3)[pred_mask]
                rot_list = y_rot[0].transpose(1,2,0).reshape(y_rot[0].size/5, 5)[pred_mask]
                prob_list = prob.flatten()[pred_mask]

                # median trim mean
                trim_idx_cpu = trim2d_idx(pos_list, xp) * trim2d_idx(rot_list, xp)
                trim_idx = xp.array(trim_idx_cpu)

                prob_weight = prob_list[trim_idx] / prob_list[trim_idx].sum()
                estimated_pos = xp.sum(prob_weight[:, xp.newaxis] * pos_list[trim_idx], axis=0)
                estimated_rot = xp.sum(prob_weight[:, xp.newaxis] * rot_list[trim_idx], axis=0)
            else:
                prob_weight = (pred_mask * prob) / (pred_mask * prob).sum()
                estimated_pos = xp.sum(xp.sum(prob_weight * (y_pos[0] + t_pc[0]), axis=1),axis=1)
                estimated_rot = xp.sum(xp.sum(prob_weight * y_rot[0], axis=1), axis=1)

            # print estimated_pos
            # print t_pos
            ret_pos = xp.sqrt(xp.square(estimated_pos - t_pos[0]).sum())
            ret_rot = xp.sqrt(xp.square(estimated_rot - t_rot[0]).sum())

            return xp.asarray(ret_pos, dtype=y_pos.dtype), xp.asarray(ret_rot, dtype=y_rot.dtype),

def single_class_pose_accuracy(y_cls, y_pos, y_rot, t_pos, t_rot, t_pc, eps=0.2, use_trim=True):
    return SingleClassPoseAccuracy(eps=eps, use_trim=use_trim)(y_cls, y_pos, y_rot, t_pos, t_rot, t_pc)
