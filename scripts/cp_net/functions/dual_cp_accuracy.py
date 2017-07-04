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

        prob = xp.max(y_cls[0], axis=0)
        pred = xp.argmax(y_cls[0], axis=0)
        pred[prob < self.eps] = 0

        # todo use bincount
        cls_count = numpy.array([(pred == i).sum() for i in range(len(y_cls[0]))])
        # print cls_count
        obj_cls = numpy.argmax(cls_count[1:]) + 1

        pred_mask = xp.invert(xp.isnan(t_pc[0,0,:])) * (pred == obj_cls)
        pred_mask = nonnan_mask.astype(numpy.bool)
        t_pc[t_pc != t_pc] = 0

        if pred_mask.sum() == 0 :
            return xp.asarray(10, dtype=y_cp.dtype), xp.asarray(10, dtype=y_ocp.dtype),
        else:
            prob_weight = (pred_mask * prob) / xp.sum(pred_mask * prob)
            estimated_cp = xp.sum((prob_weight * (y_cp[0] + t_pc[0])).reshape(3, -1), axis=1)

            # rr = xp.array([[1 ,0, 0],
            #               [0, 0,-1],
            #               [0, 1, 0]])
            # y_ocp = xp.dot(rr, t_pc.reshape(3,-1)).reshape(t_pc.shape)

            y_ocp_nonzero = y_ocp[0].reshape(3,-1)[:, pred_mask.ravel()]
            y_ocp_mean = xp.mean(y_ocp_nonzero, axis=1)
            # y_ocp_mean = xp.mean(y_ocp[0].reshape(3,-1)[:, pred_mask.ravel()], axis=1)
            y_ocp_demean = y_ocp_nonzero - y_ocp_mean[:,numpy.newaxis]

            t_pc_nonzero = t_pc[0].reshape(3,-1)[:, pred_mask.ravel()]
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

            estimated_ocp = t_pc_mean - xp.dot(R, y_ocp_mean.T)

            print "----"
            t_ocp_nonzero = y_cp[0].reshape(3,-1)[:, pred_mask.ravel()]
            # print len(y_ocp_nonzero[0])
            ver1 = xp.sqrt(xp.var(y_ocp_nonzero -  t_ocp_nonzero, axis=1))
            mean1 = xp.mean(y_ocp_nonzero -  t_ocp_nonzero, axis=1)
            print len(y_ocp_nonzero[0])
            print ver1
            print xp.max(xp.abs(y_ocp_nonzero -  t_ocp_nonzero), axis=1)
            print xp.min(xp.abs(y_ocp_nonzero -  t_ocp_nonzero), axis=1)
            print xp.mean(y_ocp_nonzero -  t_ocp_nonzero, axis=1)
            # print numpy.median(cuda.to_cpu(y_ocp_nonzero -  t_ocp_nonzero), axis=1)
            # idx1 = xp.where(y_ocp_nonzero -  t_ocp_nonzero + y_ocp > ver1)0
            # print len(numpy.unique(cuda.to_cpu(xp.where(xp.square((y_ocp_nonzero -  t_ocp_nonzero - mean1[:,numpy.newaxis])).transpose(1,0) > ver1)[0])))
            ver2 = xp.var(t_pc_demean - xp.dot(R, y_ocp_demean), axis=1)
            mean2 = xp.mean(t_pc_demean - xp.dot(R, y_ocp_demean), axis=1)
            # print numpy.median(cuda.to_cpu(t_pc_demean - xp.dot(R, y_ocp_demean)), axis=1)
            outlier = (xp.square(t_pc_demean - xp.dot(R, y_ocp_demean)) < ver2[:,numpy.newaxis] * 2)
            out_mask = outlier[0] * outlier[1] * outlier[2]
            # print ver2
            # print mean2
            mean3 = xp.mean(y_ocp_nonzero[:, out_mask] -  t_ocp_nonzero[:, out_mask], axis=1)
            # print mean3

            # print len(xp.where(xp.sum(xp.square(xp.dot(R, y_ocp_nonzero) - t_pc_nonzero), axis=0) > ver2)[0])
            # print xp.max(y_ocp[0].reshape(3,-1)[:, pred_mask.ravel()] -  y_cp[0].reshape(3,-1)[:, pred_mask.ravel()], axis=1)
            # print xp.min(y_ocp[0].reshape(3,-1)[:, pred_mask.ravel()] -  y_cp[0].reshape(3,-1)[:, pred_mask.ravel()], axis=1)
            # print R
            # print "----"
            # print rr
            # print y_ocp_mean
            # print xp.sqrt(xp.sum(xp.square(y_ocp_mean - xp.mean(y_cp[0].reshape(3,-1)[:, pred_mask.ravel()], axis=1))))
            # print xp.abs(y_ocp_mean - xp.mean(y_cp[0].reshape(3,-1)[:, pred_mask.ravel()], axis=1))
            # y_cp_nonzero = y_cp[0].reshape(3,-1)[:, pred_mask.ravel()]
            # print xp.mean(y_cp_nonzero, axis=1)
            # print xp.max(t_pc_demean.reshape(3,-1) - xp.dot(R, y_ocp_demean.reshape(3,-1)), axis=1)
            # print xp.min(t_pc_demean.reshape(3,-1) - xp.dot(R, y_ocp_demean.reshape(3,-1)), axis=1)
            # print xp.mean(t_pc_demean.reshape(3,-1) - xp.dot(R, y_ocp_demean.reshape(3,-1)), axis=1)
            # print t_cp[0]
            # print t_pc_mean
            # print estimated_ocp

            # test = ((y_cp[0] + t_pc[0]) * pred_mask).reshape(3,-1)
            # numpy.save("prob.npy", cuda.to_cpu(prob))
            # numpy.save("cp.npy", cuda.to_cpu(t_cp))
            # numpy.save("test.npy", cuda.to_cpu(test))

        # print "-------"
        # print numpy.max(((y_cp[0] + t_pc[0]) * pred_mask).reshape(3,-1), axis=1)
        # print numpy.min(((y_cp[0] + t_pc[0]) * pred_mask).reshape(3,-1), axis=1)
        # print estimated_cp
        # print estimated_ocp
        # print t_cp[0]

        ret_cp = xp.sqrt(xp.sum(xp.square(estimated_cp - t_cp[0])))
        ret_ocp = xp.sqrt(xp.sum(xp.square(estimated_ocp - t_cp[0])))
        return xp.asarray(ret_cp, dtype=y_cp.dtype), xp.asarray(ret_ocp, dtype=y_ocp.dtype),


def dual_cp_accuracy(y_cls, y_cp, y_ocp, t_cp, t_pc, nonnan_mask, eps=0.2):
    return DualCenterProposalAccuracy(eps=eps)(y_cls, y_cp, y_ocp, t_cp, t_pc, nonnan_mask)
