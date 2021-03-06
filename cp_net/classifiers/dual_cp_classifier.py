import numpy as np

import chainer.functions as F
from chainer import link
from chainer import reporter

from cp_net.functions.mask_mean_squared_error import mask_mean_squared_error

class DualCPNetClassifier(link.Chain):

    def __init__(self, predictor, distance_sanity=0.1, method="RANSAC",
                 basepath='OcclusionChallengeICCV2015',
                 im_size=(640, 480),
                 output_scale=1.0,
                 compute_class_accuracy = True, compute_pose_accuracy = True):
        super(DualCPNetClassifier, self).__init__(predictor=predictor)
        self.y = None
        self.cls_loss = None
        self.cp_loss = None
        self.ocp_loss = None
        self.loss = None
        self.cls_acc = None
        self.cp_acc = None
        self.ocp_acc = None
        self.rot_acc = None
        self.eval_rate = None
        self.ignore_label = -1
        self.lambda1 = 1e1
        self.lambda2 = 1e1
        self.distance_sanity = distance_sanity
        self.method = method
        self.compute_class_accuracy = compute_class_accuracy
        self.compute_pose_accuracy = compute_pose_accuracy

        self.output_scale = output_scale

        self.accfun = None
        if compute_pose_accuracy:
            # from cp_net.functions.old.model_base_consensus_accuracy import ModelBaseConsensusAccuracy
            from cp_net.functions.model_base_consensus_accuracy import ModelBaseConsensusAccuracy
            self.accfun = ModelBaseConsensusAccuracy(eps=0.6,
                                                     distance_sanity=self.distance_sanity,
                                                     method=self.method,
                                                     im_size=im_size,
                                                     base_path=basepath)

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-10]
        t_cls, depth, t_cp, t_ocp, cp, rot, t_pc, obj_mask, nonnan_mask, K = args[-10:]
        self.y = None
        self.loss = None
        self.cls_loss = None
        self.cp_loss = None
        self.ocp_loss = None
        self.cls_acc = None
        self.cp_acc = None
        self.ocp_acc = None
        self.rot_acc = None
        self.eval_rate = None
        self.y = self.predictor(*x)
        y_cls, y_cp, y_ocp = self.y
        self.cls_loss = F.softmax_cross_entropy(y_cls, t_cls)
        self.cp_loss = mask_mean_squared_error(
            y_cp.reshape(t_cp.shape), t_cp / self.output_scale, nonnan_mask) * 0.1
        self.cp_loss += mask_mean_squared_error(
            y_cp.reshape(t_cp.shape), t_cp / self.output_scale, obj_mask.astype(np.float32)) * 0.9
        self.ocp_loss = mask_mean_squared_error(
            y_ocp.reshape(t_ocp.shape), t_ocp / self.output_scale, nonnan_mask) * 0.1
        self.ocp_loss += mask_mean_squared_error(
            y_ocp.reshape(t_ocp.shape), t_ocp / self.output_scale, obj_mask.astype(np.float32)) * 0.9
        self.loss = self.cls_loss  + self.lambda1 * self.cp_loss + self.lambda2 * self.ocp_loss
        reporter.report({'l_cls': self.cls_loss}, self)
        reporter.report({'l_cp': self.cp_loss}, self)
        reporter.report({'l_ocp': self.ocp_loss}, self)
        reporter.report({'loss': self.loss}, self)
        if self.compute_class_accuracy:
            self.class_acc = F.accuracy(y_cls, t_cls, ignore_label=self.ignore_label)
            reporter.report({'cls_acc': self.class_acc}, self)
        if self.compute_pose_accuracy:
            self.cp_acc, self.ocp_acc, self.rot_acc, self.eval_rate= self.accfun(y_cls, y_cp * self.output_scale, y_ocp * self.output_scale,
                                                                                 cp, rot, t_pc, depth, K, args[0])
            reporter.report({'cp_acc': self.cp_acc}, self)
            reporter.report({'ocp_acc': self.ocp_acc}, self)
            reporter.report({'rot_acc': self.rot_acc}, self)
            reporter.report({'5cm5deg': self.eval_rate}, self)
        return self.loss
