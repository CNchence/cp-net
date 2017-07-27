
import numpy as np

import chainer.functions as F
from chainer import link
from chainer import reporter

from cp_net.functions import dual_cp_accuracy
from cp_net.functions.mask_mean_squared_error import mask_mean_squared_error

class DualCPNetClassifier(link.Chain):

    def __init__(self, predictor, distance_sanity=0.1, method="RANSAC",
                 ver2=False, compute_accuracy = True):
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
        self.ignore_label = -1
        self.lambda1 = 1e1
        self.lambda2 = 1e2
        self.distance_sanity = distance_sanity
        self.method = method
        self.ver2 = ver2
        self.compute_accuracy = compute_accuracy

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-8]
        t_cls, t_cp, t_ocp, cp, rot, t_pc, obj_mask, nonnan_mask = args[-8:]
        self.y = None
        self.loss = None
        self.cls_loss = None
        self.cp_loss = None
        self.ocp_loss = None
        self.cls_acc = None
        self.cp_acc = None
        self.ocp_acc = None
        self.rot_acc = None
        self.y = self.predictor(*x)
        y_cls, y_cp, y_ocp = self.y
        self.cls_loss = F.softmax_cross_entropy(y_cls, t_cls)
        self.cp_loss = mask_mean_squared_error(
            y_cp.reshape(t_cp.shape), t_cp, nonnan_mask) * 0.1
        self.cp_loss += mask_mean_squared_error(
            y_cp.reshape(t_cp.shape), t_cp, obj_mask.astype(np.float32)) * 0.9
        self.ocp_loss = mask_mean_squared_error(
            y_ocp.reshape(t_ocp.shape), t_ocp, nonnan_mask) * 0.1
        self.ocp_loss += mask_mean_squared_error(
            y_ocp.reshape(t_ocp.shape),t_ocp, obj_mask.astype(np.float32)) * 0.9
        self.loss = self.cls_loss + self.lambda1 * self.cp_loss + self.lambda2 * self.ocp_loss
        reporter.report({'l_cls': self.cls_loss}, self)
        reporter.report({'l_cp': self.cp_loss}, self)
        reporter.report({'l_ocp': self.ocp_loss}, self)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.class_acc = F.accuracy(y_cls, t_cls, ignore_label=self.ignore_label)
            self.cp_acc, self.ocp_acc, self.rot_acc= \
                dual_cp_accuracy.dual_cp_accuracy(y_cls, y_cp, y_ocp, cp, rot, t_pc,
                                                  eps=0.5,
                                                  distance_sanity=self.distance_sanity,
                                                  method=self.method,
                                                  ver2=self.ver2)
            reporter.report({'cls_acc': self.class_acc}, self)
            reporter.report({'cp_acc': self.cp_acc}, self)
            reporter.report({'ocp_acc': self.ocp_acc}, self)
            reporter.report({'rot_acc': self.rot_acc}, self)
        return self.loss
