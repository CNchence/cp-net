
import numpy as np

import chainer.functions as F
from chainer import link
from chainer import reporter

from cp_net.functions import dual_cp_accuracy
from cp_net.functions.mask_mean_squared_error import mask_mean_squared_error

class DualCPNetClassifier(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor):
        super(DualCPNetClassifier, self).__init__(predictor=predictor)
        self.y = None
        self.cls_loss = None
        self.cp_loss = None
        self.ocp_loss = None
        self.cp_mask_loss = None
        self.ocp_mask_loss = None
        self.loss = None
        self.cls_acc = None
        self.cp_acc = None
        self.ocp_acc = None
        self.cp_mask_acc= None
        self.ocp_mask_acc = None

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-7]
        t_cls, t_cp, t_ocp, cp, t_pc, obj_mask, nonnan_mask = args[-7:]
        self.y = None
        self.loss = None
        self.cls_loss = None
        self.cp_loss = None
        self.ocp_loss = None
        self.cp_mask_loss = None
        self.ocp_mask_loss = None
        self.cls_acc = None
        self.cp_acc = None
        self.ocp_acc = None
        self.y = self.predictor(*x)
        y_cls, y_cp, y_ocp, y_cp_mask, y_ocp_mask = self.y
        self.cls_loss = F.softmax_cross_entropy(y_cls, t_cls)
        self.cp_loss = mask_mean_squared_error(y_cp, t_cp, nonnan_mask) * 0.1
        self.cp_loss += mask_mean_squared_error(y_cp, t_cp, (nonnan_mask * obj_mask).astype(np.float32)) * 0.9
        self.ocp_loss = mask_mean_squared_error(y_ocp, t_ocp, nonnan_mask) * 0.1
        self.ocp_loss += mask_mean_squared_error(y_ocp, t_ocp, (nonnan_mask * obj_mask).astype(np.float32)) * 0.9

        self.loss = self.cls_loss + 1e1 * self.cp_loss + 1e2 * self.ocp_loss #+ self.cp_mask_loss + self.ocp_mask_loss
        reporter.report({'l_cls': self.cls_loss}, self)
        reporter.report({'l_cp': self.cp_loss}, self)
        reporter.report({'l_ocp': self.ocp_loss}, self)
        reporter.report({'l_cp_mask': self.cp_mask_loss}, self)
        reporter.report({'l_ocp_mask': self.ocp_mask_loss}, self)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.class_acc = F.accuracy(y_cls, t_cls)
            self.cp_acc, self.ocp_acc = \
                dual_cp_accuracy.dual_cp_accuracy(y_cls, y_cp, t_ocp, y_ocp, cp, t_pc,
                                                  y_cp_mask, y_ocp_mask)
            reporter.report({'cls_acc': self.class_acc}, self)
            reporter.report({'cp_acc': self.cp_acc}, self)
            reporter.report({'ocp_acc': self.ocp_acc}, self)
        return self.loss
