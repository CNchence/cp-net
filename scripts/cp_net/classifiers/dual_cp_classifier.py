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
        self.loss = None
        self.cls_acc = None
        self.cp_acc = None
        self.ocp_acc = None

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-7]
        t_cls, t_cp, t_ocp, cp, ocp, t_pc, nonnan_mask = args[-7:]
        self.y = None
        self.loss = None
        self.cls_loss = None
        self.cp_loss = None
        self.ocp_loss = None
        self.cls_acc = None
        self.cp_acc = None
        self.ocp_acc = None
        self.y = self.predictor(*x)
        y_cls, y_cp, y_ocp = self.y
        self.cls_loss = F.softmax_cross_entropy(y_cls, t_cls)
        self.cp_loss = mask_mean_squared_error(y_cp, t_cp, nonnan_mask)
        self.ocp_loss = mask_mean_squared_error(y_cp, t_ocp, nonnan_mask)
        self.loss = self.cls_loss + self.cp_loss + self.ocp_loss
        reporter.report({'cls_loss': self.cls_loss}, self)
        reporter.report({'cp_loss': self.cp_loss}, self)
        reporter.report({'ocp_loss': self.ocp_loss}, self)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.class_acc = F.accuracy(y_cls, t_cls)
            self.cp_acc, self.ocp_acc = \
                dual_cp_accuracy.dual_cp_accuracy(y_cls, y_cp, y_ocp, cp, ocp, t_pc, eps=0)
            reporter.report({'cls_acc': self.class_acc}, self)
            reporter.report({'cp_acc': self.cp_acc}, self)
            reporter.report({'ocp_acc': self.ocp_acc}, self)
        return self.loss
