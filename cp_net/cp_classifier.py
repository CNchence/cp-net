import chainer.functions as F
from chainer import link
from chainer import reporter

from functions import single_class_pose_accuracy

class CPNetClassifier(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor):
        super(CPNetClassifier, self).__init__(predictor=predictor)
        self.y = None
        self.cls_loss = None
        self.pos_loss = None
        self.q_loss = None
        self.loss = None
        self.class_acc = None
        self.pos_acc = None
        self.rot_acc = None

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-6]
        t_cls, t_dist, t_pos, t_rot, t_rot_map, t_pc = args[-6:]
        self.y = None
        self.loss = None
        self.class_acc = None
        self.pos_acc = None
        self.rot_acc = None
        self.y = self.predictor(*x)
        y_cls, y_pos, y_rot = self.y
        self.cls_loss = F.softmax_cross_entropy(y_cls, t_cls)
        self.pos_loss = F.mean_squared_error(y_pos, t_dist)
        self.q_loss = F.mean_squared_error(y_rot, t_rot_map)
        self.loss = self.cls_loss + self.pos_loss + self.q_loss
        reporter.report({'c_loss': self.cls_loss}, self)
        reporter.report({'p_loss': self.pos_loss}, self)
        reporter.report({'r_loss': self.q_loss}, self)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.class_acc = F.accuracy(y_cls, t_cls)
            self.pos_acc, self.rot_acc = single_class_pose_accuracy.single_class_pose_accuracy(y_cls, y_pos, y_rot, t_pos, t_rot, t_pc, eps=0)
            reporter.report({'c_acc': self.class_acc}, self)
            reporter.report({'p_acc': self.pos_acc}, self)
            reporter.report({'r_acc': self.rot_acc}, self)
        return self.loss
