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
        self.pose_acc = None

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-4]
        t_cls, t_pos, t_rot, t_pc = args[-4:]
        self.y = None
        self.loss = None
        self.class_acc = None
        self.pose_acc = None
        self.y = self.predictor(*x)
        y_cls, y_pos, y_rot = self.y
        self.cls_loss = F.softmax_cross_entropy(y_cls, t_cls)
        self.pos_loss = F.mean_squared_error(y_pos, t_pos)
        self.q_loss = F.mean_squared_error(y_rot, t_rot)
        self.loss = self.cls_loss + self.pos_loss + self.q_loss
        reporter.report({'class_loss': self.cls_loss}, self)
        reporter.report({'pos_loss': self.pos_loss}, self)
        reporter.report({'rot_loss': self.q_loss}, self)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.class_acc = F.accuracy(y_cls, t_cls)
            self.pose_acc = single_class_pose_accuracy.single_class_pose_accuracy(y_cls, y_pos, t_cls,  t_pos, t_pc)
            # self.pose_accuracy = F.mean_squared_error(y_pos, t_pos)
            reporter.report({'class_acc': self.class_acc}, self)
            reporter.report({'pose_acc': self.pose_acc}, self)
        return self.loss
