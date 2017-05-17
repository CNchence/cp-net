from chainer import Variable
import chainer.functions as F
from chainer import link
from chainer import reporter
import numpy as np
from chainer.functions.array import concat


class CPNetClassifier(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor):
        super(CPNetClassifier, self).__init__(predictor=predictor)
        self.y = None
        self.cls_loss = None
        self.pos_loss = None
        self.loss = None
        self.class_accuracy = None
        self.pose_accuracy = None

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-2]
        t1, t2 = args[-2:]
        self.y = None
        self.loss = None
        self.class_accuracy = None
        self.pose_accuracy = None
        self.y = self.predictor(*x)
        y_cls, y_pos = self.y

        t1 = F.reshape(t1, (t1.shape[0], t1.shape[2], t1.shape[3]))

        self.cls_loss = F.softmax_cross_entropy(y_cls, t1)
        self.pos_loss = F.mean_squared_error(y_pos, t2)
        self.loss = self.cls_loss + self.pos_loss
        reporter.report({'class_loss': self.cls_loss}, self)
        reporter.report({'pose_loss': self.pos_loss}, self)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.class_accuracy = F.accuracy(y_cls, t1)
            self.pose_accuracy = F.mean_squared_error(y_pos, t2)
            reporter.report({'class_accuracy': self.class_accuracy}, self)
            reporter.report({'pose_accuracy': self.pose_accuracy}, self)
        return self.loss
