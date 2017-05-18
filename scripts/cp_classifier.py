import chainer.functions as F
from chainer import link
from chainer import reporter


class CPNetClassifier(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor):
        super(CPNetClassifier, self).__init__(predictor=predictor)
        self.y = None
        self.cls_loss = None
        self.pos_loss = None
        self.q_loss = None
        self.loss = None
        self.class_accuracy = None
        self.pose_accuracy = None

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-3]
        t1, t2, t3 = args[-3:]
        self.y = None
        self.loss = None
        self.class_accuracy = None
        self.pose_accuracy = None
        self.y = self.predictor(*x)
        y_cls, y_pos, y_q = self.y

        self.cls_loss = F.softmax_cross_entropy(y_cls, t1)
        self.pos_loss = F.mean_squared_error(y_pos, t2)
        self.q_loss = F.mean_squared_error(y_q, t3)
        self.loss = self.cls_loss + self.pos_loss + self.q_loss
        reporter.report({'class_loss': self.cls_loss}, self)
        reporter.report({'position_loss': self.pos_loss}, self)
        reporter.report({'orientation_loss': self.q_loss}, self)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.class_accuracy = F.accuracy(y_cls, t1)
            self.pose_accuracy = F.mean_squared_error(y_pos, t2)
            reporter.report({'class_accuracy': self.class_accuracy}, self)
            reporter.report({'pose_accuracy': self.pose_accuracy}, self)
        return self.loss
