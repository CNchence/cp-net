from chainer import Variable
import chainer.functions as F
from chainer import link
from chainer import reporter
import numpy as np
from chainer.functions.array import concat

def class_pose_multi_loss(y, t):
    cls, pos = y
    t_cls_tmp, t_pos_tmp = F.separate(t, 1)
    t_cls = Variable(np.array([]).astype(np.float32))
    t_pos = Variable(np.array([]).astype(np.float32))
    for i in range(len(t_cls_tmp)):
        t_cls = concat.concat((t_cls, F.flatten(t_cls_tmp[i].data.astype(np.float32))), 0)
        t_pos = concat.concat((t_pos, F.flatten(t_pos_tmp[i].data.astype(np.float32))), 0)
    t_cls = F.reshape(F.flatten(t_cls), (cls.shape[0], cls.shape[2], cls.shape[3]))
    t_pos = F.reshape(F.flatten(t_pos), pos.shape)
    t_cls = Variable(t_cls.data.astype(np.int32))
    l_pos = F.mean_squared_error(pos, t_pos)
    l_cls = Variable(np.array([],dtype=np.float32))
    l_cls = F.softmax_cross_entropy(cls, t_cls)
    return  l_cls + l_pos

def class_pose_multi_acc(y, t):
    cls, pos = y
    t_cls_tmp, t_pos_tmp = F.separate(t, 1)
    t_cls = Variable(np.array([]).astype(np.float32))
    t_pos = Variable(np.array([]).astype(np.float32))
    for i in range(len(t_cls_tmp)):
        t_cls = concat.concat((t_cls, F.flatten(t_cls_tmp[i].data.astype(np.float32))), 0)
        t_pos = concat.concat((t_pos, F.flatten(t_pos_tmp[i].data.astype(np.float32))), 0)
    t_cls = F.reshape(F.flatten(t_cls), (cls.shape[0], cls.shape[2], cls.shape[3]))
    t_pos = F.reshape(F.flatten(t_pos), pos.shape)
    t_cls = Variable(t_cls.data.astype(np.int32))
    acc_pos = F.mean_squared_error(pos, t_pos)
    l_cls = Variable(np.array([],dtype=np.float32))
    acc_cls = F.accuracy(cls, t_cls)
    return  acc_cls, acc_pos


class CPNetClassifier(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=class_pose_multi_loss,
                 accfun=class_pose_multi_acc):
        super(CPNetClassifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, *args):

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.class_accuracy = None
        self.pose_accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.class_accuracy, self.pose_accuracy = self.accfun(self.y, t)
            reporter.report({'class_accuracy': self.class_accuracy}, self)
            reporter.report({'pose_accuracy': self.pose_accuracy}, self)
        return self.loss
