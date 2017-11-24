import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainer.initializers import constant
from chainer.initializers import normal
from chainer.functions.array import concat

import numpy as np
import six

import pspnet

chainer.config.comm = None


class DualCPDRN(chainer.Chain):
    ## we don't train resblock
    ## becuase resblocks is trained enough to extract features, using trained model
    def __init__(self, n_class, pre_trained_model=None, input_size=(640, 480),
                 mid_stride=True, initialW=None, train_res5=True):
        super(DualCPDRN, self).__init__()
        ## voc2012 settings
        n_blocks = [3, 4, 23, 3]
        chainer.config.mid_stride = mid_stride
        mean = np.array([123.68, 116.779, 103.939])

        if initialW is None:
            chainer.config.initialW = chainer.initializers.HeNormal()
        else:
            chainer.config.initialW = initialW

        with self.init_scope():
            self.input_size = input_size
            self.trunk = pspnet.PretrainedDilatedFCN(n_blocks=n_blocks, train_res5=train_res5)
            # Main branch
            self.cbr_main = pspnet.ConvBNReLU(2048, 512, 3, 1, 1)
            self.out_main = L.Convolution2D(
                512, n_class, 1, 1, 0, False, initialW)
            self.cbr_cp = pspnet.ConvBNReLU(2048, 512, 3, 1, 1)
            self.out_cp = L.Convolution2D(
                512, (n_class - 1) * 3, 1, 1, 0, False)
            self.cbr_ocp = pspnet.ConvBNReLU(2048, 512, 3, 1, 1)
            self.out_ocp = L.Convolution2D(
                512, (n_class - 1) * 3, 1, 1, 0, False)

        self.mean = mean
        self.n_class = n_class

    def __call__(self, x):
        h = self.trunk(x)
        h_cp = F.dropout(self.cbr_cp(h), ratio=0.1)
        h_cp = F.tanh(self.out_cp(h_cp))
        h_cp = F.resize_images(h_cp, x.shape[2:])

        h_ocp = F.dropout(self.cbr_ocp(h), ratio=0.1)
        h_ocp = F.tanh(self.out_ocp(h_ocp))
        h_ocp = F.resize_images(h_ocp, x.shape[2:])

        h = F.dropout(self.cbr_main(h), ratio=0.1)
        h = self.out_main(h)
        h = F.resize_images(h, x.shape[2:])

        return h, h_cp, h_ocp
