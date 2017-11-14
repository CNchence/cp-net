import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainer.initializers import constant
from chainer.initializers import normal
import chainer.links.model.vision.resnet as R
from chainer.functions.array import concat

import pspnet



class DualCPNetworkPSPNetBase(chainer.Chain):
    ## we don't train resblock
    ## becuase resblocks is trained enough to extract features, using trained model
    def __init__(self, n_class, pre_trained_model=None):
        super(DualCPNetworkPSPNetBase, self).__init__()

        chainer.config.mid_stride = mid_stride

        if initialW is None:
            chainer.config.initialW = chainer.initializers.HeNormal()
        else:
            chainer.config.initialW = initialW

        with self.init_scope():
            self.input_size = input_size
            self.trunk = pspnet.DilatedFCN(n_blocks=n_blocks)
            # To calculate auxirally loss
            if chainer.config.train:
                self.cbr_aux = pspnet.ConvBNReLU(None, 512, 3, 1, 1)
                self.out_aux = L.Convolution2D(
                    512, n_class, 3, 1, 1, False, initialW)

            # Main branch
            feat_size = (input_size[0] // 8, input_size[1] // 8)
            self.ppm = pspnet.PyramidPoolingModule(2048, feat_size, pyramids)
            self.cbr_main = pspnet.ConvBNReLU(4096, 512, 3, 1, 1)
            self.out_main = L.Convolution2D(
                512, n_class, 1, 1, 0, False, initialW)
            self.cbr_cp = pspnet.ConvBNReLU(4096, 512, 3, 1, 1)
            self.out_cp = L.Convolution2D(
                512, (n_class - 1) * 3, 1, 1, 0, False, initialW)
            self.cbr_ocp = pspnet.ConvBNReLU(4096, 512, 3, 1, 1)
            self.out_ocp = L.Convolution2D(
                512, (n_class - 1) * 3, 1, 1, 0, False, initialW)

        self.mean = mean
        self.n_class = n_class


    def __call__(self, x):
        if chainer.config.train:
            aux, h = self.trunk(x)
            aux = F.dropout(self.cbr_aux(aux), ratio=0.1)
            aux = self.out_aux(aux)
            aux = F.resize_images(aux, x.shape[2:])
        else:
            h = self.trunk(x)

        h = self.ppm(h)
        h_cls = F.dropout(self.cbr_main(h), ratio=0.1)
        h_cls = self.out_main(h)
        h_cls = F.resize_images(h, x.shape[2:])

        h_cp = F.dropout(self.cbr_cp(h), ratio=0.1)
        h_cp = self.out_main(h_cp)
        h_cp = F.resize_images(h_cp, x.shape[2:])

        h_ocp = F.dropout(self.cbr_ocp(h), ratio=0.1)
        h_ocp = self.out_main(h_ocp)
        h_ocp = F.resize_images(h_ocp, x.shape[2:])

        if chainer.config.train:
            return aux, h_cls, h_cp, h_ocp
        else:
            return h_cls, h_cp, h_ocp
