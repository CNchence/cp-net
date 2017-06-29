import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainer.initializers import constant
from chainer.initializers import normal
import chainer.links.model.vision.resnet as R
from chainer.functions.array import concat

from cp_net.functions.convolutional_roi_pooling import convolutional_roi_pooling

class DepthInvariantNetworkRes50FCN(chainer.Chain):
    def __init__(self, n_class=36, pretrained_model = None):
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            kwargs = {'initialW': constant.Zero()}
        else:
            # employ default initializers used in the original paper
            kwargs = {'initialW': normal.HeNormal(scale=1.0)}
        self.n_class = n_class
        super(DepthInvariantNetworkRes50FCN, self).__init__(
            # resnet50
            conv1=L.Convolution2D(3, 64, 7, 2, 3, **kwargs),
            bn1=L.BatchNormalization(64),
            res2=R.BuildingBlock(3, 64, 64, 256, 1, **kwargs), # resblock 1/2 -> 1/4
            res3=R.BuildingBlock(4, 256, 128, 512, 2, **kwargs), # resblock 1/4 ->1/8
            res4=R.BuildingBlock(6, 512, 256, 1024, 2, **kwargs), # resblock 1/8 -> 1/16
            res5=R.BuildingBlock(3, 1024, 512, 2048, 2, **kwargs), # resblock 1/16 -> 1/32

            upscore1=L.Deconvolution2D(2048, 512, 16, stride=8, pad=4),
            upscore2=L.Deconvolution2D(1024, 512, 8,stride=4, pad=2),
            upscore3=L.Deconvolution2D(512, 512, 4, stride=2, pad=1),

            bn_upscore = L.BatchNormalization(512 * 3),

            concat_conv = L.Convolution2D(512*3,  1024, 3, stride=1, pad=1),

            pool_roi_conv =  L.Convolution2D(1024, 1024, 5, stride=5, pad=0),
            conv_after_croip = L.Convolution2D(1024, 512, 3, stride=1, pad=1),

            bn_croip1 = L.BatchNormalization(1024),
            bn_croip2 = L.BatchNormalization(512),
            
            score_pool = L.Convolution2D(512, n_class, 1, stride=1, pad=0),
            upscore_final=L.Deconvolution2D(self.n_class, self.n_class, 8,
                                            stride=4, pad=2),
        )

    def __call__(self, x1, x2, eps=0.001):
        h = x1
        ksizes = x2

        h = F.relu(self.bn1(self.conv1(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        # Res Blocks
        h = self.res2(h) # 1/4
        h = self.res3(h) # 1/8
        pool1_8 = h
        h = self.res4(h) # 1/16
        pool1_16 = h
        h = self.res5(h) # 1/32
        pool1_32 = h

        # upscore 1/32 -> 1/4
        pool1_32 = self.upscore1(pool1_32)
        # upscore 1/16 -> 1/4
        pool1_16 = self.upscore2(pool1_16)
        # upscore 1/8 -> 1/4
        pool1_8 = self.upscore3(pool1_8)

        # concat 1 / 4
        h = F.leaky_relu(self.bn_upscore(concat.concat((pool1_32, pool1_16, pool1_8), axis=1)))
        h = F.relu(self.concat_conv(h))

        ksizes = F.ceil(F.resize_images((ksizes * 3), (h.data.shape[2], h.data.shape[3])))
        h = convolutional_roi_pooling(h, ksizes, out_ksize=5)
        h = F.relu(self.bn_croip1(self.pool_roi_conv(h)))

        h = F.relu(self.bn_croip2(self.conv_after_croip(h)))

        # score
        h = F.relu(self.score_pool(h))
        h = F.relu(self.upscore_final(h))
        score = h  # 1/1

        return score
