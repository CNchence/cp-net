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
        super(CenterProposalNetworkRes50FCN, self).__init__(
            # resnet50
            conv1=L.Convolution2D(3, 64, 7, 2, 3, **kwargs),
            bn1=L.BatchNormalization(64),
            res2=R.BuildingBlock(3, 64, 64, 256, 1, **kwargs), # resblock 1/2 -> 1/4
            res3=R.BuildingBlock(4, 256, 128, 512, 2, **kwargs), # resblock 1/4 ->1/8
            res4=R.BuildingBlock(6, 512, 256, 1024, 2, **kwargs), # resblock 1/8 -> 1/16
            res5=R.BuildingBlock(3, 1024, 512, 2048, 2, **kwargs), # resblock 1/16 -> 1/32

            concat_conv = L.Convolution2D(512 + 1024 + 2048,  1024, 3, stride=1, pad=1),

            pool_roi_conv =  L.Convolution2D(1024, 512, 5, stride=5, pad=0),
            conv_after_croip = L.Convolution2D(512, 512, 3, stride=1, pad=1),

            score_pool = L.Convolution2D(512, n_class, 1, stride=1, pad=0),
            upscore_final=L.Deconvolution2D(self.n_class, self.n_class, 16,
                                            stride=8, pad=4, use_cudnn=False),
        )

    def __call__(self, x1, x2, eps=0.001, test=None):
        h = x1
        ksizes = x2 # focus / depth (focus 1.0 simply)

        h = F.relu(self.bn1(self.conv1(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        # Res Blocks
        h = self.res2(h, test=test) # 1/4
        h = self.res3(h, test=test) # 1/8
        pool1_8 = h
        h = self.res4(h, test=test) # 1/16
        pool1_16 = h
        h = self.res5(h, test=test) # 1/32
        pool1_32 = h

        # upscore 1/32 -> 1/8
        h = F.unpooling_2d(h, 4)
        # upscore 1/16 -> 1/8
        pool1_16 = F.unpooling_2d(pool1_16, 2)

        # concat 1 / 8
        h = concat.concat((upscore1, upscore2, pool1_8), axis=1)
        h = F.relu(self.concat_conv(h))

        h = convolutional_roi_pooling(h, F.ceil(ksizes * 3), out_ksize=5)
        h = self.relu(self.bn_croip1(self.pool_roi_conv(h)))

        h = self.relu(self.bn_croip2(self.conv_after_croip(h)))

        # score
        h = F.relu(self.score_pool(h))
        score1_8 = h
        h = F.relu(self.upscore_final(h))
        score = h  # 1/1
        self.score = score  # XXX: for backward compatibility

        return score
