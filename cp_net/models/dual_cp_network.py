import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainer.initializers import constant
from chainer.initializers import normal
import chainer.links.model.vision.resnet as R
from chainer.functions.array import concat

class DualCenterProposalNetworkRes50FCN(chainer.Chain):
    def __init__(self, n_class=36, pretrained_model = None, output_scale=1.0):
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            kwargs = {'initialW': constant.Zero()}
        else:
            print "train resblock : True"
            # employ default initializers used in the original paper
            kwargs = {'initialW': normal.HeNormal(scale=1.0)}
        self.n_class = n_class
        self.output_scale = output_scale
        super(DualCenterProposalNetworkRes50FCN, self).__init__(
            # resnet50
            conv1=L.Convolution2D(3, 64, 7, 2, 3, **kwargs),
            bn1=L.BatchNormalization(64),
            res2=R.BuildingBlock(3, 64, 64, 256, 1, **kwargs),
            res3=R.BuildingBlock(4, 256, 128, 512, 2, **kwargs),
            res4=R.BuildingBlock(6, 512, 256, 1024, 2, **kwargs),
            res5=R.BuildingBlock(3, 1024, 512, 2048, 2, **kwargs),

            upscore32=L.Deconvolution2D(2048, 512, 8, stride=4, pad=2),
            bn_up32 = L.BatchNormalization(512),
            upscore16=L.Deconvolution2D(1024, 512, 4, stride=2, pad=1),
            bn_up16 = L.BatchNormalization(512),

            concat_conv = L.Convolution2D(512 * 3, 512 * 3, 3, stride=1, pad=1),
            bn_concat = L.BatchNormalization(512 * 3),

            score_pool = L.Convolution2D(512 * 3, n_class, 1, stride=1, pad=0),

            upscore_final=L.Deconvolution2D(self.n_class, self.n_class, 16, stride=8, pad=4),

            conv_cp1 = L.Convolution2D(512 * 3, 1024, 3, stride=1, pad=1),
            bn_cp1 = L.BatchNormalization(1024),

            # center pose network
            conv_cp2 = L.Convolution2D(1024, 512, 3, stride=1, pad=1),
            bn_cp2 = L.BatchNormalization(512),
            upscore_cp1 = L.Deconvolution2D(512, 16, 8, stride=4, pad=2),
            bn_cp3 = L.BatchNormalization(16),
            upscore_cp2 = L.Deconvolution2D(16, 3, 4, stride=2, pad=1),

            # origin center pose network
            conv_ocp2 = L.Convolution2D(1024, 512, 3, stride=1, pad=1),
            bn_ocp2 = L.BatchNormalization(512),
            upscore_ocp1 = L.Deconvolution2D(512, 16, 8, stride=4, pad=2),
            bn_ocp3 = L.BatchNormalization(16),
            upscore_ocp2 = L.Deconvolution2D(16, 3, 4, stride=2, pad=1),
        )

    def __call__(self, x1, eps=0.001):
        h = x1
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        # Res Blocks
        h = self.res2(h) # 1/4
        h = self.res3(h) # 1/8
        pool1_8 = h
        h = self.res4(h) # 1/16
        pool1_16 = h
        h = self.res5(h) # 1/32

        # upscore 1/32 -> 1/8
        h = F.elu(self.bn_up32(self.upscore32(h)))
        upscore1 = h  # 1/8
        # upscore 1/16 -> 1/8
        upscore2 = F.elu(self.bn_up16(self.upscore16(pool1_16)))

        # concat conv
        h = concat.concat((upscore1, upscore2, pool1_8), axis=1)
        h = F.relu(self.bn_concat(self.concat_conv(h)))
        concat_pool = h

        # score
        h = F.elu(self.score_pool(concat_pool))
        score1_8 = h
        h = F.relu(self.upscore_final(h))
        score = h  # 1/1

        h_cp = F.relu(self.bn_cp1(self.conv_cp1(concat_pool)))
        h_ocp = h_cp

        h_cp = F.relu(self.bn_cp2(self.conv_cp2(h_cp)))
        h_cp = F.elu(self.bn_cp3(self.upscore_cp1(h_cp)))
        h_cp = self.upscore_cp2(h_cp)

        h_ocp = F.relu(self.bn_ocp2(self.conv_ocp2(h_ocp)))
        h_ocp = F.elu(self.bn_ocp3(self.upscore_ocp1(h_ocp)))
        h_ocp = self.upscore_ocp2(h_ocp)

        cp_score = F.tanh(h_cp) * self.output_scale
        ocp_score = F.tanh(h_ocp) * self.output_scale

        return score, cp_score, ocp_score
