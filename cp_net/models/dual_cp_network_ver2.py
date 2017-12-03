import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainer.initializers import constant
from chainer.initializers import normal
import chainer.links.model.vision.resnet as R
from chainer.functions.array import concat

class DualCenterProposalNetworkRes50_predict7(chainer.Chain):
    def __init__(self, n_class=36, pretrained_model = None):
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            kwargs = {'initialW': constant.Zero()}
        else:
            print "train resblock : True"
            # employ default initializers used in the original paper
            kwargs = {'initialW': normal.HeNormal(scale=1.0)}
        self.n_class = n_class
        super(DualCenterProposalNetworkRes50_predict7, self).__init__(
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

            # score_pool = L.Convolution2D(512 * 3, 512, 3, stride=1, pad=1),
            score_conv = L.Convolution2D(512 * 3, 512, 3, stride=1, pad=1),
            bn_cls1 = L.BatchNormalization(512),
            upscore_final1=L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
            bn_cls2 = L.BatchNormalization(256),
            upscore_final2=L.Deconvolution2D(256, self.n_class, 4, stride=2, pad=1),
            # upscore_final=L.Deconvolution2D(256, self.n_class, 8, stride=4, pad=2),

            conv_cp1 = L.Convolution2D(512 * 3, 1024, 3, stride=1, pad=1),
            bn_cp1 = L.BatchNormalization(1024),

            pool_conv = L.Convolution2D(256, (n_class - 1) * 6, 3, stride=1, pad=1),
            bn_pool = L.BatchNormalization((n_class - 1) * 6),

            # center pose network
            conv_cp2 = L.Convolution2D(1024, 256, 3, stride=1, pad=1),
            bn_cp2 = L.BatchNormalization(256),
            upscore_cp1 = L.Deconvolution2D(256, (n_class - 1) * 6, 4, stride=2, pad=1),
            bn_cp3 = L.BatchNormalization((n_class - 1) * 6),
            upscore_cp2 = L.Deconvolution2D((n_class - 1) * 12, (n_class - 1) * 3, 4, stride=2, pad=1),

            # origin center pose network
            conv_ocp2 = L.Convolution2D(1024, 256, 3, stride=1, pad=1),
            bn_ocp2 = L.BatchNormalization(256),
            upscore_ocp1 = L.Deconvolution2D(256, (n_class - 1) * 6, 4, stride=2, pad=1),
            bn_ocp3 = L.BatchNormalization((n_class - 1) * 6),
            upscore_ocp2 = L.Deconvolution2D((n_class - 1) * 12, (n_class - 1) * 3, 4, stride=2, pad=1),
        )

    def __call__(self, x1):
        h = x1
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        # Res Blocks
        h = self.res2(h) # 1/4
        pool1_4 = h
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
        concat_pool = F.dropout(h, ratio=0.1)

        # score
        h_cls = F.relu(self.bn_cls1(self.score_conv(h)))
        h_cls = F.elu(self.bn_cls2(self.upscore_final1(h_cls)))
        h_cls = self.upscore_final2(h_cls)
        score = h_cls
        # h = F.elu(self.score_pool(concat_pool))
        # score1_8 = h
        # h = F.relu(self.upscore_final(h))
        # score = h  # 1/1

        pool1_4 = F.relu(self.bn_pool(self.pool_conv(pool1_4)))

        h_cp = F.relu(self.bn_cp1(self.conv_cp1(concat_pool)))
        h_ocp = h_cp

        h_cp = F.relu(self.bn_cp2(self.conv_cp2(h_cp)))
        h_cp = F.elu(self.bn_cp3(self.upscore_cp1(h_cp)))
        h_cp = concat.concat((h_cp, pool1_4), axis=1)
        h_cp = self.upscore_cp2(h_cp)

        h_ocp = F.relu(self.bn_ocp2(self.conv_ocp2(h_ocp)))
        h_ocp = F.elu(self.bn_ocp3(self.upscore_ocp1(h_ocp)))
        h_ocp = concat.concat((h_ocp, pool1_4), axis=1)
        h_ocp = self.upscore_ocp2(h_ocp)

        cp_score = F.tanh(h_cp)
        ocp_score = F.tanh(h_ocp)

        return score, cp_score, ocp_score
