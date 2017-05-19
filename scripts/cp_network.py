import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainer.initializers import constant
from chainer.initializers import normal
import chainer.links.model.vision.resnet as R
from chainer.functions.array import concat

class CenterProposalNetworkRes50FCNtest(chainer.Chain):
    def __init__(self, n_class=36, pretrained_model = None):
        self.n_class = n_class
        super(CenterProposalNetworkRes50FCNtest, self).__init__(
            conv1=L.Convolution2D(None, n_class, 3, stride=1, pad=1),
            conv2=L.Convolution2D(None, 3, 3, stride=1, pad=1)
            )
    def __call__(self, x1, x2):
        h = x1
        h_d = x2
        h = F.relu(self.conv1(h))
        h_d = F.relu(self.conv2(h))
        return h, h_d

class CenterProposalNetworkRes50FCN(chainer.Chain):
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
            res2=R.BuildingBlock(3, 64, 64, 256, 1, **kwargs),
            res3=R.BuildingBlock(4, 256, 128, 512, 2, **kwargs),
            res4=R.BuildingBlock(6, 512, 256, 1024, 2, **kwargs),
            res5=R.BuildingBlock(3, 1024, 512, 2048, 2, **kwargs),

            upscore32=L.Deconvolution2D(2048, 512, 8,
                                        stride=4, pad=2,  use_cudnn=False),
            upscore16=L.Deconvolution2D(1024, 512, 4,
                                        stride=2, pad=1, use_cudnn=False),

            concat_conv = L.Convolution2D(512 * 3, 512 * 3, 3, stride=1, pad=1),

            score_pool = L.Convolution2D(512 * 3, n_class, 1, stride=1, pad=0),
            cls_pool = L.Convolution2D(512 * 3, 128, 1, stride=1, pad=0),

            upscore_final=L.Deconvolution2D(self.n_class, self.n_class, 16,
                                            stride=8, pad=4, use_cudnn=False),

            # depth network
            conv_d1_1 = L.Convolution2D(1, 64, 3, stride=1, pad=1),
            bn_d1_1 = L.BatchNormalization(64),
            conv_d1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1),
            bn_d1_2 = L.BatchNormalization(64),

            conv_d2 = L.Convolution2D(64, 128, 3, stride=1, pad=1),
            bn_d2 = L.BatchNormalization(128),
            conv_d3 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
            bn_d3 = L.BatchNormalization(256),
            
            # center pose network
            conv_cp_1 = L.Convolution2D(256 + 512 + 128, 1024, 3, stride=1, pad=1),
            bn_cp_1 = L.BatchNormalization(1024),
            conv_cp_2 = L.Convolution2D(1024, 512, 3, stride=1, pad=1),
            bn_cp_2 = L.BatchNormalization(512),
            upscore_cp = L.Deconvolution2D(512, 3, 16, stride=8, pad=4, use_cudnn=False),

            # quaternion network
            conv_q_1 = L.Convolution2D(256 + 512 + 128, 1024, 3, stride=1, pad=1),
            bn_q_1 = L.BatchNormalization(1024),
            conv_q_2 = L.Convolution2D(1024, 512, 3, stride=1, pad=1),
            bn_q_2 = L.BatchNormalization(512),
            upscore_q = L.Deconvolution2D(512, 4, 16, stride=8, pad=4, use_cudnn=False),
        )

    def __call__(self, x1, x2,  eps=0.001, test=None):
        h = x1
        h_d = x2

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

        # upscore 1/32 -> 1/1
        h = self.upscore32(pool1_32)
        upscore1 = h  # 1/1

        # upscore 1/16 -> 1/1
        h = self.upscore16(pool1_16)
        upscore2 = h  # 1/1

        # concat conv
        h = concat.concat((upscore1, upscore2, pool1_8), axis=1)
        h = F.relu(self.concat_conv(h))
        concat_pool = h

         # score
        h = F.relu(self.score_pool(concat_pool))
        score1_8 = h
        h = F.relu(self.upscore_final(h))
        score = h  # 1/1
        self.score = score  # XXX: for backward compatibility

        h = F.relu(self.cls_pool(concat_pool))
        cls_pool1_8 = h

        h_d = F.relu(self.bn_d1_1(self.conv_d1_1(h_d)))
        h_d = F.relu(self.bn_d1_2(self.conv_d1_2(h_d)))
        h_d = F.max_pooling_2d(h_d, 2, stride=2, pad=0) # 1/2
        h_d = F.relu(self.bn_d2(self.conv_d2(h_d)))
        h_d = F.max_pooling_2d(h_d, 2, stride=2, pad=0) # 1/4
        h_d = F.relu(self.bn_d3(self.conv_d3(h_d)))
        h_d = F.max_pooling_2d(h_d, 2, stride=2, pad=0) # 1/8

        h_cp = concat.concat((h_d, pool1_8, cls_pool1_8), axis=1)
        h_cp = F.relu(self.bn_cp_1(self.conv_cp_1(h_cp)))
        h_cp = F.relu(self.bn_cp_2(self.conv_cp_2(h_cp)))
        cp_score = F.arctan(self.upscore_cp(h_cp))

        h_q = concat.concat((h_d, pool1_8, cls_pool1_8), axis=1)
        h_q = F.relu(self.bn_q_1(self.conv_q_1(h_q)))
        h_q = F.relu(self.bn_q_2(self.conv_q_2(h_q)))
        q_score = F.normalize(self.upscore_q(h_q), axis=1)

        return score, cp_score, q_score
