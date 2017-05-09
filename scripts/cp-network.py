import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class CenterProposalNetwork(chainer.Chain):
    def __init__(self):
        super(CenterProposalNetwork, self).__init__(
            # resnet50
            # depth network
            # psp net
            # label network
            # center pose network
        )

        def __call__(self, x, train_label=None, train_cp=None):

            return x



