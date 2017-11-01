import numpy

from chainer import function_node
from chainer import function
from chainer.utils import type_check

from chainer import functions as F

class MaskMeanSquaredError(function_node.FunctionNode):

    """Mask Mean squared error function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[2].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape,
            in_types[2].shape[0] == in_types[0].shape[0],
            in_types[2].shape[-2:] == in_types[0].shape[-2:]
        )

    def forward_cpu(self, inputs):
        x0, x1, x2 = inputs
        if len(x0.shape) == 5 and len(x2.shape) == 4:
            mask = x2[:, :, numpy.newaxis, :, :]
        elif len(x0.shape) == 5:
            mask = x2[:, numpy.newaxis, numpy.newaxis, :, :]
        else:
            mask = x2[:, numpy.newaxis, :, :]
        self.diff = (inputs[0] - inputs[1]) * mask
        diff = self.diff.ravel()
        self.nonzero_size = diff[diff != 0].size
        return numpy.array(diff.dot(diff) / self.nonzero_size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1, x2 = inputs
        if len(x0.shape) == 5 and len(x2.shape) == 4:
            mask = x2[:, :, numpy.newaxis, :, :]
        elif len(x0.shape) == 5:
            mask = x2[:, numpy.newaxis, numpy.newaxis, :, :]
        else:
            mask = x2[:, numpy.newaxis, :, :]
        self.diff = (inputs[0] - inputs[1]) * mask
        diff = self.diff.ravel()
        self.nonzero_size = diff[diff != 0].size
        return diff.dot(diff) / diff.dtype.type(self.nonzero_size),

    def backward(self, indexes, gy):
        ret = []
        gy0 = F.broadcast_to(gy[0], self.diff.shape) * (self.nonzero_size ==True)
        gx0 = gy0 * self.diff * (2. / self.diff.size)
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
        return ret

def mask_mean_squared_error(x0, x1, x2):
    # x2 is mask
    return MaskMeanSquaredError().apply((x0, x1, x2))[0]

