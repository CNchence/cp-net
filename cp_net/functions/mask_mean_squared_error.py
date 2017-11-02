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
            in_types[2].shape[-2:] == in_types[0].shape[-2:],
            len(in_types[2].shape) == 3 or len(in_types[2].shape) == 4
        )

    def forward_cpu(self, inputs):
        x0, x1, x2 = inputs
        x2 = (x2 == True).astype(numpy.float32)
        if len(x2.shape) == 4:
            self.diff = (inputs[0] - inputs[1]) * x2[:, :, numpy.newaxis, :, :]
        if len(x2.shape) == 3:
            self.diff = (inputs[0] - inputs[1]) * x2[:, numpy.newaxis, numpy.newaxis, :, :]
        diff = self.diff.ravel()
        self.nonzero_size = diff[diff != 0].size
        return numpy.array(diff.dot(diff) / self.nonzero_size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1, x2 = inputs
        x2 = (x2 == True).astype(numpy.float32)
        if len(x2.shape) == 4:
            self.diff = (inputs[0] - inputs[1]) * x2[:, :, numpy.newaxis, :, :]
        if len(x2.shape) == 3:
            self.diff = (inputs[0] - inputs[1]) * x2[:, numpy.newaxis, numpy.newaxis, :, :]
        diff = self.diff.ravel()
        self.nonzero_size = diff[diff != 0].size
        return diff.dot(diff) / diff.dtype.type(self.nonzero_size),

    def backward(self, indexes, gy):
        ret = []
        gy0 = F.broadcast_to(gy[0], self.diff.shape)
        gx0 = gy0 * self.diff * (2. / self.nonzero_size)
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
        return ret

def mask_mean_squared_error(x0, x1, x2):
    # x2 is mask
    return MaskMeanSquaredError().apply((x0, x1, x2))[0]

