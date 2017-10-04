import numpy

from chainer import function
from chainer.utils import type_check


class MaskMeanSquaredError(function.Function):

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
        self.diff = (x0 - x1) * mask
        diff = self.diff.ravel()
        self.non_zero_size = self.diff.size - self.diff[self.diff == 0].size
        return numpy.array(diff.dot(diff) / self.non_zero_size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1, x2 = inputs
        if len(x0.shape) == 5 and len(x2.shape) == 4:
            mask = x2[:, :, numpy.newaxis, :, :]
        elif len(x0.shape) == 5:
            mask = x2[:, numpy.newaxis, numpy.newaxis, :, :]
        else:
            mask = x2[:, numpy.newaxis, :, :]
        self.diff = (x0 - x1) * mask
        diff = self.diff.ravel()
        self.non_zero_size = self.diff.size - self.diff[self.diff == 0].size
        return diff.dot(diff) / diff.dtype.type(self.non_zero_size),

    def backward(self, inputs, gy):
        coeff = gy[0] * gy[0].dtype.type(2. / self.non_zero_size)
        gx0 = coeff * self.diff
        return gx0, -gx0, -gx0

def mask_mean_squared_error(x0, x1, x2):
    # x2 is mask
    return MaskMeanSquaredError()(x0, x1, x2)

