import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from roi_pooling_2d import roi_pooling_2d
functions.roi_pooling_2d = roi_pooling_2d


class TestConvlotionalROIPooling(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.randn(4, 3, 48, 64).astype(numpy.float32)
        self.ksize_arr = numpy.random.randint(4, 8, (48, 64)).astype(numpy.float32)
        self.out_ksize = 3
        self.gy = numpy.random.uniform(-1, 1,
                                       (4, 3, out_ksize * 48,
                                        out_ksize * 64)).astype(numpy.float32)
        #    (4, 3, 7, 7)).astype(numpy.float32)

    def check_forward(self, x_data, roi_data):
        x = chainer.Variable(x_data)
        ksize_arr = chainer.Variable(roi_data)
        y = functions.convolutinoal_roi_pooling(x, ksize_arr)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward_cpu(self.x, self.ksize_arr)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.ksize_arr))

    def check_backward(self, x_data, roi_data, y_grad):
        x = chainer.Variable(x_data)
        ksize_arr = chainer.Variable(roi_data)
        y = functions.oconvolutinoal_roi_pooling(x, ksize_arr)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, ksize_arr.data))
        gx, gr = gradient_check.numerical_grad(f, (x.data, ksize_arr.data),
                                               (y.grad,))

        gradient_check.assert_allclose(cuda.to_cpu(gx), cuda.to_cpu(x.grad))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.ksize_arr, gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.ksize_arr),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
