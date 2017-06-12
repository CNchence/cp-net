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

from convolutional_roi_pooling import convolutional_roi_pooling
convolutional_roi_pooling.roi_pooling_2d = convolutional_roi_pooling


class TestConvlotionalROIPooling(unittest.TestCase):

    def setUp(self):
        batchsize = 4
        i_height = 120
        i_width = 120
        ch = 20
        self.x = numpy.random.randn(batchsize, ch,
                                    i_height, i_width).astype(numpy.float32)
        self.ksizes = numpy.random.randint(1, 6, (batchsize, 1,
                                                  i_height, i_width)).astype(numpy.float32)
        self.out_ksize = 3
        self.gy = numpy.random.uniform(-1, 1, (batchsize, ch, self.out_ksize * i_height,
                                               self.out_ksize * i_width)).astype(numpy.float32)

    def check_forward(self, x_data, ksizes_data):
        x = chainer.Variable(x_data)
        ksizes = chainer.Variable(ksizes_data)
        y = convolutional_roi_pooling(x, ksizes,out_ksize=self.out_ksize)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        print 'test_forward_cpu'
        self.check_forward(self.x, self.ksizes)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        print 'test_forward_gpu'
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.ksizes))

    def check_backward(self, x_data, ksizes_data, y_grad):
        x = chainer.Variable(x_data)
        ksizes = chainer.Variable(ksizes_data)
        y = convolutional_roi_pooling(x, ksizes, out_ksize=self.out_ksize)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, ksizes.data))
        print "[gradient_check] numerical_grad"
        gx, gr = gradient_check.numerical_grad(f, (x.data, ksizes.data), (y.grad,))
        print "[gradient_check] assert_allclose"
        gradient_check.assert_allclose(cuda.to_cpu(gx), cuda.to_cpu(x.grad))

    # @condition.retry(3)
    # def test_backward_cpu(self):
    #     print 'test_backward_cpu'
    #     self.check_backward(self.x, self.ksizes, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        print 'test_backward_gpu'
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.ksizes),
                            cuda.to_gpu(self.gy))

testing.run_module(__name__, __file__)
