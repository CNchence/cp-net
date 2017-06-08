import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check
from chainer.utils import conv

import time

class ConvolutionROIPooling(function.Function):
    def __init__(self, out_ksize=3, stride=1, pad=0):
        self.out_ksize = out_ksize
        self.stride = stride

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, ksizes_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            ksizes_type.dtype == numpy.float32,
            # ksizes_type.ndim == 4,
        )

    def forward_cpu(self, inputs):
        t = time.time()
        x, ksizes_half = inputs
        ksizes_half = ksizes_half //2
        batchsize, channels, i_height, i_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width
        ret = numpy.empty((batchsize, channels, o_height, o_width), dtype=numpy.float32)
        self.argmax_data = numpy.empty((batchsize, channels, o_height, o_width), numpy.int32)

        patch_max = numpy.empty((batchsize, channels, self.out_ksize, self.out_ksize))
        patch_argmax = numpy.empty((2, batchsize, channels, self.out_ksize, self.out_ksize))
        tmp_argmax = numpy.empty((2, batchsize, channels, o_height, o_width))

        kmax_half = int(ksizes_half.max())
        x_pad = numpy.pad(x,
                          ((0, 0), (0, 0),
                           (kmax_half,  kmax_half), (kmax_half, kmax_half)),
                          mode='constant', constant_values=(0,))

        im_size = i_width * i_height
        mod = numpy.arange(self.out_ksize)
        mesh_b, mesh_h, mesh_w = numpy.meshgrid(six.moves.range(batchsize), mod, mod)

        arange_h = numpy.arange(i_height)
        arange_w = numpy.arange(i_width)

        mesh_y, mesh_x = numpy.meshgrid(arange_h, arange_w)
        mesh_xmin = arange_w - ksizes_half + kmax_half
        mesh_ymin = arange_h[:,numpy.newaxis] - ksizes_half + kmax_half
        stride_mesh = (ksizes_half * 2 + 1) / self.out_ksize

        # hstart = (numpy.floor(mod * stride[:,numpy.newaxis]) + ymin[:,numpy.newaxis]).ravel()
        # wstart = (numpy.floor(mod * stride[:,numpy.newaxis]) + xmin[:,numpy.newaxis]).ravel()
        # hend = (numpy.ceil((mod + 1) * stride[:,numpy.newaxis]) + ymin[:,numpy.newaxis]).ravel()
        # wend = (numpy.ceil((mod + 1) * stride[:,numpy.newaxis]) + xmin[:,numpy.newaxis]).ravel()

        # for i in six.moves.range(im_size):
        for y_root, x_root in six.moves.zip(mesh_y.ravel(), mesh_x.ravel()):
            slicey = slice(y_root * self.out_ksize, (y_root + 1) * self.out_ksize)
            slicex = slice(x_root * self.out_ksize, (x_root + 1) * self.out_ksize)

            xmin = mesh_xmin[:, 0, y_root, x_root][:,numpy.newaxis]
            ymin = mesh_ymin[:, 0, y_root, x_root][:,numpy.newaxis]
            stride = stride_mesh[:, 0, y_root, x_root][:,numpy.newaxis]

            hstart = (numpy.floor(mod * stride) + ymin).ravel()
            wstart = (numpy.floor(mod * stride) + xmin).ravel()
            hend = (numpy.ceil((mod + 1) * stride) + ymin).ravel()
            wend = (numpy.ceil((mod + 1) * stride) + xmin).ravel()

            for bt, hh, ww in zip(mesh_b.ravel(), mesh_h.ravel(), mesh_w.ravel()):
                hs = int(hstart[hh])
                he = int(hend[hh])
                ws = int(wstart[ww])
                we = int(wend[ww])
                # hs = int(hstart[bt, :, hh, ww])
                # he = int(hend[bt, :, hh, ww])
                # ws = int(wstart[bt, :, hh, ww])
                # we = int(wend[bt, :, hh, ww])
                roi_data = x_pad[bt, :, hs:he, ws:we].reshape(channels, -1)
                patch_max[bt, :, hh, ww] = numpy.max(roi_data, axis=1)
                # get the max idx respect to feature_maps coordinates
                max_idx_slice = numpy.unravel_index(numpy.argmax(roi_data, axis=1),
                                                    (he - hs, we - ws))
                patch_argmax[0, bt, :, hh, ww] = max_idx_slice[0] + hs - kmax_half # height
                patch_argmax[1, bt, :, hh, ww] = max_idx_slice[1] + ws - kmax_half # width

            ret[:, :, slicey, slicex] = patch_max
            tmp_argmax[:, :, :, slicey, slicex] = patch_argmax

        tmp_argmax[0][tmp_argmax[0] > i_height - 1] = - im_size
        tmp_argmax[1][tmp_argmax[1] > i_width - 1] = - im_size
        tmp_argmax[tmp_argmax < 0] = - im_size
        self.argmax_data = tmp_argmax[0] * i_width + tmp_argmax[1]

        print time.time() - t
        return ret,


    def forward_gpu(self, inputs):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width
        ret_data = cuda.cupy.empty((batchsize, channels, o_height, o_width), dtype=x.dtype)
        self.argmax_data = cuda.cupy.empty_like(ret_data, numpy.int32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 in_img, raw float32 ksizes, int32 in_h, int32 in_w,
            int32 channels, int32 out_ksize
            ''',
            'float32 ret, int32 argmax_data',
            '''
            int out_h = in_h * out_ksize;
            int out_w = in_w * out_ksize;
            int idx_batch = i / (out_h * out_w * channels);
            int o_x = i % out_w;
            int o_y = (i / out_w) % out_h;

            int x_root = o_x / out_ksize;
            int y_root = o_y / out_ksize;
            int x_root_mod = o_x % out_ksize;
            int y_root_mod = o_y % out_ksize;

            int ksize = ksizes[idx_batch * in_h * in_w + y_root * in_w + x_root];
            int ksize_half = max(ksize, 1) / 2;

            int ymin = y_root - ksize_half;
            int xmin = x_root - ksize_half;

            float bin_size = static_cast<float>(ksize_half * 2 + 1)
                                   / static_cast<float>(out_ksize);

            int hstart = static_cast<int>(floor(y_root_mod * bin_size));
            int wstart = static_cast<int>(floor(x_root_mod * bin_size));
            int hend = static_cast<int>(ceil((y_root_mod + 1) * bin_size));
            int wend = static_cast<int>(ceil((x_root_mod + 1) * bin_size));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + ymin, 0), in_h);
            hend = min(max(hend + ymin, 0), in_h);
            wstart = min(max(wstart + xmin, 0), in_w);
            wend = min(max(wend + xmin, 0), in_w);

            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            int maxidx = -1;
            int data_offset = i / (out_h * out_w) * in_h * in_w;
            for (int h = hstart; h < hend; ++h){
                for (int w = wstart; w < wend; ++w) {
                    int root_idx = h * in_w + w;
                    if (in_img[data_offset + root_idx] > maxval){
                        maxval = in_img[data_offset + root_idx];
                        maxidx = root_idx;
                    }
                }
            }
            ret = maxval;
            argmax_data = maxidx;
            ''',
            'convolutional_roi_pooling_fwd'
        )(x, ksizes, i_height, i_width, channels, self.out_ksize,
          ret_data, self.argmax_data)

        return ret_data,


    def backward_cpu(self, inputs, gy):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width

        # duplicated simple implimentation
        # tot = time.time()
        # ret_delta = numpy.zeros_like(x, dtype=numpy.float32)
        # max_indices = numpy.where(self.argmax_data >= 0)
        # h_list = (self.argmax_data[max_indices[0], max_indices[1],
        #                           max_indices[2], max_indices[3]] // i_width).astype(numpy.int32)
        # w_list = (self.argmax_data[max_indices[0], max_indices[1],
        #                            max_indices[2], max_indices[3]] % i_width).astype(numpy.int32)
        # for i in six.moves.range(len(h_list)):
        #     ret_delta[max_indices[0][i], max_indices[1][i], h_list[i], w_list[i]] += \
        #          gy[0][max_indices[0][i], max_indices[1][i], max_indices[2][i], max_indices[3][i]]

        tot = time.time()
        ret_delta = numpy.zeros_like(x, dtype=numpy.float32)
        for i_batch in six.moves.range(batchsize):
            for c in six.moves.range(channels):
                max_indices = numpy.where(self.argmax_data[i_batch, c] >= 0)
                h_list = (self.argmax_data[i_batch, c,max_indices[0],
                                           max_indices[1]] // i_width).astype(numpy.int32)
                w_list = (self.argmax_data[i_batch, c, max_indices[0],
                                           max_indices[1]] % i_width).astype(numpy.int32)
                for i in six.moves.range(len(h_list)):
                    ret_delta[i_batch, c, h_list[i], w_list[i]] += \
                                        gy[0][i_batch,c, max_indices[0][i], max_indices[1][i]]
                # for o_h in six.moves.range(o_height):
                #     for o_w in six.moves.range(o_width):
                #         max_idx = self.argmax_data[i_batch, c, o_h, o_w]
                #         if max_idx >= 0:
                #             h = int(max_idx // i_width)
                #             w = int(max_idx % i_width)
                #             ret_delta[i_batch, c, h, w] += gy[0][i_batch, c, o_h, o_w]

        print "total2"
        print time.time() - tot
        print "----------"
        return ret_delta, None


    def backward_gpu(self, inputs, gy):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        ret_delta = cuda.cupy.zeros_like(x, dtype=numpy.float32)
        kmax_half = int(ksizes.max() // 2)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff, raw int32 argmax_data,
            int32 in_h, int32 in_w, int32 channels,
            int32 out_ksize, int32 kmax_half
            ''',
            'float32 ret_diff',
            '''
            int w = i % in_w;
            int h = (i / in_w) % in_h;

            int out_h = in_h * out_ksize;
            int out_w = in_w * out_ksize;
            int data_offset = i / (in_h * in_w) * out_h * out_w;

            float gradient = 0;
            int hstart = max((h - kmax_half ) * out_ksize, 0);
            int hend = min((h + kmax_half + 1) * out_ksize, out_h);
            int wstart = max((w - kmax_half) * out_ksize, 0);
            int wend = min((w + kmax_half + 1) *out_ksize, out_w);

            for(int oh = hstart ; oh < hend ; ++oh){
                for(int ow = wstart ; ow < wend ; ++ow){
                    int index_ = oh * out_w + ow + data_offset;
                    if(argmax_data[index_] == in_w * h + w) {
                        gradient += top_diff[index_];
                    }
                }
            }
            ret_diff = gradient;
            ''', 'convlitional_roi_pooling_bwd'
        )(gy[0], self.argmax_data, i_height, i_width, channels,
          self.out_ksize, kmax_half,
          ret_delta)
        return ret_delta, None


def convolutional_roi_pooling(x, ksizes, out_ksize=3):
    """
    - input: feature_map (size : batchsize * K * H * W)
    - input: pixelwise kernel size  (size : batchsize * H * W)
    - output: feature_map (size: batchsize * K * (H + out_ksize) * (W * out_ksize))
    - like Average pooling but pixelwise different kernel size
    Args:
        TODO
    Returns:
        ~chainer.Variable: Output variable.
    """
    return ConvolutionROIPooling(out_ksize=out_ksize)(x, ksizes)
