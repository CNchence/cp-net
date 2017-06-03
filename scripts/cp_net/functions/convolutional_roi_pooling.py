import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _roi_pooling_slice(size, stride, max_size, roi_offset):
    start = int(numpy.floor(size * stride))
    end = int(numpy.ceil((size + 1) * stride))

    start = min(max(start + roi_offset, 0), max_size)
    end = min(max(end + roi_offset, 0), max_size)

    return slice(start, end), end - start


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
            ksizes_type.ndim == 3,
        )

    def forward_cpu(self, inputs):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        ret = numpy.empty((batchsize, channels,
                           self.out_ksize * i_height, self.out_ksize * i_width),
                          dtype=numpy.float32)

        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width
        self.argmax_data = numpy.empty((batchsize, channels,  o_height, o_width), numpy.int32)
        # cnt = 0
        for i_batch in six.moves.range(batchsize):
            for (i, i_kernel)in enumerate(ksizes[i_batch].ravel()):
                ksize = int(max(i_kernel, 1))
                x_root = i % i_width
                y_root = i // i_width

                xmin = x_root - ksize / 2
                ymin = y_root - ksize / 2

                strideh = 1. * (ksize // 2 * 2 + 1) / self.out_ksize
                stridew = 1. * (ksize // 2 * 2 + 1) / self.out_ksize

                for out_h in six.moves.range(y_root * self.out_ksize,
                                             (y_root + 1) * self.out_ksize):
                    y_mod = out_h % self.out_ksize;
                    sliceh, lenh = _roi_pooling_slice(y_mod, strideh, i_height, ymin)
                    if sliceh.stop <= sliceh.start:
                        ret[i_batch, :, out_h, :] = 0
                        self.argmax_data[i_batch, :, out_h, :] = -1
                        continue
                    for out_w in six.moves.range(x_root * self.out_ksize,
                                                 (x_root + 1) * self.out_ksize):
                        x_mod = out_w % self.out_ksize;
                        slicew, lenw = _roi_pooling_slice(x_mod, stridew, i_width, xmin)
                        if slicew.stop <= slicew.start:
                            ret[i_batch, :, out_h, out_w] = 0
                            self.argmax_data[i_batch, :, out_h, out_w] = -1
                            # print "error w"
                            # print out_h
                            # print out_w
                            # print stridew
                            # print "xmin : " + str(xmin)
                            # print "xmax : " + str(xmax)
                            # print  int(numpy.floor(x_mod * stridew))
                            # print  int(numpy.ceil((x_mod + 1) * stridew))
                            # print slicew.start
                            # print slicew.stop
                            continue
                        roi_data = x[i_batch, :, sliceh, slicew].reshape(channels, -1)
                        ret[i_batch, :, out_h, out_w] = numpy.max(roi_data, axis=1)
                        # get the max idx respect to feature_maps coordinates
                        max_idx_slice = numpy.unravel_index(
                            numpy.argmax(roi_data, axis=1), (lenh, lenw))
                        max_idx_slice_h = max_idx_slice[0] + sliceh.start
                        max_idx_slice_w = max_idx_slice[1] + slicew.start
                        max_idx_slice = max_idx_slice_h * i_width + max_idx_slice_w
                        self.argmax_data[i_batch, :, out_h, out_w] = max_idx_slice
                        # cnt +=1
        # print cnt
        return ret,


    def forward_gpu(self, inputs):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width
        ret_data = cuda.cupy.empty((batchsize, channels, o_height, o_width), dtype=x.dtype)
        self.argmax_data = cuda.cupy.empty((batchsize, channels, o_height, o_width),
                                           numpy.int32)
        # cnt = cuda.cupy.zeros(ret_data.shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 in_img, raw T ksizes, int32 in_h, int32 in_w, int32 channels,
            int32 out_h, int32 out_w, int32 out_ksize
            ''',
            'float32 ret, int32 argmax_data',
            '''
            int idx_batch = i / (out_h * out_w * channels);
            int idx_channels = (i % (out_h * out_w * channels)) / (out_h * out_w);
            int o_x = i % out_w;
            int o_y = (i / out_w) % out_h;

            int x_root = o_x / out_ksize;
            int y_root = o_y / out_ksize;
            int x_root_mod = o_x % out_ksize;
            int y_root_mod = o_y % out_ksize;

            int ksize = ksizes[idx_batch * in_h * in_w + y_root * in_w + x_root];
            ksize = max(ksize, 1);

            int ymin = y_root - ksize / 2;
            int xmin = x_root - ksize / 2;

            int ksize_half = ksize / 2;
            float bin_size_h = static_cast<float>((ksize_half * 2 + 1)
                                   / static_cast<float>(out_ksize);
            float bin_size_w = static_cast<float>(ksize_half * 2 + 1)
                                   / static_cast<float>(out_ksize);

            int hstart = static_cast<int>(floor(y_root_mod * bin_size_h)) + ymin;
            int wstart = static_cast<int>(floor(x_root_mod * bin_size_w)) + xmin;
            int hend = static_cast<int>(ceil((y_root_mod + 1) * bin_size_h)) + ymin;
            int wend = static_cast<int>(ceil((x_root_mod + 1) * bin_size_w)) + xmin;

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), in_h);
            hend = min(max(hend, 0), in_h);
            wstart = min(max(wstart, 0), in_w);
            wend = min(max(wend, 0), in_w);

            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            int maxidx = -1;
            int data_offset = (idx_batch * channels + idx_channels) * in_h * in_w;
            for (int h = hstart; h < hend; ++h){
                for (int w = wstart + 1; w < wend; ++w) {
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
        )(x, ksizes, i_height, i_width, channels, o_height, o_width, self.out_ksize,
          ret_data, self.argmax_data)

        return ret_data,


    def backward_cpu(self, inputs, gy):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width
        ret_delta = numpy.zeros_like(x, dtype=numpy.float32)
        for i_batch in six.moves.range(batchsize):
            # cnt = 0
            # duplicated simple implimentation
            for o_h in six.moves.range(o_height):
                for o_w in six.moves.range(o_width):
                    for c in six.moves.range(channels):
                        max_idx = self.argmax_data[i_batch, c, o_h, o_w]
                        if max_idx >= 0:
                            h = max_idx // i_width
                            w = max_idx % i_width
                            ret_delta[i_batch, c, h, w] += gy[0][i_batch, c, o_h, o_w]
                            # cnt +=1
                        else:
                            continue

            # print "gy size : " + str(gy[0][0].size)

            # for (i, i_kernel)in enumerate(ksizes[i_batch].ravel()):
            #     ksize = int(max(i_kernel, 1))

            #     xmin = max(i % i_width - ksize / 2, 0)
            #     xmax = min(i % i_width + ksize / 2, i_width - 1)
            #     ymin = max(i // i_width - ksize / 2, 0)
            #     ymax = min(i // i_width + ksize / 2, i_height - 1)

            #     # strideh = 1. * (ymax - ymin + 1) / self.out_ksize
            #     # stridew = 1. * (xmax - xmin + 1) / self.out_ksize

            #     phstart = i // i_width * self.out_ksize
            #     phend = (i // i_width + 1) * self.out_ksize
            #     pwstart = i % i_width * self.out_ksize
            #     pwend = (i % i_width + 1) * self.out_ksize

            #     phstart = min(max(phstart, 0), o_height)
            #     phend = min(max(phend, 0), o_height)
            #     pwstart = min(max(pwstart, 0), o_width)
            #     pwend = min(max(pwend, 0), o_width)

            #     # iterate all the w, h (from feature map) that fall into this ROIs
            #     for w in six.moves.range(xmin, xmax + 1):
            #         for h in six.moves.range(ymin, ymax + 1):
            #             for ph in six.moves.range(phstart, phend):
            #                 for pw in six.moves.range(pwstart, pwend):
            #                     max_idx_tmp = self.argmax_data[i_batch, :, ph, pw]
            #                     for c in six.moves.range(channels):
            #                         if max_idx_tmp[c] == (h * i_width + w):
            #                             ret_delta[i_batch, c, h, w] +=  gy[0][i_batch, c, ph, pw]
            #                             cnt +=1
            # print cnt
        return ret_delta, None


    def backward_gpu(self, inputs, gy):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width
        ret_delta = cuda.cupy.zeros_like(x, dtype=numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff, raw int32 argmax_data, raw T ksizes,
            int32 in_h, int32 in_w, int32 channels,
            int32 out_h, int32 out_w, int32 out_ksize
            ''',
            'float32 ret_diff',
            '''
            int idx_batch = i / (in_h * in_w * channels);
            int idx_channels = (i % (in_h * in_w * channels)) / (in_h * in_w);
            int w = i % in_w;
            int h = (i / in_w) % in_h;

            int data_offset = (idx_batch * channels + idx_channels) * out_h * out_w;

            float gradient = 0;

            // naive implimentation
            for(int oh = 0 ; oh < out_h ; oh++){
                for(int ow = 0 ; ow < out_w ; ow++){
                    int index_ = oh * out_w + ow + data_offset;
                    if(argmax_data[index_] == (h * in_w + w)) {
                        gradient += top_diff[index_];
                    }
                }
            }

            /*
            // Accumulate gradient over all ROIs that pooled this element
            for (int idx_img = 0; idx_img < in_h * in_w ; ++idx_img){
                int ksize = ksizes[idx_batch * in_h * in_w + idx_img];
                ksize = max(ksize, 1);
                int roi_start_w = max(min(idx_img % in_w - ksize / 2, in_w -1), 0);
                int roi_start_h = max(min(idx_img / in_w - ksize / 2, in_h - 1), 0);
                int roi_end_w = max(min(idx_img % in_w + ksize / 2, in_w - 1), 0);
                int roi_end_h = max(min(idx_img / in_w + ksize/ 2, in_h - 1), 0);

                // Skip if ROI doesn't include (h, w)
                const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                     h >= roi_start_h && h <= roi_end_h);
                if (!in_roi) {
                    continue;
                }

                int phstart = (idx_img / in_w) * out_ksize;
                int phend = (idx_img / in_w + 1) * out_ksize;
                int pwstart = (idx_img % in_w) * out_ksize;
                int pwend = (idx_img % in_w + 1) * out_ksize;

                phstart = min(max(phstart, 0), out_h);
                phend = min(max(phend, 0), out_h);
                pwstart = min(max(pwstart, 0), out_w);
                pwend = min(max(pwend, 0), out_h);

                for (int ph = phstart; ph < phend; ++ph) {
                    for (int pw = pwstart; pw < pwend; ++pw) {
                        int index_ = ph * out_w + pw + data_offset;
                        if (argmax_data[index_] == (h * in_w + w)) {
                            gradient += top_diff[index_];
                        }
                    }
                }
            }
            */
            ret_diff = gradient;
            ''', 'convlitional_roi_pooling_bwd'
        )(gy[0], self.argmax_data, ksizes, i_height, i_width, channels,
          o_height, o_width, self.out_ksize,
          ret_delta)
        # print ret_delta.shape
        # print ret_delta.sum()
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
