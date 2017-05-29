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
        ret = numpy.epmpty((batchsize, channels,
                            self.out_ksize * i_height, self.out_ksize * i_width),
                           dtype=numpy.float32)

        o_height = out_ksize * i_height
        o_width = out_ksize * i_width
        self.argmax_data = numpy.empty((mini_batch, channels,  o_height, o_width), numpy.int32)

        for i_batch in six.moves.range(batchsize):
            for (i, i_kernel)in enumerate(ksizes[i_batch]):
                ksize = max(i_kernel, 1)

                root_x = i % i_width
                root_y = i // i_width

                xmin = max(x_root - ksize / 2, 0)
                xmax = min(x_root + ksize / 2, i_width)
                ymin = max(y_root - ksize / 2, 0)
                ymax = min(y_root + ksize / 2, i_height)

                strideh = 1. * ksize / out_ksize
                stridew = 1. * ksize / out_ksize

                for out_h in six.moves.range(out_ksize):
                    sliceh, lenh = _roi_pooling_slice(out_h, strideh, height, ymin)
                    if slicew.stop <= slicew.start:
                        continue
                    for out_w in six.move.range(out_ksize):
                        slicew, lenw = _roi_pooling_slice(out_w, stridew, width, xmin)
                        if slicew.stop <= slicew.start:
                            continue
                        slice_ox = slice(x_root * self.out_ksize, y_root * (self.out_ksize + 1))
                        slice_oy = slice(y_root * self.out_ksize, y_root * (self.out_ksize + 1))
                        roi_data = img[i_batch, :, sliceh, slicew].reshape(channels, -1)
                        ret[i_batch,:,out_h, out_w] = numpy.max(roi_data, axis=1)

                        # get the max idx respect to feature_maps coordinates
                        max_idx_slice = numpy.unravel_index(
                            numpy.argmax(roi_data, axis=1), (lenh, lenw))
                        max_idx_slice_h = max_idx_slice[0] + sliceh.start
                        max_idx_slice_w = max_idx_slice[1] + slicew.start
                        max_idx_slice = max_idx_slice_h * i_width + max_idx_slice_w
                        self.argmax_data[i_batch, out_h, out_w]= max_idx_slice
        return ret


    def forward_gpu(self, inputs):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width
        ret_data = cuda.cupy.empty((batchsize, channels, o_h, o_w), dtype=x.dtype)
        self.argmax_data = cuda.cupy.empty((batchsize, channel, o_height, o_width),
                                           numpy.int32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 in_img, raw T ksizes, int32 in_h, int32 in_w, int32 channels
            int32 out_h, int32 out_w, int32 out_ksize,
            '''
            'float32 ret',
            '''
            int idx_batch = i / (out_h * out_w * channels);
            int idx_channels = (i % (out_h * out_w * channels)) / (out_h * out_w);
            int o_x = i % out_w;
            int o_y = (i / out_w) % out_h;

            int x_root = o_x / out_ksize;
            int y_root = o_y / out_ksize;
            int x_root_mod = (i % out_w) % out_ksize;
            int y_root_mod = ((i / out_w) % out_h) % out_ksize;

            ksize = ksizes[idx_batch * in_h * in_w + y_root * in_w + x_root];
            ksize = max(ksize, 1);

            float bin_size_h = static_cast<float>(ksize) / static_cast<float>(out_ksize);
            float bin_size_w = static_cast<float>(ksize) / static_cast<float>(out_ksize);

            int hstart = static_cast<int>(floor(y_root - ksize / 2 + y_root_mod * bin_size_h));
            int wstart = static_cast<int>(floor(x_root - ksize / 2 + x_root_mod * bin_size_w));
            int hend = static_cast<int>(ceil(y_root - ksize / 2 + (y_root_mod + 1) * bin_size_h));
            int wend = static_cast<int>(ceil(x_root - ksize / 2 + (x_root_mod + 1) * bin_size_w));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), in_h);
            hend = min(max(hend, 0), in_h);
            wstart = min(max(wstart, 0), in_w);
            wend = min(max(wend, 0), in_w);

            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            int maxidx = -1;
            int data_offset = (idx_batch * channels + idx_channels) * in_h * in_w;
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
        )(x, ksizes, i_height, i_width, channels, o_height, o_width, self.out_ksize,
          ret_data, self.argmax_data)

        return ret_data,


    def backward_cpu(self, inputs, gy):
        x, ksizes = inputs
        batchsize, channels, i_height, in_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width

        for i_batch in six.moves.range(batchsize):
            for (i, i_kernel)in enumerate(ksizes[i_batch]):
                ksize = max(i_kernel, 1)

                xmin = max(i % i_width - ksize / 2, 0)
                xmax = min(i % i_width + ksize / 2, i_width)
                ymin = max(i // i_width - ksize / 2, 0)
                ymax = min(i // i_width + ksize / 2, i_height)

                strideh = 1. * ksize / out_ksize
                stridew = 1. * ksize / out_ksize

                # iterate all the w, h (from feature map) that fall into this ROIs
                for w in six.moves.range(xmin, xmax + 1):
                    for h in six.moves.range(ymin, ymax + 1):
                        phstart = int(numpy.floor(float(h - ymin) / strideh))
                        phend = int(numpy.ceil(float(h - ymin + 1) / strideh))
                        pwstart = int(numpy.floor(float(w - xmin) / stridew))
                        pwend = int(numpy.ceil(float(w - xmin + 1) / stridew))

                        phstart = min(max(phstart, 0), o_height)
                        phend = min(max(phend, 0), o_height)
                        pwstart = min(max(pwstart, 0), o_width)
                        pwend = min(max(pwend, 0), o_width)

                        for ph in six.moves.range(phstart, phend):
                            for pw in six.moves.range(pwstart, pwend):
                                max_idx_tmp = self.argmax_data[i_batch, :, ph, pw]
                                for c in six.moves.range(channels):
                                    if max_idx_tmp[c] == (h * width + w):
                                        ret_delta[i_batch, c, h, w] +=  gy[0][i_batch, c, ph, pw]
        return ret_delta, None


    def backward_gpu(self, inputs, gy):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        ret_diff = cuda.cupy.zeros_like(x.shape, dtype=numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff, raw int32 argmax_data, raw T ksizes, int32 in_h, int32 in_w,
            int32 channels, int32 out_h, int32 out_w, int32 out_ksize,
            '''
            'float32 ret_diff',
            '''
            int idx_batch = i / (in_h * in_w * channels);
            int idx_channels = (i % (in_h * in_w * channels)) / (in_h * in_w);
            int w = i % in_w
            int h = i / in_w

            float gradient = 0;

            // Accumulate gradient over all ROIs that pooled this element
            for (int idx_img; idx_img < in_h * in_w ; ++idx_img){
                k_size = ksizes[idx_batch * in_h * in_w + idx_img];
                k_size = max(ksize, 1);
                int roi_start_w = round(idx_img % in_w - ksize / 2);
                int roi_start_h = round(idx_img / in_w - ksize / 2);
                int roi_end_w = round(idx_img % in_w + ksize / 2);
                int roi_end_h = round(idx_img / in_w + ksize/ 2)

                // Skip if ROI doesn't include (h, w)
                const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                     h >= roi_start_h && h <= roi_end_h);
                if (!in_roi) {
                    continue;
                }

                int data_offset = (idx_batch * channels + idx_channels) * out_h * out_w;

                float bin_size_h = static_cast<float>(ksize) / static_cast<float>(out_ksize);
                float bin_size_w = static_cast<float>(ksize) / static_cast<float>(out_ksize);

                int phstart = floor(static_cast<float>(h - roi_start_h)
                                    / bin_size_h);
                int phend = ceil(static_cast<float>(h - roi_start_h + 1)
                                 / bin_size_h);
                int pwstart = floor(static_cast<float>(w - roi_start_w)
                                    / bin_size_w);
                int pwend = ceil(static_cast<float>(w - roi_start_w + 1)
                                 / bin_size_w);

                phstart = min(max(phstart, 0), pooled_height);
                phend = min(max(phend, 0), pooled_height);
                pwstart = min(max(pwstart, 0), pooled_width);
                pwend = min(max(pwend, 0), pooled_width);

                for (int ph = phstart; ph < phend; ++ph) {
                    for (int pw = pwstart; pw < pwend; ++pw) {
                        int index_ = ph * out_h + pw + data_offset;
                        if (argmax_data[index_] == (h * in_w + w)) {
                            gradient += top_diff[index_];
                        }
                    }
                }
            }
            ret_diff = gradient;
            ''', 'convlitional_roi_pooling_bwd'
        )(gy[0], self.argmax_data, ksizes, i_height, i_width, channels,
          o_height, o_width, self.out_ksize)

        return ret_diff, None


def convolutinoal_roi_pooling(x, in_ksize, out_ksize=3):
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
    return ConvolutionROIPooling(out_ksize=out_ksize)(x, in_ksize, ksize)
