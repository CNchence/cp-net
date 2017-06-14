import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check
from chainer.utils import conv

# import time

class ConvolutionROIPooling(function.Function):
    def __init__(self, out_ksize=3, stride=1, pad=-10):
        self.out_ksize = int(out_ksize)
        self.stride = stride
        self.pad = pad

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, ksizes_type = in_types
        type_check.expect(
            # x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            # ksizes_type.dtype == numpy.float32,
            # ksizes_type.ndim == 4,
        )

    def forward_cpu(self, inputs):
        # t = time.time()
        x, ksizes_half = inputs
        ksizes_half = ksizes_half //2
        batchsize, channels, i_height, i_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width

        self.argmax_data = numpy.empty((batchsize, channels, o_height, o_width),
                                       dtype=numpy.int32)
        tmp_argmax = numpy.empty((2, batchsize, channels, o_height, o_width),
                                 dtype=numpy.int32)

        kmax_half = int(ksizes_half.max() + 1)
        x_pad = numpy.pad(x,
                          ((0, 0), (0, 0),
                           (kmax_half,  kmax_half), (kmax_half, kmax_half)),
                          mode='constant', constant_values=(self.pad,))

        mod = numpy.arange(self.out_ksize).reshape(self.out_ksize, 1, 1, 1, 1)

        arange_h = numpy.arange(i_height)
        arange_w = numpy.arange(i_width)

        mesh_xmin = arange_w - ksizes_half + kmax_half
        mesh_ymin = arange_h[:,numpy.newaxis] - ksizes_half + kmax_half
        stride_mesh = ((ksizes_half * 2 + 1) / self.out_ksize)[numpy.newaxis,:]

        start_mesh = numpy.floor(mod * stride_mesh)
        end_mesh = numpy.ceil((mod + 1) * stride_mesh)

        hstart_mesh = (start_mesh + mesh_ymin[numpy.newaxis,:]).astype(numpy.int32)
        hend_mesh= (end_mesh + mesh_ymin[numpy.newaxis,:]).astype(numpy.int32)
        wstart_mesh = (start_mesh + mesh_xmin[numpy.newaxis,:]).astype(numpy.int32)
        wend_mesh= (end_mesh + mesh_xmin[numpy.newaxis,:]).astype(numpy.int32)

        del mesh_xmin, mesh_ymin, start_mesh, end_mesh, stride_mesh

        hstart_out = hstart_mesh.transpose(1,2,3,0,4).reshape(batchsize, 1, o_height, i_width)
        hstart_out = numpy.repeat(hstart_out, self.out_ksize, axis=3)

        wstart_out = wstart_mesh.transpose(1,2,3,4,0).reshape(batchsize, 1, i_height, o_width)
        wstart_out = numpy.repeat(wstart_out, self.out_ksize, axis=2)

        hend_out = hend_mesh.transpose(1,2,3,0,4).reshape(batchsize, 1, o_height, i_width)
        hend_out = numpy.repeat(hend_out, self.out_ksize, axis=3)
        wend_out = wend_mesh.transpose(1,2,3,4,0).reshape(batchsize, 1, i_height, o_width)
        wend_out = numpy.repeat(wend_out, self.out_ksize, axis=2)
        delta_h_out = hend_out - hstart_out
        delta_w_out = wend_out - wstart_out
        unique_dh = numpy.unique(delta_h_out)
        unique_dw = numpy.unique(delta_w_out)

        for i in unique_dh:
            i_mask = (delta_h_out == i)
            for j in unique_dw:
                ij_mask = i_mask * (delta_w_out == j)
                if not numpy.any(ij_mask):
                    continue
                ij_indices = numpy.where(ij_mask)
                w_mod, h_mod = numpy.indices((i, j)).reshape(2, -1)
                max_idx_slice = numpy.unravel_index(
                    numpy.argmax(
                        x_pad[ij_indices[0][:,numpy.newaxis], :,
                              h_mod + hstart_out[ij_indices[0], 0, ij_indices[2],
                                                 ij_indices[3]][:,numpy.newaxis],
                              w_mod + wstart_out[ij_indices[0], 0, ij_indices[2],
                                                 ij_indices[3]][:, numpy.newaxis]], axis=1),
                    (i, j))
                tmp_argmax[0, ij_indices[0], :, ij_indices[2], ij_indices[3]] = max_idx_slice[0]
                tmp_argmax[1, ij_indices[0], :, ij_indices[2], ij_indices[3]] = max_idx_slice[1]

        # patch_argmax = numpy.empty((2, batchsize, channels, self.out_ksize * self.out_ksize),
        #                            dtype=numpy.int32)
        # k_indices_h, k_indices_w = numpy.indices((self.out_ksize, self.out_ksize))
        # delta_h = hend_mesh - hstart_mesh
        # delta_w = wend_mesh - wstart_mesh

        # vf = numpy.vectorize(slice)
        # sliceh_mesh = vf(hstart_mesh, hend_mesh)
        # slicew_mesh = vf(wstart_mesh, wend_mesh)

        # for i in six.moves.range(i_height * i_width):
        #     y_root = i // i_width
        #     x_root = i % i_width
        #     for bt in six.moves.range(batchsize):
        #         dh = delta_h[:, bt, 0, y_root, x_root]
        #         sliceh = sliceh_mesh[:, bt, 0, y_root, x_root]
        #         dw = delta_w[:, bt, 0, y_root, x_root]
        #         slicew = slicew_mesh[:, bt, 0, y_root, x_root]

        #         for j in six.moves.range(self.out_ksize **2):
        #             hh = j // self.out_ksize
        #             ww = j % self.out_ksize
        #             # get the max idx respect to feature_maps coordinates
        #             max_idx_slice = numpy.unravel_index(
        #                 numpy.argmax(
        #                     x_pad[bt, :,sliceh[hh], slicew[ww]].reshape(channels, -1), axis=1),
        #                 (dh[hh], dw[ww]))
        #             patch_argmax[0, bt, :, j] = max_idx_slice[0] # h
        #             patch_argmax[1, bt, :, j] = max_idx_slice[1] # w
        #     tmp_argmax[:,:,:, y_root * self.out_ksize + k_indices_h,
        #                x_root * self.out_ksize + k_indices_w] = \
        #     patch_argmax.reshape(2, batchsize, channels, self.out_ksize, self.out_ksize)

        tmp_argmax[0] += hstart_out
        tmp_argmax[1] += wstart_out
        tmp_argmax -=  kmax_half

        im_size = i_width * i_height

        tmp_argmax[0][tmp_argmax[0] > i_height - 1] = - im_size
        tmp_argmax[1][tmp_argmax[1] > i_width - 1] = - im_size
        tmp_argmax[tmp_argmax < 0] = - im_size
        self.argmax_data = tmp_argmax[0] * i_width + tmp_argmax[1]

        b, c, h, w = numpy.where(self.argmax_data >= 0)
        ret = numpy.zeros((batchsize, channels, o_height, o_width), dtype=x.dtype)
        ret[b, c, h, w] = x[b, c, self.argmax_data[b, c, h, w] // i_width,
                            self.argmax_data[b, c, h, w] % i_width]
        # print (time.time() - t)
        return ret,


    def forward_gpu(self, inputs):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape
        o_height = self.out_ksize * i_height
        o_width = self.out_ksize * i_width
        ret_data = cuda.cupy.empty((batchsize, channels, o_height, o_width), dtype=numpy.float64)
        self.argmax_data = cuda.cupy.empty_like(ret_data, dtype=numpy.int32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float64 in_img, raw T ksizes,
            int32 in_h, int32 in_w, int32 out_h, int32 out_w,
            int32 channels, int32 out_ksize
            ''',
            'float64 ret, int32 argmax_data',
            '''
            int idx_batch = i / (out_h * out_w * channels);
            int o_x = i % out_w;
            int o_y = (i / out_w) % out_h;

            int x_root = o_x / out_ksize;
            int y_root = o_y / out_ksize;
            int x_root_mod = o_x % out_ksize;
            int y_root_mod = o_y % out_ksize;

            int ksize_half = ksizes[idx_batch * in_h * in_w + y_root * in_w + x_root];
            ksize_half >>= 1;

            int ymin = y_root - ksize_half;
            int xmin = x_root - ksize_half;

            float bin_size = static_cast<float>(ksize_half << 1 + 1)
                                   / static_cast<float>(out_ksize);

            // Add roi offsets and clip to input boundaries
            int hstart = min(max(static_cast<int>(floor(y_root_mod * bin_size))
                                                                  + ymin, 0), in_h);
            int hend = min(max(static_cast<int>(ceil((y_root_mod + 1) * bin_size))
                                                                  + ymin, 0), in_h);
            int wstart = min(max(static_cast<int>(floor(x_root_mod * bin_size))
                                                                  + xmin, 0), in_w);
            int wend = min(max(static_cast<int>(ceil((x_root_mod + 1) * bin_size))
                                                                  + xmin, 0), in_w);

            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            double maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            int maxidx = -1;
            for (int h = hstart; h < hend; ++h){
                for (int w = wstart; w < wend; ++w) {
                    int data_offset = i / (out_h * out_w) * in_h * in_w;
                    int root_idx = h * in_w + w;
                    if (in_img[data_offset + root_idx] > maxval){
                         int maxval = static_cast<double>(in_img[data_offset + root_idx]);
                         int maxidx = root_idx;
                    }
                }
            }
            ret = maxval;
            argmax_data = maxidx;
            ''',
            'convolutional_roi_pooling_fwd'
        )(x.astype(numpy.float64), ksizes,
          i_height, i_width, o_width, o_height, channels, self.out_ksize,
          ret_data, self.argmax_data)
        ret_data = ret_data.astype(x.dtype)
        return ret_data,


    def backward_cpu(self, inputs, gy):
        x, ksizes = inputs
        batchsize, channels, i_height, i_width = x.shape

        # tot = time.time()

        i_imsize = i_height * i_width
        o_imsize = i_imsize * (self.out_ksize ** 2)
        ret_delta = numpy.zeros_like(x.ravel(), dtype=gy[0].dtype)

        max_indices, = numpy.where(self.argmax_data.ravel() >= 0)

        ## indices with offset(minibatch * channel)
        delta_indices = self.argmax_data.ravel()[max_indices] + max_indices // o_imsize * i_imsize
        ret_delta = numpy.bincount(delta_indices, weights=gy[0].ravel()[max_indices]
                                   , minlength=len(x.ravel()))
        ret_delta = ret_delta.reshape(x.shape).astype(gy[0].dtype)

        # ret_delta = numpy.empty_like(x, dtype=gy[0].dtype)
        # for i_b in six.moves.range(batchsize):
        #     for i_c in six.moves.range(channels):
        #         max_indices, = numpy.where(self.argmax_data[i_b, i_c].ravel() >= 0)
        #         delta_indices = self.argmax_data[i_b, i_c].ravel()[max_indices]
        #         gy_tmp = gy[0][i_b, i_c].ravel()
        #         patch_delta = numpy.zeros((i_height * i_width), dtype=gy[0].dtype)
        #         for ind, max_ind in six.moves.zip(delta_indices, max_indices):
        #             patch_delta[ind] += gy_tmp[max_ind]
        #         ret_delta[i_b, i_c] = patch_delta.reshape(i_height, i_width)

        # print "total"
        # print time.time() - tot
        # print "----------"
        return ret_delta, None


    def backward_gpu(self, inputs, gy):
        ## atomicAdd for float64 is not provided, so use float32
        x = inputs[0]
        i_height, i_width = x.shape[2:]
        ret_delta = cuda.cupy.zeros_like(x, dtype=cuda.cupy.float32)
        i_imsize = i_height * i_width
        o_imsize = i_imsize * (self.out_ksize **2)
        cuda.cupy.ElementwiseKernel(
            '''
            float32 top_diff, int32 argmax_data,
            int32 in_imsize, int32 out_imsize
            ''',
            'raw float32 ret_diff',
            '''
            if(argmax_data >= 0){
                atomicAdd(&ret_diff[argmax_data + i / out_imsize * in_imsize], top_diff);
            }
            ''', 'convlitional_roi_pooling_bwd'
        )(gy[0].astype(numpy.float32), self.argmax_data, i_imsize, o_imsize,
          ret_delta)
        ret_delta = ret_delta.astype(gy[0].dtype)
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
