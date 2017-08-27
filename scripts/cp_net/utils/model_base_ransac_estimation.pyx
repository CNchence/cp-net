# -*- coding: utf-8 -*-

#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: profile=True

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange

from libc.math cimport round

ctypedef np.float64_t DOUBLE_t
ctypedef np.float32_t FLOAT_t

import time


cdef extern double mean1d_up_limit(double* x, int len_x, double uplim)
cdef extern double visibility_scoring(double* x, int len_x, int percentile_thre, double max_dist)
cdef extern double calc_invisib_socre_from_map(double* depth_diff, double* mask,
                                               int im_h, int im_w, double fore_thre,
                                               double percentile_thre, double max_dist_lim)

cdef extern void pointcloud_to_depth_impl(double* pc, double* K, double* depth,
                                          int im_h, int im_w, int len_pc)


cdef inline pointcloud_to_depth_c(np.ndarray[DOUBLE_t, ndim=2] pc,
                                  np.ndarray[DOUBLE_t, ndim=2] K, int im_h, int im_w):
    cdef np.ndarray[DOUBLE_t, ndim=1] depth = np.zeros(im_h * im_w)
    pointcloud_to_depth_impl(<double *> pc.data, <double *> K.data, <double *> depth.data,
                             im_h, im_w, len(pc[0]))

    return np.asanyarray(depth).reshape(im_h, im_w)


cdef inline pointcloud_to_depth_cy(np.ndarray[DOUBLE_t, ndim=2] pc,
                                   np.ndarray[DOUBLE_t, ndim=2] K, int im_h, int im_w):

    cdef np.ndarray[np.int32_t, ndim=1] xs = (pc[0, :] * K[0, 0] / pc[2, :] + K[0, 2]).astype(np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] ys = (pc[1, :] * K[1, 1] / pc[2, :] + K[1, 2]).astype(np.int32)

    cdef np.ndarray[np.uint8_t, ndim=1] inimage_mask = (xs >= 0) * (xs < im_w) * (ys >= 0) * (ys < im_h).astype(np.uint8)
    xs = xs[inimage_mask.view(dtype=np.bool)]
    ys = ys[inimage_mask.view(dtype=np.bool)]

    cdef np.ndarray[DOUBLE_t, ndim=1] zs = pc[2, :][inimage_mask.view(dtype=np.bool)]

    cdef np.ndarray[DOUBLE_t, ndim=2] img_depth = np.zeros((im_h, im_w),dtype=np.float64)
    cdef int i = 0, xx = -1, yy = -1, len_xs = len(xs)
    cdef double val = 0

    # render depth
    for i in xrange(len_xs):
        if xx == xs[i] and yy == ys[i] and val == img_depth[yy, xx]:
            continue
        val = img_depth[ys[i], xs[i]]
        if val == 0.0:
            val= zs[i]
        else:
            val = min(zs[i], val)

        img_depth[ys[i], xs[i]] = val


    return img_depth



cdef calc_rot_by_svd_cy(np.ndarray[DOUBLE_t, ndim=2] Y,
                        np.ndarray[DOUBLE_t, ndim=2] X):
    cdef np.ndarray[DOUBLE_t, ndim=2] R, U, V, H
    cdef np.ndarray[DOUBLE_t, ndim=1] S
    cdef double VU_det
    U, S, V = np.linalg.svd(np.dot(Y, X.T))
    VU_det = np.linalg.det(np.dot(V, U))
    H = np.diag(np.array([1.0, 1.0, VU_det], dtype=np.float64))
    R = np.dot(np.dot(U, H), V).T
    return R


def model_base_ransac_estimation_cy(np.ndarray[DOUBLE_t, ndim=2] y_arr,
                                    np.ndarray[DOUBLE_t, ndim=2] x_arr,
                                    np.ndarray[DOUBLE_t, ndim=2] model,
                                    np.ndarray[DOUBLE_t, ndim=2] depth,
                                    np.ndarray[DOUBLE_t, ndim=2] K,
                                    np.ndarray[DOUBLE_t, ndim=2] obj_mask,
                                    im_size,
                                    int n_ransac=100, double max_thre=0.1,
                                    int percentile_thre = 90):

    cdef np.ndarray[np.int32_t, ndim=2] rand_sample = np.array(
        np.random.randint(0, y_arr.shape[1], (n_ransac, 3)), dtype=np.int32)
    cdef np.ndarray[DOUBLE_t, ndim=3] rand_x = x_arr[:,rand_sample]
    cdef np.ndarray[DOUBLE_t, ndim=2] rand_x_mean = np.mean(rand_x, axis=2)
    cdef np.ndarray[DOUBLE_t, ndim=3] rand_y = y_arr[:, rand_sample]
    cdef np.ndarray[DOUBLE_t, ndim=2] rand_y_mean = np.mean(rand_y, axis=2)

    ## intialize
    cdef np.ndarray[DOUBLE_t, ndim=1] _t = np.zeros(3)
    cdef np.ndarray[DOUBLE_t, ndim=2] _R = np.diag((1.0, 1.0, 1.0))
    cdef np.ndarray[DOUBLE_t, ndim=1] ret_t = np.zeros(3)
    cdef np.ndarray[DOUBLE_t, ndim=2] ret_R = np.diag((1.0, 1.0, 1.0))
    cdef np.ndarray[DOUBLE_t, ndim=1] ret_t_tri = np.zeros(3)
    cdef np.ndarray[DOUBLE_t, ndim=2] ret_R_tri = np.diag((1.0, 1.0, 1.0))

    cdef double best_score = 1e15
    cdef double best_score_tri = 1e15
    cdef double obj_visib_thre = np.sum(obj_mask) * 0.5

    cdef int imsize_h = im_size[0]
    cdef int imsize_w = im_size[1]

    cdef np.ndarray[DOUBLE_t, ndim=2] depth_mask = (depth != 0).astype(np.float64)
    cdef np.ndarray[DOUBLE_t, ndim=2] depth_obj_mask = depth_mask * obj_mask
    cdef np.ndarray[DOUBLE_t, ndim=2] depth_nonobj_mask = depth_mask * (1 - obj_mask)

    cdef np.ndarray[DOUBLE_t, ndim=2] visib_map, depth_model, depth_diff, invisib_mask
    cdef np.ndarray[DOUBLE_t, ndim=1] dist, score_visib_arr

    cdef double score, score_tri, score_visib, score_invisib

    cdef np.ndarray[DOUBLE_t, ndim=2] rand_x_demean,  rand_y_demean

    cdef size_t i_ransac = 0
    for i_ransac in xrange(n_ransac):
        rand_x_demean = rand_x[:, i_ransac, :] - rand_x_mean[:, i_ransac, np.newaxis]
        rand_y_demean = rand_y[:, i_ransac, :] - rand_y_mean[:, i_ransac, np.newaxis]

        _R = calc_rot_by_svd_cy(rand_x_demean, rand_y_demean)
        _t = rand_y_mean[:, i_ransac] - np.dot(_R, rand_x_mean[:, i_ransac])
        dist = np.sum(np.abs(np.dot(_R, x_arr) + _t[:, np.newaxis] - y_arr), axis=0)
        score = mean1d_up_limit(<double*> dist.data, <int> len(dist), max_thre)

        if score < best_score:
            best_score = score
            ret_t = _t
            ret_R = _R

        # d2 = pointcloud_to_depth_cy(np.dot(_R, model) + _t[:, np.newaxis],
        #                             K, imsize_h, imsize_w)
        depth_model = pointcloud_to_depth_c(np.dot(_R, model) + _t[:, np.newaxis],
                                            K, imsize_h, imsize_w)
        # print np.sum(d2 -depth_model)

        depth_diff = depth_model - depth
        visib_map = np.abs(depth_diff) * depth_obj_mask

        if np.sum((depth_model != 0) * depth_obj_mask) > obj_visib_thre:
            score_visib_arr = visib_map[visib_map != 0]
            score_invisib = visibility_scoring(<double*> score_visib_arr.data,
                                               len(score_visib_arr), percentile_thre, max_thre)
        else:
            continue

        invisib_mask = (depth_model != 0) * depth_nonobj_mask
        score_invisib =  calc_invisib_socre_from_map(<double *> depth_diff.data,
                                                     <double *> invisib_mask.data,
                                                     imsize_h, imsize_w,
                                                     0.015, percentile_thre, max_thre)

        score_tri = score + score_visib + score_invisib

        if score_tri < best_score_tri:
            best_score_tri = score_tri
            ret_t_tri = _t
            ret_R_tri = _R

    return ret_t, ret_R
