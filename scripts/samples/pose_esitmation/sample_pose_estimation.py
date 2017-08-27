#!/usr/bin/env python

from matplotlib import pylab as plt
import cv2
import numpy as np
import time

from cp_net.utils.pointcloud_to_depth import pointcloud_to_depth
from cp_net.functions import model_base_consensus_accuracy

from cp_net.utils import model_base_ransac_estimation
# from cp_net import model_base_ransac_estimation


# inference outputs
y = np.load("sample_data/mbr_y.npy")
obj_mask = np.load("sample_data/mbr_obj_mask.npy")

# rgb-d
rgb = cv2.imread("sample_data/mbr_rgb.jpg")
rgb = rgb.astype(np.float32)
depth = np.load("sample_data/mbr_depth.npy")
pc = np.load("sample_data/mbr_pc.npy")

# object model
model = np.load("sample_data/mbr_model.npy")
# true data
t_cp = np.load("sample_data/mbr_cp.npy")
t_rot = np.load("sample_data/mbr_rot.npy")

# camera parameter
k = np.load("sample_data/mbr_k.npy")

im_size = rgb.shape[:2]

# output fig
fig = plt.figure(figsize=(3, 2))



## pose estimation not using Cython
t1 = time.time()
ret_t, ret_R = model_base_consensus_accuracy.model_base_ransac_estimatation \
               (pc, y, pc, model, depth, k, t_cp, obj_mask)
t1 = time.time() - t1

## pose estimation using Cython
t2 = time.time()
ret_t2, ret_R2 = model_base_ransac_estimation.model_base_ransac_estimation_cy(pc, y, model,
                                                                              depth, k, obj_mask,
                                                                              im_size)
t2 = time.time() - t2


import pstats, cProfile

cProfile.runctx("model_base_ransac_estimation.model_base_ransac_estimation_cy(pc, y, model, depth, k, obj_mask, im_size, n_ransac=1000)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

## clac true pose
t_ren = pointcloud_to_depth((np.dot(t_rot, model) + t_cp[:, np.newaxis]).transpose(1,0),
                            k, im_size[::-1])

p_ren = pointcloud_to_depth((np.dot(ret_R, model) + ret_t[:, np.newaxis]).transpose(1,0),
                            k, im_size[::-1])

c_ren = pointcloud_to_depth((np.dot(ret_R2, model) + ret_t2[:, np.newaxis]).transpose(1,0),
                            k, im_size[::-1])


print "ransac without cython : ", t1
print "ransac with cython : ", t2

ax = fig.add_subplot(2, 3, 1)
ax.imshow(rgb[:,:,::-1] / 255.0)
ax = fig.add_subplot(2, 3, 2)
ax.imshow(depth)
ax = fig.add_subplot(2, 3, 3)
ax.imshow(obj_mask)
ax = fig.add_subplot(2, 3, 4)
ax.imshow(rgb[:,:,::-1] /255.0 * 0.5 + t_ren[:,:,np.newaxis] * 0.4)
ax = fig.add_subplot(2, 3, 5)
ax.imshow(rgb[:,:,::-1] /255.0 * 0.5 + p_ren[:,:,np.newaxis] * 0.4)
ax = fig.add_subplot(2, 3, 6)
ax.imshow(rgb[:,:,::-1] /255.0 * 0.5 + c_ren[:,:,np.newaxis] * 0.4)

plt.show()
