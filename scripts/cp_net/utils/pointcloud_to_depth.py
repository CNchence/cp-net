import numpy as np
import cv2


def pointcloud_to_depth(pc, K, img_size):
    # input
    # pc : input point cloud
    # K : camera parameters
    # (K = [[f * k_x,       0,  o_x],
    #       [0,       f * k_y,  o_y],
    #       [0,             0,    0]]
    #  f: focus)
    # img_size : (img_width, img_height)

    xs = np.round(pc[:, 0] * K[0, 0] / pc[:, 2] + K[0, 2])
    ys = np.round(pc[:, 1] * K[1, 1] / pc[:, 2] + K[1, 2])

    inimage_mask = (xs > 0) * (xs < img_size[0]) * \
                   (ys > 0) * (ys < img_size[1])

    xs = xs[inimage_mask].astype(np.int32)
    ys = ys[inimage_mask].astype(np.int32)
    zs = pc[:, 2][inimage_mask]

    idx = np.argsort(zs)[::-1]

    # render depth
    img_depth = np.zeros((img_size[0], img_size[1]))
    img_depth[ys[idx], xs[idx]] = zs[idx]

    return img_depth
