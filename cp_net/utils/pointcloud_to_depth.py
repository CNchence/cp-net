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

    idx = np.argsort(zs)

    # render depth
    img_depth = np.zeros(img_size[::-1])
    img_depth[ys[idx], xs[idx]] = zs[idx]

    # refine = True

    # if refine == True:
    #     grid = np.arange(5) - 1
    #     mesh_x, mesh_y = np.meshgrid(grid, grid)
    #     for dx, dy in zip(mesh_x.ravel(), mesh_y.ravel()):
    #         mask = (img_depth[ys[idx] + dy, xs[idx] + dx] - zs[idx] > 0.005)
    #         img_depth[ys[idx][mask] + dy ,xs[idx][mask] + dx] = 0

    # kernel = np.ones((5,5),np.uint8)
    # img_depth_near_half = img_depth * (img_depth < np.median(img_depth[img_depth > 0]))
    # img_depth = cv2.morphologyEx(img_depth_near_half, cv2.MORPH_CLOSE, kernel)

    return img_depth
