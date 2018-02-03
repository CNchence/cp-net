
import numpy as np
import cv2

def estimate_visib_region(depth, depth_ref):
    visib_mask = np.logical_or((depth < depth_ref),
                               np.logical_and((depth_ref == 0), (depth != 0)))
    return visib_mask


def random_affine(depth, pos, imsize, K, edge_offset=5, min_z=0.5, max_z=1.5):
    # translate and scaling
    x = np.random.randint(edge_offset, imsize[0] - edge_offset) - imsize[0] // 2
    y = np.random.randint(edge_offset, imsize[1] - edge_offset) - imsize[1] // 2
    z = min_z + np.random.rand() * (max_z - min_z)
    trans_pos = np.array([x * z / K[0, 0], y * z / K[1, 1], z])
    depth = depth + (z - pos[2])
    # Affine transform(scaling, translate, resize)
    g_x = K[0,2] + pos[0] * K[0,0] / pos[2]
    g_y = K[1,2] + pos[1] * K[1,1] / pos[2]
    Mat = np.float32([[pos[2] / z, 0, x + K[0, 2] - g_x * pos[2]/z],
                      [0, pos[2] / z, y + K[1, 2] - g_y * pos[2]/z]])
    return depth, Mat, trans_pos


def auto_context_data(concat_rgb, concat_depth, concat_mask, concat_cp, concat_ocp,
                      position, rotation, K, obj_order,
                      rgb, depth, mask, points, pos, rot,
                      edge_offset=5, min_z=0.5, max_z=1.5):
    img_height, img_width = concat_rgb.shape[:2]
    out_height, out_width = concat_depth.shape[:2]
    cp = pos[np.newaxis, np.newaxis, :] - points
    depth, M0, trans_pos = random_affine(depth, pos, (img_width, img_height), K)
    M = M0.copy() * (out_width // img_width)
    rgb = cv2.warpAffine(rgb, M0, (img_width, img_height))
    depth = cv2.warpAffine(depth, M, (out_width, out_height))
    mask = cv2.warpAffine(mask, M, (out_width, out_height))
    cp = cv2.warpAffine(cp, M, (out_width, out_height))
    # visib mask
    visib_mask = estimate_visib_region(depth, concat_depth)
    visib_mask = visib_mask * mask
    visib_mask_resize = cv2.resize(visib_mask, (img_width, img_height))
    visib_mask_resize = visib_mask_resize.astype(np.bool)
    visib_mask = visib_mask.astype(np.bool)
    # masking
    concat_rgb[visib_mask_resize, :] = rgb[visib_mask_resize, :]
    concat_depth[visib_mask] = depth[visib_mask]
    cp = (cp * visib_mask[:,:, np.newaxis]).transpose(2,0,1)
    cp[cp != cp] = 0
    concat_ocp[obj_order] = np.dot(rot.T, - cp.reshape(3, -1)).reshape(cp.shape)
    concat_cp[obj_order] = cp
    concat_mask[obj_order] = visib_mask
    # pose
    position[obj_order] = trans_pos
    rotation[obj_order] = rot

    return concat_rgb, concat_depth, concat_mask, concat_cp, concat_ocp, position, rotation
