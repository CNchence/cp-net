
import random
import numpy as np
import cv2


def add_noise(src):
    row,col,ch = src.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss_img = src + gauss
    return gauss_img

def calc_quaternion(rot):
    quat = np.zeros(4)
    quat[0] = (rot[0,0] + rot[1,1] + rot[2,2] + 1.0) / 4.0
    quat[1] = np.sign(rot[2,1] - rot[1,2]) * (rot[0,0] - rot[1,1] - rot[2,2] + 1.0) / 4.0
    quat[2] = np.sign(rot[0,2] - rot[2,0]) * (-rot[0,0] + rot[1,1] - rot[2,2] + 1.0) / 4.0
    quat[3] = np.sign(rot[1,0] - rot[0,1]) * (-rot[0,0] - rot[1,1] + rot[2,2] + 1.0) / 4.0
    return quat

def rpy_param(rot):
    return np.array([rot[0,0], rot[1,0], rot[2,0], rot[2,1], rot[2,2]])

def calc_rpy(rot, eps=10e-5):
    # r-p-y eular angle
    # return array : [tan_x, sin_y, tan_z]
    #
    if rot[2,2] != 0 :
        tan_x = rot[2,1] / rot[2,2]
    else:
        tan_x = rot[2,1] / (rot[2,2] + eps)
    sin_y = - rot[2,0]
    if rot[0,0] != 0 :
        tan_z = rot[1,0] / rot[0,0]
    else:
        tan_z = rot[1,0] / (rot[0,0] + eps)
    return np.array([tan_x, sin_y, tan_z])


# depth inpainting (inpaint Nan region)
def depth_inpainting(img_depth):
    img_depth_nan_mask = (img_depth != img_depth)
    depth_max = img_depth[img_depth==img_depth].max()
    depth_min = img_depth[img_depth==img_depth].min()
    coeff = 255 /(depth_max - depth_min)

    img_depth_shift = (img_depth - depth_min) * coeff
    img_depth_shift[img_depth_shift != img_depth_shift] = 0
    img_depth_shift[img_depth_shift > 255] = 255
    img_depth_shift[img_depth_shift < 0] = 0

    img_fill = cv2.inpaint(np.round(img_depth_shift).astype(np.uint8), img_depth_nan_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    img_fill = img_fill.astype(np.float32) / coeff + depth_min

    ret_img = img_depth
    ret_img[ret_img != ret_img] = 0
    ret_img = img_fill * img_depth_nan_mask + ret_img * (1 - img_depth_nan_mask)
    return ret_img

def roi_kernel_size(img_depth, m=1.0):
    return m / img_depth
