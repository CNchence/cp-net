
import random
import numpy as np
import cv2


def add_noise(src, sigma=5):
    src_tmp = src.copy()
    row,col,ch = src.shape
    mean = 0.0
    gauss = np.random.normal(mean, sigma, (row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss_img = src + gauss
    gauss_img[gauss_img > 255] = 255
    gauss_img[gauss_img < 0] = 0
    return gauss_img.astype(np.uint8)

def avaraging(src, ksize=5):
    average_square = (ksize, ksize)
    return cv2.blur(src, average_square)


def gamma_augmentation(src, gamma=0.75):
    LUT = np.arange(256, dtype = 'uint8' )
    for i in range(256):
        LUT[i] = 255 * pow(float(i) / 255, 1.0 / gamma)
    return cv2.LUT(src.astype('uint8'), LUT)


def salt_pepper_augmentation(src, sp_rate=0.5, amount=0.004):
    row,col,ch = src.shape
    sp_img = src.copy()
    # salt noise
    num_salt = np.ceil(amount * src.size * sp_rate)
    coords = [np.random.randint(0, i-1 , int(num_salt)) for i in src.shape]
    sp_img[coords[:-1]] = (255,255,255)
    # pepper noise
    num_pepper = np.ceil(amount* src.size * (1. - sp_rate))
    coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in src.shape]
    sp_img[coords[:-1]] = (0,0,0)

    return sp_img

class ContrastAugmentation:
    def __init__(self, min_table=50, max_table=205):
        # Look Up Table(LUT)
        diff_table = max_table - min_table
        self.LUT_HC = np.arange(256, dtype = 'uint8')
        self.LUT_LC = np.arange(256, dtype = 'uint8')

        # high contrast LUT
        for i in range(0, min_table):
            self.LUT_HC[i] = 0
        for i in range(min_table, max_table):
            self.LUT_HC[i] = 255 * (i - min_table) / diff_table
        for i in range(max_table, 255):
            self.LUT_HC[i] = 255

        # low contrast LUT
        for i in range(256):
            self.LUT_LC[i] = min_table + i * (diff_table) / 255

    def high_contrast(self, src):
        return cv2.LUT(src.astype('uint8'), self.LUT_HC)

    def low_contrast(self, src):
        return cv2.LUT(src.astype('uint8'), self.LUT_LC)


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
