import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


class ImageAugmenter(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq= iaa.Sequential([
            iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 1.0
            sometimes(
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5)),
            sometimes(
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
            sometimes(
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.0))), # emboss images
            sometimes(
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
            sometimes(iaa.Grayscale(alpha=(0.0, 1.0))),
            sometimes(
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5)),
            sometimes(
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20))),
            sometimes(
                    # randomly remove up to 10% of the pixels
                    iaa.Dropout((0.01, 0.1), per_channel=0.5)),
        ], random_order=True)

    def augment(self, img):
        return self.seq.augment_images(np.expand_dims(img.astype(np.uint8), axis=0))[0].astype(img.dtype)


class TransformAugmenter(object):
    def __init__(self, cval=0):
        sometimes = lambda aug: iaa.Sometimes(0.4, aug)
        self.seq= iaa.Sequential([
            # sometimes(
            #     iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            iaa.Affine(scale={"x": (0.75, 2.0), "y": (0.75, 2.0)},
                       rotate=(-30, 30),
                       shear=(-16, 16),
                       order=0,
                       cval=cval),
        ], random_order=True)
        self.seq_det = self.seq.to_deterministic()
    def augment(self, img):
        return self.seq.augment_images(np.expand_dims(img.astype(np.uint8), axis=0))[0].astype(img.dtype)
    def augment_deterministic(self, img):
        return self.seq_det.augment_images(np.expand_dims(img.astype(np.uint8), axis=0))[0].astype(img.dtype)
    def deterministic_update(self):
        self.seq_det = self.seq.to_deterministic()

