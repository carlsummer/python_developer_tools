# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/20/2021 4:09 PM
# @File:APR
import copy

from PIL import Image, ImageOps, ImageEnhance
import io
import random
from PIL import Image
import numpy as np

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, level, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, level, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level, _):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, _):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, _):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, img_size=(32, 32)):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(img_size,
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def shear_y(pil_img, level, img_size=(32, 32)):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(img_size,
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)


def translate_x(pil_img, level, img_size=(32, 32)):
    level = int_parameter(sample_level(level), img_size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(img_size,
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)


def translate_y(pil_img, level, img_size=(32, 32)):
    level = int_parameter(sample_level(level), img_size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(img_size,
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level, _):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


class APRecombination(object):
    def __init__(self,img_size=(32,32)):
        self.img_size = img_size
        self.aug_list = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y
        ]

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''
        op = np.random.choice(self.aug_list)
        x = op(x, 3, self.img_size)

        # p = random.uniform(0, 1)
        # if p > 0.5:
        #     return x

        x_aug = x.copy()
        op = np.random.choice(self.aug_list)
        x_aug = op(x_aug, 3, self.img_size)

        x = np.array(x).astype(np.uint8)
        x_aug = np.array(x_aug).astype(np.uint8)

        fft_1 = np.fft.fftshift(np.fft.fftn(x))
        fft_2 = np.fft.fftshift(np.fft.fftn(x_aug))

        abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)
        abs_2, angle_2 = np.abs(fft_2), np.angle(fft_2)

        fft_1 = abs_1 * np.exp((1j) * angle_2)
        fft_2 = abs_2 * np.exp((1j) * angle_1)

        p = random.uniform(0, 1)

        if p > 0.5:
            x = np.fft.ifftn(np.fft.ifftshift(fft_1))
        else:
            x = np.fft.ifftn(np.fft.ifftshift(fft_2))

        x = x.astype(np.uint8)
        x = Image.fromarray(x)

        return x
    
if __name__ == '__main__':
    img_path = r'/home/zengxh/medias/data/ext/creepageDistance/lab_datasets/lr/org/6319938267001088_0_l..jpg'
    PIL_origin_image = Image.open(img_path)

    PIL_origin_image_copy = copy.deepcopy(PIL_origin_image)
    equalizeimg = equalize(PIL_origin_image_copy,1,1)
    equalizeimg.save("equalizeimg.jpg")

    PIL_origin_image_copy = copy.deepcopy(PIL_origin_image)
    apr = APRecombination(PIL_origin_image.size)
    apr_img = apr(PIL_origin_image_copy)
    apr_img.save("apr_img.jpg")