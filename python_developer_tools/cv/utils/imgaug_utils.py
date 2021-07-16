# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/14/2021 4:20 PM
# @File:imgaug_utils
from imgaug import augmenters as iaa  # 引入数据增强的包
import cv2

def imgaug_change_color(img):
    """imgaug库增强
    https://github.com/aleju/imgaug
    """
    seq = iaa.Sequential([
        # Sometimes是指指针对50%的图片做处理
        # iaa.Sometimes(
        #     0.5,
        #     iaa.Add((-40, 40)),
        #     iaa.Multiply((0.5, 1.5))
        # ),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
        iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
        iaa.MultiplyHue((0.5, 1.5)),
        iaa.MultiplySaturation((0.5, 1.5)),
        iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        iaa.RemoveSaturation(),
        iaa.ChangeColorTemperature((1100, 10000)),
        # iaa.KMeansColorQuantization(),
        # iaa.UniformColorQuantization()
        # 使用随机组合上面的数据增强来处理图片
    ], random_order=True)
    images_aug = seq.augment_image(img)  # 是处理多张图片augment_images
    return images_aug

if __name__ == '__main__':
    img_o = cv2.imread("/home/zengxh/datasets/creepageDistance/images/7594539227002070-1_0_t.jpg")

    for i in range(30):
        img = imgaug_change_color(img_o)
        cv2.imwrite("/home/zengxh/workspace/lcnn/imgaug/sdf{}.jpg".format(i),img)