# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/27/2021 1:15 PM
# @File:骨架提取
#https://blog.csdn.net/jiiiiiaaaa/article/details/86626452

# 导入库
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io

# 将图像转为灰度图像
from PIL import Image

img = Image.open("SkeletonOrigin.png").convert('L')
img.save('greyscale.png')

# 读取灰度图像
Img_Original = io.imread('greyscale.png')

# 对图像进行预处理，二值化
from skimage import filters
from skimage.morphology import disk

# 中值滤波
Img_Original = filters.median(Img_Original, disk(5))
# 二值化
BW_Original = Img_Original < 235

# 定义像素点周围的8邻域
#                P9 P2 P3
#                P8 P1 P4
#                P7 P6 P5

def neighbours(x, y, image):
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9


# 计算邻域像素从0变化到1的次数
def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2,P3,...,P8,P9,P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)


# Zhang-Suen 细化算法
def zhangSuen(image):
    Image_Thinned = image.copy()  # Making copy to protect original image
    changing1 = changing2 = 1
    while changing1 or changing2:  # Iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0: Point P1 in the object regions
                        2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                        transitions(n) == 1 and  # Condition 2: S(P1)=1
                        P2 * P4 * P6 == 0 and  # Condition 3
                        P4 * P6 * P8 == 0):  # Condition 4
                    changing1.append((x, y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0
                        2 <= sum(n) <= 6 and  # Condition 1
                        transitions(n) == 1 and  # Condition 2
                        P2 * P4 * P8 == 0 and  # Condition 3
                        P2 * P6 * P8 == 0):  # Condition 4
                    changing2.append((x, y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned

# 对染色体图像应用Zhang-Suen细化算法
BW_Skeleton = zhangSuen(BW_Original)
import numpy as np
BW_Skeleton = np.invert(BW_Skeleton)

# 显示细化结果
fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax.ravel()
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Original binary image')
ax1.axis('off')
ax2.imshow(BW_Skeleton, cmap=plt.cm.gray)
ax2.set_title('Skeleton of the image')
ax2.axis('off')
plt.savefig('thinned.png')
plt.show()

# 生成骨架图像并保存
Skeleton = np.ones((BW_Skeleton.shape[0], BW_Skeleton.shape[1]), np.uint8) * 255  # 生成一个空灰度图像
BW_Skeleton = BW_Skeleton + 0
for i in range(BW_Skeleton.shape[0]):
    for j in range(BW_Skeleton.shape[1]):
        if BW_Skeleton[i][j] == 0:
            Skeleton[i][j] = 0

plt.axis('off')
plt.imshow(Skeleton, cmap=plt.cm.gray)

import imageio

imageio.imwrite('Skeleton.png', Skeleton)

# 利用opencv在染色体图像上画出中轴线
import numpy as np
import cv2 as cv

img = cv.imread('Skeleton.png')
binaryImg = cv.Canny(img, 100, 200)
h = cv.findContours(binaryImg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours = h[1]

