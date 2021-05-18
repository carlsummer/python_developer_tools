import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from scipy import signal


def compute_light_percentage(image, thresh):
    # 计算亮度大于某个阈值的占比
    assert len(image.shape) == 2
    img_h, img_w = image.shape[:2]
    dark_area = (image >= thresh).sum()
    area = img_h * img_w
    return 1.0 * dark_area / area


def compute_dark_percentage(image, thresh):
    # 计算亮度黑于某个阈值的占比
    assert len(image.shape) == 2
    img_h, img_w = image.shape[:2]
    dark_area = (image <= thresh).sum()
    area = img_h * img_w
    return 1.0 * dark_area / area


def avg_filt_1d(vector, kernel_size):
    # 对向量进行卷积平滑操作
    """
    Args:
            vector (array): Input vector
            kernel_size (int): kernel size of average filtering
    Returns:
            vector (array): Smoothed vector
    """
    kernel = np.ones([kernel_size]) / kernel_size
    return np.convolve(vector, kernel, mode='same')


def to_gray(image):
    # 将图片转灰度图
    if len(image.shape) == 2:
        pass
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise Exception(('Image shape error: {}').format(image.shape))
    return image


def padding(image, pad_h, pad_w):
    # 给图片添加padding
    image_shape = np.array(image.shape)
    h, w = image_shape[:2]
    pad_shape = image_shape.copy()
    pad_shape[0] = pad_h
    pad_shape[1] = pad_w
    image_pad = np.zeros(pad_shape, dtype=image.dtype)
    image_pad[:h, :w, ...] = image
    return image_pad


def get_median(vector):
    # 获取向量的中位数
    assert len(vector) >= 1
    return np.sort(vector)[int(len(vector) / 2)]


def _listornparray_2_plt(row_histogram):
    """将list或者nparray画成图 画直方图"""
    lenth = len(row_histogram)
    plt.figure(1)
    hist_x = np.linspace(0, lenth - 1, lenth)
    plt.title("row_histogram")
    plt.rcParams['figure.figsize'] = (lenth, 8)  # 单位是inches

    x_major_locator = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  # x轴按1刻度显示

    plt.plot(hist_x, row_histogram)
    higher_q = np.max(row_histogram) * (5 / 8)
    plt.plot([0, lenth - 1], [higher_q, higher_q])
    num_peak_3 = signal.find_peaks(row_histogram, distance=1)  # distance表极大值点的距离至少大于等于10个水平单位
    for ii in range(len(num_peak_3[0])):
        if (row_histogram[num_peak_3[0][ii]] > np.mean(row_histogram)) and (
                row_histogram[num_peak_3[0][ii]] != np.max(row_histogram)
        ) and (
                num_peak_3[0][ii] > 10
        ):
            plt.plot(num_peak_3[0][ii], row_histogram[num_peak_3[0][ii]], '*', markersize=10)
            plt.axvline(num_peak_3[0][ii])
            print(num_peak_3[0][ii])

    plt.savefig("row_histogram_peak.jpg")
    plt.close()


def graytoprofile(gray):
    # 边缘检测
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(gray, kernel)
    erode = cv2.erode(gray, kernel)
    profile = cv2.absdiff(dilate, erode)
    return profile


def sqrt_var_hist(histogram):
    # 求直方图的方差开根号
    return np.sqrt(np.var(histogram))


def remove_baseline_shift(vector, step=10):
    """
    Args:
            vector (array): Numpy vector with shape (num_points,)
            step (int): Kernel size of average filtering
    Returns:
            vector (array): Vector with baseline shift removed
    """
    if step % 2 == 0:
        step += 1
    vector_smoothed = signal.medfilt(vector, step)
    avg_smoothed = avg_filt_1d(vector, kernel_size=step)
    temp_step = int(len(vector) / 40)
    center_ind = int(len(vector) / 2)
    x1 = center_ind - temp_step
    x2 = center_ind + temp_step
    vector_smoothed[x1:x2] = avg_smoothed[x1:x2]
    return vector - vector_smoothed


def plt_show_cv2(image):
    # plt 显示 opencv 格式的图片
    plt.figure(1)
    if len(image.shape) == 2:
        ## 把单通道的灰度图转换成三通道的灰度图
        image = np.concatenate(
            (np.expand_dims(image, axis=2), np.expand_dims(image, axis=2), np.expand_dims(image, axis=2)), axis=-1)
        # plt.imshow(image,cmap='gray',interpolation='bicubic')#显示灰度图
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def compress_image(rgbImg_show2,ratio):
    # 压缩图片
    return cv2.resize(rgbImg_show2, (0, 0), fx=ratio, fy=ratio)

def opencvToBytes(frame):
    # opencv 转字节流
    return cv2.imencode(".jpg",frame)[1].tobytes()

def crop_image_like(image, crop_bbox):
    """
    获取bbox对应的图片
    :param image:
    :param crop_bbox:
    :return:
    """
    xmin, ymin, xmax, ymax = crop_bbox
    if xmax == -1:
        xmax = image.shape[1] + 1
    if ymax == -1:
        ymax = image.shape[0] + 1
    image_crop = image[ymin:ymax + 1, xmin:xmax + 1, ...]
    return image_crop


def cv2ToTorch(img):
    """单张cv2的格式转torch 的tensor"""
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to cxhxw
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = torch.from_numpy(img)
    return img


def group_consecutive(a, trod=10):
    """将连续的值分组
    [1, 2, 3, 7, 8, 9, 10, 100, 101, 102, 103]->[array([1, 2, 3]), array([ 7,  8,  9, 10]), array([100, 101, 102, 103])]
    """
    return np.split(a, np.where(np.diff(a) >= trod)[0] + 1)


"""
滤波器
left_rectangle = cv2.blur(left_rectangle, (3, 3))
left_rectangle = cv2.GaussianBlur(left_rectangle, (5, 5), 0)
left_rectangle = cv2.medianBlur(left_rectangle, 3)
left_rectangle = cv2.bilateralFilter(left_rectangle, 9, 75, 75)

求突然上升很大值和突然下降很大值的位置
col_histogram = np.sum(subImg2, axis=0)
nInteral = 20
cols_minInterval = subImg2.shape[1] // nInteral
## left
left_scope_diff = [np.sum(col_histogram[idx - nInteral:idx].astype(np.int)) - np.sum(
    col_histogram[idx:idx + nInteral].astype(np.int))
                   for idx in range(nInteral, cols_minInterval - nInteral)]
left_edge = nInteral + np.argmax(left_scope_diff)
## right
right_scope_diff = [np.sum(col_histogram[idx - nInteral:idx].astype(np.int)) - np.sum(
    col_histogram[idx:idx + nInteral].astype(np.int))
                    for idx in range(subImg2.shape[1] - cols_minInterval + nInteral, subImg2.shape[1] - nInteral)]
right_edge = subImg2.shape[1] - cols_minInterval + nInteral + np.argmin(right_scope_diff)
"""
