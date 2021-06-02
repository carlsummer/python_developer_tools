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


#   阿里 的代码 start
def detect_dark_to_light(vector, gray_thresh):
    u"""
            - 从左检测一个向量从暗到亮的跳变点
            Args:
                    vector (array): Vector with shape (N,)
                    gray_thresh (float): Gray threshold
            Returns:
                    index (int): index of the boundary

            Examples:
                    Assume len(vector)=100, gray_thresh=80
                    1. Normal vector
                    vector: [ 10,  10,  10, 200, 200, 200...] -> return 3
                    2. Noisy vector
                    vector: [ 10, 200,  10,  10, 200, 200...] -> return 4
                    3. All dark vector
                    vector: [ 10,  10,  10,  10,  10,  10...] -> return 100
                    4. All light vector
                    vector: [200, 200, 200, 200, 200, 200...] -> return 0
            """
    min_step = 5
    gray_sum = vector[:min_step].sum()
    if gray_sum / min_step >= gray_thresh:
        return 0
    num_points = len(vector)
    vector = np.concatenate([vector, vector[-(min_step - 1):]])
    for i in range(1, len(vector) - min_step + 1):
        gray_sum -= vector[(i - 1)]
        gray_sum += vector[(i + min_step - 1)]
        if gray_sum / min_step >= gray_thresh:
            return i

    return num_points


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


def compute_l1_dis_matrix(vector1, vector2):
    """
        Args:
                vector1 (array): Vector with shape (N,)
                vector2 (array): Vector with shape (M,)
        Returns:
                dis_matrix (array): Distance matrix with shape (N, M)
        """
    return np.abs(vector1[(Ellipsis, None)] - vector2[(None, Ellipsis)])


def compute_nms_mask_1d(vector, scores, distance_thresh):
    assert len(vector) == len(scores)
    dis_matrix = compute_l1_dis_matrix(vector, vector)
    dis_matrix += (np.eye(len(vector)) * 100000.0).astype(vector.dtype)
    order = scores.argsort()[::-1]
    mask = np.ones([len(vector)]).astype(np.bool)
    for ind in order:
        if mask[ind] == False:
            continue
        v = vector[ind]
        dis = dis_matrix[ind]
        mask[dis <= distance_thresh] = False

    return mask


def nms_1d(vector, scores, distance_thresh):
    mask = compute_nms_mask_1d(vector, scores, distance_thresh)
    return vector[mask]


def compute_local_minimum_mask(vector):
    """
        Args:
                vector (array)
        Returns:
                is_minimum (array): bool vector
        """
    if len(vector) <= 3:
        return np.zeros([len(vector)], dtype=np.bool)
    v = vector
    is_minimum = (v[1:-1] - v[:-2] <= 0) * (v[1:-1] - v[2:] <= 0)
    is_minimum = np.concatenate([is_minimum[0:1], is_minimum, is_minimum[-1:]])
    return is_minimum

def detect_negative_impulse_2(vector):
    """
        Args:
                vector (array): Input vector with shape (num_points,)
        Returns:
                inds ()
        """
    is_minimum = compute_local_minimum_mask(vector)
    mean_value = vector.mean()
    low_value = vector[vector.argsort()][:10].mean()
    thresh = low_value * 0.3 + mean_value * 0.7
    picked_inds = np.where((vector < thresh) * is_minimum)[0]
    return picked_inds

def detect_negative_impulse(vector, diff_thresh=3.0, nms_thresh=None, debug=False):
    # 检测负脉冲
    is_minimum = compute_local_minimum_mask(vector)
    if nms_thresh is not None:
        minimum_inds = np.where(is_minimum)[0]
        minimum_inds = nms_1d(minimum_inds, scores=-vector[minimum_inds], distance_thresh=nms_thresh)
        is_minimum[...] = False
        is_minimum[minimum_inds] = True
    center = get_median(vector)
    median_distance = get_median(np.abs(vector - center))
    v = vector[is_minimum]
    order = v.argsort()
    v_sort = np.sort(v)
    order_diff = v_sort[2:] - v_sort[:-2]
    if debug:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(order_diff)
        plt.plot(order_diff, '.')
        plt.plot([0, len(order_diff)], [5 * median_distance, 5 * median_distance])
        plt.subplot(2, 1, 2)
        plt.plot(vector)
        inds = np.where(is_minimum)[0]
        plt.plot(inds, vector[inds], '.')
        plt.show()
    impulse_inds = np.where(order_diff > diff_thresh)[0]
    if len(impulse_inds) == 0:
        return np.zeros([0])
    else:
        impulse_inds = impulse_inds[(impulse_inds < int(len(v) * 0.66))]
        if len(impulse_inds) == 0:
            return np.zeros([0])
        impulse_ind = impulse_inds[(-1)]
        if order_diff[impulse_ind] < 5:
            impulse_ind -= 1
        thresh = v_sort[impulse_ind]
        picked_inds = np.where((vector <= thresh) * is_minimum)[0]
        if debug:
            plt.figure(figsize=(10, 10))
            plt.subplot(3, 1, 1)
            plt.plot(order_diff)
            plt.plot(order_diff, '.')
            plt.plot([0, len(order_diff)], [diff_thresh, diff_thresh])
            plt.xlim([0, len(order_diff) - 1])
            plt.subplot(3, 1, 2)
            plt.plot(vector)
            inds = np.where(is_minimum)[0]
            plt.plot(inds, vector[inds], '.')
            plt.xlim([0, len(vector) - 1])
            plt.subplot(3, 1, 3)
            plt.plot(vector)
            plt.plot(picked_inds, vector[picked_inds], '.')
            plt.plot([0, len(vector)], [thresh, thresh])
            plt.xlim([0, len(vector) - 1])
            plt.show()
        if debug:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(order_diff)
            plt.plot(order_diff, '.')
            plt.subplot(2, 1, 2)
            plt.plot(vector)
            plt.plot(picked_inds, vector[picked_inds], '.')
            plt.plot([0, len(vector)], [thresh, thresh])
            plt.show()
        return picked_inds


# 阿里的代码end

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


def detect_best_drop_line(col_histogram):
    # 求突然上升很大值的位置
    """
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
    nInteral= 5
    # col_histogram = np.sum(vector, axis=0)
    cols_minInterval = col_histogram.shape[1] // nInteral
    left_scope_diff = [np.sum(col_histogram[idx - nInteral:idx].astype(np.int)) - np.sum(
        col_histogram[idx:idx + nInteral].astype(np.int))
                       for idx in range(nInteral, cols_minInterval - nInteral)]
    return nInteral + np.argmax(left_scope_diff)


