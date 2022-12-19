import pathlib

import numpy as np
import numpy.linalg as la
import scipy.signal as signal
import matplotlib.pyplot as plt

import cv2



def convolve(image, kernel):
    return signal.convolve2d(
        image, kernel, mode='same',
        boundary='fill', fillvalue=0)


def search_diff(im1, im2, window_size, disparity_range, ssd):
    assert im1.shape == im2.shape
    assert window_size % 2 == 1
    h, w = im1.shape
    # kernel = np.ones((window_size, window_size))
    kernel_r = np.ones((1, window_size))
    kernel_c = np.ones((window_size, 1))
    dis_map = np.zeros_like(im1, dtype=np.int32)
    score_map = np.ones_like(im1) * 1e20
    for d in range(disparity_range[0], disparity_range[1]):
        r_start, r_end = max(0, d), min(w, w + d)
        l_start, l_end = r_start - d, r_end - d
        diff = np.abs(im1[:, l_start:l_end] - im2[:, r_start:r_end])
        if ssd:
            diff = diff ** 2
        score = convolve(diff, kernel_r)
        score = convolve(score, kernel_c)
        should_update = score < score_map[:, l_start:l_end]
        dis_map[:, l_start:l_end] = np.where(
            should_update, d, dis_map[:, l_start:l_end])
        score_map[:, l_start:l_end] = np.where(
            should_update, score, score_map[:, l_start:l_end])
    return dis_map


def search_ncc(im1, im2, window_size, disparity_range):
    assert im1.shape == im2.shape
    assert window_size % 2 == 1
    h, w = im1.shape
    # kernel = np.ones((window_size, window_size))
    kernel_sum_r = np.ones((1, window_size))
    kernel_sum_l = np.ones((window_size, 1))
    # mean filter: 1 / window_size ^ 2
    kernel_mean_r = np.ones((1, window_size)) / window_size
    kernel_mean_l = np.ones((window_size, 1)) / window_size

    def window_wise_mean(patch):
        ret = convolve(patch, kernel_mean_r)
        return convolve(ret, kernel_mean_l)

    def window_wise_sum(patch):
        ret = convolve(patch, kernel_sum_r)
        return convolve(ret, kernel_sum_l)

    def point_wise_zm_mul_sum(a, b, mean_a, mean_b):
        # for every window, return sum(\bar a * \bar b)
        a_b = window_wise_sum(a * b)
        am_b = window_wise_sum(b) * mean_a
        a_bm = window_wise_sum(a) * mean_b
        am_bm = mean_a * mean_b * (window_size ** 2)
        return a_b + am_bm - am_b - a_bm

    def patched_zncc(patch1, patch2):
        mean_1 = window_wise_mean(patch1)
        mean_2 = window_wise_mean(patch2)
        corr = point_wise_zm_mul_sum(patch1, patch2, mean_1, mean_2)
        norm1 = point_wise_zm_mul_sum(patch1, patch1, mean_1, mean_1)
        norm2 = point_wise_zm_mul_sum(patch2, patch2, mean_2, mean_2)
        z = np.min(norm2*norm1)
        if z<0:
            print(norm2.min(), norm1.min())
        return corr / np.sqrt(norm1 * norm2)

    dis_map = np.zeros_like(im1, dtype=np.int32)
    score_map = np.ones_like(im1) * (-10)
    for d in range(disparity_range[0], disparity_range[1]):
        r_start, r_end = max(0, d), min(w, w + d)
        l_start, l_end = r_start - d, r_end - d
        score = patched_zncc(im1[:, l_start:l_end], im2[:, r_start:r_end])
        should_update = score > score_map[:, l_start:l_end]
        dis_map[:, l_start:l_end] = np.where(
            should_update, d, dis_map[:, l_start:l_end])
        score_map[:, l_start:l_end] = np.where(
            should_update, score, score_map[:, l_start:l_end])
    return dis_map




def pipeline(im1, im2, metric):
    im1 = cv2.imread(im1,
                     cv2.IMREAD_GRAYSCALE) / 255
    im2 = cv2.imread(im2,
                     cv2.IMREAD_GRAYSCALE) / 255
    window_size = 15
    disparity_range = (-50, 1)
    if metric == 'SSD':
        dis_map = search_diff(im1, im2, window_size,
                              disparity_range, ssd=True)
    elif metric == 'SAD':
        dis_map = search_diff(im1, im2, window_size,
                              disparity_range, ssd=False)
    else:
        dis_map = search_ncc(im1, im2, window_size, disparity_range)
    print(np.min(dis_map), np.max(dis_map))
    plt.figure()
    dis_map = np.where(dis_map < -40, 0, dis_map)
    plt.imshow(np.abs(dis_map)/50, cmap='gray')
    plt.show()
    return dis_map


if __name__ == '__main__':
    pipeline('tsukuba1.jpeg', 'tsukuba2.jpeg', 'ncc')
