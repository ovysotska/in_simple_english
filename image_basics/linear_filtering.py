# Implement box filtering
# 1) As convolution
# 2) As separated kernel convolution
# 3) using integral images
# 4) rolling window sum
#
import numpy as np
import utils


def apply_box_filter(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size))
    res_img = utils.convolve(img, kernel)
    return res_img


def apply_box_filter_separable_kernel(img, kernel_size):
    # 1D row kernel
    kernel_row = np.ones((1, kernel_size))
    # 1D column kernel
    kernel_col = np.ones((kernel_size, 1))
    res_img = utils.convolve(img, kernel_row)
    res_img = utils.convolve(res_img, kernel_col)
    return res_img


def apply_box_filter_integral_image(img, kernel_size):
    return


def create_integral_image(img):
    '''TODO: add pictures and explanations for the a,b,c,x explanation'''
    int_img = np.zeros(img.shape)
    for i in range(int_img.shape[0]):
        for j in range(int_img.shape[1]):
            if i == 0 and j == 0:  # first element
                a = 0
                b = 0
                c = 0
            elif i == 0:  # first rows
                a = 0
                b = 0
                c = int_img[i, j - 1]
            elif j == 0:  # first column
                a = 0
                c = 0
                b = int_img[i - 1, j]
            else:
                a = int_img[i - 1, j - 1]
                b = int_img[i - 1, j]
                c = int_img[i, j - 1]
            x = img[i, j]
            int_img[i, j] = c + b - a + x
    return int_img


def get_integral_img_value(int_img, i, j):
    rows = int_img.shape[0]
    cols = int_img.shape[1]
    if i < 0 or j < 0:
        return 0
    if i >= rows and j < cols:
        return int_img[rows - 1, j]
    if i < rows and j >= cols:
        return int_img[i, cols - 1]
    if i >= rows and j >= cols:
        return int_img[rows - 1, cols - 1]
    return int_img[i, j]


def convolve_integral_image(img, kernel_size):
    int_img = create_integral_image(img)
    res_img = np.zeros(img.shape)
    kh = kernel_size  # kernel height
    kw = kernel_size  # kernel width
    for i in range(res_img.shape[0]):
        for j in range(res_img.shape[1]):
            kernel_br_row = i + kh // 2  # kernel_bottom right corner row
            kernel_br_col = j + kw // 2  # kernel_bottom right corner col

            d = get_integral_img_value(
                int_img, kernel_br_row - kh, kernel_br_col - kw)
            e = get_integral_img_value(
                int_img, kernel_br_row - kh, kernel_br_col)
            f = get_integral_img_value(
                int_img, kernel_br_row, kernel_br_col - kw)
            x = get_integral_img_value(int_img, kernel_br_row, kernel_br_col)

            res_img[i, j] = x - f - e + d
    return res_img
