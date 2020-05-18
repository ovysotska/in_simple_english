import numpy as np


def convolve(img, kernel):
    ''' performs convolution of the kernel with  clamping
    for border conditions '''

    rows = img.shape[0]
    cols = img.shape[1]
    res = np.zeros((img.shape[0], img.shape[1]))
    # going over every pixel of the original image
    for r in range(0, rows):
        for c in range(0, cols):
            # print("r:",r, "c:",c)
            s = 0
            # going over pixels of the kernel
            kernel_rows = kernel.shape[0]
            kernel_cols = kernel.shape[1]
            kernel_height_half = kernel_rows // 2
            kernel_width_half = kernel_cols // 2
            for kr in range(-kernel_height_half, kernel_height_half + 1):
                for kc in range(-kernel_width_half, kernel_width_half + 1):
                    # border conditions
                    if ((r + kr) < 0 or (r + kr) >= rows or (c + kc) < 0 or (c + kc) >= cols):
                        continue
#                     print("kern. r:", r+kr, "c:", c+kc)
                    s = s + kernel[kr + kernel_height_half, kc + kernel_width_half] * img[r + kr, c + kc]
            res[r][c] = s

    return res


def convolve_reduce(img, kernel):
    ''' reduce image size, use only valid pixels'''
    rows = img.shape[0]
    cols = img.shape[1]
    kernel_size = kernel.shape[0]
    kernel_center = kernel_size // 2
    size_diff = 2 * kernel_center

    res = np.zeros((img.shape[0] - size_diff, img.shape[1] - size_diff))
    res_rows = res.shape[0]
    res_cols = res.shape[1]
    # going over every pixel of the original image
    for r in range(0, rows):
        for c in range(0, cols):
            # check if the result is outside res matrix
            if r >= res_rows or c >= res_cols:
                continue
            sum = 0
            for kr in range(0, kernel_size):
                for kc in range(0, kernel_size):
                    sum += img[r + kr, c + kc] * kernel[kr, kc]
            res[r, c] = sum
    return res


def find_derivative_x(img):
    dx_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return convolve(img, dx_kernel)


def find_derivative_y(img):
    dy_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return convolve(img, dy_kernel)


def get_Binomial_coefficients(coeff, iter, n):
    if n == 0:
        return 1
    if iter == n:
        return coeff
    new_layer_size = coeff.shape[0] + 1
    new_layer = np.zeros(new_layer_size)
    new_layer[0] = 1
    for i in range(coeff.shape[0] - 1):
        new_layer[i + 1] = coeff[i] + coeff[i + 1]
    new_layer[-1] = 1
    return get_Binomial_coefficients(new_layer, iter + 1, n)


def generate_Binomial_kernel(kernel_size):
    if kernel_size % 2 == 0:
        print("[ERROR][generate_Gaussian_kernel] Kernel_size should be an uneven number")
        return None

    iter_start = -1  # always start with -1 if start array empty
    start_array = np.array([])
    n = kernel_size - 1
    binom_1d = get_Binomial_coefficients(start_array, iter_start, n)
    kernel = np.outer(binom_1d, binom_1d)
    return kernel / np.sum(kernel)
