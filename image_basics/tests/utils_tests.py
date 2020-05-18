import unittest
import sys
sys.path.append("..")
import utils
import numpy as np
import numpy.testing as npt
import math


def proper_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def round_matrix(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = proper_round(arr[i, j])
    return arr

class TestGenerateBinomialKernel(unittest.TestCase):

    def test_kernel_size(self):
        kernel_size = 5
        expected_kernel = np.array([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]])
        kernel = utils.generate_Binomial_kernel(kernel_size)
        npt.assert_almost_equal(kernel, expected_kernel / 256, decimal=9)

    def test_binomial_coeffs(self):
        iter_start = -1  # always start with -1 if start array empty
        start_array = np.array([])

        n = 0
        coeffs_expected = np.array([1])
        coeffs = utils.get_Binomial_coefficients(start_array, iter_start, n)
        npt.assert_array_equal(coeffs, coeffs_expected)

        n = 1
        coeffs_expected = np.array([1, 1])
        coeffs = utils.get_Binomial_coefficients(start_array, iter_start, n)
        npt.assert_array_equal(coeffs, coeffs_expected)

        n = 2
        coeffs_expected = np.array([1, 2, 1])
        coeffs = utils.get_Binomial_coefficients(start_array, iter_start, n)
        npt.assert_array_equal(coeffs, coeffs_expected)

        n = 3
        coeffs_expected = np.array([1, 3, 3, 1])
        coeffs = utils.get_Binomial_coefficients(start_array, iter_start, n)
        npt.assert_array_equal(coeffs, coeffs_expected)

        n = 6
        coeffs_expected = np.array([1, 6, 15, 20, 15, 6, 1])
        coeffs = utils.get_Binomial_coefficients(start_array, iter_start, n)
        npt.assert_array_equal(coeffs, coeffs_expected)

class TestConvolution(unittest.TestCase):

    def test_convolution_big(self):
        ''' for convolutions that reduce the size of the image '''
        img = np.array([[45, 60, 98, 127, 132, 133, 137, 133],
                        [46, 65, 98, 123, 126, 128, 131, 133],
                        [47, 65, 96, 115, 119, 123, 135, 137],
                        [47, 63, 91, 107, 113, 122, 138, 134],
                        [50, 59, 80, 97, 110, 123, 133, 134],
                        [49, 53, 68, 83, 97, 113, 128, 133],
                        [50, 50, 58, 70, 84, 102, 116, 126],
                        [50, 50, 52, 58, 69, 86, 101, 120]
                        ])
        kernel = np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])
        img_expected = np.array([[69, 95, 116, 125, 129, 132],
                                 [68, 92, 110, 120, 126, 132],
                                 [66, 86, 104, 114, 124, 132],
                                 [62, 78, 94, 108, 120, 129],
                                 [57, 69, 83, 98, 112, 124],
                                 [53, 60, 71, 85, 100, 114]
                                 ])
        img_res = utils.convolve_reduce(img, kernel)
        npt.assert_almost_equal(round_matrix(img_res), img_expected, decimal=6)

    def test_1d_row_kernel(self):
        ''' testing row [-1, 0, 1] convolution preserving size'''
        kernel = np.zeros((1, 3))
        kernel[0, 0] = -1
        kernel[0, 2] = 1
        img = np.array([[2, 1, 4, 1], [3, 2, 1, 0], [5, 4, 3, 2]])
        img_exp = np.array([[1, 2, 0, -4], [2, -2, -2, -1], [4, -2, -2, -3]])
        print(kernel)
        img_res = utils.convolve(img, kernel)
        print(img_res)
        npt.assert_equal(img_res, img_exp)



    def test_1d_col_kernel(self):
        ''' testing col [-1; 0; 1] convolution preserving size'''
        kernel = np.zeros((3, 1))
        kernel[0, 0] = -1
        kernel[2, 0] = 1
        img = np.array([[2, 1, 4, 1], [3, 2, 1, 0], [5, 4, 3, 2]])
        img_exp = np.array([[3, 2, 1, 0], [3, 3, -1, 1], [-3, -2, -1, 0]])
        print(kernel)
        img_res = utils.convolve(img, kernel)
        print(img_res)
        npt.assert_equal(img_res, img_exp)


if __name__ == '__main__':
    unittest.main()
