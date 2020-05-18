import sys
sys.path.append("..")
import linear_filtering as LF
import unittest
import numpy as np
import numpy.testing as npt
import time


class TestBoxFiltering(unittest.TestCase):

    def setUp(self):
        self.img = np.array([[0.0, 1, 0, 2], [0, 1, 3, 1], [1, 0, 0, 2]])
        self.expected_img = np.array(
            [[2, 5, 8, 6], [3, 6, 10, 8], [2, 5, 7, 6]])

    def test_convolution_matrix(self):
        start = time.time()
        res_img = LF.apply_box_filter(self.img, 3)
        end = time.time()
        print("Elapsed time", end - start)
        npt.assert_equal(res_img, self.expected_img)

    def test_convolution_separable_kernel(self):
        start = time.time()
        res_img = LF.apply_box_filter_separable_kernel(self.img, 3)
        end = time.time()
        print("Elapsed time", end - start)
        npt.assert_equal(res_img, self.expected_img)

    def test_convolution_integral_image(self):
        start = time.time()
        res_img = LF.convolve_integral_image(self.img, 3)
        end = time.time()
        print("Elapsed time", end - start)
        npt.assert_equal(res_img, self.expected_img)

    def test_int_img_border_conditions(self):
        int_img = np.array([[8, 13, 9, 6],
                            [17, 25, 18, 11],
                            [14, 18, 12, 15]])
        self.assertEqual(8, LF.get_integral_img_value(int_img, 0, 0))
        self.assertEqual(0, LF.get_integral_img_value(int_img, -1, 0))
        self.assertEqual(0, LF.get_integral_img_value(int_img, -1, -1))
        self.assertEqual(0, LF.get_integral_img_value(int_img, 0, -1))
        self.assertEqual(0, LF.get_integral_img_value(int_img, 0, -10))

        self.assertEqual(12, LF.get_integral_img_value(int_img, 5, 2))
        self.assertEqual(14, LF.get_integral_img_value(int_img, 4, 0))

        self.assertEqual(6, LF.get_integral_img_value(int_img, 0, 10))
        self.assertEqual(11, LF.get_integral_img_value(int_img, 1, 4))

        self.assertEqual(15, LF.get_integral_img_value(int_img, 4, 4))

        ## this should be an impossible case
        self.assertEqual(0, LF.get_integral_img_value(int_img, 5, -10))

if __name__ == '__main__':
    unittest.main()
