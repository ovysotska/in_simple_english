import numpy as np
import utils as ut

# Sobel operator
def find_edges_sobel_operator(img, verbose=False):

    dx_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    dy_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    dx_image = ut.convolve(img, dx_kernel) / 8
    dy_image = ut.convolve(img, dy_kernel) / 8
    res_image = np.sqrt(
        np.square(dx_image) + np.square(dy_image))
    if (verbose):
        print("X derivative \n", dx_image)
        print("Y derivative \n", dy_image)
        print("Gradient\n", res_image)
    return res_image

# Canny Edge detector
