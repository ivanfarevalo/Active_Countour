import math
import cv2
import numpy as np


def create_gaussian_kernel(sigma_size):
    """
    Creates a gaussian kernel based on sigma and dimensions of kernel matrix
    :param sigma_size: standard deviation of gaussian
    :param kernel_dimensions: kernel dimensions
    """
    # Can modify kernel_dimension ration to sigma_size
    kernel_dimensions = 6 * sigma_size + 1
    gaussian_coefficient = 1/(2*math.pi*(sigma_size**2))
    gaussian_sum = 0
    gaussian_kernel = np.array([[0.0 for l in range(kernel_dimensions)] for w in range(kernel_dimensions)])

    for x in range(0, kernel_dimensions):
        for y in range(0, kernel_dimensions):
            i = x - math.floor(kernel_dimensions / 2)
            j = y - math.floor(kernel_dimensions / 2)
            gaussian_exponential = math.exp(-((i**2)+(j**2))/(2*sigma_size**2))
            # print("iteration {}:{}--> i: {} j: {}   var = {}".format(x, y, i, j, gaussian_exponential))
            gaussian_kernel[x, y] = gaussian_coefficient*gaussian_exponential
            gaussian_sum += gaussian_kernel[x, y]

    normalized_gaussian_kernel = (1/gaussian_sum)*gaussian_kernel
    return normalized_gaussian_kernel


def blur_image(gaussian_kernel, image):
    return cv2.filter2D(image, -1, gaussian_kernel)