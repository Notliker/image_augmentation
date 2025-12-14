import cv2
import numpy as np
from utils.AugmentationBase import AugmentationBase


class AverageBlurAugmentation(AugmentationBase):
    def __init__(self, kernel_size=3):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        self.kernel_size = kernel_size

    def apply(self, image):
        return cv2.blur(image, (self.kernel_size, self.kernel_size))


class GaussianBlurAugmentation(AugmentationBase):
    def __init__(self, kernel_size=3, sigma=0):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        self.kernel_size = kernel_size
        self.sigma = sigma

    def apply(self, image):
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)


class MedianBlurAugmentation(AugmentationBase):
    def __init__(self, kernel_size=3):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        self.kernel_size = kernel_size

    def apply(self, image):
        return cv2.medianBlur(image, self.kernel_size)