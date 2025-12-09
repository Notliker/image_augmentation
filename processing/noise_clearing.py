import cv2
import numpy as np

class ImageDenoiser:
    """
    Class for removing noise from images.
    """
    def __init__(self, image: np.ndarray):
        """
        Initialize the class.
        :param image: Image as a numpy array.
        """
        self.image = image

    def apply_average_filter(self, kernel_size: int = 3) -> np.ndarray:
        """
        Averaging filter.
        Smooths the image by computing the mean of the pixels under the kernel.
        
        :param kernel_size: Kernel size (odd number).
        :return: Filtered image.
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
            
        return cv2.blur(self.image, (kernel_size, kernel_size))

    def apply_gaussian_filter(self, kernel_size: int = 3, sigma: float = 0) -> np.ndarray:
        """
        Gaussian filter.
        Uses a kernel with Gaussian weights. Effective for removing Gaussian noise.
        
        :param kernel_size: Kernel size (must be odd).
        :param sigma: Standard deviation in X direction. If 0, it is computed from kernel_size.
        :return: Filtered image.
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
            
        return cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma)

    def apply_median_filter(self, kernel_size: int = 3) -> np.ndarray:
        """
        Median filter.
        Replaces the central pixel with the median of all pixels under the kernel.
        
        :param kernel_size: Aperture size (must be odd and greater than 1).
        :return: Filtered image.
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
            
        return cv2.medianBlur(self.image, kernel_size)
