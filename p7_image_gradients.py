import cv2
import numpy as np
from AugmentationBase import AugmentationBase


class GradientSharpeningAugmentation(AugmentationBase):
    """
    Base class for gradient augmentations
    all next classes - just changing tables for kernel x and y
    """
    def __init__(self, kernel_x, kernel_y, alpha=1.0):
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.alpha = alpha

    def apply(self, image):
        img_float = image.astype(np.float32)

        #filters
        grad_x = cv2.filter2D(img_float, cv2.CV_32F, self.kernel_x)
        grad_y = cv2.filter2D(img_float, cv2.CV_32F, self.kernel_y)

        #calc magnitude
        #magnitude = sqrt(Gx^2 + Gy^2)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)

        #applying
        # cv2.addWeighted: src1 * alpha + src2 * beta + gamma
        #: img_float * 1.0 + gradient * alpha
        sharpened = cv2.addWeighted(img_float, 1.0, gradient_magnitude, self.alpha, 0)

        return np.clip(sharpened, 0, 255).astype(np.uint8)

class SobelGradientAugmentation(GradientSharpeningAugmentation):
    def __init__(self, alpha=1.0):
        #sobel matrix
        # Gx
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
                       
        # Gy
        ky = np.array([[-1, -2, -1],
                       [0,  0,  0],
                       [1,  2,  1]], dtype=np.float32)
                       
        super().__init__(kx, ky, alpha)

class PrevittGradientAugmentation(GradientSharpeningAugmentation):
    def __init__(self, alpha=1.0):
        #Previtt matrix
        # Gx
        kx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float32)
                       
        # Gy
        ky = np.array([[-1, -1, -1],
                       [0,  0,  0],
                       [1,  1,  1]], dtype=np.float32)
                       
        super().__init__(kx, ky, alpha)


