import cv2
import numpy as np
import math
from AugmentationBase import AugmentationBase

class ScalingAugmentation(AugmentationBase):
    def __init__(self, scale_x=1.0, scale_y=1.0):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def apply(self, image):
        return cv2.resize(image, None, fx=self.scale_x, fy=self.scale_y, interpolation=cv2.INTER_LINEAR)

class TranslationAugmentation(AugmentationBase):
    def __init__(self, shift_x=50, shift_y=0):
        self.shift_x = shift_x
        self.shift_y = shift_y

    def apply(self, image):
        height, width = image.shape[:2]
        
        # [1, 0, tx]
        # [0, 1, ty]
        M = np.float32([[1, 0, self.shift_x], [0, 1, self.shift_y]])
        
        # warpAffine - fast method in cv2, better than cyclse
        return cv2.warpAffine(image, M, (width, height))

class RotationAugmentation(AugmentationBase):
    def __init__(self, angle=45):
        self.angle = angle

    def apply(self, image): #not cv2.rotate, cause it only 90, 180 and so on.
        height, width = image.shape[:2]
        center = (width // 2, height // 2) #around image center
        
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        
        return cv2.warpAffine(image, M, (width, height))

class GlassEffectAugmentation(AugmentationBase):
    def __init__(self, max_dist=10):
        self.max_dist = max_dist

    def apply(self, image):
        height, width = image.shape[:2]
        
        # map_x[y, x] = x, map_y[y, x] = y
        map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        #generating random shifts for every pixel
        rand_dx = np.random.uniform(-0.5, 0.5, (height, width)) * self.max_dist
        rand_dy = np.random.uniform(-0.5, 0.5, (height, width)) * self.max_dist
        
        map_x += rand_dx
        map_y += rand_dy
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

class Wave1Augmentation(AugmentationBase):
    def __init__(self, amplitude=20, period=60):
        self.amplitude = amplitude
        self.period = period

    def apply(self, image):
        height, width = image.shape[:2]
        
        map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # x(k, l) = k + 20 * sin(2*pi*l / 60)
        # l - coord Y, k - coord X
        
        offset = self.amplitude * np.sin(2 * np.pi * map_y / self.period)
        map_x = map_x + offset
        # y(k, l) = l
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

class Wave2Augmentation(AugmentationBase):
    def __init__(self, amplitude=20, period_x=30):
        self.amplitude = amplitude
        self.period_x = period_x

    def apply(self, image):
        height, width = image.shape[:2]
        
        map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # x(k, l) = k + 20 * sin(2*pi*k / 30)
        
        offset = self.amplitude * np.sin(2 * np.pi * map_x / self.period_x)
        map_x = map_x + offset
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

class MotionBlurAugmentation(AugmentationBase):
    def __init__(self, kernel_size=15):
        self.kernel_size = kernel_size

    def apply(self, image):
        #main diagonal with 1
        kernel_motion_blur = np.zeros((self.kernel_size, self.kernel_size))
        np.fill_diagonal(kernel_motion_blur, 1)
        
        #norming
        kernel_motion_blur /= self.kernel_size

        return cv2.filter2D(image, -1, kernel_motion_blur)