import cv2
import numpy as np
from AugmentationBase import AugmentationBase

class RGBtoGrayScaleAugmentation(AugmentationBase):
    def __init__(self):
        pass

    def apply(self, image):
        #color
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #color + alpha
        elif image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        
        #gray
        return image.copy()

    
class RGBtoBinaryAugmentation(AugmentationBase):
    def __init__(self, threshold=None):
        self.threshold = threshold

    def apply(self, image):
        #all to gray
        if image.ndim == 3:
            if image.shape[2] == 3:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                img_gray = image
        else:
            img_gray = image

        if self.threshold is None:
            #auto threshhold - OTSU
            _, binary_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            #fixed threshhold
            _, binary_img = cv2.threshold(
                img_gray, self.threshold, 255, cv2.THRESH_BINARY
            )

        return binary_img

        
