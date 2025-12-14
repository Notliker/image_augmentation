import cv2
import numpy as np
from AugmentationBase import AugmentationBase


class HistogramEqualizationAugmentation(AugmentationBase):
    def __init__(self, bins=256):
        self.bins = bins

    def _equalize_channel(self, channel):
        """
        equalize 1 channel (for gray - gray channel)
        for multicolor image - will be V channel in HSV
        """
        hist = cv2.calcHist([channel], [0], None, [self.bins], [0, 256]).flatten()
        
        cdf = np.cumsum(hist)
        
        #min value
        try:
            cdf_min = min(v for v in cdf if v > 0)
        except ValueError:
            return channel
            
        pixel_count = channel.shape[0] * channel.shape[1]
        
        #lookuptable
        if pixel_count == cdf_min:
            return channel
            
        numerator = (cdf - cdf_min)
        denominator = (pixel_count - cdf_min)
        lookup = (numerator / denominator) * 255
        lookup = np.clip(lookup, 0, 255).astype(np.uint8)
        
        #apply lookuptable
        if self.bins != 256:
            #here will be if we have unusual bins
            indices = (channel.astype(np.float32) / 255.0 * (self.bins - 1)).astype(np.int32)
            equalized = lookup[indices]
        else:
            equalized = cv2.LUT(channel, lookup)
            
        return equalized

    def apply(self, image):
        if image.ndim == 2:
            #gray image
            return self._equalize_channel(image)
            
        elif image.ndim == 3:
            #color image
            #strat: RGB -> HSV -> equalizatin V -> RGB
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v_eq = self._equalize_channel(v)
            hsv_eq = cv2.merge((h, s, v_eq))
            
            return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
            
        return image

    
class GammaCorrectionAugmentation(AugmentationBase):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def apply(self, image):
        if self.gamma == 1.0:
            return image.copy()

        #lookuptable
        inv_gamma = 1.0 / self.gamma
        table = np.arange(256, dtype=np.float32) / 255.0
        table = np.power(table, inv_gamma) * 255.0
        
        lookup = np.clip(table, 0, 255).astype(np.uint8)
        
        #apply lookuptable
        corrected = lookup[image]
        
        return corrected
