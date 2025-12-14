import cv2
import numpy as np
from AugmentationBase import AugmentationBase


class ImpulseNoiseAugmentation(AugmentationBase):
    def __init__(self, noise_values=[0, 255, 128], noise_probs=[0.3, 0.3, 0.4], noise_percentage=0.05):
        self.noise_values = np.array(noise_values)
        self.noise_probs = np.array(noise_probs)
        self.noise_percentage = noise_percentage
        self.cumsum_probs = np.cumsum(self.noise_probs)

    def apply(self, image):
        noisy_image = image.copy()
        
        #every image will be (H, W, Channels)
        #so if it is gray - then temporaly adding chanells to it
        
        if image.ndim == 2:
            # (H, W) -> (H, W, 1)
            work_image = noisy_image[:, :, np.newaxis]
        else:
            # (H, W, C)
            work_image = noisy_image
            
        height, width, channels = work_image.shape
        total_pixels_per_channel = height * width
        
        #amount of noisy pixels per channel
        num_noise = int(total_pixels_per_channel * self.noise_percentage)
        
        #cycle on channels
        for ch in range(channels):
            #random x y coords for current channel
            coords_x = np.random.randint(0, width, num_noise)
            coords_y = np.random.randint(0, height, num_noise)
            
            #generating noise value
            alphas = np.random.random(num_noise)
            
            #searchsorted for choosing value on probs
            indices = np.searchsorted(self.cumsum_probs, alphas)
            noise_vals = self.noise_values[indices]
            
            #apply noise for cur channel
            work_image[coords_y, coords_x, ch] = noise_vals
            
        #return noisy_image due to np views (work_image is view on noise_image)
        return noisy_image
    
class GaussianNoiseAugmentation(AugmentationBase):
    def __init__(self, mean=0, sigma=25):
        self.mean = mean
        self.sigma = sigma
        
    def apply(self, image):
        #generating noise same shape as image
        noise = np.random.normal(self.mean, self.sigma, image.shape).astype(np.float32)
        
        #convert to float to prevent overflowing int
        img_float = image.astype(np.float32)
        #cv2.add working on every channel
        noisy_image = cv2.add(img_float, noise)
    
        #clip - to prevent more than 255 due to noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
   
class RayleighNoiseAugmentation(AugmentationBase):
    def __init__(self, scale=30):
        self.scale = scale
        
    def apply(self, image):
        #generating noise same shape as image
        noise = np.random.rayleigh(self.scale, image.shape).astype(np.float32)
        
        #convert to float to prevent overflowing int
        img_float = image.astype(np.float32)
        #cv2.add working on every channel
        noisy_image = cv2.add(img_float, noise)
    
        #clip - to prevent more than 255 due to noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)


class ExponentialNoiseAugmentation(AugmentationBase):
    def __init__(self, scale=15):
        self.scale = scale
        
    def apply(self, image):
        #generating noise same shape as image
        noise = np.random.exponential(self.scale, image.shape).astype(np.float32)
        
        #convert to float to prevent overflowing int
        img_float = image.astype(np.float32)
        #cv2.add working on every channel
        noisy_image = cv2.add(img_float, noise)
    
        #clip - to prevent more than 255 due to noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
