import numpy as np
from functools import wraps
from utils.AugmentationBase import AugmentationBase

class YIQAugmentationWrapper(AugmentationBase):
    def __init__(self, inner_augmentation, apply_to_channels="Y"):
        """
        :param inner_augmentation: Объект другой аугментации (например, GaussianNoiseAugmentation(sigma=20))
        :param apply_to_channels: Строка каналов, к которым применять ("Y", "I", "Q", "YI" и т.д.)
        """
        self.inner_aug = inner_augmentation
        self.apply_to_channels = apply_to_channels
        self.channel_map = {"Y": 0, "I": 1, "Q": 2}
        self.M_RGB2YIQ = np.array([
            [0.299, 0.587, 0.114],
            [0.595716, -0.274453, -0.321263],
            [0.211456, -0.522591,  0.311135]
        ], dtype=np.float32)

        self.M_YIQ2RGB = np.array([         
            [1.0,  0.9563,  0.6210 ],
            [1.0, -0.2721, -0.6474 ],
            [1.0, -1.1070,  1.7046 ]
        ], dtype=np.float32)

    def _rgb_img_to_yiq(self, img_rgb: np.ndarray) -> np.ndarray:
        img_f = img_rgb.astype(np.float32) / 255.0
        flat = img_f.reshape(-1, 3)
        yiq_flat = flat @ self.M_RGB2YIQ.T
        return yiq_flat.reshape(img_rgb.shape)

    def _yiq_img_to_rgb(self, img_yiq: np.ndarray) -> np.ndarray:
        flat = img_yiq.reshape(-1, 3)
        rgb_flat = flat @ self.M_YIQ2RGB.T
        rgb = rgb_flat.reshape(img_yiq.shape)
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)
    def apply(self, image):
        #if gray, then YIQ has no reason
        if image.ndim == 2:
            return self.inner_aug.apply(image)

        #to YIQ
        yiq_image = self._rgb_img_to_yiq(image)

        for char in self.apply_to_channels:
            if char in self.channel_map:
                idx = self.channel_map[char]
                
                channel = yiq_image[..., idx]
                
                #Type problem, cause RGB uses 0...255,
                #but YIQ - float from 0 to 1
                
                # Y  0..1
                # I, Q  -0.6..0.6
                
                if char == 'Y':
                    # converyt Y (0..1) -> (0..255 uint8)
                    ch_uint8 = np.clip(channel * 255, 0, 255).astype(np.uint8)
                    
                    #AUGMENTATION
                    processed_uint8 = self.inner_aug.apply(ch_uint8)
                    
                    # back convert to float (0..1)
                    yiq_image[..., idx] = processed_uint8.astype(np.float32) / 255.0
                    
                else: 
                    #I Q can be minus, but int cant be
                    # hist equalization maybe be bad :(
                    # -0.6..0.6 -> 0..1 -> 0..255
                    
                    # normalization by shift
                    ch_shifted = (channel + 0.6) / 1.2
                    ch_uint8 = np.clip(ch_shifted * 255, 0, 255).astype(np.uint8)
                    
                    processed_uint8 = self.inner_aug.apply(ch_uint8)
                    
                    #back convert
                    yiq_image[..., idx] = (processed_uint8.astype(np.float32) / 255.0) * 1.2 - 0.6

        #to RGB
        return self._yiq_img_to_rgb(yiq_image)
    

# def yiq_channel_decorator(apply_to="Y"):
#     idx = {"Y": 0, "I": 1, "Q": 2}[apply_to]
#     def decorator(func):
#         @wraps(func)
#         def wrapper(img_rgb: np.ndarray, *args, **kwargs) -> np.ndarray:
#             yiq = rgb_img_to_yiq(img_rgb)
#             ch = yiq[..., idx]
#             ch_new = func(ch, *args, **kwargs)
#             yiq[..., idx] = ch_new
#             return yiq_img_to_rgb(yiq)
#         return wrapper
#     return decorator