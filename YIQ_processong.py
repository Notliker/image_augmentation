import numpy as np
from functools import wraps


M_RGB2YIQ = np.array([
    [0.299, 0.587, 0.114],
    [0.595716, -0.274453, -0.321263],
    [0.211456, -0.522591,  0.311135]
], dtype=np.float32)

M_YIQ2RGB = np.array([         
    [1.0,  0.9563,  0.6210 ],
    [1.0, -0.2721, -0.6474 ],
    [1.0, -1.1070,  1.7046 ]
], dtype=np.float32)

def rgb_img_to_yiq(img_rgb: np.ndarray) -> np.ndarray:
    img_f = img_rgb.astype(np.float32) / 255.0
    flat = img_f.reshape(-1, 3)
    yiq_flat = flat @ M_RGB2YIQ.T
    return yiq_flat.reshape(img_rgb.shape)

def yiq_img_to_rgb(img_yiq: np.ndarray) -> np.ndarray:
    flat = img_yiq.reshape(-1, 3)
    rgb_flat = flat @ M_YIQ2RGB.T
    rgb = rgb_flat.reshape(img_yiq.shape)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)

def yiq_channel_decorator(apply_to="Y"):
    idx = {"Y": 0, "I": 1, "Q": 2}[apply_to]
    def decorator(func):
        @wraps(func)
        def wrapper(img_rgb: np.ndarray, *args, **kwargs) -> np.ndarray:
            yiq = rgb_img_to_yiq(img_rgb)
            ch = yiq[..., idx]
            ch_new = func(ch, *args, **kwargs)
            yiq[..., idx] = ch_new
            return yiq_img_to_rgb(yiq)
        return wrapper
    return decorator