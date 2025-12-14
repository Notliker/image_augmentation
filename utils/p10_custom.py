import cv2
import numpy as np
import os
from utils.AugmentationBase import AugmentationBase

class AutoMixingAugmentation(AugmentationBase):
    DEFAULT_PATH = os.path.join("data", "reference_10.jpg")

    def __init__(self, 
                 dataset_path=None, 
                 win_size=(128, 128), 
                 alpha=0.0):
        """
        :param dataset_path: Path to folder with images (or single file).
        :param win_size: Size of the sliding window (h, w) to find ROI.
        :param alpha: Blending coefficient (0 = full replace, 1 = no change).
        """
        self.win_size = win_size
        self.alpha = alpha
        
        # --- Store paths for random selection ---
        target_path = dataset_path if dataset_path else self.DEFAULT_PATH
        self.image_paths = []

        if os.path.isdir(target_path):
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            for f in os.listdir(target_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    self.image_paths.append(os.path.join(target_path, f))
            if not self.image_paths:
                print(f"[Warning] AutoMixing: Folder '{target_path}' is empty.")
        elif os.path.isfile(target_path):
            self.image_paths.append(target_path)
        else:
            print(f"[Warning] AutoMixing reference not found at '{target_path}'.")

    def _best_roi_by_variance(self, img_gray, win_size):
        """Using a sliding window, find the window with the maximum brightness variance."""
        h, w = img_gray.shape
        wh, ww = win_size
        
        if h < wh or w < ww:
            return 0, 0, h, w

        img = img_gray.astype(np.float32)
        I  = cv2.integral(img)
        I2 = cv2.integral(img * img)

        best_var = -1.0
        best_yx = (0, 0)

        for y in range(0, h - wh + 1):
            y1, y2 = y, y + wh
            for x in range(0, w - ww + 1):
                x1, x2 = x, x + ww

                s  = I[y2, x2]  - I[y1, x2]  - I[y2, x1]  + I[y1, x1]
                s2 = I2[y2, x2] - I2[y1, x2] - I2[y2, x1] + I2[y1, x1]

                area = wh * ww
                mean = s / area
                var  = s2 / area - mean * mean   

                if var > best_var:
                    best_var = var
                    best_yx = (y, x)

        y, x = best_yx
        return y, x, y + wh, x + ww

    def _get_random_ref_image(self):
        """Gets random image from image_paths list."""
        if not self.image_paths:
             return None

        path = np.random.choice(self.image_paths)
        ref_image = cv2.imread(path)
        return ref_image

    def apply(self, src: np.ndarray) -> np.ndarray:
        ref = self._get_random_ref_image()
        if ref is None:
            return src

        wh, ww = self.win_size
        h_src, w_src, _ = src.shape
        h_ref, w_ref, _ = ref.shape

        work_src = src
        work_ref = ref

        if h_src < wh or w_src < ww:
             return src
        if h_ref < wh or w_ref < ww:
             work_ref = cv2.resize(ref, (max(w_ref, ww), max(h_ref, wh)))
        gray_src = cv2.cvtColor(work_src, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(work_ref, cv2.COLOR_BGR2GRAY)

        y_ref_a, x_ref_a, y_ref_b, x_ref_b = self._best_roi_by_variance(gray_ref, self.win_size)
        y_src_a, x_src_a, y_src_b, x_src_b = self._best_roi_by_variance(gray_src, self.win_size)

        patch = work_ref[y_ref_a:y_ref_b, x_ref_a:x_ref_b]
        
        box_h = y_src_b - y_src_a
        box_w = x_src_b - x_src_a

        if patch.shape[0] != box_h or patch.shape[1] != box_w:
            patch = cv2.resize(patch, (box_w, box_h))

        roi_dst = work_src[y_src_a:y_src_b, x_src_a:x_src_b].astype(np.float32)
        patch_f = patch.astype(np.float32)

        if self.alpha == 0.0:
            mixed = patch_f
        else:
            mixed = self.alpha * roi_dst + (1.0 - self.alpha) * patch_f

        out = src.copy()
        out[y_src_a:y_src_b, x_src_a:x_src_b] = mixed.astype(np.uint8)

        return out

# def _best_roi_by_variance(img_gray, win_size):
#     """Using a sliding window, find the window with the maximum brightness variance."""
#     h, w = img_gray.shape
#     wh, ww = win_size

#     img = img_gray.astype(np.float32)
#     I  = cv2.integral(img)
#     I2 = cv2.integral(img * img)

#     best_var = -1.0
#     best_yx = (0, 0)

#     for y in range(0, h - wh + 1):
#         y1, y2 = y, y + wh
#         for x in range(0, w - ww + 1):
#             x1, x2 = x, x + ww

#             s  = I[y2, x2]  - I[y1, x2]  - I[y2, x1]  + I[y1, x1]
#             s2 = I2[y2, x2] - I2[y1, x2] - I2[y2, x1] + I2[y1, x1]

#             area = wh * ww
#             mean = s / area
#             var  = s2 / area - mean * mean   

#             if var > best_var:
#                 best_var = var
#                 best_yx = (y, x)

#     y, x = best_yx
#     return y, x, y + wh, x + ww


# def mix_images_auto(path1, path2,
#                     win_size=(128, 128),
#                     alpha=0.0,
#                     out_path=None):
#     """
#     Automatic blending without markup:
#     1) search for the most "textured" win_size square on both images;
#     2) insert the square from the first image into the square of the second image;
#     3) return the resulting image.
#     """
#     img1 = cv2.imread(path1)
#     img2 = cv2.imread(path2)
#     if img1 is None or img2 is None:
#         raise ValueError("Can't upload the image")

#     h1, w1, _ = img1.shape
#     h2, w2, _ = img2.shape
#     wh, ww = win_size

#     if h1 < wh or w1 < ww:
#         img1 = cv2.resize(img1, (max(w1, ww), max(h1, wh)))
#         h1, w1, _ = img1.shape
#     if h2 < wh or w2 < ww:
#         img2 = cv2.resize(img2, (max(w2, ww), max(h2, wh)))
#         h2, w2, _ = img2.shape

#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     y1a, x1a, y1b, x1b = _best_roi_by_variance(gray1, win_size)
#     y2a, x2a, y2b, x2b = _best_roi_by_variance(gray2, win_size)

#     obj = img1[y1a:y1b, x1a:x1b]      
#     box_h = y2b - y2a
#     box_w = x2b - x2a

#     if obj.shape[0] != box_h or obj.shape[1] != box_w:
#         obj = cv2.resize(obj, (box_w, box_h))

#     roi_dst = img2[y2a:y2b, x2a:x2b].astype(np.float32)
#     obj_f   = obj.astype(np.float32)

#     if alpha == 0.0:
#         mixed = obj_f
#     else:
#         mixed = alpha * roi_dst + (1.0 - alpha) * obj_f

#     img2[y2a:y2b, x2a:x2b] = mixed.astype(np.uint8)

#     if out_path is not None:
#         cv2.imwrite(out_path, img2)

#     return img2
