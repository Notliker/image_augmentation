import numpy as np
import os
import cv2
from utils.AugmentationBase import AugmentationBase

class ImageMixingAugmentation(AugmentationBase):
    #image to mix
    DEFAULT_PATH = os.path.join("data", "reference_8.jpg")

    def __init__(self, 
                 dataset_path=None,  # Renamed from reference_path to reflect folder usage
                 mode="random", 
                 alpha=0.5, 
                 border_thickness=5,
                 min_size=(32, 32),
                 max_size=(128, 128),
                 patch_size=(64, 64)):
        """
        :param dataset_path: path to folder with images of the same class (or single image).
        :param mode: "random" or "chess".
        :param alpha: mix coef (0..1).
        :param border_thickness:
        :param min_size, max_size: minmax patch for  mode="random".
        :param patch_size: fixed size for mode="chess".
        """
        self.mode = mode
        self.alpha = alpha
        self.border_thickness = border_thickness
        self.min_size = min_size
        self.max_size = max_size
        self.patch_size = patch_size

        # --- Updated logic: Store paths instead of loading single image ---
        # If dataset_path provided, scan folder. If not, fallback to DEFAULT_PATH (single file)
        target_path = dataset_path if dataset_path else self.DEFAULT_PATH
        self.image_paths = []

        if os.path.isdir(target_path):
             # It is a directory: load all valid images
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            for f in os.listdir(target_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    self.image_paths.append(os.path.join(target_path, f))
            if not self.image_paths:
                print(f"[Warning] ImageMixing: Folder '{target_path}' is empty.")
        elif os.path.isfile(target_path):
            # It is a single file
            self.image_paths.append(target_path)
        else:
             # if no image, then we use blank image (empty list)
            print(f"[Warning] Mixing reference not found at '{target_path}'. Using blank.")

    def _get_random_ref_image(self, target_shape):
        """
        garants, that ref_image same size with src.
        Gets random image from image_paths list.
        """
        h, w = target_shape[:2]

        if not self.image_paths:
             return np.zeros((h, w, 3), dtype=np.uint8)

        # Pick random file
        path = np.random.choice(self.image_paths)
        ref_image = cv2.imread(path)

        if ref_image is None:
             return np.zeros((h, w, 3), dtype=np.uint8)

        if ref_image.shape[:2] != (h, w):
            return cv2.resize(ref_image, (w, h))
        
        return ref_image

    def apply(self, src: np.ndarray) -> np.ndarray:
        """
        Mixing src & ref.
        mode="random"  – one random patch from ref to src.
        mode="chess"   – the chess order of patches.
        """
        
        # Get random reference image for this call
        ref = self._get_random_ref_image(src.shape)
        
        h, w, _ = src.shape

        def random_patch():
            ph = np.random.randint(self.min_size[0], self.max_size[0] + 1)
            pw = np.random.randint(self.min_size[1], self.max_size[1] + 1)
            max_y = max(0, h - ph + 1)
            max_x = max(0, w - pw + 1)
            y = np.random.randint(0, max_y) if max_y > 0 else 0
            x = np.random.randint(0, max_x) if max_x > 0 else 0
            return y, x, ph, pw

        def alpha_border_mask(ph, pw):
            mask = np.ones((ph, pw), dtype=np.float32) * self.alpha
            t = min(self.border_thickness, ph // 2, pw // 2)
            if t <= 0:
                return mask
            y = np.minimum(np.arange(ph), np.arange(ph)[::-1])
            x = np.minimum(np.arange(pw), np.arange(pw)[::-1])
            dist = np.minimum.outer(y, x)
            edge = dist < t
            mask[edge] = self.alpha * dist[edge] / t
            return mask

        #main logic
        if self.mode == "random":
            y, x, ph, pw = random_patch()

            if ph > h or pw > w:
                return src

            mask = alpha_border_mask(ph, pw)[..., None]

            patch_src = src[y:y+ph, x:x+pw].astype(np.float32)
            patch_ref = ref[y:y+ph, x:x+pw].astype(np.float32)

            mixed = mask * patch_src + (1.0 - mask) * patch_ref
            out = src.copy()
            out[y:y+ph, x:x+pw] = mixed.astype(np.uint8)
            return out

        elif self.mode == "chess":
            ph, pw = self.patch_size
            out = src.copy().astype(np.float32)
            base_mask = alpha_border_mask(ph, pw)[..., None]

            for y in range(0, h, ph):
                for x in range(0, w, pw):
                    y2 = min(y + ph, h)
                    x2 = min(x + pw, w)
                    cell_h, cell_w = y2 - y, x2 - x

                    if ((y // ph) + (x // pw)) % 2 == 0:
                        continue

                    patch_src = src[y:y2, x:x2].astype(np.float32)
                    patch_ref = ref[y:y2, x:x2].astype(np.float32)

                    m = base_mask[:cell_h, :cell_w, :]
                    
                    mixed = m * patch_src + (1.0 - m) * patch_ref
                    out[y:y2, x:x2] = mixed

            return out.astype(np.uint8)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

# def mix_images(src, ref,
#                mode="random",         
#                min_size=(32, 32),
#                max_size=(128, 128),
#                patch_size=(64, 64),
#                alpha=0.5,
#                border_thickness=5):
#     """
#     Mixing src & ref.
#     mode="random"  – one random patch from ref to src.
#     mode="chess"   – the chess order of patches.
#     """
#     assert src.shape == ref.shape
#     h, w, _ = src.shape

#     def random_patch():
#         ph = np.random.randint(min_size[0], max_size[0] + 1)
#         pw = np.random.randint(min_size[1], max_size[1] + 1)
#         y = np.random.randint(0, h - ph + 1)
#         x = np.random.randint(0, w - pw + 1)
#         return y, x, ph, pw

#     def alpha_border_mask(ph, pw):
#         mask = np.ones((ph, pw), dtype=np.float32) * alpha
#         t = min(border_thickness, ph // 2, pw // 2)
#         if t <= 0:
#             return mask
#         y = np.minimum(np.arange(ph), np.arange(ph)[::-1])
#         x = np.minimum(np.arange(pw), np.arange(pw)[::-1])
#         dist = np.minimum.outer(y, x)
#         edge = dist < t
#         mask[edge] = alpha * dist[edge] / t
#         return mask

#     if mode == "random":
#         y, x, ph, pw = random_patch()
#         mask = alpha_border_mask(ph, pw)[..., None]

#         patch_src = src[y:y+ph, x:x+pw].astype(np.float32)
#         patch_ref = ref[y:y+ph, x:x+pw].astype(np.float32)

#         mixed = mask * patch_src + (1.0 - mask) * patch_ref
#         out = src.copy()
#         out[y:y+ph, x:x+pw] = mixed.astype(np.uint8)
#         return out

#     elif mode == "chess":
#         ph, pw = patch_size
#         out = src.copy().astype(np.float32)
#         base_mask = alpha_border_mask(ph, pw)[..., None]

#         for y in range(0, h, ph):
#             for x in range(0, w, pw):
#                 y2 = min(y + ph, h)
#                 x2 = min(x + pw, w)
#                 cell_h, cell_w = y2 - y, x2 - x

#                 if ((y // ph) + (x // pw)) % 2 == 0:
#                     continue

#                 patch_src = src[y:y2, x:x2].astype(np.float32)
#                 patch_ref = ref[y:y2, x:x2].astype(np.float32)

#                 m = base_mask[:cell_h, :cell_w, :]
#                 mixed = m * patch_src + (1.0 - m) * patch_ref
#                 out[y:y2, x:x2] = mixed

#         return out.astype(np.uint8)

#     else:
#         raise ValueError("mode must be 'random' or 'chess'")
    

