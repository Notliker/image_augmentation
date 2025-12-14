import cv2
import numpy as np

def build_table(ref_color_bgr: np.ndarray) -> np.ndarray:
    """
    Return the array with shape (256, 3) where for each intensity
    one BGR colour stored.
    """
    h, w, _ = ref_color_bgr.shape
    ref_gray = cv2.cvtColor(ref_color_bgr, cv2.COLOR_BGR2GRAY)

    color_hist = [dict() for _ in range(256)]

    for y in range(h):
        for x in range(w):
            I = int(ref_gray[y, x])
            b, g, r = map(int, ref_color_bgr[y, x])
            key = (b, g, r)
            color_hist[I][key] = color_hist[I].get(key, 0) + 1

    table = np.zeros((256, 3), dtype=np.uint8)
    for I in range(256):
        if color_hist[I]:
            best_color = max(color_hist[I].items(), key=lambda kv: kv[1])[0]
            table[I] = best_color
        else:
            table[I] = (I, I, I)   

    return table

def colorize(gray: np.ndarray,
             table: np.ndarray,
             smooth_ksize: int = 3) -> np.ndarray:
    """
    gray   – input white/black image.
    table  – table from build_table.
    """
    h, w = gray.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            I = int(gray[y, x])
            out[y, x] = table[I]  

    if smooth_ksize >= 3 and smooth_ksize % 2 == 1:
        out = cv2.medianBlur(out, smooth_ksize)

    return out
