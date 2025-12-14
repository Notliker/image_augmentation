import numpy as np

def mix_images(src, ref,
               mode="random",         
               min_size=(32, 32),
               max_size=(128, 128),
               patch_size=(64, 64),
               alpha=0.5,
               border_thickness=5):
    """
    Mixing src & ref.
    mode="random"  – one random patch from ref to src.
    mode="chess"   – the chess order of patches.
    """
    assert src.shape == ref.shape
    h, w, _ = src.shape

    def random_patch():
        ph = np.random.randint(min_size[0], max_size[0] + 1)
        pw = np.random.randint(min_size[1], max_size[1] + 1)
        y = np.random.randint(0, h - ph + 1)
        x = np.random.randint(0, w - pw + 1)
        return y, x, ph, pw

    def alpha_border_mask(ph, pw):
        mask = np.ones((ph, pw), dtype=np.float32) * alpha
        t = min(border_thickness, ph // 2, pw // 2)
        if t <= 0:
            return mask
        y = np.minimum(np.arange(ph), np.arange(ph)[::-1])
        x = np.minimum(np.arange(pw), np.arange(pw)[::-1])
        dist = np.minimum.outer(y, x)
        edge = dist < t
        mask[edge] = alpha * dist[edge] / t
        return mask

    if mode == "random":
        y, x, ph, pw = random_patch()
        mask = alpha_border_mask(ph, pw)[..., None]

        patch_src = src[y:y+ph, x:x+pw].astype(np.float32)
        patch_ref = ref[y:y+ph, x:x+pw].astype(np.float32)

        mixed = mask * patch_src + (1.0 - mask) * patch_ref
        out = src.copy()
        out[y:y+ph, x:x+pw] = mixed.astype(np.uint8)
        return out

    elif mode == "chess":
        ph, pw = patch_size
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
        raise ValueError("mode must be 'random' or 'chess'")
