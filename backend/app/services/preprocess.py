import io
import cv2
import numpy as np
from PIL import Image
from ..config import IMG_SIZE, JPEG_QUALITY
from .srm_filter_bank import get_srm30_kernels


def read_image_bytes(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img


def letterbox_pil(img: Image.Image, target_size=IMG_SIZE, pad_color=(0, 0, 0)):
    """Resize while keeping aspect ratio, pad to target_size."""
    tw, th = target_size
    w, h = img.size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.BICUBIC)

    new_img = Image.new("RGB", (tw, th), pad_color)
    left = (tw - nw) // 2
    top = (th - nh) // 2
    new_img.paste(resized, (left, top))
    return new_img


def srm30_residual_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Input : RGB uint8 (H,W,3)
    Output: RGB uint8 residual map (H,W,3) made by grouping 30 filter responses into 3 channels.
    """

    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernels = get_srm30_kernels()

    responses = []
    for ker in kernels:
        r = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=ker, borderType=cv2.BORDER_REFLECT)
        responses.append(np.abs(r))

    vol = np.stack(responses, axis=-1)  # (H,W,30)

    # Group filters into 3 groups -> 3 channels (mean abs response per group)
    g1 = np.mean(vol[:, :, 0:10], axis=-1)
    g2 = np.mean(vol[:, :, 10:20], axis=-1)
    g3 = np.mean(vol[:, :, 20:30], axis=-1)
    out = np.stack([g1, g2, g3], axis=-1)  # (H,W,3)

    # Robust per-channel normalization (percentile)
    out_norm = np.zeros_like(out, dtype=np.float32)
    for c in range(3):
        ch = out[:, :, c]
        lo = np.percentile(ch, 1)
        hi = np.percentile(ch, 99)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        chn = (ch - lo) / (hi - lo)
        out_norm[:, :, c] = np.clip(chn, 0, 1)

    return (out_norm * 255.0).astype(np.uint8)


def ela_rgb(pil_img: Image.Image, jpeg_quality: int = 90) -> np.ndarray:
    original = pil_img.convert("RGB")

    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=jpeg_quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    diff = np.abs(
        np.array(original, dtype=np.int16) - np.array(compressed, dtype=np.int16)
    ).astype(np.float32)

    mx = diff.max()
    if mx > 0:
        diff *= (255.0 / mx)

    return np.clip(diff, 0, 255).astype(np.uint8)


def make_model_inputs(original: Image.Image):
    """
    1) letterbox to IMG_SIZE
    2) create ELA + noise
    3) add batch dim
    """
    img_lb = letterbox_pil(original, IMG_SIZE)

    ela_vis = ela_rgb(img_lb, jpeg_quality=JPEG_QUALITY)          # (224,224,3) uint8
    noise_vis = srm30_residual_rgb(np.array(img_lb, dtype=np.uint8))              # (224,224,3) uint8

    # normalize to [0,1] float for models
    ela = (ela_vis.astype(np.float32) / 255.0)[None, ...]
    noise = (noise_vis.astype(np.float32) / 255.0)[None, ...]

    # return batches and PIL for visualization
    return ela, noise, img_lb
