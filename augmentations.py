"""
augmentations.py
Real-world camera condition augmentations for backyard bird photography.

Each augmentation simulates a degradation that a typical phone camera or
backyard bird-cam would introduce in the wild.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io


AUGMENTATION_NAMES = [
    "Gaussian Blur",        # camera shake / bird in motion
    "Low Light",            # dawn / dusk shooting
    "Sensor Noise",         # cheap phone sensor / high ISO
    "Random Crop",          # partial occlusion by branches / feeder
    "JPEG Compression",     # screenshot from a bird-cam / WhatsApp share
]


def _to_pil(img) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8))
    return img.copy()


def aug_gaussian_blur(img, intensity=0.6):
    radius = 1.0 + intensity * 3.0
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def aug_low_light(img, intensity=0.6):
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(1.0 - intensity * 0.6)

    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(1.0 - intensity * 0.3)
    return img


def aug_sensor_noise(img, intensity=0.6):
    arr = np.array(img, dtype=np.float32)
    sigma = intensity * 25.0   # slightly reduced to match training realism
    noise = np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def aug_random_crop(img, intensity=0.6):
    w, h = img.size
    crop = intensity * 0.2  # reduced from 0.3

    left = int(w * np.random.uniform(0, crop))
    top = int(h * np.random.uniform(0, crop))
    right = int(w * (1 - np.random.uniform(0, crop)))
    bottom = int(h * (1 - np.random.uniform(0, crop)))

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h))


def aug_jpeg_compression(img, intensity=0.6):
    quality = int(95 - intensity * 70)  # less destructive than before
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(20, quality))
    buf.seek(0)
    return Image.open(buf).copy()


_AUG_FNS = [
    aug_gaussian_blur,
    aug_low_light,
    aug_sensor_noise,
    aug_random_crop,
    aug_jpeg_compression,
]


def apply_all_augmentations(
    img: Image.Image,
    intensity: float = 0.6,
) -> list[Image.Image]:
    """
    Apply each augmentation independently to the source image.

    Args:
        img:       PIL Image (RGB)
        intensity: 0.0 (no effect) → 1.0 (maximum effect)

    Returns:
        List of 5 augmented PIL Images (same order as AUGMENTATION_NAMES).
    """
    np.random.seed(42)          # reproducible crops
    base = _to_pil(img).convert("RGB")
    results = []
    for fn in _AUG_FNS:
        np.random.seed(42)      # reset seed for each so crop is consistent
        results.append(fn(base, intensity=intensity))
    return results
