"""Image preprocessing for the encoder path (plan Task 9).

Handles "bad user photos" deterministically before embedding:
  * EXIF auto-orient (phone photos are frequently rotated via EXIF only);
  * convert to RGB (drop alpha / palette / grayscale differences);
  * resize the short side then center-crop to a square target so aspect ratio
    and framing are normalized regardless of how the photo was taken.

The HF image processors also resize/normalize, but we apply an explicit,
encoder-agnostic auto-orient + center-crop first so tilted/off-centre catalog
photos land consistently no matter which encoder (or processor default) runs.
"""
from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageOps

# Default working resolution before the model processor takes over. 256 short
# side -> 224 center crop matches common ViT input framing while leaving the
# processor free to do its own final resize/normalize.
DEFAULT_SHORT_SIDE = 256
DEFAULT_CROP = 224


def auto_orient(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation then strip EXIF (idempotent, safe on no-EXIF)."""
    return ImageOps.exif_transpose(image)


def to_rgb(image: Image.Image) -> Image.Image:
    """Ensure a 3-channel RGB image (handles RGBA/P/L/CMYK inputs)."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def resize_short_side(image: Image.Image, short_side: int = DEFAULT_SHORT_SIDE) -> Image.Image:
    """Resize so the shorter dimension == ``short_side``, preserving aspect."""
    w, h = image.size
    if w == 0 or h == 0:
        return image
    if w <= h:
        new_w = short_side
        new_h = max(1, round(h * short_side / w))
    else:
        new_h = short_side
        new_w = max(1, round(w * short_side / h))
    return image.resize((new_w, new_h), Image.BILINEAR)


def center_crop(image: Image.Image, crop: int = DEFAULT_CROP) -> Image.Image:
    """Center-crop a ``crop`` x ``crop`` square (pads if the image is smaller)."""
    w, h = image.size
    # Pad up if needed so the crop box is always fully inside.
    if w < crop or h < crop:
        pad_w = max(0, crop - w)
        pad_h = max(0, crop - h)
        image = ImageOps.expand(
            image,
            border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
            fill=0,
        )
        w, h = image.size
    left = (w - crop) // 2
    top = (h - crop) // 2
    return image.crop((left, top, left + crop, top + crop))


def preprocess(
    image: Image.Image,
    short_side: int = DEFAULT_SHORT_SIDE,
    crop: int = DEFAULT_CROP,
) -> Image.Image:
    """Full pipeline: auto-orient -> RGB -> resize short side -> center-crop."""
    image = auto_orient(image)
    image = to_rgb(image)
    image = resize_short_side(image, short_side)
    image = center_crop(image, crop)
    return image


def preprocess_batch(
    images: Sequence[Image.Image],
    short_side: int = DEFAULT_SHORT_SIDE,
    crop: int = DEFAULT_CROP,
) -> list[Image.Image]:
    """Apply :func:`preprocess` to each image in the batch."""
    return [preprocess(im, short_side, crop) for im in images]
