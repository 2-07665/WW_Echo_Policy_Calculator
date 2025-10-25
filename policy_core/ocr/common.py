from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

ImageInput = str | Path | Image.Image | np.ndarray | bytes | bytearray


def ensure_pil_image(image: ImageInput) -> Image.Image:
    """Return a PIL image converted to RGB regardless of the input format."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            return loaded.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as loaded:
            return loaded.convert("RGB")
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            if array.dtype != np.uint8:
                array = array.astype(np.uint8)
            return Image.fromarray(array, mode="L").convert("RGB")
        if array.ndim == 3:
            if array.dtype != np.uint8:
                array = array.astype(np.uint8)
            if array.shape[2] == 3:
                return Image.fromarray(array[:, :, ::-1], mode="RGB")
            if array.shape[2] >= 4:
                return Image.fromarray(array[:, :, :4], mode="RGBA").convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def image_to_bgr(image: ImageInput) -> tuple[np.ndarray, Tuple[int, int]]:
    """Convert arbitrary image inputs into contiguous BGR ndarrays."""
    pil_image = ensure_pil_image(image)
    width, height = pil_image.size
    array = np.asarray(pil_image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Image conversion produced an invalid array shape.")
    bgr = array[:, :, ::-1]
    return np.ascontiguousarray(bgr), (width, height)
