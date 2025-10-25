from __future__ import annotations

import io
import threading
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .common import ImageInput, ensure_pil_image

_ENGINE_LOCK = threading.Lock()
_ENGINE: RapidOCR | None = None
_LAST_DURATION = 0.0


def _to_ocr_input(image: ImageInput) -> str | np.ndarray:
    """Return a value compatible with RapidOCR (file path or BGR ndarray)."""
    if isinstance(image, (str, Path)):
        return str(image)
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        return np.array(rgb)[:, :, ::-1]
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as loaded:
            rgb = loaded.convert("RGB")
            return np.array(rgb)[:, :, ::-1]
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("NumPy input must be HxWxC with at least 3 channels.")
        if array.shape[2] > 3:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return np.ascontiguousarray(array)
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def get_ocr_engine() -> RapidOCR:
    """Return a shared RapidOCR engine instance."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = RapidOCR()
        return _ENGINE


def run_ocr(
    image: ImageInput,
    *,
    engine: RapidOCR | None = None,
) -> tuple[Sequence[Sequence[object]], tuple[int, int]]:
    """
    Execute RapidOCR on the provided input.

    Returns the raw detection list alongside the image size (width, height).
    """
    global _LAST_DURATION

    ocr_engine = engine or get_ocr_engine()
    parsed = _to_ocr_input(image)

    start = perf_counter()
    result, _ = ocr_engine(parsed)
    _LAST_DURATION = perf_counter() - start

    if isinstance(parsed, np.ndarray):
        height, width = parsed.shape[:2]
    else:
        pil = ensure_pil_image(image)
        width, height = pil.size

    return result or (), (width, height)


def extract_text(
    image: ImageInput,
    *,
    engine: RapidOCR | None = None,
) -> str:
    """
    Convenience wrapper returning newline-delimited text only.

    `run_ocr` should be used when layout or bounding boxes are required.
    """
    raw_result, _ = run_ocr(image, engine=engine)
    lines: list[str] = []
    for entry in raw_result:
        if len(entry) >= 2 and entry[1]:
            lines.append(str(entry[1]))
    return "\n".join(lines)


def measure_last_call_duration() -> float:
    """Return the duration in seconds of the most recent OCR call."""
    return _LAST_DURATION
