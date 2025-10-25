from __future__ import annotations

import io
from time import perf_counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

ImageInput = str | Path | Image.Image | np.ndarray | bytes | bytearray

_ENGINE: RapidOCR | None = None


def _get_engine() -> RapidOCR:
    """Lazily create a shared RapidOCR engine instance."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RapidOCR()
    return _ENGINE


def _to_ocr_input(image: ImageInput) -> str | np.ndarray:
    """Return a value compatible with RapidOCR (file path or BGR ndarray)."""
    if isinstance(image, (str, Path)):
        return str(image)
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        # RapidOCR expects BGR channel ordering when given an ndarray.
        return np.array(rgb)[:, :, ::-1]
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as loaded:
            rgb = loaded.convert("RGB")
            return np.array(rgb)[:, :, ::-1]
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def _collect_lines(result: Sequence[Sequence[object]]) -> str:
    """Concatenate recognized text entries with newlines."""
    lines: List[str] = []
    for entry in result:
        if len(entry) >= 2 and entry[1]:
            lines.append(entry[1])
    return "\n".join(lines)


def extract_text(image: ImageInput, *, engine: RapidOCR | None = None) -> str:
    """
    Run RapidOCR on the provided image-like input and return concatenated text lines.

    Args:
        image: Path-like value, PIL Image, ndarray (BGR), or bytes.
        engine: Optional pre-existing RapidOCR instance for reuse.
    """
    ocr_engine = engine or _get_engine()
    parsed = _to_ocr_input(image)

    start = perf_counter()
    result, _ = ocr_engine(parsed)
    duration = perf_counter() - start
    setattr(extract_text, "_last_duration", duration)

    if not result:
        return ""
    return _collect_lines(result)


def measure_last_call_duration() -> float:
    """Return the duration (seconds) of the most recent `extract_text` call."""
    return getattr(extract_text, "_last_duration", 0.0)


def _ensure_pil_image(image: ImageInput) -> Image.Image:
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


def _comparison_array(image: ImageInput, size: tuple[int, int]) -> np.ndarray:
    pil_image = _ensure_pil_image(image)
    resized = pil_image.resize(size, Image.Resampling.LANCZOS)
    gray = resized.convert("L")
    array = np.asarray(gray, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Image conversion produced an empty array.")
    return array / 255.0


def is_upgrade_page(
    image: ImageInput,
    upgrade_page_logo: ImageInput,
    *,
    tolerance: float = 0.01,
    similarity_threshold: float = 0.99,
    sample_size: tuple[int, int] = (128, 128),
) -> bool:
    """
    Compare a capture against the predefined upgrade-page logo.

    The comparison is resolution-independent (aspect ratio must match) and uses a
    combination of mean absolute error and cosine similarity on downscaled
    grayscale representations.
    """
    reference = _comparison_array(upgrade_page_logo, sample_size)
    target = _comparison_array(image, sample_size)
    mae = float(np.abs(target - reference).mean())
    print(mae)
    if mae <= tolerance:
        return True

    numerator = float(np.dot(target.ravel(), reference.ravel()))
    denominator = float(np.linalg.norm(target.ravel()) * np.linalg.norm(reference.ravel()))
    similarity = numerator / denominator if denominator else 0.0
    return similarity >= similarity_threshold
