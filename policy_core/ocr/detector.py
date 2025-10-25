from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

from .common import ImageInput, ensure_pil_image


@dataclass(frozen=True)
class PageMatchConfig:
    tolerance: float = 0.01
    similarity_threshold: float = 0.99
    sample_size: Tuple[int, int] = (128, 128)


def _normalised_array(image: ImageInput, size: Tuple[int, int]) -> np.ndarray:
    pil_image = ensure_pil_image(image)
    resized = pil_image.resize(size, resample=Image.Resampling.LANCZOS)
    gray = resized.convert("L")
    array = np.asarray(gray, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Image conversion produced an empty array.")
    return array / 255.0


def is_upgrade_page(
    target: ImageInput,
    reference: ImageInput,
    config: PageMatchConfig | None = None,
) -> bool:
    """Compare a capture against the predefined upgrade-page logo."""
    cfg = config or PageMatchConfig()
    reference_array = _normalised_array(reference, cfg.sample_size)
    target_array = _normalised_array(target, cfg.sample_size)

    mae = float(np.abs(target_array - reference_array).mean())
    if mae <= cfg.tolerance:
        return True

    numerator = float(np.dot(target_array.ravel(), reference_array.ravel()))
    denominator = float(np.linalg.norm(target_array.ravel()) * np.linalg.norm(reference_array.ravel()))
    similarity = numerator / denominator if denominator else 0.0
    return similarity >= cfg.similarity_threshold
