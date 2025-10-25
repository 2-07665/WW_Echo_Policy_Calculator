from __future__ import annotations

import io
import os
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, cast

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from ocr import _get_engine, is_upgrade_page

try:
    from policy_core.data import BUFF_LABELS, BUFF_TYPE_COUNTS, BUFF_TYPES
except ModuleNotFoundError:  # pragma: no cover - allow running inside OCR folder
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from policy_core.data import BUFF_LABELS, BUFF_TYPE_COUNTS, BUFF_TYPES

ImageInput = str | Path | Image.Image | np.ndarray | bytes | bytearray

_LOCK_PATTERN = re.compile(r"强化至\+?(\d+)\s*可[调調][谐諧]")
_PENDING_PATTERN = re.compile(r"待[调調][谐諧]")
_LABEL_KEY_STRIP = re.compile(r"[\s·•:_：／/|\\-]+")

WINDOW_TITLE = os.environ.get("OCR_WINDOW_TITLE", "鸣潮  ")
# Update these defaults to tune the capture regions. Values are (left, top, right, bottom)
# expressed as percentages of the client window.
DEFAULT_DETECTION_CROP: tuple[float, float, float, float] = (0.00, 0.00, 0.15, 0.1)
DEFAULT_OCR_CROP: tuple[float, float, float, float] = (0.110, 0.29, 0.37, 0.5)
DETECTION_INTERVAL = float(os.environ.get("OCR_DETECTION_INTERVAL", "15.0"))
DETAILED_CAPTURE_INTERVAL = float(os.environ.get("OCR_DETAILED_INTERVAL", "0.1"))



@dataclass(frozen=True)
class BuffDefinition:
    type_name: str
    label: str
    allowed_values: frozenset[int]
    counts: Dict[int, int]


@dataclass
class _ValueCandidates:
    flat: int | None
    percent: int | None
    explicit_percent: bool


@dataclass
class _OCRDetection:
    text: str
    confidence: float
    center_x: float
    center_y: float


@dataclass
class BuffSlotResult:
    index: int
    status: str
    buff_type: str | None
    buff_label: str | None
    normalized_value: int | None
    raw_type_text: str | None
    raw_value_text: str | None
    cleaned_value_text: str | None
    type_valid: bool
    value_valid: bool
    unlock_requirement: int | None = None


@dataclass
class BuffWorkflowResult:
    slots: List[BuffSlotResult]
    buff_types: List[str | None]
    buff_values: List[int | None]

    def to_ui_payload(self) -> Dict[str, List[str | int | None]]:
        return {"buff_names": self.buff_types, "buff_values": self.buff_values}


def _image_to_bgr(image: ImageInput) -> tuple[np.ndarray, tuple[int, int]]:
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        width, height = rgb.size
        return np.array(rgb)[:, :, ::-1], (width, height)
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            rgb = loaded.convert("RGB")
            width, height = rgb.size
            return np.array(rgb)[:, :, ::-1], (width, height)
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as loaded:
            rgb = loaded.convert("RGB")
            width, height = rgb.size
            return np.array(rgb)[:, :, ::-1], (width, height)
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("NumPy image input must be HxWxC with at least 3 channels.")
        if array.shape[2] > 3:
            array = array[:, :, :3]
        height, width = array.shape[:2]
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return np.ascontiguousarray(array), (width, height)
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def _cleanup_buff_type(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u3000", " ").strip()


def _cleanup_value_token(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\uFF05", "%")
    cleaned = cleaned.replace("O", "0").replace("o", "0")
    cleaned = cleaned.replace(" ", "").replace("\u3000", "")
    cleaned = cleaned.replace("\n", "").replace("\r", "")
    cleaned = cleaned.replace(",", "").replace("\uFF0C", "")
    cleaned = cleaned.replace("\uFF0E", ".").replace("\u3002", ".")
    cleaned = cleaned.replace("＋", "+")
    return cleaned.strip()


def _normalize_label_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u3000", " ")
    normalized = normalized.strip()
    return _LABEL_KEY_STRIP.sub("", normalized)


def _parse_single_value(text: str) -> _ValueCandidates:
    cleaned = _cleanup_value_token(text)
    explicit_percent = "%" in cleaned
    numeric = cleaned.replace("%", "")
    if not numeric:
        return _ValueCandidates(flat=None, percent=None, explicit_percent=explicit_percent)
    try:
        value = Decimal(numeric)
    except InvalidOperation:
        return _ValueCandidates(flat=None, percent=None, explicit_percent=explicit_percent)
    flat_candidate: int | None = None
    percent_candidate: int | None = None
    if value == value.to_integral_value():
        flat_candidate = int(value)
    if explicit_percent or "." in numeric or percent_candidate is None:
        scaled = (value * Decimal(10)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        percent_candidate = int(scaled)
    return _ValueCandidates(flat=flat_candidate, percent=percent_candidate, explicit_percent=explicit_percent)


def _value_candidate_list(text: str) -> List[Tuple[str, _ValueCandidates]]:
    normalized = unicodedata.normalize("NFKC", text).replace("\r", "\n")
    raw_parts: List[str] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        segments = stripped.split()
        if segments:
            raw_parts.extend(segments)
        else:
            raw_parts.append(stripped)

    cleaned_parts = [part for part in (_cleanup_value_token(part) for part in raw_parts) if part]
    if not cleaned_parts:
        fallback = _cleanup_value_token(normalized.replace("\n", ""))
        if fallback:
            cleaned_parts = [fallback]

    variants: List[str] = []
    if cleaned_parts:
        variants.append("".join(cleaned_parts))
        variants.extend(cleaned_parts)
        for idx in range(len(cleaned_parts) - 1):
            variants.append(cleaned_parts[idx] + cleaned_parts[idx + 1])
    else:
        variants.append("")

    ordered: List[str] = []
    seen: set[str] = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        ordered.append(variant)

    interpretations: List[Tuple[str, _ValueCandidates]] = []
    for variant in ordered:
        parsed = _parse_single_value(variant)
        if parsed.flat is None and parsed.percent is None:
            continue
        interpretations.append((variant, parsed))
    return interpretations


def _detections_from_result(raw_result: Sequence[Sequence[object]]) -> List[_OCRDetection]:
    detections: List[_OCRDetection] = []
    for entry in raw_result:
        if len(entry) < 3:
            continue
        points, text, score = entry[:3]
        if not text:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        detections.append(
            _OCRDetection(
                text=str(text),
                confidence=float(score),
                center_x=center_x,
                center_y=center_y,
            )
        )
    return detections


def _group_by_rows(detections: Sequence[_OCRDetection], tolerance: float) -> List[List[_OCRDetection]]:
    rows: List[List[_OCRDetection]] = []
    centers: List[float] = []
    for detection in sorted(detections, key=lambda det: det.center_y):
        placed = False
        for idx, center in enumerate(centers):
            if abs(center - detection.center_y) <= tolerance:
                rows[idx].append(detection)
                centers[idx] = sum(item.center_y for item in rows[idx]) / len(rows[idx])
                placed = True
                break
        if not placed:
            rows.append([detection])
            centers.append(detection.center_y)
    for row in rows:
        row.sort(key=lambda det: det.center_x)
    return rows


class BuffWorkflow:
    def __init__(
        self,
        *,
        engine: RapidOCR | None = None,
        max_slots: int = 5,
        row_tolerance_ratio: float = 0.035,
    ) -> None:
        self.engine = engine or _get_engine()
        if not isinstance(self.engine, RapidOCR):
            raise TypeError("BuffWorkflow currently supports RapidOCR engines only.")
        self.max_slots = max(1, max_slots)
        self.row_tolerance_ratio = max(0.0, row_tolerance_ratio)

        self.buff_definitions_by_type = self._build_definitions()
        self._label_alias_map: Dict[str, Tuple[BuffDefinition, ...]] = {}
        for definition in self.buff_definitions_by_type.values():
            self._register_alias(definition.label, (definition.type_name,))
            self._register_alias(_cleanup_buff_type(definition.label), (definition.type_name,))

        # Handle ambiguous UI labels missing 百分比.
        self._register_alias("攻击", ("Attack", "Attack_Flat"))
        self._register_alias("防御", ("Defence", "Defence_Flat"))
        self._register_alias("生命", ("HP", "HP_Flat"))

    def _build_definitions(self) -> Dict[str, BuffDefinition]:
        definitions: Dict[str, BuffDefinition] = {}
        for idx, type_name in enumerate(BUFF_TYPES):
            label = BUFF_LABELS.get(type_name, type_name)
            raw_counts = BUFF_TYPE_COUNTS[idx] if idx < len(BUFF_TYPE_COUNTS) else {}
            counts: Dict[int, int] = {int(value): int(freq) for value, freq in raw_counts.items()}
            definitions[type_name] = BuffDefinition(
                type_name=type_name,
                label=label,
                allowed_values=frozenset(counts.keys()),
                counts=counts,
            )
        return definitions

    def _register_alias(self, alias: str, type_names: Iterable[str]) -> None:
        key = _normalize_label_key(alias)
        if not key:
            return
        definitions = tuple(
            self.buff_definitions_by_type[name] for name in type_names if name in self.buff_definitions_by_type
        )
        if not definitions:
            return
        existing = {definition.type_name: definition for definition in self._label_alias_map.get(key, ())}
        for definition in definitions:
            existing[definition.type_name] = definition
        self._label_alias_map[key] = tuple(existing.values())

    def process_image(self, image: ImageInput) -> BuffWorkflowResult:
        bgr, (_, height) = _image_to_bgr(image)
        ocr_result, _ = self.engine(bgr)
        detections = _detections_from_result(ocr_result or [])
        if not detections:
            return BuffWorkflowResult(slots=[], buff_types=[], buff_values=[])

        tolerance = max(12.0, height * self.row_tolerance_ratio)
        grouped_rows = _group_by_rows(detections, tolerance)

        slots: List[BuffSlotResult] = []
        buff_types: List[str | None] = []
        buff_values: List[int | None] = []

        for index, row in enumerate(grouped_rows[: self.max_slots]):
            if not row:
                continue
            raw_type_text = _cleanup_buff_type(row[0].text)
            status_key = raw_type_text.replace(" ", "")

            if not raw_type_text:
                slots.append(
                    BuffSlotResult(
                        index=index,
                        status="empty",
                        buff_type=None,
                        buff_label=None,
                        normalized_value=None,
                        raw_type_text=None,
                        raw_value_text=None,
                        cleaned_value_text=None,
                        type_valid=False,
                        value_valid=False,
                    )
                )
                continue

            if _PENDING_PATTERN.search(status_key):
                slots.append(
                    BuffSlotResult(
                        index=index,
                        status="pending",
                        buff_type=None,
                        buff_label=None,
                        normalized_value=None,
                        raw_type_text=raw_type_text,
                        raw_value_text=None,
                        cleaned_value_text=None,
                        type_valid=False,
                        value_valid=False,
                    )
                )
                buff_types.append(None)
                buff_values.append(None)
                break

            lock_match = _LOCK_PATTERN.search(status_key)
            if lock_match:
                requirement = int(lock_match.group(1))
                slots.append(
                    BuffSlotResult(
                        index=index,
                        status="locked",
                        buff_type=None,
                        buff_label=None,
                        normalized_value=None,
                        raw_type_text=raw_type_text,
                        raw_value_text=None,
                        cleaned_value_text=None,
                        type_valid=False,
                        value_valid=False,
                        unlock_requirement=requirement,
                    )
                )
                buff_types.append(None)
                buff_values.append(None)
                break

            value_text = "".join(det.text for det in row[1:]) if len(row) > 1 else ""
            value_candidates = _value_candidate_list(value_text)
            cleaned_value_text = value_candidates[0][0] if value_candidates else _cleanup_value_token(value_text)
            definition, matched_value, matched_text = self._resolve_definition(raw_type_text, value_candidates)

            type_valid = definition is not None
            value_valid = matched_value is not None and type_valid

            slot = BuffSlotResult(
                index=index,
                status="buff" if type_valid else "unknown",
                buff_type=definition.type_name if definition and value_valid else None,
                buff_label=definition.label if definition else None,
                normalized_value=matched_value if value_valid else None,
                raw_type_text=raw_type_text or None,
                raw_value_text=value_text or None,
                cleaned_value_text=matched_text or cleaned_value_text or None,
                type_valid=type_valid,
                value_valid=value_valid,
            )
            slots.append(slot)
            if type_valid and value_valid:
                buff_types.append(slot.buff_type)
                buff_values.append(slot.normalized_value)
            else:
                buff_types.append(None)
                buff_values.append(None)

        current_index = len(slots)
        while current_index < self.max_slots:
            slots.append(
                BuffSlotResult(
                    index=current_index,
                    status="missing",
                    buff_type=None,
                    buff_label=None,
                    normalized_value=None,
                    raw_type_text=None,
                    raw_value_text=None,
                    cleaned_value_text=None,
                    type_valid=False,
                    value_valid=False,
                )
            )
            buff_types.append(None)
            buff_values.append(None)
            current_index += 1

        return BuffWorkflowResult(slots=slots, buff_types=buff_types, buff_values=buff_values)

    def _resolve_definition(
        self,
        raw_type_text: str,
        value_candidates: List[Tuple[str, _ValueCandidates]],
    ) -> Tuple[BuffDefinition | None, int | None, str | None]:
        key = _normalize_label_key(raw_type_text)
        options = list(self._label_alias_map.get(key, ()))
        if not options:
            collected: Dict[str, BuffDefinition] = {}
            for alias_key, definitions in self._label_alias_map.items():
                if alias_key and alias_key in key:
                    for definition in definitions:
                        collected.setdefault(definition.type_name, definition)
            options = list(collected.values())

        if not options:
            return None, None, None

        if len(options) == 1:
            definition = options[0]
            matched_value, matched_text = self._match_value(definition, value_candidates)
            return definition, matched_value, matched_text

        matches: List[Tuple[BuffDefinition, int, str | None, _ValueCandidates]] = []
        for definition in options:
            matched_value, matched_text, meta = self._match_value(definition, value_candidates, with_meta=True)
            if matched_value is not None:
                matches.append((definition, matched_value, matched_text, meta))

        if not matches:
            return None, None, None

        definition, value, text = self._select_preferred_match(matches)
        return definition, value, text

    def _match_value(
        self,
        definition: BuffDefinition,
        candidates: List[Tuple[str, _ValueCandidates]],
        *,
        with_meta: bool = False,
    ) -> Tuple[int | None, str | None, _ValueCandidates | None]:
        if not candidates:
            return (None, None, None) if with_meta else (None, None)

        best_fallback: Tuple[float, int, str, _ValueCandidates] | None = None
        for text, candidate in candidates:
            possible_values: List[int] = []
            if candidate.percent is not None:
                possible_values.append(candidate.percent)
            if candidate.flat is not None:
                possible_values.append(candidate.flat)

            for candidate_value in possible_values:
                if candidate_value in definition.allowed_values:
                    return (
                        (candidate_value, text, candidate)
                        if with_meta
                        else (candidate_value, text)
                    )
                if definition.allowed_values:
                    nearest = min(definition.allowed_values, key=lambda value: abs(value - candidate_value))
                    diff = abs(nearest - candidate_value)
                    if diff <= 2:
                        if best_fallback is None or diff < best_fallback[0]:
                            best_fallback = (diff, nearest, text, candidate)

        if best_fallback:
            _, value, text, candidate = best_fallback
            return (
                (value, text, candidate)
                if with_meta
                else (value, text)
            )
        return (None, None, None) if with_meta else (None, None)

    def _select_preferred_match(
        self,
        matches: List[Tuple[BuffDefinition, int, str | None, _ValueCandidates]],
    ) -> Tuple[BuffDefinition, int, str | None]:
        def sort_key(item: Tuple[BuffDefinition, int, str | None, _ValueCandidates]) -> Tuple[int, int]:
            definition, value, _, candidate_meta = item
            if candidate_meta.percent is not None and value == candidate_meta.percent:
                return (0, 0 if not definition.type_name.endswith("_Flat") else 1)
            if candidate_meta.flat is not None and value == candidate_meta.flat:
                return (1, 0 if definition.type_name.endswith("_Flat") else 1)
            return (2, 0 if not definition.type_name.endswith("_Flat") else 1)

        definition, value, text, _ = min(matches, key=sort_key)
        return definition, value, text


def _process_and_print(workflow: BuffWorkflow, image: Image.Image) -> None:
    result = workflow.process_image(image)
    for slot in result.slots:
        print(
            f"[debug] Slot {slot.index + 1}: status={slot.status} "
            f"type={slot.raw_type_text!r} value={slot.raw_value_text!r} cleaned={slot.cleaned_value_text!r}"
        )
        if slot.status == "buff" and slot.type_valid and slot.value_valid:
            print(f" -> {slot.buff_label} ({slot.buff_type}) normalized={slot.normalized_value}")
        elif slot.status in {"pending", "locked"} and slot.unlock_requirement:
            print(f" -> {slot.status} until +{slot.unlock_requirement}")
    payload = result.to_ui_payload()
    print("Buff names:", payload["buff_names"])
    print("Buff values:", payload["buff_values"])


def _parse_crop_env(name: str, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    raw = os.environ.get(name)
    if not raw:
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 4:
        print(f"Invalid {name} value '{raw}', expected four comma-separated numbers. Using default.")
        return default
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        print(f"Non-numeric {name} value '{raw}'. Using default.")
        return default
    return cast(tuple[float, float, float, float], values)


def _main_loop() -> None:
    try:
        from capture_window import capture_frames
    except ImportError as error:  # pragma: no cover - capture only works on Windows
        raise SystemExit(f"capture_window module unavailable: {error}") from error

    screenshot_interval = DETECTION_INTERVAL
    logo_path = Path("upgrade_page_logo.png")
    if not logo_path.exists():
        raise SystemExit(f"Upgrade page logo not found at {logo_path!s}")
    with Image.open(logo_path) as loaded_logo:
        logo_image = loaded_logo.convert("RGB")

    detection_crop = _parse_crop_env("OCR_DETECTION_CROP", DEFAULT_DETECTION_CROP)
    ocr_crop = _parse_crop_env("OCR_VALUE_CROP", DEFAULT_OCR_CROP)
    workflow = BuffWorkflow()

    print("Starting OCR watcher. Press Ctrl+C to stop.")
    try:
        for frame in capture_frames(
            title=WINDOW_TITLE,
            interval=screenshot_interval,
            count=0,
            crop=detection_crop,
        ):
            try:
                if not is_upgrade_page(frame, logo_image):
                    print("Page mismatch; waiting...")
                    continue
                print("Upgrade page detected. Capturing detailed frame...")
                detailed_frame = next(
                    capture_frames(
                        title=WINDOW_TITLE,
                        interval=DETAILED_CAPTURE_INTERVAL,
                        count=1,
                        crop=ocr_crop,
                    )
                )
                _process_and_print(workflow, detailed_frame)
            except StopIteration:
                print("Capture interrupted.")
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


if __name__ == "__main__":
    _main_loop()
