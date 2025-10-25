"""Domain constants, user-data helpers, and shared type aliases."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

BUFF_TYPES: Final[list[str]] = [
    "Crit_Rate",
    "Crit_Damage",
    "Attack",
    "Defence",
    "HP",
    "Attack_Flat",
    "Defence_Flat",
    "HP_Flat",
    "ER",
    "Basic_Attack_Damage",
    "Heavy_Attack_Damage",
    "Skill_Damage",
    "Ult_Damage",
]

BUFF_LABELS: Final[dict[str, str]] = {
    "Crit_Rate": "暴击",
    "Crit_Damage": "暴击伤害",
    "Attack": "攻击百分比",
    "Defence": "防御百分比",
    "HP": "生命百分比",
    "Attack_Flat": "攻击",
    "Defence_Flat": "防御",
    "HP_Flat": "生命",
    "ER": "共鸣效率",
    "Basic_Attack_Damage": "普攻伤害加成",
    "Heavy_Attack_Damage": "重击伤害加成",
    "Skill_Damage": "共鸣技能伤害加成",
    "Ult_Damage": "共鸣解放伤害加成",
}

BUFF_NAME_TO_INDEX: Final[dict[str, int]] = {name: idx for idx, name in enumerate(BUFF_TYPES)}

TOTAL_BUFF_TYPES: Final[int] = len(BUFF_TYPES)

BUFF_TYPE_COUNTS: Final[list[dict[int, int]]] = [
    {63: 848, 69: 778, 75: 871, 81: 310, 87: 257, 93: 281, 99: 108, 105: 83},  # Crit_Rate
    {126: 813, 138: 825, 150: 879, 162: 272, 174: 296, 186: 311, 198: 107, 210: 93},  # Crit_Damage
    {64: 60, 71: 68, 79: 183, 86: 210, 94: 150, 101: 124, 109: 55, 116: 24},  # Attack
    {81: 56, 90: 91, 100: 178, 109: 244, 118: 180, 128: 121, 138: 56, 147: 33},  # Defence
    {64: 69, 71: 76, 79: 195, 86: 256, 94: 150, 101: 125, 109: 49, 116: 26},  # HP
    {30: 79, 40: 513, 50: 369, 60: 27},  # Attack_Flat
    {40: 150, 50: 412, 60: 389, 70: 27},  # Defence_Flat
    {320: 59, 360: 86, 390: 194, 430: 231, 470: 169, 510: 130, 540: 60, 580: 44},  # HP_Flat
    {68: 66, 76: 72, 84: 199, 92: 249, 100: 177, 108: 111, 116: 55, 124: 26},  # ER
    {64: 62, 71: 93, 79: 197, 86: 222, 94: 176, 101: 149, 109: 56, 116: 38},  # Basic_Attack_Damage
    {64: 58, 71: 73, 79: 199, 86: 239, 94: 159, 101: 132, 109: 56, 116: 25},  # Heavy_Attack_Damage
    {64: 70, 71: 71, 79: 192, 86: 226, 94: 180, 101: 134, 109: 57, 116: 26},  # Skill_Damage
    {64: 66, 71: 62, 79: 184, 86: 232, 94: 171, 101: 140, 109: 69, 116: 34},  # Ult_Damage
]

BUFF_TYPE_MAX_VALUES: Final[list[int]] = [max(counts) for counts in BUFF_TYPE_COUNTS]

MAX_SELECTED_TYPES: Final[int] = 5

ProbabilityMassFunction = dict[int, float]
RawCountMap = dict[int, int]
RawValueEntry = tuple[int, float, int]

DEFAULT_BUFF_WEIGHTS: Final[dict[str, float]] = {
    "Crit_Rate": 100,
    "Crit_Damage": 100,
    "Attack": 70,
    "Defence": 0,
    "HP": 0,
    "Attack_Flat": 30,
    "Defence_Flat": 0,
    "HP_Flat": 0,
    "ER": 10,
    "Basic_Attack_Damage": 0,
    "Heavy_Attack_Damage": 0,
    "Skill_Damage": 0,
    "Ult_Damage": 0,
}

# ---- User-maintained count data ---------------------------------------------

USER_COUNTS_FILENAME: Final[str] = "user_counts_data.json"
USER_COUNTS_JSON_PATH: Final[Path] = Path(__file__).with_name(USER_COUNTS_FILENAME)


def empty_user_buff_type_counts() -> list[dict[int, int]]:
    """Return an empty counts list aligned with BUFF_TYPES length."""

    return [{} for _ in BUFF_TYPES]


def clone_count_maps(raw_counts: Sequence[Mapping[int, int]]) -> list[dict[int, int]]:
    """Return a copy of the supplied count mappings with integer keys and values."""

    return [{int(value): int(amount) for value, amount in mapping.items()} for mapping in raw_counts]


def normalize_user_buff_type_counts(
    raw_counts: Iterable[Mapping[object, object]],
) -> list[dict[int, int]]:
    """Coerce JSON-compatible input into the expected list-of-dicts structure."""

    counts_list = [dict(mapping) for mapping in raw_counts]
    target_len = len(BUFF_TYPES)
    if len(counts_list) < target_len:
        counts_list.extend({} for _ in range(target_len - len(counts_list)))
    elif len(counts_list) > target_len:
        counts_list = counts_list[:target_len]

    normalized: list[dict[int, int]] = []
    for mapping in counts_list:
        converted: dict[int, int] = {}
        for key, value in mapping.items():
            try:
                converted[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
        normalized.append(converted)
    return normalized


def load_user_buff_type_counts(
    path: str | Path | None = None,
) -> list[dict[int, int]]:
    """Load user-maintained buff counts from the JSON file, if present."""

    counts_path = Path(path) if path is not None else USER_COUNTS_JSON_PATH
    try:
        raw_text = counts_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return empty_user_buff_type_counts()
    try:
        raw_data = json.loads(raw_text)
    except json.JSONDecodeError:
        return empty_user_buff_type_counts()

    if not isinstance(raw_data, list):
        return empty_user_buff_type_counts()

    return normalize_user_buff_type_counts(raw_data)


def build_active_counts(
    include_user_counts: bool,
    user_counts: Iterable[Mapping[int, int]] | None = None,
) -> list[dict[int, int]]:
    """Combine built-in counts with optional user data."""

    base_counts = clone_count_maps(BUFF_TYPE_COUNTS)
    if not include_user_counts or user_counts is None:
        return base_counts
    extra_counts = normalize_user_buff_type_counts(user_counts)
    for idx, extra_mapping in enumerate(extra_counts):
        for value, amount in extra_mapping.items():
            base_counts[idx][value] = base_counts[idx].get(value, 0) + amount
    return base_counts


def save_user_buff_type_counts(
    counts: Iterable[Mapping[int, int]],
    path: str | Path | None = None,
) -> None:
    """Persist user-maintained buff counts to the JSON file."""

    counts_path = Path(path) if path is not None else USER_COUNTS_JSON_PATH
    normalized = normalize_user_buff_type_counts(counts)
    counts_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = [
        {str(value): int(amount) for value, amount in sorted(mapping.items())}
        for mapping in normalized
    ]
    counts_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def load_character_presets(
    preset_path: str | Path | None,
) -> dict[str, dict[str, float]]:
    """Load character buff weight presets from the given JSON file."""

    if not preset_path:
        return {}

    path = Path(preset_path)
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw_data, Mapping):
        return {}

    presets: dict[str, dict[str, float]] = {}
    for name, weights in raw_data.items():
        if not isinstance(name, str) or not isinstance(weights, Mapping):
            continue

        parsed_weights: dict[str, float] = {}
        for buff_name, value in weights.items():
            if buff_name not in BUFF_TYPES:
                continue
            try:
                parsed_weights[buff_name] = float(value)
            except (TypeError, ValueError):
                continue

        if parsed_weights:
            presets[name] = parsed_weights

    return presets
