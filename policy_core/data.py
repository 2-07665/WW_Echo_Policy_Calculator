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
    {63: 1036, 69: 964, 75: 1053, 81: 362, 87: 322, 93: 328, 99: 131, 105: 112},  # Crit_Rate
    {126: 995, 138: 1005, 150: 1090, 162: 335, 174: 362, 186: 387, 198: 129, 210: 119},  # Crit_Damage
    {64: 316, 71: 373, 79: 921, 86: 1125, 94: 781, 101: 707, 109: 254, 116: 139},  # Attack
    {81: 318, 90: 413, 100: 995, 109: 1277, 118: 872, 128: 718, 138: 295, 147: 152},  # Defence
    {64: 321, 71: 386, 79: 1005, 86: 1213, 94: 800, 101: 669, 109: 275, 116: 137},  # HP
    {30: 326, 40: 2496, 50: 1838, 60: 120},  # Attack_Flat
    {40: 700, 50: 2128, 60: 1846, 70: 141},  # Defence_Flat
    {320: 298, 360: 419, 390: 971, 430: 1205, 470: 864, 510: 680, 540: 258, 580: 168},  # HP_Flat
    {68: 302, 76: 375, 84: 975, 92: 1199, 100: 871, 108: 643, 116: 274, 124: 126},  # ER
    {64: 316, 71: 360, 79: 959, 86: 1199, 94: 859, 101: 723, 109: 263, 116: 160},  # Basic_Attack_Damage
    {64: 319, 71: 369, 79: 968, 86: 1187, 94: 809, 101: 697, 109: 283, 116: 150},  # Heavy_Attack_Damage
    {64: 328, 71: 357, 79: 978, 86: 1173, 94: 847, 101: 731, 109: 283, 116: 149},  # Skill_Damage
    {64: 292, 71: 358, 79: 973, 86: 1162, 94: 823, 101: 694, 109: 280, 116: 144},  # Ult_Damage
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
