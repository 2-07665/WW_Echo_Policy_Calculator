"""Scoring utilities and PMF builders for buff distributions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .data import (
    BUFF_TYPE_MAX_VALUES,
    BUFF_TYPES,
    MAX_SELECTED_TYPES,
    TOTAL_BUFF_TYPES,
    ProbabilityMassFunction,
    RawCountMap,
)


def create_buff_weight_list(buff_weights: Mapping[str, float]) -> list[float]:
    """Translate a name-to-weight mapping into an ordered weight list.

    Parameters
    ----------
    buff_weights:
        Mapping from buff name to weight. Missing entries default to zero.
    """

    return [float(buff_weights.get(name, 0.0)) for name in BUFF_TYPES]


def make_linear_score_fn(buff_weight_list: Sequence[float]) -> Callable[[int, float], float]:
    """Scale weights so that the maximum achievable total equals 100 points."""

    max_weight = sum(sorted(buff_weight_list, reverse=True)[:MAX_SELECTED_TYPES])
    if max_weight <= 0.0:
        raise ValueError("Sum of top five weights must be positive to build a scorer.")

    def scorer(buff_type_index: int, value: float) -> float:
        return (
            100.0
            * buff_weight_list[buff_type_index]
            / max_weight
            * value
            / BUFF_TYPE_MAX_VALUES[buff_type_index]
        )

    return scorer


def make_simple_score_fn(buff_weight_list: Sequence[float]) -> Callable[[int, float], float]:
    """Return a scorer that assigns fixed weight per buff regardless of raw value."""

    def scorer(buff_type_index: int, _: float) -> float:
        return buff_weight_list[buff_type_index]

    return scorer


def build_score_pmfs_from_counts(
    counts: Sequence[RawCountMap],
    scorer: Callable[[int, float], float],
) -> list[ProbabilityMassFunction]:
    """Convert raw value histograms into score PMFs for each buff type."""

    pmfs: list[ProbabilityMassFunction] = []
    for buff_index in range(TOTAL_BUFF_TYPES):
        raw_counts = counts[buff_index]
        total = sum(raw_counts.values())
        if total <= 0:
            pmfs.append({})
            continue
        probability_lookup: ProbabilityMassFunction = {}
        for raw_value, frequency in raw_counts.items():
            score_bucket = round(100 * scorer(buff_index, float(raw_value)))
            probability_lookup[score_bucket] = (
                probability_lookup.get(score_bucket, 0.0) + (frequency / total)
            )
        pmfs.append(probability_lookup)
    return pmfs
