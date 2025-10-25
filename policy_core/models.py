"""Dataclasses shared across policy and simulation modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FirstUpgradeOption:
    """Probability of continuing after the first reveal for a given value."""

    raw_value: int
    score: float
    probability: float


@dataclass
class FirstUpgradeGroup:
    """Group of viable first-slot outcomes for a specific buff type."""

    buff_name: str
    options: list[FirstUpgradeOption]


@dataclass
class SimulationSummary:
    """Aggregated Monte Carlo metrics for a computed policy."""

    success_rate: float
    echo_per_success: float
    dkq_per_success: float
    exp_per_success: float
    max_slot_scores: list[int]
    total_runs: int
