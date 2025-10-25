"""High-level entry points used by the UI and callers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from .cost import CostModel
from .data import BUFF_NAME_TO_INDEX, BUFF_TYPE_COUNTS
from .models import FirstUpgradeGroup, SimulationSummary
from .scoring import (
    build_score_pmfs_from_counts,
    create_buff_weight_list,
    make_linear_score_fn,
    make_simple_score_fn,
)
from .simulation import simulate_many
from .solver import PolicySolver


def build_scorer(buff_weights: Mapping[str, float]) -> Callable[[int, float], float]:
    """Return the linear scoring function consistent with the DP solver.

    Parameters
    ----------
    buff_weights:
        Mapping from buff type name to weight assigned by the user.
    """

    weight_list = create_buff_weight_list(buff_weights)
    return make_linear_score_fn(weight_list)


def build_simple_scorer(buff_weights: Mapping[str, float]) -> Callable[[int, float], float]:
    """Return the simple scoring function for fixed-per-type contributions.

    Parameters
    ----------
    buff_weights:
        Mapping from buff type name to weight assigned by the user.
    """

    weight_list = create_buff_weight_list(buff_weights)
    return make_simple_score_fn(weight_list)


def buff_names_to_indices(names: Sequence[str]) -> list[int]:
    """Map buff type names to their canonical indices.

    Parameters
    ----------
    names:
        Buff type names chosen by the caller.

    Returns
    -------
    list[int]
        Indices corresponding to the supplied buff names.

    Raises
    ------
    ValueError
        If an unknown buff name is supplied.
    """

    indices: list[int] = []
    for name in names:
        try:
            indices.append(BUFF_NAME_TO_INDEX[name])
        except KeyError as exc:
            raise ValueError(f"Unknown buff name '{name}'") from exc
    return indices


def score_to_int(score: float) -> int:
    """Convert a displayed score to the integer grid used by the solver."""

    return int(round(score * 100))


def make_cost_model(w_echo: float, w_dkq: float, w_exp: float) -> CostModel:
    """Factory helper that keeps the UI decoupled from the solver class."""

    return CostModel(w_echo=w_echo, w_dkq=w_dkq, w_exp=w_exp)


@dataclass
class PolicyComputationResult:
    """Bundle containing the solver and derived reporting artefacts."""

    solver: PolicySolver
    target_score: float
    lambda_star: float
    expected_cost_per_success: float
    compute_seconds: float
    simulation: Optional[SimulationSummary]
    cost_model: CostModel
    first_upgrade_table: list[FirstUpgradeGroup]


def compute_optimal_policy(
    buff_weights: Mapping[str, float],
    target_score: float,
    cost_model: Optional[CostModel] = None,
    simulation_runs: int = 0,
    simulation_seed: int = 42,
    buff_type_counts: Optional[Sequence[Mapping[int, int]]] = None,
) -> PolicyComputationResult:
    """Compute the optimal policy and optional simulation summary.

    Parameters
    ----------
    buff_weights:
        User-specified weights per buff type.
    target_score:
        Total score threshold to treat a run as successful.
    cost_model:
        Optional override for the default cost weighting model.
    simulation_runs:
        Number of Monte Carlo runs to execute (0 disables simulation).
    simulation_seed:
        Seed forwarded to the RNG used for simulations.
    buff_type_counts:
        Optional replacement for the built-in buff type count data. When supplied it must
        contain an entry for every buff type defined in ``BUFF_TYPES``.

    Returns
    -------
    PolicyComputationResult
        Bundle containing the solved policy, cost model, and optional simulation stats.
    """

    if cost_model is None:
        cost_model = CostModel(w_echo=0.0, w_dkq=0.5, w_exp=2.06)

    if buff_type_counts is None:
        active_counts = [dict(counts) for counts in BUFF_TYPE_COUNTS]
    else:
        active_counts = [
            {int(value): int(amount) for value, amount in counts.items()}
            for counts in buff_type_counts
        ]
        expected_len = len(BUFF_TYPE_COUNTS)
        if len(active_counts) != expected_len:
            raise ValueError(
                f"buff_type_counts must contain {expected_len} entries, "
                f"received {len(active_counts)}"
            )

    scorer = build_scorer(buff_weights)
    pmfs = build_score_pmfs_from_counts(active_counts, scorer)
    solver = PolicySolver(
        pmfs=pmfs,
        target=target_score,
        cost_model=cost_model,
        raw_counts=active_counts,
        scorer_fn=scorer,
    )

    compute_start = perf_counter()
    lambda_star = solver.lambda_search()
    compute_seconds = perf_counter() - compute_start

    first_upgrade_table = solver.continue_after_first()

    succ_extra_cost = cost_model.succ_extra_cost()
    expected_cost_per_success = (
        float("inf") if lambda_star <= 0 else (1.0 / lambda_star + succ_extra_cost)
    )

    simulation: Optional[SimulationSummary] = None
    if simulation_runs > 0:
        (
            success_probability,
            echo_per_success,
            dkq_per_success,
            exp_per_success,
            max_slot_scores,
            total_runs,
        ) = simulate_many(
            solver=solver,
            runs=simulation_runs,
            seed=simulation_seed,
        )
        simulation = SimulationSummary(
            success_rate=success_probability,
            echo_per_success=echo_per_success,
            dkq_per_success=dkq_per_success,
            exp_per_success=exp_per_success,
            max_slot_scores=max_slot_scores,
            total_runs=total_runs,
        )

    return PolicyComputationResult(
        solver=solver,
        target_score=target_score,
        lambda_star=lambda_star,
        expected_cost_per_success=expected_cost_per_success,
        compute_seconds=compute_seconds,
        simulation=simulation,
        cost_model=cost_model,
        first_upgrade_table=first_upgrade_table,
    )
