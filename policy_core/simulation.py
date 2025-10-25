"""Monte Carlo helpers and heuristic policies for upgrade simulations."""

from __future__ import annotations

import random
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable
from typing import Optional

from .cost import DKQ_COST, SUCCESS_DKQ_ADDITIONAL_COST, success_exp_additional_cost
from .data import TOTAL_BUFF_TYPES
from .solver import PolicySolver

DecisionFn = Callable[[list[int], int], bool]


def simulate_once(
    solver: PolicySolver,
    rng: random.Random,
    decision: Optional[DecisionFn] = None,
) -> tuple[bool, float, float, int, Optional[int]]:
    """Simulate a single upgrade run using the supplied policy decision.

    Parameters
    ----------
    solver:
        Policy solver providing scoring tables and decisions.
    rng:
        Deterministic random number generator.
    decision:
        Optional override for the solver's decision rule.
    """

    policy = decision or solver.decision
    remaining_types = list(range(TOTAL_BUFF_TYPES))
    used_types: list[int] = []
    cumulative_score = 0
    dkq_cost = 0.0
    exp_cost = 0.0
    for _ in range(solver.max_slots):
        if not used_types:
            should_continue = True
        else:
            should_continue = policy(used_types, cumulative_score)
        if not should_continue:
            return False, dkq_cost, exp_cost, cumulative_score, None
        slot_index = len(used_types)
        dkq_cost += DKQ_COST
        exp_cost += solver.exp_cost_for_slot(slot_index)
        choice_index = rng.randrange(len(remaining_types))
        chosen_type = remaining_types.pop(choice_index)
        used_types.append(chosen_type)
        scores = solver.type_scores[chosen_type]
        cumulative = solver.type_cumulative_probabilities[chosen_type]
        sample = rng.random()
        pick_index = bisect_left(cumulative, sample)
        if pick_index >= len(scores):
            pick_index = len(scores) - 1
        sampled_value = scores[pick_index]
        cumulative_score += sampled_value
    success = cumulative_score >= solver.target_score
    return success, dkq_cost, exp_cost, cumulative_score, cumulative_score


def simulate_many(
    solver: PolicySolver,
    runs: int = 10000,
    seed: int = 42,
    decision: Optional[DecisionFn] = None,
) -> tuple[float, float, float, float, list[int], int]:
    """Run Monte Carlo simulation to estimate success rate, costs, and full-run scores."""

    rng = random.Random(seed)
    successes = 0
    total_dkq_cost = 0.0
    total_exp_cost = 0.0
    max_slot_scores: list[int] = []
    for _ in range(runs):
        success, dkq_cost, exp_cost, _, max_slot_score = simulate_once(
            solver, rng, decision=decision
        )
        total_dkq_cost += dkq_cost
        total_exp_cost += exp_cost
        if success:
            successes += 1
        if max_slot_score is not None:
            max_slot_scores.append(max_slot_score)

    success_rate = successes / runs if runs > 0 else 0.0
    echo_per_success = runs / successes if successes > 0 else float("inf")
    dkq_cost_per_success = (
        SUCCESS_DKQ_ADDITIONAL_COST + (total_dkq_cost / successes)
        if successes > 0
        else float("inf")
    )
    exp_cost_per_success = (
        success_exp_additional_cost() + (total_exp_cost / successes)
        if successes > 0
        else float("inf")
    )
    return (
        success_rate,
        echo_per_success,
        dkq_cost_per_success,
        exp_cost_per_success,
        max_slot_scores,
        runs,
    )


def always_upgrade(_: list[int], __: int) -> bool:
    """Baseline policy that never abandons (for comparison)."""

    return True


def upgrade_until_impossible(solver: PolicySolver) -> DecisionFn:
    """Return a decision rule that continues while success remains feasible."""

    def _decision(used_types: list[int], score: int) -> bool:
        mask = solver.mask_for_used_types(used_types)
        remaining_slots = solver.max_slots - len(used_types)
        best_case = solver.best_case_sum(mask, remaining_slots)
        return score + best_case >= solver.target_score

    return _decision


def strategy_two_one_two(used_types: list[int], score: int) -> bool:
    """Heuristic policy matching community '2-1-2' guidance."""

    if score < 0:  # guard for unexpected inputs
        return False

    slot_count = len(used_types)
    counts = Counter(used_types)
    crit_counts = counts[0] + counts[1]
    sustain_counts = counts[4] + counts[7] + counts[9]

    if slot_count == 0:
        return True
    if slot_count in (1, 2) and (crit_counts + sustain_counts) >= 1:
        return True
    if slot_count == 3 and crit_counts >= 1:
        return True
    if slot_count == 4 and (
        crit_counts == 2 or (crit_counts >= 1 and sustain_counts >= 1)
    ):
        return True
    return False
