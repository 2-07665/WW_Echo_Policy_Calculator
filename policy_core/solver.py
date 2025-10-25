"""Dynamic-programming solver that searches for optimal upgrade policies."""

from __future__ import annotations

from collections.abc import Callable
from itertools import accumulate
from typing import Optional

from .cost import CostModel, exp_cost_of_k
from .data import (
    BUFF_TYPES,
    MAX_SELECTED_TYPES,
    TOTAL_BUFF_TYPES,
    ProbabilityMassFunction,
    RawCountMap,
    RawValueEntry,
)
from .models import FirstUpgradeGroup, FirstUpgradeOption


class PolicySolver:
    """Dynamic-programming solver that searches for the optimal upgrade policy."""

    def __init__(
        self,
        pmfs: list[ProbabilityMassFunction],
        target: float,
        cost_model: Optional[CostModel] = None,
        raw_counts: Optional[list[RawCountMap]] = None,
        scorer_fn: Optional[Callable[[int, float], float]] = None,
    ) -> None:
        """Initialise the solver state and pre-compute lookup structures.

        Parameters
        ----------
        pmfs:
            Score probability mass functions per buff type.
        target:
            Desired score threshold on the original (float) scale.
        cost_model:
            Weighting model applied to resource costs (defaults to a baseline).
        raw_counts:
            Optional raw histogram counts used for first-upgrade summaries.
        scorer_fn:
            Optional scorer consistent with ``raw_counts`` for interpreting raw values.
        """
        self.pmfs = pmfs
        self.type_vals = [tuple(p.items()) for p in self.pmfs]
        self._type_scores = [tuple(score for score, _ in values) for values in self.type_vals]
        self._type_probabilities = [tuple(prob for _, prob in values) for values in self.type_vals]
        cumulative_tables: list[tuple[float, ...]] = []
        for probs in self._type_probabilities:
            cumulative = list(accumulate(probs))
            if cumulative:
                cumulative[-1] = 1.0
            cumulative_tables.append(tuple(cumulative))
        self._type_cumulative_probabilities = cumulative_tables
        self._exp_cost_cache = [exp_cost_of_k(k) for k in range(MAX_SELECTED_TYPES)]
        self.type_max_score = [max(p.keys()) for p in pmfs]
        self.target = round(100 * target)
        self._lam = 0.0
        self._raw_counts = raw_counts
        self._scorer_fn = scorer_fn

        if cost_model is None:
            cost_model = CostModel(w_echo=0.0, w_dkq=0.5, w_exp=2.06)
        self.cost = cost_model
        self._fixed = self.cost.fixed_cost()

        self._max_slots = min(MAX_SELECTED_TYPES, TOTAL_BUFF_TYPES)
        self._all_mask = (1 << TOTAL_BUFF_TYPES) - 1
        self._reveal_cost_cache = [self.cost.reveal_cost(k) for k in range(self._max_slots)]
        self._mask_cache: dict[tuple[int, ...], int] = {}

        mask_count = self._all_mask + 1
        self._remaining_types: list[tuple[int, ...]] = [tuple() for _ in range(mask_count)]
        self._best_case_lookup: list[list[int]] = [
            [0] * (self._max_slots + 1) for _ in range(mask_count)
        ]
        self._popcount: list[int] = [0] * mask_count
        self._inv_remaining: list[float] = [0.0] * mask_count
        all_indices = list(range(TOTAL_BUFF_TYPES))
        for mask in range(mask_count):
            if mask:
                self._popcount[mask] = self._popcount[mask >> 1] + (mask & 1)
            remaining = [idx for idx in all_indices if not (mask & (1 << idx))]
            self._remaining_types[mask] = tuple(remaining)
            remaining_count = TOTAL_BUFF_TYPES - self._popcount[mask]
            self._inv_remaining[mask] = 0.0 if remaining_count == 0 else 1.0 / remaining_count
            sorted_scores = sorted(
                (self.type_max_score[idx] for idx in remaining), reverse=True
            )
            prefix = [0]
            for sc in sorted_scores:
                prefix.append(prefix[-1] + sc)
            scores_len = len(sorted_scores)
            for r in range(self._max_slots + 1):
                capped = r if r <= scores_len else scores_len
                self._best_case_lookup[mask][r] = prefix[capped]

        self._V_cache: list[dict[int, float]] = [dict() for _ in range(mask_count)]
        self._decision_cache: list[dict[int, bool]] = [dict() for _ in range(mask_count)]
        self._raw_value_cache: Optional[list[list[RawValueEntry]]] = None
        self._V: Callable[[int, int], float]

        if self._raw_counts is not None and self._scorer_fn is not None:
            cache: list[list[RawValueEntry]] = []
            for idx, counts in enumerate(self._raw_counts):
                total = sum(counts.values())
                entries: list[RawValueEntry] = []
                for raw_value, freq in counts.items():
                    prob = freq / total if total > 0 else 0.0
                    score_int = round(100 * self._scorer_fn(idx, float(raw_value)))
                    entries.append((int(raw_value), prob, score_int))
                entries.sort(key=lambda item: item[0], reverse=True)
                cache.append(entries)
            self._raw_value_cache = cache

    def _clear(self) -> None:
        """Reset cached value and decision tables."""

        for bucket in self._V_cache:
            bucket.clear()
        for bucket in self._decision_cache:
            bucket.clear()

    @property
    def max_slots(self) -> int:
        """Return the maximum number of upgrade slots the solver evaluates."""

        return self._max_slots

    @property
    def type_scores(self) -> list[tuple[int, ...]]:
        """Return the discrete score tables used for each buff type."""

        return self._type_scores

    @property
    def type_cumulative_probabilities(self) -> list[tuple[float, ...]]:
        """Return cumulative probability tables aligned with `type_scores`."""

        return self._type_cumulative_probabilities

    def exp_cost_for_slot(self, slot_index: int) -> float:
        """Return the EXP cost applied when revealing the given slot."""

        if 0 <= slot_index < len(self._exp_cost_cache):
            return self._exp_cost_cache[slot_index]
        return exp_cost_of_k(slot_index)

    @property
    def target_score(self) -> int:
        """Return the target score on the integer grid used internally."""

        return self.target

    @staticmethod
    def _indices_to_mask(indices: list[int]) -> int:
        """Convert a list of buff indices to the corresponding bitmask."""

        mask = 0
        for idx in indices:
            mask |= 1 << idx
        return mask

    def mask_for_used_types(self, used_types: list[int]) -> int:
        """Return a cached mask for the supplied used type sequence."""

        key = tuple(sorted(used_types))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        mask = self._indices_to_mask(list(key))
        self._mask_cache[key] = mask
        return mask

    def best_case_sum(self, mask: int, r: int) -> int:
        """Return the best-case cumulative score achievable with `r` slots left."""

        if r <= 0:
            return 0
        return self._best_case_lookup[mask][r]

    def _score_key(self, score: int) -> int:
        """Return the canonical score bucket used for memoisation."""

        return score if score < self.target else self.target

    def _make_V(self) -> None:
        """Bootstrap the recursive value function closure for the current λ."""

        lam = self._lam
        value_cache = self._V_cache
        decision_cache = self._decision_cache
        remaining_types = self._remaining_types
        type_vals = self.type_vals
        reveal_costs = self._reveal_cost_cache
        max_slots = self._max_slots
        fixed_cost = self._fixed
        inv_remaining = self._inv_remaining
        target = self.target
        popcount = self._popcount
        best_case_lookup = self._best_case_lookup

        def value(mask: int, score: int) -> float:
            key_score = score if score < target else target
            cache_bucket = value_cache[mask]
            cached = cache_bucket.get(key_score)
            if cached is not None:
                return cached

            used_slots = popcount[mask]
            remaining_slots = max_slots - used_slots
            decision_bucket = decision_cache[mask]
            if remaining_slots == 0:
                result = 1.0 if score >= target else 0.0
                cache_bucket[key_score] = result
                decision_bucket[key_score] = False
                return result

            if score >= target:
                score = target

            best_case_total = best_case_lookup[mask][remaining_slots]
            if score + best_case_total < target:
                cache_bucket[key_score] = 0.0
                decision_bucket[key_score] = False
                return 0.0

            total_probability = 0.0
            for index in remaining_types[mask]:
                next_mask = mask | (1 << index)
                for buff_score, probability in type_vals[index]:
                    total_probability += probability * value(next_mask, score + buff_score)

            reveal_cost = (
                reveal_costs[used_slots]
                if used_slots < len(reveal_costs)
                else self.cost.reveal_cost(used_slots)
            )
            if used_slots == 0:
                reveal_cost += fixed_cost
            advantage = -lam * reveal_cost + total_probability * inv_remaining[mask]

            if advantage > 0.0:
                cache_bucket[key_score] = advantage
                decision_bucket[key_score] = True
            else:
                cache_bucket[key_score] = 0.0
                decision_bucket[key_score] = False
            return cache_bucket[key_score]

        self._V = value

    def evaluate_lambda(self, lam: float) -> float:
        """Evaluate the value function at the specified λ and return V(Ø, 0)."""

        self._lam = lam
        self._clear()
        self._make_V()
        return self._V(0, 0)

    def derive_policy_at_lambda(self, lam: float) -> None:
        """Populate caches for the supplied λ without returning the objective."""

        self.evaluate_lambda(lam)

    def lambda_search(
        self,
        lo: float = 0.0,
        hi: float = 1.0,
        tol: float = 1e-7,
        max_iter: int = 100,
    ) -> float:
        """Binary search λ such that V(Ø, 0) crosses zero."""

        value_low = self.evaluate_lambda(lo)
        value_high = self.evaluate_lambda(hi)
        widen = 0
        while value_high > 0 and widen < 20:
            hi *= 2.0
            value_high = self.evaluate_lambda(hi)
            widen += 1

        iteration = 0
        while hi - lo > tol and iteration < max_iter:
            mid = 0.5 * (lo + hi)
            value_mid = self.evaluate_lambda(mid)
            if value_mid > 0:
                lo = mid
            else:
                hi = mid
            iteration += 1

        lam_star = 0.5 * (lo + hi)
        self.derive_policy_at_lambda(lam_star)
        return lam_star

    def decision(self, used_types: list[int], score: int) -> bool:
        """Return True when the policy advises continuing with the next reveal."""

        if not used_types:
            return True

        mask = self.mask_for_used_types(used_types)
        score_key = self._score_key(score)
        bucket = self._decision_cache[mask]
        if score_key not in bucket:
            _ = self._V(mask, score)
        return bucket[score_key]

    def decision_output(self, used_types: list[int], score: int) -> str:
        """Return a human-readable recommendation for the current run state."""

        if not used_types:
            return "Continue"
        if len(used_types) >= self._max_slots:
            return "达成目标分数！" if score >= self.target else "下次好运"
        mask = self.mask_for_used_types(used_types)
        score_key = self._score_key(score)
        bucket = self._decision_cache[mask]
        if score_key not in bucket:
            _ = self._V(mask, score)
        return "Continue" if bucket[score_key] else "Abandon"

    def continue_after_first(self) -> list[FirstUpgradeGroup]:
        """Summarise first-slot outcomes that keep the policy in 'continue'."""

        if not hasattr(self, "_V"):
            raise RuntimeError(
                "Policy not decided; call lambda_search or derive_policy_at_lambda first."
            )

        root_bucket = self._decision_cache[0]
        if 0 not in root_bucket:
            _ = self._V(0, 0)

        if self._raw_value_cache is None:
            raise RuntimeError(
                "Raw value cache unavailable; ensure solver was constructed with "
                "raw_counts and scorer_fn."
            )

        groups: list[FirstUpgradeGroup] = []
        for idx, name in enumerate(BUFF_TYPES):
            outcomes: list[FirstUpgradeOption] = []
            for raw_value, raw_prob, score_int in self._raw_value_cache[idx]:
                if raw_prob <= 0.0:
                    continue
                if self.decision([idx], score_int):
                    outcomes.append(
                        FirstUpgradeOption(
                            raw_value=raw_value,
                            score=score_int / 100.0,
                            probability=raw_prob,
                        )
                    )
            if outcomes:
                groups.append(
                    FirstUpgradeGroup(
                        buff_name=name,
                        options=outcomes,
                    )
                )
        return groups
