"""Resource cost modelling utilities for the policy calculator."""

from __future__ import annotations

from typing import Final

ECHO_COST: Final[int] = 1
DKQ_COST: Final[int] = 7
EXP_COST_BY_LEVEL: Final[list[int]] = [4400, 16500, 39600, 79100, 142600]
# EXP_COST_BY_LEVEL: Final[list[int]] = [4500, 16500, 40000, 79500, 143000] # rounded to 500
EXP_INCREMENTAL_COSTS: Final[list[int]] = [
    current - (EXP_COST_BY_LEVEL[index - 1] if index > 0 else 0)
    for index, current in enumerate(EXP_COST_BY_LEVEL)
]
EXP_REFUND_RATIO_DEFAULT: Final[float] = 0.66  # 75% is not practical in real runs
_EXP_REFUND_RATIO: float = EXP_REFUND_RATIO_DEFAULT
SUCCESS_DKQ_ADDITIONAL_COST: Final[int] = 15


def _compute_success_exp_additional_cost(ratio: float) -> float:
    """Return the additional EXP cost incurred on success for the supplied ratio."""

    return ratio * EXP_COST_BY_LEVEL[-1] / 5000


def set_exp_refund_ratio(ratio: float) -> None:
    """Update the global EXP refund ratio used by the cost model helpers.

    Parameters
    ----------
    ratio:
        Target refund ratio expressed on ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If the ratio lies outside the inclusive ``[0, 1]`` interval.
    """

    if not 0.0 <= ratio <= 1.0:
        raise ValueError("EXP refund ratio must be between 0 and 1.")
    global _EXP_REFUND_RATIO
    _EXP_REFUND_RATIO = ratio


def get_exp_refund_ratio() -> float:
    """Return the currently configured EXP refund ratio."""

    return _EXP_REFUND_RATIO


def success_exp_additional_cost() -> float:
    """Return the extra EXP cost incurred on a successful run."""

    return _compute_success_exp_additional_cost(_EXP_REFUND_RATIO)


def exp_cost_of_k(level_index: int) -> float:
    """Return the effective EXP cost at a given upgrade depth.

    Parameters
    ----------
    level_index:
        Zero-based index representing the upgrade stage.

    Returns
    -------
    float
        Effective EXP cost scaled by the current refund ratio.
    """

    incremental = EXP_INCREMENTAL_COSTS[level_index]
    return (1 - get_exp_refund_ratio()) * incremental / 5000


class CostModel:
    """Resource weighting model used to evaluate upgrade policies."""

    def __init__(self, w_echo: float, w_dkq: float, w_exp: float) -> None:
        """Initialise the cost model with user-specified weights."""

        self.w_echo = w_echo
        self.w_dkq = w_dkq
        self.w_exp = w_exp

    def fixed_cost(self) -> float:
        """Return the fixed cost paid before any reveal occurs."""

        return self.w_echo * ECHO_COST

    def reveal_cost(self, slot_index: int) -> float:
        """Return the incremental cost paid when revealing the given slot."""

        return self.w_exp * exp_cost_of_k(slot_index) + self.w_dkq * DKQ_COST

    def succ_extra_cost(self) -> float:
        """Return the extra resource cost incurred on a successful run."""

        return self.w_dkq * SUCCESS_DKQ_ADDITIONAL_COST + self.w_exp * success_exp_additional_cost()
