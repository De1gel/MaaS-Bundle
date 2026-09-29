"""Shared numerical defaults and run-wide cost feasibility policy.

Cost and margin tolerances are in per-traveler monetary units. Objective
profit tolerances are in weighted aggregate profit units; they are independent.
"""

from __future__ import annotations

import numpy as np

DEFAULT_COST_TOLERANCE = 1e-7
DEFAULT_MARGIN_TOLERANCE = 1e-9
DEFAULT_PROFIT_TOLERANCE = 1e-9


def validate_tolerance(value: float, *, name: str) -> None:
    """Reject invalid tolerances before taking maxima or comparing values."""

    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def effective_cost_tolerance(
    solver: object,
    *,
    choice_tolerance: float = DEFAULT_COST_TOLERANCE,
    violation_tolerance: float = DEFAULT_COST_TOLERANCE,
) -> float:
    """Fix one cost threshold before evaluating the initial assignment.

    Adapters may advertise ``feasibility_tolerance``. Legacy adapters without
    that property use the shared default. A later solve certificate never
    silently widens this threshold: actual residuals must pass the fixed policy.
    """

    solver_tolerance = getattr(solver, "feasibility_tolerance", DEFAULT_COST_TOLERANCE)
    for name, value in (
        ("choice_tolerance", choice_tolerance),
        ("violation_tolerance", violation_tolerance),
        ("solver.feasibility_tolerance", solver_tolerance),
    ):
        validate_tolerance(value, name=name)
    return float(max(choice_tolerance, violation_tolerance, solver_tolerance))
