"""Per-subscriber provision costs, independent of route valuation and prices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ProvisionCost(Protocol):
    """Batch interface for separable, possibly bundle-specific costs K_b.

    Inputs have shape (B, M), containing subscriptions 1..B, without PAYG.
    values returns (B,); gradients returns (B, M), row b being dK_b/d epsilon_b.
    Each output cost must depend only on its own input row, but can couple
    modes arbitrarily within that row. Costs must be finite, nonnegative and
    zero at zero allowance. No convexity or monotonicity is assumed.
    """

    @property
    def mode_count(self) -> int:
        """Number of allowance dimensions."""

    def values(self, allowances: NDArray[np.float64]) -> ArrayLike:
        """Return per-subscriber provision costs, one per subscription."""

    def gradients(self, allowances: NDArray[np.float64]) -> ArrayLike:
        """Return allowance derivatives for each subscription."""


@dataclass(frozen=True)
class LinearProvisionCost:
    """Shared linear cost K_b(epsilon_b) = k @ epsilon_b."""

    unit_cost: NDArray[np.float64]

    def __post_init__(self) -> None:
        cost = np.asarray(self.unit_cost, dtype=float)
        if cost.ndim != 1 or cost.size == 0:
            raise ValueError("unit_allowance_cost must be a nonempty vector")
        if not np.all(np.isfinite(cost)) or np.any(cost < 0):
            raise ValueError("unit_allowance_cost must be finite and nonnegative")
        cost = cost.copy()
        cost.setflags(write=False)
        object.__setattr__(self, "unit_cost", cost)

    @property
    def mode_count(self) -> int:
        return self.unit_cost.size

    def values(self, allowances: NDArray[np.float64]) -> NDArray[np.float64]:
        return allowances @ self.unit_cost

    def gradients(self, allowances: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.broadcast_to(self.unit_cost, allowances.shape)


@dataclass(frozen=True)
class CallableProvisionCost:
    """Adapt batch cost/gradient functions; optionally approximate derivatives.

    All subscription rows are passed in their original order, allowing
    bundle-specific coefficients. Callbacks must be deterministic, must not
    mutate inputs and must obey ProvisionCost's row-separability contract.
    Without gradient, bounded three-point differences perturb one mode at a
    time across all rows, requiring roughly 2*M value evaluations per gradient.
    At zero allowance forward differences avoid negative-domain evaluations.

    Smooth functions work with the existing NLP adapters. Piecewise or
    discontinuous functions can be evaluated, but numerical derivatives do not
    provide smooth-solver convergence or local-optimality guarantees at kinks.
    """

    mode_count: int
    value: Callable[[NDArray[np.float64]], ArrayLike]
    gradient: Callable[[NDArray[np.float64]], ArrayLike] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode_count, (int, np.integer)) or self.mode_count <= 0:
            raise ValueError("mode_count must be a positive integer")
        if not callable(self.value):
            raise TypeError("value must be callable")
        if self.gradient is not None and not callable(self.gradient):
            raise TypeError("gradient must be callable or None")

    def values(self, allowances: NDArray[np.float64]) -> ArrayLike:
        return self.value(allowances)

    def gradients(self, allowances: NDArray[np.float64]) -> ArrayLike:
        if self.gradient is not None:
            return self.gradient(allowances)
        result = np.empty_like(allowances)
        step = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(allowances))
        base = None
        for mode in range(self.mode_count):
            h = step[:, mode]
            central = allowances[:, mode] >= h
            plus = allowances.copy()
            plus[:, mode] += h
            plus.setflags(write=False)
            # Callbacks may reuse a work buffer between evaluations.
            high = checked_cost_values(self, plus).copy()
            other = allowances.copy()
            other[:, mode] += np.where(central, -h, 2 * h)
            other.setflags(write=False)
            low_or_far = checked_cost_values(self, other).copy()
            result[:, mode] = (high - low_or_far) / (2 * h)
            if not np.all(central):
                if base is None:
                    base = checked_cost_values(self, allowances).copy()
                result[~central, mode] = (
                    -3 * base[~central] + 4 * high[~central] - low_or_far[~central]
                ) / (2 * h[~central])
        return result


def checked_cost_values(cost: ProvisionCost, allowances: NDArray[np.float64]) -> NDArray:
    """Validate evaluated values at the boundary to user-defined cost code."""

    values = np.asarray(cost.values(allowances), dtype=float)
    if values.shape != (allowances.shape[0],):
        raise ValueError("provision cost values must have shape (B,)")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("provision cost values must be finite and nonnegative")
    return values


def checked_cost_gradients(cost: ProvisionCost, allowances: NDArray[np.float64]) -> NDArray:
    """Validate derivatives without imposing monotonicity on the costs."""

    gradients = np.asarray(cost.gradients(allowances), dtype=float)
    if gradients.shape != allowances.shape:
        raise ValueError("provision cost gradients must have shape (B, M)")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("provision cost gradients must be finite")
    return gradients
