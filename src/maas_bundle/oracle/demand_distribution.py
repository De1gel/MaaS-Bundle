"""Trip-frequency distributions used by expected route valuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class TripFrequencyDistribution(Protocol):
    """Distribution interface needed to integrate a piecewise-affine value.

    Intervals use the convention ``[lower, upper)``. The final interval may
    have ``upper = infinity``. Implementations must represent a nonnegative
    random variable with finite first moment.
    """

    def probability(self, lower: float, upper: float) -> float:
        """Return ``P(lower <= mu < upper)``."""

    def first_moment(self, lower: float, upper: float) -> float:
        """Return ``E[mu * 1{lower <= mu < upper}]``."""


@dataclass(frozen=True)
class UniformDemandDistribution:
    """Continuous uniform trip-frequency distribution on ``[lower, upper]``."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate the finite, nonnegative support."""

        lower = float(self.lower)
        upper = float(self.upper)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("uniform support bounds must be finite")
        if lower < 0:
            raise ValueError("uniform lower support must be nonnegative")
        if upper <= lower:
            raise ValueError("uniform upper support must be greater than lower support")

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def probability(self, lower: float, upper: float) -> float:
        """Return the probability in ``[lower, upper)``."""

        interval_lower, interval_upper = _validate_interval(lower, upper)
        overlap_lower = max(interval_lower, self.lower)
        overlap_upper = min(interval_upper, self.upper)
        if overlap_upper <= overlap_lower:
            return 0.0
        return (overlap_upper - overlap_lower) / (self.upper - self.lower)

    def first_moment(self, lower: float, upper: float) -> float:
        """Return the unnormalized first moment in ``[lower, upper)``."""

        interval_lower, interval_upper = _validate_interval(lower, upper)
        overlap_lower = max(interval_lower, self.lower)
        overlap_upper = min(interval_upper, self.upper)
        if overlap_upper <= overlap_lower:
            return 0.0
        return 0.5 * (overlap_upper**2 - overlap_lower**2) / (self.upper - self.lower)


@dataclass(frozen=True)
class DiscreteDemandDistribution:
    """Finite empirical trip-frequency distribution."""

    support: NDArray[np.float64]
    probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate, sort, and protect the support and probability masses."""

        support = np.asarray(self.support, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if support.ndim != 1 or support.size == 0:
            raise ValueError("support must be a nonempty vector")
        if probabilities.shape != support.shape:
            raise ValueError("probabilities must have one entry per support value")
        if not np.all(np.isfinite(support)) or np.any(support < 0):
            raise ValueError("support must be finite and nonnegative")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("probabilities must be finite and nonnegative")
        if not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("probabilities must sum to one")

        order = np.argsort(support, kind="stable")
        immutable_support = support[order].copy()
        immutable_probabilities = probabilities[order].copy()
        immutable_support.setflags(write=False)
        immutable_probabilities.setflags(write=False)
        object.__setattr__(self, "support", immutable_support)
        object.__setattr__(self, "probabilities", immutable_probabilities)

    def probability(self, lower: float, upper: float) -> float:
        """Return the probability mass in ``[lower, upper)``."""

        interval_lower, interval_upper = _validate_interval(lower, upper)
        selected = (self.support >= interval_lower) & (self.support < interval_upper)
        return float(np.sum(self.probabilities[selected]))

    def first_moment(self, lower: float, upper: float) -> float:
        """Return the unnormalized first moment in ``[lower, upper)``."""

        interval_lower, interval_upper = _validate_interval(lower, upper)
        selected = (self.support >= interval_lower) & (self.support < interval_upper)
        return float(self.support[selected] @ self.probabilities[selected])


@dataclass(frozen=True)
class TriangularDemandDistribution:
    """Continuous triangular trip-frequency distribution on finite support."""

    lower: float
    mode: float
    upper: float

    def __post_init__(self) -> None:
        """Require a finite, nonnegative support with an interior mode."""

        lower = float(self.lower)
        mode = float(self.mode)
        upper = float(self.upper)
        if not np.isfinite(lower) or not np.isfinite(mode) or not np.isfinite(upper):
            raise ValueError("triangular parameters must be finite")
        if lower < 0:
            raise ValueError("triangular lower support must be nonnegative")
        if upper <= lower:
            raise ValueError("triangular upper support must be greater than lower support")
        if not lower < mode < upper:
            raise ValueError("triangular mode must lie strictly inside its support")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "upper", upper)

    def probability(self, lower: float, upper: float) -> float:
        """Return the probability in ``[lower, upper)``."""

        interval_lower, interval_upper = _validate_interval(lower, upper)
        clipped_lower = max(interval_lower, self.lower)
        clipped_upper = min(interval_upper, self.upper)
        if clipped_upper <= clipped_lower:
            return 0.0
        return self._left_probability(clipped_lower, min(clipped_upper, self.mode)) + (
            self._right_probability(max(clipped_lower, self.mode), clipped_upper)
        )

    def first_moment(self, lower: float, upper: float) -> float:
        """Return the unnormalized first moment in ``[lower, upper)``."""

        interval_lower, interval_upper = _validate_interval(lower, upper)
        clipped_lower = max(interval_lower, self.lower)
        clipped_upper = min(interval_upper, self.upper)
        if clipped_upper <= clipped_lower:
            return 0.0
        return self._left_first_moment(clipped_lower, min(clipped_upper, self.mode)) + (
            self._right_first_moment(max(clipped_lower, self.mode), clipped_upper)
        )

    def _left_probability(self, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.0
        denominator = (self.upper - self.lower) * (self.mode - self.lower)
        return ((upper - self.lower) ** 2 - (lower - self.lower) ** 2) / denominator

    def _right_probability(self, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.0
        denominator = (self.upper - self.lower) * (self.upper - self.mode)
        def primitive(value: float) -> float:
            return self.upper * value - 0.5 * value**2

        return 2.0 * (primitive(upper) - primitive(lower)) / denominator

    def _left_first_moment(self, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.0
        denominator = (self.upper - self.lower) * (self.mode - self.lower)
        def primitive(value: float) -> float:
            return value**3 / 3.0 - 0.5 * self.lower * value**2

        return 2.0 * (primitive(upper) - primitive(lower)) / denominator

    def _right_first_moment(self, lower: float, upper: float) -> float:
        if upper <= lower:
            return 0.0
        denominator = (self.upper - self.lower) * (self.upper - self.mode)
        def primitive(value: float) -> float:
            return 0.5 * self.upper * value**2 - value**3 / 3.0

        return 2.0 * (primitive(upper) - primitive(lower)) / denominator


def _validate_interval(lower: float, upper: float) -> tuple[float, float]:
    interval_lower = float(lower)
    interval_upper = float(upper)
    if np.isnan(interval_lower) or np.isnan(interval_upper):
        raise ValueError("interval bounds must not be NaN")
    if interval_upper < interval_lower:
        raise ValueError("interval upper bound must be at least its lower bound")
    return interval_lower, interval_upper
