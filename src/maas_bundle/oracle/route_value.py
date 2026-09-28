"""High-level orchestration for route-choice value functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from maas_bundle.oracle.demand_distribution import TripFrequencyDistribution
from maas_bundle.oracle.dual_polytope import (
    build_dual_polytope,
    enumerate_vertices,
)
from maas_bundle.oracle.piecewise_linear import (
    PiecewiseLinearFunction,
    build_piecewise_phi,
    sort_dual_vertices,
)


@dataclass(frozen=True)
class RouteValuation:
    """Expected route cost and allowance shadow price for one distribution."""

    value: float
    expected_omega: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Store a validated, immutable expected shadow-price vector."""

        value = float(self.value)
        expected_omega = np.asarray(self.expected_omega, dtype=float)
        if not np.isfinite(value):
            raise ValueError("expected route value must be finite")
        if expected_omega.ndim != 1:
            raise ValueError("expected_omega must be one-dimensional")
        if not np.all(np.isfinite(expected_omega)):
            raise ValueError("expected_omega must contain only finite values")
        if np.any(expected_omega < 0):
            raise ValueError("expected_omega must be nonnegative")

        immutable_omega = expected_omega.copy()
        immutable_omega.setflags(write=False)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "expected_omega", immutable_omega)

    @property
    def allowance_subgradient(self) -> NDArray[np.float64]:
        """Return the immutable allowance subgradient ``-E[omega*]``."""

        subgradient = -self.expected_omega
        subgradient.setflags(write=False)
        return subgradient


@dataclass(frozen=True)
class RouteValueOracle:
    """Reusable lower-level value oracle for fixed costs, usage, and tariffs.

    The H-to-V conversion and vertex sorting are performed once when the oracle
    is built. Different allowance vectors only require rebuilding the
    one-dimensional upper envelope.
    """

    vertices: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Enforce the sorted, immutable vertex-cache invariant."""

        object.__setattr__(self, "vertices", sort_dual_vertices(self.vertices))

    @property
    def vertex_count(self) -> int:
        """Return the number of vertices in the dual feasible region."""

        return self.vertices.shape[0]

    @property
    def mode_count(self) -> int:
        """Return the number of allowance dimensions."""

        return self.vertices.shape[1] - 1

    @property
    def slopes(self) -> NDArray[np.float64]:
        """Return a read-only view of the sorted ``pi`` coordinates."""

        return self.vertices[:, 0]

    @property
    def omega(self) -> NDArray[np.float64]:
        """Return a read-only view of the sorted shadow-price coordinates."""

        return self.vertices[:, 1:]

    def for_allowance(
        self,
        allowance: ArrayLike,
        *,
        domain_start: float = 0.0,
        tolerance: float = 1e-12,
    ) -> PiecewiseLinearFunction:
        """Return the complete piecewise-linear value function for an allowance."""

        return build_piecewise_phi(
            self.vertices,
            allowance,
            domain_start=domain_start,
            tolerance=tolerance,
        )

    def value(
        self,
        mu: ArrayLike,
        allowance: ArrayLike,
        *,
        domain_start: float = 0.0,
    ) -> float | NDArray[np.float64]:
        """Build the allowance envelope and evaluate it at trip frequencies."""

        return self.for_allowance(allowance, domain_start=domain_start)(mu)

    def integral(
        self,
        lower: float,
        upper: float,
        allowance: ArrayLike,
        *,
        domain_start: float = 0.0,
    ) -> float:
        """Build the allowance envelope and integrate it over a finite interval."""

        phi = self.for_allowance(allowance, domain_start=domain_start)
        return phi.integral(lower, upper)

    def expected_valuation(
        self,
        allowance: ArrayLike,
        distribution: TripFrequencyDistribution,
        *,
        probability_tolerance: float = 1e-10,
    ) -> RouteValuation:
        """Return ``G(epsilon)`` and an allowance subgradient under a demand law.

        The expectation is exact whenever the distribution supplies exact
        interval probabilities and unnormalized first moments. Each affine
        segment is integrated using those two quantities, while its active
        dual vertex supplies the corresponding shadow price.
        """

        if probability_tolerance < 0:
            raise ValueError("probability_tolerance must be nonnegative")

        phi = self.for_allowance(allowance)
        segment_count = phi.breakpoints.size
        probabilities = np.empty(segment_count, dtype=float)
        first_moments = np.empty(segment_count, dtype=float)

        for segment, lower in enumerate(phi.breakpoints):
            upper = phi.breakpoints[segment + 1] if segment + 1 < segment_count else np.inf
            probability = float(distribution.probability(float(lower), float(upper)))
            first_moment = float(distribution.first_moment(float(lower), float(upper)))
            if not np.isfinite(probability) or probability < -probability_tolerance:
                raise ValueError("distribution returned an invalid interval probability")
            if probability > 1.0 + probability_tolerance:
                raise ValueError("distribution returned an invalid interval probability")
            if not np.isfinite(first_moment) or first_moment < -probability_tolerance:
                raise ValueError("distribution returned an invalid interval first moment")
            probabilities[segment] = max(probability, 0.0)
            first_moments[segment] = max(first_moment, 0.0)

        total_probability = float(np.sum(probabilities))
        if not np.isclose(total_probability, 1.0, rtol=0.0, atol=probability_tolerance):
            msg = (
                "distribution must place total probability one on the "
                f"route-value domain, got {total_probability}"
            )
            raise ValueError(msg)

        expected_value = float(phi.slopes @ first_moments + phi.intercepts @ probabilities)
        active_omega = self.omega[phi.active_vertex_indices]
        expected_omega = probabilities @ active_omega
        return RouteValuation(value=expected_value, expected_omega=expected_omega)

    def expected_value(
        self,
        allowance: ArrayLike,
        distribution: TripFrequencyDistribution,
    ) -> float:
        """Return only ``G(epsilon)`` for callers that do not need derivatives."""

        return self.expected_valuation(allowance, distribution).value


def build_route_value_oracle(
    path_cost: ArrayLike,
    mode_usage: ArrayLike,
    payg_tariff: ArrayLike,
) -> RouteValueOracle:
    """Build and preprocess a reusable route-choice value oracle."""

    polytope = build_dual_polytope(
        path_cost=path_cost,
        mode_usage=mode_usage,
        payg_tariff=payg_tariff,
    )
    vertices = enumerate_vertices(polytope)
    return RouteValueOracle(vertices=vertices)
