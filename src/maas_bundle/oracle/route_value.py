"""High-level orchestration for route-choice value functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
