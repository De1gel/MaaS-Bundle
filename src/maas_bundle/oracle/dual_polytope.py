"""Construction of the dual feasible region for a route-choice problem."""

from __future__ import annotations

from dataclasses import dataclass

import cdd
import numpy as np
from numpy.typing import ArrayLike, NDArray


class UnboundedPolytopeError(ValueError):
    """Raised when vertex-only output is requested for an unbounded polytope."""


@dataclass(frozen=True)
class DualPolytope:
    """Half-space representation of the route-choice dual feasible region.

    The dual vector is ordered as ``(pi, omega_1, ..., omega_M)`` and the
    feasible region is represented by ``matrix @ x <= bounds``.
    """

    matrix: NDArray[np.float64]
    bounds: NDArray[np.float64]

    @property
    def dimension(self) -> int:
        """Return the number of dual variables."""

        return self.matrix.shape[1]

    def contains(self, point: ArrayLike, *, tolerance: float = 1e-9) -> bool:
        """Return whether a dual point satisfies every inequality."""

        candidate = np.asarray(point, dtype=float)
        if candidate.shape != (self.dimension,):
            msg = f"point must have shape ({self.dimension},), got {candidate.shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(candidate)):
            raise ValueError("point must contain only finite values")
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")

        return bool(np.all(self.matrix @ candidate <= self.bounds + tolerance))


def build_dual_polytope(
    path_cost: ArrayLike,
    mode_usage: ArrayLike,
    payg_tariff: ArrayLike,
) -> DualPolytope:
    """Build the half-space representation of the route-choice dual.

    Parameters
    ----------
    path_cost:
        Generalized baseline path costs ``beta * c_bar`` with shape ``(K,)``.
    mode_usage:
        Mode-usage matrix ``N`` with shape ``(M, K)``.
    payg_tariff:
        Nonnegative PAYG tariff vector ``rho`` with shape ``(M,)``.

    Returns
    -------
    DualPolytope
        The inequalities for

        ``pi * 1 - N.T @ omega <= path_cost``,
        ``0 <= omega <= payg_tariff``, and ``pi >= 0``.
    """

    costs = _as_nonnegative_vector(path_cost, name="path_cost", require_nonempty=True)
    usage = _as_nonnegative_matrix(mode_usage, name="mode_usage")
    tariffs = _as_nonnegative_vector(payg_tariff, name="payg_tariff")

    mode_count, path_count = usage.shape
    if path_count != costs.size:
        msg = (
            "mode_usage must have one column per path_cost entry: "
            f"got {path_count} columns and {costs.size} costs"
        )
        raise ValueError(msg)
    if mode_count != tariffs.size:
        msg = (
            "mode_usage must have one row per payg_tariff entry: "
            f"got {mode_count} rows and {tariffs.size} tariffs"
        )
        raise ValueError(msg)

    # Each path r gives pi - N[:, r].T @ omega <= path_cost[r].
    path_rows = np.column_stack((np.ones(path_count), -usage.T))

    # The remaining rows encode omega <= rho, -omega <= 0, and -pi <= 0.
    omega_upper_rows = np.column_stack((np.zeros(mode_count), np.eye(mode_count)))
    omega_lower_rows = np.column_stack((np.zeros(mode_count), -np.eye(mode_count)))
    pi_lower_row = np.concatenate(([-1.0], np.zeros(mode_count)))[None, :]

    matrix = np.vstack((path_rows, omega_upper_rows, omega_lower_rows, pi_lower_row))
    bounds = np.concatenate(
        (costs, tariffs, np.zeros(mode_count), np.zeros(1)),
    )

    # Frozen dataclasses do not make contained arrays immutable by themselves.
    matrix.setflags(write=False)
    bounds.setflags(write=False)
    return DualPolytope(matrix=matrix, bounds=bounds)


def enumerate_vertices(
    polytope: DualPolytope,
) -> NDArray[np.float64]:
    """Convert a bounded H-polytope to its vertices using double description.

    ``pycddlib`` represents an inequality as ``b + A @ x >= 0``. The stored
    inequalities ``G @ x <= h`` are therefore passed to cddlib as ``[h, -G]``.
    Generator rows with leading coordinate one are vertices; rows with leading
    coordinate zero are rays.

    Polytopes returned by :func:`build_dual_polytope` are bounded because every
    component of ``omega`` lies in ``[0, rho]`` and any finite-cost path gives a
    finite upper bound on ``pi``. A manually constructed ``DualPolytope`` may be
    unbounded; this function detects cddlib rays and rejects that case.
    """

    h_representation = np.column_stack((polytope.bounds, -polytope.matrix))
    inequalities = cdd.matrix_from_array(
        h_representation.tolist(),
        rep_type=cdd.RepType.INEQUALITY,
    )
    cdd_polytope = cdd.polyhedron_from_matrix(inequalities)
    generators = cdd.copy_generators(cdd_polytope)
    generator_array = np.asarray(generators.array, dtype=float)

    if generator_array.size == 0:
        return np.empty((0, polytope.dimension), dtype=float)

    homogeneous_coordinates = generator_array[:, 0]
    if np.any(np.isclose(homogeneous_coordinates, 0.0)):
        raise UnboundedPolytopeError("polytope is unbounded; cddlib returned rays")

    vertices = generator_array[:, 1:] / homogeneous_coordinates[:, None]
    order = np.lexsort(vertices[:, ::-1].T)
    result = vertices[order]
    result.setflags(write=False)
    return result


def _as_nonnegative_vector(
    values: ArrayLike,
    *,
    name: str,
    require_nonempty: bool = False,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if require_nonempty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _as_nonnegative_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array
