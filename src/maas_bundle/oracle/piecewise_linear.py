"""One-dimensional piecewise-linear route-value functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PiecewiseLinearFunction:
    """Upper envelope on ``[breakpoints[0], infinity)``.

    Segment ``r`` is ``slopes[r] * mu + intercepts[r]`` from
    ``breakpoints[r]`` up to the next breakpoint. The final segment extends to
    infinity.
    """

    breakpoints: NDArray[np.float64]
    slopes: NDArray[np.float64]
    intercepts: NDArray[np.float64]
    active_vertex_indices: NDArray[np.int64]
    cumulative_integrals: NDArray[np.float64]

    def __call__(self, mu: ArrayLike) -> float | NDArray[np.float64]:
        """Evaluate the value function at one or more trip frequencies."""

        frequencies = np.asarray(mu, dtype=float)
        if not np.all(np.isfinite(frequencies)):
            raise ValueError("mu must contain only finite values")
        if np.any(frequencies < self.breakpoints[0]):
            raise ValueError(f"mu must be at least {self.breakpoints[0]}")

        segments = np.searchsorted(self.breakpoints, frequencies, side="right") - 1
        values = self.slopes[segments] * frequencies + self.intercepts[segments]
        if values.ndim == 0:
            return float(values.item())
        return values

    def integral(self, lower: float, upper: float) -> float:
        """Return the exact integral of the value function on a finite interval."""

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("integration bounds must be finite")
        if lower < self.breakpoints[0]:
            raise ValueError(f"lower must be at least {self.breakpoints[0]}")
        if upper < lower:
            raise ValueError("upper must be greater than or equal to lower")

        return self._integral_from_domain_start(upper) - self._integral_from_domain_start(lower)

    def _integral_from_domain_start(self, upper: float) -> float:
        segment = int(np.searchsorted(self.breakpoints, upper, side="right") - 1)
        lower = self.breakpoints[segment]
        slope = self.slopes[segment]
        intercept = self.intercepts[segment]
        remainder = 0.5 * slope * (upper**2 - lower**2) + intercept * (upper - lower)
        return float(self.cumulative_integrals[segment] + remainder)


def sort_dual_vertices(vertices: ArrayLike) -> NDArray[np.float64]:
    """Return an immutable vertex matrix sorted by its ``pi`` coordinate."""

    array = _as_vertex_matrix(vertices)

    order = np.argsort(array[:, 0], kind="stable")
    result = array[order].copy()
    result.setflags(write=False)
    return result


def build_piecewise_phi(
    sorted_vertices: ArrayLike,
    allowance: ArrayLike,
    *,
    domain_start: float = 0.0,
    tolerance: float = 1e-12,
) -> PiecewiseLinearFunction:
    """Build ``max_j(pi_j * mu - omega_j @ allowance)`` on a half-line."""

    if not np.isfinite(domain_start):
        raise ValueError("domain_start must be finite")
    if domain_start < 0:
        raise ValueError("domain_start must be nonnegative")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    vertices = _as_vertex_matrix(sorted_vertices)
    if np.any(np.diff(vertices[:, 0]) < 0):
        raise ValueError("sorted_vertices must be ordered by pi")

    mode_count = vertices.shape[1] - 1
    allowance_array = np.asarray(allowance, dtype=float)
    if allowance_array.shape != (mode_count,):
        msg = f"allowance must have shape ({mode_count},), got {allowance_array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(allowance_array)):
        raise ValueError("allowance must contain only finite values")
    if np.any(allowance_array < 0):
        raise ValueError("allowance must be nonnegative")

    intercepts = -(vertices[:, 1:] @ allowance_array)
    unique_lines = _best_line_per_slope(vertices, intercepts, tolerance=tolerance)
    hull = _build_upper_hull(unique_lines, tolerance=tolerance)

    starts = np.asarray([line[3] for line in hull])
    first_active = int(np.searchsorted(starts, domain_start, side="right") - 1)
    active_hull = hull[first_active:]

    breakpoints = np.asarray(
        [domain_start, *(line[3] for line in active_hull[1:])],
        dtype=float,
    )
    slopes = np.asarray([line[0] for line in active_hull], dtype=float)
    active_intercepts = np.asarray([line[1] for line in active_hull], dtype=float)
    active_indices = np.asarray([line[2] for line in active_hull], dtype=np.int64)
    cumulative_integrals = _cumulative_integrals(breakpoints, slopes, active_intercepts)
    for result in (
        breakpoints,
        slopes,
        active_intercepts,
        active_indices,
        cumulative_integrals,
    ):
        result.setflags(write=False)

    return PiecewiseLinearFunction(
        breakpoints=breakpoints,
        slopes=slopes,
        intercepts=active_intercepts,
        active_vertex_indices=active_indices,
        cumulative_integrals=cumulative_integrals,
    )


def _cumulative_integrals(
    breakpoints: NDArray[np.float64],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
) -> NDArray[np.float64]:
    cumulative = np.zeros_like(breakpoints)
    if breakpoints.size == 1:
        return cumulative

    lower = breakpoints[:-1]
    upper = breakpoints[1:]
    segment_integrals = 0.5 * slopes[:-1] * (upper**2 - lower**2) + intercepts[:-1] * (
        upper - lower
    )
    cumulative[1:] = np.cumsum(segment_integrals)
    return cumulative


def _best_line_per_slope(
    vertices: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    *,
    tolerance: float,
) -> list[tuple[float, float, int]]:
    lines: list[tuple[float, float, int]] = []
    for vertex_index, (vertex, intercept) in enumerate(zip(vertices, intercepts, strict=True)):
        candidate = (float(vertex[0]), float(intercept), vertex_index)
        if not lines or abs(candidate[0] - lines[-1][0]) > tolerance:
            lines.append(candidate)
        elif candidate[1] > lines[-1][1] + tolerance:
            lines[-1] = candidate
    return lines


def _as_vertex_matrix(vertices: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(vertices, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"vertices must be two-dimensional, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("vertices must contain at least one vertex")
    if array.shape[1] == 0:
        raise ValueError("vertices must contain a pi coordinate")
    if not np.all(np.isfinite(array)):
        raise ValueError("vertices must contain only finite values")
    return array


def _build_upper_hull(
    lines: list[tuple[float, float, int]],
    *,
    tolerance: float,
) -> list[tuple[float, float, int, float]]:
    hull: list[tuple[float, float, int, float]] = []
    for slope, intercept, vertex_index in lines:
        start = -np.inf
        while hull:
            previous_slope, previous_intercept, _, previous_start = hull[-1]
            start = (previous_intercept - intercept) / (slope - previous_slope)
            if start > previous_start + tolerance:
                break
            hull.pop()
        if not hull:
            start = -np.inf
        hull.append((slope, intercept, vertex_index, start))
    return hull
