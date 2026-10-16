"""The two-panel A1 valuation structure figure; no changes to oracle evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

if TYPE_CHECKING:
    from maas_bundle.benchmarks.stylized_oracle import StylizedOracleResult

PT_LABEL = r"PT allowance $\varepsilon_{\mathrm{PT}}$"
SM_LABEL = r"SM allowance $\varepsilon_{\mathrm{SM}}$"
STYLES = (
    ("#0072B2", (0, (1.4, 2.2))),
    ("#3d3d3d", "solid"),
    ("#D55E00", (0, (6, 3))),
)


@dataclass(frozen=True)
class RegimeBoundary:
    """A vertex-pair equality segment clipped to the dual upper envelope."""

    vertex_pair: tuple[int, int]
    endpoints: NDArray[np.float64]


@dataclass(frozen=True)
class RealizedRegimes:
    """A partition computed directly at one demand, using cached oracle vertices."""

    frequency: float
    vertices: NDArray[np.float64]
    values: NDArray[np.float64]
    indices: NDArray[np.int64]
    active_vertices: tuple[int, ...]
    boundaries: tuple[RegimeBoundary, ...]
    junction_verified: bool


def _clip_boundary(vertices, pair, frequency, maximum):
    """Intersect pair equality with the scan box and all dual-dominance halfspaces."""

    first, second = pair
    pi = vertices[:, 0]
    omega = vertices[:, 1:]
    normal = omega[first] - omega[second]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    origin = frequency * (pi[first] - pi[second]) * normal / norm**2
    direction = np.array([-normal[1], normal[0]]) / norm
    normals = np.vstack(([1, 0], [-1, 0], [0, 1], [0, -1], omega[first] - omega))
    limits = np.concatenate(([maximum, 0, maximum, 0], frequency * (pi[first] - pi)))
    lower, upper = -np.inf, np.inf
    for coefficient, limit in zip(normals, limits, strict=True):
        slope = float(coefficient @ direction)
        remaining = float(limit - coefficient @ origin)
        if abs(slope) <= 1e-12:
            if remaining < -1e-9:
                return None
        elif slope > 0:
            upper = min(upper, remaining / slope)
        else:
            lower = max(lower, remaining / slope)
    if upper - lower <= 1e-8:
        return None  # An isolated multiway tie is not a boundary segment.
    endpoints = np.clip(origin + np.outer([lower, upper], direction), 0, maximum)
    endpoints.setflags(write=False)
    return RegimeBoundary(pair, endpoints)


def _realized_regimes(result: StylizedOracleResult, frequency: float) -> RealizedRegimes:
    """Find full-dimensional active regions, then extract their exact shared edges."""

    pt, sm = np.meshgrid(result.allowance_axis, result.allowance_axis)
    allowances = np.stack((pt, sm), axis=-1)
    vertices = result.vertices
    values_by_vertex = frequency * vertices[:, 0] - np.einsum(
        "ijm,vm->ijv", allowances, vertices[:, 1:]
    )
    values = np.max(values_by_vertex, axis=-1)
    indices = np.argmax(values_by_vertex, axis=-1)
    tied = np.sum(np.isclose(values_by_vertex, values[..., None], rtol=0, atol=1e-9), axis=-1) > 1
    indices[tied] = -1
    active_vertices = tuple(int(index) for index in np.unique(indices[~tied]))
    boundaries = tuple(
        boundary
        for pair in combinations(active_vertices, 2)
        if (boundary := _clip_boundary(vertices, pair, frequency, result.settings.allowance_max))
        is not None
    )
    # Mark (mu,mu) only after verifying a multiway regime junction and full coverage.
    point = np.array([frequency, frequency])
    point_values = frequency * vertices[:, 0] - vertices[:, 1:] @ point
    incident = sum(
        np.min(np.linalg.norm(boundary.endpoints - point, axis=1)) < 1e-7 for boundary in boundaries
    )
    tied_at_point = np.count_nonzero(
        np.isclose(point_values, point_values.max(), rtol=0, atol=1e-9)
    )
    covered_value = frequency * min(result.config.baseline_path_costs[1])
    junction_verified = bool(
        frequency <= result.settings.allowance_max
        and incident >= 3
        and tied_at_point >= 3
        and np.isclose(point_values.max(), covered_value, rtol=0, atol=1e-8)
    )
    values.setflags(write=False)
    indices.setflags(write=False)
    return RealizedRegimes(
        frequency, vertices, values, indices, active_vertices, boundaries, junction_verified
    )


def prepare_stylized_regimes(result: StylizedOracleResult) -> tuple[RealizedRegimes, ...]:
    """Evaluate the same upper envelope separately at the support endpoints and mean."""

    lower, upper = result.config.frequency_supports[1]
    return tuple(_realized_regimes(result, mu) for mu in (lower, (lower + upper) / 2, upper))


def validate_stylized_figure(result, regimes) -> dict[str, object]:
    """Validate plotting anchors, old/new partition agreement and extracted segments."""

    traveler = result.config.class_names.index("M")
    expected_origin = result.config.payg_expected_costs[traveler]
    mean = sum(result.config.frequency_supports[1]) / 2
    expected_floor = mean * min(result.config.baseline_path_costs[1])
    reference = next(
        (regime for regime in regimes if regime.frequency == result.settings.realized_frequency),
        None,
    )
    if reference is None:
        reference = _realized_regimes(result, result.settings.realized_frequency)
    original_map_matches = bool(
        np.array_equal(reference.indices, result.realized_dual_indices)
        and np.allclose(reference.values, result.realized_values, rtol=0, atol=1e-9)
    )
    equality_error = dominance_error = 0.0
    for regime in regimes:
        for boundary in regime.boundaries:
            # The full segment is affine, so endpoints and midpoint suffice for dominance.
            probes = np.vstack((boundary.endpoints, boundary.endpoints.mean(axis=0)))
            values = regime.frequency * result.vertices[:, 0] - probes @ result.vertices[:, 1:].T
            first, second = boundary.vertex_pair
            equality_error = max(
                equality_error, float(np.max(abs(values[:, first] - values[:, second])))
            )
            dominance_error = max(
                dominance_error, float(np.max(values.max(axis=1) - values[:, first]))
            )
    origin_matches = bool(
        np.isclose(result.expected_values[0, 0], expected_origin, rtol=0, atol=1e-8)
    )
    floor_matches = bool(
        np.isclose(result.expected_values[-1, -1], expected_floor, rtol=0, atol=1e-8)
    )
    same_vertices = all(regime.vertices is result.vertices for regime in regimes)
    passed = bool(
        result.diagnostics["passed"]
        and original_map_matches
        and origin_matches
        and floor_matches
        and same_vertices
        and equality_error <= 1e-8
        and dominance_error <= 1e-8
    )
    return {
        "passed": passed,
        "mu_values": [regime.frequency for regime in regimes],
        "allowance_scan_range": {
            "PT": [0, result.settings.allowance_max],
            "SM": [0, result.settings.allowance_max],
        },
        "G_M_origin": float(result.expected_values[0, 0]),
        "G_M_large_allowance": float(result.expected_values[-1, -1]),
        "original_realized_partition_matches": original_map_matches,
        "same_cached_oracle_vertices": same_vertices,
        "maximum_boundary_equality_error": equality_error,
        "maximum_boundary_dominance_violation": dominance_error,
        "cached_dual_vertex_count": int(result.vertices.shape[0]),
        "active_dual_vertex_counts": {
            f"{regime.frequency:g}": len(regime.active_vertices) for regime in regimes
        },
        "boundary_segment_counts": {
            f"{regime.frequency:g}": len(regime.boundaries) for regime in regimes
        },
        "verified_junctions": [
            [regime.frequency, regime.frequency] for regime in regimes if regime.junction_verified
        ],
    }


def build_stylized_figure(result, regimes):
    """Create exactly two allowance panels plus the expected-cost colorbar."""

    figure, axes = plt.subplots(
        1, 2, figsize=(12.8, 5.8), layout="constrained", gridspec_kw={"wspace": 0.13}
    )
    axis = result.allowance_axis
    heatmap = axes[0].pcolormesh(
        axis,
        axis,
        result.expected_values,
        shading="nearest",
        cmap="viridis",
        vmin=float(result.expected_values.min()),
        vmax=float(result.expected_values.max()),
        rasterized=True,
    )
    levels = MaxNLocator(nbins=6).tick_values(
        result.expected_values.min(), result.expected_values.max()
    )
    levels = levels[
        (levels > result.expected_values.min()) & (levels < result.expected_values.max())
    ]
    contours = axes[0].contour(
        axis,
        axis,
        result.expected_values,
        levels=levels,
        colors="white",
        linewidths=0.55,
        alpha=0.7,
    )
    axes[0].clabel(contours, inline=True, fontsize=8, fmt="%g")
    axes[0].set_title(
        r"(a) EMTC $G_M(\boldsymbol{\varepsilon})$", loc="left", pad=12
    )
    colorbar = figure.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.025)
    colorbar.set_label("EMTC [MU / person / month]")
    handles = []
    for regime, (color, linestyle) in zip(regimes, STYLES, strict=True):
        segments = [boundary.endpoints for boundary in regime.boundaries]
        axes[1].add_collection(
            LineCollection(segments, colors=color, linewidths=1.65, linestyles=linestyle)
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                linewidth=1.65,
                label=rf"$\mu={regime.frequency:g}$",
            )
        )
        if regime.junction_verified:
            mu = regime.frequency
            axes[1].plot(mu, mu, "o", color=color, markersize=4, zorder=4)
            axes[1].annotate(
                rf"$({mu:g},{mu:g})$",
                (mu, mu),
                xytext=(7, 9),
                textcoords="offset points",
                color=color,
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.6},
            )
    axes[1].set_title(r"(b) Realized-demand regime boundaries across $\mu$", loc="left", pad=12)
    axes[1].legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95)
    axes[1].grid(color="#d1d5db", linewidth=0.5, alpha=0.45)
    for panel in axes:
        panel.set(
            xlabel=PT_LABEL,
            ylabel=SM_LABEL,
            xlim=(0, result.settings.allowance_max),
            ylim=(0, result.settings.allowance_max),
            aspect="equal",
        )
    return figure


def plot_stylized_oracle(
    result: StylizedOracleResult, directory: Path, *, regimes=None
) -> list[str]:
    """Save one valuation-structure figure as PNG and PDF, at 320 dpi."""

    if regimes is None:
        regimes = prepare_stylized_regimes(result)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
        }
    ):
        figure = build_stylized_figure(result, regimes)
        filenames = ["a1_valuation_structure.pdf", "a1_valuation_structure.png"]
        try:
            for filename in filenames:
                figure.savefig(directory / filename, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(figure)
    return filenames
