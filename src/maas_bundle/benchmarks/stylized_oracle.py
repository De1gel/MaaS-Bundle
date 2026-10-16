"""A1: exact expected valuation and allowance shadow prices for class M."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import linprog

from maas_bundle.examples.stylized import build_stylized_problem
from maas_bundle.examples.stylized_config import (
    DEFAULT_STYLIZED_CONFIG,
    MODE_USAGE,
    StylizedExperimentConfig,
)
from maas_bundle.oracle.demand_distribution import UniformDemandDistribution
from maas_bundle.oracle.route_value import RouteValueOracle


@dataclass(frozen=True)
class StylizedOracleSettings:
    """Plotting resolution and diagnostic settings, not optimization bounds."""

    allowance_max: float = 40.0
    grid_points: int = 161
    realized_frequency: float = 20.0
    gradient_samples: int = 64
    seed: int = 20261002
    diagnostic_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not np.isfinite(self.allowance_max) or self.allowance_max <= 0:
            raise ValueError("allowance_max must be finite and positive")
        if not np.isfinite(self.realized_frequency) or self.realized_frequency < 0:
            raise ValueError("realized_frequency must be finite and nonnegative")
        for name, minimum in (("grid_points", 3), ("gradient_samples", 1)):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(self.diagnostic_tolerance) or self.diagnostic_tolerance <= 0:
            raise ValueError("diagnostic_tolerance must be finite and positive")


DEFAULT_STYLIZED_ORACLE_SETTINGS = StylizedOracleSettings()


@dataclass(frozen=True)
class StylizedOracleResult:
    """Arrays are indexed [SM allowance, PT allowance]; omega's last axis is mode."""

    config: StylizedExperimentConfig
    settings: StylizedOracleSettings
    allowance_axis: NDArray[np.float64]
    expected_values: NDArray[np.float64]
    expected_omega: NDArray[np.float64]
    realized_values: NDArray[np.float64]
    realized_dual_indices: NDArray[np.int64]
    vertices: NDArray[np.float64]
    gradient_checks: tuple[dict[str, object], ...]
    diagnostics: dict[str, object]
    evaluation_seconds: float


def run_stylized_oracle(
    *,
    config: StylizedExperimentConfig = DEFAULT_STYLIZED_CONFIG,
    settings: StylizedOracleSettings = DEFAULT_STYLIZED_ORACLE_SETTINGS,
) -> StylizedOracleResult:
    """Evaluate A1 without solving or altering the upper-level design problem.

    Values and expected shadow prices use exact interval probabilities and
    first moments. Realized dual indices are -1 at ties instead of assigning
    an arbitrary vertex. The plot window imposes no restriction on menu design.
    """

    started = perf_counter()
    problem = build_stylized_problem(config)
    class_index = config.class_names.index("M")
    oracle = problem.route_oracles[class_index][0]
    distribution = problem.demand_distributions[class_index]
    axis = np.linspace(0, settings.allowance_max, settings.grid_points)
    values = np.empty((axis.size, axis.size))
    omega = np.empty((*values.shape, 2))
    for row, sm in enumerate(axis):
        for column, pt in enumerate(axis):
            valuation = oracle.expected_valuation([pt, sm], distribution)
            values[row, column] = valuation.value
            omega[row, column] = valuation.expected_omega

    pt_grid, sm_grid = np.meshgrid(axis, axis)
    allowance_grid = np.stack((pt_grid, sm_grid), axis=-1)
    dual_values = settings.realized_frequency * oracle.slopes - np.einsum(
        "ijm,vm->ijv", allowance_grid, oracle.omega
    )
    realized_values = np.max(dual_values, axis=-1)
    dual_indices = np.argmax(dual_values, axis=-1)
    tied = (
        np.sum(np.isclose(dual_values, realized_values[..., None], rtol=0, atol=1e-9), axis=-1) > 1
    )
    dual_indices[tied] = -1
    checks, diagnostics = _validate_oracle(config, settings, oracle, distribution, values, omega)
    diagnostics["realized_unique_active_vertices"] = np.unique(dual_indices[~tied]).tolist()
    diagnostics["realized_tied_grid_points"] = int(np.count_nonzero(tied))
    diagnostics["grid_evaluations"] = int(values.size)
    for array in (axis, values, omega, realized_values, dual_indices):
        array.setflags(write=False)
    return StylizedOracleResult(
        config=config,
        settings=settings,
        allowance_axis=axis,
        expected_values=values,
        expected_omega=omega,
        realized_values=realized_values,
        realized_dual_indices=dual_indices,
        vertices=oracle.vertices,
        gradient_checks=checks,
        diagnostics=diagnostics,
        evaluation_seconds=perf_counter() - started,
    )


def _validate_oracle(
    config: StylizedExperimentConfig,
    settings: StylizedOracleSettings,
    oracle: RouteValueOracle,
    distribution: UniformDemandDistribution,
    values: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Check the dual oracle against an independent primal LP and derivatives."""

    usage = np.asarray(MODE_USAGE)
    cost = np.concatenate((config.baseline_path_costs[1], config.payg_tariff))
    inequality = np.column_stack((usage, -np.eye(2)))
    equality = np.array([[1.0, 1.0, 1.0, 0.0, 0.0]])
    lp_calls = 0

    def primal_value(mu: float, allowance: NDArray[np.float64]) -> float:
        nonlocal lp_calls
        lp_calls += 1
        result = linprog(
            cost,
            A_ub=inequality,
            b_ub=allowance,
            A_eq=equality,
            b_eq=[mu],
            bounds=(0, None),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"independent path-allocation LP failed: {result.message}")
        return float(result.fun)

    scale = settings.allowance_max
    reference_allowances = scale * np.array(
        [[0, 0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.25], [1, 1]]
    )
    lp_error = 0.0
    expectation_error = 0.0
    expectation_checks = []
    mean = (distribution.lower + distribution.upper) / 2
    for allowance in reference_allowances:
        for mu in (distribution.lower, mean, distribution.upper, settings.realized_frequency):
            lp_error = max(lp_error, abs(primal_value(mu, allowance) - oracle.value(mu, allowance)))
        reference, quadrature_error = quad(
            lambda mu, fixed_allowance=allowance: primal_value(mu, fixed_allowance),
            distribution.lower,
            distribution.upper,
            points=[mean],
            epsabs=1e-8,
            epsrel=1e-10,
        )
        reference /= distribution.upper - distribution.lower
        exact = oracle.expected_value(allowance, distribution)
        expectation_error = max(expectation_error, abs(exact - reference))
        expectation_checks.append(
            {
                "allowance": allowance.tolist(),
                "oracle_expected_value": exact,
                "primal_lp_integrated_value": reference,
                "quadrature_error_bound": quadrature_error
                / (distribution.upper - distribution.lower),
            }
        )

    random = np.random.default_rng(settings.seed)
    records = []
    gradient_error = 0.0
    h = min(1e-4, scale / 1000)
    for allowance in random.uniform(0.1 * scale, 0.9 * scale, (settings.gradient_samples, 2)):
        valuation = oracle.expected_valuation(allowance, distribution)
        differences = []
        side_disagreement = []
        for direction in np.eye(2):
            plus = oracle.expected_value(allowance + h * direction, distribution)
            minus = oracle.expected_value(allowance - h * direction, distribution)
            differences.append((plus - minus) / (2 * h))
            side_disagreement.append(
                abs((plus - valuation.value) / h - (valuation.value - minus) / h)
            )
        differentiable = max(side_disagreement) <= 1e-3
        error = float(np.max(np.abs(np.asarray(differences) + valuation.expected_omega)))
        if differentiable:
            gradient_error = max(gradient_error, error)
        records.append(
            {
                "epsilon_PT": float(allowance[0]),
                "epsilon_SM": float(allowance[1]),
                "minus_omega_PT": float(-valuation.expected_omega[0]),
                "minus_omega_SM": float(-valuation.expected_omega[1]),
                "finite_difference_PT": differences[0],
                "finite_difference_SM": differences[1],
                "maximum_error": error,
                "smooth_at_difference_scale": differentiable,
            }
        )
    smooth_count = sum(record["smooth_at_difference_scale"] for record in records)
    monotonicity_violation = max(
        float(np.max(np.diff(values, axis=0), initial=0)),
        float(np.max(np.diff(values, axis=1), initial=0)),
    )
    shadow_violation = max(
        float(np.max(-omega, initial=0)),
        float(np.max(omega - np.asarray(config.payg_tariff), initial=0)),
    )
    passed = (
        smooth_count > 0
        and max(
            lp_error, expectation_error, gradient_error, monotonicity_violation, shadow_violation
        )
        <= settings.diagnostic_tolerance
    )
    return tuple(records), {
        "passed": bool(passed),
        "diagnostic_tolerance": settings.diagnostic_tolerance,
        "maximum_realized_primal_dual_error": lp_error,
        "maximum_expected_primal_oracle_error": expectation_error,
        "maximum_gradient_error_at_smooth_samples": gradient_error,
        "gradient_sample_count": settings.gradient_samples,
        "smooth_gradient_sample_count": smooth_count,
        "finite_difference_step": h,
        "maximum_monotonicity_violation": monotonicity_violation,
        "maximum_shadow_price_bound_violation": shadow_violation,
        "independent_primal_lp_calls": lp_calls,
        "expectation_checks": expectation_checks,
    }


def save_stylized_oracle_results(result: StylizedOracleResult, output_dir: str | Path) -> dict:
    """Validate the visualization and save only the final two-panel PDF and PNG."""

    from maas_bundle.benchmarks.stylized_oracle_plots import (
        plot_stylized_oracle,
        prepare_stylized_regimes,
        validate_stylized_figure,
    )

    regimes = prepare_stylized_regimes(result)
    visualization = validate_stylized_figure(result, regimes)
    if not visualization["passed"]:
        raise ValueError(f"A1 visualization validation failed: {visualization}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figures = plot_stylized_oracle(result, directory, regimes=regimes)

    # Remove only the retired A1 artifacts produced by this experiment's old exporter.
    for name in ("a1_overview", "a1_dual_regimes", "a1_allowance_slices", "a1_value_surface"):
        for extension in ("png", "pdf", "svg"):
            (directory / f"{name}.{extension}").unlink(missing_ok=True)
    for name in (
        "a1_oracle_grid.csv",
        "a1_oracle_grid.npz",
        "a1_gradient_checks.csv",
        "a1_summary.json",
    ):
        (directory / name).unlink(missing_ok=True)
    readme = directory / "README.md"
    if readme.is_file() and readme.read_text(encoding="utf-8").startswith(
        "# A1 — Oracle structure"
    ):
        readme.unlink()
    return {
        "experiment": "A1 — Valuation oracle structure",
        "traveler_class": "M",
        "instance": result.config.metadata(),
        "plot_window_is_optimization_bound": False,
        "value_range": [float(result.expected_values.min()), float(result.expected_values.max())],
        "diagnostics": result.diagnostics,
        "visualization": visualization,
        "figures": figures,
    }
