"""Single-start and multi-start experiments for the sioux falls instance."""

from __future__ import annotations

import numpy as np

from maas_bundle.benchmarks.initialization import sample_profitable_menus
from maas_bundle.benchmarks.multistart import MultiStartResult, run_multistart
from maas_bundle.benchmarks.settings import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIOUX_FALLS_SETTINGS,
    DEFAULT_START_COUNT,
    BenchmarkSettings,
)
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import solve_primal
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import ScipyRegionalSolver
from maas_bundle.design.validation import CandidateDiagnostics
from maas_bundle.examples.sioux_falls import (
    build_sioux_falls_problem,
    draw_sioux_falls_menu,
    initial_sioux_falls_menu,
)
from maas_bundle.examples.sioux_falls_config import (
    DEFAULT_SIOUX_FALLS_CONFIG,
    SiouxFallsExperimentConfig,
)


def generate_sioux_falls_initial_menus(
    start_count: int = DEFAULT_START_COUNT,
    *,
    seed: int | None = DEFAULT_RANDOM_SEED,
    problem: BundleDesignProblem | None = None,
    config: SiouxFallsExperimentConfig = DEFAULT_SIOUX_FALLS_CONFIG,
    minimum_profit: float = DEFAULT_SIOUX_FALLS_SETTINGS.profit_tolerance,
) -> tuple[BundleMenu, ...]:
    """Generate monotone, positive-profit starts at Sioux Falls credit scales."""

    if not isinstance(start_count, (int, np.integer)) or start_count <= 0:
        raise ValueError("start_count must be a positive integer")
    random = np.random.default_rng(seed)
    if problem is None:
        problem = build_sioux_falls_problem(config=config)

    def draw_menu() -> BundleMenu:
        return draw_sioux_falls_menu(random, config=config)

    return sample_profitable_menus(
        problem,
        draw_menu,
        start_count,
        minimum_profit=minimum_profit,
    )


def run_sioux_falls_example(
    *,
    oracle_cache_path=None,
    config: SiouxFallsExperimentConfig = DEFAULT_SIOUX_FALLS_CONFIG,
    settings: BenchmarkSettings = DEFAULT_SIOUX_FALLS_SETTINGS,
) -> CandidateDiagnostics:
    """Run the full primal pipeline on the large Sioux Falls instance."""

    solver = ScipyRegionalSolver(
        "SLSQP",
        options={"maxiter": settings.nlp_max_iterations, "ftol": settings.nlp_ftol},
        check_kkt=settings.check_kkt,
        slsqp_restarts=settings.slsqp_restarts,
        enable_fallback=settings.solver_fallback,
        numerical_constraint_recovery=settings.numerical_constraint_recovery,
    )
    return solve_primal(
        build_sioux_falls_problem(
            config=config, **({"oracle_cache_path": oracle_cache_path} if oracle_cache_path else {})
        ),
        initial_sioux_falls_menu(config=config),
        solver,
        max_region_iterations=settings.max_region_iterations,
        profit_tolerance=settings.profit_tolerance,
    )


def run_sioux_falls_multistart(
    start_count: int = DEFAULT_START_COUNT,
    *,
    oracle_cache_path=None,
    seed: int | None = DEFAULT_RANDOM_SEED,
    config: SiouxFallsExperimentConfig = DEFAULT_SIOUX_FALLS_CONFIG,
    settings: BenchmarkSettings = DEFAULT_SIOUX_FALLS_SETTINGS,
) -> MultiStartResult:
    """Run the reproducible multi-start benchmark on Sioux Falls."""

    solver = ScipyRegionalSolver(
        "SLSQP",
        options={"maxiter": settings.nlp_max_iterations, "ftol": settings.nlp_ftol},
        check_kkt=settings.check_kkt,
        slsqp_restarts=settings.slsqp_restarts,
        enable_fallback=settings.solver_fallback,
        numerical_constraint_recovery=settings.numerical_constraint_recovery,
    )
    problem = build_sioux_falls_problem(
        config=config, **({"oracle_cache_path": oracle_cache_path} if oracle_cache_path else {})
    )
    return run_multistart(
        problem,
        generate_sioux_falls_initial_menus(
            start_count,
            seed=seed,
            problem=problem,
            config=config,
            minimum_profit=settings.profit_tolerance,
        ),
        solver,
        max_region_iterations=settings.max_region_iterations,
        profit_tolerance=settings.profit_tolerance,
    )
