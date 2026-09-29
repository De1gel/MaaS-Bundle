"""Single-start and multi-start experiments for the two route instance."""

from __future__ import annotations

import numpy as np

from maas_bundle.benchmarks.assignment_enumeration import enumerate_assignment_regions
from maas_bundle.benchmarks.initialization import sample_profitable_menus
from maas_bundle.benchmarks.multistart import MultiStartResult, run_multistart
from maas_bundle.benchmarks.settings import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_START_COUNT,
    DEFAULT_TWO_ROUTE_SETTINGS,
    BenchmarkSettings,
)
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import solve_primal
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import ScipyRegionalSolver
from maas_bundle.design.validation import CandidateDiagnostics
from maas_bundle.examples.two_route import (
    build_two_route_problem,
    draw_two_route_menu,
    initial_two_route_menu,
)


def generate_two_route_initial_menus(
    start_count: int = DEFAULT_START_COUNT,
    *,
    seed: int | None = DEFAULT_RANDOM_SEED,
    problem: BundleDesignProblem | None = None,
    minimum_profit: float = DEFAULT_TWO_ROUTE_SETTINGS.profit_tolerance,
) -> tuple[BundleMenu, ...]:
    """Generate monotone, positive-profit starts outside the all-PAYG plateau."""

    if not isinstance(start_count, (int, np.integer)) or start_count <= 0:
        raise ValueError("start_count must be a positive integer")
    random = np.random.default_rng(seed)
    if problem is None:
        problem = build_two_route_problem()

    def draw_menu() -> BundleMenu:
        return draw_two_route_menu(random)

    return sample_profitable_menus(problem, draw_menu, start_count, minimum_profit=minimum_profit)


def run_two_route_example(
    *,
    settings: BenchmarkSettings = DEFAULT_TWO_ROUTE_SETTINGS,
) -> CandidateDiagnostics:
    """Run the complete primal pipeline on the small open-data instance."""

    solver = ScipyRegionalSolver(
        "SLSQP",
        options={"maxiter": settings.nlp_max_iterations, "ftol": settings.nlp_ftol},
    )
    return solve_primal(
        build_two_route_problem(),
        initial_two_route_menu(),
        solver,
        max_region_iterations=settings.max_region_iterations,
        profit_tolerance=settings.profit_tolerance,
    )


def run_two_route_multistart(
    start_count: int = DEFAULT_START_COUNT,
    *,
    seed: int | None = DEFAULT_RANDOM_SEED,
    settings: BenchmarkSettings = DEFAULT_TWO_ROUTE_SETTINGS,
) -> MultiStartResult:
    """Run a reproducible multi-start experiment on the two-route instance."""

    solver = ScipyRegionalSolver(
        "SLSQP",
        options={"maxiter": settings.nlp_max_iterations, "ftol": settings.nlp_ftol},
    )
    problem = build_two_route_problem()
    return run_multistart(
        problem,
        generate_two_route_initial_menus(
            start_count, seed=seed, problem=problem, minimum_profit=settings.profit_tolerance
        ),
        solver,
        max_region_iterations=settings.max_region_iterations,
        profit_tolerance=settings.profit_tolerance,
    )


def compare_two_route(
    *,
    settings: BenchmarkSettings = DEFAULT_TWO_ROUTE_SETTINGS,
) -> dict[str, object]:
    """Compare one primal run with all fixed-assignment regional solves."""

    problem = build_two_route_problem()
    initial_menu = initial_two_route_menu()
    solver = ScipyRegionalSolver(
        "SLSQP",
        options={"maxiter": settings.nlp_max_iterations, "ftol": settings.nlp_ftol},
    )
    primal = solve_primal(
        problem,
        initial_menu,
        solver,
        max_region_iterations=settings.max_region_iterations,
        profit_tolerance=settings.profit_tolerance,
    )
    enumeration = enumerate_assignment_regions(problem, initial_menu, solver)
    best = enumeration.best_candidate
    primal_profit = primal.candidate.true_profit
    best_profit = None if best is None else best.true_profit
    profit_gap = (
        None if primal_profit is None or best_profit is None else best_profit - primal_profit
    )
    return {
        "example": "two-route",
        "warning": "enumeration compares local regional solves; it is not a global proof",
        "primal": {
            "status": primal.status.value,
            "assignment": primal.candidate.assignment.tolist(),
            "true_profit": primal_profit,
            "region_iterations": primal.candidate.region_iteration_count,
            "boundary_search_complete": primal.candidate.boundary_search_complete,
        },
        "enumeration": {
            **enumeration.metrics(),
            "best_assignment": None if best is None else best.assigned_bundles.tolist(),
            "best_prices": None if best is None else best.true_assignment.menu.prices.tolist(),
            "best_allowances": (
                None if best is None else best.true_assignment.menu.allowances.tolist()
            ),
        },
        "enumerated_minus_primal_profit": profit_gap,
    }
