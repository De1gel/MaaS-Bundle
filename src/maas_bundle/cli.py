"""Command-line entry point for reproducible MaaS bundle-design runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from time import perf_counter

import numpy as np

from maas_bundle.benchmarks import enumerate_assignment_regions
from maas_bundle.design.primal import solve_primal
from maas_bundle.design.regional import ScipyRegionalSolver
from maas_bundle.examples.sioux_falls import (
    DIRECTED_LINK_COUNT,
    NODE_COUNT,
    run_sioux_falls_example,
    run_sioux_falls_multistart,
    sioux_falls_od_matrix,
)
from maas_bundle.examples.sioux_falls import (
    SOURCE_URL as SIOUX_FALLS_SOURCE_URL,
)
from maas_bundle.examples.sioux_falls_config import DEFAULT_SIOUX_FALLS_CONFIG
from maas_bundle.examples.two_route import (
    SOURCE_AM_DEMAND,
    SOURCE_URL,
    build_two_route_problem,
    initial_two_route_menu,
    run_two_route_example,
    run_two_route_multistart,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a selected command and return a process exit status."""

    parser = argparse.ArgumentParser(prog="maas-bundle")
    subcommands = parser.add_subparsers(dest="command", required=True)
    example = subcommands.add_parser("example", help="run a bundled reproducible example")
    example.add_argument("name", choices=("two-route", "sioux-falls", "souxf"))
    compare = subcommands.add_parser("compare", help="compare primal and enumeration")
    compare.add_argument("name", choices=("two-route",))
    multistart = subcommands.add_parser(
        "multistart",
        help="run a reproducible multi-start region experiment",
    )
    multistart.add_argument("name", choices=("two-route", "sioux-falls", "souxf"))
    multistart.add_argument("--starts", type=int, default=30)
    multistart.add_argument("--seed", type=int, default=20260901)
    arguments = parser.parse_args(argv)

    if arguments.command == "example" and arguments.name == "two-route":
        diagnostics = run_two_route_example()
        candidate = diagnostics.candidate
        payload = {
            "example": "two-route",
            "source": SOURCE_URL,
            "status": diagnostics.status.value,
            "valid": diagnostics.valid,
            "prices": candidate.menu.prices.tolist(),
            "allowances": candidate.menu.allowances.tolist(),
            "assignment": candidate.assignment.tolist(),
            "normalized_profit": candidate.true_profit,
            "am_profit_at_source_demand": (
                None
                if candidate.true_profit is None
                else SOURCE_AM_DEMAND * candidate.true_profit
            ),
            "region_iterations": candidate.region_iteration_count,
            "boundary_search_complete": candidate.boundary_search_complete,
            "generation_iterations": candidate.fixed_generation_iteration_count,
            "double_ties": diagnostics.double_tie_count,
        }
        print(json.dumps(payload, indent=2))
        return 0 if diagnostics.valid else 1

    if arguments.command == "example" and arguments.name in {"sioux-falls", "souxf"}:
        payload = _run_sioux_falls()
        print(json.dumps(payload, indent=2))
        return 0 if payload["valid"] else 1

    if arguments.command == "compare" and arguments.name == "two-route":
        payload = _compare_two_route()
        print(json.dumps(payload, indent=2))
        return 0 if payload["enumeration"]["best_assignment"] is not None else 1

    if arguments.command == "multistart" and arguments.name == "two-route":
        payload = _multistart_two_route(arguments.starts, arguments.seed)
        print(json.dumps(payload, indent=2))
        return 0 if payload["consistent_starts"] else 1

    if arguments.command == "multistart" and arguments.name in {"sioux-falls", "souxf"}:
        payload = _multistart_sioux_falls(arguments.starts, arguments.seed)
        print(json.dumps(payload, indent=2))
        return 0 if payload["consistent_starts"] else 1

    raise AssertionError("argparse accepted an unsupported command")  # pragma: no cover


def _compare_two_route() -> dict[str, object]:
    problem = build_two_route_problem()
    initial_menu = initial_two_route_menu()
    solver = ScipyRegionalSolver(
        "SLSQP",
        options={"maxiter": 2_000, "ftol": 1e-10},
    )
    primal = solve_primal(problem, initial_menu, solver, max_region_iterations=20)
    enumeration = enumerate_assignment_regions(problem, initial_menu, solver)
    best = enumeration.best_candidate
    primal_profit = primal.candidate.true_profit
    best_profit = None if best is None else best.true_profit
    profit_gap = (
        None
        if primal_profit is None or best_profit is None
        else best_profit - primal_profit
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


def _multistart_two_route(start_count: int, seed: int) -> dict[str, object]:
    result = run_two_route_multistart(start_count, seed=seed)
    best = result.best_run
    return {
        "example": "two-route",
        "seed": seed,
        "warning": "multi-start samples local attraction regions; it is not a global proof",
        **result.metrics(),
        "all_consistent_starts_same_region": result.distinct_region_count == 1,
        "best_assignment": None if best is None else list(best.final_assignment),
        "best_start": None if best is None else best.number,
        "regions": result.region_summaries(),
    }


def _run_sioux_falls() -> dict[str, object]:
    started = perf_counter()
    diagnostics = run_sioux_falls_example()
    candidate = diagnostics.candidate
    return {
        "example": "sioux-falls",
        "source": SIOUX_FALLS_SOURCE_URL,
        "experiment": DEFAULT_SIOUX_FALLS_CONFIG.metadata(),
        "synthetic_extension": True,
        "nodes": NODE_COUNT,
        "directed_links": DIRECTED_LINK_COUNT,
        "positive_od_pairs": int(np.count_nonzero(sioux_falls_od_matrix())),
        "traveler_classes": candidate.assignment.size,
        "status": diagnostics.status.value,
        "valid": diagnostics.valid,
        "prices": candidate.menu.prices.tolist(),
        "allowances": candidate.menu.allowances.tolist(),
        "assignment_counts": np.bincount(candidate.assignment, minlength=3).tolist(),
        "normalized_profit": candidate.true_profit,
        "region_iterations": candidate.region_iteration_count,
        "boundary_search_complete": candidate.boundary_search_complete,
        "generation_iterations": candidate.fixed_generation_iteration_count,
        "solve_time_seconds": candidate.solve_time_seconds,
        "end_to_end_seconds": perf_counter() - started,
        "double_ties": diagnostics.double_tie_count,
    }


def _multistart_sioux_falls(start_count: int, seed: int) -> dict[str, object]:
    started = perf_counter()
    result = run_sioux_falls_multistart(start_count, seed=seed)
    best = result.best_run
    regions = []
    for summary in result.region_summaries():
        assignment = np.asarray(summary.pop("assignment"), dtype=np.int64)
        regions.append(
            {
                "assignment_signature": hashlib.sha256(assignment.tobytes()).hexdigest()[:12],
                "assignment_counts": np.bincount(assignment, minlength=3).tolist(),
                **summary,
            }
        )
    return {
        "example": "sioux-falls",
        "source": SIOUX_FALLS_SOURCE_URL,
        "experiment": DEFAULT_SIOUX_FALLS_CONFIG.metadata(),
        "synthetic_extension": True,
        "seed": seed,
        "traveler_classes": 2 * int(np.count_nonzero(sioux_falls_od_matrix())),
        "warning": "multi-start samples local attraction regions; it is not a global proof",
        **result.metrics(),
        "end_to_end_seconds": perf_counter() - started,
        "all_consistent_starts_same_region": result.distinct_region_count == 1,
        "best_assignment_counts": (
            None
            if best is None
            else np.bincount(best.final_assignment, minlength=3).tolist()
        ),
        "best_start": None if best is None else best.number,
        "regions": regions,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
