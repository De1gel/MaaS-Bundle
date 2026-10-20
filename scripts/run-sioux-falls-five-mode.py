"""Attempt bundle design using saved vertices, with a finite wall-time budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path
from time import perf_counter

import numpy as np

from maas_bundle.benchmarks.sioux_falls import generate_sioux_falls_initial_menus
from maas_bundle.benchmarks.sioux_falls_preprocessing import DEFAULT_VERTEX_CACHE
from maas_bundle.design.primal import solve_primal
from maas_bundle.design.regional import ScipyRegionalSolver
from maas_bundle.examples.sioux_falls import build_sioux_falls_problem, initial_sioux_falls_menu
from maas_bundle.examples.sioux_falls_config import DEFAULT_SIOUX_FALLS_CONFIG


def assignment_signature(assignment):
    return hashlib.sha256(np.asarray(assignment, dtype="<i8").tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--oracle-cache", default=DEFAULT_VERTEX_CACHE)
    parser.add_argument(
        "--output", default="results/sioux_falls_five_modes/design_exploratory.json"
    )
    parser.add_argument("--wall-seconds", type=int, default=180)
    parser.add_argument("--nlp-iterations", type=int, default=200)
    parser.add_argument("--max-regions", type=int, default=10)
    parser.add_argument("--seed", type=int, help="use a positive-profit random start instead")
    parser.add_argument("--slsqp-restarts", type=int, default=4)
    parser.add_argument("--search-restarts", type=int, default=3)
    parser.add_argument(
        "--numerical-constraint-recovery", action=argparse.BooleanOptionalAction, default=True,
        help="use numerical constraint derivatives on the final SLSQP retry",
    )
    parser.add_argument(
        "--solver-fallback", action="store_true", help="allow trust-constr after SLSQP retries"
    )
    parser.add_argument(
        "--strict-kkt", action="store_true", help="enable independent KKT acceptance"
    )
    parser.add_argument(
        "--no-recovery", action="store_true", help="disable solver restarts/fallback"
    )
    args = parser.parse_args()
    if min(args.wall_seconds, args.nlp_iterations, args.max_regions) <= 0:
        parser.error("all run budgets must be positive")
    if min(args.slsqp_restarts, args.search_restarts) < 0:
        parser.error("restart counts must be nonnegative")
    search_restarts = args.search_restarts if not args.no_recovery else 0
    started = perf_counter()
    problem = build_sioux_falls_problem(oracle_cache_path=args.oracle_cache)
    menu = initial_sioux_falls_menu()
    if args.seed is not None:
        menu = generate_sioux_falls_initial_menus(1, seed=args.seed, problem=problem)[0]
    incumbent = problem.evaluate_assignment(menu)
    initial_profit = incumbent.total_profit
    payload = {
        "experiment": DEFAULT_SIOUX_FALLS_CONFIG.metadata(),
        "oracle_cache": args.oracle_cache,
        "traveler_types": problem.traveler_count,
        "upper_variables": 12,
        "settings": {
            "wall_seconds": args.wall_seconds,
            "nlp_max_iterations": args.nlp_iterations,
            "max_regions": args.max_regions,
            "recovery": not args.no_recovery,
            "slsqp_restarts": args.slsqp_restarts,
            "search_restarts": search_restarts,
            "numerical_constraint_recovery": args.numerical_constraint_recovery,
            "solver_fallback": args.solver_fallback and not args.no_recovery,
            "check_kkt": args.strict_kkt,
            "seed": args.seed,
        },
        "initial_profit": initial_profit,
        "initial_prices": menu.prices.tolist(),
        "initial_allowances": menu.allowances.tolist(),
        "initial_assignment_sha256": assignment_signature(incumbent.bundle_indices),
        "status": "running",
        "valid": False,
        "nlp_attempts": [],
        "search_attempts": [],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        payload.update(
            elapsed_seconds=perf_counter() - started,
            best_observed_profit=incumbent.total_profit,
            best_observed_prices=incumbent.menu.prices.tolist(),
            best_observed_allowances=incumbent.menu.allowances.tolist(),
        )
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)

    backend = ScipyRegionalSolver(
        options={"maxiter": args.nlp_iterations, "ftol": 1e-9},
        enable_recovery=not args.no_recovery,
        check_kkt=args.strict_kkt,
        slsqp_restarts=args.slsqp_restarts,
        enable_fallback=args.solver_fallback,
        numerical_constraint_recovery=args.numerical_constraint_recovery,
    )

    class LoggedSolver:
        name = backend.name
        feasibility_tolerance = backend.feasibility_tolerance
        check_kkt = backend.check_kkt
        optimality_tolerance = backend.optimality_tolerance
        active_set_tolerance = backend.active_set_tolerance
        objective_tolerance = backend.objective_tolerance

        def solve(self, regional_problem, initial_menu):
            nonlocal incumbent
            number = len(payload["nlp_attempts"]) + 1
            print(f"NLP {number}: {regional_problem.constraint_count} constraints", flush=True)
            result = backend.solve(regional_problem, initial_menu)
            payload["nlp_attempts"].append(result.metrics())
            observed = problem.evaluate_assignment(result.menu)
            if observed.total_profit > incumbent.total_profit:
                incumbent = observed
            persist()
            print(json.dumps(result.metrics()), flush=True)
            return result

    def time_limit(signum, frame):
        raise TimeoutError("bundle-design wall-time budget exhausted")

    def record_diagnostics(diagnostics):
        candidate = diagnostics.candidate
        payload.update(
            status=diagnostics.status.value,
            valid=diagnostics.valid,
            diagnostics=diagnostics.metrics(),
            diagnostic_message=diagnostics.message,
            candidate_metrics=candidate.metrics(),
            candidate_prices=candidate.menu.prices.tolist(),
            candidate_allowances=candidate.menu.allowances.tolist(),
            candidate_assignment_counts=np.bincount(candidate.assignment, minlength=3).tolist(),
            candidate_assignment_sha256=assignment_signature(candidate.assignment),
            candidate_region_history=[
                {
                    "region": step.number,
                    "assumed_assignment_sha256": assignment_signature(step.assumed_assignment),
                    "revisited_assignment": step.revisited_assignment,
                    "regional_objective": step.regional_objective,
                    "true_profit": step.true_profit,
                    "accepted": step.accepted,
                    "changed_travelers": step.changed_traveler_count,
                    "boundary_origin": step.boundary_origin,
                    "boundary_returned": step.boundary_returned,
                    "user_ties": (
                        int(np.count_nonzero(step.true_assignment.has_numerical_cost_ties))
                        if step.true_assignment is not None else None
                    ),
                    "double_ties": (
                        int(np.count_nonzero(step.true_assignment.has_double_ties))
                        if step.true_assignment is not None else None
                    ),
                    "generation_iterations": step.fixed_assignment_result.iteration_count,
                    "nlp_objectives": [
                        iteration.solve_result.objective_value
                        for iteration in step.fixed_assignment_result.iterations
                    ],
                    "prices": step.fixed_assignment_result.menu.prices.tolist(),
                    "allowances": step.fixed_assignment_result.menu.allowances.tolist(),
                }
                for step in candidate.iterations
            ],
        )

    persist()
    previous_handler = signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(args.wall_seconds)
    try:
        for search_number in range(search_restarts + 1):
            payload.update(status="running", active_search_attempt=search_number + 1)
            first_nlp = len(payload["nlp_attempts"])
            search_initial_profit = problem.evaluate_assignment(menu).total_profit
            diagnostics = solve_primal(
                problem, menu, LoggedSolver(), max_region_iterations=args.max_regions
            )
            record_diagnostics(diagnostics)
            payload["search_attempts"].append({
                "number": search_number + 1,
                "initial_profit": search_initial_profit,
                "initial_prices": menu.prices.tolist(),
                "initial_allowances": menu.allowances.tolist(),
                "initial_assignment_sha256": assignment_signature(
                    diagnostics.candidate.initial_assignment.bundle_indices
                ),
                "first_nlp_index": first_nlp,
                "completed_nlp_calls": len(payload["nlp_attempts"]) - first_nlp,
                "diagnostics": diagnostics.metrics(),
                "candidate_metrics": diagnostics.candidate.metrics(),
                "region_history": payload["candidate_region_history"],
            })
            persist()
            if diagnostics.valid:
                break
            # Every logged NLP endpoint has already had its full optimistic
            # response recomputed. Re-enter search at the best such feasible
            # menu rather than prescribing a random assignment or losing it.
            menu = incumbent.menu
            if search_number < search_restarts:
                print(json.dumps({
                    "restarting_search": search_number + 2,
                    "best_feasible_profit": incumbent.total_profit,
                }), flush=True)
    except TimeoutError as error:
        payload.update(status="time_limit", message=str(error), valid=False)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        persist()
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "valid",
                    "initial_profit",
                    "best_observed_profit",
                    "elapsed_seconds",
                )
            }
        ),
        flush=True,
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
