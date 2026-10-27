"""Replay the problematic call with KKT acceptance and backend fallback enabled."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from maas_bundle.design.regional import ScipyRegionalSolver

_SPEC = importlib.util.spec_from_file_location(
    "selected_nlp_checks", Path(__file__).with_name("verify-selected-nlps.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


def diagnose(regional, full, menu):
    x = regional.layout.pack(menu)
    h = regional.constraint_values(x)
    result = _HELPERS.candidate_record(regional, full, menu)
    result["first_order_checks"] = []
    for tolerance in (1e-8, 1e-6, 1e-4, 1e-3):
        certificate = regional.first_order_certificate(
            x,
            feasibility_tolerance=1e-7,
            stationarity_tolerance=1e-5,
            active_tolerance=tolerance,
        )
        comp = [0.0]
        for index, weight in zip(
            certificate.active_inequality_indices, certificate.inequality_multipliers, strict=True
        ):
            comp.append(abs(weight * h[index]))
        for index, weight in zip(
            certificate.active_lower_bound_indices, certificate.lower_bound_multipliers, strict=True
        ):
            comp.append(abs(weight * x[index]))
        for index, weight in zip(
            certificate.active_upper_bound_indices, certificate.upper_bound_multipliers, strict=True
        ):
            comp.append(abs(weight * (regional.upper_bounds[index] - x[index])))
        result["first_order_checks"].append(
            {
                "active_tolerance": tolerance,
                "normalized_stationarity": certificate.normalized_stationarity_residual,
                "primal_violation": certificate.primal_violation,
                "absolute_complementarity": max(comp),
                "adapter_first_order_passed": certificate.satisfies_kkt,
                "independent_screen_passed": bool(certificate.satisfies_kkt and max(comp) <= 1e-6),
            }
        )
    return result


def run(args):
    trace = json.loads(args.trace.read_text())
    assert trace["complete"] and trace["matches_saved_fields"]
    call = trace["calls"][args.call - 1]
    settings = trace["source_settings"]
    problem = _HELPERS.build_sioux_falls_problem(
        oracle_cache_path="results/sioux_falls_five_modes/dual_vertices.npz"
    )
    alpha = np.array(trace["regions"][call["assignment_sha256"]]["assigned_bundles"])
    regional = _HELPERS.make_regional(problem, alpha, call["choice_constraints"])
    full = _HELPERS._HELPERS._HELPERS.regional_problem(problem, alpha)
    saved_return = _HELPERS._HELPERS.restore_menu(call["menu"])
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "retry.json"
    report = {
        "call": args.call,
        "region": call["region"],
        "cg_round": call["cg_round"],
        "assignment_sha256": call["assignment_sha256"],
        "choice_constraints": call["choice_constraints"],
        "source_settings": settings,
        "source_metrics": call["metrics"],
        "source_return": diagnose(regional, full, saved_return),
        "changed_settings": {"check_kkt": True, "enable_fallback": True},
        "same_source_nlp_options": {"maxiter": settings["nlp_max_iterations"], "ftol": 1e-9},
        "fallback_policy": {
            "backend": "trust-constr",
            "gtol": 1e-8,
            "active_set_tolerance": 1e-3,
            "independent_recheck_active_tolerance": 1e-4,
        },
        "runs": [],
        "complete": False,
        "local_or_global_optimality_certificate": False,
    }

    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    class RecordingSolver(ScipyRegionalSolver):
        def _solve_once(self, restricted, initial, *, numerical_constraints=False):
            result = super()._solve_once(
                restricted, initial, numerical_constraints=numerical_constraints
            )
            row = {
                "metrics": result.metrics(),
                "numerical_constraints": numerical_constraints,
                "initial_menu": _HELPERS.record_menu(initial),
                "returned": diagnose(restricted, full, result.menu),
            }
            current["slsqp_attempts"].append(row)
            save()
            print(
                json.dumps(
                    {
                        "start": current["start"],
                        "attempt": len(current["slsqp_attempts"]),
                        "solver_status": result.status.value,
                        "native_success": result.native_success,
                        "profit": result.objective_value,
                        "kkt_residual": (
                            result.first_order_certificate.normalized_stationarity_residual
                        ),
                        "violation": result.maximum_constraint_violation,
                    }
                ),
                flush=True,
            )
            return result

    save()
    for label, initial_record in (
        ("original_nlp_input", call["initial_menu"]),
        ("previous_bad_return", call["menu"]),
    ):
        current = {"start": label, "slsqp_attempts": [], "complete": False}
        report["runs"].append(current)
        solver = RecordingSolver(
            options={"maxiter": settings["nlp_max_iterations"], "ftol": 1e-9},
            check_kkt=True,
            enable_recovery=settings["recovery"],
            slsqp_restarts=settings["slsqp_restarts"],
            numerical_constraint_recovery=settings["numerical_constraint_recovery"],
            enable_fallback=True,
        )
        initial = _HELPERS._HELPERS.restore_menu(initial_record)
        current["initial"] = diagnose(regional, full, initial)
        started = perf_counter()
        solved = solver.solve(regional, initial)
        current.update(
            {
                "complete": True,
                "seconds": perf_counter() - started,
                "final_metrics": solved.metrics(),
                "final": diagnose(regional, full, solved.menu),
                "fallback_used": any("trust-constr" in name for name in solved.attempted_solvers),
            }
        )
        # Independent price optimization via direct integration and HiGHS.
        priced = _HELPERS.independent_price_check(regional, solved.menu.allowances)
        current["conditional_price_gain"] = priced["profit"] - solved.objective_value
        current["independent_price_check"] = priced
        save()
        print(
            json.dumps(
                {
                    "start": label,
                    "complete": True,
                    "fallback_used": current["fallback_used"],
                    "metrics": solved.metrics(),
                    "seconds": current["seconds"],
                }
            ),
            flush=True,
        )
    report["complete"] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--call", type=int, default=15)
    parser.add_argument(
        "--trace", type=Path, default=Path("results/longest_trajectory_sosc_20261022/replay.json")
    )
    parser.add_argument("--output", type=Path, default=Path("results/call_15_kkt_retry_20261027"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
