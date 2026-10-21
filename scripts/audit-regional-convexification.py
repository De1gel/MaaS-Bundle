"""Recompute prototype feasibility from individual cached integration oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from maas_bundle.design.boundary import ChoiceConstraintIndex, CriticalConstraintSet
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.regional import RestrictedRegionalProblem, ScipyRegionalSolver
from maas_bundle.examples.sioux_falls import build_sioux_falls_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--polish", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    problem = build_sioux_falls_problem(
        oracle_cache_path="results/sioux_falls_five_modes/dual_vertices.npz"
    )

    def values(allowances):
        return np.array(
            [
                [
                    oracle.expected_value(allowances[b], problem.demand_distributions[i])
                    for b, oracle in enumerate(oracles)
                ]
                for i, oracles in enumerate(problem.route_oracles)
            ]
        )

    audit = {"scope": "independent final-point integration and full choice checks", "runs": []}
    for run in report["runs"]:
        initial = BundleMenu(np.array(run["initial_prices"]), np.array(run["initial_allowances"]))
        # Recover the documented fixed assignment, including optimistic ties.
        alpha = problem.evaluate_assignment(initial, choice_tolerance=1e-9).bundle_indices
        final = BundleMenu(np.array(run["final_prices"]), np.array(run["final_allowances"]))
        g = values(final.allowances)
        generalized = g + final.prices
        violation = float(
            np.max(generalized[np.arange(len(alpha)), alpha] - generalized.min(axis=1))
        )
        margins = final.prices.copy()
        margins[1:] -= final.allowances[1:] @ problem.unit_allowance_cost
        profit = float(problem.traveler_weights @ margins[alpha])
        initial_margins = initial.prices.copy()
        initial_margins[1:] -= initial.allowances[1:] @ problem.unit_allowance_cost
        initial_profit = float(problem.traveler_weights @ initial_margins[alpha])

        # Price-only LP is an independent exact conditional optimization check.
        ii, aa = np.where(np.arange(problem.bundle_count)[None, :] != alpha[:, None])
        bb = alpha[ii]
        matrix = np.zeros((len(ii) + 1, problem.bundle_count - 1))
        for row, (assigned, alternative) in enumerate(zip(bb, aa, strict=True)):
            if assigned:
                matrix[row, assigned - 1] += 1
            if alternative:
                matrix[row, alternative - 1] -= 1
        matrix[-1] = [1, -1]
        rhs = np.concatenate((g[ii, aa] - g[ii, bb], [0.0]))
        demand = np.bincount(alpha, weights=problem.traveler_weights, minlength=3)
        lp = linprog(-demand[1:], A_ub=matrix, b_ub=rhs, bounds=(0, None), method="highs")
        assert lp.success, lp.message
        conditional_price_gain = float(-lp.fun - demand[1:] @ final.prices[1:])

        full = CriticalConstraintSet(alpha, problem.bundle_count)
        full.add_all(
            tuple(
                ChoiceConstraintIndex(int(i), int(b), int(a))
                for i, b, a in zip(ii, bb, aa, strict=True)
            )
        )
        regional = RestrictedRegionalProblem(problem, full)
        certificate = regional.first_order_certificate(
            regional.layout.pack(final), stationarity_tolerance=1e-5, active_tolerance=1e-4
        )
        max_trace_violation = max(h["original_choice_violation"] for h in run["history"])
        all_profits = [initial_profit] + [h["fixed_assignment_profit"] for h in run["history"]]
        min_improvement = float(np.min(np.diff(all_profits)))
        order_violation = max(0.0, float(np.max(-np.diff(final.prices))))
        entry = {
            "start": run["start"],
            "initial_profit": initial_profit,
            "final_profit": profit,
            "profit_recomputation_difference": profit - run["final_fixed_assignment_profit"],
            "final_original_choice_violation": violation,
            "max_recorded_original_choice_violation": max_trace_violation,
            "price_order_violation": order_violation,
            "minimum_recorded_profit_improvement": min_improvement,
            "conditional_price_lp_gain": conditional_price_gain,
            "normalized_kkt_stationarity_residual": certificate.normalized_stationarity_residual,
            "kkt_is_not_a_local_optimality_proof": True,
            "outer_iterations": len(run["history"]),
            "lp_solves": sum(h["lp_solves"] for h in run["history"]),
            "elapsed_seconds": run["elapsed_seconds"],
            "termination": run["status"],
        }
        if args.polish:
            solver = ScipyRegionalSolver(
                options={"maxiter": 200, "ftol": 1e-11},
                enable_recovery=False,
                enable_fallback=False,
            )
            polished = solver.solve(regional, final)
            pg = values(polished.menu.allowances) + polished.menu.prices
            pv = float(np.max(pg[np.arange(len(alpha)), alpha] - pg.min(axis=1)))
            polished_margins = polished.menu.prices.copy()
            polished_margins[1:] -= polished.menu.allowances[1:] @ problem.unit_allowance_cost
            direct_profit = float(problem.traveler_weights @ polished_margins[alpha])
            assert polished.success and pv <= 1e-7, "original-constraint polish failed acceptance"
            assert abs(direct_profit - polished.objective_value) <= 1e-8
            assert direct_profit >= profit - 1e-7
            entry["original_constraint_polish"] = {
                **polished.metrics(),
                "direct_oracle_choice_violation": pv,
                "direct_recomputed_profit": direct_profit,
                "profit_recomputation_difference": direct_profit - polished.objective_value,
                "prices": polished.menu.prices.tolist(),
                "allowances": polished.menu.allowances.tolist(),
                "profit_change": polished.objective_value - profit,
                "recovery_and_backend_fallback_disabled": True,
            }
            print(json.dumps({"start": run["start"], "polish": polished.metrics()}), flush=True)
        assert abs(entry["profit_recomputation_difference"]) <= 1e-8
        assert violation <= 1e-8 and max_trace_violation <= 1e-8
        assert order_violation <= 1e-8 and min_improvement >= -1e-7
        audit["runs"].append(entry)

    witness = report["nonconvex_choice_counterexample"]
    allowances = [
        np.array([np.zeros(5), a, witness["premium_allowance_fixed"]])
        for a in witness["basic_allowance_endpoints"]
    ]
    midpoint = (allowances[0] + allowances[1]) / 2
    i = witness["traveler_index"]
    v = [values(a)[i] for a in (*allowances, midpoint)]
    p = witness["premium_price_endpoints"]
    residuals = [p[k] - 10 + v[k][2] - v[k][1] for k in range(2)]
    middle_residual = float(sum(p) / 2 - 10 + v[2][2] - v[2][1])
    assert max(abs(r) for r in residuals) < 1e-8 and middle_residual > 1e-3
    audit["nonconvex_constraint_witness"] = {
        "endpoint_residuals": residuals,
        "midpoint_violation": middle_residual,
    }
    audit["passed"] = True
    output = args.report.with_name("audit_polish.json" if args.polish else "audit.json")
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
