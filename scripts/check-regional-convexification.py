"""Verify a feasible CCP prototype with LP cutting planes and the exact oracle.

This is a fixed-assignment experiment, not a replacement regional solver or a
global-optimality claim. Run with PYTHONPATH=src in the maas-bundle environment.
No demand discretization or new dual-vertex enumeration is performed.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.regional import MenuVariableLayout
from maas_bundle.examples.sioux_falls import (
    build_sioux_falls_problem,
    initial_sioux_falls_menu,
)


def convex_choice_oracle(problem, ii, aa, bb, price_matrix, anchor_grad, intercept, query):
    """Return exact convex/original residuals and a convex subgradient matrix."""
    v = problem.evaluate_valuations(query)
    g = -v.expected_omega
    alt_affine = intercept + np.sum(anchor_grad[ii, aa] * query.allowances[aa], axis=1)
    residual = query.prices[bb] - query.prices[aa] + v.route_values[ii, bb] - alt_affine
    original = query.prices[bb] - query.prices[aa] + v.route_values[ii, bb] - v.route_values[ii, aa]
    matrix = price_matrix.copy()
    row = np.arange(len(ii))
    mask_b, mask_a = bb > 0, aa > 0
    designed, modes = problem.bundle_count - 1, problem.mode_count
    for mode in range(modes):
        matrix[row[mask_b], designed + (bb[mask_b] - 1) * modes + mode] += g[
            ii[mask_b], bb[mask_b], mode
        ]
        matrix[row[mask_a], designed + (aa[mask_a] - 1) * modes + mode] -= anchor_grad[
            ii[mask_a], aa[mask_a], mode
        ]
    return residual, original, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-iterations", type=int, default=15)
    parser.add_argument("--cut-iterations", type=int, default=120)
    parser.add_argument("--radius", type=float, default=48.0)
    parser.add_argument("--starts", type=int, default=3)
    args = parser.parse_args()
    problem = build_sioux_falls_problem(
        oracle_cache_path="results/sioux_falls_five_modes/dual_vertices.npz"
    )
    initial = initial_sioux_falls_menu()
    layout = MenuVariableLayout(problem.bundle_count, problem.mode_count)
    report = {
        "method": "feasible CCP; convex subproblems solved by LP cutting planes",
        "scope": "fixed assignment; linear provision cost; bounded step box",
        "traveler_count": problem.traveler_count,
        "mode_count": problem.mode_count,
        "upper_variables": layout.variable_count,
        "normalized_profit_units": "USD/representative weighted person/month",
        "feasibility_tolerance": 1e-8,
        "allowance_step_radius": args.radius,
        "price_step_radius": 100.0,
        "outer_iteration_limit": args.outer_iterations,
        "cut_iteration_limit": args.cut_iterations,
        "runs": [],
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    # A rank > 1 rules out a differentiable common scalar valuation coordinate.
    coordinate_allowances = initial.allowances.copy()
    coordinate_allowances[1] += np.array([0.123, 0.456, 0.789, 0.246, 0.357])
    coordinate_menu = BundleMenu(initial.prices, coordinate_allowances)
    initial_values = problem.evaluate_valuations(coordinate_menu)
    gradients = -initial_values.expected_omega[:, 1]
    finite_difference = np.empty_like(gradients)
    step = 1e-4
    for mode in range(problem.mode_count):
        high, low = coordinate_allowances.copy(), coordinate_allowances.copy()
        high[1, mode] += step
        low[1, mode] -= step
        vh = problem.evaluate_valuations(BundleMenu(initial.prices, high)).route_values[:, 1]
        vl = problem.evaluate_valuations(BundleMenu(initial.prices, low)).route_values[:, 1]
        finite_difference[:, mode] = (vh - vl) / (2 * step)
    report["valuation_coordinate_check"] = {
        "allowance_point": coordinate_allowances[1].tolist(),
        "oracle_gradient_singular_values": np.linalg.svd(gradients, compute_uv=False).tolist(),
        "finite_difference_singular_values": np.linalg.svd(
            finite_difference, compute_uv=False
        ).tolist(),
        "gradient_max_difference": float(np.max(np.abs(gradients - finite_difference))),
    }

    # Construct actual paired-choice boundary points; search for a strict Jensen
    # gap using the same cached oracle used in the full experiment.
    rng = np.random.default_rng(20261021)
    best_gap = -np.inf
    counterexample = None
    for _ in range(16):
        low = rng.uniform(0, 20, problem.mode_count)
        high = low + rng.uniform(10, 70, problem.mode_count)
        premium = high + 40
        values = []
        for allowance in (low, high, (low + high) / 2):
            menu = BundleMenu(
                np.array([0.0, 10.0, 20.0]),
                np.array([np.zeros(problem.mode_count), allowance, premium]),
            )
            values.append(problem.evaluate_valuations(menu).route_values)
        gap = (values[0][:, 1] + values[1][:, 1]) / 2 - values[2][:, 1]
        traveler = int(np.argmax(gap))
        if gap[traveler] > best_gap:
            best_gap = float(gap[traveler])
            prices = [10 + v[traveler, 1] - v[traveler, 2] for v in values[:2]]
            counterexample = {
                "traveler_index": traveler,
                "assigned_option": 2,
                "alternative_option": 1,
                "basic_allowance_endpoints": [low.tolist(), high.tolist()],
                "premium_allowance_fixed": premium.tolist(),
                "basic_price_fixed": 10.0,
                "premium_price_endpoints": [float(p) for p in prices],
                "endpoint_choice_residuals": [
                    float(prices[j] - 10 + values[j][traveler, 2] - values[j][traveler, 1])
                    for j in range(2)
                ],
                "midpoint_choice_violation": best_gap,
                "scope": "one exact choice constraint; not the full assignment region",
            }
    report["nonconvex_choice_counterexample"] = counterexample
    save()

    for start in range(args.starts):
        if start == 0:
            menu = initial
        else:
            source_dir = Path("results/sioux_falls_five_modes/multistart_100_20261019")
            record = json.loads((source_dir / f"start_{start + 1:03d}.json").read_text())
            menu = BundleMenu(
                np.array(record["initial_prices"]), np.array(record["initial_allowances"])
            )
        started = perf_counter()
        assignment = problem.evaluate_assignment(menu, choice_tolerance=1e-9)
        alpha = assignment.bundle_indices
        demand = assignment.bundle_demand
        # Linear minimization objective is minus the fixed-assignment profit.
        objective = np.concatenate(
            (
                -demand[1:],
                (demand[1:, None] * problem.unit_allowance_cost).reshape(-1),
            )
        )
        ii, aa = np.where(np.arange(problem.bundle_count)[None, :] != alpha[:, None])
        bb = alpha[ii]
        count = len(ii)
        row = np.arange(count)
        designed = problem.bundle_count - 1
        modes = problem.mode_count
        pmat = np.zeros((count, layout.variable_count))
        mask_b, mask_a = bb > 0, aa > 0
        pmat[row[mask_b], bb[mask_b] - 1] += 1
        pmat[row[mask_a], aa[mask_a] - 1] -= 1
        order = np.zeros((max(designed - 1, 0), layout.variable_count))
        for b in range(designed - 1):
            order[b, b], order[b, b + 1] = 1, -1
        run = {
            "start": start + 1,
            "initial_fixed_assignment_profit": assignment.total_profit,
            "choice_constraint_count": count,
            "initial_prices": menu.prices.tolist(),
            "initial_allowances": menu.allowances.tolist(),
            "history": [],
            "status": "outer_iteration_limit",
        }
        report["runs"].append(run)
        for outer in range(args.outer_iterations):
            anchor = menu
            x0 = layout.pack(anchor)
            v0 = problem.evaluate_valuations(anchor)
            g0 = -v0.expected_omega
            alternative_intercept = v0.route_values[ii, aa] - np.sum(
                g0[ii, aa] * anchor.allowances[aa], axis=1
            )

            evaluate = partial(
                convex_choice_oracle, problem, ii, aa, bb, pmat, g0, alternative_intercept
            )

            residual, original, matrix = evaluate(anchor)
            assert np.max(original) <= 1e-8, "entry menu violates the original region"
            cuts = [order, matrix]
            rhs = [np.zeros(len(order)), matrix @ x0 - residual]
            radius = np.concatenate(
                (np.full(designed, 100.0), np.full(designed * modes, args.radius))
            )
            bounds = list(zip(np.maximum(0, x0 - radius), x0 + radius, strict=True))
            inner_success = False
            for inner in range(args.cut_iterations):
                lp_solves = inner + 1
                lp = linprog(
                    objective,
                    A_ub=np.vstack(cuts),
                    b_ub=np.concatenate(rhs),
                    bounds=bounds,
                    method="highs",
                    options={
                        "primal_feasibility_tolerance": 1e-9,
                        "dual_feasibility_tolerance": 1e-9,
                    },
                )
                if not lp.success:
                    run["status"] = "lp_failed"
                    run["lp_message"] = lp.message
                    break
                assert np.min(lp.x) >= -1e-8, "LP nonnegativity violation exceeds tolerance"
                query_vector = np.maximum(lp.x, 0.0)
                query = layout.unpack(query_vector)
                residual, original, matrix = evaluate(query)
                if np.max(residual) <= 1e-8:
                    inner_success = True
                    break
                violated = residual > 1e-8
                cuts.append(matrix[violated])
                rhs.append(matrix[violated] @ query_vector - residual[violated])
            if not inner_success:
                if run["status"] != "lp_failed":
                    run["status"] = "cut_iteration_limit"
                break
            profit = float(-objective @ query_vector)
            previous = float(-objective @ x0)
            assert profit >= previous - 1e-7, "profit decreased"
            assert np.max(original) <= 1e-8, "full original region check failed"
            # The LP is an outer relaxation of the convex inner approximation.
            # A feasible LP optimum therefore certifies its objective to the
            # stated constraint tolerance (not exact arithmetic feasibility).
            entry = {
                "iteration": outer + 1,
                "fixed_assignment_profit": profit,
                "improvement": profit - previous,
                "lp_solves": lp_solves,
                "lp_upper_minus_candidate_profit": float(-lp.fun - profit),
                "lp_cut_count": sum(len(c) for c in cuts),
                "convex_subproblem_violation": max(0.0, float(np.max(residual))),
                "original_choice_violation": max(0.0, float(np.max(original))),
                "step_inf_norm": float(np.max(np.abs(query_vector - x0))),
                "elapsed_seconds": perf_counter() - started,
            }
            run["history"].append(entry)
            menu = query
            run["final_prices"] = menu.prices.tolist()
            run["final_allowances"] = menu.allowances.tolist()
            run["final_fixed_assignment_profit"] = profit
            run["elapsed_seconds"] = perf_counter() - started
            print(json.dumps({"start": start + 1, **entry}), flush=True)
            save()
            if profit - previous <= 1e-6:
                run["status"] = "small_improvement; not a local-optimality certificate"
                break
        run["elapsed_seconds"] = perf_counter() - started
        save()
    print(
        json.dumps({"output": str(args.output), "statuses": [r["status"] for r in report["runs"]]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
