"""Independent checks of selected or all saved restricted NLPs.

Profile prices exactly, exhaust a small allowance grid, restart SLSQP with
numerical constraint derivatives, and bound the objective in explicit boxes
using a convexity-based LP relaxation. No global certificate for the unbounded
original NLP is claimed. The production model and saved menus are not changed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import gurobipy as gp
import numpy as np
from scipy.optimize import linprog

from maas_bundle.design.boundary import ChoiceConstraintIndex, CriticalConstraintSet
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.regional import RestrictedRegionalProblem, ScipyRegionalSolver
from maas_bundle.examples.sioux_falls import build_sioux_falls_problem

_SPEC = importlib.util.spec_from_file_location(
    "trajectory_checks", Path(__file__).with_name("check-longest-trajectory-sosc.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


def record_menu(menu):
    return {"prices": menu.prices.tolist(), "allowances": menu.allowances.tolist()}


def make_regional(problem, alpha, choices):
    critical = CriticalConstraintSet(alpha, problem.bundle_count)
    critical.add_all(tuple(ChoiceConstraintIndex(*index) for index in choices))
    return RestrictedRegionalProblem(problem, critical)


def price_lp(regional, allowances, env):
    """Solve the original retained price constraints with Gurobi, no approximation."""
    problem = regional.problem
    menu = BundleMenu(np.zeros(3), allowances)
    x = regional.layout.pack(menu)
    h = regional.constraint_values(x)
    matrix = np.zeros((len(h), 2))
    for j, index in enumerate(regional.active_indices):
        if index.assigned_bundle:
            matrix[j, index.assigned_bundle - 1] += 1
        if index.alternative_bundle:
            matrix[j, index.alternative_bundle - 1] -= 1
    matrix[-1] = [1, -1]
    with gp.Model(env=env) as model:
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        p = model.addMVar(2, lb=0, ub=problem.price_upper_bounds)
        model.addMConstr(matrix, p, "<", -h)
        model.setObjective(regional.bundle_demand[1:] @ p, gp.GRB.MAXIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            return None
        return BundleMenu(np.r_[0, np.maximum(p.X, 0)], allowances)


def profile_two_prices(g0, g1, g2, choices, price_caps):
    """Vectorized exact two-price LP for nonnegative bundle demands.

    The componentwise greatest feasible price pair maximizes every nonnegative
    linear price objective. Constraints reduce to price bounds and bounds on
    p2-p1. Return all pairs on the Cartesian allowance grid, plus feasibility.
    """
    shape = (g1.shape[1], g2.shape[1])
    low1, low2 = np.zeros(shape), np.zeros(shape)
    up1, up2 = np.full(shape, price_caps[0]), np.full(shape, price_caps[1])
    low_delta, up_delta = np.zeros(shape), np.full(shape, np.inf)
    for traveler, b, a in choices:
        values = [g0[traveler], g1[traveler, :, None], g2[traveler, None, :]]
        rhs = values[a] - values[b]
        if (b, a) == (1, 0):
            up1 = np.minimum(up1, rhs)
        elif (b, a) == (2, 0):
            up2 = np.minimum(up2, rhs)
        elif (b, a) == (0, 1):
            low1 = np.maximum(low1, -rhs)
        elif (b, a) == (0, 2):
            low2 = np.maximum(low2, -rhs)
        elif (b, a) == (1, 2):
            low_delta = np.maximum(low_delta, -rhs)
        elif (b, a) == (2, 1):
            up_delta = np.minimum(up_delta, rhs)
        else:
            raise ValueError((b, a))
    with np.errstate(invalid="ignore"):
        p1 = np.minimum(up1, up2 - low_delta)
        p2 = np.minimum(up2, up1 + up_delta)
        feasible = (
            np.isfinite(p1)
            & np.isfinite(p2)
            & (p1 >= low1 - 1e-9)
            & (p2 >= low2 - 1e-9)
            & (p2 - p1 >= low_delta - 1e-9)
            & (p2 - p1 <= up_delta + 1e-9)
        )
    return p1, p2, feasible


def allowance_grid(regional, center, radius):
    problem = regional.problem
    candidates = []
    for b in (1, 2):
        levels = [
            np.unique(np.maximum(0, value + radius * max(1.0, value) * np.array([-1, 0, 1])))
            for value in center.allowances[b]
        ]
        candidates.append(np.array(list(product(*levels))))
    travelers = regional.valuation_travelers
    rows = {traveler: row for row, traveler in enumerate(travelers)}
    g0 = np.array(
        [
            problem.route_oracles[i][0].expected_value(np.zeros(5), problem.demand_distributions[i])
            for i in travelers
        ]
    )
    g = []
    for b, vectors in enumerate(candidates, 1):
        g.append(
            np.array(
                [
                    [
                        problem.route_oracles[i][b].expected_value(
                            v, problem.demand_distributions[i]
                        )
                        for v in vectors
                    ]
                    for i in travelers
                ]
            )
        )
    choices = [
        (rows[k.traveler], k.assigned_bundle, k.alternative_bundle) for k in regional.active_indices
    ]
    p1, p2, feasible = profile_two_prices(g0, *g, choices, problem.price_upper_bounds)
    cost1 = problem.provision_cost.values(candidates[0])
    cost2 = problem.provision_cost.values(candidates[1])
    demand = regional.bundle_demand
    profit = demand[1] * (p1 - cost1[:, None]) + demand[2] * (p2 - cost2[None, :])
    profit = np.where(feasible, profit, -np.inf)
    row, col = np.unravel_index(np.argmax(profit), profit.shape)
    assert np.isfinite(profit[row, col]), "Grid includes a known feasible allowance vector"
    menu = BundleMenu(
        np.array([0, p1[row, col], p2[row, col]]),
        np.vstack([np.zeros(5), candidates[0][row], candidates[1][col]]),
    )
    displacement = [
        np.max(np.abs(vectors - center.allowances[b]) / np.maximum(1, center.allowances[b]), axis=1)
        for b, vectors in enumerate(candidates, 1)
    ]
    distinct = (displacement[0][:, None] >= radius / 2) | (displacement[1][None, :] >= radius / 2)
    other_profit = np.where(distinct, profit, -np.inf)
    r2, c2 = np.unravel_index(np.argmax(other_profit), other_profit.shape)
    other = None
    if np.isfinite(other_profit[r2, c2]):
        other = BundleMenu(
            np.array([0, p1[r2, c2], p2[r2, c2]]),
            np.vstack([np.zeros(5), candidates[0][r2], candidates[1][c2]]),
        )
    return (
        menu,
        other,
        {
            "relative_radius": radius,
            "grid_points": int(feasible.size),
            "feasible_grid_points": int(feasible.sum()),
            "feasible_noncentral_grid_points": int((feasible & distinct).sum()),
            "allowance_vectors_per_bundle": [len(v) for v in candidates],
            "profiled_profit": float(profit[row, col]),
        },
    )


def affine_lower_and_corner_upper(oracle, allowance, distribution):
    """Bounds derived from feasible dual selections and convexity in mu.

    Fixed envelope selections produce a globally valid affine lower bound in
    allowance, even when they are not the unique maximizers. Chords between
    actual max-of-all-vertices values dominate the convex route value in mu.
    This avoids assuming that an arbitrary boundary subgradient is smooth.
    """
    phi = oracle.for_allowance(allowance)
    intercept, omega = 0.0, np.zeros(oracle.mode_count)
    for j, vertex in enumerate(phi.active_vertex_indices):
        lower = phi.breakpoints[j]
        upper = phi.breakpoints[j + 1] if j + 1 < len(phi.breakpoints) else np.inf
        intercept += oracle.slopes[vertex] * distribution.first_moment(lower, upper)
        omega += oracle.omega[vertex] * distribution.probability(lower, upper)
    knots = np.unique(
        np.r_[
            distribution.lower,
            phi.breakpoints[
                (phi.breakpoints > distribution.lower) & (phi.breakpoints < distribution.upper)
            ],
            distribution.upper,
        ]
    )
    values = np.max(oracle.slopes[:, None] * knots - (oracle.omega @ allowance)[:, None], axis=0)
    upper_value = 0.0
    for left, right, yl, yr in zip(knots[:-1], knots[1:], values[:-1], values[1:], strict=True):
        slope = (yr - yl) / (right - left)
        upper_value += slope * distribution.first_moment(left, right) + (
            yl - slope * left
        ) * distribution.probability(left, right)
    # Numerical padding; these are float bounds, not interval-arithmetic proofs.
    return intercept - 1e-9, -omega, upper_value + 1e-9


def box_upper_bound(regional, center, radius, env):
    """Gurobi LP upper bound valid for the explicit allowance box only.

    For convex G, any convex combination of corner G values overestimates G
    at the same convex combination of corners. Support planes underestimate G.
    Use lower bounds for +G and upper bounds for -G in every retained h<=0.
    """
    started = perf_counter()
    problem = regional.problem
    corners = {}
    for b in (1, 2):
        eps = center.allowances[b]
        lo = np.maximum(0, eps - radius * np.maximum(1, eps))
        hi = eps + radius * np.maximum(1, eps)
        corners[b] = np.array(list(product(*zip(lo, hi, strict=True))))
    lower_needed = {
        (k.traveler, k.assigned_bundle) for k in regional.active_indices if k.assigned_bundle
    }
    upper_needed = {
        (k.traveler, k.alternative_bundle) for k in regional.active_indices if k.alternative_bundle
    }
    data = {}
    for i, b in sorted(lower_needed | upper_needed):
        data[i, b] = [
            affine_lower_and_corner_upper(
                problem.route_oracles[i][b], corner, problem.demand_distributions[i]
            )
            for corner in corners[b]
        ]
    with gp.Model(env=env) as model:
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        prices = model.addMVar(2, lb=0, ub=problem.price_upper_bounds, name="prices")
        eps_vars, lambdas = {}, {}
        for b in (1, 2):
            lambdas[b] = model.addMVar(len(corners[b]), lb=0, name=f"lambda_{b}")
            model.addConstr(lambdas[b].sum() == 1)
            eps_vars[b] = model.addMVar(5, lb=0, name=f"eps_{b}")
            model.addConstr(eps_vars[b] == corners[b].T @ lambdas[b])
        lower_g, upper_g = {}, {}
        for i, b in lower_needed:
            lower_g[i, b] = model.addVar(lb=0, name=f"lower_g_{i}_{b}")
            for intercept, gradient, _ in data[i, b]:
                model.addConstr(lower_g[i, b] >= intercept + gradient @ eps_vars[b])
        for i, b in upper_needed:
            upper_g[i, b] = np.array([value[2] for value in data[i, b]]) @ lambdas[b]
        g0 = {
            i: problem.route_oracles[i][0].expected_value(np.zeros(5), dist)
            for i, dist in enumerate(problem.demand_distributions)
            if i in regional.valuation_travelers
        }
        for index in regional.active_indices:
            i, b, a = index.traveler, index.assigned_bundle, index.alternative_bundle
            cb, ca = (prices[b - 1] if b else 0), (prices[a - 1] if a else 0)
            gb = lower_g[i, b] if b else g0[i]
            ga = upper_g[i, a] if a else g0[i]
            model.addConstr(cb - ca + gb - ga <= 0)
        model.addConstr(prices[0] <= prices[1])
        demand = regional.bundle_demand
        objective = demand[1:] @ prices
        for b in (1, 2):
            objective -= demand[b] * (problem.unit_allowance_cost @ eps_vars[b])
        model.setObjective(objective, gp.GRB.MAXIMIZE)
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL, model.Status
        upper_bound = float(model.ObjVal)
    profit = regional.profit(regional.layout.pack(center))
    assert upper_bound >= profit - 1e-7
    return {
        "relative_radius": radius,
        "upper_bound": upper_bound,
        "upper_bound_minus_original_profit": upper_bound - profit,
        "box_lower_allowances": [corners[b].min(axis=0).tolist() for b in (1, 2)],
        "box_upper_allowances": [corners[b].max(axis=0).tolist() for b in (1, 2)],
        "scope": "original retained NLP restricted to this allowance box; prices unrestricted",
        "seconds": perf_counter() - started,
    }


def candidate_record(regional, full, menu):
    x = regional.layout.pack(menu)
    return {
        "profit": regional.profit(x),
        "restricted_violation": regional.maximum_constraint_violation(x),
        "full_region_violation": full.maximum_constraint_violation(x),
        "menu": record_menu(menu),
    }


def independent_price_check(regional, allowances):
    """HiGHS prices with a fresh explicit integration of every retained G."""
    problem = regional.problem
    values, matrix, rhs = {}, [], []
    for index in regional.active_indices:
        i, b, a = index.traveler, index.assigned_bundle, index.alternative_bundle
        for option in (b, a):
            if (i, option) not in values:
                phi = problem.route_oracles[i][option].for_allowance(allowances[option])
                dist = problem.demand_distributions[i]
                value = 0.0
                for j, lower in enumerate(phi.breakpoints):
                    upper = phi.breakpoints[j + 1] if j + 1 < len(phi.breakpoints) else np.inf
                    value += phi.slopes[j] * dist.first_moment(lower, upper)
                    value += phi.intercepts[j] * dist.probability(lower, upper)
                values[i, option] = value
        line = np.zeros(2)
        if b:
            line[b - 1] += 1
        if a:
            line[a - 1] -= 1
        matrix.append(line)
        rhs.append(values[i, a] - values[i, b])
    matrix.append([1, -1])
    rhs.append(0)
    solved = linprog(
        -regional.bundle_demand[1:],
        A_ub=matrix,
        b_ub=rhs,
        bounds=list(zip(np.zeros(2), problem.price_upper_bounds, strict=True)),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    assert solved.success, solved.message
    profit = -solved.fun - regional.bundle_demand[1:] @ (
        allowances[1:] @ problem.unit_allowance_cost
    )
    return {
        "profit": float(profit),
        "violation": float(max(0, np.max(np.array(matrix) @ solved.x - rhs))),
        "menu": {"prices": [0, *solved.x.tolist()], "allowances": allowances.tolist()},
    }


def independently_check_outliers(problem, trace, report, output):
    checks = []
    for result in report["runs"]:
        if summarize_run(result)["best_gain"] <= 1e-5:
            continue
        call = trace["calls"][result["call"] - 1]
        alpha = np.array(trace["regions"][call["assignment_sha256"]]["assigned_bundles"])
        regional = make_regional(problem, alpha, call["choice_constraints"])
        original = _HELPERS.restore_menu(result["original"]["menu"])
        candidates = [result["price_lp"], *result["grids"], *result["restarts"]]
        best = max(
            (v for v in candidates if v["restricted_violation"] <= 1e-7),
            key=lambda v: v["profit"],
        )
        independent = independent_price_check(regional, np.array(best["menu"]["allowances"]))
        independent["gain"] = independent["profit"] - result["original"]["profit"]
        grid = next(g for g in result["grids"] if g["relative_radius"] == 0.01)
        delta = np.array(grid["menu"]["allowances"]) - original.allowances
        check = {
            "call": call["call"],
            "baseline_profit": result["original"]["profit"],
            "scope": "same retained NLP; direct interval integration and independent HiGHS LP",
            "independent_best": independent,
            "ray_checks": [],
        }
        base_profile = independent_price_check(regional, original.allowances)
        check["baseline_conditionally_optimal_prices"] = base_profile
        for factor in (1, 0.1, 0.01, 0.001, 0.0001):
            eps = original.allowances + factor * delta
            value = independent_price_check(regional, eps)
            value.update(
                {
                    "factor": factor,
                    "gain": value["profit"] - result["original"]["profit"],
                    "gain_over_conditionally_priced_baseline": value["profit"]
                    - base_profile["profit"],
                    "normalized_allowance_displacement": float(
                        np.max(
                            np.abs(eps - original.allowances) / np.maximum(1, original.allowances)
                        )
                    ),
                }
            )
            check["ray_checks"].append(value)
        (output / f"call_{call['call']}_independent_check.json").write_text(
            json.dumps(check, indent=2) + "\n"
        )
        checks.append(check)
    report["independent_outlier_checks"] = checks


def summarize_run(result, feasibility=1e-7):
    """Never count an infeasible restart as an improvement of the same NLP."""
    candidates = [("original", result["original"]), ("price_lp", result["price_lp"])]
    for grid in result["grids"]:
        candidates.append((f"grid_{grid['relative_radius']}", grid))
        if "best_noncentral" in grid:
            candidates.append(
                (f"noncentral_grid_{grid['relative_radius']}", grid["best_noncentral"])
            )
    for restart in result["restarts"]:
        candidates.append((f"restart_{restart['start']}", restart))
    feasible = [(name, v) for name, v in candidates if v["restricted_violation"] <= feasibility]
    name, best = max(feasible, key=lambda pair: pair[1]["profit"])
    full = [(n, v) for n, v in feasible if v["full_region_violation"] <= feasibility]
    original = result["original"]
    row = {
        "call": result["call"],
        "region": result["region"],
        "cg_round": result["cg_round"],
        "is_region_endpoint": result["is_region_endpoint"],
        "retained_choice_constraints": result["retained_choice_constraints"],
        "original_profit": original["profit"],
        "original_restricted_violation": original["restricted_violation"],
        "original_full_violation": original["full_region_violation"],
        "price_gain": result["price_lp"]["profit"] - original["profit"],
        "grid_gain": max([original["profit"]] + [g["profit"] for g in result["grids"]])
        - original["profit"],
        "best_source": name,
        "best_profit": best["profit"],
        "best_gain": best["profit"] - original["profit"],
        "relative_gain": (best["profit"] - original["profit"]) / max(1, abs(original["profit"])),
        "best_restricted_violation": best["restricted_violation"],
        "best_full_violation": best["full_region_violation"],
        "best_full_gain": None,
        "restart_returns_infeasible": sum(
            r["restricted_violation"] > feasibility for r in result["restarts"]
        ),
    }
    if full and original["full_region_violation"] <= feasibility:
        row["best_full_gain"] = max(v["profit"] for _, v in full) - original["profit"]
    for bound in result["box_bounds"]:
        row[f"box_upper_gap_{bound['relative_radius']}"] = bound[
            "upper_bound_minus_original_profit"
        ]
    return row


def audit_internal_retries(problem, trace, selected_calls):
    """Record failed intermediate returns separately from the final NLP returns."""
    retries = []
    for call in trace["calls"]:
        if call["call"] not in selected_calls:
            continue
        for number in call["backend_attempts"][:-1]:
            attempt = trace["backend_attempts"][number - 1]
            alpha = np.array(trace["regions"][call["assignment_sha256"]]["assigned_bundles"])
            regional = make_regional(problem, alpha, call["choice_constraints"])
            full = _HELPERS._HELPERS.regional_problem(problem, alpha)
            retries.append(
                {
                    "attempt": number,
                    "call": call["call"],
                    "source_metrics": attempt["metrics"],
                    **candidate_record(regional, full, _HELPERS.restore_menu(attempt["menu"])),
                }
            )
    return retries


def write_summary(report, output):
    rows = [summarize_run(run) for run in report["runs"]]
    for filename, selected in (
        ("nlp_calls.csv", rows),
        ("region_endpoints.csv", [r for r in rows if r["is_region_endpoint"]]),
    ):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    report["summary"] = {
        "nlp_calls_checked": len(rows),
        "region_endpoints_checked": sum(r["is_region_endpoint"] for r in rows),
        "grid_combinations_checked": sum(
            g["grid_points"] for run in report["runs"] for g in run["grids"]
        ),
        "gain_threshold_counts": {
            str(t): sum(r["best_gain"] > t for r in rows)
            for t in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
        },
        "largest_gain": max(rows, key=lambda r: r["best_gain"]),
        "largest_gain_excluding_worst_call": (
            sorted((r["best_gain"] for r in rows), reverse=True)[1] if len(rows) > 1 else None
        ),
        "maximum_region_endpoint_gain": max(
            (
                r["best_full_gain"]
                for r in rows
                if r["is_region_endpoint"] and r["best_full_gain"] is not None
            ),
            default=None,
        ),
        "restart_returns_infeasible": sum(r["restart_returns_infeasible"] for r in rows),
        "internal_retry_returns_checked": len(report.get("internal_retry_returns", [])),
        "internal_retry_returns_infeasible": sum(
            v["restricted_violation"] > report["feasibility_tolerance"]
            for v in report.get("internal_retry_returns", [])
        ),
        "region_endpoints_with_full_feasible_gain_above_1e_5": sum(
            r["is_region_endpoint"]
            and r["best_full_gain"] is not None
            and r["best_full_gain"] > 1e-5
            for r in rows
        ),
        "total_verification_seconds": sum(run["seconds"] for run in report["runs"]),
    }
    write_markdown_report(report, rows, output)


def write_markdown_report(report, rows, output):
    selection = report["source_selection"]
    summary = report["summary"]
    lines = [
        "# NLP verification along the longest Sioux Falls path",
        "",
        f"Source: start {selection['start']}, {selection['regions']} regions, "
        f"{selection['transitions']} transitions. The path was selected by the maximum "
        "transition count across the same 100 completed source trajectories. "
        "Deterministic replay matched the saved objectives, menus, assignments and "
        "constraint counts.",
        "",
        f"Checked {len(rows)} restricted NLP return points, including "
        f"{summary['region_endpoints_checked']} final regional endpoints. "
        f"The source contains {report['backend_attempts_in_source']} backend attempts; "
        "the per-call rows refer to each combined call's final return. "
        f"An additional {summary['internal_retry_returns_checked']} internal retry returns "
        "were checked for feasibility; "
        f"{summary['internal_retry_returns_infeasible']} were infeasible for their "
        "retained NLP. These are not alternative feasible optima.",
        "",
        "## Method and acceptance",
        "",
        "Each comparison uses the exact original fixed assignment and retained choice "
        "constraints. No reassignment, rounding of dual vertices, or modification of "
        "the production solver is performed. Original and alternative menus are also "
        "checked against every choice constraint in that region. Feasibility threshold: "
        "1e-7. Infeasible restarts are retained in the raw record and excluded from "
        "objective-gain calculations.",
        "",
        "1. Gurobi reoptimizes both prices at the original allowance vector.",
        "2. Exhaust three allowance grids, with r=0.1, 0.03, 0.01. Coordinate e takes "
        "max(0,e-r max(1,e)), e, e+r max(1,e). The two-price LP is optimized at "
        "every combination, using the independently tested exact two-price profiler. "
        "Grid winners are checked with Gurobi and the original oracle constraints.",
        "3. Restart SLSQP from the original return and the best noncentral feasible "
        "grid point, using numerical constraint derivatives (eps=1e-6), maxiter=500, "
        "ftol=1e-11, with recovery and fallback disabled.",
        "4. Solve Gurobi LP outer relaxations in allowance boxes at r=0.1, 0.01, "
        "0.001, 0.0001; prices retain their original domain. For convex G, a convex "
        "combination of corner values is an upper bound at the same combination of "
        "corners. Integrated feasible dual selections yield affine lower bounds. "
        "In each h=C_b-C_a+G_ib-G_ia<=0, replace +G_ib by a lower bound and "
        "-G_ia by the negative upper bound. This enlarges the feasible set, so "
        "maximizing the relaxed profit supplies an objective upper bound in that box.",
        "",
        "Corner upper values integrate chords of the max over all cached dual "
        "vertices at support endpoints and envelope breakpoints. These chords "
        "dominate the convex integrand in trip frequency. Lower constants receive "
        "-1e-9 padding and upper values +1e-9; Gurobi LP feasibility and optimality "
        "tolerances are 1e-9. These are floating-point bounds, not interval proofs.",
        "",
        "## Results",
        "",
        f"Largest observed feasible gain: {summary['largest_gain']['best_gain']:.12g} "
        f"at call {summary['largest_gain']['call']} "
        f"(region {summary['largest_gain']['region']}, CG round "
        f"{summary['largest_gain']['cg_round']}). "
        f"This is {100 * summary['largest_gain']['relative_gain']:.4f}% of its original profit.",
        "",
        f"The largest remaining gain is {summary['largest_gain_excluding_worst_call']}; "
        "the largest gain among full-region-feasible final endpoints is "
        f"{summary['maximum_region_endpoint_gain']}. "
        "A large-gain intermediate call is assessed as its retained NLP, "
        "not as a feasible solution of its complete region.",
        "",
        f"Grid combinations: {summary['grid_combinations_checked']:,}. "
        f"Total verification time: {summary['total_verification_seconds']:.1f} seconds "
        "after cached-problem construction.",
        "",
        "Number of calls with best feasible gain above each absolute threshold:",
        "",
        "| Gain threshold | Calls |",
        "| --- | ---: |",
    ]
    lines += [
        f"| {threshold} | {count} |"
        for threshold, count in summary["gain_threshold_counts"].items()
    ]
    lines += [
        "",
        "The following table includes all checked restricted NLP returns. "
        "An asterisk denotes the last CG call in its region. A large original "
        "full-region violation at an intermediate CG call does not invalidate "
        "its feasibility for the restricted NLP actually solved.",
        "",
        "| Call | Region | Original profit | Best feasible gain | Best source | "
        "Original full-region violation | Best full-region violation |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        marker = "*" if row["is_region_endpoint"] else ""
        lines.append(
            f"| {row['call']}{marker} | {row['region']} | {row['original_profit']:.12f} | "
            f"{row['best_gain']:.6g} | {row['best_source']} | "
            f"{row['original_full_violation']:.4g} | {row['best_full_violation']:.4g} |"
        )
    lines += [
        "",
        "## Scope of the conclusion",
        "",
        "A positive gain is an observed improvement of the same retained NLP, "
        "not by itself a proof that the original return was not locally optimal. "
        "No gain found is numerical evidence, not a local or global certificate. "
        "Full-region improvements are reported separately and require both the "
        "original and alternative points to be full-region feasible. Box upper "
        "bounds apply only to the stated boxes, not the entire original "
        "unbounded allowance domain. SLSQP success is not used as a certificate "
        "of KKT, SOSC or global optimality.",
        "",
        "See nlp_calls.csv for every gain and box bound; region_endpoints.csv "
        "for the final CG calls; verification.json for all original and "
        "alternative menus, grid counts, initial points, constraints and statuses.",
        "For calls with gain above 1e-5, call_N_independent_check.json contains "
        "a separate direct-interval-integration and HiGHS price-LP check, plus "
        "a shrinking sequence of feasible improving allowance perturbations. "
        "These numerical witnesses are not an exact local-optimality theorem.",
        "",
        "## Reproduce",
        "",
        "```sh",
        "PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\",
        "  /opt/homebrew/Caskroom/miniforge/base/envs/maas-bundle/bin/python \\",
        "  scripts/verify-selected-nlps.py --all-calls \\",
        "  --output results/longest_trajectory_nlp_verification_20261027",
        "```",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def run(args):
    trace = json.loads(args.trace.read_text())
    assert trace["complete"] and trace["matches_saved_fields"]
    if args.all_calls:
        selection, _ = _HELPERS.select_longest(Path.cwd())
        assert selection == trace["selection"], "Saved trace is no longer the selected longest path"
        args.calls = [call["call"] for call in trace["calls"]]
    region_ends = {call["region"]: call["call"] for call in trace["calls"]}
    problem = build_sioux_falls_problem(
        oracle_cache_path="results/sioux_falls_five_modes/dual_vertices.npz"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "source_trace": str(args.trace),
        "calls_selected": args.calls,
        "gurobi_version": list(gp.gurobi.version()),
        "scope": "original restricted NLP returns; no reassignment or model normalization",
        "source_selection": trace["selection"],
        "backend_attempts_in_source": len(trace["backend_attempts"]),
        "feasibility_tolerance": 1e-7,
        "runs": [],
        "full_domain_global_certificate": False,
    }
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        for number in args.calls:
            started = perf_counter()
            call = trace["calls"][number - 1]
            alpha = np.array(trace["regions"][call["assignment_sha256"]]["assigned_bundles"])
            regional = make_regional(problem, alpha, call["choice_constraints"])
            full = _HELPERS._HELPERS.regional_problem(problem, alpha)
            original = _HELPERS.restore_menu(call["menu"])
            baseline = candidate_record(regional, full, original)
            priced = price_lp(regional, original.allowances, env)
            assert priced is not None
            result = {
                "call": number,
                "region": call["region"],
                "cg_round": call["cg_round"],
                "is_region_endpoint": region_ends[call["region"]] == number,
                "assignment_sha256": call["assignment_sha256"],
                "choice_constraints": call["choice_constraints"],
                "retained_choice_constraints": len(call["choice_constraints"]),
                "original": baseline,
                "price_lp": candidate_record(regional, full, priced),
                "grids": [],
                "restarts": [],
                "box_bounds": [],
            }
            distinct_starts = []
            for radius in (0.1, 0.03, 0.01):
                menu, other, grid = allowance_grid(regional, original, radius)
                # Independent Gurobi check of the vectorized price LP's winner.
                verified = price_lp(regional, menu.allowances, env)
                assert verified is not None
                exact = candidate_record(regional, full, verified)
                assert abs(exact["profit"] - grid["profiled_profit"]) < 1e-7
                assert exact["restricted_violation"] <= 1e-7
                if other is not None:
                    other = price_lp(regional, other.allowances, env)
                    assert other is not None
                    grid["best_noncentral"] = candidate_record(regional, full, other)
                    distinct_starts.append(other)
                result["grids"].append(grid | exact)
            assert distinct_starts
            best = max(distinct_starts, key=lambda m: regional.profit(regional.layout.pack(m)))
            # Two starts only, with no hidden fallback, recovery or restarts.
            solver = ScipyRegionalSolver(
                options={"maxiter": 500, "ftol": 1e-11},
                check_kkt=False,
                enable_recovery=False,
                enable_fallback=False,
            )
            for label, initial in (("original_return", original), ("best_noncentral_grid", best)):
                solved = solver._solve_once(regional, initial, numerical_constraints=True)
                check = candidate_record(regional, full, solved.menu)
                result["restarts"].append(
                    {
                        "start": label,
                        "numerical_constraint_jacobian": True,
                        "initial": candidate_record(regional, full, initial),
                        "solver_metrics": solved.metrics(),
                        **check,
                    }
                )
            for radius in (0.1, 0.01, 0.001, 0.0001):
                result["box_bounds"].append(box_upper_bound(regional, original, radius, env))
            result["seconds"] = perf_counter() - started
            report["runs"].append(result)
            (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "call": number,
                        "original_profit": baseline["profit"],
                        "price_lp_profit": result["price_lp"]["profit"],
                        "best_grid_profit": max(v["profit"] for v in result["grids"]),
                        "restarts": [
                            {
                                "profit": v["profit"],
                                "violation": v["restricted_violation"],
                                "full_violation": v["full_region_violation"],
                            }
                            for v in result["restarts"]
                        ],
                        "smallest_box_upper_gap": result["box_bounds"][-1][
                            "upper_bound_minus_original_profit"
                        ],
                        "seconds": result["seconds"],
                    }
                ),
                flush=True,
            )
    report["complete"] = True
    report["internal_retry_returns"] = audit_internal_retries(problem, trace, args.calls)
    independently_check_outliers(problem, trace, report, args.output)
    write_summary(report, args.output)
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, nargs="+", default=[4, 28, 41])
    parser.add_argument("--all-calls", action="store_true", help="Audit every call on longest path")
    parser.add_argument(
        "--trace", type=Path, default=Path("results/longest_trajectory_sosc_20261022/replay.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/selected_nlp_verification_20261027")
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
