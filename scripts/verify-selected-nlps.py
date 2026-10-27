"""Small independent checks of three saved restricted NLPs.

Profile prices exactly, exhaust a small allowance grid, restart SLSQP with
numerical constraint derivatives, and bound the objective in explicit boxes
using a convexity-based LP relaxation. No global certificate for the unbounded
original NLP is claimed. The production model and saved menus are not changed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import gurobipy as gp
import numpy as np

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


def run(args):
    trace = json.loads(args.trace.read_text())
    assert trace["complete"] and trace["matches_saved_fields"]
    problem = build_sioux_falls_problem(
        oracle_cache_path="results/sioux_falls_five_modes/dual_vertices.npz"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "source_trace": str(args.trace),
        "calls_selected": args.calls,
        "gurobi_version": list(gp.gurobi.version()),
        "scope": "three original restricted NLPs; no reassignment or model normalization",
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
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, nargs="+", default=[4, 28, 41])
    parser.add_argument(
        "--trace", type=Path, default=Path("results/longest_trajectory_sosc_20261022/replay.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/selected_nlp_verification_20261027")
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
