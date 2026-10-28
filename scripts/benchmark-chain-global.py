"""Compare one-shot SLSQP with full-domain spatial B&B on fixed saved NLPs.

This is a custom spatial B&B using Gurobi node LPs, not Gurobi's native
nonlinear solver. Root bounds follow from original constraints and incumbent
dominance, not arbitrary allowance caps. A time limit never counts as a global
certificate. Numerical bounds use explicit padding, not interval arithmetic.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import gurobipy as gp
import numpy as np

from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.regional import ScipyRegionalSolver

_SPEC = importlib.util.spec_from_file_location(
    "verification_helpers", Path(__file__).with_name("verify-selected-nlps.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_H = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_H)
PAD = 1e-8


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def lp_dual_upper_bound(model):
    """Use a sign-feasible LP dual and correct all reduced-cost residuals.

    For max c'x, pi'Ax <= pi'b. With finite l<=x<=u, the remaining
    r'x=(c-A'pi)'x is bounded by sum max(r_j*l_j,r_j*u_j). This remains an
    upper bound when solver dual/reduced-cost residuals are nonzero.
    """
    variables, constraints = model.getVars(), model.getConstrs()
    lower = np.array(model.getAttr("LB", variables), dtype=np.longdouble)
    upper = np.array(model.getAttr("UB", variables), dtype=np.longdouble)
    if np.any(np.abs(lower) >= gp.GRB.INFINITY / 2) or np.any(np.abs(upper) >= gp.GRB.INFINITY / 2):
        raise ValueError("Residual-corrected dual bound requires finite LP variable bounds")
    pi = np.array(model.getAttr("Pi", constraints), dtype=np.longdouble)
    senses = np.array(model.getAttr("Sense", constraints))
    pi[senses == "<"] = np.maximum(pi[senses == "<"], 0)
    pi[senses == ">"] = np.minimum(pi[senses == ">"], 0)
    residual = np.array(model.getAttr("Obj", variables), dtype=np.longdouble)
    residual -= model.getA().astype(np.longdouble).T @ pi
    bound = np.longdouble(model.ObjCon) + pi @ np.array(
        model.getAttr("RHS", constraints), dtype=np.longdouble
    )
    bound += np.maximum(residual * lower, residual * upper).sum()
    return max(float(bound), float(model.ObjVal)) + PAD


def price_caps(regional):
    """h_iba<=0 implies C_b<=C_a+G_ia(0), since 0<=G_ib<=G_ib(0)."""
    p = regional.problem
    g0 = {
        i: p.route_oracles[i][0].expected_value(np.zeros(p.mode_count), p.demand_distributions[i])
        for i in regional.valuation_travelers
    }
    caps = np.r_[0.0, np.full(p.bundle_count - 1, np.inf)]
    for _ in range(p.bundle_count):
        for k in regional.active_indices:
            assert (
                p.route_oracles[k.traveler][k.alternative_bundle] is p.route_oracles[k.traveler][0]
            )
            caps[k.assigned_bundle] = min(
                caps[k.assigned_bundle], caps[k.alternative_bundle] + g0[k.traveler]
            )
        for b in range(1, p.bundle_count - 1):
            caps[b] = min(caps[b], caps[b + 1])
    caps[1:] = np.minimum(caps[1:] + PAD, p.price_upper_bounds)
    if not np.all(np.isfinite(caps)):
        raise ValueError("No finite price bound derived; do not silently truncate the NLP")
    return caps, g0


class SpatialSearch:
    def __init__(self, regional, env, seed_menus, *, seconds, gap):
        self.regional, self.p, self.env = regional, regional.problem, env
        self.seconds, self.target_gap = seconds, gap
        self.m = self.p.mode_count
        assert self.p.bundle_count == 3
        self.caps, self.g0 = price_caps(regional)
        self.travelers = regional.valuation_travelers
        self.rows = {i: j for j, i in enumerate(self.travelers)}
        self.choices = [
            (self.rows[k.traveler], k.assigned_bundle, k.alternative_bundle)
            for k in regional.active_indices
        ]
        self.lower_keys = sorted(
            {(k.traveler, k.assigned_bundle) for k in regional.active_indices if k.assigned_bundle}
        )
        self.upper_keys = sorted(
            {
                (k.traveler, k.alternative_bundle)
                for k in regional.active_indices
                if k.alternative_bundle
            }
        )
        self.keys = sorted(set(self.lower_keys) | set(self.upper_keys))
        self.cache = {}
        self.count_evaluations = 0
        self.seed_planes = {key: [] for key in self.lower_keys}
        self.cost_coefficients = regional.bundle_demand[1:, None] * self.p.unit_allowance_cost
        if not np.all(self.cost_coefficients > 0):
            raise ValueError("Benchmark requires positive demand and unit cost in both bundles")
        self.best_value = 0.0
        self.best_menu = BundleMenu(np.zeros(3), np.zeros((3, self.m)))
        zero_x = regional.layout.pack(self.best_menu)
        assert regional.maximum_constraint_violation(zero_x) <= 1e-7
        for seed in [self.best_menu, *seed_menus]:
            self.profile(seed.allowances[1:])
            for i, b in self.lower_keys:
                datum = self.evaluate(i, b, seed.allowances[b])
                self.seed_planes[i, b].append(datum[:2])
        self.minimum_g = {key: 0.0 for key in self.keys}
        # Bound propagation over the incumbent-dominating set. At its upper
        # allowance corner, monotonic G is a lower bound throughout that box.
        # This strengthens C_b-C_a <= G_ia(0)-min G_ib, then the cost budget.
        for _ in range(4):
            budget = regional.bundle_demand @ self.caps - self.best_value + PAD
            upper_eps = np.minimum(budget / self.cost_coefficients, self.p.allowance_upper_bounds)
            for key in self.keys:
                self.minimum_g[key] = max(
                    self.minimum_g[key], self.evaluate(*key, upper_eps[key[1] - 1])[3] - PAD
                )
            new_caps = self.caps.copy()
            for _ in range(self.p.bundle_count):
                for k in regional.active_indices:
                    if not k.assigned_bundle:
                        continue
                    low = self.minimum_g[k.traveler, k.assigned_bundle]
                    new_caps[k.assigned_bundle] = min(
                        new_caps[k.assigned_bundle],
                        new_caps[k.alternative_bundle] + self.g0[k.traveler] - low + PAD,
                    )
                new_caps[1] = min(new_caps[1], new_caps[2])
            self.caps = new_caps
        budget = regional.bundle_demand @ self.caps - self.best_value + PAD
        self.root_lower = np.zeros((2, self.m))
        self.root_upper = np.minimum(budget / self.cost_coefficients, self.p.allowance_upper_bounds)

    def evaluate(self, i, b, allowance):
        allowance = np.asarray(allowance, dtype=float)
        key = i, b, allowance.tobytes()
        if key not in self.cache:
            c, gradient, upper = _H.affine_lower_and_corner_upper(
                self.p.route_oracles[i][b], allowance, self.p.demand_distributions[i]
            )
            # Recover the oracle value from the integrated fixed dual selection.
            value = c + gradient @ allowance + 1e-9
            self.cache[key] = c, gradient, upper, float(value)
            self.count_evaluations += 1
        return self.cache[key]

    def profile(self, eps):
        g = [np.zeros((len(self.travelers), 1)) for _ in (1, 2)]
        for j, i in enumerate(self.travelers):
            for b in (1, 2):
                g[b - 1][j, 0] = self.evaluate(i, b, eps[b - 1])[3]
        p1, p2, feasible = _H.profile_two_prices(
            np.array([self.g0[i] for i in self.travelers]), *g, self.choices, self.caps[1:]
        )
        if not feasible[0, 0]:
            return
        menu = BundleMenu(
            np.maximum(0, np.array([0, p1[0, 0], p2[0, 0]])), np.vstack([np.zeros(self.m), eps])
        )
        x = self.regional.layout.pack(menu)
        if self.regional.maximum_constraint_violation(x) <= 1e-7:
            value = self.regional.profit(x)
            if value > self.best_value:
                self.best_value, self.best_menu = value, menu

    def tighten(self, lo, hi):
        budget = self.regional.bundle_demand @ self.caps - self.best_value + PAD
        left = budget - np.sum(self.cost_coefficients * lo)
        if left < -PAD:
            return None
        upper = np.minimum(hi, lo + max(0, left) / self.cost_coefficients)
        return None if np.any(upper < lo - PAD) else np.maximum(upper, lo)

    def solve_box(self, lo, hi, inherited_upper, deadline):
        """LP outer bound, optionally refined by valid convex support cuts."""
        corners = {
            b: np.array(list(product(*zip(lo[b - 1], hi[b - 1], strict=True)))) for b in (1, 2)
        }
        data = {key: [self.evaluate(*key, v) for v in corners[key[1]]] for key in self.keys}
        with gp.Model(env=self.env) as model:
            model.Params.Threads = 1
            model.Params.FeasibilityTol = 1e-9
            model.Params.OptimalityTol = 1e-9
            prices = model.addMVar(2, lb=0, ub=self.caps[1:])
            eps = {b: model.addMVar(self.m, lb=lo[b - 1], ub=hi[b - 1]) for b in (1, 2)}
            lam = {b: model.addMVar(len(corners[b]), lb=0, ub=1) for b in (1, 2)}
            for b in (1, 2):
                model.addConstr(lam[b].sum() == 1)
                model.addConstr(eps[b] == corners[b].T @ lam[b])
            lower = {}
            for key in self.lower_keys:
                i, b = key
                lower[key] = model.addVar(lb=self.minimum_g[key], ub=self.g0[i] + PAD)
                for c, grad in self.seed_planes[key] + [d[:2] for d in data[key]]:
                    model.addConstr(lower[key] >= c + grad @ eps[b])
            upper = {
                key: np.array([d[2] for d in data[key]]) @ lam[key[1]] for key in self.upper_keys
            }
            choice_rows = []
            for k in self.regional.active_indices:
                i, b, a = k.traveler, k.assigned_bundle, k.alternative_bundle
                cb, ca = prices[b - 1] if b else 0, prices[a - 1] if a else 0
                gb = lower[i, b] if b else self.g0[i]
                ga = upper[i, a] if a else self.g0[i]
                choice_rows.append(model.addConstr(cb - ca + gb - ga <= 0))
            model.addConstr(prices[0] <= prices[1])
            objective = self.regional.bundle_demand[1:] @ prices
            for b in (1, 2):
                objective -= self.cost_coefficients[b - 1] @ eps[b]
            model.setObjective(objective, gp.GRB.MAXIMIZE)
            bound = inherited_upper
            point, branch_bundle, branch_traveler = 0.5 * (lo + hi), 1, self.travelers[0]
            for _ in range(6):
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    break
                model.Params.TimeLimit = max(0.001, remaining)
                model.optimize()
                if model.Status == gp.GRB.INFEASIBLE:
                    return -np.inf, point, branch_bundle, branch_traveler
                if model.Status != gp.GRB.OPTIMAL:
                    # No primal LP objective is substituted for an unresolved dual bound.
                    break
                bound = min(bound, lp_dual_upper_bound(model))
                point = np.array([eps[b].X for b in (1, 2)])
                point = np.clip(point, lo, hi)
                self.profile(point)
                if bound <= self.best_value + self.target_gap:
                    break
                upper_weights = {key: 0.0 for key in self.upper_keys}
                for k, constraint in zip(self.regional.active_indices, choice_rows, strict=True):
                    if k.alternative_bundle:
                        upper_weights[k.traveler, k.alternative_bundle] += max(
                            0, float(constraint.Pi)
                        )
                worst_upper, worst_lower, worst_score = 0.0, 0.0, -1.0
                for i, b in self.upper_keys:
                    exact = self.evaluate(i, b, point[b - 1])[3]
                    error = float(upper[i, b].getValue()) - exact
                    worst_upper = max(worst_upper, error)
                    score = max(0, error) * (upper_weights[i, b] + 1e-8)
                    if score > worst_score:
                        worst_score, branch_bundle, branch_traveler = score, b, i
                for i, b in self.lower_keys:
                    c, grad, _, exact = self.evaluate(i, b, point[b - 1])
                    worst_lower = max(worst_lower, exact - lower[i, b].X)
                    model.addConstr(lower[i, b] >= c + grad @ eps[b])
                if worst_lower < max(1e-6, 0.05 * worst_upper):
                    break
            return bound, point, branch_bundle, branch_traveler

    def run(self):
        started = perf_counter()
        deadline = started + self.seconds
        root_gross = float(self.regional.bundle_demand @ self.caps)
        heap = [(-root_gross, 0, self.root_lower, self.root_upper)]
        counter, nodes, retired = 0, 0, self.best_value
        milestones, history = {}, []
        status = "time_limit"
        while heap and perf_counter() < deadline:
            neg_upper, _, lo, hi = heapq.heappop(heap)
            inherited = -neg_upper
            if inherited <= self.best_value + self.target_gap:
                retired = max(retired, inherited)
                continue
            tightened = self.tighten(lo, hi)
            if tightened is None:
                retired = max(retired, self.best_value + PAD)
                continue
            hi = tightened
            bound, point, bundle, traveler = self.solve_box(lo, hi, inherited, deadline)
            nodes += 1
            if np.isneginf(bound):
                pass
            elif bound <= self.best_value + self.target_gap:
                retired = max(retired, bound)
            else:
                widths = hi[bundle - 1] - lo[bundle - 1]
                grads = np.array(
                    [
                        self.evaluate(traveler, bundle, v)[1]
                        for v in product(*zip(lo[bundle - 1], hi[bundle - 1], strict=True))
                    ]
                )
                variation = np.ptp(grads, axis=0)
                fraction = np.divide(
                    point[bundle - 1] - lo[bundle - 1],
                    widths,
                    out=np.zeros_like(widths),
                    where=widths > 0,
                )
                scores = (
                    widths
                    * np.maximum(variation, 1e-8)
                    * np.maximum(fraction * (1 - fraction), 1e-4)
                )
                column = int(np.argmax(scores))
                if widths[column] <= 1e-10:
                    heapq.heappush(heap, (-bound, counter + 1, lo, hi))
                    status = "numerically_unresolved"
                    break
                split = float(
                    np.clip(
                        point[bundle - 1, column],
                        lo[bundle - 1, column] + 0.01 * widths[column],
                        hi[bundle - 1, column] - 0.01 * widths[column],
                    )
                )
                for side in (0, 1):
                    child_lo, child_hi = lo.copy(), hi.copy()
                    if side:
                        child_lo[bundle - 1, column] = split
                    else:
                        child_hi[bundle - 1, column] = split
                    counter += 1
                    heapq.heappush(heap, (-bound, counter, child_lo, child_hi))
            global_upper = max(self.best_value, retired, -heap[0][0] if heap else -np.inf)
            gap = global_upper - self.best_value
            elapsed = perf_counter() - started
            for tolerance in (1e-2, 1e-3, 1e-4, 1e-5):
                if gap <= tolerance and str(tolerance) not in milestones:
                    milestones[str(tolerance)] = elapsed
            if not history or elapsed - history[-1]["seconds"] >= 1:
                history.append(
                    {
                        "seconds": elapsed,
                        "lower": self.best_value,
                        "upper": global_upper,
                        "gap": gap,
                        "nodes": nodes,
                    }
                )
            if gap <= self.target_gap:
                status = "gap_closed"
                break
        else:
            if not heap:
                status = "gap_closed"
        upper = max(self.best_value, retired, -heap[0][0] if heap else -np.inf)
        return {
            "status": status,
            "seconds": perf_counter() - started,
            "lower_bound": self.best_value,
            "upper_bound": upper,
            "absolute_gap": upper - self.best_value,
            "relative_gap": (upper - self.best_value) / max(1, abs(self.best_value)),
            "target_absolute_gap": self.target_gap,
            "certified_to_requested_gap": upper - self.best_value <= self.target_gap,
            "nodes": nodes,
            "open_nodes": len(heap),
            "oracle_evaluations": self.count_evaluations,
            "root_price_upper_bounds": self.caps.tolist(),
            "root_allowance_upper_bounds": self.root_upper.tolist(),
            "milestone_seconds": milestones,
            "progress": history,
            "best_menu": _H.record_menu(self.best_menu),
            "best_retained_violation": self.regional.maximum_constraint_violation(
                self.regional.layout.pack(self.best_menu)
            ),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, nargs="+", default=list(range(1, 42)))
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--gap", type=float, default=1e-4)
    parser.add_argument(
        "--trace", type=Path, default=Path("results/longest_trajectory_sosc_20261022/replay.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/longest_chain_global_timing_20261027")
    )
    parser.add_argument("--phase", choices=["local", "global", "all"], default="all")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    assert trace["complete"] and trace["matches_saved_fields"]
    problem = _H.build_sioux_falls_problem(
        oracle_cache_path="results/sioux_falls_five_modes/dual_vertices.npz"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    local_path = args.output / "single_slsqp.json"
    if args.phase in ("local", "all"):
        local = {
            "calls": [],
            "complete": False,
            "scope": "fixed original NLP inputs; one SLSQP call, no recovery/restart/fallback",
        }
        for number in args.calls:
            call = trace["calls"][number - 1]
            alpha = np.array(trace["regions"][call["assignment_sha256"]]["assigned_bundles"])
            regional = _H.make_regional(problem, alpha, call["choice_constraints"])
            initial = _H._HELPERS.restore_menu(call["initial_menu"])
            solver = ScipyRegionalSolver(
                options={"maxiter": 2000, "ftol": 1e-9},
                check_kkt=False,
                enable_recovery=False,
                enable_fallback=False,
            )
            start = perf_counter()
            solved = solver.solve(regional, initial)
            local["calls"].append(
                {
                    "call": number,
                    "region": call["region"],
                    "wall_seconds": perf_counter() - start,
                    "metrics": solved.metrics(),
                    "menu": _H.record_menu(solved.menu),
                }
            )
            save(local_path, local)
        local["complete"] = True
        local["total_wall_seconds"] = sum(r["wall_seconds"] for r in local["calls"])
        local["total_backend_seconds"] = sum(r["metrics"]["time_seconds"] for r in local["calls"])
        save(local_path, local)
    if args.phase == "local":
        return
    local = json.loads(local_path.read_text())
    assert local["complete"]
    by_call = {r["call"]: r for r in local["calls"]}
    report = {
        "implementation": "custom spatial B&B with Gurobi LP relaxations",
        "native_gurobi_spatial_bnb": False,
        "target_absolute_gap": args.gap,
        "node_search_time_limit_per_call": args.seconds,
        "source_trace": str(args.trace),
        "calls": [],
        "complete": False,
    }
    path = args.output / "global_timing.json"
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        for number in args.calls:
            call = trace["calls"][number - 1]
            alpha = np.array(trace["regions"][call["assignment_sha256"]]["assigned_bundles"])
            regional = _H.make_regional(problem, alpha, call["choice_constraints"])
            initial = _H._HELPERS.restore_menu(call["initial_menu"])
            one_shot = _H._HELPERS.restore_menu(by_call[number]["menu"])
            started = perf_counter()
            search = SpatialSearch(
                regional, env, [initial, one_shot], seconds=args.seconds, gap=args.gap
            )
            setup = perf_counter() - started
            result = search.run()
            result.update(
                {
                    "call": number,
                    "region": call["region"],
                    "setup_seconds": setup,
                    "slsqp_seconds": by_call[number]["wall_seconds"],
                    "global_total_seconds_including_slsqp": by_call[number]["wall_seconds"]
                    + setup
                    + result["seconds"],
                }
            )
            report["calls"].append(result)
            save(path, report)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "call",
                            "status",
                            "seconds",
                            "lower_bound",
                            "upper_bound",
                            "absolute_gap",
                            "nodes",
                            "certified_to_requested_gap",
                        )
                    }
                ),
                flush=True,
            )
    report["complete"] = True
    report["certified_count"] = sum(r["certified_to_requested_gap"] for r in report["calls"])
    report["total_global_seconds"] = sum(
        r["global_total_seconds_including_slsqp"] for r in report["calls"]
    )
    report["total_local_seconds"] = sum(r["slsqp_seconds"] for r in report["calls"])
    save(path, report)


if __name__ == "__main__":
    main()
