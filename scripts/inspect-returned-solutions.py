"""Inspect saved menus without changing the production solver or manuscript.

Checks full feasibility, multiplier residuals, optimistic reassignment, branch
stability, and nearby allowance perturbations with an exact price-only LP.
Hessian results describe a selected smooth envelope branch, not a classical
SOSC certificate when switching or derivative instability is detected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from maas_bundle.design.boundary import ChoiceConstraintIndex, CriticalConstraintSet
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.regional import RestrictedRegionalProblem
from maas_bundle.examples.sioux_falls import build_sioux_falls_problem
from maas_bundle.oracle.demand_distribution import (
    TriangularDemandDistribution,
    UniformDemandDistribution,
)


def density(distribution, value):
    if value <= distribution.lower or value >= distribution.upper:
        return 0.0
    if isinstance(distribution, UniformDemandDistribution):
        return 1.0 / (distribution.upper - distribution.lower)
    if isinstance(distribution, TriangularDemandDistribution):
        if value <= distribution.mode:
            return (
                2
                * (value - distribution.lower)
                / (
                    (distribution.upper - distribution.lower)
                    * (distribution.mode - distribution.lower)
                )
            )
        return (
            2
            * (distribution.upper - value)
            / ((distribution.upper - distribution.lower) * (distribution.upper - distribution.mode))
        )
    raise TypeError("This diagnostic supports the actual continuous benchmark distributions only")


def envelope_curvature(oracle, allowance, distribution):
    """Analytic Hessian on the selected envelope branch, plus switching flags.

    At a transition p -> q at tau, the Hessian contribution is
    pdf(tau) * (omega_q - omega_p)(omega_q - omega_p)' / (pi_q - pi_p).
    Parallel maximizing faces or multiple envelope intersections require
    separate nonsmooth analysis and invalidate a blanket smooth-SOSC claim.
    """
    phi = oracle.for_allowance(allowance)
    hessian = np.zeros((oracle.mode_count, oracle.mode_count))
    parallel_switches = 0
    multiple_intersections = 0
    uniform_support_switches = 0
    min_parallel_distance = np.inf
    vertices = oracle.vertices
    intercepts = -oracle.omega @ allowance
    for segment, chosen in enumerate(phi.active_vertex_indices):
        lower = phi.breakpoints[segment]
        upper = phi.breakpoints[segment + 1] if segment + 1 < len(phi.breakpoints) else np.inf
        if distribution.probability(lower, upper) <= 1e-10:
            continue
        difference = oracle.omega - oracle.omega[chosen]
        norm = np.linalg.norm(difference, axis=1)
        parallel = (np.abs(oracle.slopes - oracle.slopes[chosen]) <= 1e-10) & (norm > 1e-10)
        if np.any(parallel):
            distance = np.abs(intercepts[parallel] - intercepts[chosen]) / norm[parallel]
            min_parallel_distance = min(min_parallel_distance, float(np.min(distance)))
            parallel_switches += int(np.any(distance <= 1e-8))
    for segment in range(1, len(phi.breakpoints)):
        tau = phi.breakpoints[segment]
        before, after = phi.active_vertex_indices[segment - 1 : segment + 1]
        delta = oracle.omega[after] - oracle.omega[before]
        hessian += (
            density(distribution, tau)
            * np.outer(delta, delta)
            / (vertices[after, 0] - vertices[before, 0])
        )
        if isinstance(distribution, UniformDemandDistribution):
            uniform_support_switches += int(
                min(abs(tau - distribution.lower), abs(tau - distribution.upper)) <= 1e-8
            )
        if distribution.lower < tau < distribution.upper:
            scores = oracle.slopes * tau + intercepts
            tied = scores >= np.max(scores) - 1e-9
            # Count distinct slopes: parallel-face ties are reported separately.
            tied_slopes = np.unique(np.round(oracle.slopes[tied], 9))
            multiple_intersections += int(len(tied_slopes) > 2)
    return hessian, {
        "parallel_switch_segments_with_positive_probability": parallel_switches,
        "multiple_envelope_intersections_in_support": multiple_intersections,
        "uniform_support_switches": uniform_support_switches,
        "minimum_parallel_switch_distance": (
            float(min_parallel_distance) if np.isfinite(min_parallel_distance) else None
        ),
    }


def regional_problem(problem, assignment, selected=None):
    critical = CriticalConstraintSet(assignment, problem.bundle_count)
    if selected is None:
        selected = tuple(
            ChoiceConstraintIndex(i, int(b), a)
            for i, b in enumerate(assignment)
            for a in range(problem.bundle_count)
            if a != b
        )
    critical.add_all(tuple(selected))
    return RestrictedRegionalProblem(problem, critical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-directions", type=int, default=4)
    parser.add_argument("--reuse-neighborhood", action="store_true")
    args = parser.parse_args()
    previous = {}
    if args.reuse_neighborhood and args.output.exists():
        previous = {case["name"]: case for case in json.loads(args.output.read_text())["cases"]}
    problem = build_sioux_falls_problem(
        oracle_cache_path="results/sioux_falls_five_modes/dual_vertices.npz"
    )
    directory = Path("results/regional_convexification_20261021")
    raw = json.loads((directory / "verification.json").read_text())
    polished = json.loads((directory / "audit_polish.json").read_text())
    candidates = []
    for initial, run in zip(raw["runs"], polished["runs"], strict=True):
        entry = run["original_constraint_polish"]
        menu = BundleMenu(np.array(entry["prices"]), np.array(entry["allowances"]))
        initial_menu = BundleMenu(
            np.array(initial["initial_prices"]), np.array(initial["initial_allowances"])
        )
        alpha = problem.evaluate_assignment(initial_menu, choice_tolerance=1e-9).bundle_indices
        candidates.append((f"recent_fixed_assignment_{run['start']}", menu, alpha))
    best_path = Path("results/sioux_falls_five_modes/multistart_100_20261019/best_candidate.json")
    best = json.loads(best_path.read_text())
    best_menu = BundleMenu(np.array(best["prices"]), np.array(best["allowances"]))
    candidates.append(
        (
            "saved_best_completed_candidate",
            best_menu,
            problem.evaluate_assignment(best_menu, choice_tolerance=1e-9).bundle_indices,
        )
    )
    report = {
        "scope": "four saved menus; numerical diagnostics, not rigorous SOSC certificates",
        "units": "USD/representative weighted person/month",
        "neighborhood_radii_in_relative_infinity_norm": [1e-4, 1e-2],
        "radii_scale": "coordinate scale=max(1, abs(candidate_coordinate))",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for name, menu, alpha in candidates:
        regional = regional_problem(problem, alpha)
        x = regional.layout.pack(menu)
        h = regional.constraint_values(x)
        certificate = regional.first_order_certificate(
            x, stationarity_tolerance=1e-5, active_tolerance=1e-4
        )
        multipliers = np.zeros(len(h))
        multipliers[list(certificate.active_inequality_indices)] = (
            certificate.inequality_multipliers
        )
        mu = np.zeros(len(x))
        mu[list(certificate.active_lower_bound_indices)] = certificate.lower_bound_multipliers
        complementarity = max(float(np.max(np.abs(multipliers * h))), float(np.max(np.abs(mu * x))))
        optimistic = problem.evaluate_assignment(menu, choice_tolerance=1e-9)
        case = {
            "name": name,
            "fixed_assignment_profit": regional.profit(x),
            "optimistic_profit_at_same_menu": optimistic.total_profit,
            "classes_reassigned_at_same_menu": int(
                np.count_nonzero(alpha != optimistic.bundle_indices)
            ),
            "traveler_cost_ties": int(np.count_nonzero(optimistic.has_numerical_cost_ties)),
            "double_ties": int(np.count_nonzero(optimistic.has_double_ties)),
            "primal_violation": certificate.primal_violation,
            "normalized_stationarity": certificate.normalized_stationarity_residual,
            "absolute_complementarity": complementarity,
            "minimum_nonnegative_multiplier": min(
                (*certificate.inequality_multipliers, *certificate.lower_bound_multipliers),
                default=0.0,
            ),
            "active_inequalities": certificate.active_inequality_count,
            "active_lower_bounds": certificate.active_lower_bound_count,
        }
        report["cases"].append(case)

        # Use only active traveler constraints for expensive derivative diagnostics.
        active_choices = [
            regional.active_indices[j]
            for j in certificate.active_inequality_indices
            if j < len(regional.active_indices)
        ]
        compact = regional_problem(problem, alpha, active_choices)
        compact_indices = {index: j for j, index in enumerate(compact.active_indices)}
        lam = np.zeros(compact.constraint_count)
        for j, weight in enumerate(multipliers):
            if j < len(regional.active_indices):
                if regional.active_indices[j] in compact_indices:
                    lam[compact_indices[regional.active_indices[j]]] = weight
            else:
                lam[len(compact.active_indices) + j - len(regional.active_indices)] = weight
        jacobian = compact.constraint_jacobian(x)
        active_rows = np.flatnonzero(h >= -1e-4)
        all_normals = [regional.constraint_jacobian(x)[active_rows]]
        bound_rows = np.flatnonzero(x <= 1e-4)
        if len(bound_rows):
            all_normals.append(-np.eye(len(x))[bound_rows])
        normals = np.vstack(all_normals)
        case["active_constraint_rank"] = int(np.linalg.matrix_rank(normals, tol=1e-8))
        case["active_constraint_count_including_bounds"] = len(normals)

        hessian = np.zeros((len(x), len(x)))  # min(-profit), with linear provision cost
        geometry = []
        curvature_cache = {}
        for index, weight in zip(compact.active_indices, lam, strict=False):
            for bundle, sign in ((index.assigned_bundle, 1), (index.alternative_bundle, -1)):
                if bundle == 0:
                    continue
                key = (index.traveler, bundle)
                if key not in curvature_cache:
                    hg, flags = envelope_curvature(
                        problem.route_oracles[index.traveler][bundle],
                        menu.allowances[bundle],
                        problem.demand_distributions[index.traveler],
                    )
                    curvature_cache[key] = hg
                    geometry.append({"traveler": index.traveler, "bundle": bundle, **flags})
                columns = slice(2 + (bundle - 1) * 5, 2 + bundle * 5)
                hessian[columns, columns] += sign * weight * curvature_cache[key]
        case["active_valuation_geometry"] = geometry

        grad0 = -compact.profit_gradient(x) + jacobian.T @ lam
        finite_difference_checks = []
        for factor in (1e-3, 1e-4, 1e-5):
            numeric_hessian = np.zeros_like(hessian)
            max_jacobian_error = 0.0
            worst_component = None
            for column in range(2, len(x)):
                step = factor * max(1.0, abs(x[column]))
                high = x.copy()
                high[column] += step
                gh = -compact.profit_gradient(high) + compact.constraint_jacobian(high).T @ lam
                vh = compact.constraint_values(high)
                if x[column] > step:
                    low = x.copy()
                    low[column] -= step
                    gl = -compact.profit_gradient(low) + compact.constraint_jacobian(low).T @ lam
                    vl = compact.constraint_values(low)
                    numeric_hessian[:, column] = (gh - gl) / (2 * step)
                    derivative = (vh - vl) / (2 * step)
                else:
                    far = x.copy()
                    far[column] += 2 * step
                    gf = -compact.profit_gradient(far) + compact.constraint_jacobian(far).T @ lam
                    vf = compact.constraint_values(far)
                    numeric_hessian[:, column] = (-3 * grad0 + 4 * gh - gf) / (2 * step)
                    derivative = (-3 * compact.constraint_values(x) + 4 * vh - vf) / (2 * step)
                component_errors = np.abs(derivative - jacobian[:, column])
                error = float(np.max(component_errors))
                if error > max_jacobian_error:
                    max_jacobian_error = error
                    worst_row = int(np.argmax(component_errors))
                    worst_component = {
                        "variable_column": column,
                        "variable_value": float(x[column]),
                        "compact_constraint_row": worst_row,
                        "oracle_derivative": float(jacobian[worst_row, column]),
                        "finite_difference_derivative": float(derivative[worst_row]),
                    }
            finite_difference_checks.append(
                {
                    "relative_step": factor,
                    "active_constraint_jacobian_max_error": max_jacobian_error,
                    "worst_component": worst_component,
                    "lagrangian_hessian_max_error_against_branch_formula": float(
                        np.max(np.abs(numeric_hessian - hessian))
                    ),
                    "lagrangian_hessian_asymmetry": float(
                        np.max(np.abs(numeric_hessian - numeric_hessian.T))
                    ),
                }
            )
        case["finite_difference_checks"] = finite_difference_checks

        # Differentiability requires additivity of directional derivatives even
        # along feasible positive coordinate rays. Persistent failure at smaller
        # steps is stronger evidence than a single selected subgradient mismatch.
        directional_checks = []
        base_values = compact.constraint_values(x)
        for factor in (1e-4, 1e-5, 1e-6):
            steps = factor * np.maximum(1.0, np.abs(x))
            one_values = {}
            for column in range(2, len(x)):
                trial = x.copy()
                trial[column] += steps[column]
                one_values[column] = compact.constraint_values(trial)
            max_error = 0.0
            witness = None
            for first in range(2, len(x)):
                for second in range(first + 1, len(x)):
                    if (first - 2) // 5 != (second - 2) // 5:
                        continue
                    trial = x.copy()
                    trial[first] += steps[first]
                    trial[second] += steps[second]
                    difference = (
                        compact.constraint_values(trial)
                        - one_values[first]
                        - one_values[second]
                        + base_values
                    ) / factor
                    row = int(np.argmax(np.abs(difference)))
                    if abs(difference[row]) > max_error:
                        max_error = float(abs(difference[row]))
                        witness = {
                            "variable_columns": [first, second],
                            "variable_values": [float(x[first]), float(x[second])],
                            "compact_constraint_row": row,
                            "signed_additivity_error": float(difference[row]),
                        }
            directional_checks.append(
                {
                    "relative_step": factor,
                    "maximum_directional_additivity_error": max_error,
                    "witness": witness,
                }
            )
        case["directional_additivity_checks"] = directional_checks

        # The nullspace of positive-multiplier normals CONTAINS the critical
        # cone. Positive definiteness here is conservative sufficient curvature;
        # failure here is inconclusive when weakly active inequalities exist.
        strong_normals = [compact.constraint_jacobian(x)[lam > 1e-7]]
        if np.any(mu > 1e-7):
            strong_normals.append(-np.eye(len(x))[mu > 1e-7])
        strong = np.vstack(strong_normals)
        z = null_space(strong, rcond=1e-8) if len(strong) else np.eye(len(x))
        eig = np.linalg.eigvalsh(z.T @ hessian @ z)
        case["selected_branch_conservative_curvature"] = {
            "nullspace_dimension": z.shape[1],
            "projected_minimization_hessian_eigenvalues": eig.tolist(),
            "is_not_a_classical_sosc_certificate": True,
        }
        save()
        print(
            json.dumps(
                {
                    "case": name,
                    "stage": "first_order_and_branch_checks",
                    "stationarity": case["normalized_stationarity"],
                    "reassigned": case["classes_reassigned_at_same_menu"],
                    "derivative_checks": finite_difference_checks,
                }
            ),
            flush=True,
        )
        if name in previous and "neighborhood_probes" in previous[name]:
            assert (
                abs(previous[name]["fixed_assignment_profit"] - case["fixed_assignment_profit"])
                < 1e-8
            )
            case["neighborhood_probes"] = previous[name]["neighborhood_probes"]
            save()
            continue

        scenarios = [("original_fixed_assignment", alpha)]
        if np.any(alpha != optimistic.bundle_indices):
            scenarios.append(("optimistic_assignment_at_candidate", optimistic.bundle_indices))
        rng = np.random.default_rng(20261022)
        directions = [*np.eye(10), *(-np.eye(10))]
        for _ in range(args.random_directions):
            direction = rng.normal(size=10)
            directions.append(direction / np.max(np.abs(direction)))
        probes = []
        for label, trial_alpha in scenarios:
            trial_regional = regional_problem(problem, trial_alpha)
            demand = trial_regional.bundle_demand
            baseline_profit = trial_regional.profit(x)
            ii, aa = np.where(np.arange(3)[None, :] != trial_alpha[:, None])
            bb = trial_alpha[ii]
            price_matrix = np.zeros((len(ii) + 1, 2))
            rows = np.arange(len(ii))
            mb, ma = bb > 0, aa > 0
            price_matrix[rows[mb], bb[mb] - 1] += 1
            price_matrix[rows[ma], aa[ma] - 1] -= 1
            price_matrix[-1] = [1, -1]
            for radius in (1e-4, 1e-2):
                summary = {
                    "assignment": label,
                    "radius": radius,
                    "attempted": 0,
                    "price_lp_feasible": 0,
                    "original_audit_feasible": 0,
                    "baseline_fixed_profit": baseline_profit,
                    "best_fixed_improvement": 0.0,
                    "best_optimistic_improvement": 0.0,
                }
                for direction in directions:
                    allowances = menu.allowances.copy()
                    allowances[1:] = np.maximum(
                        0.0,
                        allowances[1:]
                        + (radius * np.maximum(1.0, x[2:]) * direction).reshape(2, 5),
                    )
                    if np.array_equal(allowances, menu.allowances):
                        continue
                    summary["attempted"] += 1
                    valuation = problem.evaluate_valuations(BundleMenu(menu.prices, allowances))
                    g = valuation.route_values
                    rhs = np.concatenate((g[ii, aa] - g[ii, bb], [0.0]))
                    price_radius = radius * np.maximum(1.0, menu.prices[1:])
                    bounds = list(
                        zip(
                            np.maximum(0, menu.prices[1:] - price_radius),
                            menu.prices[1:] + price_radius,
                            strict=True,
                        )
                    )
                    lp = linprog(
                        -demand[1:],
                        A_ub=price_matrix,
                        b_ub=rhs,
                        bounds=bounds,
                        method="highs",
                        options={"primal_feasibility_tolerance": 1e-9},
                    )
                    if not lp.success:
                        continue
                    summary["price_lp_feasible"] += 1
                    prices = np.concatenate(([0.0], np.maximum(lp.x, 0.0)))
                    violation = float(np.max(prices[bb] - prices[aa] + g[ii, bb] - g[ii, aa]))
                    if violation > 1e-8 or np.any(np.diff(prices) < -1e-9):
                        continue
                    summary["original_audit_feasible"] += 1
                    candidate = BundleMenu(prices, allowances)
                    profit = float(demand @ problem.bundle_margins(candidate))
                    true_profit = problem.evaluate_assignment(
                        candidate, choice_tolerance=1e-9
                    ).total_profit
                    improvement = profit - baseline_profit
                    true_improvement = true_profit - optimistic.total_profit
                    if improvement > summary["best_fixed_improvement"]:
                        summary["best_fixed_improvement"] = improvement
                        summary["best_fixed_witness"] = {
                            "prices": prices.tolist(),
                            "allowances": allowances.tolist(),
                            "profit": profit,
                            "original_violation": max(0.0, violation),
                            "optimistic_profit": true_profit,
                        }
                    summary["best_optimistic_improvement"] = max(
                        summary["best_optimistic_improvement"], true_improvement
                    )
                probes.append(summary)
                print(
                    json.dumps(
                        {
                            "case": name,
                            "stage": "neighborhood",
                            **{k: v for k, v in summary.items() if k != "best_fixed_witness"},
                        }
                    ),
                    flush=True,
                )
        case["neighborhood_probes"] = probes
        save()
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"])}), flush=True)


if __name__ == "__main__":
    main()
