"""Replay and audit every NLP endpoint on the saved longest regional path.

The original selection/acceptance policy is preserved during replay. Diagnostics
run afterwards and distinguish restricted-NLP feasibility from full-region
feasibility. Float diagnostics are not interval-certified mathematical proofs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import null_space

from maas_bundle.design.boundary import ChoiceConstraintIndex, CriticalConstraintSet
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import solve_primal
from maas_bundle.design.regional import RestrictedRegionalProblem, ScipyRegionalSolver
from maas_bundle.examples.sioux_falls import build_sioux_falls_problem

_HELPER_PATH = Path(__file__).with_name("inspect-returned-solutions.py")
_SPEC = importlib.util.spec_from_file_location("returned_solution_helpers", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


def signature(assignment):
    return hashlib.sha256(np.asarray(assignment, dtype="<i8").tobytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def menu_record(menu):
    return {"prices": menu.prices.tolist(), "allowances": menu.allowances.tolist()}


def restore_menu(record):
    return BundleMenu(np.array(record["prices"]), np.array(record["allowances"]))


def select_longest(root):
    folder = root / "results/sioux_falls_five_modes/multistart_100_20261019"
    summary_path = folder / "regional_workload_summary.json"
    summary = json.loads(summary_path.read_text())
    all_paths = []
    for number, relative in summary["sources_by_start"].items():
        record = json.loads((root / relative).read_text())
        assert record["valid"]
        all_paths.append((int(number), relative, record))
    start, relative, record = max(
        all_paths, key=lambda item: item[2]["candidate_metrics"]["assignment_changes"]
    )
    assert start == summary["selected_trajectory"]["start"]
    return {
        "start": start,
        "source": relative,
        "selection": "maximum assignment_changes among all 100 completed trajectories",
        "regions": len(record["candidate_region_history"]),
        "transitions": record["candidate_metrics"]["assignment_changes"],
        "restricted_calls": len(record["nlp_attempts"]),
        "backend_attempts": sum(item["attempts"] for item in record["nlp_attempts"]),
    }, record


def replay(problem, selection, source, output):
    path = output / "replay.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("complete"):
            price_error = max(
                float(
                    np.max(np.abs(np.array(current["menu"]["prices"]) - np.array(saved["prices"])))
                )
                for current, saved in zip(
                    cached["replayed_region_history"],
                    source["candidate_region_history"],
                    strict=True,
                )
            )
            cached["comparison"]["maximum_saved_region_price_difference"] = price_error
            cached["matches_saved_fields"] = cached["matches_saved_fields"] and price_error <= 1e-8
            save_json(path, cached)
            return cached
        raise RuntimeError("Incomplete replay exists; use a different output directory")
    settings = source["settings"]
    assert not settings["solver_fallback"], "This source path uses SLSQP only"
    region_numbers = {
        row["assumed_assignment_sha256"]: row["region"]
        for row in source["candidate_region_history"]
    }
    trace = {
        "selection": selection,
        "source_settings": settings,
        "scope": "reconstructed intermediate endpoints; checked against saved fields",
        "regions": {},
        "calls": [],
        "backend_attempts": [],
        "complete": False,
    }
    context = {"current": None}
    started = perf_counter()

    class RecordingBackend(ScipyRegionalSolver):
        def _solve_once(self, regional, initial, *, numerical_constraints=False):
            result = super()._solve_once(
                regional, initial, numerical_constraints=numerical_constraints
            )
            call = context["current"]
            number = len(trace["backend_attempts"]) + 1
            trace["backend_attempts"].append(
                {
                    "attempt": number,
                    "call": call["call"],
                    "region": call["region"],
                    "cg_round": call["cg_round"],
                    "numerical_constraint_jacobian": numerical_constraints,
                    "initial_menu": menu_record(initial),
                    "menu": menu_record(result.menu),
                    "metrics": result.metrics(),
                }
            )
            call["backend_attempts"].append(number)
            save_json(path, trace)
            return result

    backend = RecordingBackend(
        options={"maxiter": settings["nlp_max_iterations"], "ftol": 1e-9},
        enable_recovery=settings["recovery"],
        check_kkt=settings["check_kkt"],
        slsqp_restarts=settings["slsqp_restarts"],
        enable_fallback=False,
        numerical_constraint_recovery=settings["numerical_constraint_recovery"],
    )
    rounds = Counter()

    class RecordingSolver:
        name = backend.name
        feasibility_tolerance = backend.feasibility_tolerance
        check_kkt = backend.check_kkt
        optimality_tolerance = backend.optimality_tolerance
        active_set_tolerance = backend.active_set_tolerance
        objective_tolerance = backend.objective_tolerance

        def solve(self, regional, initial):
            key = signature(regional.assigned_bundles)
            number = region_numbers.get(key)
            rounds[key] += 1
            if key not in trace["regions"]:
                trace["regions"][key] = {
                    "region": number,
                    "assigned_bundles": regional.assigned_bundles.tolist(),
                }
            call = {
                "call": len(trace["calls"]) + 1,
                "region": number,
                "assignment_sha256": key,
                "cg_round": rounds[key],
                "choice_constraints": [
                    [index.traveler, index.assigned_bundle, index.alternative_bundle]
                    for index in regional.active_indices
                ],
                "initial_menu": menu_record(initial),
                "backend_attempts": [],
            }
            trace["calls"].append(call)
            context["current"] = call
            result = backend.solve(regional, initial)
            call.update(menu=menu_record(result.menu), metrics=result.metrics())
            save_json(path, trace)
            print(
                json.dumps(
                    {
                        "phase": "replay",
                        "call": call["call"],
                        "region": number,
                        "cg_round": call["cg_round"],
                        "status": result.status.value,
                        "backend_attempts": len(call["backend_attempts"]),
                    }
                ),
                flush=True,
            )
            return result

    initial = BundleMenu(np.array(source["initial_prices"]), np.array(source["initial_allowances"]))
    assert (
        signature(problem.evaluate_assignment(initial).bundle_indices)
        == source["initial_assignment_sha256"]
    )
    diagnostics = solve_primal(
        problem, initial, RecordingSolver(), max_region_iterations=settings["max_regions"]
    )
    candidate = diagnostics.candidate
    history = []
    for step in candidate.iterations:
        history.append(
            {
                "region": step.number,
                "assignment_sha256": signature(step.assumed_assignment),
                "cg_rounds": step.fixed_assignment_result.iteration_count,
                "regional_profit": step.regional_objective,
                "true_profit": step.true_profit,
                "menu": menu_record(step.fixed_assignment_result.menu),
            }
        )
    objective_differences = [
        abs(call["metrics"]["objective"] - saved["objective"])
        for call, saved in zip(trace["calls"], source["nlp_attempts"], strict=False)
    ]
    endpoint_differences = [
        float(
            np.max(np.abs(np.array(current["menu"]["allowances"]) - np.array(saved["allowances"])))
        )
        for current, saved in zip(history, source["candidate_region_history"], strict=False)
    ]
    price_differences = [
        float(np.max(np.abs(np.array(current["menu"]["prices"]) - np.array(saved["prices"]))))
        for current, saved in zip(history, source["candidate_region_history"], strict=False)
    ]
    trace.update(
        {
            "complete": True,
            "replay_elapsed_seconds": perf_counter() - started,
            "diagnostics": diagnostics.metrics(),
            "replayed_region_history": history,
            "comparison": {
                "same_call_count": len(trace["calls"]) == len(source["nlp_attempts"]),
                "same_backend_attempt_count": len(trace["backend_attempts"])
                == selection["backend_attempts"],
                "same_region_assignment_hashes": [h["assignment_sha256"] for h in history]
                == [h["assumed_assignment_sha256"] for h in source["candidate_region_history"]],
                "maximum_saved_call_objective_difference": max(objective_differences, default=0),
                "maximum_saved_region_allowance_difference": max(endpoint_differences, default=0),
                "maximum_saved_region_price_difference": max(price_differences, default=0),
                "same_constraint_counts": len(trace["calls"]) == len(source["nlp_attempts"])
                and all(
                    len(call["choice_constraints"]) == saved["critical_constraints"]
                    for call, saved in zip(trace["calls"], source["nlp_attempts"], strict=False)
                ),
            },
        }
    )
    comparison = trace["comparison"]
    trace["matches_saved_fields"] = (
        comparison["same_call_count"]
        and comparison["same_backend_attempt_count"]
        and comparison["same_region_assignment_hashes"]
        and comparison["same_constraint_counts"]
        and comparison["maximum_saved_call_objective_difference"] <= 1e-8
        and comparison["maximum_saved_region_allowance_difference"] <= 1e-7
        and comparison["maximum_saved_region_price_difference"] <= 1e-8
    )
    save_json(path, trace)
    return trace


def curvature_upper_space(hessian, jacobian, multipliers, lower_multipliers, tolerance=1e-7):
    """Use a subspace CONTAINING the critical cone, never all active equalities."""
    rows = [jacobian[multipliers > tolerance]]
    if np.any(lower_multipliers > tolerance):
        rows.append(-np.eye(hessian.shape[0])[lower_multipliers > tolerance])
    matrix = np.vstack(rows)
    basis = null_space(matrix, rcond=1e-8) if len(matrix) else np.eye(hessian.shape[0])
    eigenvalues = np.linalg.eigvalsh(basis.T @ hessian @ basis)
    return {
        "containing_subspace_dimension": basis.shape[1],
        "eigenvalues": eigenvalues.tolist(),
        "minimum_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) else None,
        "positive_on_containing_subspace": not len(eigenvalues) or bool(eigenvalues[0] > 1e-7),
        "multiplier_threshold": tolerance,
    }


def inspect_endpoint(problem, regional, menu, full_region):
    x = regional.layout.pack(menu)
    h = regional.constraint_values(x)
    certificate = regional.first_order_certificate(
        x, active_tolerance=1e-4, stationarity_tolerance=1e-5
    )
    lam = np.zeros(len(h))
    lam[list(certificate.active_inequality_indices)] = certificate.inequality_multipliers
    mu = np.zeros(len(x))
    mu[list(certificate.active_lower_bound_indices)] = certificate.lower_bound_multipliers
    comp = max(float(np.max(np.abs(lam * h))), float(np.max(np.abs(mu * x))))
    full_violation = full_region.maximum_constraint_violation(x)
    record = {
        "restricted_primal_violation": certificate.primal_violation,
        "full_region_primal_violation": full_violation,
        "normalized_stationarity": certificate.normalized_stationarity_residual,
        "absolute_complementarity": comp,
        "restricted_profit": regional.profit(x),
        "active_choice_constraints": [],
        "active_set_sensitivity": [],
    }
    for tolerance in (1e-8, 1e-6, 1e-4):
        record["active_set_sensitivity"].append(
            {
                "tolerance": tolerance,
                "inequalities": np.flatnonzero(h >= -tolerance).tolist(),
                "lower_bounds": np.flatnonzero(x <= tolerance).tolist(),
            }
        )
    record["active_set_stable"] = all(
        item["inequalities"] == record["active_set_sensitivity"][0]["inequalities"]
        and item["lower_bounds"] == record["active_set_sensitivity"][0]["lower_bounds"]
        for item in record["active_set_sensitivity"]
    )
    active = np.flatnonzero(h >= -1e-4)
    choices = [regional.active_indices[j] for j in active if j < len(regional.active_indices)]
    record["active_choice_constraints"] = [
        [index.traveler, index.assigned_bundle, index.alternative_bundle] for index in choices
    ]
    compact = _HELPERS.regional_problem(problem, regional.assigned_bundles, choices)
    weights = np.zeros(compact.constraint_count)
    lookup = {index: row for row, index in enumerate(compact.active_indices)}
    for row, weight in enumerate(lam):
        if row < len(regional.active_indices):
            if regional.active_indices[row] in lookup:
                weights[lookup[regional.active_indices[row]]] = weight
        else:
            weights[len(compact.active_indices) + row - len(regional.active_indices)] = weight
    jac = compact.constraint_jacobian(x)
    hessian = np.zeros((len(x), len(x)))  # f=-profit is linear for the saved instance
    geometry, cache = [], {}
    designed, modes = problem.bundle_count - 1, problem.mode_count
    for index, weight in zip(compact.active_indices, weights, strict=False):
        for bundle, sign in ((index.assigned_bundle, 1), (index.alternative_bundle, -1)):
            if not bundle:
                continue
            key = index.traveler, bundle
            if key not in cache:
                hg, flags = _HELPERS.envelope_curvature(
                    problem.route_oracles[index.traveler][bundle],
                    menu.allowances[bundle],
                    problem.demand_distributions[index.traveler],
                )
                cache[key] = hg
                geometry.append({"traveler": index.traveler, "bundle": bundle, **flags})
            columns = slice(designed + (bundle - 1) * modes, designed + bundle * modes)
            hessian[columns, columns] += sign * weight * cache[key]
    record["valuation_branch_geometry"] = geometry
    record["selected_branch_lagrangian_hessian"] = hessian.tolist()
    base_values = compact.constraint_values(x)
    grad0 = -compact.profit_gradient(x) + jac.T @ weights
    differences = []
    for factor in (1e-3, 1e-4, 1e-5):
        estimated = np.zeros_like(hessian)
        max_jac_error, worst = 0.0, None
        for column in range(designed, len(x)):
            step = factor * max(1.0, abs(x[column]))
            high = x.copy()
            high[column] += step
            jh = compact.constraint_jacobian(high)
            gh = -compact.profit_gradient(high) + jh.T @ weights
            vh = compact.constraint_values(high)
            if x[column] > step:
                low = x.copy()
                low[column] -= step
                gl = -compact.profit_gradient(low) + compact.constraint_jacobian(low).T @ weights
                vl = compact.constraint_values(low)
                estimated[:, column] = (gh - gl) / (2 * step)
                derivative = (vh - vl) / (2 * step)
                side = "central"
            else:
                far = x.copy()
                far[column] += 2 * step
                gf = -compact.profit_gradient(far) + compact.constraint_jacobian(far).T @ weights
                vf = compact.constraint_values(far)
                estimated[:, column] = (-3 * grad0 + 4 * gh - gf) / (2 * step)
                derivative = (-3 * base_values + 4 * vh - vf) / (2 * step)
                side = "forward"
            error = float(np.max(np.abs(derivative - jac[:, column])))
            if error > max_jac_error:
                max_jac_error = error
                worst = {"column": column, "coordinate": float(x[column]), "side": side}
        differences.append(
            {
                "relative_step": factor,
                "jacobian_max_error": max_jac_error,
                "hessian_max_error": float(np.max(np.abs(estimated - hessian))),
                "hessian_asymmetry": float(np.max(np.abs(estimated - estimated.T))),
                "worst_jacobian_component": worst,
            }
        )
    record["derivative_checks"] = differences
    record["selected_branch_curvature"] = curvature_upper_space(hessian, jac, weights, mu)
    branch_flags = sum(
        item["parallel_switch_segments_with_positive_probability"]
        + item["multiple_envelope_intersections_in_support"]
        + item["uniform_support_switches"]
        for item in geometry
    )
    record["branch_flag_count"] = branch_flags
    norm_h = float(np.max(np.abs(hessian)))
    derivative_ok = (
        differences[-1]["jacobian_max_error"] <= 1e-5
        and differences[-1]["hessian_max_error"] <= 1e-4 + 0.02 * norm_h
        and branch_flags == 0
    )
    record["direct_classical_derivative_screen_passed"] = derivative_ok
    if certificate.primal_violation > 1e-7:
        status = "restricted_nlp_infeasible"
    elif certificate.normalized_stationarity_residual > 1e-5 or comp > 1e-6:
        status = "first_order_not_verified"
    elif not derivative_ok:
        status = "classical_sosc_not_certified_derivative_or_branch"
    elif not record["active_set_stable"]:
        status = "classical_sosc_not_certified_active_set_ambiguous"
    elif record["selected_branch_curvature"]["positive_on_containing_subspace"]:
        status = "numerical_sosc_supported_not_rigorous_certificate"
    else:
        status = "curvature_screen_inconclusive"
    record["sosc_status"] = status
    record["full_region_transfer_eligible"] = full_violation <= 1e-7
    return record


def audit(problem, trace, output):
    path = output / "sosc_checks.json"
    report = {
        "selection": trace["selection"],
        "replay_matches_saved_fields": trace["matches_saved_fields"],
        "scope": (
            "each normalized backend-return endpoint, using the restricted NLP actually solved"
        ),
        "thresholds": {
            "feasibility": 1e-7,
            "stationarity": 1e-5,
            "complementarity": 1e-6,
            "active_set": [1e-8, 1e-6, 1e-4],
        },
        "not_a_rigorous_mathematical_certificate": True,
        "attempts": [],
        "calls": [],
    }
    existing = {}
    if path.exists():
        previous = json.loads(path.read_text())
        existing = {row["attempt"]: row for row in previous["attempts"]}
    full_regions = {}
    for attempt in trace["backend_attempts"]:
        if attempt["attempt"] in existing:
            report["attempts"].append(existing[attempt["attempt"]])
            continue
        call = trace["calls"][attempt["call"] - 1]
        key = call["assignment_sha256"]
        alpha = np.array(trace["regions"][key]["assigned_bundles"])
        critical = CriticalConstraintSet(alpha, problem.bundle_count)
        critical.add_all(
            tuple(ChoiceConstraintIndex(*index) for index in call["choice_constraints"])
        )
        restricted = RestrictedRegionalProblem(problem, critical)
        if key not in full_regions:
            full_regions[key] = _HELPERS.regional_problem(problem, alpha)
        result = inspect_endpoint(
            problem, restricted, restore_menu(attempt["menu"]), full_regions[key]
        )
        result.update(
            {
                "attempt": attempt["attempt"],
                "call": call["call"],
                "region": call["region"],
                "cg_round": call["cg_round"],
                "source_solver_status": attempt["metrics"]["status"],
                "source_native_success": attempt["metrics"]["native_success"],
                "choice_constraint_count": len(call["choice_constraints"]),
                "numerical_constraint_jacobian": attempt["numerical_constraint_jacobian"],
            }
        )
        report["attempts"].append(result)
        save_json(path, report)
        print(
            json.dumps(
                {
                    "phase": "sosc",
                    "attempt": result["attempt"],
                    "call": result["call"],
                    "region": result["region"],
                    "status": result["sosc_status"],
                    "first_order": result["normalized_stationarity"],
                    "restricted_violation": result["restricted_primal_violation"],
                    "full_region_violation": result["full_region_primal_violation"],
                }
            ),
            flush=True,
        )
    by_attempt = {row["attempt"]: row for row in report["attempts"]}
    for call in trace["calls"]:
        # A successful combined call returns the final backend attempt. Verify
        # rather than assuming this relationship when producing per-call rows.
        number = call["backend_attempts"][-1]
        endpoint = trace["backend_attempts"][number - 1]["menu"]
        same = endpoint == call["menu"]
        assert same, "Combined call endpoint differs from final backend attempt"
        final = by_attempt[number]
        report["calls"].append(
            {
                k: final[k]
                for k in (
                    "call",
                    "region",
                    "cg_round",
                    "attempt",
                    "source_solver_status",
                    "choice_constraint_count",
                    "restricted_profit",
                    "restricted_primal_violation",
                    "full_region_primal_violation",
                    "normalized_stationarity",
                    "absolute_complementarity",
                    "active_set_stable",
                    "branch_flag_count",
                    "direct_classical_derivative_screen_passed",
                    "sosc_status",
                    "full_region_transfer_eligible",
                )
            }
            | {
                "selected_branch_min_eigenvalue": final["selected_branch_curvature"][
                    "minimum_eigenvalue"
                ],
                "containing_subspace_dimension": final["selected_branch_curvature"][
                    "containing_subspace_dimension"
                ],
            }
        )
    report["summary"] = {
        "backend_attempt_count": len(report["attempts"]),
        "restricted_call_count": len(report["calls"]),
        "backend_status_counts": dict(Counter(row["sosc_status"] for row in report["attempts"])),
        "call_status_counts": dict(Counter(row["sosc_status"] for row in report["calls"])),
        "complete_region_feasible_calls": sum(
            row["full_region_transfer_eligible"] for row in report["calls"]
        ),
    }
    report["complete"] = True
    save_json(path, report)
    for filename, rows in (
        ("nlp_calls.csv", report["calls"]),
        ("backend_attempts.csv", report["attempts"]),
    ):
        keys = list(report["calls"][0])
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            if filename == "backend_attempts.csv":
                rows = [
                    row
                    | {
                        "selected_branch_min_eigenvalue": row["selected_branch_curvature"][
                            "minimum_eigenvalue"
                        ],
                        "containing_subspace_dimension": row["selected_branch_curvature"][
                            "containing_subspace_dimension"
                        ],
                    }
                    for row in rows
                ]
            writer.writerows(rows)
    return report


def write_report(trace, report, output):
    selection = trace["selection"]
    calls = report["calls"]
    attempts = report["attempts"]
    endpoints = {}
    for row in calls:
        endpoints[row["region"]] = row
    region_rows = []
    for region, endpoint in sorted(endpoints.items()):
        members = [row for row in calls if row["region"] == region]
        region_rows.append(
            {
                "region": region,
                "first_call": members[0]["call"],
                "last_call": endpoint["call"],
                "restricted_calls": len(members),
                "backend_attempts": sum(item["region"] == region for item in attempts),
                "endpoint_profit": endpoint["restricted_profit"],
                "full_region_violation": endpoint["full_region_primal_violation"],
                "stationarity": endpoint["normalized_stationarity"],
                "complementarity": endpoint["absolute_complementarity"],
                "selected_branch_min_eigenvalue": endpoint["selected_branch_min_eigenvalue"],
                "sosc_status": endpoint["sosc_status"],
            }
        )
    with (output / "regions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(region_rows[0]))
        writer.writeheader()
        writer.writerows(region_rows)
    label = {
        "classical_sosc_not_certified_derivative_or_branch": "未确认：导数/分支",
        "first_order_not_verified": "未通过当前一阶验收",
        "restricted_nlp_infeasible": "当次 NLP 不可行",
        "classical_sosc_not_certified_active_set_ambiguous": "未确认：活跃集不稳定",
        "curvature_screen_inconclusive": "未确认：保守曲率检查不通过",
        "numerical_sosc_supported_not_rigorous_certificate": "数值 SOSC 支持（非严格证明）",
    }
    positive_calls = sum(
        row["selected_branch_min_eigenvalue"] is None
        or row["selected_branch_min_eigenvalue"] > 1e-7
        for row in calls
    )
    first_order_failures = [
        row for row in calls if row["sosc_status"] == "first_order_not_verified"
    ]
    final = calls[-1]
    text = [
        "# 最长区域切换路径的逐次 NLP SOSC 检查",
        "",
        "日期：2026-10-22。",
        "",
        f"对象：start {selection['start']}，{selection['regions']} 个区域、"
        f"{selection['transitions']} 次切换、{len(calls)} 次受限 NLP 调用、"
        f"{len(attempts)} 次底层 SLSQP 尝试。",
        "",
        "**结论：本次没有任何返回点获得经典 SOSC 的数值确认。** "
        "这是适用条件未确认或前置检查未通过，不等于证明这些点均非局部最优。",
        "",
        "## 选取与补录",
        "",
        "在工作量统计使用的全部 100 条已完成路径中，按 assignment_changes 选最长路径。"
        f"实际数据源为 `{selection['source']}`。",
        "",
        "旧文件没有保存每个中间 NLP 的完整菜单与约束索引。因此保留原初始菜单、"
        "原求解器参数、原接受规则及缓存极点，确定性重放补录。"
        "记录每次受限 NLP 的真实约束子集、固定 assignment、输入菜单、返回菜单，"
        "并捕获恢复过程中的每次底层尝试。二阶分析在重放之后进行，不影响原路径。",
        "",
        f"重放吻合标志：`{trace['matches_saved_fields']}`。",
        "```json",
        json.dumps(trace["comparison"], indent=2),
        "```",
        "",
        "原接受配置 `check_kkt=False` 保持不变；本次在求解完成后另行检验。"
        "所有 44 次底层调用仍为 SLSQP，没有更换后端。第 15 次受限调用发生 4 次底层尝试，"
        "前三次返回点不满足当次可行性要求，第 4 次被原策略接受。",
        "",
        "## 数值判定规则",
        "",
        "- 当次受限 NLP 可行性阈值 `1e-7`；归一化一阶平衡阈值 `1e-5`；"
        "绝对互补残差阈值 `1e-6`。乘子由非负最小二乘拟合。",
        "- 活跃集以 `1e-4` 建立，另用 `1e-8`、`1e-6` 检查敏感性。",
        "- 每个返回点同时检查完整固定 assignment 区域的全部旅客约束。"
        "受限 NLP 的 SOSC 即使成立，也只有在该点对完整区域可行时才可转移。",
        "- 用当前积分 envelope 构造真实单分支的解析 Hessian，"
        "不使用 SLSQP 的 BFGS 矩阵。成本为原实例的线性 allowance 成本。",
        "- 二阶方向使用正乘子约束法向量的零空间，"
        "它包含所选乘子对应的临界锥。零乘子活跃不等式没有被强行当作等式。"
        "因此在该较大子空间上正定是保守曲率支持，不通过则不能直接否定真实临界锥上的 SOSC。",
        "- 对非线性坐标使用相对步长 `1e-3`、`1e-4`、`1e-5` 检查导数及 Hessian，"
        "下界附近使用单侧公式。平行活跃面、多重交点和支持端点被单独标记。",
        "- 只有可行性、一阶、互补性、导数/分支、活跃集和曲率的检查全部通过，"
        "才报告数值 SOSC 支持；仍不当作区间算术意义的严格证明。",
        "",
        "## 逐次调用的汇总",
        "",
        "```json",
        json.dumps(report["summary"], indent=2),
        "```",
        "",
        f"{positive_calls}/{len(calls)} 个受限 NLP 返回点的**当前单分支**保守投影最小特征值为正。"
        "但一阶与导数适用条件不能被这个正特征值替代。",
        "",
        "9 个未通过本次一阶验收的调用如下。注意第 18 次调用是一阶平衡通过，"
        "但互补残差略超过所设阈值；改变验收阈值可能改变这种边界分类。",
        "",
        "| NLP 调用 | 区域 | 一阶平衡残差 | 互补残差 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in first_order_failures:
        text.append(
            f"| {row['call']} | {row['region']} | {row['normalized_stationarity']:.3e} "
            f"| {row['absolute_complementarity']:.3e} |"
        )
    text.extend(
        [
            "",
            "这说明 solver 原生 success 并未自动完成独立一阶验收。"
            "反过来，这些拟合残差也不证明不存在其他次梯度或乘子满足某种广义驻点条件。",
            "",
            "## 14 个区域终点",
            "",
            "| 区域 | NLP 调用范围 | 一阶平衡残差 | 互补残差 | 单分支投影最小特征值 | SOSC 状态 |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in region_rows:
        eigen = row["selected_branch_min_eigenvalue"]
        eigen_text = f"{eigen:.3e}" if eigen is not None else "临界子空间为零"
        text.append(
            f"| {row['region']} | {row['first_call']}–{row['last_call']} "
            f"| {row['stationarity']:.3e} | {row['complementarity']:.3e} "
            f"| {eigen_text} | {label[row['sosc_status']]} |"
        )
    text.extend(
        [
            "",
            "14 个终点均对完整区域数值可行。其中第 10、11 区域的终点分别是一阶平衡"
            "残差约 `3.10e-3` 和 `9.48e-4`，未通过本次一阶验收。其余 12 个终点"
            "通过选定次梯度下一阶验收，但经典二阶导数条件仍未确认。",
            "",
            "## 最后一次 NLP 的具体结果",
            "",
            f"第 {final['call']} 次调用，区域 {final['region']}；固定分配利润 "
            f"`{final['restricted_profit']:.12f}` USD/代表性加权旅客/月。",
            f"原区域违反量 `{final['full_region_primal_violation']:.3e}`，"
            f"归一化一阶平衡残差 `{final['normalized_stationarity']:.3e}`，"
            f"互补残差 `{final['absolute_complementarity']:.3e}`。",
            "",
            "该点 premium bike allowance 在零下界。其相关活跃约束的 oracle 次梯度与"
            "可行侧差分导数的最大差异约 `0.006757`，步长降低后仍存在。"
            "直接对原 oracle Jacobian 再差分时，和单分支解析 Hessian 的误差从约 "
            "`0.654` 增长至 `65.52`。这不能当作已获得真实光滑 Hessian。",
            "",
            "并列面的存在本身不是对可行域不存在光滑延拓的证明。"
            "本次结论仅是：当前 oracle 次梯度、单分支 Hessian 与完整经典 SOSC 的适用条件"
            "尚未建立一致性，因而不颁发经典 SOSC 确认。",
            "",
            "## 文件与复现",
            "",
            "- `replay.json`：41 次调用、44 次底层尝试的菜单、真实约束索引和固定 assignment。",
            "- `sosc_checks.json`：每个点的矩阵、乘子残差、导数检查、分支标记及状态。",
            "- `nlp_calls.csv`：41 次受限 NLP 返回点的汇总。",
            "- `backend_attempts.csv`：含 3 次失败重试的 44 个返回点汇总。",
            "- `regions.csv`：14 个完整区域终点。",
            "",
            "```sh",
            "PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\",
            "  /opt/homebrew/Caskroom/miniforge/base/envs/maas-bundle/bin/python \\",
            "  scripts/check-longest-trajectory-sosc.py",
            "```",
            "",
            "已有完整 `replay.json` 时直接复用；已有逐点审计时断点复用。"
            "如需从头独立重放，传入一个新的 `--output` 目录。"
            "原实验、论文和生产求解器均未改写。",
            "",
            "方法依据：[临界锥与二阶条件课程资料]"
            "(https://faculty.ucmerced.edu/mcarreira-perpinan/teaching/EECS260/2014-Fall/lecture-notes.pdf)、"
            "[IPOPT 导数检查说明](https://coin-or.github.io/Ipopt/SPECIALS.html)。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/longest_trajectory_sosc_20261022")
    )
    parser.add_argument("--phase", choices=("replay", "audit", "all"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    selection, source = select_longest(root)
    problem = build_sioux_falls_problem(
        oracle_cache_path=root / "results/sioux_falls_five_modes/dual_vertices.npz"
    )
    assert problem.unit_allowance_cost is not None, (
        "This Hessian audit requires the saved linear cost"
    )
    print(json.dumps({"selection": selection}), flush=True)
    if args.phase == "audit":
        trace = json.loads((args.output / "replay.json").read_text())
    else:
        trace = replay(problem, selection, source, args.output)
    if args.phase != "replay":
        report = audit(problem, trace, args.output)
        write_report(trace, report, args.output)
        print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
