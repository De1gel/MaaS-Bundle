"""A2: observe regional search and enumerate 26 non-PAYG candidate regions.

The enumeration exhausts assignments, not continuous local attraction basins.
No model, optimistic response, or search acceptance rule is reimplemented here.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from maas_bundle.design.assignment import AssignmentResult
from maas_bundle.design.boundary import ChoiceConstraintIndex, CriticalConstraintSet
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import solve_fixed_assignment, solve_primal
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import (
    MenuVariableLayout,
    RegionalSolveResult,
    RestrictedRegionalProblem,
    ScipyRegionalSolver,
)
from maas_bundle.examples.stylized import (
    build_stylized_problem,
    draw_stylized_menu,
    initial_stylized_menu,
)
from maas_bundle.examples.stylized_config import DEFAULT_STYLIZED_CONFIG

CLASSES = ("P", "M", "S")
OPTIONS = ("PAYG", "Basic", "Premium")


@dataclass(frozen=True)
class StylizedRegionalSettings:
    """A single solver/domain/tolerance policy shared by all three methods."""

    starts: int = 1
    seed: int = 20261004
    max_nlp_iterations: int = 500
    max_region_iterations: int = 100
    feasibility_tolerance: float = 1e-7
    margin_tolerance: float = 1e-9
    profit_tolerance: float = 1e-9
    optimality_tolerance: float = 1e-5
    active_set_tolerance: float = 1e-4
    ftol: float = 1e-10
    trajectory_start: str = "switch"

    def __post_init__(self) -> None:
        if self.trajectory_start not in {"switch", "baseline"}:
            raise ValueError("trajectory_start must be 'switch' or 'baseline'")
        for name in ("starts", "max_nlp_iterations", "max_region_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.starts != 1 and self.starts < 4:
            raise ValueError("starts must be 1, or at least 4 to include all common anchors")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in (
            "feasibility_tolerance",
            "margin_tolerance",
            "profit_tolerance",
            "optimality_tolerance",
            "active_set_tolerance",
            "ftol",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def solver(self) -> ScipyRegionalSolver:
        # One independent initial menu per assignment; numerical recovery is
        # internal to the adapter and every backend attempt remains counted.
        return ScipyRegionalSolver(
            options={"maxiter": self.max_nlp_iterations, "ftol": self.ftol},
            feasibility_tolerance=self.feasibility_tolerance,
            optimality_tolerance=self.optimality_tolerance,
            active_set_tolerance=self.active_set_tolerance,
            enable_recovery=True,
            numerical_constraint_recovery=True,
            objective_tolerance=self.profit_tolerance,
        )


class _RecordingSolver:
    """Capture actual NLP entry menus, including CG restarts and boundary trials."""

    def __init__(self, solver: ScipyRegionalSolver) -> None:
        self.solver = solver
        self.feasibility_tolerance = solver.feasibility_tolerance
        self.optimality_tolerance = solver.optimality_tolerance
        self.active_set_tolerance = solver.active_set_tolerance
        self.objective_tolerance = solver.objective_tolerance
        self.entries: dict[int, BundleMenu] = {}

    @property
    def name(self) -> str:
        return self.solver.name

    def solve(
        self, regional_problem: RestrictedRegionalProblem, initial_menu: BundleMenu
    ) -> RegionalSolveResult:
        result = self.solver.solve(regional_problem, initial_menu)
        self.entries[id(result)] = initial_menu
        return result

    def entry_for(self, result: RegionalSolveResult) -> BundleMenu:
        if result.initial_menu is not None:
            return result.initial_menu
        return self.entries[id(result)]


DEFAULT_STYLIZED_REGIONAL_SETTINGS = StylizedRegionalSettings()


def regional_illustration_menu() -> BundleMenu:
    """A positive-profit, strict-choice start illustrating a real S switch.

    Only the experiment's starting menu changes. Traveler/model parameters
    and the algorithm's production optimistic response remain unchanged.
    """
    return BundleMenu(
        prices=np.array([0.0, 50.0, 60.0]),
        allowances=np.array([[0.0, 0.0], [0.0, 30.0], [40.0, 12.0]]),
    )


def _menu(menu: BundleMenu) -> dict:
    return {"prices": menu.prices.tolist(), "allowances": menu.allowances.tolist()}


def _restore_menu(record: dict) -> BundleMenu:
    return BundleMenu(np.asarray(record["prices"]), np.asarray(record["allowances"]))


def _response(
    problem: BundleDesignProblem, menu: BundleMenu, settings: StylizedRegionalSettings
) -> AssignmentResult:
    return problem.evaluate_assignment(
        menu,
        choice_tolerance=settings.feasibility_tolerance,
        profit_tolerance=settings.margin_tolerance,
    )


def _response_record(response: AssignmentResult) -> dict:
    double_ties = []
    for traveler, mask in enumerate(response.numerical_tie_mask):
        for a, b in combinations(np.flatnonzero(mask).tolist(), 2):
            if (
                abs(response.bundle_margins[b] - response.bundle_margins[a])
                > response.margin_tolerance
            ):
                continue
            double_ties.append(
                {
                    "class": CLASSES[traveler],
                    "pair": [a, b],
                    "cost_gap": float(
                        response.generalized_costs[traveler, b]
                        - response.generalized_costs[traveler, a]
                    ),
                    "margin_gap": float(response.bundle_margins[b] - response.bundle_margins[a]),
                    "optimistic_pair": bool(
                        response.numerical_optimistic_mask[traveler, a]
                        and response.numerical_optimistic_mask[traveler, b]
                    ),
                }
            )
    return {
        "assignment": response.bundle_indices.tolist(),
        "true_profit": response.total_profit,
        "generalized_costs": response.generalized_costs.tolist(),
        "minimum_cost_sets": [
            np.flatnonzero(mask).tolist() for mask in response.numerical_tie_mask
        ],
        "optimistic_sets": [
            np.flatnonzero(mask).tolist() for mask in response.numerical_optimistic_mask
        ],
        "bundle_margins": response.bundle_margins.tolist(),
        "double_ties": double_ties,
    }


def _full_region(problem: BundleDesignProblem, assignment) -> RestrictedRegionalProblem:
    pool = CriticalConstraintSet(np.asarray(assignment), problem.bundle_count)
    pool.add_all(
        ChoiceConstraintIndex(i, int(a), b)
        for i, a in enumerate(assignment)
        for b in range(problem.bundle_count)
        if b != a
    )
    return RestrictedRegionalProblem(problem, pool)


def _check_menu(problem, menu, assignment, settings) -> dict:
    """Check every choice inequality, box bound, price ordering and true response."""
    assignment = np.asarray(assignment, dtype=int)
    response = _response(problem, menu, settings)
    costs = response.generalized_costs
    selected = costs[np.arange(problem.traveler_count), assignment]
    choice_violation = float(np.max(selected[:, None] - costs, initial=0.0))
    domain_violation = float(
        max(
            0.0,
            np.max(menu.prices[1:] - problem.price_upper_bounds),
            np.max(menu.allowances[1:] - problem.allowance_upper_bounds),
            np.max(-menu.prices),
            np.max(-menu.allowances),
            np.max(-np.diff(menu.prices)),
        )
    )
    profit = float(problem.traveler_weights @ problem.bundle_margins(menu)[assignment])
    optimistic_profit = float(
        problem.traveler_weights @ problem.bundle_margins(menu)[response.bundle_indices]
    )
    return {
        "choice_violation": choice_violation,
        "domain_violation": domain_violation,
        "maximum_violation": max(choice_violation, domain_violation),
        "design_bounds_satisfied": domain_violation <= settings.feasibility_tolerance,
        "assignment_consistent": bool(np.array_equal(assignment, response.bundle_indices)),
        "independent_assigned_profit": profit,
        "independent_true_profit": optimistic_profit,
        "profit_recomputation_error": abs(optimistic_profit - response.total_profit),
    }


def _search(problem, initial, settings, *, all_constraints: bool) -> dict:
    solver = _RecordingSolver(settings.solver())
    cpu, wall = process_time(), perf_counter()
    diagnostics = solve_primal(
        problem,
        initial,
        solver,
        initial_per_pair=problem.traveler_count if all_constraints else 1,
        max_region_iterations=settings.max_region_iterations,
        choice_tolerance=settings.feasibility_tolerance,
        violation_tolerance=settings.feasibility_tolerance,
        profit_tolerance=settings.profit_tolerance,
        margin_tolerance=settings.margin_tolerance,
    )
    cpu, wall = process_time() - cpu, perf_counter() - wall
    candidate = diagnostics.candidate
    initial_response = candidate.initial_assignment
    rows = []
    incumbent_profit = initial_response.total_profit
    incumbent_assignment = initial_response.bundle_indices.copy()
    accepted_assignment_path = [incumbent_assignment.tolist()]
    visited = {tuple(initial_response.bundle_indices)}
    solved = set()
    for step in candidate.iterations:
        fixed = step.fixed_assignment_result
        first = fixed.iterations[0].solve_result
        entry = solver.entry_for(first)
        entry_response = _response(problem, entry, settings)
        final = fixed.final_solve_result
        response = _response(problem, fixed.menu, settings)
        switches = []
        for i, (a, b) in enumerate(
            zip(step.assumed_assignment, response.bundle_indices, strict=True)
        ):
            if a == b:
                continue
            gap = float(response.generalized_costs[i, b] - response.generalized_costs[i, a])
            switches.append(
                {
                    "class": CLASSES[i],
                    "from": int(a),
                    "to": int(b),
                    "from_name": OPTIONS[a],
                    "to_name": OPTIONS[b],
                    "pair": [int(a), int(b)],
                    "cost_gap": gap,
                    "tied": abs(gap) <= settings.feasibility_tolerance,
                    "margin_gap": float(response.bundle_margins[b] - response.bundle_margins[a]),
                }
            )
        checked = _check_menu(problem, fixed.menu, step.assumed_assignment, settings)
        retained_changed = step.accepted and not np.array_equal(
            response.bundle_indices, incumbent_assignment
        )
        mechanism = None
        if retained_changed:
            if step.boundary_origin is not None:
                mechanism = "boundary_neighbor_optimization"
            elif switches and all(
                abs(switch["margin_gap"]) <= settings.margin_tolerance for switch in switches
            ):
                mechanism = "double_tie_resolution"
            else:
                mechanism = "optimistic_reassignment"
        rows.append(
            {
                "iteration": step.number,
                "starting_menu": _menu(entry),
                "starting_assignment": step.assumed_assignment.tolist(),
                "starting_optimistic_assignment": entry_response.bundle_indices.tolist(),
                "profit_before_solve": entry_response.total_profit,
                "assigned_profit_before_solve": float(
                    problem.traveler_weights
                    @ problem.bundle_margins(entry)[step.assumed_assignment]
                ),
                "incumbent_profit_before_solve": incumbent_profit,
                "retained_assignment_before_solve": incumbent_assignment.tolist(),
                "retained_assignment_changed": bool(retained_changed),
                "assignment_change_mechanism": mechanism,
                "optimized_menu": _menu(fixed.menu),
                "regional_profit": final.objective_value,
                "same_menu_reassignment_profit_gain": response.total_profit - final.objective_value,
                "response": _response_record(response),
                "switches": switches,
                "accepted": step.accepted,
                "boundary_origin": step.boundary_origin,
                "boundary_returned": step.boundary_returned,
                "assignment_revisited": step.revisited_assignment,
                "assumed_region_revisited": tuple(step.assumed_assignment) in solved,
                "fixed_assignment_status": fixed.status.value,
                "solver": final.metrics(),
                "first_order_certificate": asdict(final.first_order_certificate),
                "validation": checked,
                "active_constraints": len(fixed.active_constraints),
                "generated_constraints": fixed.added_constraint_count,
                "nlp_solves": fixed.nlp_solve_count,
                "generation_iterations": [
                    {
                        "iteration": t.number,
                        "starting_menu": _menu(solver.entry_for(t.solve_result)),
                        "optimized_menu": _menu(t.solve_result.menu),
                        "solver": t.solve_result.metrics(),
                        "certificate": asdict(t.solve_result.first_order_certificate),
                        "message": t.solve_result.message,
                        "active_constraints": t.active_constraint_count,
                        "added_constraints": [asdict(index) for index in t.added_constraints],
                        "full_separation_violation": (
                            None
                            if t.separation is None
                            else _check_menu(
                                problem, t.solve_result.menu, step.assumed_assignment, settings
                            )["choice_violation"]
                        ),
                    }
                    for t in fixed.iterations
                ],
            }
        )
        visited.add(tuple(step.assumed_assignment))
        solved.add(tuple(step.assumed_assignment))
        if step.accepted:
            visited.add(tuple(response.bundle_indices))
            incumbent_profit = response.total_profit
            incumbent_assignment = response.bundle_indices.copy()
            if retained_changed:
                accepted_assignment_path.append(incumbent_assignment.tolist())
    response = _response(problem, candidate.menu, settings)
    return {
        "method": "Regional search + all traveler constraints"
        if all_constraints
        else "Regional search + CG",
        "status": candidate.status.value,
        "diagnostics": diagnostics.metrics(),
        "diagnostic_message": diagnostics.message,
        "initial_menu": _menu(initial),
        "initial_response": _response_record(initial_response),
        "iterations": rows,
        "final_menu": _menu(candidate.menu),
        "final_response": _response_record(response),
        "final_validation": _check_menu(problem, candidate.menu, candidate.assignment, settings),
        "region_iterations": candidate.region_iteration_count,
        "terminal_region": candidate.terminal_iteration.number,
        "distinct_assignments_visited": len(visited),
        "assignments_visited": [[int(a) for a in signature] for signature in sorted(visited)],
        "nlp_solves": candidate.nlp_solve_count,
        "generated_constraints": sum(r["generated_constraints"] for r in rows),
        "initial_active_constraints": sum(
            r["generation_iterations"][0]["active_constraints"] for r in rows
        ),
        "accepted_iterations": sum(r["accepted"] for r in rows),
        "accepted_assignment_path": accepted_assignment_path,
        "accepted_assignment_transitions": sum(r["retained_assignment_changed"] for r in rows),
        "accepted_optimistic_reassignments": sum(
            r["accepted"] and bool(r["switches"]) for r in rows
        ),
        "accepted_strict_margin_reassignments": sum(
            r["accepted"]
            and bool(r["switches"])
            and r["same_menu_reassignment_profit_gain"] > settings.profit_tolerance
            for r in rows
        ),
        "accepted_boundary_transitions": sum(
            r["assignment_change_mechanism"] == "boundary_neighbor_optimization" for r in rows
        ),
        "rejected_iterations": sum(not r["accepted"] for r in rows),
        "assignment_revisited": any(
            r["assignment_revisited"] or r["assumed_region_revisited"] for r in rows
        ),
        "double_tie_encountered": bool(_response_record(initial_response)["double_ties"])
        or any(r["response"]["double_ties"] for r in rows),
        "accepted_profit_nondecreasing": all(
            not r["accepted"]
            or r["response"]["true_profit"]
            >= r["incumbent_profit_before_solve"] - settings.profit_tolerance
            for r in rows
        ),
        "cpu_time_seconds": cpu,
        "wall_time_seconds": wall,
    }


def _enumerate(problem, starts, settings, progress) -> dict:
    records = []
    solver = settings.solver()
    total_cpu, total_wall = process_time(), perf_counter()
    assignments = [a for a in product(range(3), repeat=3) if a != (0, 0, 0)]
    for number, assignment in enumerate(assignments, 1):
        region = _full_region(problem, assignment)
        cpu, wall = process_time(), perf_counter()
        attempts, feasible = [], []
        for start_number, menu in enumerate(starts, 1):
            fixed = solve_fixed_assignment(
                problem,
                menu,
                assignment,
                solver,
                initial_per_pair=problem.traveler_count,
                violation_tolerance=settings.feasibility_tolerance,
            )
            result = fixed.final_solve_result
            response = _response(problem, fixed.menu, settings)
            checked = _check_menu(problem, fixed.menu, assignment, settings)
            is_feasible = checked["maximum_violation"] <= settings.feasibility_tolerance
            attempt = {
                "start": start_number,
                "fixed_assignment_status": fixed.status.value,
                "solver": result.metrics(),
                "message": result.message,
                "certificate": asdict(result.first_order_certificate),
                "validation": checked,
                "feasible": is_feasible,
                "regional_profit": checked["independent_assigned_profit"],
                "optimistic_assignment": response.bundle_indices.tolist(),
                "true_profit": checked["independent_true_profit"],
                "double_ties": _response_record(response)["double_ties"],
                "nlp_solves": fixed.nlp_solve_count,
                "generated_constraints": fixed.added_constraint_count,
            }
            attempts.append(attempt)
            if is_feasible:
                feasible.append(
                    {
                        **attempt,
                        "menu": _menu(fixed.menu),
                        "response": _response_record(response),
                        "certified_regional": fixed.converged,
                        "full_region_certificate": asdict(
                            region.first_order_certificate(
                                region.layout.pack(fixed.menu),
                                feasibility_tolerance=settings.feasibility_tolerance,
                                stationarity_tolerance=settings.optimality_tolerance,
                                active_tolerance=settings.active_set_tolerance,
                            )
                        ),
                    }
                )
        best = max(feasible, key=lambda r: r["regional_profit"], default=None)
        best_true = max(feasible, key=lambda r: r["true_profit"], default=None)
        successful = sum(a["fixed_assignment_status"] == "converged" for a in attempts)
        record = {
            "assignment": list(assignment),
            "starts": len(starts),
            "successful_starts": successful,
            "failed_starts": len(starts) - successful,
            "feasible_starts": len(feasible),
            "status": "feasible solution found"
            if best
            else "no feasible solution found by "
            + ("single-start" if len(starts) == 1 else "multi-start"),
            "solver_status": "success" if successful else "solver failure",
            "best_regional_profit": None if best is None else best["regional_profit"],
            "best_feasible_menu": None if best is None else best["menu"],
            "best_start": None if best is None else best["start"],
            "best": best,
            "best_optimistic": best_true,
            "attempts": attempts,
            "nlp_solves": sum(a["nlp_solves"] for a in attempts),
            "generated_constraints": sum(a["generated_constraints"] for a in attempts),
            "cpu_time_seconds": process_time() - cpu,
            "wall_time_seconds": perf_counter() - wall,
        }
        records.append(record)
        if progress:
            progress(
                f"Enumeration {number}/{len(assignments)} {assignment}: {successful}/{len(starts)} "
                f"successful, best regional profit={record['best_regional_profit']}"
            )
    # Every feasible fixed-region solution is also a valid menu. Its production
    # optimistic response can select another region at a choice boundary.
    best = max(
        (r["best_optimistic"] for r in records if r["best_optimistic"]),
        key=lambda r: r["true_profit"],
        default=None,
    )
    origin = next((r["assignment"] for r in records if r["best_optimistic"] is best), None)
    return {
        "method": "Exhaustive assignment enumeration + "
        + ("single start" if len(starts) == 1 else "multi-start"),
        "interpretation": "exhaustive assignments with local continuous solves; no global proof",
        "assignment_count": len(records),
        "assignment_combinations": 27,
        "excluded_assignments": [
            {"assignment": [0, 0, 0], "reason": "all-PAYG assigned profit is identically zero"}
        ],
        "starts_per_assignment": len(starts),
        "start_menus": [_menu(menu) for menu in starts],
        "start_responses": [
            _response_record(_response(problem, menu, settings)) for menu in starts
        ],
        "start_scheme": "one numerical seed shared across all fixed regions"
        if len(starts) == 1
        else "trajectory initial, original baseline initial, zero menu, CG endpoint, "
        "then independent bounded random menus; identical starts in all 26 candidate regions",
        "records": records,
        "best": best,
        "best_origin_assignment": origin,
        "nlp_solves": sum(r["nlp_solves"] for r in records),
        "generated_constraints": sum(r["generated_constraints"] for r in records),
        "successful_starts": sum(r["successful_starts"] for r in records),
        "failed_starts": sum(r["failed_starts"] for r in records),
        "double_tie_encountered": any(
            attempt["double_ties"] for r in records for attempt in r["attempts"]
        )
        or any(
            _response_record(_response(problem, menu, settings))["double_ties"] for menu in starts
        ),
        "cpu_time_seconds": process_time() - total_cpu,
        "wall_time_seconds": perf_counter() - total_wall,
    }


def run_stylized_regional(
    settings: StylizedRegionalSettings = DEFAULT_STYLIZED_REGIONAL_SETTINGS,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run A2 with one bounded problem shared by CG, all constraints and enumeration."""
    total_cpu, total_wall = process_time(), perf_counter()
    problem = build_stylized_problem(price_upper_bound=190.0, allowance_upper_bound=64.0)
    baseline_menu = initial_stylized_menu()
    initial = (
        regional_illustration_menu() if settings.trajectory_start == "switch" else baseline_menu
    )
    initial_response = _response(problem, initial, settings)
    cg = _search(problem, initial, settings, all_constraints=False)
    all_constraints = _search(problem, initial, settings, all_constraints=True)
    baseline_reference = (
        _search(problem, baseline_menu, settings, all_constraints=False)
        if settings.trajectory_start == "switch"
        else cg
    )
    if progress:
        progress(
            f"CG: {cg['status']}, profit={cg['final_response']['true_profit']:.9f}; "
            f"all constraints: {all_constraints['status']}, "
            f"profit={all_constraints['final_response']['true_profit']:.9f}"
        )
    if settings.starts == 1:
        starts = [settings.solver().default_initial_menu(_full_region(problem, [1, 1, 1]))]
    else:
        random = np.random.default_rng(settings.seed)
        starts = [
            initial,
            baseline_menu,
            BundleMenu(np.zeros(3), np.zeros((3, 2))),
            _restore_menu(cg["final_menu"]),
        ]
        starts += [draw_stylized_menu(random) for _ in range(settings.starts - len(starts))]
    enumeration = _enumerate(problem, starts, settings, progress)
    if settings.starts == 1:
        enumeration["initialization_owner"] = "ScipyRegionalSolver.default_initial_menu"
    best = enumeration["best"]
    enum_profit = None if best is None else best["true_profit"]
    cg_profit = cg["final_response"]["true_profit"]
    gap = None if best is None else enum_profit - cg_profit
    layout = MenuVariableLayout(3, 2)
    menu_difference = float(
        np.max(
            np.abs(
                layout.pack(_restore_menu(cg["final_menu"]))
                - layout.pack(_restore_menu(all_constraints["final_menu"]))
            )
        )
    )
    comparison = []
    for run in (cg, all_constraints):
        comparison.append(
            {
                "method": run["method"],
                "profit": run["final_response"]["true_profit"],
                "assignment": run["final_response"]["assignment"],
                "status": run["status"],
                "max_constraint_violation": run["final_validation"]["maximum_violation"],
                "assignments_visited": run["distinct_assignments_visited"],
                "regional_nlp_solves": run["nlp_solves"],
                "generated_constraints": run["generated_constraints"],
                "initial_active_constraints": run["initial_active_constraints"],
                "cpu_time_seconds": run["cpu_time_seconds"],
                "wall_time_seconds": run["wall_time_seconds"],
            }
        )
    comparison.append(
        {
            "method": enumeration["method"],
            "profit": enum_profit,
            "assignment": None if best is None else best["optimistic_assignment"],
            "status": "feasible solution found"
            if best
            else "no feasible solution found by "
            + ("single-start" if settings.starts == 1 else "multi-start"),
            "max_constraint_violation": None
            if best is None
            else _check_menu(
                problem, _restore_menu(best["menu"]), best["optimistic_assignment"], settings
            )["maximum_violation"],
            "assignments_visited": enumeration["assignment_count"],
            "regional_nlp_solves": enumeration["nlp_solves"],
            "generated_constraints": enumeration["generated_constraints"],
            "initial_active_constraints": 6 * settings.starts * enumeration["assignment_count"],
            "cpu_time_seconds": enumeration["cpu_time_seconds"],
            "wall_time_seconds": enumeration["wall_time_seconds"],
        }
    )
    agreement = {
        "same_final_assignment": cg["final_response"]["assignment"]
        == all_constraints["final_response"]["assignment"],
        "profit_difference": cg_profit - all_constraints["final_response"]["true_profit"],
        "max_menu_variable_difference": menu_difference,
        "profit_matches": abs(cg_profit - all_constraints["final_response"]["true_profit"])
        <= settings.profit_tolerance,
        "menu_matches": menu_difference <= settings.feasibility_tolerance,
    }
    used_bundles = sorted(
        set(cg["final_response"]["assignment"])
        | set(all_constraints["final_response"]["assignment"])
    )
    used_bundles = [b for b in used_bundles if b != 0]
    cg_menu = _restore_menu(cg["final_menu"])
    full_menu = _restore_menu(all_constraints["final_menu"])
    agreement["unused_subscription_bundles"] = [b for b in (1, 2) if b not in used_bundles]
    agreement["max_used_menu_variable_difference"] = float(
        max(
            np.max(
                np.abs(cg_menu.prices[used_bundles] - full_menu.prices[used_bundles]), initial=0.0
            ),
            np.max(
                np.abs(cg_menu.allowances[used_bundles] - full_menu.allowances[used_bundles]),
                initial=0.0,
            ),
        )
    )
    # Check all menus that are actually exported, including inner CG solutions.
    reported = [initial, *starts]
    for run in (cg, all_constraints, baseline_reference):
        reported.append(_restore_menu(run["final_menu"]))
        for row in run["iterations"]:
            for generation in row["generation_iterations"]:
                reported.extend(
                    _restore_menu(generation[key]) for key in ("starting_menu", "optimized_menu")
                )
    for record in enumeration["records"]:
        for key in ("best", "best_optimistic"):
            if record[key]:
                reported.append(_restore_menu(record[key]["menu"]))
    domain_checks = [
        _check_menu(problem, m, _response(problem, m, settings).bundle_indices, settings)
        for m in reported
    ]
    endpoint_checks = [cg["final_validation"], all_constraints["final_validation"]]
    if best:
        endpoint_checks.append(
            _check_menu(
                problem, _restore_menu(best["menu"]), best["optimistic_assignment"], settings
            )
        )
    validity = {
        "all_reported_design_bounds_satisfied": all(
            c["design_bounds_satisfied"] for c in domain_checks
        ),
        "max_reported_domain_violation": max(c["domain_violation"] for c in domain_checks),
        "all_final_assignments_consistent": all(
            c["assignment_consistent"] for c in endpoint_checks
        ),
        "max_final_constraint_violation": max(c["maximum_violation"] for c in endpoint_checks),
        "true_profit_recomputation_max_error": max(
            c["profit_recomputation_error"] for c in endpoint_checks
        ),
        "accepted_profit_nondecreasing": cg["accepted_profit_nondecreasing"]
        and all_constraints["accepted_profit_nondecreasing"],
        "shared_model_solver_domain_tolerances": True,
        "accepted_switches_are_cost_ties": all(
            switch["tied"]
            for run in (cg, all_constraints)
            for row in run["iterations"]
            if row["accepted"]
            for switch in row["switches"]
        ),
        "accepted_switches_have_nondecreasing_margin": all(
            switch["margin_gap"] >= -settings.margin_tolerance
            for run in (cg, all_constraints)
            for row in run["iterations"]
            if row["accepted"]
            for switch in row["switches"]
        ),
        "double_tie_in_search": cg["double_tie_encountered"]
        or all_constraints["double_tie_encountered"],
        "double_tie_in_enumeration": enumeration["double_tie_encountered"],
        "max_enumeration_region_choice_violation": max(
            (
                r["best"]["validation"]["choice_violation"]
                for r in enumeration["records"]
                if r["best"]
            ),
            default=0.0,
        ),
    }
    validity["passed"] = (
        validity["all_reported_design_bounds_satisfied"]
        and validity["all_final_assignments_consistent"]
        and validity["accepted_profit_nondecreasing"]
        and validity["accepted_switches_are_cost_ties"]
        and validity["accepted_switches_have_nondecreasing_margin"]
        and validity["max_final_constraint_violation"] <= settings.feasibility_tolerance
        and validity["max_enumeration_region_choice_violation"] <= settings.feasibility_tolerance
        and validity["true_profit_recomputation_max_error"] <= settings.profit_tolerance
    )
    model_metadata = DEFAULT_STYLIZED_CONFIG.metadata()
    model_metadata["design_domain"].update(
        finite_upper_bounds=True,
        price_upper_bounds=problem.price_upper_bounds.tolist(),
        allowance_upper_bounds=problem.allowance_upper_bounds.tolist(),
    )
    return {
        "experiment": "A2 — Regional search and assignment benchmark",
        "model": model_metadata,
        "settings": asdict(settings),
        "domain": {
            "price_upper_bounds": problem.price_upper_bounds.tolist(),
            "allowance_upper_bounds": problem.allowance_upper_bounds.tolist(),
            "nonnegative": True,
            "ordered_prices": True,
            "fixed_zero_payg": True,
        },
        "solver": {
            "method": "SLSQP",
            "enable_recovery": settings.solver().enable_recovery,
            "numerical_constraint_recovery": settings.solver().numerical_constraint_recovery,
            "options": dict(settings.solver().options),
            "nlp_count_definition": "actual backend minimize calls, including retries",
            "generated_constraints_definition": "new separation cuts after initialization",
        },
        "initial_menu": _menu(initial),
        "initial_response": _response_record(initial_response),
        "initial_menu_selection": {
            "name": settings.trajectory_start,
            "purpose": "demonstrate a production optimistic switch at a traveler cost tie"
            if settings.trajectory_start == "switch"
            else "original baseline reference",
            "baseline_menu": _menu(baseline_menu),
            "changed": "experiment starting menu only; no model or transition-rule changes",
        },
        "baseline_reference": baseline_reference,
        "regional_search_cg": cg,
        "regional_search_all_constraints": all_constraints,
        "enumeration": enumeration,
        "comparison": comparison,
        "cg_vs_all": agreement,
        "benchmark_gap": {
            "absolute": gap,
            "relative": None if gap is None else gap / max(1.0, abs(enum_profit)),
            "matches_best_found": gap is not None and abs(gap) <= settings.profit_tolerance,
        },
        "numerical_validity": validity,
        "timing": {
            "total_cpu_seconds": process_time() - total_cpu,
            "total_wall_seconds": perf_counter() - total_wall,
            "excludes": "file export and plotting",
        },
    }


def terminal_summary(result: dict) -> dict:
    cg, enum = result["regional_search_cg"], result["enumeration"]
    return {
        "regional_search": {
            "initial_assignment": result["initial_response"]["assignment"],
            "final_assignment": cg["final_response"]["assignment"],
            "initial_profit": result["initial_response"]["true_profit"],
            "final_profit": cg["final_response"]["true_profit"],
            "final_menu": cg["final_menu"],
            **{
                key: cg[key]
                for key in (
                    "status",
                    "region_iterations",
                    "distinct_assignments_visited",
                    "nlp_solves",
                    "accepted_assignment_transitions",
                    "rejected_iterations",
                    "assignment_revisited",
                    "cpu_time_seconds",
                )
            },
        },
        "enumeration": {
            "assignments_enumerated": enum["assignment_count"],
            "starts_per_assignment": enum["starts_per_assignment"],
            "best_assignment_found": None
            if enum["best"] is None
            else enum["best"]["optimistic_assignment"],
            "best_profit_found": None if enum["best"] is None else enum["best"]["true_profit"],
            "successful_starts": enum["successful_starts"],
            "failed_starts": enum["failed_starts"],
            "cpu_time_seconds": enum["cpu_time_seconds"],
            "gap_to_regional_search": result["benchmark_gap"],
            "interpretation": enum["interpretation"],
        },
        "cg_vs_all_constraints": result["cg_vs_all"],
        "numerical_validity": result["numerical_validity"],
    }


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def save_stylized_regional_results(result: dict, output_dir: str | Path) -> None:
    """Write exactly one two-format trajectory, three tables and the complete JSON."""
    from maas_bundle.benchmarks.stylized_regional_plots import plot_regional_trajectory

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "a2_full_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _csv(
        output / "a2_iteration_table.csv",
        [
            {
                "Iter.": row["iteration"],
                "Assignment": row["starting_assignment"],
                "Profit before solve": row["profit_before_solve"],
                "Regional optimum profit": row["regional_profit"],
                "Next assignment": row["response"]["assignment"],
                "Switching classes": "; ".join(
                    s["class"] + ": " + s["from_name"] + " -> " + s["to_name"]
                    for s in row["switches"]
                ),
                "Max violation": row["validation"]["maximum_violation"],
                "NLP status": row["solver"]["status"],
                "Accepted": row["accepted"],
                "Active constraints": row["active_constraints"],
                "Generated constraints": row["generated_constraints"],
                "Same-menu reassignment gain": row["same_menu_reassignment_profit_gain"],
            }
            for row in result["regional_search_cg"]["iterations"]
        ],
    )
    _csv(
        output / "a2_assignment_enumeration.csv",
        [
            {
                "Assignment": row["assignment"],
                "Best regional objective": row["best_regional_profit"],
                "Best feasible menu": row["best_feasible_menu"],
                "Optimistic assignment at regional best": None
                if row["best"] is None
                else row["best"]["optimistic_assignment"],
                "True profit at regional best": None
                if row["best"] is None
                else row["best"]["true_profit"],
                "Status": row["status"],
                "Solver status": row["solver_status"],
                "Best start solver status": None
                if row["best"] is None
                else row["best"]["solver"]["status"],
                "Best start KKT": None
                if row["best"] is None
                else row["best"]["solver"]["kkt_satisfied"],
                "Max choice violation": None
                if row["best"] is None
                else row["best"]["validation"]["choice_violation"],
                "Starts": row["starts"],
                "Successful starts": row["successful_starts"],
                "Failed starts": row["failed_starts"],
                "Feasible starts": row["feasible_starts"],
                "NLP solves": row["nlp_solves"],
                "CPU time": row["cpu_time_seconds"],
            }
            for row in result["enumeration"]["records"]
        ],
    )
    _csv(output / "a2_algorithm_comparison.csv", result["comparison"])
    plot_regional_trajectory(result, output)
