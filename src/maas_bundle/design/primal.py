"""Critical-traveler generation and boundary assignment search."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from maas_bundle.design.assignment import AssignmentResult
from maas_bundle.design.boundary import (
    ChoiceConstraintIndex,
    CriticalConstraintSet,
    SeparationResult,
    initialize_critical_constraints,
    separate_choice_constraints,
)
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import (
    RegionalNlpSolver,
    RegionalSolveResult,
    RestrictedRegionalProblem,
    backtrack_to_feasible_menu,
)
from maas_bundle.design.tolerances import (
    DEFAULT_COST_TOLERANCE,
    DEFAULT_MARGIN_TOLERANCE,
    DEFAULT_PROFIT_TOLERANCE,
    effective_cost_tolerance,
    validate_tolerance,
)
from maas_bundle.design.validation import CandidateDiagnostics, validate_candidate


class FixedAssignmentStatus(StrEnum):
    """Termination status of fixed-assignment constraint generation."""

    CONVERGED = "converged"
    SOLVER_FAILED = "solver_failed"
    ITERATION_LIMIT = "iteration_limit"
    STALLED = "stalled"


@dataclass(frozen=True)
class CriticalGenerationIteration:
    """Immutable diagnostics for one solve-and-separate iteration."""

    number: int
    active_constraint_count: int
    solve_result: RegionalSolveResult
    separation: SeparationResult | None
    added_constraints: tuple[ChoiceConstraintIndex, ...] = ()


@dataclass(frozen=True)
class FixedAssignmentResult:
    """Outcome of critical-constraint generation for one fixed assignment."""

    status: FixedAssignmentStatus
    assigned_bundles: NDArray[np.int64]
    active_constraints: tuple[ChoiceConstraintIndex, ...]
    iterations: tuple[CriticalGenerationIteration, ...]
    solve_time_seconds: float
    feasibility_tolerance: float = DEFAULT_COST_TOLERANCE

    def __post_init__(self) -> None:
        """Protect the fixed assignment and validate the recorded outcome."""

        validate_tolerance(self.feasibility_tolerance, name="feasibility_tolerance")
        assignments = np.asarray(self.assigned_bundles)
        if assignments.ndim != 1 or not np.issubdtype(assignments.dtype, np.integer):
            raise ValueError("assigned_bundles must be a one-dimensional integer array")
        if not self.iterations:
            raise ValueError("a fixed-assignment result must record at least one iteration")
        elapsed = float(self.solve_time_seconds)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("solve_time_seconds must be finite and nonnegative")

        immutable_assignments = assignments.astype(np.int64, copy=True)
        immutable_assignments.setflags(write=False)
        object.__setattr__(self, "assigned_bundles", immutable_assignments)
        object.__setattr__(self, "active_constraints", tuple(self.active_constraints))
        object.__setattr__(self, "iterations", tuple(self.iterations))
        object.__setattr__(self, "solve_time_seconds", elapsed)

    @property
    def converged(self) -> bool:
        """Return whether full traveler separation found no violation."""

        return self.status is FixedAssignmentStatus.CONVERGED

    @property
    def menu(self) -> BundleMenu:
        """Return the final menu produced by the regional solver."""

        return self.iterations[-1].solve_result.menu

    @property
    def final_solve_result(self) -> RegionalSolveResult:
        """Return the final regional NLP result."""

        return self.iterations[-1].solve_result

    @property
    def iteration_count(self) -> int:
        """Return the number of solve-and-separate generation iterations."""

        return len(self.iterations)

    @property
    def nlp_solve_count(self) -> int:
        """Return the number of backend attempts including solver recovery."""

        return sum(iteration.solve_result.attempt_count for iteration in self.iterations)

    @property
    def added_constraint_count(self) -> int:
        """Return the total number of constraints generated after initialization."""

        return sum(len(iteration.added_constraints) for iteration in self.iterations)


class AssignmentConsistencyStatus(StrEnum):
    """Termination status of the outer assignment-consistency search."""

    CONSISTENT = "consistent"
    FIXED_ASSIGNMENT_FAILED = "fixed_assignment_failed"
    REGION_ITERATION_LIMIT = "region_iteration_limit"
    ASSIGNMENT_REVISIT = "assignment_revisit"
    PROFIT_DECREASE = "profit_decrease"


@dataclass(frozen=True)
class AssignmentRegionIteration:
    """One fixed-assignment solve followed by numerical optimistic reassignment."""

    number: int
    assumed_assignment: NDArray[np.int64]
    fixed_assignment_result: FixedAssignmentResult
    true_assignment: AssignmentResult | None
    revisited_assignment: bool = False
    accepted: bool = True
    boundary_origin: int | None = None
    boundary_returned: bool = False

    def __post_init__(self) -> None:
        """Validate and protect the assignment used to define this region."""

        assumed = np.asarray(self.assumed_assignment)
        if assumed.ndim != 1 or not np.issubdtype(assumed.dtype, np.integer):
            raise ValueError("assumed_assignment must be a one-dimensional integer array")
        if self.number <= 0:
            raise ValueError("region iteration number must be positive")
        if not np.array_equal(assumed, self.fixed_assignment_result.assigned_bundles):
            raise ValueError("region assignment must match the fixed-assignment result")
        if self.true_assignment is not None and self.true_assignment.bundle_indices.shape != (
            assumed.size,
        ):
            raise ValueError("true assignment must use the same travelers")
        if self.true_assignment is not None:
            fixed_menu = self.fixed_assignment_result.menu
            if not (
                np.array_equal(self.true_assignment.menu.prices, fixed_menu.prices)
                and np.array_equal(
                    self.true_assignment.menu.allowances,
                    fixed_menu.allowances,
                )
            ):
                raise ValueError("true assignment must be evaluated at the fixed-result menu")

        immutable_assumed = assumed.astype(np.int64, copy=True)
        immutable_assumed.setflags(write=False)
        object.__setattr__(self, "assumed_assignment", immutable_assumed)

    @property
    def changed_traveler_count(self) -> int | None:
        """Return how many travelers change bundle, or None after an inner failure."""

        if self.true_assignment is None:
            return None
        return int(np.count_nonzero(self.true_assignment.bundle_indices != self.assumed_assignment))

    @property
    def regional_objective(self) -> float:
        """Return the objective evaluated under the assumed assignment."""

        return self.fixed_assignment_result.final_solve_result.objective_value

    @property
    def true_profit(self) -> float | None:
        """Return optimistic true profit at the candidate menu when available."""

        return None if self.true_assignment is None else self.true_assignment.total_profit


@dataclass(frozen=True)
class AssignmentConsistencyResult:
    """Candidate or diagnostic returned by the outer assignment-region search."""

    status: AssignmentConsistencyStatus
    initial_assignment: AssignmentResult
    iterations: tuple[AssignmentRegionIteration, ...]
    solve_time_seconds: float
    retained_assignment: AssignmentResult | None = None
    terminal_region_number: int | None = None
    boundary_search_complete: bool = False

    def __post_init__(self) -> None:
        """Freeze iteration history and validate elapsed time."""

        if not self.iterations:
            raise ValueError("an assignment-consistency result requires an iteration")
        elapsed = float(self.solve_time_seconds)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("solve_time_seconds must be finite and nonnegative")
        object.__setattr__(self, "iterations", tuple(self.iterations))
        object.__setattr__(self, "solve_time_seconds", elapsed)
        if self.terminal_region_number is not None and not (
            1 <= self.terminal_region_number <= len(self.iterations)
        ):
            raise ValueError("terminal_region_number must identify a recorded region")

    @property
    def terminal_iteration(self) -> AssignmentRegionIteration:
        """Return the solved region supporting the retained terminal boundary."""

        if self.terminal_region_number is not None:
            return self.iterations[self.terminal_region_number - 1]
        return self.iterations[-1]

    @property
    def consistent(self) -> bool:
        """Return whether the final menu and optimistic assignment agree."""

        return self.status is AssignmentConsistencyStatus.CONSISTENT

    @property
    def menu(self) -> BundleMenu:
        """Return the accepted menu, retaining the incumbent after rejection."""

        if self.retained_assignment is not None:
            return self.retained_assignment.menu
        final_iteration = self.iterations[-1]
        if final_iteration.true_assignment is not None:
            return final_iteration.true_assignment.menu
        return final_iteration.fixed_assignment_result.menu

    @property
    def assignment(self) -> NDArray[np.int64]:
        """Return the retained response or the latest candidate assignment."""

        if self.retained_assignment is not None:
            return self.retained_assignment.bundle_indices
        final_iteration = self.iterations[-1]
        if final_iteration.true_assignment is not None:
            return final_iteration.true_assignment.bundle_indices
        return final_iteration.assumed_assignment

    @property
    def true_profit(self) -> float | None:
        """Return accepted profit; rejected trial profits stay in the history."""

        if self.retained_assignment is not None:
            return self.retained_assignment.total_profit
        return self.iterations[-1].true_profit

    @property
    def region_iteration_count(self) -> int:
        """Return the number of visited fixed-assignment regions."""

        return len(self.iterations)

    @property
    def assignment_change_count(self) -> int:
        """Return the number of region transitions performed or diagnosed."""

        return sum(
            iteration.changed_traveler_count != 0
            for iteration in self.iterations
            if iteration.accepted and iteration.changed_traveler_count is not None
        )

    @property
    def fixed_generation_iteration_count(self) -> int:
        """Return total inner generation iterations over every visited region."""

        return sum(
            iteration.fixed_assignment_result.iteration_count for iteration in self.iterations
        )

    @property
    def nlp_solve_count(self) -> int:
        """Return all backend attempts across regions and generation iterations."""

        return sum(
            iteration.fixed_assignment_result.nlp_solve_count for iteration in self.iterations
        )

    def metrics(self) -> dict[str, str | int | float | None]:
        """Return a terminal-report-friendly outer-loop record."""

        return {
            "status": self.status.value,
            "regions": self.region_iteration_count,
            "assignment_changes": self.assignment_change_count,
            "assignment_revisits": sum(
                iteration.revisited_assignment for iteration in self.iterations
            ),
            "changed_travelers": sum(
                iteration.changed_traveler_count or 0 for iteration in self.iterations
            ),
            "generation_iterations": self.fixed_generation_iteration_count,
            "nlp_solves": self.nlp_solve_count,
            "regional_objective": self.terminal_iteration.regional_objective,
            "boundary_trials": sum(i.boundary_origin is not None for i in self.iterations),
            "boundary_returns": sum(i.boundary_returned for i in self.iterations),
            "boundary_search_complete": self.boundary_search_complete,
            "true_profit": self.true_profit,
            "time_seconds": self.solve_time_seconds,
        }


def solve_fixed_assignment(
    problem: BundleDesignProblem,
    initial_menu: BundleMenu,
    assigned_bundles: ArrayLike,
    solver: RegionalNlpSolver,
    *,
    initial_per_pair: int = 1,
    tie_tolerance: float = 1e-12,
    violation_tolerance: float = DEFAULT_COST_TOLERANCE,
    max_iterations: int | None = None,
) -> FixedAssignmentResult:
    """Solve one fixed-assignment region by finite constraint generation.

    Each nonterminal iteration adds at most one most-violated omitted traveler
    constraint per occupied ordered bundle pair. Existing constraints are never
    removed. Without an explicit iteration limit, finiteness follows from the
    finite set of ``traveler_count * (bundle_count - 1)`` possible constraints.
    """

    if max_iterations is not None and (
        not isinstance(max_iterations, (int, np.integer)) or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer or None")

    violation_tolerance = effective_cost_tolerance(
        solver, choice_tolerance=violation_tolerance, violation_tolerance=violation_tolerance
    )
    initial_valuation = problem.evaluate_valuations(initial_menu)
    critical = initialize_critical_constraints(
        initial_menu,
        initial_valuation,
        assigned_bundles,
        initial_per_pair=initial_per_pair,
        tie_tolerance=tie_tolerance,
    )
    initial_regional = RestrictedRegionalProblem(problem, critical)
    initial_separation = separate_choice_constraints(
        initial_menu,
        initial_valuation,
        critical,
        violation_tolerance=violation_tolerance,
    )
    initial_vector = initial_regional.layout.pack(initial_menu)
    feasible_anchor = (
        initial_menu
        if initial_separation.is_feasible
        and initial_regional.maximum_constraint_violation(initial_vector) <= violation_tolerance
        else None
    )
    natural_limit = (
        problem.traveler_count * (problem.bundle_count - 1) - critical.constraint_count + 1
    )
    iteration_limit = natural_limit if max_iterations is None else int(max_iterations)

    menu = initial_menu
    history: list[CriticalGenerationIteration] = []
    started = perf_counter()

    for number in range(1, iteration_limit + 1):
        active_count = critical.constraint_count
        regional = RestrictedRegionalProblem(problem, critical)
        solve_result = solver.solve(regional, menu)

        if (
            not solve_result.success
            or solve_result.maximum_constraint_violation > violation_tolerance
        ):
            history.append(
                CriticalGenerationIteration(
                    number=number,
                    active_constraint_count=active_count,
                    solve_result=solve_result,
                    separation=None,
                )
            )
            return _fixed_assignment_result(
                FixedAssignmentStatus.SOLVER_FAILED,
                critical,
                history,
                started,
                violation_tolerance,
            )

        # Retain a certified full-region anchor if the restricted CG solves
        # introduce only a roundoff loss. The most recent NLP entry may itself
        # violate omitted constraints, so its warm-start guard cannot do this.
        if feasible_anchor is not None:
            anchor_vector = regional.layout.pack(feasible_anchor)
            anchor_profit = regional.profit(anchor_vector)
            loss = anchor_profit - solve_result.objective_value
            if 0 < loss <= getattr(solver, "objective_tolerance", DEFAULT_PROFIT_TOLERANCE):
                certificate = regional.first_order_certificate(
                    anchor_vector,
                    feasibility_tolerance=violation_tolerance,
                    stationarity_tolerance=getattr(solver, "optimality_tolerance", 1e-5),
                    active_tolerance=getattr(solver, "active_set_tolerance", 1e-4),
                    check_stationarity=getattr(solver, "check_kkt", True),
                )
                accepted = (
                    certificate.satisfies_kkt
                    if certificate.stationarity_checked
                    else certificate.primal_violation <= violation_tolerance
                )
                if accepted:
                    solve_result = replace(
                        solve_result,
                        menu=feasible_anchor,
                        objective_value=anchor_profit,
                        maximum_constraint_violation=certificate.primal_violation,
                        first_order_certificate=certificate,
                        message=solve_result.message
                        + (
                            "; retained certified CG anchor after roundoff loss"
                            if certificate.stationarity_checked
                            else "; retained feasible CG anchor after roundoff loss (KKT unchecked)"
                        ),
                    )
        menu = solve_result.menu
        valuation = problem.evaluate_valuations(menu)
        separation = separate_choice_constraints(
            menu,
            valuation,
            critical,
            violation_tolerance=violation_tolerance,
        )
        additions = tuple(violation.index for violation in separation.additions)
        added_count = critical.add_all(additions)
        history.append(
            CriticalGenerationIteration(
                number=number,
                active_constraint_count=active_count,
                solve_result=solve_result,
                separation=separation,
                added_constraints=additions,
            )
        )

        if separation.is_feasible:
            return _fixed_assignment_result(
                FixedAssignmentStatus.CONVERGED,
                critical,
                history,
                started,
                violation_tolerance,
            )
        if not separation.can_add_constraints or added_count == 0:
            return _fixed_assignment_result(
                FixedAssignmentStatus.STALLED,
                critical,
                history,
                started,
                violation_tolerance,
            )
        if number == iteration_limit:
            return _fixed_assignment_result(
                FixedAssignmentStatus.ITERATION_LIMIT,
                critical,
                history,
                started,
                violation_tolerance,
            )

        if feasible_anchor is not None:
            expanded_regional = RestrictedRegionalProblem(problem, critical)
            menu = backtrack_to_feasible_menu(
                expanded_regional,
                menu,
                feasible_anchor,
                tolerance=violation_tolerance,
            )

    raise AssertionError(  # pragma: no cover - guarded by positive iteration_limit
        "the positive iteration limit must execute at least once"
    )


@dataclass
class _BoundaryFrame:
    """A solved menu and remaining equal-margin neighboring assignments."""

    assignment: AssignmentResult
    region_number: int
    neighbors: Iterator[NDArray[np.int64]]


def _boundary_neighbors(assignment: AssignmentResult) -> Iterator[NDArray[np.int64]]:
    for traveler, alternative in zip(
        *np.nonzero(assignment.numerical_optimistic_mask), strict=True
    ):
        selected = assignment.bundle_indices[traveler]
        if alternative == selected:
            continue
        neighbor = assignment.bundle_indices.copy()
        neighbor[traveler] = alternative
        neighbor.setflags(write=False)
        yield neighbor


def _same_menu(
    left: BundleMenu,
    right: BundleMenu,
    atol: float,
    rtol: float,
    allowance_atol: float = 1e-7,
) -> bool:
    """Symmetric componentwise absolute + relative menu comparison."""

    return all(
        np.all(np.abs(a - b) <= absolute + rtol * np.maximum(np.abs(a), np.abs(b)))
        for a, b, absolute in (
            (left.prices, right.prices, atol),
            (left.allowances, right.allowances, allowance_atol),
        )
    )


def _regional_checks_at_menu(
    problem: BundleDesignProblem,
    result: FixedAssignmentResult,
    menu: BundleMenu,
) -> bool:
    """Check a nearby solve at the actual retained point, not its own endpoint.

    This supplies the same feasibility/first-order guarantees as the NLP
    adapter; it does not upgrade them to a second-order certificate.
    """

    if np.array_equal(menu.prices, result.menu.prices) and np.array_equal(
        menu.allowances, result.menu.allowances
    ):
        return result.converged and result.final_solve_result.success
    certificate = result.final_solve_result.first_order_certificate
    critical = CriticalConstraintSet(result.assigned_bundles, problem.bundle_count)
    critical.add_all(result.active_constraints)
    valuation = problem.evaluate_valuations(menu)
    separation = separate_choice_constraints(
        menu,
        valuation,
        critical,
        violation_tolerance=result.feasibility_tolerance,
    )
    if not separation.is_feasible:
        return False
    regional = RestrictedRegionalProblem(problem, critical)
    checked = regional.first_order_certificate(
        regional.layout.pack(menu),
        feasibility_tolerance=result.feasibility_tolerance,
        stationarity_tolerance=certificate.stationarity_tolerance,
        active_tolerance=result.feasibility_tolerance,
        check_stationarity=certificate.stationarity_checked,
    )
    return (
        checked.satisfies_kkt
        if certificate.stationarity_checked
        else checked.primal_violation <= result.feasibility_tolerance
    )


def find_assignment_consistent_menu(
    problem: BundleDesignProblem,
    initial_menu: BundleMenu,
    solver: RegionalNlpSolver,
    *,
    choice_tolerance: float = DEFAULT_COST_TOLERANCE,
    profit_tolerance: float = DEFAULT_PROFIT_TOLERANCE,
    margin_tolerance: float = DEFAULT_MARGIN_TOLERANCE,
    max_region_iterations: int = 100,
    stop_on_assignment_revisit: bool = False,
    initial_per_pair: int = 1,
    critical_tie_tolerance: float = 1e-12,
    violation_tolerance: float = DEFAULT_COST_TOLERANCE,
    max_generation_iterations: int | None = None,
    boundary_tolerance: float = 1e-7,
    boundary_relative_tolerance: float = 1e-6,
    allowance_boundary_tolerance: float = 1e-7,
) -> AssignmentConsistencyResult:
    """Follow optimistic transitions and check equal-margin neighboring regions.

    Returning to an assignment at a different menu always requires a solve at
    that new menu, even with unchanged profit. A same-point trial is complete
    only when its regional checks also hold at the saved anchor. Menu matching
    is independent of tie type; only selection of extra neighbors uses margins.
    """

    if not isinstance(max_region_iterations, (int, np.integer)) or max_region_iterations <= 0:
        raise ValueError("max_region_iterations must be a positive integer")
    if not isinstance(stop_on_assignment_revisit, (bool, np.bool_)):
        raise ValueError("stop_on_assignment_revisit must be Boolean")
    for name, value in (
        ("boundary_tolerance", boundary_tolerance),
        ("boundary_relative_tolerance", boundary_relative_tolerance),
        ("allowance_boundary_tolerance", allowance_boundary_tolerance),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    choice_tolerance = violation_tolerance = effective_cost_tolerance(
        solver, choice_tolerance=choice_tolerance, violation_tolerance=violation_tolerance
    )
    validate_tolerance(profit_tolerance, name="profit_tolerance")
    validate_tolerance(margin_tolerance, name="margin_tolerance")
    started = perf_counter()
    initial_assignment = problem.evaluate_assignment(
        initial_menu,
        choice_tolerance=choice_tolerance,
        profit_tolerance=margin_tolerance,
    )
    if np.any(initial_menu.prices[1:] > getattr(problem, "price_upper_bounds", np.inf)) or np.any(
        initial_menu.allowances[1:] > getattr(problem, "allowance_upper_bounds", np.inf)
    ):
        raise ValueError("initial menu must satisfy the problem design upper bounds")
    incumbent = initial_assignment
    menu = initial_menu
    assumed_assignment = initial_assignment.bundle_indices
    visited_assignments = {_assignment_signature(assumed_assignment)}
    visited_menus = {_assignment_signature(assumed_assignment): initial_menu}
    history: list[AssignmentRegionIteration] = []
    frames: list[_BoundaryFrame] = []

    def finish(status, *, terminal=None, complete=False):
        return AssignmentConsistencyResult(
            status=status,
            initial_assignment=initial_assignment,
            iterations=tuple(history),
            solve_time_seconds=perf_counter() - started,
            retained_assignment=incumbent,
            terminal_region_number=terminal,
            boundary_search_complete=complete,
        )

    for number in range(1, int(max_region_iterations) + 1):
        origin = frames[-1].region_number if frames else None
        # An exploratory assignment uses its own entry objective. The saved
        # incumbent is never overwritten by a lower-profit numerical trial.
        entry_profit = None
        if frames:
            entry_profit = float(
                problem.traveler_weights @ problem.bundle_margins(menu)[assumed_assignment]
            )
        fixed_result = solve_fixed_assignment(
            problem,
            menu,
            assumed_assignment,
            solver,
            initial_per_pair=initial_per_pair,
            tie_tolerance=critical_tie_tolerance,
            violation_tolerance=violation_tolerance,
            max_iterations=max_generation_iterations,
        )
        if not fixed_result.converged:
            history.append(
                AssignmentRegionIteration(
                    number,
                    assumed_assignment,
                    fixed_result,
                    None,
                    accepted=False,
                    boundary_origin=origin,
                )
            )
            return finish(AssignmentConsistencyStatus.FIXED_ASSIGNMENT_FAILED)

        response = problem.evaluate_assignment(
            fixed_result.menu,
            choice_tolerance=choice_tolerance,
            profit_tolerance=margin_tolerance,
        )
        objective = fixed_result.final_solve_result.objective_value
        if frames:
            # Numerical tolerance only applies to exploratory regional solves;
            # incumbent profit is never decreased, even by a tiny amount.
            decreased = objective < entry_profit - profit_tolerance
        else:
            decreased = (
                objective < incumbent.total_profit or response.total_profit < incumbent.total_profit
            )
        consistent = np.array_equal(response.bundle_indices, assumed_assignment)
        signature = _assignment_signature(response.bundle_indices)
        revisit = not consistent and signature in visited_assignments
        improved = response.total_profit > incumbent.total_profit + profit_tolerance
        same_point = bool(frames) and _same_menu(
            response.menu,
            frames[-1].assignment.menu,
            boundary_tolerance,
            boundary_relative_tolerance,
            allowance_boundary_tolerance,
        )
        returned = False
        if same_point and not improved and not decreased:
            anchor = frames[-1].assignment
            # Both values must agree to numerical profit tolerance as well;
            # nearby coordinates alone never erase a material objective change.
            if (
                abs(response.total_profit - anchor.total_profit) <= profit_tolerance
                and abs(objective - entry_profit) <= profit_tolerance
            ):
                returned = _regional_checks_at_menu(problem, fixed_result, anchor.menu)
        history.append(
            AssignmentRegionIteration(
                number,
                assumed_assignment,
                fixed_result,
                response,
                revisited_assignment=revisit,
                accepted=not decreased
                and not returned
                and response.total_profit >= incumbent.total_profit,
                boundary_origin=origin,
                boundary_returned=returned,
            )
        )
        if decreased:
            return finish(AssignmentConsistencyStatus.PROFIT_DECREASE)
        if returned:
            # Keep the exact anchor. Its original region was solved there,
            # and this neighbor has now passed its checks at that same point.
            pass
        else:
            # A new endpoint invalidates all pending checks at the old point,
            # including equal-profit moves and returns to an old assignment.
            frames.clear()
            if response.total_profit >= incumbent.total_profit:
                incumbent = response
            else:
                return finish(AssignmentConsistencyStatus.PROFIT_DECREASE)
            if not consistent:
                if (
                    revisit
                    and stop_on_assignment_revisit
                    and origin is None
                    and not improved
                    and _same_menu(
                        response.menu,
                        visited_menus[signature],
                        boundary_tolerance,
                        boundary_relative_tolerance,
                        allowance_boundary_tolerance,
                    )
                ):
                    return finish(AssignmentConsistencyStatus.ASSIGNMENT_REVISIT)
                visited_assignments.add(signature)
                visited_menus[signature] = response.menu
                menu = fixed_result.menu
                assumed_assignment = response.bundle_indices
                continue
            frames.append(_BoundaryFrame(response, number, _boundary_neighbors(response)))

        frame = frames[-1]
        neighbor = next(frame.neighbors, None)
        if neighbor is None:
            return finish(
                AssignmentConsistencyStatus.CONSISTENT,
                terminal=frame.region_number,
                complete=True,
            )
        menu = frame.assignment.menu
        assumed_assignment = neighbor
        visited_assignments.add(_assignment_signature(neighbor))
        visited_menus[_assignment_signature(neighbor)] = menu

    return finish(AssignmentConsistencyStatus.REGION_ITERATION_LIMIT)


def solve_primal(
    problem: BundleDesignProblem,
    initial_menu: BundleMenu,
    solver: RegionalNlpSolver,
    *,
    choice_tolerance: float = DEFAULT_COST_TOLERANCE,
    profit_tolerance: float = DEFAULT_PROFIT_TOLERANCE,
    margin_tolerance: float = DEFAULT_MARGIN_TOLERANCE,
    max_region_iterations: int = 100,
    stop_on_assignment_revisit: bool = False,
    initial_per_pair: int = 1,
    critical_tie_tolerance: float = 1e-12,
    violation_tolerance: float = DEFAULT_COST_TOLERANCE,
    max_generation_iterations: int | None = None,
    boundary_tolerance: float = 1e-7,
    boundary_relative_tolerance: float = 1e-6,
    allowance_boundary_tolerance: float = 1e-7,
) -> CandidateDiagnostics:
    """Search from a profitable menu and validate the resulting candidate.

    Positive initial profit excludes all-PAYG plateaus. The outer search
    retains its incumbent on failure or profit decrease, so unsuccessful
    trials cannot replace the positive-profit menu returned to the caller.
    """

    choice_tolerance = violation_tolerance = effective_cost_tolerance(
        solver, choice_tolerance=choice_tolerance, violation_tolerance=violation_tolerance
    )
    validate_tolerance(profit_tolerance, name="profit_tolerance")
    validate_tolerance(margin_tolerance, name="margin_tolerance")
    if np.any(np.diff(initial_menu.prices) < 0):
        raise ValueError("initial menu prices must be nondecreasing")
    initial_assignment = problem.evaluate_assignment(
        initial_menu,
        choice_tolerance=choice_tolerance,
        profit_tolerance=margin_tolerance,
    )
    if initial_assignment.total_profit <= profit_tolerance:
        raise ValueError(
            "initial menu must have positive optimistic profit above profit_tolerance; "
            "all-PAYG, zero-profit and loss-making starts are not allowed"
        )

    candidate = find_assignment_consistent_menu(
        problem,
        initial_menu,
        solver,
        choice_tolerance=choice_tolerance,
        profit_tolerance=profit_tolerance,
        margin_tolerance=margin_tolerance,
        max_region_iterations=max_region_iterations,
        stop_on_assignment_revisit=stop_on_assignment_revisit,
        initial_per_pair=initial_per_pair,
        critical_tie_tolerance=critical_tie_tolerance,
        violation_tolerance=violation_tolerance,
        max_generation_iterations=max_generation_iterations,
        boundary_tolerance=boundary_tolerance,
        boundary_relative_tolerance=boundary_relative_tolerance,
        allowance_boundary_tolerance=allowance_boundary_tolerance,
    )
    return validate_candidate(candidate)


def _fixed_assignment_result(
    status: FixedAssignmentStatus,
    critical: CriticalConstraintSet,
    history: list[CriticalGenerationIteration],
    started: float,
    feasibility_tolerance: float,
) -> FixedAssignmentResult:
    return FixedAssignmentResult(
        status=status,
        assigned_bundles=critical.assigned_bundles,
        active_constraints=critical.indices(),
        iterations=tuple(history),
        solve_time_seconds=perf_counter() - started,
        feasibility_tolerance=feasibility_tolerance,
    )


def _assignment_signature(assignment: NDArray[np.int64]) -> tuple[int, ...]:
    return tuple(int(bundle) for bundle in assignment)
