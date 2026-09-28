"""Tests for algorithm-level primal candidate validation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from maas_bundle import (
    AssignmentConsistencyResult,
    AssignmentConsistencyStatus,
    AssignmentRegionIteration,
    BundleDesignProblem,
    BundleMenu,
    CandidateValidationStatus,
    ChoiceConstraintIndex,
    CriticalGenerationIteration,
    FixedAssignmentResult,
    FixedAssignmentStatus,
    LinearProvisionCost,
    RegionalFirstOrderCertificate,
    RegionalSolveResult,
    RegionalSolveStatus,
    ScipyRegionalSolver,
    SeparationResult,
    ValuationTable,
    solve_primal,
    validate_candidate,
)


def _menu(price: float, allowance: float) -> BundleMenu:
    return BundleMenu(
        prices=np.array([0.0, price]),
        allowances=np.array([[0.0], [allowance]]),
    )


class _SmoothIntersectionProblem(BundleDesignProblem):
    """Two binding smooth constraints with a known regional solution."""

    bundle_count = 2
    mode_count = 1

    def __init__(self, traveler_count: int = 10) -> None:
        object.__setattr__(self, "traveler_weights", np.full(traveler_count, 1 / traveler_count))
        object.__setattr__(self, "provision_cost", LinearProvisionCost(np.array([1.0])))
        object.__setattr__(self, "route_oracles", ())
        object.__setattr__(self, "demand_distributions", ())

    @property
    def traveler_count(self) -> int:
        return self.traveler_weights.size

    def evaluate_valuations(self, menu: BundleMenu) -> ValuationTable:
        allowance = menu.allowances[1, 0]
        cap = np.full(self.traveler_count, 4.0)
        cap[0] = 1.0 + 2.0 * allowance - 0.5 * allowance**2 if allowance <= 2 else 3.0
        cap[1] = 1.5 + 0.25 * allowance
        values = np.empty((self.traveler_count, 2))
        values[:, 0] = 10.0
        values[:, 1] = 10.0 - cap
        omega = np.zeros((self.traveler_count, 2, 1))
        omega[0, 1, 0] = max(2.0 - allowance, 0.0)
        omega[1, 1, 0] = 0.25
        return ValuationTable(values, omega)


def _solver() -> ScipyRegionalSolver:
    return ScipyRegionalSolver("SLSQP", options={"maxiter": 500, "ftol": 1e-9})


def _manual_candidate(
    problem: BundleDesignProblem,
    menu: BundleMenu,
    *,
    outer_status: AssignmentConsistencyStatus = AssignmentConsistencyStatus.CONSISTENT,
    fixed_status: FixedAssignmentStatus = FixedAssignmentStatus.CONVERGED,
    solve_status: RegionalSolveStatus = RegionalSolveStatus.SUCCESS,
    full_separation: bool = True,
) -> AssignmentConsistencyResult:
    assignment = problem.evaluate_assignment(menu)
    assigned = assignment.bundle_indices
    indices = tuple(
        ChoiceConstraintIndex(traveler, int(bundle), 1 - int(bundle))
        for traveler, bundle in enumerate(assigned)
    )
    certificate = RegionalFirstOrderCertificate(
        0.0,
        0.0,
        0.0,
        0,
        0,
        1e-7,
        1e-6,
    )
    solve_result = RegionalSolveResult(
        solver_name="manual",
        status=solve_status,
        menu=menu,
        objective_value=0.0,
        solve_time_seconds=0.0,
        iteration_count=1,
        maximum_constraint_violation=0.0,
        optimality_measure=0.0,
        first_order_certificate=certificate,
        active_choice_constraint_count=len(indices),
        variable_count=2,
        message="manual",
    )
    separation = SeparationResult((), ()) if full_separation else None
    generation = CriticalGenerationIteration(1, len(indices), solve_result, separation)
    fixed = FixedAssignmentResult(fixed_status, assigned, indices, (generation,), 0.0)
    region = AssignmentRegionIteration(1, assigned, fixed, assignment)
    return AssignmentConsistencyResult(outer_status, assignment, (region,), 0.0)


def test_solve_primal_returns_valid_consistent_candidate(record_property) -> None:
    result = solve_primal(_SmoothIntersectionProblem(), _menu(0.5, 0.0), _solver())
    record_property("validation_metric", result.metrics())

    assert result.status is CandidateValidationStatus.CONSISTENT_CANDIDATE
    assert result.valid
    assert result.assignment_consistent
    assert result.full_separation_satisfied
    assert result.regional_solve_succeeded
    assert result.regional_kkt_satisfied
    assert result.active_user_tie_count == 2
    assert result.double_tie_count == 0
    assert result.candidate.boundary_search_complete
    allowance = (3.5 - np.sqrt(8.25)) / 2
    assert result.candidate.true_profit == pytest.approx(1.5 - 0.75 * allowance)


def test_validation_reports_numerical_double_tie() -> None:
    valid = solve_primal(_SmoothIntersectionProblem(), _menu(0.5, 0.0), _solver())
    candidate = valid.candidate
    final_iteration = candidate.terminal_iteration
    assert final_iteration.true_assignment is not None
    double_tie_assignment = replace(
        final_iteration.true_assignment,
        bundle_margins=np.zeros(2),
        total_profit=0.0,
    )
    double_tie_candidate = replace(
        candidate,
        iterations=tuple(
            replace(step, true_assignment=double_tie_assignment)
            if step.number == final_iteration.number
            else step
            for step in candidate.iterations
        ),
        boundary_search_complete=False,
    )

    result = validate_candidate(double_tie_candidate)

    assert result.status is CandidateValidationStatus.DOUBLE_TIE
    assert not result.valid
    assert result.double_tie_count == 2


def test_validation_requires_assignment_consistency() -> None:
    problem = _SmoothIntersectionProblem(2)
    candidate = _manual_candidate(
        problem,
        _menu(0.5, 0.0),
        outer_status=AssignmentConsistencyStatus.REGION_ITERATION_LIMIT,
    )

    result = validate_candidate(candidate)

    assert result.status is CandidateValidationStatus.NOT_ASSIGNMENT_CONSISTENT
    assert result.regional_solve_succeeded


def test_validation_requires_successful_regional_solve() -> None:
    problem = _SmoothIntersectionProblem(2)
    candidate = _manual_candidate(
        problem,
        _menu(0.5, 0.0),
        outer_status=AssignmentConsistencyStatus.FIXED_ASSIGNMENT_FAILED,
        fixed_status=FixedAssignmentStatus.SOLVER_FAILED,
        solve_status=RegionalSolveStatus.FAILED,
    )

    result = validate_candidate(candidate)

    assert result.status is CandidateValidationStatus.REGIONAL_SOLVE_FAILED
    assert not result.regional_solve_succeeded


def test_validation_requires_full_separation() -> None:
    problem = _SmoothIntersectionProblem(2)
    candidate = _manual_candidate(
        problem,
        _menu(0.5, 0.0),
        outer_status=AssignmentConsistencyStatus.FIXED_ASSIGNMENT_FAILED,
        fixed_status=FixedAssignmentStatus.ITERATION_LIMIT,
        full_separation=False,
    )

    result = validate_candidate(candidate)

    assert result.status is CandidateValidationStatus.SEPARATION_FAILED
    assert not result.full_separation_satisfied


def test_validation_rejects_negative_margin_tolerance() -> None:
    candidate = _manual_candidate(_SmoothIntersectionProblem(2), _menu(0.5, 0.0))

    with pytest.raises(ValueError, match="margin_tolerance"):
        validate_candidate(candidate, margin_tolerance=-1.0)
