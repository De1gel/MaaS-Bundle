"""Tests for fixed-assignment critical-constraint generation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import maas_bundle.design.primal as primal_module
from maas_bundle import (
    AssignmentConsistencyStatus,
    AssignmentRegionIteration,
    AssignmentResult,
    BundleDesignProblem,
    BundleMenu,
    ChoiceConstraintIndex,
    CriticalConstraintSet,
    CriticalGenerationIteration,
    FixedAssignmentResult,
    FixedAssignmentStatus,
    LinearProvisionCost,
    RegionalFirstOrderCertificate,
    RegionalSolveResult,
    RegionalSolveStatus,
    RestrictedRegionalProblem,
    ScipyRegionalSolver,
    SeparationResult,
    ValuationTable,
    aggregate_bundle_demand,
    backtrack_to_feasible_menu,
    choice_constraint_slacks,
    select_optimistic_bundles,
    solve_fixed_assignment,
)


class _SyntheticProblem(BundleDesignProblem):
    """Minimal problem double that supplies G and omega directly, without an oracle."""

    bundle_count = 2
    mode_count = 1

    def __init__(
        self,
        outside_values: list[float],
        subscription_intercepts: list[float],
        subscription_omega: list[float],
    ) -> None:
        outside = np.asarray(outside_values, dtype=float)
        object.__setattr__(self, "_outside_values", outside)
        object.__setattr__(
            self,
            "_subscription_intercepts",
            np.asarray(subscription_intercepts, dtype=float),
        )
        object.__setattr__(
            self,
            "_subscription_omega",
            np.asarray(subscription_omega, dtype=float),
        )
        object.__setattr__(self, "traveler_weights", np.full(outside.size, 1.0 / outside.size))
        object.__setattr__(self, "provision_cost", LinearProvisionCost(np.array([0.75])))
        object.__setattr__(self, "route_oracles", ())
        object.__setattr__(self, "demand_distributions", ())

    @property
    def traveler_count(self) -> int:
        return self._outside_values.size

    def evaluate_valuations(self, menu: BundleMenu) -> ValuationTable:
        allowance = menu.allowances[1, 0]
        route_values = np.column_stack(
            (
                self._outside_values,
                self._subscription_intercepts - self._subscription_omega * allowance,
            )
        )
        expected_omega = np.zeros((self.traveler_count, self.bundle_count, self.mode_count))
        expected_omega[:, 1, 0] = self._subscription_omega
        return ValuationTable(route_values, expected_omega)


class _GenerationOptimizationProblem(BundleDesignProblem):
    """Smooth known-optimum problem for generation tests, without route oracles."""

    bundle_count = 2
    mode_count = 1

    def __init__(self, traveler_count: int) -> None:
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

        route_values = np.empty((self.traveler_count, self.bundle_count))
        route_values[:, 0] = 10.0
        route_values[:, 1] = 10.0 - cap
        expected_omega = np.zeros((self.traveler_count, self.bundle_count, self.mode_count))
        expected_omega[0, 1, 0] = max(2.0 - allowance, 0.0)
        expected_omega[1, 1, 0] = 0.25
        return ValuationTable(route_values, expected_omega)


def _switching_problem() -> _SyntheticProblem:
    return _SyntheticProblem(
        outside_values=[6.0, 6.5],
        subscription_intercepts=[5.0, 5.0],
        subscription_omega=[1.0, 0.0],
    )


def _bounded_problem() -> _SyntheticProblem:
    return _SyntheticProblem(
        outside_values=[5.0, 4.0, 3.0],
        subscription_intercepts=[0.0, 0.0, 0.0],
        subscription_omega=[0.0, 0.0, 0.0],
    )


class _MenuSequenceSolver:
    """Deterministic adapter used to isolate generation-loop behavior."""

    name = "menu-sequence"

    def __init__(
        self,
        menus: list[BundleMenu],
        *,
        statuses: list[RegionalSolveStatus] | None = None,
    ) -> None:
        self._menus = menus
        self._statuses = statuses or [RegionalSolveStatus.SUCCESS] * len(menus)
        self.active_counts: list[int] = []
        self.warm_starts: list[BundleMenu] = []

    def solve(
        self,
        regional_problem: RestrictedRegionalProblem,
        initial_menu: BundleMenu,
    ) -> RegionalSolveResult:
        call = len(self.active_counts)
        menu = self._menus[call]
        status = self._statuses[call]
        self.active_counts.append(regional_problem.active_choice_constraint_count)
        self.warm_starts.append(initial_menu)
        certificate = RegionalFirstOrderCertificate(
            primal_violation=0.0,
            stationarity_residual=0.0,
            normalized_stationarity_residual=0.0,
            active_inequality_count=0,
            active_lower_bound_count=0,
            feasibility_tolerance=1e-7,
            stationarity_tolerance=1e-6,
        )
        return RegionalSolveResult(
            solver_name=self.name,
            status=status,
            menu=menu,
            objective_value=0.0,
            solve_time_seconds=0.0,
            iteration_count=1,
            maximum_constraint_violation=certificate.primal_violation,
            optimality_measure=None,
            first_order_certificate=certificate,
            active_choice_constraint_count=regional_problem.active_choice_constraint_count,
            variable_count=regional_problem.variable_count,
            message=status.value,
        )


def _menu(price: float, allowance: float) -> BundleMenu:
    return BundleMenu(
        prices=np.array([0.0, price]),
        allowances=np.array([[0.0], [allowance]]),
    )


def _assignment_snapshot(
    menu: BundleMenu,
    bundle_indices: np.ndarray,
    true_profit: float,
) -> AssignmentResult:
    traveler_count = bundle_indices.size
    costs = np.full((traveler_count, 2), 2.0)
    costs[np.arange(traveler_count), bundle_indices] = 1.0
    demand = np.bincount(bundle_indices, minlength=2).astype(float)
    return AssignmentResult(
        menu=menu,
        valuation=ValuationTable(
            np.zeros((traveler_count, 2)),
            np.zeros((traveler_count, 2, 1)),
        ),
        bundle_indices=bundle_indices,
        generalized_costs=costs,
        bundle_margins=np.zeros(2),
        bundle_demand=demand,
        total_profit=true_profit,
    )


class _AssignmentSequenceProblem(BundleDesignProblem):
    """Return prescribed optimistic assignments while recording evaluated menus."""

    bundle_count = 2
    mode_count = 1

    def __init__(self, states: list[tuple[np.ndarray, float]]) -> None:
        traveler_count = states[0][0].size
        object.__setattr__(self, "traveler_weights", np.ones(traveler_count))
        object.__setattr__(self, "provision_cost", LinearProvisionCost(np.array([1.0])))
        object.__setattr__(self, "route_oracles", ())
        object.__setattr__(self, "demand_distributions", ())
        object.__setattr__(self, "_states", states)
        object.__setattr__(self, "evaluated_menus", [])

    @property
    def traveler_count(self) -> int:
        return self.traveler_weights.size

    def evaluate_assignment(
        self,
        menu: BundleMenu,
        *,
        choice_tolerance: float = 1e-9,
        profit_tolerance: float = 1e-9,
    ) -> AssignmentResult:
        del choice_tolerance, profit_tolerance
        call = len(self.evaluated_menus)
        self.evaluated_menus.append(menu)
        indices, true_profit = self._states[call]
        return _assignment_snapshot(menu, indices, true_profit)


class _FixedAssignmentSequence:
    """Prescribe inner-loop menus/statuses and record warm starts and assignments."""

    def __init__(
        self,
        menus: list[BundleMenu],
        objectives: list[float],
        *,
        statuses: list[FixedAssignmentStatus] | None = None,
    ) -> None:
        self._menus = menus
        self._objectives = objectives
        self._statuses = statuses or [FixedAssignmentStatus.CONVERGED] * len(menus)
        self.warm_starts: list[BundleMenu] = []
        self.assignments: list[np.ndarray] = []

    def __call__(
        self,
        problem,
        initial_menu,
        assigned_bundles,
        solver,
        **kwargs,
    ) -> FixedAssignmentResult:
        del problem, solver, kwargs
        call = len(self.warm_starts)
        menu = self._menus[call]
        fixed_status = self._statuses[call]
        solve_status = (
            RegionalSolveStatus.SUCCESS
            if fixed_status is FixedAssignmentStatus.CONVERGED
            else RegionalSolveStatus.FAILED
        )
        assignments = np.asarray(assigned_bundles, dtype=np.int64)
        self.warm_starts.append(initial_menu)
        self.assignments.append(assignments.copy())
        certificate = RegionalFirstOrderCertificate(0.0, 0.0, 0.0, 0, 0, 1e-7, 1e-6)
        solve_result = RegionalSolveResult(
            solver_name="fixed-sequence",
            status=solve_status,
            menu=menu,
            objective_value=self._objectives[call],
            solve_time_seconds=0.0,
            iteration_count=1,
            maximum_constraint_violation=0.0,
            optimality_measure=0.0,
            first_order_certificate=certificate,
            active_choice_constraint_count=0,
            variable_count=2,
            message=solve_status.value,
        )
        generation_iteration = CriticalGenerationIteration(
            number=1,
            active_constraint_count=0,
            solve_result=solve_result,
            separation=SeparationResult((), ()) if solve_result.success else None,
        )
        return FixedAssignmentResult(
            status=fixed_status,
            assigned_bundles=assignments,
            active_constraints=(),
            iterations=(generation_iteration,),
            solve_time_seconds=0.0,
        )


class _BoundarySequenceProblem(BundleDesignProblem):
    """Prescribed economic responses with actual margins and weighted profits."""

    def __init__(self, responses):
        response = responses[0]
        object.__setattr__(self, "traveler_weights", np.ones(response.bundle_indices.size))
        object.__setattr__(self, "provision_cost", LinearProvisionCost(np.zeros(1)))
        object.__setattr__(self, "route_oracles", ((None,) * response.menu.bundle_count,))
        object.__setattr__(self, "_responses", iter(responses))

    def evaluate_assignment(self, menu, **kwargs):
        response = next(self._responses)
        np.testing.assert_array_equal(response.menu.prices, menu.prices)
        np.testing.assert_array_equal(response.menu.allowances, menu.allowances)
        return response


def _boundary_response(costs, *, prices=(0.0, 1.0, 1.0), allowance=0.0):
    costs = np.asarray(costs, dtype=float)
    prices = np.array(prices)
    menu = BundleMenu(prices, np.array([[0.0]] + [[allowance]] * (len(prices) - 1)))
    assignment = select_optimistic_bundles(costs, prices)
    demand = aggregate_bundle_demand(assignment, np.ones(len(costs)), len(prices))
    return AssignmentResult(
        menu=menu,
        valuation=ValuationTable(costs - prices, np.zeros((*costs.shape, 1))),
        bundle_indices=assignment,
        generalized_costs=costs,
        bundle_margins=prices,
        bundle_demand=demand,
        total_profit=float(demand @ prices),
    )


def _run_boundary_sequence(monkeypatch, responses, objectives, **kwargs):
    problem = _BoundarySequenceProblem(responses)
    fixed = _FixedAssignmentSequence([r.menu for r in responses[1:]], objectives)
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)
    result = primal_module.find_assignment_consistent_menu(
        problem,
        responses[0].menu,
        _MenuSequenceSolver([]),
        **kwargs,
    )
    return result, fixed


def test_ordinary_tie_does_not_force_a_lower_margin_assignment(monkeypatch):
    response = _boundary_response([[1, 1], [5, 1]], prices=(0.0, 1.0))
    result, fixed = _run_boundary_sequence(monkeypatch, [response] * 2, [2.0])
    assert result.consistent and result.boundary_search_complete
    assert result.true_profit == 2.0
    assert result.terminal_region_number == 1
    assert [a.tolist() for a in fixed.assignments] == [[1, 1]]
    assert primal_module.validate_candidate(result).valid


def test_multiple_double_ties_try_other_switch_after_return(monkeypatch):
    response = _boundary_response([[5, 1, 1], [5, 1, 1]])
    result, fixed = _run_boundary_sequence(monkeypatch, [response] * 4, [2.0] * 3)
    assert result.consistent and result.boundary_search_complete
    assert [a.tolist() for a in fixed.assignments] == [[1, 1], [2, 1], [1, 2]]
    assert all(i.boundary_returned for i in result.iterations[1:])
    assert all(menu is response.menu for menu in fixed.warm_starts)
    diagnostics = primal_module.validate_candidate(result)
    assert diagnostics.valid
    assert diagnostics.double_tie_count == 2


def test_roundoff_profit_gain_does_not_restart_boundary_search(monkeypatch):
    checked = []

    def check_at_anchor(problem, result, menu):
        checked.append(menu)
        return True

    monkeypatch.setattr(primal_module, "_regional_checks_at_menu", check_at_anchor)
    boundary = _boundary_response([[5, 1, 1], [5, 1, 5]])
    rounded = _boundary_response(
        [[5, 2, 2], [5, 2, 5]],
        prices=(0.0, 1.0 + 3e-12, 1.0 + 3e-12),
        allowance=1e-8,
    )
    result, fixed = _run_boundary_sequence(
        monkeypatch,
        [boundary, boundary, rounded],
        [2.0, rounded.total_profit],
    )
    assert result.consistent and result.boundary_search_complete
    assert result.iterations[-1].boundary_returned
    assert result.menu is boundary.menu
    assert result.true_profit == 2.0
    assert len(fixed.assignments) == 2
    assert checked == [boundary.menu]


def test_boundary_trial_that_improves_continues_at_new_assignment(monkeypatch):
    boundary = _boundary_response([[5, 1, 1], [5, 1, 5]])
    improved = _boundary_response([[5, 4, 2], [5, 1, 5]], prices=(0.0, 1.0, 2.0))
    result, fixed = _run_boundary_sequence(
        monkeypatch,
        [boundary, boundary, improved],
        [2.0, 3.0],
    )
    assert result.consistent
    assert result.true_profit == 3.0
    assert not result.iterations[-1].boundary_returned
    assert result.menu is improved.menu
    assert [a.tolist() for a in fixed.assignments] == [[1, 1], [2, 1]]


def test_equal_profit_return_at_different_menu_reoptimizes_assignment(monkeypatch):
    boundary = _boundary_response([[5, 1, 1], [5, 1, 5]])
    moved = _boundary_response([[5, 1, 2], [5, 1, 5]], allowance=0.5)
    result, fixed = _run_boundary_sequence(
        monkeypatch,
        [boundary, boundary, moved, moved],
        [2.0] * 3,
    )
    assert result.consistent
    assert not result.iterations[1].boundary_returned
    assert [a.tolist() for a in fixed.assignments] == [[1, 1], [2, 1], [1, 1]]
    assert result.menu is moved.menu
    assert result.terminal_region_number == 3
    assert fixed.warm_starts[2] is moved.menu


def test_improvement_on_same_assignment_reopens_boundary_checks(monkeypatch):
    boundary = _boundary_response([[5, 1, 1], [5, 1, 5]])
    improved = _boundary_response([[5, 2, 2], [5, 2, 5]], prices=(0.0, 2.0, 2.0))
    result, fixed = _run_boundary_sequence(
        monkeypatch,
        [boundary, boundary, improved, improved, improved],
        [2, 4, 4, 4],
    )
    assert result.consistent
    assert result.true_profit == 4.0
    assert not result.iterations[1].boundary_returned
    assert result.iterations[-1].boundary_returned
    assert [a.tolist() for a in fixed.assignments] == [[1, 1], [2, 1], [1, 1], [2, 1]]


def test_equal_profit_move_invalidates_checks_at_old_boundary(monkeypatch):
    boundary = _boundary_response([[5, 1, 1], [5, 1, 5]])
    moved = _boundary_response([[5, 1, 1], [5, 1, 5]], allowance=0.5)
    result, fixed = _run_boundary_sequence(
        monkeypatch,
        [boundary, boundary, moved, moved, moved],
        [2.0] * 4,
    )
    assert result.consistent and result.boundary_search_complete
    assert not result.iterations[1].boundary_returned
    assert result.iterations[-1].boundary_returned
    assert result.terminal_region_number == 3
    assert result.menu is moved.menu
    assert [a.tolist() for a in fixed.assignments] == [[1, 1], [2, 1], [1, 1], [2, 1]]
    assert fixed.warm_starts[-1] is moved.menu


def test_equal_profit_new_point_in_neighbor_keeps_new_menu(monkeypatch):
    boundary = _boundary_response([[5, 1, 1], [5, 1, 5]])
    moved = _boundary_response([[5, 2, 1], [5, 1, 5]], allowance=0.5)
    result, fixed = _run_boundary_sequence(
        monkeypatch,
        [boundary, boundary, moved],
        [2.0, 2.0],
    )
    assert result.consistent and result.boundary_search_complete
    assert not result.iterations[-1].boundary_returned
    assert result.iterations[-1].accepted
    assert result.menu is moved.menu
    assert result.terminal_region_number == 2
    assert [a.tolist() for a in fixed.assignments] == [[1, 1], [2, 1]]


def test_nearby_coordinates_without_anchor_checks_cannot_finish_trial(monkeypatch):
    boundary = _boundary_response([[5, 1, 1], [5, 1, 5]])
    moved = _boundary_response([[5, 1, 2], [5, 1, 5]], allowance=1e-8)
    monkeypatch.setattr(primal_module, "_regional_checks_at_menu", lambda *args: False)
    result, fixed = _run_boundary_sequence(
        monkeypatch,
        [boundary, boundary, moved, moved],
        [2.0] * 3,
    )
    assert result.consistent
    assert not result.iterations[1].boundary_returned
    assert len(fixed.assignments) == 3
    assert result.menu is moved.menu


def test_anchor_recheck_uses_feasibility_and_stationarity_at_that_menu():
    problem = _SyntheticProblem([1.0], [0.0], [0.0])
    result = solve_fixed_assignment(problem, _menu(0.5, 0.0), [1], ScipyRegionalSolver())
    assert result.converged
    assert primal_module._regional_checks_at_menu(problem, result, _menu(1.0 - 1e-8, 0.0))
    assert not primal_module._regional_checks_at_menu(problem, result, _menu(0.5, 0.0))
    assert not primal_module._regional_checks_at_menu(problem, result, _menu(1.1, 0.0))


def test_menu_matching_is_symmetric_and_scaled():
    left = _menu(10.0, 100.0)
    right = _menu(10.0 + 1e-6, 100.0 + 2e-6)
    assert primal_module._same_menu(left, right, 1e-7, 1e-6)
    assert primal_module._same_menu(right, left, 1e-7, 1e-6)
    assert not primal_module._same_menu(left, right, 1e-7, 0.0)


def test_boundary_limit_does_not_mark_untried_ties_resolved(monkeypatch):
    response = _boundary_response([[5, 1, 1], [5, 1, 1]])
    result, fixed = _run_boundary_sequence(
        monkeypatch,
        [response] * 3,
        [2.0] * 2,
        max_region_iterations=2,
    )
    assert result.status is AssignmentConsistencyStatus.REGION_ITERATION_LIMIT
    assert not result.boundary_search_complete
    assert len(fixed.assignments) == 2
    diagnostics = primal_module.validate_candidate(result)
    assert not diagnostics.valid
    assert diagnostics.status.value == "boundary_search_incomplete"


def test_failed_boundary_trial_keeps_positive_incumbent_and_remains_unresolved(monkeypatch):
    response = _boundary_response([[5, 1, 1], [5, 1, 5]])
    problem = _BoundarySequenceProblem([response, response])
    fixed = _FixedAssignmentSequence(
        [response.menu] * 2,
        [2, 0],
        statuses=[FixedAssignmentStatus.CONVERGED, FixedAssignmentStatus.SOLVER_FAILED],
    )
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)
    result = primal_module.find_assignment_consistent_menu(
        problem,
        response.menu,
        _MenuSequenceSolver([]),
    )
    assert result.status is AssignmentConsistencyStatus.FIXED_ASSIGNMENT_FAILED
    assert not result.boundary_search_complete
    assert result.true_profit == 2.0
    assert result.menu is response.menu
    assert not result.iterations[-1].accepted


@pytest.mark.parametrize("field", ["boundary_tolerance", "boundary_relative_tolerance"])
@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf])
def test_boundary_search_rejects_invalid_menu_tolerance(field, tolerance):
    with pytest.raises(ValueError, match=field):
        primal_module.find_assignment_consistent_menu(
            None,
            None,
            None,
            **{field: tolerance},
        )


def test_generation_adds_violated_traveler_monotonically_and_converges() -> None:
    """Converge in the finite two-constraint synthetic example.

    The constraints are ``C-epsilon <= 1`` and ``C <= 1.5``. Initialization
    selects the first one. The first candidate ``(2, 1)`` violates only the
    omitted second constraint; after adding it, ``(1.5, 0.5)`` satisfies both.
    """

    problem = _switching_problem()
    solver = _MenuSequenceSolver([_menu(2.0, 1.0), _menu(1.5, 0.5)])

    result = solve_fixed_assignment(problem, _menu(0.0, 0.0), [1, 1], solver)

    assert result.status is FixedAssignmentStatus.CONVERGED
    assert result.converged
    assert result.iteration_count == 2
    assert result.added_constraint_count == 1
    assert solver.active_counts == [1, 2]
    assert result.iterations[0].active_constraint_count == 1
    assert result.iterations[0].added_constraints == (ChoiceConstraintIndex(1, 1, 0),)
    first_separation = result.iterations[0].separation
    assert first_separation is not None
    assert first_separation.violations[0].slack == pytest.approx(-0.5)
    assert result.iterations[1].added_constraints == ()
    assert result.iterations[1].separation is not None
    assert result.iterations[1].separation.is_feasible
    assert result.active_constraints == (
        ChoiceConstraintIndex(0, 1, 0),
        ChoiceConstraintIndex(1, 1, 0),
    )
    assert result.menu is result.final_solve_result.menu
    assert result.solve_time_seconds >= 0
    assert not result.assigned_bundles.flags.writeable


def test_generation_backtracks_to_nearby_feasible_warm_start() -> None:
    problem = _switching_problem()
    initial_menu = _menu(0.0, 0.0)
    solver = _MenuSequenceSolver([_menu(2.0, 1.0), _menu(1.5, 0.5)])

    result = solve_fixed_assignment(problem, initial_menu, [1, 1], solver)

    assert result.converged
    assert solver.warm_starts[0] is initial_menu
    restored = solver.warm_starts[1]
    expanded = CriticalConstraintSet([1, 1], bundle_count=2)
    expanded.add_all(
        (
            ChoiceConstraintIndex(0, 1, 0),
            ChoiceConstraintIndex(1, 1, 0),
        )
    )
    regional = RestrictedRegionalProblem(problem, expanded)
    vector = regional.layout.pack(restored)
    assert regional.maximum_constraint_violation(vector) <= 1e-9
    assert restored.prices[1] == pytest.approx(1.4985, abs=1e-8)
    assert restored.allowances[1, 0] == pytest.approx(0.74925, abs=1e-8)


def test_backtracking_keeps_an_already_feasible_candidate() -> None:
    problem = _switching_problem()
    critical = CriticalConstraintSet([1, 1], bundle_count=2)
    critical.add_all(
        (
            ChoiceConstraintIndex(0, 1, 0),
            ChoiceConstraintIndex(1, 1, 0),
        )
    )
    regional = RestrictedRegionalProblem(problem, critical)
    candidate = _menu(1.0, 0.5)

    restored = backtrack_to_feasible_menu(
        regional,
        candidate,
        _menu(0.0, 0.0),
        tolerance=1e-9,
    )

    assert restored is candidate


def test_backtracking_keeps_candidate_when_anchor_is_not_feasible() -> None:
    problem = _switching_problem()
    critical = CriticalConstraintSet([1, 1], bundle_count=2)
    critical.add_all(
        (
            ChoiceConstraintIndex(0, 1, 0),
            ChoiceConstraintIndex(1, 1, 0),
        )
    )
    regional = RestrictedRegionalProblem(problem, critical)
    candidate = _menu(2.0, 1.0)

    restored = backtrack_to_feasible_menu(
        regional,
        candidate,
        candidate,
        tolerance=1e-9,
    )

    assert restored is candidate


@pytest.mark.parametrize("reject_restored", [False, True])
def test_backtracking_defensively_falls_back_to_anchor(
    monkeypatch,
    reject_restored,
) -> None:
    problem = _switching_problem()
    critical = CriticalConstraintSet([1, 1], bundle_count=2)
    regional = RestrictedRegionalProblem(problem, critical)
    anchor = _menu(0.0, 0.0)
    calls = 0

    def synthetic_violation(vector) -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 1.0
        if reject_restored and calls == 12:
            return 1.0
        if reject_restored:
            return 0.0
        return 0.0 if calls == 2 else 1.0

    monkeypatch.setattr(regional, "maximum_constraint_violation", synthetic_violation)

    restored = backtrack_to_feasible_menu(
        regional,
        _menu(2.0, 1.0),
        anchor,
        tolerance=1e-9,
    )

    assert restored is anchor


@pytest.mark.parametrize("traveler_count", [10, 1000])
def test_generation_with_slsqp_reaches_known_fixed_assignment_optimum(
    traveler_count,
    record_property,
) -> None:
    """Run the complete generation loop on small and large synthetic populations."""

    problem = _GenerationOptimizationProblem(traveler_count)
    assignments = np.ones(traveler_count, dtype=np.int64)
    solver = ScipyRegionalSolver("SLSQP", options={"maxiter": 500, "ftol": 1e-9})

    result = solve_fixed_assignment(problem, _menu(0.5, 0.0), assignments, solver)

    expected_allowance = (3.5 - np.sqrt(8.25)) / 2.0
    expected_price = 1.5 + 0.25 * expected_allowance
    expected_objective = expected_price - expected_allowance
    valuation = problem.evaluate_valuations(result.menu)
    slacks = choice_constraint_slacks(result.menu, valuation, assignments)
    maximum_full_violation = float(np.max(np.maximum(-slacks[:, 0], 0.0)))
    metric = {
        "travelers": traveler_count,
        "iterations": result.iteration_count,
        "nlp_iterations": sum(
            iteration.solve_result.iteration_count for iteration in result.iterations
        ),
        "initial_critical": result.iterations[0].active_constraint_count,
        "final_critical": len(result.active_constraints),
        "added": result.added_constraint_count,
        "objective": result.final_solve_result.objective_value,
        "price": result.menu.prices[1],
        "allowance": result.menu.allowances[1, 0],
        "max_violation": maximum_full_violation,
        "kkt": result.final_solve_result.first_order_certificate.satisfies_kkt,
        "time_seconds": result.solve_time_seconds,
    }
    record_property("generation_metric", metric)

    assert result.converged
    assert result.iteration_count == 2
    assert result.iterations[0].active_constraint_count == 1
    assert result.added_constraint_count == 1
    assert len(result.active_constraints) == 2
    assert result.menu.allowances[1, 0] == pytest.approx(expected_allowance, abs=2e-5)
    assert result.menu.prices[1] == pytest.approx(expected_price, abs=2e-5)
    assert result.final_solve_result.objective_value == pytest.approx(
        expected_objective,
        abs=2e-5,
    )
    assert maximum_full_violation <= 1e-7
    assert result.final_solve_result.first_order_certificate.satisfies_kkt


def test_generation_stops_immediately_when_regional_solver_fails() -> None:
    solver = _MenuSequenceSolver(
        [_menu(0.0, 0.0)],
        statuses=[RegionalSolveStatus.FAILED],
    )

    result = solve_fixed_assignment(_switching_problem(), _menu(0.0, 0.0), [1, 1], solver)

    assert result.status is FixedAssignmentStatus.SOLVER_FAILED
    assert not result.converged
    assert result.iteration_count == 1
    assert result.iterations[0].separation is None
    assert result.added_constraint_count == 0


def test_generation_reports_iteration_limit_after_adding_new_constraint() -> None:
    solver = _MenuSequenceSolver([_menu(2.0, 1.0)])

    result = solve_fixed_assignment(
        _switching_problem(),
        _menu(0.0, 0.0),
        [1, 1],
        solver,
        max_iterations=1,
    )

    assert result.status is FixedAssignmentStatus.ITERATION_LIMIT
    assert result.iteration_count == 1
    assert result.added_constraint_count == 1
    assert len(result.active_constraints) == 2


def test_generation_reports_stall_if_only_active_constraint_is_violated() -> None:
    problem = _bounded_problem()
    # Initialization activates traveler 2 (C <= 3); this deliberately broken
    # adapter reports success at C=4, so separation has a violation but no
    # omitted violated constraint that could repair the restricted solve.
    solver = _MenuSequenceSolver([_menu(4.0, 0.0)])

    result = solve_fixed_assignment(problem, _menu(1.0, 0.0), [1, 1, 1], solver)

    assert result.status is FixedAssignmentStatus.STALLED
    separation = result.iterations[0].separation
    assert separation is not None
    assert not separation.is_feasible
    assert not separation.can_add_constraints


@pytest.mark.parametrize("max_iterations", [0, -1, 1.5])
def test_generation_rejects_invalid_iteration_limit(max_iterations) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        solve_fixed_assignment(
            _switching_problem(),
            _menu(0.0, 0.0),
            [1, 1],
            _MenuSequenceSolver([_menu(1.0, 0.0)]),
            max_iterations=max_iterations,
        )


def test_fixed_assignment_result_validates_immutable_record() -> None:
    valid = solve_fixed_assignment(
        _switching_problem(),
        _menu(0.0, 0.0),
        [1, 1],
        _MenuSequenceSolver([_menu(2.0, 1.0), _menu(1.5, 0.5)]),
    )

    with pytest.raises(ValueError, match="one-dimensional integer"):
        replace(valid, assigned_bundles=np.array([[1, 1, 1]]))
    with pytest.raises(ValueError, match="one-dimensional integer"):
        replace(valid, assigned_bundles=np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="at least one iteration"):
        replace(valid, iterations=())
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(valid, solve_time_seconds=-1.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(valid, solve_time_seconds=np.nan)


def test_outer_search_returns_consistent_candidate_and_separate_objectives(
    monkeypatch,
    record_property,
) -> None:
    assignment = np.array([0, 0], dtype=np.int64)
    problem = _AssignmentSequenceProblem([(assignment, 3.0), (assignment, 4.0)])
    fixed = _FixedAssignmentSequence([_menu(1.0, 0.0)], [8.0])
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)

    result = primal_module.find_assignment_consistent_menu(
        problem,
        _menu(0.0, 0.0),
        _MenuSequenceSolver([]),
    )
    record_property("assignment_metric", result.metrics())

    assert result.status is AssignmentConsistencyStatus.CONSISTENT
    assert result.consistent
    assert result.region_iteration_count == 1
    assert result.assignment_change_count == 0
    assert result.iterations[0].regional_objective == pytest.approx(8.0)
    assert result.iterations[0].true_profit == pytest.approx(4.0)
    assert result.true_profit == pytest.approx(4.0)
    assert result.menu is fixed._menus[0]
    np.testing.assert_array_equal(result.assignment, assignment)


def test_outer_search_rebuilds_region_after_bulk_assignment_change_and_warm_starts(
    monkeypatch,
    record_property,
) -> None:
    initial_assignment = np.zeros(100, dtype=np.int64)
    changed_assignment = np.ones(100, dtype=np.int64)
    initial_menu = _menu(0.0, 0.0)
    first_candidate = _menu(2.0, 0.5)
    second_candidate = _menu(1.5, 0.25)
    problem = _AssignmentSequenceProblem(
        [
            (initial_assignment, 2.0),
            (changed_assignment, 3.0),
            (changed_assignment, 4.0),
        ]
    )
    fixed = _FixedAssignmentSequence(
        [first_candidate, second_candidate],
        [20.0, 7.0],
    )
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)

    result = primal_module.find_assignment_consistent_menu(
        problem,
        initial_menu,
        _MenuSequenceSolver([]),
    )
    record_property("assignment_metric", result.metrics())

    assert result.consistent
    assert result.region_iteration_count == 2
    assert result.assignment_change_count == 1
    assert result.iterations[0].changed_traveler_count == 100
    assert result.iterations[0].regional_objective == pytest.approx(20.0)
    assert result.iterations[0].true_profit == pytest.approx(3.0)
    assert fixed.warm_starts == [initial_menu, first_candidate]
    np.testing.assert_array_equal(fixed.assignments[0], initial_assignment)
    np.testing.assert_array_equal(fixed.assignments[1], changed_assignment)


def test_outer_search_continues_after_revisit_by_default(monkeypatch) -> None:
    first = np.array([0, 0], dtype=np.int64)
    second = np.array([1, 0], dtype=np.int64)
    third = np.array([1, 1], dtype=np.int64)
    problem = _AssignmentSequenceProblem(
        [(first, 0.0), (second, 1.0), (first, 2.0), (third, 3.0), (third, 4.0)]
    )
    menus = [_menu(float(index), 0.0) for index in range(1, 5)]
    fixed = _FixedAssignmentSequence(menus, [1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)

    result = primal_module.find_assignment_consistent_menu(
        problem,
        _menu(0.0, 0.0),
        _MenuSequenceSolver([]),
    )

    assert result.status is AssignmentConsistencyStatus.CONSISTENT
    assert result.region_iteration_count == 4
    assert result.iterations[1].revisited_assignment
    assert fixed.warm_starts[2] is menus[1]


def test_revisit_option_does_not_stop_a_new_menu_with_improved_profit(monkeypatch) -> None:
    first = np.array([0, 0], dtype=np.int64)
    second = np.array([1, 0], dtype=np.int64)
    third = np.array([1, 1], dtype=np.int64)
    problem = _AssignmentSequenceProblem(
        [(first, 0.0), (second, 1.0), (third, 2.0), (first, 3.0), (first, 4.0)]
    )
    menus = [_menu(float(index), 0.0) for index in range(1, 5)]
    fixed = _FixedAssignmentSequence(menus, [1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)

    result = primal_module.find_assignment_consistent_menu(
        problem,
        _menu(0.0, 0.0),
        _MenuSequenceSolver([]),
        stop_on_assignment_revisit=True,
    )

    assert result.status is AssignmentConsistencyStatus.CONSISTENT
    assert result.consistent
    assert result.region_iteration_count == 4
    assert result.iterations[2].revisited_assignment
    assert result.menu is menus[-1]


def test_outer_search_uses_region_limit_instead_of_stopping_on_default_revisit(
    monkeypatch,
) -> None:
    first = np.array([0, 0], dtype=np.int64)
    second = np.array([1, 1], dtype=np.int64)
    problem = _AssignmentSequenceProblem([(first, 0.0), (second, 1.0), (first, 2.0)])
    fixed = _FixedAssignmentSequence([_menu(1.0, 0.0), _menu(2.0, 0.0)], [1.0, 2.0])
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)

    result = primal_module.find_assignment_consistent_menu(
        problem,
        _menu(0.0, 0.0),
        _MenuSequenceSolver([]),
        max_region_iterations=2,
    )

    assert result.status is AssignmentConsistencyStatus.REGION_ITERATION_LIMIT
    assert result.iterations[-1].revisited_assignment


def test_outer_search_propagates_fixed_assignment_failure(monkeypatch) -> None:
    assignment = np.array([0, 0], dtype=np.int64)
    problem = _AssignmentSequenceProblem([(assignment, 0.0)])
    fixed = _FixedAssignmentSequence(
        [_menu(1.0, 0.0)],
        [1.0],
        statuses=[FixedAssignmentStatus.SOLVER_FAILED],
    )
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)

    result = primal_module.find_assignment_consistent_menu(
        problem,
        _menu(0.0, 0.0),
        _MenuSequenceSolver([]),
    )

    assert result.status is AssignmentConsistencyStatus.FIXED_ASSIGNMENT_FAILED
    assert result.iterations[0].true_assignment is None
    assert result.iterations[0].changed_traveler_count is None
    assert result.true_profit == 0.0
    assert result.menu is result.initial_assignment.menu
    assert not result.iterations[0].accepted
    np.testing.assert_array_equal(result.assignment, assignment)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_region_iterations": 0}, "max_region_iterations"),
        ({"max_region_iterations": 1.5}, "max_region_iterations"),
        ({"stop_on_assignment_revisit": 1}, "stop_on_assignment_revisit"),
    ],
)
def test_outer_search_rejects_invalid_configuration(arguments, message) -> None:
    assignment = np.array([0], dtype=np.int64)
    problem = _AssignmentSequenceProblem([(assignment, 0.0)])

    with pytest.raises(ValueError, match=message):
        primal_module.find_assignment_consistent_menu(
            problem,
            _menu(0.0, 0.0),
            _MenuSequenceSolver([]),
            **arguments,
        )


def test_assignment_region_iteration_validates_its_assignment() -> None:
    assignment = np.array([0, 0], dtype=np.int64)
    problem = _AssignmentSequenceProblem([(assignment, 0.0), (assignment, 0.0)])
    fixed_factory = _FixedAssignmentSequence([_menu(1.0, 0.0)], [1.0])
    fixed = fixed_factory(problem, _menu(0.0, 0.0), assignment, None)
    true_assignment = problem.evaluate_assignment(_menu(1.0, 0.0))

    with pytest.raises(ValueError, match="one-dimensional integer"):
        AssignmentRegionIteration(1, np.array([[0, 0]]), fixed, true_assignment)
    with pytest.raises(ValueError, match="number must be positive"):
        AssignmentRegionIteration(0, assignment, fixed, true_assignment)
    with pytest.raises(ValueError, match="must match"):
        AssignmentRegionIteration(1, np.array([1, 1]), fixed, true_assignment)
    wrong_size = _assignment_snapshot(_menu(1.0, 0.0), np.array([0]), 0.0)
    with pytest.raises(ValueError, match="same travelers"):
        AssignmentRegionIteration(1, assignment, fixed, wrong_size)
    wrong_menu = _assignment_snapshot(_menu(2.0, 0.0), assignment, 0.0)
    with pytest.raises(ValueError, match="fixed-result menu"):
        AssignmentRegionIteration(1, assignment, fixed, wrong_menu)


def test_assignment_consistency_result_validates_history_and_time(monkeypatch) -> None:
    assignment = np.array([0], dtype=np.int64)
    problem = _AssignmentSequenceProblem([(assignment, 0.0), (assignment, 0.0)])
    fixed = _FixedAssignmentSequence([_menu(1.0, 0.0)], [1.0])
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)
    result = primal_module.find_assignment_consistent_menu(
        problem,
        _menu(0.0, 0.0),
        _MenuSequenceSolver([]),
    )

    with pytest.raises(ValueError, match="requires an iteration"):
        replace(result, iterations=())
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(result, solve_time_seconds=-1.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(result, solve_time_seconds=np.nan)


@pytest.mark.parametrize(
    ("regional_profit", "true_profit", "final_assignment"),
    [
        (2.0, 0.0, [0, 0]),  # Numerical reassignment must not erase profit.
        (0.5, 2.0, [1, 1]),  # Even favorable reassignment cannot hide regional loss.
        (2.0, 1.0 - 1e-12, [1, 1]),  # No cumulative tolerance-based drift.
    ],
)
def test_outer_search_rejects_profit_loss_and_retains_incumbent(
    monkeypatch,
    regional_profit,
    true_profit,
    final_assignment,
) -> None:
    initial_menu = _menu(1.0, 0.0)
    assignment = np.ones(2, dtype=np.int64)
    problem = _AssignmentSequenceProblem(
        [(assignment, 1.0), (np.array(final_assignment), true_profit)]
    )
    fixed = _FixedAssignmentSequence([_menu(2.0, 0.0)], [regional_profit])
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)

    result = primal_module.find_assignment_consistent_menu(
        problem,
        initial_menu,
        _MenuSequenceSolver([]),
    )
    diagnostics = primal_module.validate_candidate(result)

    assert result.status is AssignmentConsistencyStatus.PROFIT_DECREASE
    assert not result.consistent
    assert result.menu is initial_menu
    assert result.true_profit == 1.0
    np.testing.assert_array_equal(result.assignment, assignment)
    assert not result.iterations[-1].accepted
    assert result.iterations[-1].true_profit == true_profit
    assert result.assignment_change_count == 0
    assert diagnostics.status.value == "profit_decrease"
    assert not diagnostics.valid


def test_outer_search_retains_latest_accepted_menu_after_later_failure(monkeypatch) -> None:
    first = np.array([1, 0], dtype=np.int64)
    second = np.array([1, 1], dtype=np.int64)
    problem = _AssignmentSequenceProblem([(first, 1.0), (second, 2.0)])
    accepted_menu = _menu(2.0, 0.0)
    failed_menu = _menu(0.0, 0.0)
    fixed = _FixedAssignmentSequence(
        [accepted_menu, failed_menu],
        [2.0, 0.0],
        statuses=[FixedAssignmentStatus.CONVERGED, FixedAssignmentStatus.SOLVER_FAILED],
    )
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)

    result = primal_module.find_assignment_consistent_menu(
        problem,
        _menu(1.0, 0.0),
        _MenuSequenceSolver([]),
    )

    assert result.status is AssignmentConsistencyStatus.FIXED_ASSIGNMENT_FAILED
    assert result.menu is accepted_menu
    assert result.true_profit == 2.0
    np.testing.assert_array_equal(result.assignment, second)
    assert result.iterations[-1].fixed_assignment_result.menu is failed_menu
    assert not result.iterations[-1].accepted


def test_initial_and_final_assignment_share_solver_cost_tolerance(monkeypatch):
    problem = _SyntheticProblem([1.0], [0.0], [0.0])
    menu = _menu(1.0 + 5e-8, 0.0)
    fixed = _FixedAssignmentSequence([menu], [menu.prices[1]])
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)
    result = primal_module.solve_primal(
        problem,
        menu,
        ScipyRegionalSolver(),
        choice_tolerance=1e-9,
        violation_tolerance=1e-9,
    )
    assert result.valid
    assert fixed.assignments[0].tolist() == [1]
    assert result.candidate.initial_assignment.choice_tolerance == 1e-7
    assert result.candidate.terminal_iteration.true_assignment.choice_tolerance == 1e-7


def test_numerically_feasible_entry_remains_a_backtracking_anchor(monkeypatch):
    problem = _switching_problem()
    initial = _menu(1.5 + 5e-8, 1.0)
    solver = _MenuSequenceSolver([_menu(1.5, 0.0), _menu(1.5, 0.5)])
    calls = []
    original = primal_module.backtrack_to_feasible_menu

    def traced(regional, menu, anchor, *, tolerance):
        calls.append((anchor, tolerance))
        return original(regional, menu, anchor, tolerance=tolerance)

    monkeypatch.setattr(primal_module, "backtrack_to_feasible_menu", traced)
    result = solve_fixed_assignment(problem, initial, [1, 1], solver)
    assert result.converged
    assert calls == [(initial, 1e-7)]
    assert solver.warm_starts[1].allowances[1, 0] > 0.49
    assert result.feasibility_tolerance == 1e-7


def test_later_looser_solver_certificate_cannot_widen_cost_policy():
    class LooserCertificateSolver(_MenuSequenceSolver):
        feasibility_tolerance = 1e-7

        def solve(self, regional_problem, initial_menu):
            result = super().solve(regional_problem, initial_menu)
            return replace(
                result,
                maximum_constraint_violation=5e-7,
                first_order_certificate=replace(
                    result.first_order_certificate,
                    primal_violation=5e-7,
                    feasibility_tolerance=1e-6,
                ),
            )

    result = solve_fixed_assignment(
        _bounded_problem(),
        _menu(1.0, 0.0),
        [1, 1, 1],
        LooserCertificateSolver([_menu(3.0 + 5e-7, 0.0)]),
    )
    assert result.status is FixedAssignmentStatus.SOLVER_FAILED
    assert result.feasibility_tolerance == 1e-7


def test_double_tie_neighbors_use_best_margin_not_pairwise_closeness():
    response = _boundary_response([[5, 1, 1, 1]], prices=(0, 1, 1 + 0.75e-9, 1 + 1.5e-9))
    assert response.bundle_indices.tolist() == [2]
    # Option 1 is close to the selected option, but outside the best-margin set.
    assert response.numerical_optimistic_mask.tolist() == [[False, False, True, True]]
    assert [a.tolist() for a in primal_module._boundary_neighbors(response)] == [[3]]
    assert response.has_double_ties.tolist() == [True]
    assert not response.numerical_optimistic_mask.flags.writeable


@pytest.mark.parametrize("margin_tolerance,expected_regions", [(1e-9, 1), (1e-8, 2)])
def test_margin_policy_controls_search_and_validation_together(
    monkeypatch,
    margin_tolerance,
    expected_regions,
):
    class NearMarginProblem(_SyntheticProblem):
        bundle_count = 3

        def bundle_margins(self, menu):
            return menu.prices

        def evaluate_valuations(self, menu):
            return ValuationTable(np.array([[5.0, 0.0, -5e-9]]), np.zeros((1, 3, 1)))

    problem = NearMarginProblem([5.0], [0.0], [0.0])
    menu = BundleMenu(np.array([0, 1, 1 + 5e-9]), np.zeros((3, 1)))
    # A deliberately wide objective tolerance must not enlarge the margin tie set.
    fixed = _FixedAssignmentSequence([menu] * expected_regions, [1 + 5e-9, 1][:expected_regions])
    monkeypatch.setattr(primal_module, "solve_fixed_assignment", fixed)
    result = primal_module.solve_primal(
        problem,
        menu,
        ScipyRegionalSolver(),
        margin_tolerance=margin_tolerance,
        profit_tolerance=1e-4,
    )
    assert result.valid
    assert result.candidate.region_iteration_count == expected_regions
    assert result.double_tie_count == expected_regions - 1
    assert result.candidate.initial_assignment.margin_tolerance == margin_tolerance
    if expected_regions == 2:
        assert result.candidate.iterations[-1].boundary_returned
        assert [a.tolist() for a in fixed.assignments] == [[1], [2]]


def test_price_and_allowance_matching_have_independent_units():
    left, right = _menu(10, 100), _menu(10, 100 + 5e-5)
    assert not primal_module._same_menu(left, right, 1e-3, 0, 1e-7)
    assert primal_module._same_menu(left, right, 1e-7, 0, 1e-4)


@pytest.mark.parametrize(
    "field",
    ["choice_tolerance", "violation_tolerance", "margin_tolerance", "profit_tolerance"],
)
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_search_tolerances_are_rejected_before_assignment(field, value):
    with pytest.raises(ValueError, match=field):
        primal_module.solve_primal(
            _bounded_problem(),
            _menu(1.0, 0.0),
            ScipyRegionalSolver(),
            **{field: value},
        )
