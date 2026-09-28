"""Tests for the exhaustive small-instance assignment baseline."""

from dataclasses import replace

import numpy as np
import pytest

import maas_bundle.benchmarks.assignment_enumeration as enumeration_module
from maas_bundle import (
    BundleDesignProblem,
    BundleMenu,
    FixedAssignmentStatus,
    ScipyRegionalSolver,
    UniformDemandDistribution,
    build_route_value_oracle,
)
from maas_bundle.benchmarks import enumerate_assignment_regions


def _tiny_problem() -> tuple[BundleDesignProblem, BundleMenu, ScipyRegionalSolver]:
    oracle = build_route_value_oracle([1.0], [[1.0]], [1.0])
    problem = BundleDesignProblem(
        traveler_weights=np.array([1.0]),
        unit_allowance_cost=np.array([0.2]),
        route_oracles=((oracle, oracle),),
        demand_distributions=(UniformDemandDistribution(1.0, 2.0),),
    )
    menu = BundleMenu(
        prices=np.array([0.0, 0.5]),
        allowances=np.array([[0.0], [1.0]]),
    )
    solver = ScipyRegionalSolver("SLSQP", options={"maxiter": 500, "ftol": 1e-10})
    return problem, menu, solver


def test_enumeration_visits_every_assignment_and_selects_best_consistent_candidate() -> None:
    problem, menu, solver = _tiny_problem()

    result = enumerate_assignment_regions(problem, menu, solver)

    assert result.assignment_count == 2
    assert result.converged_region_count == 2
    assert result.consistent_candidate_count == 1
    assert result.best_candidate is not None
    assert result.best_candidate.assigned_bundles.tolist() == [1]
    assert result.best_candidate.true_profit == pytest.approx(1.12)
    assert result.metrics()["best_true_profit"] == pytest.approx(1.12)


@pytest.mark.parametrize("limit", [0, -1, 1.5])
def test_enumeration_rejects_invalid_assignment_limit(limit) -> None:
    problem, menu, solver = _tiny_problem()

    with pytest.raises(ValueError, match="maximum_assignments"):
        enumerate_assignment_regions(problem, menu, solver, maximum_assignments=limit)


def test_enumeration_rejects_instance_larger_than_limit() -> None:
    problem, menu, solver = _tiny_problem()

    with pytest.raises(ValueError, match="requires 2 assignments"):
        enumerate_assignment_regions(problem, menu, solver, maximum_assignments=1)


def test_enumeration_can_disable_limit_and_skips_failed_regions(monkeypatch) -> None:
    problem, menu, solver = _tiny_problem()
    original = enumeration_module.solve_fixed_assignment

    def fail_outside_assignment(problem, initial_menu, assigned_bundles, solver, **kwargs):
        result = original(problem, initial_menu, assigned_bundles, solver, **kwargs)
        if assigned_bundles[0] == 0:
            return replace(result, status=FixedAssignmentStatus.SOLVER_FAILED)
        return result

    monkeypatch.setattr(
        enumeration_module,
        "solve_fixed_assignment",
        fail_outside_assignment,
    )

    result = enumerate_assignment_regions(
        problem,
        menu,
        solver,
        maximum_assignments=None,
    )

    assert result.assignment_count == 2
    assert result.converged_region_count == 1
    assert result.best_candidate is not None
