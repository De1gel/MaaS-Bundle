"""Tests for reusable multi-start region diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest

import maas_bundle.benchmarks.multistart as multistart_module
from maas_bundle import (
    BundleDesignProblem,
    BundleMenu,
    ScipyRegionalSolver,
    UniformDemandDistribution,
    build_route_value_oracle,
)
from maas_bundle.benchmarks import MultiStartResult, MultiStartRun, run_multistart


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
    solver = ScipyRegionalSolver("SLSQP")
    return problem, menu, solver


def _diagnostics(
    initial: tuple[int, ...],
    final: tuple[int, ...],
    profit: float | None,
    *,
    consistent: bool = True,
    valid: bool = True,
    iterations: int = 1,
):
    candidate = SimpleNamespace(
        consistent=consistent,
        initial_assignment=SimpleNamespace(bundle_indices=np.array(initial)),
        assignment=np.array(final),
        true_profit=profit,
        region_iteration_count=iterations,
    )
    return SimpleNamespace(candidate=candidate, valid=valid)


def test_run_multistart_keeps_each_start_and_forwards_solver_options(monkeypatch) -> None:
    problem, menu, solver = _tiny_problem()
    second_menu = BundleMenu(
        prices=np.array([0.0, 0.8]),
        allowances=np.array([[0.0], [1.5]]),
    )
    calls = []

    def fake_solve(problem, initial_menu, solver, **kwargs):
        calls.append((problem, initial_menu, solver, kwargs))
        return _diagnostics((0,), (1,), 1.2)

    monkeypatch.setattr(multistart_module, "solve_primal", fake_solve)

    result = run_multistart(
        problem,
        [menu, second_menu],
        solver,
        max_region_iterations=7,
        stop_on_assignment_revisit=True,
    )

    assert len(result.runs) == 2
    assert [run.number for run in result.runs] == [1, 2]
    assert result.runs[0].initial_menu is menu
    assert result.runs[0].initial_assignment == (0,)
    assert result.runs[0].final_assignment == (1,)
    assert result.runs[0].true_profit == pytest.approx(1.2)
    assert all(call[3]["max_region_iterations"] == 7 for call in calls)
    assert all(call[3]["stop_on_assignment_revisit"] for call in calls)
    assert result.solve_time_seconds >= 0


def test_multistart_groups_consistent_regions_and_selects_best_run() -> None:
    _, menu, _ = _tiny_problem()
    runs = (
        MultiStartRun(1, menu, _diagnostics((0,), (1,), 3.0, iterations=2)),
        MultiStartRun(2, menu, _diagnostics((1,), (1,), 4.0)),
        MultiStartRun(3, menu, _diagnostics((0,), (0,), None, valid=False)),
        MultiStartRun(
            4,
            menu,
            _diagnostics((1,), (1,), 8.0, consistent=False, valid=False),
        ),
    )
    result = MultiStartResult(runs=runs, solve_time_seconds=0.25)

    assert len(result.consistent_runs) == 3
    assert result.distinct_region_count == 2
    assert result.best_run is runs[1]
    assert result.metrics() == {
        "starts": 4,
        "consistent_starts": 3,
        "valid_starts": 2,
        "distinct_regions": 2,
        "best_true_profit": 4.0,
        "time_seconds": 0.25,
    }
    summaries = result.region_summaries()
    assert summaries[0]["assignment"] == [0]
    assert summaries[0]["best_true_profit"] is None
    assert summaries[1]["assignment"] == [1]
    assert summaries[1]["starts"] == 2
    assert summaries[1]["valid_starts"] == 2
    assert summaries[1]["best_true_profit"] == pytest.approx(4.0)
    assert summaries[1]["worst_true_profit"] == pytest.approx(3.0)
    assert summaries[1]["mean_region_iterations"] == pytest.approx(1.5)
    assert summaries[1]["start_numbers"] == [1, 2]


def test_multistart_without_consistent_run_has_no_best() -> None:
    _, menu, _ = _tiny_problem()
    run = MultiStartRun(
        1,
        menu,
        _diagnostics((0,), (1,), 2.0, consistent=False, valid=False),
    )
    result = MultiStartResult((run,), 0.0)

    assert result.best_run is None
    assert result.distinct_region_count == 0
    assert result.region_summaries() == ()
    assert result.metrics()["best_true_profit"] is None


def test_run_multistart_rejects_empty_or_wrong_dimension_menus() -> None:
    problem, menu, solver = _tiny_problem()
    wrong_menu = BundleMenu(
        prices=np.array([0.0]),
        allowances=np.array([[0.0]]),
    )

    with pytest.raises(ValueError, match="at least one"):
        run_multistart(problem, [], solver)
    with pytest.raises(ValueError, match="problem dimensions"):
        run_multistart(problem, [menu, wrong_menu], solver)
