"""Experiment orchestration, positive starts and end-to-end regression."""

from types import SimpleNamespace

import numpy as np
import pytest

import maas_bundle.benchmarks.two_route as two_route_module
from maas_bundle import CandidateValidationStatus
from maas_bundle.benchmarks.two_route import (
    generate_two_route_initial_menus,
    run_two_route_example,
    run_two_route_multistart,
)


def test_two_route_example_runs_complete_primal_pipeline() -> None:
    result = run_two_route_example()

    assert result.status is CandidateValidationStatus.CONSISTENT_CANDIDATE
    assert result.valid
    assert result.double_tie_count == 0
    assert result.candidate.region_iteration_count == 2
    assert result.candidate.boundary_search_complete
    assert result.candidate.iterations[0].assumed_assignment.tolist() == [1, 2, 1, 2]
    assert result.candidate.iterations[0].true_assignment is not None
    assert result.candidate.iterations[0].true_assignment.bundle_indices.tolist() == [1, 2, 2, 2]
    assert result.candidate.assignment.tolist() == [1, 2, 2, 2]
    assert result.candidate.true_profit == pytest.approx(3.54506977, abs=1e-6)
    assert result.candidate.menu.prices == pytest.approx([0.0, 2.91225236, 5.27554881])
    assert np.all(np.diff(result.candidate.menu.prices) >= 0)


def test_two_route_multistart_menus_are_reproducible_and_monotone() -> None:
    first = generate_two_route_initial_menus(3, seed=17)
    second = generate_two_route_initial_menus(3, seed=17)

    assert len(first) == 3
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.prices, right.prices)
        assert np.array_equal(left.allowances, right.allowances)
        assert np.all(np.diff(left.prices) >= 0)
        assert np.all(np.diff(left.allowances, axis=0) >= 0)
        assert left.prices[0] == 0
        assert np.all(left.allowances[0] == 0)


@pytest.mark.parametrize("start_count", [0, -1, 1.5])
def test_two_route_multistart_menu_generator_rejects_invalid_count(start_count) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        generate_two_route_initial_menus(start_count)


def test_two_route_multistart_wrapper_builds_problem_and_solver(monkeypatch) -> None:
    sentinel = object()
    captured = {}
    monkeypatch.setattr(two_route_module, "build_two_route_problem", lambda: "problem")
    monkeypatch.setattr(
        two_route_module,
        "generate_two_route_initial_menus",
        lambda count, seed, problem, minimum_profit: (count, seed),
    )
    monkeypatch.setattr(two_route_module, "ScipyRegionalSolver", lambda *args, **kwargs: "solver")

    def fake_multistart(problem, menus, solver, **kwargs):
        captured.update(problem=problem, menus=menus, solver=solver, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(two_route_module, "run_multistart", fake_multistart)

    assert run_two_route_multistart(4, seed=8) is sentinel
    assert captured == {
        "problem": "problem",
        "menus": (4, 8),
        "solver": "solver",
        "kwargs": {"max_region_iterations": 20, "profit_tolerance": 1e-9},
    }


def test_compare_payload_uses_primal_and_enumeration_results(monkeypatch) -> None:
    menu = SimpleNamespace(
        prices=np.array([0.0, 2.0]),
        allowances=np.array([[0.0], [3.0]]),
    )
    primal_candidate = SimpleNamespace(
        true_profit=3.0,
        assignment=np.array([1]),
        region_iteration_count=2,
        boundary_search_complete=True,
    )
    primal = SimpleNamespace(
        status=SimpleNamespace(value="consistent_candidate"),
        candidate=primal_candidate,
    )
    best = SimpleNamespace(
        true_profit=4.0,
        assigned_bundles=np.array([1]),
        true_assignment=SimpleNamespace(menu=menu),
    )
    enumeration = SimpleNamespace(
        best_candidate=best,
        metrics=lambda: {"assignments": 2, "best_true_profit": 4.0},
    )
    monkeypatch.setattr(two_route_module, "build_two_route_problem", lambda: object())
    monkeypatch.setattr(two_route_module, "initial_two_route_menu", lambda: object())
    monkeypatch.setattr(two_route_module, "ScipyRegionalSolver", lambda *args, **kwargs: object())
    monkeypatch.setattr(two_route_module, "solve_primal", lambda *args, **kwargs: primal)
    monkeypatch.setattr(
        two_route_module,
        "enumerate_assignment_regions",
        lambda *args, **kwargs: enumeration,
    )

    payload = two_route_module.compare_two_route()

    assert payload["primal"]["assignment"] == [1]
    assert payload["enumeration"]["best_assignment"] == [1]
    assert payload["enumerated_minus_primal_profit"] == pytest.approx(1.0)
