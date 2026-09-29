"""Experiment orchestration, positive starts and end-to-end regression."""

import numpy as np
import pytest

import maas_bundle.benchmarks.sioux_falls as sioux_module
from maas_bundle import CandidateValidationStatus
from maas_bundle.benchmarks.sioux_falls import (
    generate_sioux_falls_initial_menus,
    run_sioux_falls_example,
    run_sioux_falls_multistart,
)
from maas_bundle.examples.sioux_falls import build_sioux_falls_problem, initial_sioux_falls_menu


def test_sioux_falls_initial_menu_and_generated_starts_are_monotone() -> None:
    fixed = initial_sioux_falls_menu()
    first = generate_sioux_falls_initial_menus(3, seed=19)
    second = generate_sioux_falls_initial_menus(3, seed=19)

    assert fixed.prices.tolist() == [0.0, 20.0, 40.0]
    assert fixed.allowances.tolist() == [[0.0, 0.0], [32.0, 16.0], [80.0, 48.0]]
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.prices, right.prices)
        assert np.array_equal(left.allowances, right.allowances)
        assert np.all(np.diff(left.prices) >= 0)
        assert np.all(np.diff(left.allowances, axis=0) >= 0)


@pytest.mark.parametrize("start_count", [0, -1, 1.5])
def test_sioux_falls_start_generator_rejects_invalid_count(start_count) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        generate_sioux_falls_initial_menus(start_count)


def test_sioux_falls_run_wrappers_use_full_problem(monkeypatch) -> None:
    primal_sentinel = object()
    multistart_sentinel = object()
    captured = []
    monkeypatch.setattr(sioux_module, "build_sioux_falls_problem", lambda **kwargs: "problem")
    monkeypatch.setattr(sioux_module, "initial_sioux_falls_menu", lambda **kwargs: "menu")
    monkeypatch.setattr(
        sioux_module,
        "generate_sioux_falls_initial_menus",
        lambda count, seed, problem, config, minimum_profit: (count, seed),
    )
    monkeypatch.setattr(sioux_module, "ScipyRegionalSolver", lambda *args, **kwargs: "solver")

    def fake_primal(*args, **kwargs):
        captured.append(("primal", args, kwargs))
        return primal_sentinel

    def fake_multistart(*args, **kwargs):
        captured.append(("multistart", args, kwargs))
        return multistart_sentinel

    monkeypatch.setattr(sioux_module, "solve_primal", fake_primal)
    monkeypatch.setattr(sioux_module, "run_multistart", fake_multistart)

    assert run_sioux_falls_example() is primal_sentinel
    assert run_sioux_falls_multistart(4, seed=8) is multistart_sentinel
    assert captured == [
        (
            "primal",
            ("problem", "menu", "solver"),
            {"max_region_iterations": 30, "profit_tolerance": 1e-9},
        ),
        (
            "multistart",
            ("problem", (4, 8), "solver"),
            {"max_region_iterations": 30, "profit_tolerance": 1e-9},
        ),
    ]


@pytest.mark.sioux_falls_scale
def test_full_sioux_falls_primal_pipeline() -> None:
    result = run_sioux_falls_example()

    # Complete network path geometry, with no forced ordinary-tie trials.
    assert result.status is CandidateValidationStatus.CONSISTENT_CANDIDATE
    assert result.valid
    assert result.candidate.boundary_search_complete
    assert result.double_tie_count == 0
    assert result.candidate.assignment.size == 1056
    assert result.candidate.true_profit == pytest.approx(10.16182917, abs=1e-5)
    assert result.candidate.region_iteration_count == 5
    assert all(step.boundary_origin is None for step in result.candidate.iterations)
    assert result.candidate.terminal_region_number == 5


def test_sioux_falls_starts_have_positive_profit_on_full_instance() -> None:
    problem = build_sioux_falls_problem()
    fixed = problem.evaluate_assignment(initial_sioux_falls_menu())
    assert fixed.total_profit > 1e-9
    menus = generate_sioux_falls_initial_menus(30, seed=20260901, problem=problem)
    assert len(menus) == 30
    for menu in menus:
        assignment = problem.evaluate_assignment(menu)
        assert assignment.total_profit > 1e-9
        assert np.sum(assignment.bundle_demand[1:]) > 0
