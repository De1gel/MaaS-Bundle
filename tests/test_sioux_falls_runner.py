"""Budgeted experiments re-enter search from the best verified menu on failure."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maas_bundle.design.menu import BundleMenu
from maas_bundle.examples.two_route import build_two_route_problem, initial_two_route_menu


@pytest.mark.parametrize(
    "finish_successfully,no_recovery,expected_calls",
    [(True, False, 2), (False, False, 4), (False, True, 1)],
)
def test_search_restarts_use_best_observed_menu_without_falsifying_success(
    monkeypatch, tmp_path, finish_successfully, no_recovery, expected_calls
):
    script = Path(__file__).resolve().parents[1] / "scripts/run-sioux-falls-five-mode.py"
    spec = importlib.util.spec_from_file_location("sioux_runner", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    problem = build_two_route_problem()
    initial = initial_two_route_menu()
    better = BundleMenu(initial.prices + np.array([0, .02, .02]), initial.allowances)
    better_response = problem.evaluate_assignment(better)
    assert better_response.total_profit > problem.evaluate_assignment(initial).total_profit
    calls = []

    def backend_solve(self, regional, menu):
        assert self.slsqp_restarts == 4
        assert not self.enable_fallback
        assert self.numerical_constraint_recovery
        return SimpleNamespace(menu=better, metrics=lambda: {"attempts": 1})

    def primal(problem, menu, solver, **kwargs):
        calls.append(menu)
        solver.solve(SimpleNamespace(constraint_count=0), menu)
        valid = finish_successfully and len(calls) > 1
        response = better_response if valid else problem.evaluate_assignment(menu)
        metrics = {"regions": 1, "true_profit": response.total_profit}
        candidate = SimpleNamespace(
            menu=response.menu, assignment=response.bundle_indices, iterations=(),
            initial_assignment=problem.evaluate_assignment(menu), metrics=lambda: metrics,
        )
        status = "consistent_candidate" if valid else "regional_solve_failed"
        return SimpleNamespace(
            candidate=candidate, status=SimpleNamespace(value=status), valid=valid,
            message="synthetic diagnostic", metrics=lambda: {"valid": valid},
        )

    output = tmp_path / "result.json"
    monkeypatch.setattr(runner, "build_sioux_falls_problem", lambda **kwargs: problem)
    monkeypatch.setattr(runner, "initial_sioux_falls_menu", lambda: initial)
    monkeypatch.setattr(runner.ScipyRegionalSolver, "solve", backend_solve)
    monkeypatch.setattr(runner, "solve_primal", primal)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runner.signal, "alarm", lambda *args: None)
    argv = [str(script), "--output", str(output)]
    if no_recovery:
        argv.append("--no-recovery")
    monkeypatch.setattr("sys.argv", argv)
    assert runner.main() == (0 if finish_successfully else 1)
    assert len(calls) == expected_calls
    for menu in calls[1:]:
        np.testing.assert_array_equal(menu.prices, better.prices)
        np.testing.assert_array_equal(menu.allowances, better.allowances)
    saved = json.loads(output.read_text())
    assert saved["valid"] is finish_successfully
    assert len(saved["search_attempts"]) == expected_calls
    assert saved["best_observed_profit"] == better_response.total_profit
