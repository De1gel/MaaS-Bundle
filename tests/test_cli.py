"""Tests for the installed command-line entry point."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import maas_bundle.cli as cli_module
from maas_bundle.cli import main


def test_two_route_cli_prints_machine_readable_result(capsys) -> None:
    exit_status = main(["example", "two-route"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_status == 0
    assert payload["example"] == "two-route"
    assert payload["status"] == "consistent_candidate"
    assert payload["valid"]
    assert payload["assignment"] == [1, 2, 2, 2]
    assert payload["boundary_search_complete"]
    assert payload["am_profit_at_source_demand"] > 0


@pytest.mark.parametrize(("best_assignment", "expected_exit"), [([1], 0), (None, 1)])
def test_compare_cli_reports_helper_payload(
    monkeypatch,
    capsys,
    best_assignment,
    expected_exit,
) -> None:
    payload = {
        "enumeration": {"best_assignment": best_assignment},
        "enumerated_minus_primal_profit": 0.5,
    }
    monkeypatch.setattr(cli_module, "_compare_two_route", lambda: payload)

    exit_status = main(["compare", "two-route"])

    assert exit_status == expected_exit
    assert json.loads(capsys.readouterr().out) == payload


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
    monkeypatch.setattr(cli_module, "build_two_route_problem", lambda: object())
    monkeypatch.setattr(cli_module, "initial_two_route_menu", lambda: object())
    monkeypatch.setattr(cli_module, "ScipyRegionalSolver", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli_module, "solve_primal", lambda *args, **kwargs: primal)
    monkeypatch.setattr(
        cli_module,
        "enumerate_assignment_regions",
        lambda *args, **kwargs: enumeration,
    )

    payload = cli_module._compare_two_route()

    assert payload["primal"]["assignment"] == [1]
    assert payload["enumeration"]["best_assignment"] == [1]
    assert payload["enumerated_minus_primal_profit"] == pytest.approx(1.0)


@pytest.mark.parametrize("consistent_starts", [3, 0])
def test_multistart_cli_reports_helper_payload(
    monkeypatch,
    capsys,
    consistent_starts,
) -> None:
    payload = {"consistent_starts": consistent_starts, "distinct_regions": 2}
    monkeypatch.setattr(cli_module, "_multistart_two_route", lambda starts, seed: payload)

    exit_status = main(
        ["multistart", "two-route", "--starts", "3", "--seed", "17"]
    )

    assert exit_status == (0 if consistent_starts else 1)
    assert json.loads(capsys.readouterr().out) == payload


def test_multistart_payload_reports_regions_and_best_start(monkeypatch) -> None:
    best = SimpleNamespace(number=2, final_assignment=(1, 2))
    result = SimpleNamespace(
        best_run=best,
        distinct_region_count=2,
        metrics=lambda: {
            "starts": 3,
            "consistent_starts": 3,
            "distinct_regions": 2,
        },
        region_summaries=lambda: ({"assignment": [1, 2], "starts": 2},),
    )
    monkeypatch.setattr(cli_module, "run_two_route_multistart", lambda *args, **kwargs: result)

    payload = cli_module._multistart_two_route(3, 17)

    assert payload["seed"] == 17
    assert not payload["all_consistent_starts_same_region"]
    assert payload["best_assignment"] == [1, 2]
    assert payload["best_start"] == 2
    assert payload["regions"][0]["starts"] == 2


def test_multistart_payload_handles_no_best_run(monkeypatch) -> None:
    result = SimpleNamespace(
        best_run=None,
        distinct_region_count=0,
        metrics=lambda: {"starts": 1, "consistent_starts": 0},
        region_summaries=lambda: (),
    )
    monkeypatch.setattr(cli_module, "run_two_route_multistart", lambda *args, **kwargs: result)

    payload = cli_module._multistart_two_route(1, 17)

    assert payload["best_assignment"] is None
    assert payload["best_start"] is None


@pytest.mark.parametrize("name", ["sioux-falls", "souxf"])
def test_sioux_falls_example_cli_accepts_long_and_short_names(
    monkeypatch,
    capsys,
    name,
) -> None:
    payload = {"valid": True, "example": "sioux-falls"}
    monkeypatch.setattr(cli_module, "_run_sioux_falls", lambda: payload)

    assert main(["example", name]) == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_sioux_falls_example_cli_reports_invalid_result(monkeypatch, capsys) -> None:
    payload = {"valid": False, "example": "sioux-falls"}
    monkeypatch.setattr(cli_module, "_run_sioux_falls", lambda: payload)

    assert main(["example", "souxf"]) == 1
    assert json.loads(capsys.readouterr().out) == payload


@pytest.mark.parametrize("consistent_starts", [2, 0])
def test_sioux_falls_multistart_cli_uses_consistency_exit_status(
    monkeypatch,
    capsys,
    consistent_starts,
) -> None:
    payload = {"consistent_starts": consistent_starts}
    monkeypatch.setattr(cli_module, "_multistart_sioux_falls", lambda starts, seed: payload)

    status = main(["multistart", "souxf", "--starts", "2", "--seed", "9"])

    assert status == (0 if consistent_starts else 1)
    assert json.loads(capsys.readouterr().out) == payload


def test_sioux_falls_example_payload_is_compact(monkeypatch) -> None:
    menu = SimpleNamespace(
        prices=np.array([0.0, 2.0, 4.0]),
        allowances=np.array([[0.0], [1.0], [2.0]]),
    )
    candidate = SimpleNamespace(
        assignment=np.array([0, 1, 2, 2]),
        menu=menu,
        true_profit=3.0,
        region_iteration_count=2,
        boundary_search_complete=True,
        fixed_generation_iteration_count=3,
        solve_time_seconds=0.2,
    )
    diagnostics = SimpleNamespace(
        candidate=candidate,
        status=SimpleNamespace(value="consistent_candidate"),
        valid=True,
        double_tie_count=0,
    )
    monkeypatch.setattr(cli_module, "run_sioux_falls_example", lambda: diagnostics)
    monkeypatch.setattr(cli_module, "sioux_falls_od_matrix", lambda: np.eye(2))

    payload = cli_module._run_sioux_falls()

    assert payload["assignment_counts"] == [1, 1, 2]
    assert "assignment" not in payload
    assert payload["positive_od_pairs"] == 2
    assert payload["traveler_classes"] == 4
    assert payload["end_to_end_seconds"] >= 0
    assert payload["experiment"]["units"]["prices"] == "USD/person/month"
    assert payload["experiment"]["source_time_to_minutes"] == 0.6


def test_sioux_falls_multistart_payload_compacts_assignment_vectors(monkeypatch) -> None:
    best = SimpleNamespace(number=1, final_assignment=(0, 1, 2, 2))
    result = SimpleNamespace(
        best_run=best,
        distinct_region_count=1,
        metrics=lambda: {"starts": 2, "consistent_starts": 2},
        region_summaries=lambda: (
            {
                "assignment": [0, 1, 2, 2],
                "starts": 2,
                "best_true_profit": 3.0,
            },
        ),
    )
    monkeypatch.setattr(cli_module, "run_sioux_falls_multistart", lambda *args, **kwargs: result)
    monkeypatch.setattr(cli_module, "sioux_falls_od_matrix", lambda: np.eye(2))

    payload = cli_module._multistart_sioux_falls(2, 9)

    assert payload["traveler_classes"] == 4
    assert payload["all_consistent_starts_same_region"]
    assert payload["best_assignment_counts"] == [1, 1, 2]
    assert payload["best_start"] == 1
    assert payload["regions"][0]["assignment_counts"] == [1, 1, 2]
    assert len(payload["regions"][0]["assignment_signature"]) == 12
    assert "assignment" not in payload["regions"][0]
    assert payload["experiment"]["units"]["true_profit"] == "USD/person/month"


def test_sioux_falls_multistart_payload_handles_no_best(monkeypatch) -> None:
    result = SimpleNamespace(
        best_run=None,
        distinct_region_count=0,
        metrics=lambda: {"starts": 1, "consistent_starts": 0},
        region_summaries=lambda: (),
    )
    monkeypatch.setattr(cli_module, "run_sioux_falls_multistart", lambda *args, **kwargs: result)
    monkeypatch.setattr(cli_module, "sioux_falls_od_matrix", lambda: np.eye(1))

    payload = cli_module._multistart_sioux_falls(1, 9)

    assert payload["best_assignment_counts"] is None
    assert payload["best_start"] is None
