"""Check genuine boundary transitions and reproducible complete-search multi-starts."""

import csv
import json

import numpy as np
import pytest

from maas_bundle.benchmarks.stylized_regional import StylizedRegionalSettings, _search
from maas_bundle.benchmarks.stylized_regional_multistart import (
    _audit_run,
    run_stylized_regional_multistart,
    save_stylized_regional_multistart,
)
from maas_bundle.cli import main
from maas_bundle.design.menu import BundleMenu
from maas_bundle.examples import build_stylized_problem


@pytest.mark.parametrize("all_constraints", [False, True])
def test_accepted_neighbor_is_counted_but_rejected_boundary_return_is_excluded(all_constraints):
    problem = build_stylized_problem(price_upper_bound=190, allowance_upper_bound=64)
    # Start 450 in the recorded seed-20261004, 500-start experiment.
    initial = BundleMenu(
        np.array([0, 63.158512605798805, 64.52341864578479]),
        np.array(
            [
                [0, 0],
                [32.65803042369091, 51.25499736747309],
                [42.483284508941324, 14.095663447850043],
            ]
        ),
    )
    settings = StylizedRegionalSettings()
    run = _search(problem, initial, settings, all_constraints=all_constraints)
    assert run["accepted_assignment_path"] == [[2, 1, 1], [1, 1, 1], [1, 2, 1]]
    assert run["accepted_assignment_transitions"] == 2
    assert run["accepted_optimistic_reassignments"] == 1
    assert run["accepted_strict_margin_reassignments"] == 0
    assert run["accepted_boundary_transitions"] == 1
    first, unchanged, rejected, neighbor = run["iterations"]
    assert first["assignment_change_mechanism"] == "double_tie_resolution"
    assert abs(first["same_menu_reassignment_profit_gain"]) <= settings.profit_tolerance
    assert not unchanged["retained_assignment_changed"]
    assert rejected["boundary_returned"] and not rejected["accepted"]
    assert not rejected["retained_assignment_changed"]
    assert neighbor["accepted"] and neighbor["boundary_origin"] == unchanged["iteration"]
    assert neighbor["starting_assignment"] == neighbor["response"]["assignment"]
    assert neighbor["switches"] == []
    assert neighbor["retained_assignment_changed"]
    assert neighbor["response"]["true_profit"] > neighbor["incumbent_profit_before_solve"] + 9
    assert run["final_response"]["true_profit"] == pytest.approx(61)
    assert _audit_run(problem, run, settings)["passed"]


def test_multistart_reuses_positive_reference_menus_and_exports_three_artifacts(tmp_path):
    result = run_stylized_regional_multistart(4, seed=20261004)
    assert result["summary"]["audited_valid_starts"] == 4
    assert result["sampling"]["reference_starts"] == 2
    assert result["sampling"]["random_positive_profit_starts"] == 2
    assert result["model"]["frequency_supports"] == ((14, 22), (16, 24), (18, 26))
    assert all(run["initial_response"]["true_profit"] > 0 for run in result["runs"])
    assert result["runs"][0]["initial_response"]["assignment"] == [1, 2, 2]
    assert result["runs"][1]["accepted_assignment_path"] == [[2, 2, 1], [2, 2, 2]]
    assert result["summary"]["representative_all_constraints_agreement"]["audited_valid"]
    save_stylized_regional_multistart(result, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "a2_multistart_diagnostics.json",
        "a2_multistart_runs.csv",
        "a2_multistart.png",
    }
    loaded = json.loads((tmp_path / "a2_multistart_diagnostics.json").read_text())
    assert len(loaded["runs"]) == 4
    with (tmp_path / "a2_multistart_runs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert json.loads(rows[1]["accepted_assignment_path"]) == [[2, 2, 1], [2, 2, 2]]


def test_stylized_cli_initializes_each_assignment_once_with_a_solver_owned_seed(tmp_path, capsys):
    assert (
        main(
            [
                "multistart",
                "stylized",
                "--starts",
                "26",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    summary = json.loads(output[output.index("{\n") :])
    assert summary["initial_assignments"] == summary["initialization_regional_solves"] == 26
    assert summary["initialization_nlp_solves"] >= 26
    assert summary["certified_initial_regions"] == 26
    assert summary["failed_initial_regions"] == 0
    assert summary["continuations_attempted"] == 26
    assert summary["continuation_entry_errors"] == []
    assert summary["starts_per_assignment"] == 1
    loaded = json.loads((tmp_path / "a2_assignment_starts.json").read_text())
    assert loaded["initialization"]["owner"] == "ScipyRegionalSolver.default_initial_menu"
    assert loaded["initialization"]["solver_initial_menu"]["prices"] == [0, 0, 0]
    for row in loaded["runs"]:
        region = row["initial_region_result"]
        assert row["initial_assignment"] != [0, 0, 0]
        assert region["starts"] == 1
        assert region["nlp_solves"] >= 1
        if row["initial_assignment"] in ([0, 1, 1], [1, 2, 2], [2, 1, 2]):
            assert row["continuation"]["audit"]["passed"]
    representative = next(row for row in loaded["runs"] if row["initial_assignment"] == [0, 0, 2])
    assert representative["path_including_initial_region_response"] == [
        [0, 0, 2],
        [0, 2, 2],
        [1, 2, 2],
    ]
    assert representative["initial_response_profit_gain"] == pytest.approx(23.5)
    continuation = representative["continuation"]
    assert continuation["audit"]["passed"]
    assert continuation["accepted_assignment_transitions"] == 1
    assert continuation["accepted_assignment_path"] == [[0, 2, 2], [1, 2, 2]]
    assert continuation["accepted_boundary_transitions"] == 1
    assert {p.name for p in tmp_path.iterdir()} == {
        "a2_assignment_starts.json",
        "a2_assignment_starts.csv",
        "a2_assignment_starts.png",
    }
    with (tmp_path / "a2_assignment_starts.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 26


def test_stylized_cli_rejects_multiple_random_starts_per_assignment():
    with pytest.raises(SystemExit):
        main(["multistart", "stylized", "--starts", "500"])


@pytest.mark.parametrize("start_count,seed", [(1, 0), (2.5, 0), (2, -1), (2, 1.5)])
def test_multistart_rejects_invalid_sampling_budgets(start_count, seed):
    with pytest.raises(ValueError):
        run_stylized_regional_multistart(start_count, seed=seed)
