"""Independent feasibility/profit audits and artifact checks for A2."""

import csv
import json
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pytest

from maas_bundle.benchmarks.stylized_regional import (
    StylizedRegionalSettings,
    _restore_menu,
    run_stylized_regional,
    save_stylized_regional_results,
)
from maas_bundle.benchmarks.stylized_regional_plots import build_regional_trajectory
from maas_bundle.examples.stylized import build_stylized_problem


@pytest.fixture(scope="module")
def experiment():
    return run_stylized_regional(StylizedRegionalSettings(starts=1))


def test_baseline_and_every_search_endpoint_are_audited(experiment):
    problem = build_stylized_problem(price_upper_bound=190, allowance_upper_bound=64)
    assert experiment["initial_response"]["assignment"] == [2, 2, 1]
    assert experiment["initial_response"]["true_profit"] == pytest.approx(76 / 3)
    assert experiment["baseline_reference"]["initial_response"]["assignment"] == [1, 2, 2]
    assert experiment["baseline_reference"]["initial_response"]["true_profit"] == pytest.approx(30)
    assert experiment["numerical_validity"]["passed"]
    for key in ("regional_search_cg", "regional_search_all_constraints"):
        run = experiment[key]
        previous = experiment["initial_response"]["true_profit"]
        for row in run["iterations"]:
            menu = _restore_menu(row["optimized_menu"])
            response = problem.evaluate_assignment(menu)
            assert row["response"]["assignment"] == response.bundle_indices.tolist()
            assert row["response"]["true_profit"] == pytest.approx(
                problem.traveler_weights @ problem.bundle_margins(menu)[response.bundle_indices]
            )
            if row["accepted"]:
                assert response.total_profit >= previous - 1e-9
                previous = response.total_profit
            for switch in row["switches"]:
                i = ("P", "M", "S").index(switch["class"])
                a, b = switch["pair"]
                assert switch["cost_gap"] == pytest.approx(
                    response.generalized_costs[i, b] - response.generalized_costs[i, a]
                )
        assert run["final_validation"]["assignment_consistent"]
        assert run["final_validation"]["maximum_violation"] <= 1e-7
    assert experiment["cg_vs_all"]["same_final_assignment"]
    assert abs(experiment["cg_vs_all"]["profit_difference"]) <= 1e-9
    assert experiment["benchmark_gap"]["absolute"] >= -1e-9


def test_s_switch_is_a_true_boundary_transition_and_next_region_uses_its_menu(experiment):
    for key in ("regional_search_cg", "regional_search_all_constraints"):
        run = experiment[key]
        first, second = run["iterations"]
        assert first["starting_assignment"] == [2, 2, 1]
        assert first["response"]["assignment"] == second["starting_assignment"] == [2, 2, 2]
        assert second["starting_menu"] == first["optimized_menu"]
        assert first["accepted"] and second["accepted"]
        assert run["accepted_assignment_transitions"] == 1
        (switch,) = first["switches"]
        assert switch["class"] == "S"
        assert switch["pair"] == [1, 2]
        assert abs(switch["cost_gap"]) <= 1e-7
        assert switch["margin_gap"] > 1
        assert first["same_menu_reassignment_profit_gain"] == pytest.approx(
            switch["margin_gap"] / 3, abs=1e-9
        )
        assert second["response"]["true_profit"] > first["response"]["true_profit"]
    assert experiment["baseline_reference"]["accepted_assignment_transitions"] == 0
    assert not experiment["benchmark_gap"]["matches_best_found"]


def test_26_candidate_assignments_use_one_seed_and_full_constraints(experiment):
    enumeration = experiment["enumeration"]
    assert [r["assignment"] for r in enumeration["records"]] == [
        list(a) for a in product(range(3), repeat=3) if a != (0, 0, 0)
    ]
    assert enumeration["starts_per_assignment"] == 1
    assert enumeration["nlp_solves"] >= 26
    assert enumeration["excluded_assignments"][0]["assignment"] == [0, 0, 0]
    problem = build_stylized_problem(price_upper_bound=190, allowance_upper_bound=64)
    for record in enumeration["records"]:
        assert record["starts"] == 1
        assert record["successful_starts"] == 1
        assert record["failed_starts"] == 0
        assert record["status"] != "infeasible"
        assert record["generated_constraints"] == 0
        if record["best"]:
            menu = _restore_menu(record["best_feasible_menu"])
            costs = problem.evaluate_valuations(menu).route_values + menu.prices
            gaps = costs[np.arange(3), record["assignment"]][:, None] - costs
            assert gaps.max() <= 1e-7
            assert record["best_regional_profit"] == pytest.approx(
                problem.traveler_weights @ problem.bundle_margins(menu)[record["assignment"]]
            )
    assert enumeration["failed_starts"] == 0
    assert all(r["status"] == "feasible solution found" for r in enumeration["records"])
    assert enumeration["double_tie_encountered"]


def test_export_has_exactly_requested_six_artifacts_and_complete_records(experiment, tmp_path):
    save_stylized_regional_results(experiment, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "a2_regional_search_trajectory.pdf",
        "a2_regional_search_trajectory.png",
        "a2_iteration_table.csv",
        "a2_assignment_enumeration.csv",
        "a2_algorithm_comparison.csv",
        "a2_full_results.json",
    }
    loaded = json.loads((tmp_path / "a2_full_results.json").read_text())
    assert loaded["domain"]["price_upper_bounds"] == [190, 190]
    assert len(loaded["enumeration"]["records"]) == 26
    assert loaded["regional_search_cg"]["iterations"][0]["generation_iterations"]
    for name, count in (
        ("a2_iteration_table.csv", 2),
        ("a2_assignment_enumeration.csv", 26),
        ("a2_algorithm_comparison.csv", 3),
    ):
        with (tmp_path / name).open() as stream:
            assert len(list(csv.DictReader(stream))) == count


def test_trajectory_labels_show_post_response_assignment_and_retained_profit(experiment):
    figure = build_regional_trajectory(experiment)
    try:
        axis = figure.axes[0]
        accepted = experiment["regional_search_cg"]["iterations"]
        np.testing.assert_allclose(
            axis.lines[0].get_ydata(),
            [
                experiment["initial_response"]["true_profit"],
                *[r["response"]["true_profit"] for r in accepted if r["accepted"]],
            ],
        )
        assert not any("No assignment switch" in t.get_text() for t in axis.texts)
        assert any("S: Basic → Premium" in t.get_text() for t in axis.texts)
        assert any("(2,2,1)" in t.get_text() for t in axis.texts)
        assert any("(2,2,2)" in t.get_text() for t in axis.texts)
        assert any("After switch:" in t.get_text() for t in axis.texts)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "changes",
    [
        {"starts": 3},
        {"starts": 5.5},
        {"seed": -1},
        {"feasibility_tolerance": 0},
        {"ftol": np.nan},
        {"trajectory_start": "forced"},
    ],
)
def test_settings_reject_invalid_run_policies(changes):
    with pytest.raises(ValueError):
        StylizedRegionalSettings(**changes)
