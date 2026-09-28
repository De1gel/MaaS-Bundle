"""Tests for optimistic bundle selection and assignment results."""

import numpy as np
import pytest

from maas_bundle import (
    AssignmentResult,
    BundleMenu,
    ValuationTable,
    aggregate_bundle_demand,
    select_optimistic_bundles,
)


def test_selects_unique_minimum_cost_bundle() -> None:
    assignment = select_optimistic_bundles(
        generalized_costs=[[1.0, 2.0], [3.0, 1.0]],
        bundle_margins=[100.0, 0.0],
    )

    np.testing.assert_array_equal(assignment, [0, 1])


def test_cost_tie_is_broken_by_platform_margin() -> None:
    assignment = select_optimistic_bundles(
        generalized_costs=[[1.0, 1.0, 2.0]],
        bundle_margins=[2.0, 3.0, 100.0],
    )

    np.testing.assert_array_equal(assignment, [1])


def test_double_tie_is_broken_by_smallest_bundle_index() -> None:
    assignment = select_optimistic_bundles(
        generalized_costs=[[1.0, 1.0, 1.0]],
        bundle_margins=[2.0, 3.0, 3.0],
    )

    np.testing.assert_array_equal(assignment, [1])


def test_tolerances_control_which_bundles_are_tied() -> None:
    costs = [[1.0, 1.0 + 5e-10]]
    margins = [1.0, 2.0]

    inside_tolerance = select_optimistic_bundles(costs, margins, choice_tolerance=1e-9)
    outside_tolerance = select_optimistic_bundles(costs, margins, choice_tolerance=1e-12)
    margin_tie = select_optimistic_bundles(
        [[1.0, 1.0]],
        [2.0, 2.0 + 5e-10],
        profit_tolerance=1e-9,
    )

    np.testing.assert_array_equal(inside_tolerance, [1])
    np.testing.assert_array_equal(outside_tolerance, [0])
    np.testing.assert_array_equal(margin_tie, [0])


def test_select_returns_immutable_indices_for_empty_traveler_set() -> None:
    assignment = select_optimistic_bundles(np.empty((0, 2)), [1.0, 2.0])

    assert assignment.shape == (0,)
    assert not assignment.flags.writeable


@pytest.mark.parametrize(
    ("costs", "margins", "arguments", "message"),
    [
        ([1.0, 2.0], [1.0, 2.0], {}, "must be two-dimensional"),
        (np.empty((1, 0)), [], {}, "at least one bundle"),
        ([[1.0, 2.0]], [1.0], {}, "one entry per bundle"),
        ([[np.nan, 2.0]], [1.0, 2.0], {}, "only finite values"),
        ([[1.0, 2.0]], [1.0, np.inf], {}, "only finite values"),
        ([[1.0]], [1.0], {"choice_tolerance": -1.0}, "choice_tolerance"),
        ([[1.0]], [1.0], {"profit_tolerance": np.nan}, "profit_tolerance"),
    ],
)
def test_select_rejects_invalid_input(costs, margins, arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        select_optimistic_bundles(costs, margins, **arguments)


def test_aggregates_weighted_bundle_demand() -> None:
    demand = aggregate_bundle_demand(
        bundle_indices=[0, 2, 2, 1],
        traveler_weights=[100.0, 50.0, 80.0, 20.0],
        bundle_count=3,
    )

    np.testing.assert_allclose(demand, [100.0, 20.0, 130.0])
    assert not demand.flags.writeable


@pytest.mark.parametrize(
    ("indices", "weights", "bundle_count", "message"),
    [
        ([[0]], [1.0], 1, "must be one-dimensional"),
        ([0.0], [1.0], 1, "must contain integers"),
        ([0], [1.0, 2.0], 1, "must match"),
        ([0], [np.nan], 1, "finite and nonnegative"),
        ([0], [-1.0], 1, "finite and nonnegative"),
        ([0], [1.0], 0, "positive integer"),
        ([0], [1.0], 1.5, "positive integer"),
        ([-1], [1.0], 1, "invalid bundle index"),
        ([1], [1.0], 1, "invalid bundle index"),
    ],
)
def test_aggregate_rejects_invalid_input(indices, weights, bundle_count, message) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_bundle_demand(indices, weights, bundle_count)


def _valid_assignment_result_arguments():
    return {
        "menu": BundleMenu(prices=np.array([0.0]), allowances=np.array([[0.0]])),
        "valuation": ValuationTable(
            route_values=np.array([[1.0]]),
            expected_omega=np.array([[[0.0]]]),
        ),
        "bundle_indices": np.array([0]),
        "generalized_costs": np.array([[1.0]]),
        "bundle_margins": np.array([0.0]),
        "bundle_demand": np.array([1.0]),
        "total_profit": 0.0,
    }


def test_assignment_result_exposes_valuation_and_immutable_arrays() -> None:
    result = AssignmentResult(**_valid_assignment_result_arguments())

    np.testing.assert_allclose(result.route_values, [[1.0]])
    np.testing.assert_allclose(result.expected_omega, [[[0.0]]])
    assert all(
        not array.flags.writeable
        for array in (
            result.bundle_indices,
            result.generalized_costs,
            result.bundle_margins,
            result.bundle_demand,
            result.minimum_cost_gaps,
            result.numerical_tie_mask,
            result.optimistic_margin_gaps,
        )
    )


def test_assignment_result_records_strict_optimistic_tie_diagnostics() -> None:
    result = AssignmentResult(
        menu=BundleMenu(np.zeros(3), np.zeros((3, 1))),
        valuation=ValuationTable(np.zeros((1, 3)), np.zeros((1, 3, 1))),
        bundle_indices=np.array([1]),
        generalized_costs=np.array([[1.0, 1.0, 2.0]]),
        bundle_margins=np.array([2.0, 3.0, 100.0]),
        bundle_demand=np.array([0.0, 1.0, 0.0]),
        total_profit=3.0,
    )

    np.testing.assert_allclose(result.minimum_cost_gaps, [[0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(result.numerical_tie_mask, [[True, True, False]])
    np.testing.assert_allclose(result.optimistic_margin_gaps, [1.0])
    np.testing.assert_array_equal(result.has_numerical_cost_ties, [True])
    np.testing.assert_array_equal(result.has_double_ties, [False])


def test_assignment_result_records_deterministic_numerical_double_tie() -> None:
    costs = np.array([[1.0, 1.0 + 5e-10, 2.0]])
    margins = np.array([2.0, 2.0 + 5e-10, 100.0])
    selected = select_optimistic_bundles(
        costs,
        margins,
        choice_tolerance=1e-9,
        profit_tolerance=1e-9,
    )
    result = AssignmentResult(
        menu=BundleMenu(np.zeros(3), np.zeros((3, 1))),
        valuation=ValuationTable(np.zeros((1, 3)), np.zeros((1, 3, 1))),
        bundle_indices=selected,
        generalized_costs=costs,
        bundle_margins=margins,
        bundle_demand=np.array([1.0, 0.0, 0.0]),
        total_profit=2.0,
        choice_tolerance=1e-9,
        profit_tolerance=1e-9,
    )

    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_array_equal(result.numerical_tie_mask, [[True, True, False]])
    assert result.optimistic_margin_gaps[0] == pytest.approx(-5e-10)
    np.testing.assert_array_equal(result.has_double_ties, [True])


def test_assignment_result_uses_infinite_margin_gap_without_competing_tie() -> None:
    result = AssignmentResult(
        menu=BundleMenu(np.zeros(2), np.zeros((2, 1))),
        valuation=ValuationTable(np.zeros((1, 2)), np.zeros((1, 2, 1))),
        bundle_indices=np.array([0]),
        generalized_costs=np.array([[1.0, 2.0]]),
        bundle_margins=np.array([0.0, 10.0]),
        bundle_demand=np.array([1.0, 0.0]),
        total_profit=0.0,
    )

    assert np.isinf(result.optimistic_margin_gaps[0])
    np.testing.assert_array_equal(result.has_numerical_cost_ties, [False])
    np.testing.assert_array_equal(result.has_double_ties, [False])


def test_assignment_result_rejects_nonoptimistic_bundle_inside_cost_tie() -> None:
    with pytest.raises(ValueError, match="optimistic margin tie-breaking"):
        AssignmentResult(
            menu=BundleMenu(np.zeros(2), np.zeros((2, 1))),
            valuation=ValuationTable(np.zeros((1, 2)), np.zeros((1, 2, 1))),
            bundle_indices=np.array([0]),
            generalized_costs=np.array([[1.0, 1.0]]),
            bundle_margins=np.array([1.0, 2.0]),
            bundle_demand=np.array([1.0, 0.0]),
            total_profit=1.0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "menu",
            BundleMenu(np.zeros(2), np.zeros((2, 1))),
            "same bundles",
        ),
        (
            "menu",
            BundleMenu(np.zeros(1), np.zeros((1, 2))),
            "same allowance dimensions",
        ),
        ("bundle_indices", [0, 0], "one entry per traveler"),
        ("generalized_costs", [[1.0, 2.0]], "one entry per traveler and bundle"),
        ("bundle_margins", [], "one entry per bundle"),
        ("bundle_demand", [], "one entry per bundle"),
        ("bundle_indices", [1], "invalid bundle index"),
        ("bundle_indices", [-1], "invalid bundle index"),
        ("generalized_costs", [[np.nan]], "only finite values"),
        ("bundle_demand", [-1.0], "bundle_demand must be nonnegative"),
        ("total_profit", np.inf, "total_profit must be finite"),
        ("choice_tolerance", -1.0, "choice_tolerance"),
        ("profit_tolerance", np.nan, "profit_tolerance"),
        ("bundle_indices", [0], "numerical minimum-cost"),
    ],
)
def test_assignment_result_rejects_invalid_input(field, value, message) -> None:
    arguments = _valid_assignment_result_arguments()
    if message == "numerical minimum-cost":
        arguments.update(
            menu=BundleMenu(np.zeros(2), np.zeros((2, 1))),
            valuation=ValuationTable(np.zeros((1, 2)), np.zeros((1, 2, 1))),
            generalized_costs=np.array([[2.0, 1.0]]),
            bundle_margins=np.zeros(2),
            bundle_demand=np.array([1.0, 0.0]),
        )
    arguments[field] = np.asarray(value) if field not in {"menu", "total_profit"} else value

    with pytest.raises(ValueError, match=message):
        AssignmentResult(**arguments)


def test_assignment_result_accepts_an_empty_traveler_set() -> None:
    result = AssignmentResult(
        menu=BundleMenu(np.array([0.0]), np.array([[0.0]])),
        valuation=ValuationTable(np.empty((0, 1)), np.empty((0, 1, 1))),
        bundle_indices=np.empty(0, dtype=np.int64),
        generalized_costs=np.empty((0, 1)),
        bundle_margins=np.array([0.0]),
        bundle_demand=np.array([0.0]),
        total_profit=0.0,
    )

    assert result.bundle_indices.size == 0
    assert result.minimum_cost_gaps.shape == (0, 1)
    assert result.numerical_tie_mask.shape == (0, 1)
    assert result.optimistic_margin_gaps.shape == (0,)
