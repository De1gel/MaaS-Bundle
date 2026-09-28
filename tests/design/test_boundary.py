"""Tests for pairwise boundaries and critical-traveler separation."""

import numpy as np
import pytest

from maas_bundle import (
    BundleMenu,
    ChoiceConstraintIndex,
    ChoiceConstraintViolation,
    CriticalConstraintSet,
    ValuationTable,
    assigned_pairwise_values,
    choice_constraint_slacks,
    initialize_critical_constraints,
    separate_choice_constraints,
)


@pytest.fixture
def menu() -> BundleMenu:
    return BundleMenu(
        prices=np.array([0.0, 1.0, 2.0]),
        allowances=np.zeros((3, 1)),
    )


@pytest.fixture
def valuation() -> ValuationTable:
    return ValuationTable(
        route_values=np.array(
            [
                [1.0, 2.0, 4.0],
                [3.0, 1.0, 2.0],
                [2.0, 1.0, 1.5],
            ]
        ),
        expected_omega=np.zeros((3, 3, 1)),
    )


@pytest.fixture
def assignments() -> np.ndarray:
    return np.array([0, 1, 1], dtype=np.int64)


def test_computes_pairwise_values_from_each_assigned_bundle(valuation, assignments) -> None:
    differences = assigned_pairwise_values(valuation, assignments)

    np.testing.assert_allclose(
        differences,
        [
            [0.0, 1.0, 3.0],
            [2.0, 0.0, 1.0],
            [1.0, 0.0, 0.5],
        ],
    )
    assert not differences.flags.writeable


def test_computes_assignment_constraint_slacks(menu, valuation, assignments) -> None:
    slacks = choice_constraint_slacks(menu, valuation, assignments)

    np.testing.assert_allclose(
        slacks,
        [
            [0.0, 2.0, 5.0],
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 1.5],
        ],
    )
    assert not slacks.flags.writeable


def test_initializes_smallest_slack_traveler_for_each_occupied_pair(
    menu,
    valuation,
    assignments,
) -> None:
    pool = initialize_critical_constraints(menu, valuation, assignments)

    assert pool.traveler_count == 3
    assert pool.bundle_count == 3
    assert pool.constraint_count == 4
    np.testing.assert_array_equal(pool.active_travelers(0, 1), [0])
    np.testing.assert_array_equal(pool.active_travelers(0, 2), [0])
    np.testing.assert_array_equal(pool.active_travelers(1, 0), [2])
    np.testing.assert_array_equal(pool.active_travelers(1, 2), [2])
    np.testing.assert_array_equal(pool.active_travelers(2, 0), [])
    assert not pool.active_mask.flags.writeable
    assert not pool.assigned_bundles.flags.writeable


def test_initialization_includes_ties_and_requested_number(menu, assignments) -> None:
    valuation = ValuationTable(
        route_values=np.array(
            [
                [1.0, 2.0, 4.0],
                [2.0, 1.0, 2.0],
                [2.0 + 5e-13, 1.0, 2.5],
            ]
        ),
        expected_omega=np.zeros((3, 3, 1)),
    )

    tied = initialize_critical_constraints(
        menu,
        valuation,
        assignments,
        tie_tolerance=1e-12,
    )
    requested_two = initialize_critical_constraints(
        menu,
        valuation,
        assignments,
        initial_per_pair=2,
        tie_tolerance=0.0,
    )

    np.testing.assert_array_equal(tied.active_travelers(1, 0), [1, 2])
    np.testing.assert_array_equal(requested_two.active_travelers(1, 0), [1, 2])


def test_constraint_set_adds_constraints_monotonically(assignments) -> None:
    pool = CriticalConstraintSet(assignments, bundle_count=3)
    index = ChoiceConstraintIndex(1, 1, 0)

    assert not pool.contains(index)
    assert pool.add(index)
    assert pool.contains(index)
    assert not pool.add(index)
    assert pool.add_all((index, ChoiceConstraintIndex(2, 1, 2))) == 1
    assert pool.constraint_count == 2


def test_separation_finds_most_violated_omitted_traveler(menu, valuation, assignments) -> None:
    pool = initialize_critical_constraints(menu, valuation, assignments)
    candidate = ValuationTable(
        route_values=np.array(
            [
                [1.0, 2.0, 4.0],
                [0.0, 1.0, 2.0],
                [0.5, 1.0, 1.5],
            ]
        ),
        expected_omega=np.zeros((3, 3, 1)),
    )

    result = separate_choice_constraints(menu, candidate, pool)

    assert not result.is_feasible
    assert result.can_add_constraints
    assert len(result.violations) == 1
    assert len(result.additions) == 1
    assert result.violations[0].index == ChoiceConstraintIndex(1, 1, 0)
    assert result.violations[0].slack == pytest.approx(-2.0)
    assert result.additions[0].index == ChoiceConstraintIndex(1, 1, 0)

    assert pool.add_all(tuple(item.index for item in result.additions)) == 1
    repeated = separate_choice_constraints(menu, candidate, pool)
    assert not repeated.is_feasible
    assert not repeated.can_add_constraints


def test_separation_reports_complete_feasibility(menu, valuation, assignments) -> None:
    pool = initialize_critical_constraints(menu, valuation, assignments)

    result = separate_choice_constraints(menu, valuation, pool)

    assert result.is_feasible
    assert not result.can_add_constraints
    assert result.violations == ()
    assert result.additions == ()


@pytest.mark.parametrize("slack", [0.0, 1.0, np.inf, np.nan])
def test_violation_rejects_nonnegative_or_nonfinite_slack(slack) -> None:
    with pytest.raises(ValueError, match="finite negative slack"):
        ChoiceConstraintViolation(ChoiceConstraintIndex(0, 0, 1), slack)


@pytest.mark.parametrize(
    ("assigned", "bundle_count", "message"),
    [
        ([[0]], 1, "must be one-dimensional"),
        ([0.0], 1, "must contain integers"),
        ([0], 0, "positive integer"),
        ([-1], 1, "invalid bundle index"),
        ([1], 1, "invalid bundle index"),
    ],
)
def test_constraint_set_rejects_invalid_assignment(assigned, bundle_count, message) -> None:
    with pytest.raises(ValueError, match=message):
        CriticalConstraintSet(assigned, bundle_count)


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (ChoiceConstraintIndex(-1, 0, 1), "traveler index"),
        (ChoiceConstraintIndex(2, 0, 1), "traveler index"),
        (ChoiceConstraintIndex(0, -1, 1), "assigned bundle index"),
        (ChoiceConstraintIndex(0, 2, 1), "assigned bundle index"),
        (ChoiceConstraintIndex(0, 0, -1), "alternative bundle index"),
        (ChoiceConstraintIndex(0, 0, 2), "alternative bundle index"),
        (ChoiceConstraintIndex(0, 0, 0), "two different bundles"),
        (ChoiceConstraintIndex(0, 1, 0), "does not match"),
    ],
)
def test_constraint_set_rejects_invalid_constraint_index(index, message) -> None:
    pool = CriticalConstraintSet([0, 1], bundle_count=2)

    with pytest.raises(ValueError, match=message):
        pool.add(index)


def test_active_travelers_rejects_invalid_bundle_pair(assignments) -> None:
    pool = CriticalConstraintSet(assignments, bundle_count=3)

    with pytest.raises(ValueError, match="two different bundles"):
        pool.active_travelers(1, 1)


@pytest.mark.parametrize(
    ("assigned", "message"),
    [
        ([0, 1], "one entry per traveler"),
        ([0.0, 1.0, 1.0], "must contain integers"),
        ([0, 1, 3], "invalid bundle index"),
    ],
)
def test_pairwise_values_rejects_invalid_assignment(valuation, assigned, message) -> None:
    with pytest.raises(ValueError, match=message):
        assigned_pairwise_values(valuation, assigned)


@pytest.mark.parametrize(
    ("menu", "valuation", "message"),
    [
        (
            BundleMenu(np.zeros(2), np.zeros((2, 1))),
            ValuationTable(np.zeros((1, 3)), np.zeros((1, 3, 1))),
            "same bundles",
        ),
        (
            BundleMenu(np.zeros(3), np.zeros((3, 2))),
            ValuationTable(np.zeros((1, 3)), np.zeros((1, 3, 1))),
            "same allowance dimensions",
        ),
    ],
)
def test_slacks_rejects_incompatible_menu_and_valuation(menu, valuation, message) -> None:
    with pytest.raises(ValueError, match=message):
        choice_constraint_slacks(menu, valuation, [0])


@pytest.mark.parametrize(
    ("initial_per_pair", "tie_tolerance", "message"),
    [
        (0, 0.0, "positive integer"),
        (1.5, 0.0, "positive integer"),
        (1, -1.0, "tie_tolerance"),
    ],
)
def test_initialization_rejects_invalid_options(
    menu,
    valuation,
    assignments,
    initial_per_pair,
    tie_tolerance,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        initialize_critical_constraints(
            menu,
            valuation,
            assignments,
            initial_per_pair=initial_per_pair,
            tie_tolerance=tie_tolerance,
        )


def test_separation_rejects_incompatible_dimensions(menu, valuation, assignments) -> None:
    wrong_travelers = CriticalConstraintSet([0, 1], bundle_count=3)
    wrong_bundles = CriticalConstraintSet(assignments, bundle_count=4)

    with pytest.raises(ValueError, match="same travelers"):
        separate_choice_constraints(menu, valuation, wrong_travelers)
    with pytest.raises(ValueError, match="same bundles"):
        separate_choice_constraints(menu, valuation, wrong_bundles)


def test_separation_rejects_invalid_tolerance(menu, valuation, assignments) -> None:
    pool = CriticalConstraintSet(assignments, bundle_count=3)

    with pytest.raises(ValueError, match="violation_tolerance"):
        separate_choice_constraints(menu, valuation, pool, violation_tolerance=np.nan)
