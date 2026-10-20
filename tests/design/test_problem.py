"""Tests for static bundle-design instances and menu evaluation."""

import numpy as np
import pytest

from maas_bundle import (
    BundleDesignProblem,
    BundleMenu,
    RouteValueOracle,
    UniformDemandDistribution,
)


def _example_problem() -> BundleDesignProblem:
    return BundleDesignProblem(
        traveler_weights=np.array([10.0, 20.0]),
        unit_allowance_cost=np.array([0.2]),
        demand_distributions=(
            UniformDemandDistribution(0.0, 2.0),
            UniformDemandDistribution(0.0, 4.0),
        ),
        route_oracles=(
            (RouteValueOracle([[2.0, 0.0]]), RouteValueOracle([[0.5, 0.0]])),
            (RouteValueOracle([[0.5, 0.0]]), RouteValueOracle([[1.0, 0.0]])),
        ),
    )


def test_problem_validates_static_data_once_and_exposes_dimensions() -> None:
    problem = _example_problem()

    assert problem.traveler_count == 2
    assert problem.bundle_count == 2
    assert problem.mode_count == 1
    assert isinstance(problem.route_oracles, tuple)
    assert isinstance(problem.route_oracles[0], tuple)
    assert isinstance(problem.demand_distributions, tuple)
    assert not problem.traveler_weights.flags.writeable
    assert not problem.unit_allowance_cost.flags.writeable


def test_problem_evaluates_valuations_independently_of_assignment() -> None:
    problem = _example_problem()
    menu = BundleMenu(np.array([0.0, 1.0]), np.array([[0.0], [1.0]]))

    valuation = problem.evaluate_valuations(menu)

    np.testing.assert_allclose(valuation.route_values, [[2.0, 0.5], [1.0, 2.0]])
    np.testing.assert_allclose(valuation.expected_omega, np.zeros((2, 2, 1)))


def test_selected_valuations_and_exact_allowance_cache_match_direct_oracle(monkeypatch) -> None:
    oracle = RouteValueOracle([[0, 0], [2, 0.5], [4, 2]])
    problem = BundleDesignProblem(
        traveler_weights=np.ones(3),
        unit_allowance_cost=np.array([0.2]),
        route_oracles=((oracle, oracle, oracle),) * 3,
        demand_distributions=tuple(UniformDemandDistribution(0, n) for n in (1, 2, 3)),
    )
    original = RouteValueOracle.expected_valuation
    calls = []

    def counted(self, allowance, distribution, **kwargs):
        calls.append((tuple(allowance), distribution.upper))
        return original(self, allowance, distribution, **kwargs)

    monkeypatch.setattr(RouteValueOracle, "expected_valuation", counted)
    menu = BundleMenu(np.array([0, 1, 2]), np.array([[0], [0.5], [2]]))
    selected = problem.evaluate_valuations_for_travelers(menu, [2, 0])
    assert len(calls) == 6
    for row, traveler in enumerate((2, 0)):
        for bundle in range(3):
            reference = original(
                oracle, menu.allowances[bundle], problem.demand_distributions[traveler]
            )
            assert selected.route_values[row, bundle] == pytest.approx(reference.value)
            np.testing.assert_allclose(
                selected.expected_omega[row, bundle], reference.expected_omega
            )
    # Changing only prices must not rebuild any envelope.
    prices_only = BundleMenu(np.array([0, 4, 7]), menu.allowances)
    repeated = problem.evaluate_valuations_for_travelers(prices_only, [2, 0])
    assert len(calls) == 6
    np.testing.assert_array_equal(repeated.route_values, selected.route_values)
    # Only the changed bundle is invalidated, not PAYG or the other bundle.
    changed = BundleMenu(prices_only.prices, np.array([[0], [0.7], [2]]))
    updated = problem.evaluate_valuations_for_travelers(changed, [2, 0])
    assert len(calls) == 8
    assert not np.allclose(updated.route_values[:, 1], selected.route_values[:, 1])
    # Full separation must still visit the omitted traveler, with every bundle.
    full = problem.evaluate_valuations(changed)
    assert len(calls) == 11
    np.testing.assert_array_equal(full.route_values[[2, 0]], updated.route_values)
    assert full.route_values.shape == (3, 3)


@pytest.mark.parametrize("travelers", [[-1], [2], [0.5], [[0]]])
def test_selected_valuations_reject_invalid_traveler_indices(travelers) -> None:
    with pytest.raises(ValueError):
        _example_problem().evaluate_valuations_for_travelers(
            BundleMenu(np.zeros(2), np.zeros((2, 1))), travelers
        )


def test_problem_evaluates_complete_optimistic_assignment() -> None:
    problem = _example_problem()
    menu = BundleMenu(np.array([0.0, 1.0]), np.array([[0.0], [1.0]]))

    result = problem.evaluate_assignment(menu)

    assert result.menu is menu
    np.testing.assert_allclose(result.generalized_costs, [[2.0, 1.5], [1.0, 3.0]])
    np.testing.assert_allclose(result.bundle_margins, [0.0, 0.8])
    np.testing.assert_array_equal(result.bundle_indices, [1, 0])
    np.testing.assert_allclose(result.bundle_demand, [20.0, 10.0])
    assert result.total_profit == pytest.approx(8.0)


def _single_oracle_problem_arguments():
    return {
        "traveler_weights": np.array([1.0]),
        "unit_allowance_cost": np.array([1.0]),
        "demand_distributions": (UniformDemandDistribution(0.0, 1.0),),
        "route_oracles": ((RouteValueOracle([[1.0, 0.0]]),),),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("traveler_weights", [[1.0]], "nonempty vector"),
        ("traveler_weights", [], "nonempty vector"),
        ("traveler_weights", [np.nan], "finite and nonnegative"),
        ("traveler_weights", [-1.0], "finite and nonnegative"),
        ("unit_allowance_cost", [[1.0]], "nonempty vector"),
        ("unit_allowance_cost", [], "nonempty vector"),
        ("unit_allowance_cost", [np.inf], "finite and nonnegative"),
        ("unit_allowance_cost", [-1.0], "finite and nonnegative"),
        ("demand_distributions", (), "one entry per traveler"),
        ("route_oracles", (), "one row per traveler"),
        ("route_oracles", ((),), "at least one bundle"),
        (
            "route_oracles",
            ((RouteValueOracle([[1.0, 0.0, 0.0]]),),),
            "problem allowance dimensions",
        ),
    ],
)
def test_problem_rejects_invalid_static_data(field, value, message) -> None:
    arguments = _single_oracle_problem_arguments()
    if field in {"traveler_weights", "unit_allowance_cost"}:
        value = np.asarray(value)
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        BundleDesignProblem(**arguments)


def test_problem_rejects_ragged_oracle_rows() -> None:
    arguments = _single_oracle_problem_arguments()
    arguments["traveler_weights"] = np.array([1.0, 1.0])
    arguments["demand_distributions"] *= 2
    arguments["route_oracles"] = (
        (RouteValueOracle([[1.0, 0.0]]),),
        (RouteValueOracle([[1.0, 0.0]]), RouteValueOracle([[1.0, 0.0]])),
    )

    with pytest.raises(ValueError, match="same bundles in every row"):
        BundleDesignProblem(**arguments)


@pytest.mark.parametrize(
    ("menu", "message"),
    [
        (BundleMenu(np.zeros(3), np.zeros((3, 1))), "one entry per problem bundle"),
        (BundleMenu(np.zeros(2), np.zeros((2, 2))), "problem allowance dimensions"),
        (BundleMenu(np.array([1.0, 2.0]), np.zeros((2, 1))), "bundle zero"),
        (BundleMenu(np.zeros(2), np.array([[1.0], [0.0]])), "bundle zero"),
    ],
)
def test_problem_rejects_menu_incompatible_with_model(menu, message) -> None:
    with pytest.raises(ValueError, match=message):
        _example_problem().evaluate_valuations(menu)
