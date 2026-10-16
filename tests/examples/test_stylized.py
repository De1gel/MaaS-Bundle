"""Check network/resource consistency and valuation of the stylized instance."""

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from maas_bundle.design.menu import BundleMenu
from maas_bundle.examples import (
    DEFAULT_STYLIZED_CONFIG,
    HIGH_UNCERTAINTY_STYLIZED_CONFIG,
    build_stylized_graph,
    build_stylized_problem,
    draw_stylized_menu,
    initial_stylized_menu,
)
from maas_bundle.examples.stylized import STYLIZED_PATHS
from maas_bundle.examples.stylized_config import MODE_NAMES, MODE_USAGE


def test_network_paths_reproduce_the_mode_usage_matrix() -> None:
    graph = build_stylized_graph()
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 4
    actual_paths = {tuple(path) for path in nx.all_simple_edge_paths(graph, "O", "D")}
    assert actual_paths == set(STYLIZED_PATHS)
    usage = np.zeros((2, 3))
    for column, path in enumerate(STYLIZED_PATHS):
        for tail, head, key in path:
            edge = graph.edges[tail, head, key]
            usage[MODE_NAMES.index(edge["mode"]), column] += edge["resource_units"]
    np.testing.assert_array_equal(usage, MODE_USAGE)


@pytest.mark.parametrize("config", [DEFAULT_STYLIZED_CONFIG, HIGH_UNCERTAINTY_STYLIZED_CONFIG])
def test_oracle_payg_cost_and_full_coverage_agree_with_analytic_values(config) -> None:
    problem = build_stylized_problem(config)
    menu = BundleMenu(
        prices=np.zeros(3),
        allowances=np.array([[0, 0], [64, 64], [64, 64]], dtype=float),
    )
    valuations = problem.evaluate_valuations(menu)
    np.testing.assert_allclose(valuations.route_values[:, 0], config.payg_expected_costs)
    np.testing.assert_allclose(config.payg_expected_costs[:3], [144, 180, 209])
    means = np.array([(lower + upper) / 2 for lower, upper in config.frequency_supports])
    covered_costs = means * np.min(config.path_costs, axis=1)
    np.testing.assert_allclose(valuations.route_values[:, 1], covered_costs)
    np.testing.assert_allclose(valuations.expected_omega[:, 1:], 0, atol=1e-12)


def test_uncertainty_extension_preserves_network_and_mean_and_changes_valuation() -> None:
    config = HIGH_UNCERTAINTY_STYLIZED_CONFIG
    problem = build_stylized_problem(config)
    assert problem.route_oracles[1][0] is problem.route_oracles[3][0]
    low, high = problem.demand_distributions[1], problem.demand_distributions[3]
    assert low.first_moment(0, np.inf) == high.first_moment(0, np.inf) == 20
    assert (low.upper - low.lower) ** 2 / 12 == pytest.approx(16 / 3)
    assert (high.upper - high.lower) ** 2 / 12 == 48
    values = problem.evaluate_valuations(initial_stylized_menu(config)).route_values
    assert values[1, 0] == values[3, 0]
    assert values[3, 2] > values[1, 2]


def test_class_specific_demand_breaks_identical_pt_allowance_savings() -> None:
    problem = build_stylized_problem()
    assert [(d.lower, d.upper) for d in problem.demand_distributions] == [
        (14, 22),
        (16, 24),
        (18, 26),
    ]
    menu = BundleMenu(np.zeros(3), np.array([[0, 0], [44, 0], [44, 0]], dtype=float))
    values = problem.evaluate_valuations(menu).route_values
    np.testing.assert_allclose(values[:, 0] - values[:, 1], [72, 79, 84])


@pytest.mark.parametrize("supports", [((14, 22),), ((14, 22), (24, 16), (18, 26))])
def test_class_specific_demand_requires_three_valid_supports(supports) -> None:
    with pytest.raises(ValueError, match="baseline_frequency_supports"):
        replace(DEFAULT_STYLIZED_CONFIG, baseline_frequency_supports=supports)


@pytest.mark.parametrize(
    "config,assignment,profit",
    [
        (DEFAULT_STYLIZED_CONFIG, [1, 2, 2], 30),
        (HIGH_UNCERTAINTY_STYLIZED_CONFIG, [1, 2, 2, 2], 31),
    ],
)
def test_initial_menu_is_a_positive_profit_start(config, assignment, profit) -> None:
    problem = build_stylized_problem(config)
    menu = initial_stylized_menu(config)
    config.validate_menu(menu)
    result = problem.evaluate_assignment(menu)
    np.testing.assert_array_equal(result.bundle_indices, assignment)
    np.testing.assert_allclose(result.bundle_margins, [0, 20, 35])
    assert result.total_profit == pytest.approx(profit)


def test_menu_and_population_validation() -> None:
    config = DEFAULT_STYLIZED_CONFIG
    large_menu = BundleMenu(np.array([0, 240, 300]), np.array([[0, 0], [72, 80], [100, 120]]))
    config.validate_menu(large_menu)
    large_config = replace(
        config,
        initial_prices=tuple(large_menu.prices),
        initial_allowances=tuple(tuple(row) for row in large_menu.allowances),
    )
    result = build_stylized_problem(large_config).evaluate_assignment(
        initial_stylized_menu(large_config)
    )
    np.testing.assert_array_equal(result.bundle_indices, [0, 0, 0])
    with pytest.raises(ValueError, match="nondecreasing"):
        config.validate_menu(BundleMenu(np.array([0, 75, 40]), np.zeros((3, 2))))
    with pytest.raises(ValueError, match="PAYG"):
        config.validate_menu(BundleMenu(np.array([1, 40, 75]), np.zeros((3, 2))))
    with pytest.raises(ValueError, match="one entry"):
        replace(config, include_high_uncertainty=True)
    with pytest.raises(ValueError, match="sum to one"):
        replace(config, traveler_weights=(0.2, 0.2, 0.2))
    # Price ordering is required, componentwise allowance ordering is not.
    config.validate_menu(BundleMenu(np.array([0, 40, 75]), np.array([[0, 0], [32, 8], [24, 40]])))
    random = np.random.default_rng(20261002)
    for _ in range(10):
        menu = draw_stylized_menu(random, config)
        config.validate_menu(menu)
        assert np.all(menu.prices[1:] >= config.sampled_price_bounds[0])
        assert np.all(menu.prices[1:] <= config.sampled_price_bounds[1])
        assert np.all(menu.allowances[1:] >= config.sampled_allowance_bounds[0])
        assert np.all(menu.allowances[1:] <= config.sampled_allowance_bounds[1])
