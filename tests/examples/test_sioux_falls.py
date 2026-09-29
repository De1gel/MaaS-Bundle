"""Tests for the TNTP Sioux Falls MaaS benchmark extension."""

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import linprog

import maas_bundle.examples.sioux_falls as sioux_module
from maas_bundle.examples.sioux_falls import (
    DIRECTED_LINK_COUNT,
    NODE_COUNT,
    SOURCE_TOTAL_OD_FLOW,
    build_sioux_falls_graph,
    build_sioux_falls_problem,
    initial_sioux_falls_menu,
    sioux_falls_od_matrix,
)
from maas_bundle.examples.sioux_falls_config import DEFAULT_SIOUX_FALLS_CONFIG


def test_sioux_falls_source_network_and_od_table() -> None:
    graph = build_sioux_falls_graph()
    demand = sioux_falls_od_matrix()

    assert graph.number_of_nodes() == NODE_COUNT == 24
    assert graph.number_of_edges() == DIRECTED_LINK_COUNT == 76
    assert graph.edges[1, 2]["source_free_flow_time"] == 6
    assert graph.edges[1, 2]["free_flow_time"] == pytest.approx(3.6)
    assert graph.graph["time_unit"] == "minute"
    assert graph.edges[2, 1]["capacity"] == pytest.approx(25900.20064)
    assert demand.shape == (24, 24)
    assert np.count_nonzero(demand) == 528
    assert np.sum(demand) == SOURCE_TOTAL_OD_FLOW


def test_sioux_falls_problem_can_select_highest_demand_od_pairs() -> None:
    full_problem = build_sioux_falls_problem()
    problem = build_sioux_falls_problem(maximum_od_pairs=3)

    assert full_problem.traveler_count == 1056
    assert problem.traveler_count == 6
    assert problem.bundle_count == 3
    assert problem.mode_count == 2
    assert np.sum(problem.traveler_weights) == pytest.approx(1.0)
    assert [type(item).__name__ for item in problem.demand_distributions] == [
        "UniformDemandDistribution",
        "TriangularDemandDistribution",
    ] * 3
    assert all(len(row) == 3 for row in problem.route_oracles)
    assert all(row[0] is row[1] is row[2] for row in problem.route_oracles)
    assert all(
        row[0] is full_row[0]
        for row, full_row in zip(problem.route_oracles, full_problem.route_oracles[:6], strict=True)
    )
    assert problem.demand_distributions[0].lower == 8
    assert problem.demand_distributions[0].upper == 20
    assert problem.demand_distributions[1].mode == 32
    assert problem.demand_distributions[1].upper == 44


def test_sioux_falls_paths_include_a_better_route_beyond_the_first_five() -> None:
    graph = nx.DiGraph()
    for via in range(1, 7):
        # The sixth path is slower in free-flow time but cheaper with PAYG.
        attributes = {
            "free_flow_time": 1.0 + 0.01 * via,
            "capacity": 20000.0 if via == 6 else 5000.0,
        }
        graph.add_edge(0, via, **attributes)
        graph.add_edge(via, 7, **attributes)

    times, usage = sioux_module._od_path_data(graph, 0, 7)
    assert times.size == 6
    costs = 0.18 * times + np.array([0.02, 0.10]) @ usage
    oracle = sioux_module.build_route_value_oracle(costs, usage, [0.45, 0.75])
    expected = 0.18 * 2.12 + 2 * 0.02 + 2 * 0.45
    assert oracle.value(1.0, [0.0, 0.0]) == pytest.approx(expected)
    first_five = np.argsort(times)[:5]
    truncated = sioux_module.build_route_value_oracle(
        costs[first_five], usage[:, first_five], [0.45, 0.75]
    )
    assert truncated.value(1.0, [0.0, 0.0]) > expected + 0.5


@pytest.mark.parametrize(("origin", "destination"), [(1, 24), (10, 16), (3, 20)])
@pytest.mark.parametrize("segment", [0, 1])
def test_full_network_oracle_matches_independent_link_flow_lp(origin, destination, segment) -> None:
    """Verify the all-path oracle against a formulation with every network arc."""

    graph = build_sioux_falls_graph()
    edges = list(graph.edges(data=True))
    edge_count = len(edges)
    incidence = np.zeros((NODE_COUNT, edge_count + 2))
    usage = np.zeros((2, edge_count + 2))
    costs = np.zeros(edge_count + 2)
    config = DEFAULT_SIOUX_FALLS_CONFIG
    beta = config.value_of_time_usd_per_minute[segment]
    preference = config.preference_usd_per_link_use[segment]
    for column, (tail, head, edge) in enumerate(edges):
        incidence[tail - 1, column] = 1.0
        incidence[head - 1, column] = -1.0
        mode = 0 if edge["capacity"] >= config.trunk_capacity_threshold_source_flow_units else 1
        usage[mode, column] = 1.0
        costs[column] = beta * edge["free_flow_time"] + preference[mode]
    usage[:, -2:] = -np.eye(2)
    costs[-2:] = config.payg_usd_per_link_use
    oracle = sioux_module._od_route_oracles(origin, destination)[segment]
    for mu in (0.0, 8.0, 32.0):
        balance = np.zeros(NODE_COUNT)
        balance[origin - 1] = mu
        balance[destination - 1] = -mu
        for allowance in ([0.0, 0.0], [32.0, 16.0], [0.0, 120.0], [120.0, 0.0]):
            reference = linprog(
                costs,
                A_ub=usage,
                b_ub=allowance,
                A_eq=incidence,
                b_eq=balance,
                bounds=(0, None),
                method="highs",
            )
            assert reference.success, reference.message
            assert oracle.value(mu, allowance) == pytest.approx(reference.fun, abs=1e-8)


def test_sioux_falls_reuses_precomputed_geometry_across_menus(monkeypatch) -> None:
    cached = sioux_module._od_route_oracles(1, 24)

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail("cached oracle must not rebuild paths or enumerate vertices")

    monkeypatch.setattr(sioux_module, "_od_path_data", unexpected_rebuild)
    monkeypatch.setattr(sioux_module, "build_route_value_oracle", unexpected_rebuild)
    assert sioux_module._od_route_oracles(1, 24) is cached
    for oracle in cached:
        assert np.isfinite(oracle.value(3.0, [2.0, 8.0]))
        assert np.isfinite(oracle.value(9.0, [20.0, 0.0]))


def test_sioux_falls_changed_economic_config_cannot_reuse_old_vertices() -> None:
    config = DEFAULT_SIOUX_FALLS_CONFIG
    changed = replace(config, value_of_time_usd_per_minute=(0.36, 0.64))
    original = sioux_module._od_route_oracles(1, 24, config)
    updated = sioux_module._od_route_oracles(1, 24, changed)
    assert updated is not original
    assert updated[0].value(8.0, [32.0, 16.0]) > original[0].value(8.0, [32.0, 16.0])
    menu = initial_sioux_falls_menu(
        config=replace(config, initial_prices_usd_per_month=(0, 22, 44))
    )
    assert menu.prices.tolist() == [0, 22, 44]


@pytest.mark.parametrize("maximum_od_pairs", [0, -1, 1.5])
def test_sioux_falls_problem_rejects_invalid_od_limit(maximum_od_pairs) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_sioux_falls_problem(maximum_od_pairs=maximum_od_pairs)
