"""Tests for the TNTP Sioux Falls MaaS benchmark extension."""

from dataclasses import replace

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


def test_five_modes_share_each_physical_link() -> None:
    graph = sioux_module.build_sioux_falls_multimodal_graph()
    assert graph.number_of_nodes() == 24
    assert graph.number_of_edges() == 380
    for tail, head in build_sioux_falls_graph().edges:
        assert set(graph[tail][head]) == set(sioux_module.MODE_NAMES)
        for mode, name in enumerate(sioux_module.MODE_NAMES):
            assert graph[tail][head][name]["mode_index"] == mode


@pytest.mark.parametrize(
    "field,value",
    [
        ("payg_usd_per_link_use", (0.45, 0.75)),
        ("provision_usd_per_link_use", (0.05,) * 4 + (-0.1,)),
        ("preference_usd_per_link_use", ((0.0,) * 5,)),
        ("initial_allowances_link_uses_per_month", ((0.0,) * 2,) * 3),
        ("travel_time_factors", (1.0, 2.0)),
        ("travel_time_factors", (1.0, 1.0, 1.0, 1.0, 0.0)),
    ],
)
def test_five_mode_config_rejects_inconsistent_dimensions(field, value) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_SIOUX_FALLS_CONFIG, **{field: value})


def test_sioux_falls_problem_can_select_highest_demand_od_pairs() -> None:
    full_problem = build_sioux_falls_problem()
    problem = build_sioux_falls_problem(maximum_od_pairs=3)

    assert full_problem.traveler_count == 1056
    assert problem.traveler_count == 6
    assert problem.bundle_count == 3
    assert full_problem.mode_count == problem.mode_count == 5
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


@pytest.mark.parametrize(("origin", "destination"), [(1, 24), (10, 16), (3, 20)])
@pytest.mark.parametrize("segment", [0, 1])
def test_full_network_oracle_matches_independent_link_flow_lp(origin, destination, segment) -> None:
    """Verify the all-path oracle against a formulation with every network arc."""

    graph = sioux_module.build_sioux_falls_multimodal_graph()
    edges = list(graph.edges(data=True))
    edge_count = len(edges)
    incidence = np.zeros((NODE_COUNT, edge_count + 5))
    usage = np.zeros((5, edge_count + 5))
    costs = np.zeros(edge_count + 5)
    config = DEFAULT_SIOUX_FALLS_CONFIG
    beta = config.value_of_time_usd_per_minute[segment]
    preference = config.preference_usd_per_link_use[segment]
    for column, (tail, head, edge) in enumerate(edges):
        incidence[tail - 1, column] = 1.0
        incidence[head - 1, column] = -1.0
        mode = edge["mode_index"]
        usage[mode, column] = 1.0
        costs[column] = beta * edge["travel_time"] + preference[mode]
    usage[:, -5:] = -np.eye(5)
    costs[-5:] = config.payg_usd_per_link_use
    oracle = sioux_module._od_route_oracles(origin, destination)[segment]
    for mu in (0.0, 8.0, 32.0):
        balance = np.zeros(NODE_COUNT)
        balance[origin - 1] = mu
        balance[destination - 1] = -mu
        for allowance in (np.zeros(5), [16, 16, 4, 6, 6], *list(120 * np.eye(5))):
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

    monkeypatch.setattr(sioux_module, "build_sioux_falls_multimodal_graph", unexpected_rebuild)
    monkeypatch.setattr(sioux_module, "build_network_route_value_oracle", unexpected_rebuild)
    assert sioux_module._od_route_oracles(1, 24) is cached
    for oracle in cached:
        assert np.isfinite(oracle.value(3.0, [2.0, 8.0, 1.0, 4.0, 6.0]))
        assert np.isfinite(oracle.value(9.0, [20.0, 0.0, 4.0, 2.0, 8.0]))


def test_sioux_falls_changed_economic_config_cannot_reuse_old_vertices() -> None:
    config = DEFAULT_SIOUX_FALLS_CONFIG
    changed = replace(config, value_of_time_usd_per_minute=(0.36, 0.64))
    original = sioux_module._od_route_oracles(1, 24, config)
    updated = sioux_module._od_route_oracles(1, 24, changed)
    assert updated is not original
    assert updated[0].value(8.0, [16.0, 16.0, 4.0, 6.0, 6.0]) > original[0].value(
        8.0, [16.0, 16.0, 4.0, 6.0, 6.0]
    )
    menu = initial_sioux_falls_menu(
        config=replace(config, initial_prices_usd_per_month=(0, 22, 44))
    )
    assert menu.prices.tolist() == [0, 22, 44]


@pytest.mark.parametrize("maximum_od_pairs", [0, -1, 1.5])
def test_sioux_falls_problem_rejects_invalid_od_limit(maximum_od_pairs) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_sioux_falls_problem(maximum_od_pairs=maximum_od_pairs)
