"""Large synthetic MaaS extension of the standard Sioux Falls network.

The 24-node, 76-link network and 360,600-trip OD table are transcribed from
the Transportation Networks for Research TNTP data.  The original data are a
road traffic-assignment instance; the two MaaS credit modes, traveler demand
distributions, mode preferences, and bundle costs below are synthetic and are
kept explicit so experimental results do not imply that they came from TNTP.

Source: https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls
The source repository permits the donated data for academic research and asks
users to cite the Transportation Networks for Research Core Team.
"""

from __future__ import annotations

from functools import lru_cache

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.examples.sioux_falls_config import (
    DEFAULT_SIOUX_FALLS_CONFIG,
    TNTP_TIME_TO_MINUTES,
    SiouxFallsExperimentConfig,
)
from maas_bundle.oracle.demand_distribution import (
    TriangularDemandDistribution,
    UniformDemandDistribution,
)
from maas_bundle.oracle.route_value import RouteValueOracle, build_route_value_oracle

SOURCE_URL = "https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls"
SOURCE_TOTAL_OD_FLOW = 360_600.0
NODE_COUNT = 24
DIRECTED_LINK_COUNT = 76
MODE_NAMES = ("trunk", "feeder")
TRAVELER_SEGMENTS_PER_OD = 2

# The TNTP network is symmetric. Each tuple is
# (tail, head, source capacity, source time in 0.01 hour), inserted both ways.
_UNDIRECTED_LINKS = (
    (1, 2, 25900.200640, 6.0),
    (1, 3, 23403.473190, 4.0),
    (2, 6, 4958.180928, 5.0),
    (3, 4, 17110.523720, 4.0),
    (3, 12, 23403.473190, 4.0),
    (4, 5, 17782.794100, 2.0),
    (4, 11, 4908.826730, 6.0),
    (5, 6, 4947.995469, 4.0),
    (5, 9, 10000.000000, 5.0),
    (6, 8, 4898.587646, 2.0),
    (7, 8, 7841.811310, 3.0),
    (7, 18, 23403.473190, 2.0),
    (8, 9, 5050.193156, 10.0),
    (8, 16, 5045.822583, 5.0),
    (9, 10, 13915.788420, 3.0),
    (10, 11, 10000.000000, 5.0),
    (10, 15, 13512.001550, 6.0),
    (10, 16, 4854.917717, 4.0),
    (10, 17, 4993.510694, 8.0),
    (11, 12, 4908.826730, 6.0),
    (11, 14, 4876.508287, 4.0),
    (12, 13, 25900.200640, 3.0),
    (13, 24, 5091.256152, 4.0),
    (14, 15, 5127.526119, 5.0),
    (14, 23, 4924.790605, 4.0),
    (15, 19, 14564.753150, 3.0),
    (15, 22, 9599.180565, 3.0),
    (16, 17, 5229.910063, 2.0),
    (16, 18, 19679.896710, 3.0),
    (17, 19, 4823.950831, 2.0),
    (18, 20, 23403.473190, 4.0),
    (19, 20, 5002.607563, 4.0),
    (20, 21, 5059.912340, 6.0),
    (20, 22, 5075.697193, 5.0),
    (21, 22, 5229.910063, 2.0),
    (21, 24, 4885.357564, 3.0),
    (22, 23, 5000.000000, 4.0),
    (23, 24, 5078.508436, 2.0),
)

# OD flow in hundreds of TNTP scaled flow units, in node order 1..24.
_OD_FLOW_HUNDREDS = (
    (0, 1, 1, 5, 2, 3, 5, 8, 5, 13, 5, 2, 5, 3, 5, 5, 4, 1, 3, 3, 1, 4, 3, 1),
    (1, 0, 1, 2, 1, 4, 2, 4, 2, 6, 2, 1, 3, 1, 1, 4, 2, 0, 1, 1, 0, 1, 0, 0),
    (1, 1, 0, 2, 1, 3, 1, 2, 1, 3, 3, 2, 1, 1, 1, 2, 1, 0, 0, 0, 0, 1, 1, 0),
    (5, 2, 2, 0, 5, 4, 4, 7, 7, 12, 14, 6, 6, 5, 5, 8, 5, 1, 2, 3, 2, 4, 5, 2),
    (2, 1, 1, 5, 0, 2, 2, 5, 8, 10, 5, 2, 2, 1, 2, 5, 2, 0, 1, 1, 1, 2, 1, 0),
    (3, 4, 3, 4, 2, 0, 4, 8, 4, 8, 4, 2, 2, 1, 2, 9, 5, 1, 2, 3, 1, 2, 1, 1),
    (5, 2, 1, 4, 2, 4, 0, 10, 6, 19, 5, 7, 4, 2, 5, 14, 10, 2, 4, 5, 2, 5, 2, 1),
    (8, 4, 2, 7, 5, 8, 10, 0, 8, 16, 8, 6, 6, 4, 6, 22, 14, 3, 7, 9, 4, 5, 3, 2),
    (5, 2, 1, 7, 8, 4, 6, 8, 0, 28, 14, 6, 6, 6, 9, 14, 9, 2, 4, 6, 3, 7, 5, 2),
    (13, 6, 3, 12, 10, 8, 19, 16, 28, 0, 40, 20, 19, 21, 40, 44, 39, 7, 18, 25, 12, 26, 18, 8),
    (5, 2, 3, 15, 5, 4, 5, 8, 14, 39, 0, 14, 10, 16, 14, 14, 10, 1, 4, 6, 4, 11, 13, 6),
    (2, 1, 2, 6, 2, 2, 7, 6, 6, 20, 14, 0, 13, 7, 7, 7, 6, 2, 3, 4, 3, 7, 7, 5),
    (5, 3, 1, 6, 2, 2, 4, 6, 6, 19, 10, 13, 0, 6, 7, 6, 5, 1, 3, 6, 6, 13, 8, 8),
    (3, 1, 1, 5, 1, 1, 2, 4, 6, 21, 16, 7, 6, 0, 13, 7, 7, 1, 3, 5, 4, 12, 11, 4),
    (5, 1, 1, 5, 2, 2, 5, 6, 10, 40, 14, 7, 7, 13, 0, 12, 15, 2, 8, 11, 8, 26, 10, 4),
    (5, 4, 2, 8, 5, 9, 14, 22, 14, 44, 14, 7, 6, 7, 12, 0, 28, 5, 13, 16, 6, 12, 5, 3),
    (4, 2, 1, 5, 2, 5, 10, 14, 9, 39, 10, 6, 5, 7, 15, 28, 0, 6, 17, 17, 6, 17, 6, 3),
    (1, 0, 0, 1, 0, 1, 2, 3, 2, 7, 2, 2, 1, 1, 2, 5, 6, 0, 3, 4, 1, 3, 1, 0),
    (3, 1, 0, 2, 1, 2, 4, 7, 4, 18, 4, 3, 3, 3, 8, 13, 17, 3, 0, 12, 4, 12, 3, 1),
    (3, 1, 0, 3, 1, 3, 5, 9, 6, 25, 6, 5, 6, 5, 11, 16, 17, 4, 12, 0, 12, 24, 7, 4),
    (1, 0, 0, 2, 1, 1, 2, 4, 3, 12, 4, 3, 6, 4, 8, 6, 6, 1, 4, 12, 0, 18, 7, 5),
    (4, 1, 1, 4, 2, 2, 5, 5, 7, 26, 11, 7, 13, 12, 26, 12, 17, 3, 12, 24, 18, 0, 21, 11),
    (3, 0, 1, 5, 1, 1, 2, 3, 5, 18, 13, 7, 8, 11, 10, 5, 6, 1, 3, 7, 7, 21, 0, 7),
    (1, 0, 0, 2, 0, 1, 1, 2, 2, 8, 6, 5, 7, 4, 4, 3, 3, 0, 1, 4, 5, 11, 7, 0),
)


def build_sioux_falls_graph() -> nx.DiGraph:
    """Return the source network with physical travel times in minutes.

    Source length is an arbitrary copy of source time, not a distance. Source
    capacity and OD flow are used only for classification and relative weights.
    """

    graph = nx.DiGraph(name="TNTP Sioux Falls", time_unit="minute")
    graph.add_nodes_from(range(1, NODE_COUNT + 1))
    for tail, head, capacity, travel_time in _UNDIRECTED_LINKS:
        attributes = {
            "capacity": capacity,
            "source_length": travel_time,
            "source_free_flow_time": travel_time,
            "free_flow_time": TNTP_TIME_TO_MINUTES * travel_time,
        }
        graph.add_edge(tail, head, **attributes)
        graph.add_edge(head, tail, **attributes)
    return graph


def sioux_falls_od_matrix() -> NDArray[np.float64]:
    """Return an independent 24-by-24 TNTP OD-flow matrix."""

    return 100.0 * np.asarray(_OD_FLOW_HUNDREDS, dtype=float)


def build_sioux_falls_problem(
    *,
    maximum_od_pairs: int | None = None,
    config: SiouxFallsExperimentConfig = DEFAULT_SIOUX_FALLS_CONFIG,
) -> BundleDesignProblem:
    """Build the large two-segment-per-OD synthetic MaaS instance."""

    if maximum_od_pairs is not None and (
        not isinstance(maximum_od_pairs, (int, np.integer)) or maximum_od_pairs <= 0
    ):
        raise ValueError("maximum_od_pairs must be a positive integer or None")

    od_matrix = sioux_falls_od_matrix()
    positive_pairs = [
        (float(od_matrix[origin - 1, destination - 1]), origin, destination)
        for origin in range(1, NODE_COUNT + 1)
        for destination in range(1, NODE_COUNT + 1)
        if od_matrix[origin - 1, destination - 1] > 0
    ]
    positive_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    if maximum_od_pairs is not None:
        positive_pairs = positive_pairs[: int(maximum_od_pairs)]

    total_selected_flow = sum(flow for flow, _, _ in positive_pairs)
    traveler_weights: list[float] = []
    route_oracles = []
    distributions = []
    for flow, origin, destination in positive_pairs:
        od_oracles = _od_route_oracles(origin, destination, config)
        for segment, share in enumerate(config.segment_shares):
            oracle = od_oracles[segment]
            traveler_weights.append(share * flow / total_selected_flow)
            route_oracles.append((oracle, oracle, oracle))
            distributions.append(
                UniformDemandDistribution(*config.low_frequency_trips_per_month)
                if segment == 0
                else TriangularDemandDistribution(*config.high_frequency_trips_per_month)
            )
    return BundleDesignProblem(
        traveler_weights=np.asarray(traveler_weights),
        unit_allowance_cost=np.asarray(config.provision_usd_per_link_use),
        route_oracles=tuple(route_oracles),
        demand_distributions=tuple(distributions),
    )


def initial_sioux_falls_menu(
    *,
    config: SiouxFallsExperimentConfig = DEFAULT_SIOUX_FALLS_CONFIG,
) -> BundleMenu:
    """Return one feasible PAYG/basic/premium warm start."""

    return BundleMenu(
        prices=np.asarray(config.initial_prices_usd_per_month),
        allowances=np.asarray(config.initial_allowances_link_uses_per_month),
    )


@lru_cache(maxsize=2 * NODE_COUNT * (NODE_COUNT - 1))
def _od_route_oracles(
    origin: int,
    destination: int,
    config: SiouxFallsExperimentConfig = DEFAULT_SIOUX_FALLS_CONFIG,
) -> tuple[RouteValueOracle, ...]:
    """Cache complete dual vertices for both segments of this fixed benchmark.

    Geometry is independent of menus, allowances, and trip-frequency draws.
    Immutable oracles are shared across bundle options and repeated runs in
    this process; the temporary all-path arrays need not remain in memory.
    """

    path_costs, mode_usage = _od_path_data(
        build_sioux_falls_graph(),
        origin,
        destination,
        trunk_capacity_threshold=config.trunk_capacity_threshold_source_flow_units,
    )
    oracles = []
    for value_of_time, preference in zip(
        config.value_of_time_usd_per_minute,
        config.preference_usd_per_link_use,
        strict=True,
    ):
        generalized_costs = value_of_time * path_costs + np.asarray(preference) @ mode_usage
        oracles.append(
            build_route_value_oracle(
                generalized_costs,
                mode_usage,
                payg_tariff=np.asarray(config.payg_usd_per_link_use),
            )
        )
    return tuple(oracles)


def _od_path_data(
    graph: nx.DiGraph,
    origin: int,
    destination: int,
    *,
    trunk_capacity_threshold: float = (
        DEFAULT_SIOUX_FALLS_CONFIG.trunk_capacity_threshold_source_flow_units
    ),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Use every simple OD path, without a path-count or length cutoff.

    All benchmark edge costs are positive and mode usage is nonnegative.
    Removing a cycle can only reduce generalized cost and mode usage, so
    simple paths suffice for the complete network route-value function.
    """

    paths = tuple(nx.all_simple_paths(graph, source=origin, target=destination))
    path_costs = np.zeros(len(paths), dtype=float)
    mode_usage = np.zeros((len(MODE_NAMES), len(paths)), dtype=float)
    for path_index, path in enumerate(paths):
        for tail, head in nx.utils.pairwise(path):
            edge = graph.edges[tail, head]
            path_costs[path_index] += edge["free_flow_time"]
            # High-capacity links form the synthetic trunk network; remaining
            # links consume feeder credits. This classification is not TNTP data.
            mode = 0 if edge["capacity"] >= trunk_capacity_threshold else 1
            mode_usage[mode, path_index] += 1.0
    return path_costs, mode_usage


def draw_sioux_falls_menu(
    random: np.random.Generator,
    *,
    config: SiouxFallsExperimentConfig = DEFAULT_SIOUX_FALLS_CONFIG,
) -> BundleMenu:
    """Draw one candidate at the instance's monthly scales, without screening."""

    prices = np.concatenate(
        ([0.0], np.sort(random.uniform(*config.sampled_price_bounds_usd_per_month, size=2)))
    )
    basic = random.uniform(*config.sampled_basic_allowance_bounds_link_uses_per_month, size=2)
    premium = basic + random.uniform(
        *config.sampled_premium_increment_bounds_link_uses_per_month, size=2
    )
    return BundleMenu(prices=prices, allowances=np.vstack((np.zeros(2), basic, premium)))
