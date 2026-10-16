"""Construct the small synthetic PT/SM instance and candidate menus."""

import networkx as nx
import numpy as np

from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.examples.stylized_config import (
    DEFAULT_STYLIZED_CONFIG,
    MODE_USAGE,
    StylizedExperimentConfig,
)
from maas_bundle.oracle.demand_distribution import UniformDemandDistribution
from maas_bundle.oracle.route_value import build_route_value_oracle

# Edge keys distinguish parallel PT and SM links from O to H.
STYLIZED_PATHS = (
    (("O", "H", "PT"), ("H", "D", "PT")),
    (("O", "D", "SM"),),
    (("O", "H", "SM"), ("H", "D", "PT")),
)


def build_stylized_graph() -> nx.MultiDiGraph:
    """Return three nodes, four service links and exactly three OD paths.

    The direct SM service consumes two resource units per trip. Resource
    usage is deliberately distinct from the number of traversed links.
    """

    graph = nx.MultiDiGraph(name="Stylized 3-path PT/SM network")
    graph.add_nodes_from(("O", "H", "D"))
    for tail, head, mode, usage in (
        ("O", "H", "PT", 1.0),
        ("H", "D", "PT", 1.0),
        ("O", "D", "SM", 2.0),
        ("O", "H", "SM", 1.0),
    ):
        graph.add_edge(tail, head, key=mode, mode=mode, resource_units=usage)
    return graph


def build_stylized_problem(
    config: StylizedExperimentConfig = DEFAULT_STYLIZED_CONFIG,
    *,
    price_upper_bound: float = np.inf,
    allowance_upper_bound: float = np.inf,
) -> BundleDesignProblem:
    """Build three baseline classes or four classes with the optional M_H.

    All options share each class's path set, tariffs and route oracle. M_H
    reuses M's oracle by identity; only its demand distribution differs.
    The menu domain uses nonnegativity, ordered prices and fixed zero PAYG.
    Candidate sampling ranges do not impose finite optimization upper bounds.
    Explicit upper bounds may be supplied by experiments such as A2.
    """

    oracles = tuple(
        build_route_value_oracle(costs, MODE_USAGE, config.payg_tariff)
        for costs in config.baseline_path_costs
    )
    if config.include_high_uncertainty:
        oracles += (oracles[1],)
    return BundleDesignProblem(
        traveler_weights=np.asarray(config.traveler_weights, dtype=float),
        unit_allowance_cost=np.asarray(config.unit_allowance_cost, dtype=float),
        route_oracles=tuple((oracle,) * 3 for oracle in oracles),
        demand_distributions=tuple(
            UniformDemandDistribution(lower, upper) for lower, upper in config.frequency_supports
        ),
        price_upper_bounds=price_upper_bound,
        allowance_upper_bounds=allowance_upper_bound,
    )


def initial_stylized_menu(
    config: StylizedExperimentConfig = DEFAULT_STYLIZED_CONFIG,
) -> BundleMenu:
    """Return the configured menu, checked against the instance design domain."""

    menu = BundleMenu(
        prices=np.asarray(config.initial_prices, dtype=float),
        allowances=np.asarray(config.initial_allowances, dtype=float),
    )
    config.validate_menu(menu)
    return menu


def draw_stylized_menu(
    random: np.random.Generator,
    config: StylizedExperimentConfig = DEFAULT_STYLIZED_CONFIG,
) -> BundleMenu:
    """Sample within candidate ranges, ordering prices and allowing crossed allowances.

    Candidate draws do not promise positive profit. Future benchmark drivers
    can use the existing sample_profitable_menus helper to screen them.
    Sampling ranges do not constrain the subsequent regional optimization.
    """

    menu = BundleMenu(
        prices=np.concatenate(
            ([0.0], np.sort(random.uniform(*config.sampled_price_bounds, size=2)))
        ),
        allowances=np.vstack(
            (np.zeros(2), random.uniform(*config.sampled_allowance_bounds, size=(2, 2)))
        ),
    )
    config.validate_menu(menu)
    return menu
