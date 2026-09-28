"""Small multimodal example derived from TAPLab's CC0 two-route network.

Source network
--------------
TAPLab, ``tapbench/diagnostic/two_route``:
https://github.com/asu-trans-ai-lab/TAPLab/tree/main/tapbench/diagnostic/two_route

The source has one OD pair, 2,000 AM trips, and two symmetric paths with
free-flow time 12.  Its data manifest declares the instance CC0.  This MaaS
example retains the topology and free-flow times, interprets the upper path as
bus and the lower path as rail, and adds synthetic traveler and product data.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from maas_bundle.benchmarks.initialization import sample_profitable_menus
from maas_bundle.benchmarks.multistart import MultiStartResult, run_multistart
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import solve_primal
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import ScipyRegionalSolver
from maas_bundle.design.validation import CandidateDiagnostics
from maas_bundle.oracle.demand_distribution import (
    TriangularDemandDistribution,
    UniformDemandDistribution,
)
from maas_bundle.oracle.route_value import build_route_value_oracle

SOURCE_AM_DEMAND = 2_000.0
SOURCE_URL = "https://github.com/asu-trans-ai-lab/TAPLab/tree/main/tapbench/diagnostic/two_route"
MODE_NAMES = ("bus", "rail")


def build_two_route_graph() -> nx.DiGraph:
    """Return the four-node, two-path CC0 source topology."""

    graph = nx.DiGraph(name="TAPLab CC0 two-route diagnostic")
    graph.add_node(1, node_type="centroid", position=(0.0, 0.0))
    graph.add_node(2, node_type="intersection", position=(1.0, 1.0))
    graph.add_node(3, node_type="intersection", position=(1.0, -1.0))
    graph.add_node(4, node_type="centroid", position=(2.0, 0.0))
    graph.add_edge(1, 2, free_flow_time=2.0, capacity=1_000.0, mode="bus")
    graph.add_edge(2, 4, free_flow_time=10.0, capacity=1_000.0, mode="bus")
    graph.add_edge(1, 3, free_flow_time=2.0, capacity=1_000.0, mode="rail")
    graph.add_edge(3, 4, free_flow_time=10.0, capacity=1_000.0, mode="rail")
    return graph


def build_two_route_problem() -> BundleDesignProblem:
    """Build a four-class, two-mode, two-subscription MaaS instance."""

    graph = build_two_route_graph()
    paths = tuple(sorted(nx.all_simple_paths(graph, source=1, target=4)))
    free_flow_cost = np.array(
        [
            sum(graph.edges[tail, head]["free_flow_time"] for tail, head in nx.utils.pairwise(path))
            for path in paths
        ],
        dtype=float,
    )
    mode_usage = np.zeros((len(MODE_NAMES), len(paths)), dtype=float)
    for path_index, path in enumerate(paths):
        used_modes = {graph.edges[tail, head]["mode"] for tail, head in nx.utils.pairwise(path)}
        for mode in used_modes:
            mode_usage[MODE_NAMES.index(mode), path_index] = 1.0

    # Generalized path cost = value-of-time factor * free-flow time plus a
    # synthetic mode-preference penalty.  The first two classes prefer bus;
    # the last two prefer rail.
    value_of_time = np.array([0.7, 0.9, 1.1, 1.3])
    preference_penalty = np.array(
        [
            [0.0, 1.4],
            [0.0, 0.9],
            [1.1, 0.0],
            [2.6, 0.0],
        ]
    )
    generalized_path_costs = value_of_time[:, None] * free_flow_cost + preference_penalty
    payg_tariff = np.array([1.2, 1.0])
    traveler_oracles = tuple(
        build_route_value_oracle(costs, mode_usage, payg_tariff) for costs in generalized_path_costs
    )

    return BundleDesignProblem(
        # Normalized shares of the source network's 2,000 AM trips.
        traveler_weights=np.array([0.25, 0.35, 0.25, 0.15]),
        unit_allowance_cost=np.array([0.15, 0.12]),
        route_oracles=tuple((oracle, oracle, oracle) for oracle in traveler_oracles),
        demand_distributions=(
            UniformDemandDistribution(2.0, 4.0),
            TriangularDemandDistribution(3.0, 4.5, 6.0),
            UniformDemandDistribution(5.0, 8.0),
            TriangularDemandDistribution(7.0, 8.5, 10.0),
        ),
    )


def initial_two_route_menu() -> BundleMenu:
    """Return the feasible PAYG/basic/premium warm start for the example."""

    return BundleMenu(
        # This deliberately starts in assignment region [1, 2, 1, 2]. The
        # first optimized region reassigns class 3 to premium, forcing a
        # second outer iteration before consistency is reached.
        prices=np.array([0.0, 2.0, 2.5]),
        allowances=np.array(
            [
                [0.0, 0.0],
                [3.0, 7.0],
                [9.0, 9.0],
            ]
        ),
    )


def run_two_route_example() -> CandidateDiagnostics:
    """Run the complete primal pipeline on the small open-data instance."""

    solver = ScipyRegionalSolver(
        "SLSQP",
        options={"maxiter": 2_000, "ftol": 1e-10},
    )
    return solve_primal(
        build_two_route_problem(),
        initial_two_route_menu(),
        solver,
        max_region_iterations=20,
    )


def generate_two_route_initial_menus(
    start_count: int = 30,
    *,
    seed: int | None = 20260901,
    problem: BundleDesignProblem | None = None,
) -> tuple[BundleMenu, ...]:
    """Generate monotone, positive-profit starts outside the all-PAYG plateau."""

    if not isinstance(start_count, (int, np.integer)) or start_count <= 0:
        raise ValueError("start_count must be a positive integer")
    random = np.random.default_rng(seed)
    if problem is None:
        problem = build_two_route_problem()

    def draw_menu() -> BundleMenu:
        prices = np.concatenate(([0.0], np.sort(random.uniform(0.2, 8.0, size=2))))
        basic_allowance = random.uniform(0.0, 6.0, size=2)
        premium_allowance = basic_allowance + random.uniform(0.0, 6.0, size=2)
        allowances = np.vstack((np.zeros(2), basic_allowance, premium_allowance))
        return BundleMenu(prices=prices, allowances=allowances)

    return sample_profitable_menus(problem, draw_menu, start_count)


def run_two_route_multistart(
    start_count: int = 30,
    *,
    seed: int | None = 20260901,
) -> MultiStartResult:
    """Run a reproducible multi-start experiment on the two-route instance."""

    solver = ScipyRegionalSolver(
        "SLSQP",
        options={"maxiter": 2_000, "ftol": 1e-10},
    )
    problem = build_two_route_problem()
    return run_multistart(
        problem,
        generate_two_route_initial_menus(start_count, seed=seed, problem=problem),
        solver,
        max_region_iterations=20,
    )
