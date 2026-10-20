"""Complete route-dual geometry on networks with parallel modal arcs."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from maas_bundle.oracle.dual_polytope import build_dual_polytope, enumerate_vertices
from maas_bundle.oracle.route_value import RouteValueOracle


def build_network_route_value_oracle(
    graph: nx.MultiDiGraph,
    origin: int,
    destination: int,
    payg_tariff: ArrayLike,
    *,
    tolerance: float = 1e-9,
) -> RouteValueOracle:
    """Enumerate every dual vertex without enumerating all modal paths.

    Arc attributes are ``generalized_cost`` and ``mode_index``. Start with one
    valid route inequality, enumerate the bounded outer polytope, and separate
    every vertex with a shortest-path oracle. When every vertex passes, the
    outer polytope equals the complete route-dual polytope to ``tolerance``:
    every route inequality is linear and hence holds on its entire convex hull.
    Positive arc costs allow unrestricted transfers at nodes and exclude any
    benefit from cycles. No route-count or iteration cutoff is imposed.
    """

    tariffs = np.asarray(payg_tariff, dtype=float)
    if tariffs.ndim != 1 or not tariffs.size or np.any(tariffs < 0):
        raise ValueError("payg_tariff must be a nonempty nonnegative vector")
    if not np.all(np.isfinite(tariffs)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tariffs and positive tolerance must be finite")
    if origin == destination:
        raise ValueError("origin and destination must differ")
    alternatives = {}
    for tail, head, edge in graph.edges(data=True):
        cost, mode = float(edge["generalized_cost"]), edge["mode_index"]
        if not np.isfinite(cost) or cost <= 0 or not 0 <= mode < tariffs.size:
            raise ValueError("modal arcs require positive costs and valid mode indices")
        alternatives.setdefault((tail, head), []).append((cost, mode))

    def shortest_route(omega):
        weighted = nx.DiGraph()
        weighted.add_nodes_from(graph.nodes)
        for (tail, head), options in alternatives.items():
            cost, mode = min(options, key=lambda item: item[0] + omega[item[1]])
            weighted.add_edge(tail, head, weight=cost + omega[mode], cost=cost, mode=mode)
        path = nx.shortest_path(weighted, origin, destination, weight="weight")
        usage = np.zeros(tariffs.size)
        baseline = 0.0
        for tail, head in nx.utils.pairwise(path):
            edge = weighted.edges[tail, head]
            baseline += edge["cost"]
            usage[edge["mode"]] += 1
        return baseline, usage

    baseline, usage = shortest_route(np.zeros(tariffs.size))
    routes = {tuple(usage): baseline}
    while True:
        usages = np.asarray(list(routes))
        costs = np.asarray(list(routes.values()))
        vertices = enumerate_vertices(build_dual_polytope(costs, usages.T, tariffs))
        cuts = {}
        for vertex in vertices:
            baseline, usage = shortest_route(vertex[1:])
            if vertex[0] > baseline + usage @ vertex[1:] + tolerance:
                key = tuple(usage)
                cuts[key] = min(baseline, cuts.get(key, np.inf))
        if not cuts:
            return RouteValueOracle(vertices)
        changed = False
        for key, cost in cuts.items():
            if cost < routes.get(key, np.inf):
                routes[key] = cost
                changed = True
        if not changed:
            raise RuntimeError("route separation stalled above the requested tolerance")
