"""Check complete modal-network geometry against explicit path enumeration."""

import networkx as nx
import numpy as np

from maas_bundle.oracle.network import build_network_route_value_oracle
from maas_bundle.oracle.route_value import build_route_value_oracle


def test_vertex_separation_matches_all_multimodal_paths():
    graph = nx.MultiDiGraph()
    for tail, head, time in [(0, 1, 1), (1, 3, 3), (0, 2, 2), (2, 3, 1), (1, 2, 0.5)]:
        for mode in range(5):
            graph.add_edge(
                tail,
                head,
                mode_index=mode,
                generalized_cost=time * (1 + mode * 0.3) + 0.1 / (mode + 1),
            )
    costs, usages = [], []
    for path in nx.all_simple_edge_paths(graph, 0, 3):
        cost, usage = 0.0, np.zeros(5)
        for edge in path:
            data = graph.edges[edge]
            cost += data["generalized_cost"]
            usage[data["mode_index"]] += 1
        costs.append(cost)
        usages.append(usage)
    tariffs = [1.5, 1.2, 0.9, 0.6, 0.3]
    explicit = build_route_value_oracle(costs, np.asarray(usages).T, tariffs)
    separated = build_network_route_value_oracle(graph, 0, 3, tariffs)
    assert len(costs) == 175
    assert separated.vertex_count == explicit.vertex_count
    for vertex in explicit.vertices:
        assert np.min(np.max(np.abs(separated.vertices - vertex), axis=1)) < 1e-8
