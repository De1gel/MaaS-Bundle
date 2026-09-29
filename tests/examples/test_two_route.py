"""End-to-end checks for the CC0 two-route example."""

from maas_bundle.examples.two_route import (
    build_two_route_graph,
    build_two_route_problem,
    initial_two_route_menu,
)


def test_two_route_source_topology_and_problem_dimensions() -> None:
    graph = build_two_route_graph()
    problem = build_two_route_problem()
    menu = initial_two_route_menu()

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4
    assert problem.traveler_count == 4
    assert problem.bundle_count == 3
    assert problem.mode_count == 2
    assert menu.bundle_count == problem.bundle_count
    assert menu.mode_count == problem.mode_count
    assert [type(distribution).__name__ for distribution in problem.demand_distributions] == [
        "UniformDemandDistribution",
        "TriangularDemandDistribution",
        "UniformDemandDistribution",
        "TriangularDemandDistribution",
    ]
