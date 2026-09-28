"""Reproducible example instances for MaaS bundle design."""

from maas_bundle.examples.sioux_falls import (
    build_sioux_falls_graph,
    build_sioux_falls_problem,
    generate_sioux_falls_initial_menus,
    initial_sioux_falls_menu,
    run_sioux_falls_example,
    run_sioux_falls_multistart,
    sioux_falls_od_matrix,
)
from maas_bundle.examples.sioux_falls_config import (
    DEFAULT_SIOUX_FALLS_CONFIG,
    SiouxFallsExperimentConfig,
)
from maas_bundle.examples.two_route import (
    SOURCE_AM_DEMAND,
    build_two_route_graph,
    build_two_route_problem,
    generate_two_route_initial_menus,
    initial_two_route_menu,
    run_two_route_example,
    run_two_route_multistart,
)

__all__ = [
    "DEFAULT_SIOUX_FALLS_CONFIG",
    "SiouxFallsExperimentConfig",
    "SOURCE_AM_DEMAND",
    "build_sioux_falls_graph",
    "build_sioux_falls_problem",
    "build_two_route_graph",
    "build_two_route_problem",
    "generate_sioux_falls_initial_menus",
    "generate_two_route_initial_menus",
    "initial_sioux_falls_menu",
    "initial_two_route_menu",
    "run_sioux_falls_example",
    "run_sioux_falls_multistart",
    "run_two_route_example",
    "run_two_route_multistart",
    "sioux_falls_od_matrix",
]
