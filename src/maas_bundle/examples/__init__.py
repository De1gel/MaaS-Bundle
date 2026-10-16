"""Reproducible example instances for MaaS bundle design."""

from maas_bundle.examples.sioux_falls import (
    build_sioux_falls_graph,
    build_sioux_falls_problem,
    draw_sioux_falls_menu,
    initial_sioux_falls_menu,
    sioux_falls_od_matrix,
)
from maas_bundle.examples.sioux_falls_config import (
    DEFAULT_SIOUX_FALLS_CONFIG,
    SiouxFallsExperimentConfig,
)
from maas_bundle.examples.stylized import (
    build_stylized_graph,
    build_stylized_problem,
    draw_stylized_menu,
    initial_stylized_menu,
)
from maas_bundle.examples.stylized_config import (
    DEFAULT_STYLIZED_CONFIG,
    HIGH_UNCERTAINTY_STYLIZED_CONFIG,
    StylizedExperimentConfig,
)
from maas_bundle.examples.two_route import (
    SOURCE_AM_DEMAND,
    build_two_route_graph,
    build_two_route_problem,
    draw_two_route_menu,
    initial_two_route_menu,
)

__all__ = [
    "DEFAULT_STYLIZED_CONFIG",
    "HIGH_UNCERTAINTY_STYLIZED_CONFIG",
    "StylizedExperimentConfig",
    "DEFAULT_SIOUX_FALLS_CONFIG",
    "SiouxFallsExperimentConfig",
    "SOURCE_AM_DEMAND",
    "build_sioux_falls_graph",
    "build_sioux_falls_problem",
    "build_two_route_graph",
    "build_two_route_problem",
    "draw_sioux_falls_menu",
    "initial_sioux_falls_menu",
    "draw_two_route_menu",
    "initial_two_route_menu",
    "sioux_falls_od_matrix",
    "build_stylized_graph",
    "build_stylized_problem",
    "draw_stylized_menu",
    "initial_stylized_menu",
]
