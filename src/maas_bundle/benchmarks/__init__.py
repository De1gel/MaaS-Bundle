"""Experiment drivers, shared run settings and comparison utilities."""

from maas_bundle.benchmarks.assignment_enumeration import (
    AssignmentEnumerationResult,
    EnumeratedCandidate,
    enumerate_assignment_regions,
)
from maas_bundle.benchmarks.multistart import (
    MultiStartResult,
    MultiStartRun,
    run_multistart,
)
from maas_bundle.benchmarks.settings import (
    DEFAULT_SIOUX_FALLS_SETTINGS,
    DEFAULT_TWO_ROUTE_SETTINGS,
    BenchmarkSettings,
)
from maas_bundle.benchmarks.sioux_falls import (
    generate_sioux_falls_initial_menus,
    run_sioux_falls_example,
    run_sioux_falls_multistart,
)
from maas_bundle.benchmarks.stylized_assignment_starts import (
    run_stylized_assignment_starts,
    save_stylized_assignment_starts,
)
from maas_bundle.benchmarks.stylized_oracle import (
    DEFAULT_STYLIZED_ORACLE_SETTINGS,
    StylizedOracleResult,
    StylizedOracleSettings,
    run_stylized_oracle,
    save_stylized_oracle_results,
)
from maas_bundle.benchmarks.stylized_regional import (
    DEFAULT_STYLIZED_REGIONAL_SETTINGS,
    StylizedRegionalSettings,
    run_stylized_regional,
    save_stylized_regional_results,
)
from maas_bundle.benchmarks.stylized_regional_multistart import (
    run_stylized_regional_multistart,
    save_stylized_regional_multistart,
)
from maas_bundle.benchmarks.two_route import (
    compare_two_route,
    generate_two_route_initial_menus,
    run_two_route_example,
    run_two_route_multistart,
)

__all__ = [
    "DEFAULT_STYLIZED_REGIONAL_SETTINGS",
    "StylizedRegionalSettings",
    "run_stylized_regional",
    "save_stylized_regional_results",
    "run_stylized_assignment_starts",
    "save_stylized_assignment_starts",
    "run_stylized_regional_multistart",
    "save_stylized_regional_multistart",
    "DEFAULT_STYLIZED_ORACLE_SETTINGS",
    "StylizedOracleResult",
    "StylizedOracleSettings",
    "run_stylized_oracle",
    "save_stylized_oracle_results",
    "BenchmarkSettings",
    "DEFAULT_SIOUX_FALLS_SETTINGS",
    "DEFAULT_TWO_ROUTE_SETTINGS",
    "compare_two_route",
    "generate_sioux_falls_initial_menus",
    "run_sioux_falls_example",
    "run_sioux_falls_multistart",
    "generate_two_route_initial_menus",
    "run_two_route_example",
    "run_two_route_multistart",
    "AssignmentEnumerationResult",
    "EnumeratedCandidate",
    "MultiStartResult",
    "MultiStartRun",
    "enumerate_assignment_regions",
    "run_multistart",
]
