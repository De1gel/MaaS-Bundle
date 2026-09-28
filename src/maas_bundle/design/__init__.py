"""Public API for MaaS bundle-design components."""

from maas_bundle.design.assignment import (
    AssignmentResult,
    aggregate_bundle_demand,
    select_optimistic_bundles,
)
from maas_bundle.design.boundary import (
    ChoiceConstraintIndex,
    ChoiceConstraintViolation,
    CriticalConstraintSet,
    SeparationResult,
    assigned_pairwise_values,
    choice_constraint_slacks,
    initialize_critical_constraints,
    separate_choice_constraints,
)
from maas_bundle.design.cost import CallableProvisionCost, LinearProvisionCost, ProvisionCost
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import (
    AssignmentConsistencyResult,
    AssignmentConsistencyStatus,
    AssignmentRegionIteration,
    CriticalGenerationIteration,
    FixedAssignmentResult,
    FixedAssignmentStatus,
    find_assignment_consistent_menu,
    solve_fixed_assignment,
    solve_primal,
)
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import (
    MenuVariableLayout,
    RegionalFirstOrderCertificate,
    RegionalNlpSolver,
    RegionalSolveResult,
    RegionalSolveStatus,
    RestrictedRegionalProblem,
    ScipyRegionalSolver,
    backtrack_to_feasible_menu,
)
from maas_bundle.design.validation import (
    CandidateDiagnostics,
    CandidateValidationStatus,
    validate_candidate,
)
from maas_bundle.design.valuation import ValuationTable

__all__ = [
    "ProvisionCost",
    "LinearProvisionCost",
    "CallableProvisionCost",
    "AssignmentResult",
    "AssignmentConsistencyResult",
    "AssignmentConsistencyStatus",
    "AssignmentRegionIteration",
    "BundleDesignProblem",
    "BundleMenu",
    "CandidateDiagnostics",
    "CandidateValidationStatus",
    "ChoiceConstraintIndex",
    "ChoiceConstraintViolation",
    "CriticalConstraintSet",
    "CriticalGenerationIteration",
    "FixedAssignmentResult",
    "FixedAssignmentStatus",
    "MenuVariableLayout",
    "RegionalFirstOrderCertificate",
    "RegionalNlpSolver",
    "RegionalSolveResult",
    "RegionalSolveStatus",
    "RestrictedRegionalProblem",
    "ScipyRegionalSolver",
    "SeparationResult",
    "ValuationTable",
    "aggregate_bundle_demand",
    "assigned_pairwise_values",
    "backtrack_to_feasible_menu",
    "choice_constraint_slacks",
    "find_assignment_consistent_menu",
    "initialize_critical_constraints",
    "separate_choice_constraints",
    "select_optimistic_bundles",
    "solve_fixed_assignment",
    "solve_primal",
    "validate_candidate",
]
