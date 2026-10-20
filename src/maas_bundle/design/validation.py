"""Algorithm-level validation of assignment-consistent primal candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from maas_bundle.design.regional import RegionalFirstOrderCertificate

if TYPE_CHECKING:
    from maas_bundle.design.primal import AssignmentConsistencyResult


class CandidateValidationStatus(StrEnum):
    """Outcome of the checks owned by the primal algorithm."""

    CONSISTENT_CANDIDATE = "consistent_candidate"
    NOT_ASSIGNMENT_CONSISTENT = "not_assignment_consistent"
    SEPARATION_FAILED = "separation_failed"
    REGIONAL_SOLVE_FAILED = "regional_solve_failed"
    DOUBLE_TIE = "double_tie"
    PROFIT_DECREASE = "profit_decrease"
    BOUNDARY_SEARCH_INCOMPLETE = "boundary_search_incomplete"


@dataclass(frozen=True)
class CandidateDiagnostics:
    """Diagnostics for the candidate returned by the primal algorithm.

    A valid candidate passes regional checks and the implemented numerical
    boundary search. Return cycles and one-tie switches are not independent
    second-order or multiple-boundary-intersection optimality certificates.
    """

    status: CandidateValidationStatus
    candidate: AssignmentConsistencyResult
    assignment_consistent: bool
    full_separation_satisfied: bool
    regional_solve_succeeded: bool | None
    regional_kkt_satisfied: bool | None
    active_user_tie_count: int
    double_tie_count: int
    first_order_certificate: RegionalFirstOrderCertificate | None
    message: str

    @property
    def valid(self) -> bool:
        """Return whether every algorithm-level validation check passed."""

        return self.status is CandidateValidationStatus.CONSISTENT_CANDIDATE

    def metrics(self) -> dict[str, str | int | bool | None]:
        """Return a terminal-report-friendly diagnostic record."""

        return {
            "status": self.status.value,
            "valid": self.valid,
            "assignment_consistent": self.assignment_consistent,
            "full_separation": self.full_separation_satisfied,
            "regional_success": self.regional_solve_succeeded,
            "regional_kkt": self.regional_kkt_satisfied,
            "user_ties": self.active_user_tie_count,
            "double_ties": self.double_tie_count,
        }


def validate_candidate(
    candidate: AssignmentConsistencyResult,
    *,
    margin_tolerance: float | None = None,
) -> CandidateDiagnostics:
    """Validate the final outer, separation, regional, and tie conditions."""

    if margin_tolerance is not None and (not np.isfinite(margin_tolerance) or margin_tolerance < 0):
        raise ValueError("margin_tolerance must be finite and nonnegative")

    # Validation must not reinterpret a completed search under a new tie set.
    searched_margin_tolerance = candidate.initial_assignment.margin_tolerance
    if margin_tolerance is not None and margin_tolerance != searched_margin_tolerance:
        raise ValueError("margin_tolerance must match the boundary search tolerance")

    final_region = candidate.terminal_iteration.fixed_assignment_result
    final_solve = final_region.final_solve_result
    first_order = final_solve.first_order_certificate

    if not candidate.consistent:
        if candidate.status.value == "profit_decrease":
            status = CandidateValidationStatus.PROFIT_DECREASE
            message = "regional trial decreased profit; retained the previous feasible menu"
        elif candidate.status.value == "region_iteration_limit" and any(
            step.boundary_origin is not None for step in candidate.iterations
        ):
            status = CandidateValidationStatus.BOUNDARY_SEARCH_INCOMPLETE
            message = "boundary exploration reached the region limit; retained the best menu"
        elif (
            candidate.status.value == "fixed_assignment_failed"
            and final_region.status.value == "solver_failed"
        ):
            status = CandidateValidationStatus.REGIONAL_SOLVE_FAILED
            message = "fixed-assignment generation stopped after a regional NLP failure"
        elif candidate.status.value == "fixed_assignment_failed":
            status = CandidateValidationStatus.SEPARATION_FAILED
            message = "fixed-assignment generation stopped before full separation"
        else:
            status = CandidateValidationStatus.NOT_ASSIGNMENT_CONSISTENT
            message = "outer search did not return an assignment-consistent candidate"
        return _diagnostics(
            status,
            candidate,
            regional_solve_succeeded=final_solve.success,
            regional_kkt_satisfied=(
                first_order.satisfies_kkt if first_order.stationarity_checked else None
            ),
            first_order_certificate=first_order,
            message=message,
        )

    # CONSISTENT is produced only after solve_fixed_assignment returns
    # CONVERGED. That transition guarantees acceptance under the selected
    # solver policy and feasible full separation; do not recompute them here.
    assignment = candidate.terminal_iteration.true_assignment
    assert assignment is not None
    tied = assignment.has_numerical_cost_ties
    double_ties = assignment.has_double_ties
    tie_count = int(np.count_nonzero(tied))
    double_tie_count = int(np.count_nonzero(double_ties))
    status = (
        CandidateValidationStatus.DOUBLE_TIE
        if double_tie_count and not candidate.boundary_search_complete
        else CandidateValidationStatus.CONSISTENT_CANDIDATE
    )
    if double_tie_count and not candidate.boundary_search_complete:
        message = "a numerical user tie lacks a strict platform-margin gap"
    elif candidate.boundary_search_complete:
        message = (
            "candidate passes regional checks and completed boundary search; "
            "not a second-order certificate"
        )
    else:
        message = "candidate passes regional checks; boundary search was not recorded"
    if not first_order.stationarity_checked:
        message += "; exploratory candidate: independent KKT was not checked"
    return CandidateDiagnostics(
        status=status,
        candidate=candidate,
        assignment_consistent=True,
        full_separation_satisfied=True,
        regional_solve_succeeded=True,
        regional_kkt_satisfied=(
            first_order.satisfies_kkt if first_order.stationarity_checked else None
        ),
        active_user_tie_count=tie_count,
        double_tie_count=double_tie_count,
        first_order_certificate=first_order,
        message=message,
    )


def _diagnostics(
    status: CandidateValidationStatus,
    candidate: AssignmentConsistencyResult,
    *,
    assignment_consistent: bool = False,
    full_separation_satisfied: bool = False,
    regional_solve_succeeded: bool | None = None,
    regional_kkt_satisfied: bool | None = None,
    first_order_certificate: RegionalFirstOrderCertificate | None = None,
    message: str,
) -> CandidateDiagnostics:
    return CandidateDiagnostics(
        status=status,
        candidate=candidate,
        assignment_consistent=assignment_consistent,
        full_separation_satisfied=full_separation_satisfied,
        regional_solve_succeeded=regional_solve_succeeded,
        regional_kkt_satisfied=regional_kkt_satisfied,
        active_user_tie_count=0,
        double_tie_count=0,
        first_order_certificate=first_order_certificate,
        message=message,
    )
