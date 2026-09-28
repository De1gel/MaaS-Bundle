"""Exhaustive assignment-region baseline for very small instances."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from maas_bundle.design.assignment import AssignmentResult
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import FixedAssignmentResult, solve_fixed_assignment
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import RegionalNlpSolver


@dataclass(frozen=True)
class EnumeratedCandidate:
    """One fixed assignment whose optimized menu reproduces that assignment."""

    assigned_bundles: NDArray[np.int64]
    fixed_assignment_result: FixedAssignmentResult
    true_assignment: AssignmentResult

    @property
    def true_profit(self) -> float:
        """Return the original optimistic profit at the candidate menu."""

        return self.true_assignment.total_profit


@dataclass(frozen=True)
class AssignmentEnumerationResult:
    """Summary of exhaustive assignment-region traversal."""

    assignment_count: int
    converged_region_count: int
    candidates: tuple[EnumeratedCandidate, ...]
    solve_time_seconds: float

    @property
    def consistent_candidate_count(self) -> int:
        """Return how many optimized regions reproduced their assignment."""

        return len(self.candidates)

    @property
    def best_candidate(self) -> EnumeratedCandidate | None:
        """Return the highest-true-profit consistent candidate, if one exists."""

        return None if not self.candidates else self.candidates[0]

    def metrics(self) -> dict[str, int | float | None]:
        """Return compact enumeration diagnostics."""

        best = self.best_candidate
        return {
            "assignments": self.assignment_count,
            "converged_regions": self.converged_region_count,
            "consistent_candidates": self.consistent_candidate_count,
            "best_true_profit": None if best is None else best.true_profit,
            "time_seconds": self.solve_time_seconds,
        }


def enumerate_assignment_regions(
    problem: BundleDesignProblem,
    initial_menu: BundleMenu,
    solver: RegionalNlpSolver,
    *,
    maximum_assignments: int | None = 100_000,
    choice_tolerance: float = 1e-9,
    profit_tolerance: float = 1e-9,
    initial_per_pair: int = 1,
    tie_tolerance: float = 1e-12,
    violation_tolerance: float = 1e-9,
    max_generation_iterations: int | None = None,
) -> AssignmentEnumerationResult:
    """Solve every fixed-assignment region and retain consistent candidates.

    This is an exponential small-instance baseline, not a global optimizer for
    the continuous problem: every regional NLP is still solved by the supplied
    local solver.  It reuses the production fixed-assignment generation and
    exact optimistic-assignment evaluator without duplicating their formulas.
    """

    assignment_count = problem.bundle_count**problem.traveler_count
    if maximum_assignments is not None:
        if not isinstance(maximum_assignments, (int, np.integer)) or maximum_assignments <= 0:
            raise ValueError("maximum_assignments must be a positive integer or None")
        if assignment_count > maximum_assignments:
            raise ValueError(
                f"enumeration requires {assignment_count} assignments, "
                f"exceeding maximum_assignments={maximum_assignments}"
            )

    started = perf_counter()
    converged_count = 0
    candidates: list[EnumeratedCandidate] = []
    assignments = product(range(problem.bundle_count), repeat=problem.traveler_count)
    for assignment_tuple in assignments:
        assigned_bundles = np.asarray(assignment_tuple, dtype=np.int64)
        fixed_result = solve_fixed_assignment(
            problem,
            initial_menu,
            assigned_bundles,
            solver,
            initial_per_pair=initial_per_pair,
            tie_tolerance=tie_tolerance,
            violation_tolerance=violation_tolerance,
            max_iterations=max_generation_iterations,
        )
        if not fixed_result.converged:
            continue
        converged_count += 1

        true_assignment = problem.evaluate_assignment(
            fixed_result.menu,
            choice_tolerance=choice_tolerance,
            profit_tolerance=profit_tolerance,
        )
        if not np.array_equal(true_assignment.bundle_indices, assigned_bundles):
            continue
        immutable_assignment = assigned_bundles.copy()
        immutable_assignment.setflags(write=False)
        candidates.append(
            EnumeratedCandidate(
                assigned_bundles=immutable_assignment,
                fixed_assignment_result=fixed_result,
                true_assignment=true_assignment,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.true_profit,
            tuple(int(bundle) for bundle in candidate.assigned_bundles),
        )
    )
    return AssignmentEnumerationResult(
        assignment_count=assignment_count,
        converged_region_count=converged_count,
        candidates=tuple(candidates),
        solve_time_seconds=perf_counter() - started,
    )
