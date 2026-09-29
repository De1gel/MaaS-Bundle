"""Multi-start diagnostics for assignment-consistent primal search."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import solve_primal
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import RegionalNlpSolver
from maas_bundle.design.tolerances import (
    DEFAULT_COST_TOLERANCE,
    DEFAULT_MARGIN_TOLERANCE,
    DEFAULT_PROFIT_TOLERANCE,
)
from maas_bundle.design.validation import CandidateDiagnostics


@dataclass(frozen=True)
class MultiStartRun:
    """Outcome of one primal solve from one initial menu."""

    number: int
    initial_menu: BundleMenu
    diagnostics: CandidateDiagnostics

    @property
    def consistent(self) -> bool:
        """Return whether this start reached an assignment-consistent region."""

        return self.diagnostics.candidate.consistent

    @property
    def initial_assignment(self) -> tuple[int, ...]:
        """Return the optimistic assignment at the initial menu."""

        indices = self.diagnostics.candidate.initial_assignment.bundle_indices
        return tuple(int(bundle) for bundle in indices)

    @property
    def final_assignment(self) -> tuple[int, ...]:
        """Return the final optimistic assignment reached by this start."""

        return tuple(int(bundle) for bundle in self.diagnostics.candidate.assignment)

    @property
    def true_profit(self) -> float | None:
        """Return the final true optimistic profit."""

        return self.diagnostics.candidate.true_profit


@dataclass(frozen=True)
class MultiStartResult:
    """All runs and assignment-region aggregation for a multi-start experiment."""

    runs: tuple[MultiStartRun, ...]
    solve_time_seconds: float

    @property
    def consistent_runs(self) -> tuple[MultiStartRun, ...]:
        """Return starts that reached an assignment-consistent candidate."""

        return tuple(run for run in self.runs if run.consistent)

    @property
    def distinct_region_count(self) -> int:
        """Return the number of distinct consistent final assignments."""

        return len({run.final_assignment for run in self.consistent_runs})

    @property
    def best_run(self) -> MultiStartRun | None:
        """Return the highest-profit consistent run, if one exists."""

        eligible = [run for run in self.consistent_runs if run.true_profit is not None]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda run: (
                float(run.true_profit),
                tuple(-bundle for bundle in run.final_assignment),
                -run.number,
            ),
        )

    def region_summaries(self) -> tuple[dict[str, object], ...]:
        """Group consistent starts by their final assignment region."""

        grouped: dict[tuple[int, ...], list[MultiStartRun]] = {}
        for run in self.consistent_runs:
            grouped.setdefault(run.final_assignment, []).append(run)

        summaries: list[dict[str, object]] = []
        for assignment, runs in sorted(grouped.items()):
            profits = [float(run.true_profit) for run in runs if run.true_profit is not None]
            region_iterations = [run.diagnostics.candidate.region_iteration_count for run in runs]
            summaries.append(
                {
                    "assignment": list(assignment),
                    "starts": len(runs),
                    "valid_starts": sum(run.diagnostics.valid for run in runs),
                    "best_true_profit": max(profits) if profits else None,
                    "worst_true_profit": min(profits) if profits else None,
                    "mean_region_iterations": float(np.mean(region_iterations)),
                    "start_numbers": [run.number for run in runs],
                }
            )
        return tuple(summaries)

    def metrics(self) -> dict[str, int | float | None]:
        """Return compact experiment-level diagnostics."""

        best = self.best_run
        return {
            "starts": len(self.runs),
            "consistent_starts": len(self.consistent_runs),
            "valid_starts": sum(run.diagnostics.valid for run in self.runs),
            "distinct_regions": self.distinct_region_count,
            "best_true_profit": None if best is None else best.true_profit,
            "time_seconds": self.solve_time_seconds,
        }


def run_multistart(
    problem: BundleDesignProblem,
    initial_menus: Iterable[BundleMenu],
    solver: RegionalNlpSolver,
    *,
    choice_tolerance: float = DEFAULT_COST_TOLERANCE,
    profit_tolerance: float = DEFAULT_PROFIT_TOLERANCE,
    max_region_iterations: int = 100,
    stop_on_assignment_revisit: bool = False,
    initial_per_pair: int = 1,
    critical_tie_tolerance: float = 1e-12,
    violation_tolerance: float = DEFAULT_COST_TOLERANCE,
    max_generation_iterations: int | None = None,
    margin_tolerance: float = DEFAULT_MARGIN_TOLERANCE,
    boundary_tolerance: float = 1e-7,
    boundary_relative_tolerance: float = 1e-6,
    allowance_boundary_tolerance: float = 1e-7,
) -> MultiStartResult:
    """Run the complete primal pipeline from each supplied initial menu."""

    menus = tuple(initial_menus)
    if not menus:
        raise ValueError("initial_menus must contain at least one menu")
    for menu in menus:
        if menu.bundle_count != problem.bundle_count or menu.mode_count != problem.mode_count:
            raise ValueError("every initial menu must match the problem dimensions")

    started = perf_counter()
    runs = tuple(
        MultiStartRun(
            number=number,
            initial_menu=menu,
            diagnostics=solve_primal(
                problem,
                menu,
                solver,
                choice_tolerance=choice_tolerance,
                profit_tolerance=profit_tolerance,
                max_region_iterations=max_region_iterations,
                stop_on_assignment_revisit=stop_on_assignment_revisit,
                initial_per_pair=initial_per_pair,
                critical_tie_tolerance=critical_tie_tolerance,
                violation_tolerance=violation_tolerance,
                max_generation_iterations=max_generation_iterations,
                margin_tolerance=margin_tolerance,
                boundary_tolerance=boundary_tolerance,
                boundary_relative_tolerance=boundary_relative_tolerance,
                allowance_boundary_tolerance=allowance_boundary_tolerance,
            ),
        )
        for number, menu in enumerate(menus, start=1)
    )
    return MultiStartResult(runs=runs, solve_time_seconds=perf_counter() - started)
