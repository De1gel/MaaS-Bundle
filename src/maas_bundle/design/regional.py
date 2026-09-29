"""Restricted fixed-assignment NLP and interchangeable solver adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from time import perf_counter
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import BFGS, Bounds, NonlinearConstraint, OptimizeResult, minimize, nnls

from maas_bundle.design.assignment import aggregate_bundle_demand
from maas_bundle.design.boundary import ChoiceConstraintIndex, CriticalConstraintSet
from maas_bundle.design.cost import LinearProvisionCost
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.tolerances import (
    DEFAULT_COST_TOLERANCE,
)
from maas_bundle.design.valuation import ValuationTable


@dataclass(frozen=True)
class MenuVariableLayout:
    """Pack designed bundles into a solver vector while omitting PAYG."""

    bundle_count: int
    mode_count: int

    def __post_init__(self) -> None:
        """Require PAYG plus at least one designed bundle and one mode."""

        if not isinstance(self.bundle_count, (int, np.integer)) or self.bundle_count < 2:
            raise ValueError("bundle_count must include PAYG and at least one designed bundle")
        if not isinstance(self.mode_count, (int, np.integer)) or self.mode_count <= 0:
            raise ValueError("mode_count must be a positive integer")
        object.__setattr__(self, "bundle_count", int(self.bundle_count))
        object.__setattr__(self, "mode_count", int(self.mode_count))

    @property
    def designed_bundle_count(self) -> int:
        """Return the number of optimized subscription bundles."""

        return self.bundle_count - 1

    @property
    def variable_count(self) -> int:
        """Return ``B * (M + 1)`` for the designed subscriptions."""

        return self.designed_bundle_count * (self.mode_count + 1)

    def pack(self, menu: BundleMenu) -> NDArray[np.float64]:
        """Return ``[C_1..C_B, epsilon_1,1..epsilon_B,M]``."""

        self._validate_menu(menu)
        vector = np.concatenate((menu.prices[1:], menu.allowances[1:].reshape(-1)))
        vector.setflags(write=False)
        return vector

    def unpack(self, vector: ArrayLike) -> BundleMenu:
        """Reconstruct a complete menu and insert the fixed zero PAYG row."""

        values = np.asarray(vector, dtype=float)
        if values.shape != (self.variable_count,):
            raise ValueError(f"solver vector must have shape ({self.variable_count},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("solver vector must contain only finite values")
        if np.any(values < 0):
            raise ValueError("solver vector must be nonnegative")

        designed = self.designed_bundle_count
        prices = np.zeros(self.bundle_count, dtype=float)
        allowances = np.zeros((self.bundle_count, self.mode_count), dtype=float)
        prices[1:] = values[:designed]
        allowances[1:] = values[designed:].reshape(designed, self.mode_count)
        return BundleMenu(prices=prices, allowances=allowances)

    def _validate_menu(self, menu: BundleMenu) -> None:
        if menu.bundle_count != self.bundle_count:
            raise ValueError("menu and layout must use the same bundles")
        if menu.mode_count != self.mode_count:
            raise ValueError("menu and layout must use the same allowance dimensions")
        if menu.prices[0] != 0.0 or np.any(menu.allowances[0] != 0.0):
            raise ValueError("bundle zero must be the fixed PAYG option")


class RestrictedRegionalProblem:
    """Numerical functions for one fixed assignment and active constraint set."""

    def __init__(
        self,
        problem: BundleDesignProblem,
        critical_constraints: CriticalConstraintSet,
    ) -> None:
        if critical_constraints.traveler_count != problem.traveler_count:
            raise ValueError("critical constraints and problem must use the same travelers")
        if critical_constraints.bundle_count != problem.bundle_count:
            raise ValueError("critical constraints and problem must use the same bundles")
        if problem.bundle_count < 2:
            raise ValueError("a regional problem requires at least one designed bundle")

        self.problem = problem
        self.layout = MenuVariableLayout(problem.bundle_count, problem.mode_count)
        self.assigned_bundles = critical_constraints.assigned_bundles
        self.active_indices = critical_constraints.indices()
        self.bundle_demand = aggregate_bundle_demand(
            self.assigned_bundles,
            problem.traveler_weights,
            problem.bundle_count,
        )
        self._cached_vector: NDArray[np.float64] | None = None
        self._cached_menu: BundleMenu | None = None
        self._cached_valuation: ValuationTable | None = None
        self._cached_profit: float | None = None
        self._cached_profit_gradient: NDArray[np.float64] | None = None

    @property
    def variable_count(self) -> int:
        """Return the number of solver variables."""

        return self.layout.variable_count

    @property
    def active_choice_constraint_count(self) -> int:
        """Return the number of generated traveler choice constraints."""

        return len(self.active_indices)

    @property
    def price_monotonicity_constraint_count(self) -> int:
        """Return the number of adjacent subscription-price comparisons."""

        return max(self.layout.designed_bundle_count - 1, 0)

    @property
    def constraint_count(self) -> int:
        """Return choice plus price-monotonicity constraints."""

        return self.active_choice_constraint_count + self.price_monotonicity_constraint_count

    def profit(self, vector: ArrayLike) -> float:
        """Return the fixed-assignment regional objective to maximize."""

        menu = self._menu(vector)
        if self._cached_profit is None:
            self._cached_profit = float(self.bundle_demand @ self.problem.bundle_margins(menu))
        return self._cached_profit

    def profit_gradient(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Return the objective gradient using the provision cost interface."""

        menu = self._menu(vector)
        if self._cached_profit_gradient is None:
            demand = self.bundle_demand[1:]
            cost_gradient = self.problem.provision_cost_gradients(menu)
            allowance_gradient = -(demand[:, None] * cost_gradient).reshape(-1)
            gradient = np.concatenate((demand, allowance_gradient))
            gradient.setflags(write=False)
            self._cached_profit_gradient = gradient
        return self._cached_profit_gradient

    def constraint_values(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Return all regional inequalities in canonical form ``h(x) <= 0``."""

        menu, valuation = self._evaluate(vector)
        values = np.empty(self.constraint_count, dtype=float)
        for row, index in enumerate(self.active_indices):
            values[row] = _choice_constraint_value(menu, valuation, index)

        offset = self.active_choice_constraint_count
        designed_prices = menu.prices[1:]
        if self.price_monotonicity_constraint_count:
            values[offset:] = designed_prices[:-1] - designed_prices[1:]
        values.setflags(write=False)
        return values

    def constraint_jacobian(self, vector: ArrayLike) -> NDArray[np.float64]:
        """Return a subgradient Jacobian for all inequalities ``h(x) <= 0``."""

        _, valuation = self._evaluate(vector)
        jacobian = np.zeros((self.constraint_count, self.variable_count), dtype=float)
        for row, index in enumerate(self.active_indices):
            self._fill_choice_constraint_jacobian(jacobian[row], valuation, index)

        offset = self.active_choice_constraint_count
        for comparison in range(self.price_monotonicity_constraint_count):
            jacobian[offset + comparison, comparison] = 1.0
            jacobian[offset + comparison, comparison + 1] = -1.0
        jacobian.setflags(write=False)
        return jacobian

    def maximum_constraint_violation(self, vector: ArrayLike) -> float:
        """Return the largest nonnegativity or regional inequality violation."""

        values = np.asarray(vector, dtype=float)
        bound_violation = float(np.max(np.maximum(-values, 0.0), initial=0.0))
        if np.any(values < 0):
            projected = np.maximum(values, 0.0)
        else:
            projected = values
        constraints = self.constraint_values(projected)
        inequality_violation = float(np.max(np.maximum(constraints, 0.0), initial=0.0))
        return max(bound_violation, inequality_violation)

    def first_order_certificate(
        self,
        vector: ArrayLike,
        *,
        feasibility_tolerance: float = DEFAULT_COST_TOLERANCE,
        stationarity_tolerance: float = 1e-6,
        active_tolerance: float = 1e-6,
    ) -> RegionalFirstOrderCertificate:
        """Independently check primal feasibility and first-order KKT stationarity.

        Nonnegative Lagrange multipliers are fitted to the active regional
        inequalities and active lower bounds with nonnegative least squares.
        A small residual is necessary for a local maximum, but is not by itself
        sufficient without a second-order or problem-specific argument.
        """

        for name, tolerance in (
            ("feasibility_tolerance", feasibility_tolerance),
            ("stationarity_tolerance", stationarity_tolerance),
            ("active_tolerance", active_tolerance),
        ):
            if not np.isfinite(tolerance) or tolerance < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        values = self._validate_vector(vector)
        projected = np.maximum(values, 0.0)
        constraints = self.constraint_values(projected)
        active_inequalities = constraints >= -active_tolerance
        active_lower_bounds = projected <= active_tolerance
        gradient = self.profit_gradient(projected)

        columns: list[NDArray[np.float64]] = []
        active_inequality_indices = np.flatnonzero(active_inequalities)
        active_lower_bound_indices = np.flatnonzero(active_lower_bounds)
        if np.any(active_inequalities):
            columns.append(self.constraint_jacobian(projected)[active_inequalities].T)
        if np.any(active_lower_bounds):
            columns.append(-np.eye(self.variable_count)[:, active_lower_bounds])

        if columns:
            multiplier_matrix = np.column_stack(columns)
            multipliers, _ = nnls(multiplier_matrix, gradient)
            residual_vector = multiplier_matrix @ multipliers - gradient
        else:
            multipliers = np.empty(0, dtype=float)
            residual_vector = -gradient
        stationarity_residual = float(np.linalg.norm(residual_vector, ord=np.inf))
        gradient_scale = 1.0 + float(np.linalg.norm(gradient, ord=np.inf))

        return RegionalFirstOrderCertificate(
            primal_violation=self.maximum_constraint_violation(values),
            stationarity_residual=stationarity_residual,
            normalized_stationarity_residual=stationarity_residual / gradient_scale,
            active_inequality_count=int(np.count_nonzero(active_inequalities)),
            active_lower_bound_count=int(np.count_nonzero(active_lower_bounds)),
            feasibility_tolerance=float(feasibility_tolerance),
            stationarity_tolerance=float(stationarity_tolerance),
            active_inequality_indices=tuple(int(index) for index in active_inequality_indices),
            active_lower_bound_indices=tuple(int(index) for index in active_lower_bound_indices),
            inequality_multipliers=tuple(
                float(value) for value in multipliers[: active_inequality_indices.size]
            ),
            lower_bound_multipliers=tuple(
                float(value) for value in multipliers[active_inequality_indices.size :]
            ),
        )

    def _menu(self, vector: ArrayLike) -> BundleMenu:
        values = self._validate_vector(vector)
        if self._cached_vector is None or not np.array_equal(values, self._cached_vector):
            menu = self.layout.unpack(values)
            self._cached_vector = values.copy()
            self._cached_menu = menu
            self._cached_valuation = None
            self._cached_profit = None
            self._cached_profit_gradient = None
        assert self._cached_menu is not None
        return self._cached_menu

    def _evaluate(self, vector: ArrayLike) -> tuple[BundleMenu, ValuationTable]:
        menu = self._menu(vector)
        if self._cached_valuation is None:
            self._cached_valuation = self.problem.evaluate_valuations(menu)
        assert self._cached_valuation is not None
        return menu, self._cached_valuation

    def _validate_vector(self, vector: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(vector, dtype=float)
        if values.shape != (self.variable_count,):
            raise ValueError(f"solver vector must have shape ({self.variable_count},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("solver vector must contain only finite values")
        return values

    def _fill_choice_constraint_jacobian(
        self,
        row: NDArray[np.float64],
        valuation: ValuationTable,
        index: ChoiceConstraintIndex,
    ) -> None:
        assigned = index.assigned_bundle
        alternative = index.alternative_bundle
        traveler = index.traveler
        if assigned > 0:
            row[assigned - 1] += 1.0
            start = self.layout.designed_bundle_count + (assigned - 1) * self.problem.mode_count
            row[start : start + self.problem.mode_count] -= valuation.expected_omega[
                traveler, assigned
            ]
        if alternative > 0:
            row[alternative - 1] -= 1.0
            start = self.layout.designed_bundle_count + (alternative - 1) * self.problem.mode_count
            row[start : start + self.problem.mode_count] += valuation.expected_omega[
                traveler, alternative
            ]


class RegionalSolveStatus(StrEnum):
    """Solver-independent regional termination status."""

    SUCCESS = "success"
    ITERATION_LIMIT = "iteration_limit"
    INFEASIBLE = "infeasible"
    FAILED = "failed"


@dataclass(frozen=True)
class RegionalFirstOrderCertificate:
    """Independent primal-feasibility and first-order KKT diagnostics.

    This is a necessary local-optimality check, not a second-order certificate.
    """

    primal_violation: float
    stationarity_residual: float
    normalized_stationarity_residual: float
    active_inequality_count: int
    active_lower_bound_count: int
    feasibility_tolerance: float
    stationarity_tolerance: float
    active_inequality_indices: tuple[int, ...] = ()
    active_lower_bound_indices: tuple[int, ...] = ()
    inequality_multipliers: tuple[float, ...] = ()
    lower_bound_multipliers: tuple[float, ...] = ()

    @property
    def satisfies_kkt(self) -> bool:
        """Return whether primal feasibility and stationarity pass tolerance."""

        return (
            self.primal_violation <= self.feasibility_tolerance
            and self.normalized_stationarity_residual <= self.stationarity_tolerance
        )


@dataclass(frozen=True)
class RegionalSolveResult:
    """Unified regional solution, diagnostics, and observable metrics."""

    solver_name: str
    status: RegionalSolveStatus
    menu: BundleMenu
    objective_value: float
    solve_time_seconds: float
    iteration_count: int
    maximum_constraint_violation: float
    optimality_measure: float | None
    first_order_certificate: RegionalFirstOrderCertificate
    active_choice_constraint_count: int
    variable_count: int
    message: str
    raw_status: int | None = None
    attempt_count: int = 1
    attempted_solvers: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        """Return whether the adapter accepted the solution as feasible success."""

        return self.status is RegionalSolveStatus.SUCCESS

    def metrics(self) -> dict[str, str | int | float | bool | None]:
        """Return a terminal-report-friendly metrics record."""

        return {
            "solver": self.solver_name,
            "status": self.status.value,
            "time_seconds": self.solve_time_seconds,
            "iterations": self.iteration_count,
            "objective": self.objective_value,
            "max_violation": self.maximum_constraint_violation,
            "optimality": self.optimality_measure,
            "kkt_residual": self.first_order_certificate.normalized_stationarity_residual,
            "kkt_satisfied": self.first_order_certificate.satisfies_kkt,
            "critical_constraints": self.active_choice_constraint_count,
            "variables": self.variable_count,
            "attempts": self.attempt_count,
        }


class RegionalNlpSolver(Protocol):
    """Adapter contract for a local restricted-regional NLP solver.

    A result marked ``SUCCESS`` is expected to have passed the adapter's
    feasibility and first-order KKT tolerances.
    """

    @property
    def name(self) -> str:
        """Return a stable solver/backend name."""

    def solve(
        self,
        regional_problem: RestrictedRegionalProblem,
        initial_menu: BundleMenu,
    ) -> RegionalSolveResult:
        """Solve one restricted fixed-assignment regional problem."""


@dataclass(frozen=True)
class ScipyRegionalSolver:
    """SciPy local NLP adapter with an on-failure recovery chain.

    A successful first solve returns immediately.  Failed SLSQP solves are
    restarted once from the best feasible point available and then fall back
    to ``trust-constr``. Direct ``trust-constr`` use remains a single attempt.
    """

    method: str = "SLSQP"
    options: Mapping[str, int | float | bool] | None = None
    feasibility_tolerance: float = DEFAULT_COST_TOLERANCE
    optimality_tolerance: float = 1e-5
    active_set_tolerance: float = 1e-4
    enable_recovery: bool = True

    def __post_init__(self) -> None:
        """Validate the algorithm and freeze its option mapping."""

        normalized = self.method.lower()
        names = {"slsqp": "SLSQP", "trust-constr": "trust-constr"}
        if normalized not in names:
            raise ValueError("method must be 'SLSQP' or 'trust-constr'")
        if not np.isfinite(self.feasibility_tolerance) or self.feasibility_tolerance < 0:
            raise ValueError("feasibility_tolerance must be finite and nonnegative")
        if not np.isfinite(self.optimality_tolerance) or self.optimality_tolerance < 0:
            raise ValueError("optimality_tolerance must be finite and nonnegative")
        if not np.isfinite(self.active_set_tolerance) or self.active_set_tolerance < 0:
            raise ValueError("active_set_tolerance must be finite and nonnegative")
        if not isinstance(self.enable_recovery, (bool, np.bool_)):
            raise ValueError("enable_recovery must be Boolean")
        options = {} if self.options is None else dict(self.options)
        object.__setattr__(self, "method", names[normalized])
        object.__setattr__(self, "options", MappingProxyType(options))
        object.__setattr__(self, "enable_recovery", bool(self.enable_recovery))

    @property
    def name(self) -> str:
        """Return the SciPy algorithm name used by this adapter."""

        return f"scipy-{self.method}"

    def solve(
        self,
        regional_problem: RestrictedRegionalProblem,
        initial_menu: BundleMenu,
    ) -> RegionalSolveResult:
        """Solve once normally and activate recovery only after rejection."""

        first = self._solve_once(regional_problem, initial_menu)
        if first.success or self.method != "SLSQP" or not self.enable_recovery:
            return first

        attempts = [first]
        initial_vector = regional_problem.layout.pack(initial_menu)
        feasible_anchor = (
            initial_menu
            if regional_problem.maximum_constraint_violation(initial_vector)
            <= self.feasibility_tolerance
            else None
        )
        restart_menu, feasible_anchor = _prepare_recovery_start(
            regional_problem,
            first.menu,
            feasible_anchor,
            tolerance=self.feasibility_tolerance,
        )
        restarted = self._solve_once(regional_problem, restart_menu)
        attempts.append(restarted)
        if restarted.success:
            return _combine_attempts(attempts)

        fallback_menu, _ = _prepare_recovery_start(
            regional_problem,
            restarted.menu,
            feasible_anchor,
            tolerance=self.feasibility_tolerance,
        )
        primary_iteration_limit = int(self.options.get("maxiter", 1_000))
        fallback_options: dict[str, int | float | bool] = {
            "maxiter": max(primary_iteration_limit, 500),
            "gtol": max(min(self.optimality_tolerance, 1e-8), np.finfo(float).eps),
        }
        fallback = ScipyRegionalSolver(
            "trust-constr",
            options=fallback_options,
            feasibility_tolerance=self.feasibility_tolerance,
            optimality_tolerance=self.optimality_tolerance,
            active_set_tolerance=max(self.active_set_tolerance, 1e-3),
            enable_recovery=False,
        ).solve(regional_problem, fallback_menu)
        attempts.append(fallback)
        return _combine_attempts(attempts)

    def _solve_once(
        self,
        regional_problem: RestrictedRegionalProblem,
        initial_menu: BundleMenu,
    ) -> RegionalSolveResult:
        """Run exactly one SciPy minimize call."""

        initial = regional_problem.layout.pack(initial_menu)
        bounds = Bounds(
            np.zeros(regional_problem.variable_count),
            np.full(regional_problem.variable_count, np.inf),
            keep_feasible=True,
        )
        constraints: tuple[object, ...] = ()
        if regional_problem.constraint_count:
            if self.method == "SLSQP":
                constraints = (
                    {
                        "type": "ineq",
                        "fun": lambda x: -regional_problem.constraint_values(x),
                        "jac": lambda x: -regional_problem.constraint_jacobian(x),
                    },
                )
            else:
                constraints = (
                    NonlinearConstraint(
                        fun=lambda x: regional_problem.constraint_values(np.maximum(x, 0.0)),
                        lb=np.full(regional_problem.constraint_count, -np.inf),
                        ub=np.zeros(regional_problem.constraint_count),
                        jac=lambda x: regional_problem.constraint_jacobian(np.maximum(x, 0.0)),
                        hess="2-point",
                    ),
                )

        started = perf_counter()
        solver_options = dict(self.options)
        if (
            self.method == "trust-constr"
            and regional_problem.constraint_count > regional_problem.variable_count
        ):
            # Dense SVD handles the linearly dependent traveler constraints
            # that arise when several critical classes support the same pair.
            solver_options.setdefault("factorization_method", "SVDFactorization")
        arguments: dict[str, object] = {
            "fun": lambda x: -regional_problem.profit(x),
            "x0": initial,
            "jac": lambda x: -regional_problem.profit_gradient(x),
            "method": self.method,
            "bounds": bounds,
            "constraints": constraints,
            "options": solver_options,
        }
        if self.method == "trust-constr":
            if isinstance(regional_problem.problem.provision_cost, LinearProvisionCost):
                arguments["hess"] = lambda x: np.zeros(
                    (regional_problem.variable_count, regional_problem.variable_count)
                )
            else:
                arguments["hess"] = BFGS()
        result = minimize(
            **arguments,
        )
        elapsed = perf_counter() - started
        return self._normalize_result(regional_problem, result, elapsed)

    def _normalize_result(
        self,
        regional_problem: RestrictedRegionalProblem,
        result: OptimizeResult,
        elapsed: float,
    ) -> RegionalSolveResult:
        raw_vector = np.asarray(result.x, dtype=float)
        projected_vector = np.maximum(raw_vector, 0.0)
        menu = regional_problem.layout.unpack(projected_vector)
        violation = regional_problem.maximum_constraint_violation(raw_vector)
        raw_optimality = getattr(result, "optimality", None)
        optimality = float(raw_optimality) if raw_optimality is not None else None
        acceptable_optimality = optimality is None or optimality <= self.optimality_tolerance
        certificate = regional_problem.first_order_certificate(
            raw_vector,
            feasibility_tolerance=self.feasibility_tolerance,
            stationarity_tolerance=self.optimality_tolerance,
            active_tolerance=self.active_set_tolerance,
        )
        status = _map_scipy_status(
            success=(acceptable_optimality and certificate.satisfies_kkt),
            message=str(result.message),
            maximum_violation=violation,
            feasibility_tolerance=self.feasibility_tolerance,
        )
        return RegionalSolveResult(
            solver_name=self.name,
            status=status,
            menu=menu,
            objective_value=regional_problem.profit(projected_vector),
            solve_time_seconds=float(elapsed),
            iteration_count=int(getattr(result, "nit", 0)),
            maximum_constraint_violation=violation,
            optimality_measure=optimality,
            first_order_certificate=certificate,
            active_choice_constraint_count=regional_problem.active_choice_constraint_count,
            variable_count=regional_problem.variable_count,
            message=str(result.message),
            raw_status=int(result.status) if result.status is not None else None,
            attempted_solvers=(self.name,),
        )


def backtrack_to_feasible_menu(
    regional_problem: RestrictedRegionalProblem,
    candidate: BundleMenu,
    feasible_anchor: BundleMenu,
    *,
    tolerance: float,
) -> BundleMenu:
    """Return a nearby feasible point on the anchor-to-candidate segment."""

    layout = regional_problem.layout
    candidate_vector = layout.pack(candidate)
    if regional_problem.maximum_constraint_violation(candidate_vector) <= tolerance:
        return candidate

    anchor_vector = layout.pack(feasible_anchor)
    if regional_problem.maximum_constraint_violation(anchor_vector) > tolerance:
        return candidate

    direction = candidate_vector - anchor_vector
    infeasible_factor = 1.0
    feasible_factor = 0.0
    for _ in range(50):
        factor = 0.5 * infeasible_factor
        trial = anchor_vector + factor * direction
        if regional_problem.maximum_constraint_violation(trial) <= tolerance:
            feasible_factor = factor
            break
        infeasible_factor = factor
    else:
        return feasible_anchor

    for _ in range(8):
        factor = 0.5 * (feasible_factor + infeasible_factor)
        trial = anchor_vector + factor * direction
        if regional_problem.maximum_constraint_violation(trial) <= tolerance:
            feasible_factor = factor
        else:
            infeasible_factor = factor

    feasible_factor *= 0.999
    restored = anchor_vector + feasible_factor * direction
    if regional_problem.maximum_constraint_violation(restored) > tolerance:
        return feasible_anchor
    return layout.unpack(restored)


def _prepare_recovery_start(
    regional_problem: RestrictedRegionalProblem,
    candidate: BundleMenu,
    feasible_anchor: BundleMenu | None,
    *,
    tolerance: float,
) -> tuple[BundleMenu, BundleMenu | None]:
    """Choose a feasible restart when one is available."""

    candidate_vector = regional_problem.layout.pack(candidate)
    if regional_problem.maximum_constraint_violation(candidate_vector) <= tolerance:
        return candidate, candidate
    if feasible_anchor is None:
        return candidate, None
    return (
        backtrack_to_feasible_menu(
            regional_problem,
            candidate,
            feasible_anchor,
            tolerance=tolerance,
        ),
        feasible_anchor,
    )


def _combine_attempts(attempts: list[RegionalSolveResult]) -> RegionalSolveResult:
    """Return the final result with cumulative retry metrics."""

    final = attempts[-1]
    trace = " -> ".join(f"{attempt.solver_name}:{attempt.status.value}" for attempt in attempts)
    return replace(
        final,
        solve_time_seconds=sum(attempt.solve_time_seconds for attempt in attempts),
        iteration_count=sum(attempt.iteration_count for attempt in attempts),
        message=f"recovery chain [{trace}]; final: {final.message}",
        attempt_count=len(attempts),
        attempted_solvers=tuple(
            solver for attempt in attempts for solver in attempt.attempted_solvers
        ),
    )


def _choice_constraint_value(
    menu: BundleMenu,
    valuation: ValuationTable,
    index: ChoiceConstraintIndex,
) -> float:
    traveler = index.traveler
    assigned = index.assigned_bundle
    alternative = index.alternative_bundle
    return float(
        menu.prices[assigned]
        - menu.prices[alternative]
        + valuation.route_values[traveler, assigned]
        - valuation.route_values[traveler, alternative]
    )


def _map_scipy_status(
    *,
    success: bool,
    message: str,
    maximum_violation: float,
    feasibility_tolerance: float,
) -> RegionalSolveStatus:
    if success and maximum_violation <= feasibility_tolerance:
        return RegionalSolveStatus.SUCCESS
    normalized_message = message.lower()
    exhausted_budget = (
        "iteration" in normalized_message or "function evaluation" in normalized_message
    ) and ("limit" in normalized_message or "maximum" in normalized_message)
    if exhausted_budget:
        return RegionalSolveStatus.ITERATION_LIMIT
    if maximum_violation > feasibility_tolerance:
        return RegionalSolveStatus.INFEASIBLE
    return RegionalSolveStatus.FAILED
