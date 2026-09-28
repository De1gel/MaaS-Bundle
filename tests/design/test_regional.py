"""Tests for restricted regional NLP assembly and solver adapters."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import maas_bundle.design.regional as regional_module
from maas_bundle import (
    BundleDesignProblem,
    BundleMenu,
    ChoiceConstraintIndex,
    CriticalConstraintSet,
    MenuVariableLayout,
    RegionalSolveStatus,
    RestrictedRegionalProblem,
    RouteValueOracle,
    ScipyRegionalSolver,
    UniformDemandDistribution,
)
from maas_bundle.design.regional import _map_scipy_status


def _constant_oracle(slope: float, omega: float = 0.0) -> RouteValueOracle:
    return RouteValueOracle(np.array([[slope, omega]]))


def _bounded_price_problem() -> BundleDesignProblem:
    distributions = tuple(UniformDemandDistribution(0.5, 1.5) for _ in range(3))
    return BundleDesignProblem(
        traveler_weights=np.ones(3),
        unit_allowance_cost=np.array([1.0]),
        route_oracles=tuple(
            (_constant_oracle(outside_value), _constant_oracle(0.0))
            for outside_value in (5.0, 4.0, 3.0)
        ),
        demand_distributions=distributions,
    )


def _scaled_bounded_price_problem(
    traveler_count: int,
) -> tuple[BundleDesignProblem, CriticalConstraintSet]:
    """Build an affine benchmark whose unique optimum is C=3, epsilon=0.

    Every traveler is assigned to subscription 1 and contributes one constraint
    ``C_1 <= cap_i``.  Since ``min(cap_i)=3`` and the normalized fixed-assignment
    objective is ``C_1 - epsilon_1``, the analytical global optimum is known.
    """

    caps = np.linspace(5.0, 3.0, traveler_count)
    distribution = UniformDemandDistribution(0.5, 1.5)
    problem = BundleDesignProblem(
        traveler_weights=np.full(traveler_count, 1.0 / traveler_count),
        unit_allowance_cost=np.array([1.0]),
        route_oracles=tuple(
            (_constant_oracle(cap), _constant_oracle(0.0)) for cap in caps
        ),
        demand_distributions=tuple(distribution for _ in range(traveler_count)),
    )
    pool = CriticalConstraintSet(np.ones(traveler_count, dtype=np.int64), bundle_count=2)
    pool.add_all(
        ChoiceConstraintIndex(traveler, 1, 0) for traveler in range(traveler_count)
    )
    return problem, pool


def _critical_pool(active_count: int, *, assigned_bundle: int = 1) -> CriticalConstraintSet:
    pool = CriticalConstraintSet(np.full(3, assigned_bundle, dtype=np.int64), bundle_count=2)
    for traveler in range(active_count):
        pool.add(ChoiceConstraintIndex(traveler, assigned_bundle, 0))
    return pool


def test_layout_round_trips_menu_and_omits_payg() -> None:
    layout = MenuVariableLayout(bundle_count=3, mode_count=2)
    menu = BundleMenu(
        prices=np.array([0.0, 2.0, 4.0]),
        allowances=np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]),
    )

    vector = layout.pack(menu)
    restored = layout.unpack(vector)

    assert layout.designed_bundle_count == 2
    assert layout.variable_count == 6
    np.testing.assert_allclose(vector, [2.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(restored.prices, menu.prices)
    np.testing.assert_allclose(restored.allowances, menu.allowances)
    assert not vector.flags.writeable


@pytest.mark.parametrize(
    ("bundle_count", "mode_count", "message"),
    [
        (1, 1, "PAYG and at least one"),
        (2.5, 1, "PAYG and at least one"),
        (2, 0, "positive integer"),
        (2, 1.5, "positive integer"),
    ],
)
def test_layout_rejects_invalid_dimensions(bundle_count, mode_count, message) -> None:
    with pytest.raises(ValueError, match=message):
        MenuVariableLayout(bundle_count, mode_count)


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([1.0], "must have shape"),
        ([np.nan, 0.0], "only finite values"),
        ([-1.0, 0.0], "must be nonnegative"),
    ],
)
def test_layout_rejects_invalid_solver_vector(vector, message) -> None:
    with pytest.raises(ValueError, match=message):
        MenuVariableLayout(2, 1).unpack(vector)


@pytest.mark.parametrize(
    ("menu", "message"),
    [
        (BundleMenu(np.zeros(3), np.zeros((3, 1))), "same bundles"),
        (BundleMenu(np.zeros(2), np.zeros((2, 2))), "same allowance dimensions"),
        (BundleMenu(np.array([1.0, 2.0]), np.zeros((2, 1))), "fixed PAYG"),
        (BundleMenu(np.zeros(2), np.array([[1.0], [0.0]])), "fixed PAYG"),
    ],
)
def test_layout_rejects_incompatible_menu(menu, message) -> None:
    with pytest.raises(ValueError, match=message):
        MenuVariableLayout(2, 1).pack(menu)


def _two_direction_problem() -> tuple[BundleDesignProblem, CriticalConstraintSet, BundleMenu]:
    distribution = UniformDemandDistribution(2.0, 3.0)
    problem = BundleDesignProblem(
        traveler_weights=np.ones(2),
        unit_allowance_cost=np.array([0.5]),
        route_oracles=(
            (_constant_oracle(2.0, 1.0), _constant_oracle(2.0, 2.0)),
            (_constant_oracle(2.0, 1.0), _constant_oracle(2.0, 3.0)),
        ),
        demand_distributions=(distribution, distribution),
    )
    pool = CriticalConstraintSet([1, 0], bundle_count=2)
    pool.add_all(
        (
            ChoiceConstraintIndex(0, 1, 0),
            ChoiceConstraintIndex(1, 0, 1),
        )
    )
    menu = BundleMenu(np.array([0.0, 1.0]), np.array([[0.0], [0.5]]))
    return problem, pool, menu


def test_regional_objective_constraints_and_jacobians() -> None:
    problem, pool, menu = _two_direction_problem()
    regional = RestrictedRegionalProblem(problem, pool)
    vector = regional.layout.pack(menu)

    assert regional.variable_count == 2
    assert regional.active_choice_constraint_count == 2
    assert regional.price_monotonicity_constraint_count == 0
    assert regional.constraint_count == 2
    assert regional.profit(vector) == pytest.approx(0.75)
    np.testing.assert_allclose(regional.profit_gradient(vector), [1.0, -0.5])
    np.testing.assert_allclose(regional.constraint_values(vector), [0.0, 0.5])
    np.testing.assert_allclose(
        regional.constraint_jacobian(vector),
        [[1.0, -2.0], [-1.0, 3.0]],
    )


def _finite_difference(function, vector, step=1e-6):
    vector = np.asarray(vector, dtype=float)
    baseline = np.asarray(function(vector))
    jacobian = np.empty((baseline.size, vector.size))
    for column in range(vector.size):
        forward = vector.copy()
        backward = vector.copy()
        forward[column] += step
        backward[column] -= step
        jacobian[:, column] = (
            np.asarray(function(forward)).reshape(-1) - np.asarray(function(backward)).reshape(-1)
        ) / (2.0 * step)
    return jacobian.reshape((*baseline.shape, vector.size))


def test_regional_derivatives_match_finite_differences() -> None:
    problem, pool, menu = _two_direction_problem()
    regional = RestrictedRegionalProblem(problem, pool)
    vector = regional.layout.pack(menu)

    objective_fd = _finite_difference(lambda x: np.array([regional.profit(x)]), vector)[0]
    constraints_fd = _finite_difference(regional.constraint_values, vector)

    np.testing.assert_allclose(regional.profit_gradient(vector), objective_fd, atol=1e-7)
    np.testing.assert_allclose(regional.constraint_jacobian(vector), constraints_fd, atol=1e-7)


def test_regional_adds_adjacent_price_monotonicity_constraints() -> None:
    distribution = UniformDemandDistribution(0.5, 1.5)
    problem = BundleDesignProblem(
        traveler_weights=np.array([1.0]),
        unit_allowance_cost=np.array([1.0]),
        route_oracles=((_constant_oracle(1.0), _constant_oracle(1.0), _constant_oracle(1.0)),),
        demand_distributions=(distribution,),
    )
    pool = CriticalConstraintSet([0], bundle_count=3)
    regional = RestrictedRegionalProblem(problem, pool)
    menu = BundleMenu(np.array([0.0, 3.0, 2.0]), np.zeros((3, 1)))
    vector = regional.layout.pack(menu)

    assert regional.price_monotonicity_constraint_count == 1
    np.testing.assert_allclose(regional.constraint_values(vector), [1.0])
    np.testing.assert_allclose(regional.constraint_jacobian(vector), [[1.0, -1.0, 0.0, 0.0]])
    assert regional.maximum_constraint_violation(vector) == pytest.approx(1.0)


def test_regional_reports_bound_violation_without_evaluating_negative_allowance() -> None:
    regional = RestrictedRegionalProblem(_bounded_price_problem(), _critical_pool(1))

    assert regional.maximum_constraint_violation([1.0, -0.25]) == pytest.approx(0.25)


def test_regional_reuses_valuation_for_identical_vector() -> None:
    regional = RestrictedRegionalProblem(_bounded_price_problem(), _critical_pool(1))
    vector = np.array([1.0, 0.0])

    first_menu, first_valuation = regional._evaluate(vector)
    second_menu, second_valuation = regional._evaluate(vector.copy())

    assert first_menu is second_menu
    assert first_valuation is second_valuation


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([1.0], "must have shape"),
        ([np.nan, 0.0], "only finite values"),
    ],
)
def test_regional_rejects_invalid_vector(vector, message) -> None:
    regional = RestrictedRegionalProblem(_bounded_price_problem(), _critical_pool(1))

    with pytest.raises(ValueError, match=message):
        regional.profit_gradient(vector)


def test_first_order_certificate_distinguishes_stationary_and_nonstationary_points() -> None:
    regional = RestrictedRegionalProblem(_bounded_price_problem(), _critical_pool(3))

    optimum = regional.first_order_certificate([3.0, 0.0])
    interior_nonoptimum = regional.first_order_certificate([1.0, 1.0])
    infeasible = regional.first_order_certificate([6.0, 0.0])

    assert optimum.satisfies_kkt
    assert optimum.active_inequality_count == 1
    assert optimum.active_lower_bound_count == 1
    assert optimum.stationarity_residual <= 1e-12
    assert not interior_nonoptimum.satisfies_kkt
    assert interior_nonoptimum.stationarity_residual > 0
    assert not infeasible.satisfies_kkt
    assert infeasible.primal_violation == pytest.approx(3.0)


@pytest.mark.parametrize(
    "keyword",
    ["feasibility_tolerance", "stationarity_tolerance", "active_tolerance"],
)
def test_first_order_certificate_rejects_invalid_tolerances(keyword) -> None:
    regional = RestrictedRegionalProblem(_bounded_price_problem(), _critical_pool(1))

    with pytest.raises(ValueError, match=keyword):
        regional.first_order_certificate([1.0, 0.0], **{keyword: -1.0})


def test_regional_rejects_incompatible_critical_set() -> None:
    problem = _bounded_price_problem()

    with pytest.raises(ValueError, match="same travelers"):
        RestrictedRegionalProblem(problem, CriticalConstraintSet([1], 2))
    with pytest.raises(ValueError, match="same bundles"):
        RestrictedRegionalProblem(problem, CriticalConstraintSet([1, 1, 1], 3))


def test_regional_requires_at_least_one_designed_bundle() -> None:
    problem = BundleDesignProblem(
        traveler_weights=np.array([1.0]),
        unit_allowance_cost=np.array([1.0]),
        route_oracles=((_constant_oracle(1.0),),),
        demand_distributions=(UniformDemandDistribution(0.5, 1.5),),
    )

    with pytest.raises(ValueError, match="at least one designed bundle"):
        RestrictedRegionalProblem(problem, CriticalConstraintSet([0], 1))


@pytest.mark.parametrize(
    ("method", "active_count", "expected_price"),
    [
        ("SLSQP", 1, 5.0),
        ("SLSQP", 2, 4.0),
        ("SLSQP", 3, 3.0),
        ("trust-constr", 1, 5.0),
        ("trust-constr", 2, 4.0),
        ("trust-constr", 3, 3.0),
    ],
)
def test_solvers_handle_different_critical_constraint_counts(
    method,
    active_count,
    expected_price,
    record_property,
) -> None:
    problem = _bounded_price_problem()
    regional = RestrictedRegionalProblem(problem, _critical_pool(active_count))
    initial = BundleMenu(np.array([0.0, 1.0]), np.zeros((2, 1)))
    options = (
        {"maxiter": 500, "ftol": 1e-10} if method == "SLSQP" else {"maxiter": 500, "gtol": 1e-9}
    )
    solver = ScipyRegionalSolver(method=method, options=options)

    result = solver.solve(regional, initial)
    record_property("regional_metric", result.metrics())

    assert result.success, result.message
    assert result.status is RegionalSolveStatus.SUCCESS
    assert result.menu.prices[1] == pytest.approx(expected_price, abs=2e-3)
    assert result.objective_value == pytest.approx(3.0 * expected_price, abs=6e-3)
    assert result.maximum_constraint_violation <= 1e-7
    assert result.active_choice_constraint_count == active_count
    assert result.variable_count == 2
    assert result.solve_time_seconds >= 0
    assert result.iteration_count >= 0
    assert result.metrics()["solver"] == solver.name


@pytest.mark.parametrize("method", ["SLSQP", "trust-constr"])
def test_solvers_support_zero_active_choice_constraints(method, record_property) -> None:
    problem = _bounded_price_problem()
    regional = RestrictedRegionalProblem(problem, _critical_pool(0, assigned_bundle=0))
    initial = BundleMenu(np.array([0.0, 1.0]), np.zeros((2, 1)))
    solver = ScipyRegionalSolver(method=method, options={"maxiter": 100})

    result = solver.solve(regional, initial)
    record_property("regional_metric", result.metrics())

    assert result.active_choice_constraint_count == 0
    assert result.objective_value == pytest.approx(0.0)
    if method == "SLSQP":
        assert result.success, result.message
    else:
        assert result.status in {
            RegionalSolveStatus.SUCCESS,
            RegionalSolveStatus.ITERATION_LIMIT,
            RegionalSolveStatus.FAILED,
        }


def _synthetic_optimize_result(
    vector,
    *,
    success: bool = False,
    status: int = 8,
    message: str = "synthetic stop",
    iterations: int = 1,
) -> OptimizeResult:
    return OptimizeResult(
        x=np.asarray(vector, dtype=float),
        success=success,
        status=status,
        message=message,
        nit=iterations,
    )


def test_slsqp_success_uses_one_attempt_without_recovery(monkeypatch) -> None:
    problem = _bounded_price_problem()
    regional = RestrictedRegionalProblem(problem, _critical_pool(3))
    initial = BundleMenu(np.array([0.0, 1.0]), np.zeros((2, 1)))
    calls = []

    def fake_minimize(**arguments):
        calls.append(arguments)
        return _synthetic_optimize_result([3.0, 0.0], success=True, status=0)

    monkeypatch.setattr(regional_module, "minimize", fake_minimize)

    result = ScipyRegionalSolver("SLSQP").solve(regional, initial)

    assert result.success
    assert len(calls) == 1
    assert result.attempt_count == 1
    assert result.attempted_solvers == ("scipy-SLSQP",)
    assert result.metrics()["attempts"] == 1


def test_slsqp_restarts_once_from_feasible_nonstationary_point(monkeypatch) -> None:
    problem = _bounded_price_problem()
    regional = RestrictedRegionalProblem(problem, _critical_pool(3))
    initial = BundleMenu(np.array([0.0, 1.0]), np.zeros((2, 1)))
    returned = iter(
        (
            _synthetic_optimize_result([1.0, 1.0], iterations=2),
            _synthetic_optimize_result([3.0, 0.0], success=True, status=0, iterations=3),
        )
    )
    calls = []

    def fake_minimize(**arguments):
        calls.append(arguments)
        return next(returned)

    monkeypatch.setattr(regional_module, "minimize", fake_minimize)

    result = ScipyRegionalSolver("SLSQP").solve(regional, initial)

    assert result.success
    assert [call["method"] for call in calls] == ["SLSQP", "SLSQP"]
    np.testing.assert_allclose(calls[1]["x0"], [1.0, 1.0])
    assert result.attempt_count == 2
    assert result.iteration_count == 5
    assert result.attempted_solvers == ("scipy-SLSQP", "scipy-SLSQP")
    assert "recovery chain" in result.message


def test_slsqp_backtracks_infeasible_result_before_restart(monkeypatch) -> None:
    problem = _bounded_price_problem()
    regional = RestrictedRegionalProblem(problem, _critical_pool(3))
    initial = BundleMenu(np.array([0.0, 1.0]), np.zeros((2, 1)))
    returned = iter(
        (
            _synthetic_optimize_result([6.0, 0.0]),
            _synthetic_optimize_result([3.0, 0.0], success=True, status=0),
        )
    )
    calls = []

    def fake_minimize(**arguments):
        calls.append(arguments)
        return next(returned)

    monkeypatch.setattr(regional_module, "minimize", fake_minimize)

    result = ScipyRegionalSolver("SLSQP").solve(regional, initial)

    assert result.success
    restart = np.asarray(calls[1]["x0"])
    assert 1.0 < restart[0] < 3.0
    assert regional.maximum_constraint_violation(restart) <= 1e-7


def test_slsqp_uses_infeasible_candidate_when_no_anchor_exists(monkeypatch) -> None:
    problem = _bounded_price_problem()
    regional = RestrictedRegionalProblem(problem, _critical_pool(3))
    initial = BundleMenu(np.array([0.0, 6.0]), np.zeros((2, 1)))
    returned = iter(
        (
            _synthetic_optimize_result([5.0, 0.0]),
            _synthetic_optimize_result([3.0, 0.0], success=True, status=0),
        )
    )
    calls = []

    def fake_minimize(**arguments):
        calls.append(arguments)
        return next(returned)

    monkeypatch.setattr(regional_module, "minimize", fake_minimize)

    result = ScipyRegionalSolver("SLSQP").solve(regional, initial)

    assert result.success
    np.testing.assert_allclose(calls[1]["x0"], [5.0, 0.0])


def test_slsqp_falls_back_to_trust_constr_with_numerical_constraint_hessian(
    monkeypatch,
) -> None:
    problem = _bounded_price_problem()
    regional = RestrictedRegionalProblem(problem, _critical_pool(3))
    initial = BundleMenu(np.array([0.0, 1.0]), np.zeros((2, 1)))
    returned = iter(
        (
            _synthetic_optimize_result([1.0, 1.0], iterations=2),
            _synthetic_optimize_result([1.0, 1.0], iterations=3),
            _synthetic_optimize_result([3.0, 0.0], success=True, status=1, iterations=4),
        )
    )
    calls = []

    def fake_minimize(**arguments):
        calls.append(arguments)
        return next(returned)

    monkeypatch.setattr(regional_module, "minimize", fake_minimize)

    result = ScipyRegionalSolver("SLSQP", options={"maxiter": 20}).solve(
        regional,
        initial,
    )

    assert result.success
    assert [call["method"] for call in calls] == ["SLSQP", "SLSQP", "trust-constr"]
    fallback_constraint = calls[2]["constraints"][0]
    assert fallback_constraint.hess == "2-point"
    assert calls[2]["options"] == {
        "maxiter": 500,
        "gtol": 1e-8,
        "factorization_method": "SVDFactorization",
    }
    assert result.attempt_count == 3
    assert result.iteration_count == 9
    assert result.attempted_solvers == (
        "scipy-SLSQP",
        "scipy-SLSQP",
        "scipy-trust-constr",
    )


def test_recovery_chain_fallback_reaches_known_optimum_with_real_trust_constr(
    monkeypatch,
) -> None:
    problem = _bounded_price_problem()
    regional = RestrictedRegionalProblem(problem, _critical_pool(3))
    initial = BundleMenu(np.array([0.0, 1.0]), np.zeros((2, 1)))
    scipy_minimize = regional_module.minimize
    slsqp_calls = 0

    def force_slsqp_failure(**arguments):
        nonlocal slsqp_calls
        if arguments["method"] == "SLSQP":
            slsqp_calls += 1
            return _synthetic_optimize_result([1.0, 1.0])
        return scipy_minimize(**arguments)

    monkeypatch.setattr(regional_module, "minimize", force_slsqp_failure)

    result = ScipyRegionalSolver("SLSQP", options={"maxiter": 20}).solve(
        regional,
        initial,
    )

    assert result.success, result.message
    assert slsqp_calls == 2
    assert result.attempt_count == 3
    assert result.solver_name == "scipy-trust-constr"
    assert result.menu.prices[1] == pytest.approx(3.0, abs=2e-3)
    assert result.menu.allowances[1, 0] == pytest.approx(0.0, abs=2e-3)
    assert result.objective_value == pytest.approx(9.0, abs=6e-3)
    assert result.first_order_certificate.satisfies_kkt


@pytest.mark.regional_scale
@pytest.mark.parametrize("traveler_count", [100, 1000])
def test_solver_scale_and_known_global_optimum(traveler_count, record_property) -> None:
    """Measure both solvers and reject false success on a known global optimum."""

    problem, pool = _scaled_bounded_price_problem(traveler_count)
    regional = RestrictedRegionalProblem(problem, pool)
    initial = BundleMenu(np.array([0.0, 1.0]), np.zeros((2, 1)))

    for method in ("SLSQP", "trust-constr"):
        options = (
            {"maxiter": 500, "ftol": 1e-10}
            if method == "SLSQP"
            else {"maxiter": 500, "gtol": 1e-9}
        )
        result = ScipyRegionalSolver(method=method, options=options).solve(regional, initial)
        known_global_optimum = (
            result.success
            and result.first_order_certificate.satisfies_kkt
            and result.objective_value == pytest.approx(3.0, abs=2e-3)
            and result.menu.prices[1] == pytest.approx(3.0, abs=2e-3)
            and result.menu.allowances[1, 0] == pytest.approx(0.0, abs=2e-3)
        )
        metric = result.metrics()
        metric["known_optimum"] = "global+local" if known_global_optimum else "not verified"
        record_property("regional_metric", metric)

        # An accepted result must match the independently known global optimum;
        # a backend may instead be rejected by the adapter's independent KKT check.
        assert not result.success or known_global_optimum
        assert result.maximum_constraint_violation <= 1e-7
        if method == "SLSQP":
            assert known_global_optimum, result.message


@pytest.mark.parametrize(
    ("method", "tolerance", "message"),
    [
        ("unknown", 1e-7, "method must be"),
        ("SLSQP", -1.0, "feasibility_tolerance"),
        ("SLSQP", np.nan, "feasibility_tolerance"),
    ],
)
def test_scipy_solver_rejects_invalid_configuration(method, tolerance, message) -> None:
    with pytest.raises(ValueError, match=message):
        ScipyRegionalSolver(method=method, feasibility_tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [-1.0, np.nan])
def test_scipy_solver_rejects_invalid_optimality_tolerance(tolerance) -> None:
    with pytest.raises(ValueError, match="optimality_tolerance"):
        ScipyRegionalSolver(optimality_tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [-1.0, np.nan])
def test_scipy_solver_rejects_invalid_active_set_tolerance(tolerance) -> None:
    with pytest.raises(ValueError, match="active_set_tolerance"):
        ScipyRegionalSolver(active_set_tolerance=tolerance)


def test_scipy_solver_rejects_non_boolean_recovery_flag() -> None:
    with pytest.raises(ValueError, match="enable_recovery"):
        ScipyRegionalSolver(enable_recovery=1)


@pytest.mark.parametrize(
    ("success", "message", "violation", "expected"),
    [
        (True, "done", 0.0, RegionalSolveStatus.SUCCESS),
        (False, "Iteration limit reached", 0.0, RegionalSolveStatus.ITERATION_LIMIT),
        (False, "failed", 1.0, RegionalSolveStatus.INFEASIBLE),
        (False, "failed", 0.0, RegionalSolveStatus.FAILED),
    ],
)
def test_maps_solver_statuses(success, message, violation, expected) -> None:
    assert (
        _map_scipy_status(
            success=success,
            message=message,
            maximum_violation=violation,
            feasibility_tolerance=1e-7,
        )
        is expected
    )
