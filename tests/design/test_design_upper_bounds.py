"""Finite design boxes must reach every backend and the independent KKT check."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import maas_bundle.design.regional as regional_module
from maas_bundle.design.boundary import ChoiceConstraintIndex, CriticalConstraintSet
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import solve_primal
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import RestrictedRegionalProblem, ScipyRegionalSolver
from maas_bundle.examples.stylized import build_stylized_problem, initial_stylized_menu
from maas_bundle.oracle.demand_distribution import UniformDemandDistribution
from maas_bundle.oracle.route_value import RouteValueOracle


def _regional(*, price_cap=2.0, allowance_cap=0.75, subscription_slope=0.0, omega=0.0):
    problem = BundleDesignProblem(
        traveler_weights=np.ones(1),
        unit_allowance_cost=np.ones(1),
        route_oracles=(
            (
                RouteValueOracle(np.array([[4.0, 0.0]])),
                RouteValueOracle(np.array([[subscription_slope, omega]])),
            ),
        ),
        demand_distributions=(UniformDemandDistribution(0.5, 1.5),),
        price_upper_bounds=price_cap,
        allowance_upper_bounds=allowance_cap,
    )
    pool = CriticalConstraintSet(np.array([1]), 2)
    pool.add(ChoiceConstraintIndex(0, 1, 0))
    return RestrictedRegionalProblem(problem, pool)


def test_upper_price_is_a_valid_stationary_maximum_and_violation_is_measured():
    region = _regional()
    certificate = region.first_order_certificate([2, 0])
    assert certificate.satisfies_kkt
    assert certificate.active_upper_bound_indices == (0,)
    assert certificate.upper_bound_multipliers == pytest.approx((1,))
    assert region.maximum_constraint_violation([2.4, 0]) == pytest.approx(0.4)
    assert region.maximum_constraint_violation([1, 1]) == pytest.approx(0.25)
    result = ScipyRegionalSolver().solve(region, BundleMenu(np.array([0, 1]), np.zeros((2, 1))))
    assert result.success
    assert result.menu.prices[1] == pytest.approx(2)


def test_upper_allowance_can_support_the_optimum():
    region = _regional(price_cap=10, allowance_cap=1, subscription_slope=3, omega=2)
    initial = BundleMenu(np.array([0, 0.5]), np.array([[0], [0.5]]))
    result = ScipyRegionalSolver(options={"ftol": 1e-12}).solve(region, initial)
    assert result.success
    np.testing.assert_allclose(region.layout.pack(result.menu), [3, 1], atol=1e-7)
    assert result.first_order_certificate.active_upper_bound_indices == (1,)


def test_solver_supplies_its_default_numerical_seed_without_a_caller_menu():
    region = _regional()
    solver = ScipyRegionalSolver(enable_recovery=False)
    seed = solver.default_initial_menu(region)
    np.testing.assert_array_equal(region.layout.pack(seed), [0, 0])
    result = solver.solve(region)
    assert result.success
    assert result.attempt_count == 1
    np.testing.assert_allclose(region.layout.pack(result.menu), [2, 0], atol=1e-7)


def test_restarts_and_trust_constr_receive_identical_finite_bounds(monkeypatch):
    region = _regional()
    outputs = iter(([1, 0.5], [1, 0.5], [2, 0]))
    calls = []

    def fake_minimize(**arguments):
        calls.append(arguments)
        return OptimizeResult(x=np.array(next(outputs)), nit=1, message="synthetic stop", status=8)

    monkeypatch.setattr(regional_module, "minimize", fake_minimize)
    result = ScipyRegionalSolver().solve(
        region, BundleMenu(np.array([0, 1]), np.array([[0], [0.2]]))
    )
    assert result.success
    assert result.attempt_count == 3
    assert [c["method"] for c in calls] == ["SLSQP", "SLSQP", "trust-constr"]
    for call in calls:
        np.testing.assert_array_equal(call["bounds"].ub, [2, 0.75])
        np.testing.assert_array_equal(call["bounds"].lb, [0, 0])
        assert np.all(call["x0"] <= call["bounds"].ub)


@pytest.mark.parametrize("bounds", [-1, np.nan])
def test_invalid_upper_bounds_rejected(bounds):
    with pytest.raises(ValueError):
        _regional(price_cap=bounds)


def test_default_domain_stays_unbounded_and_search_rejects_an_outside_initial_menu():
    unbounded = build_stylized_problem()
    assert np.isinf(unbounded.price_upper_bounds).all()
    assert np.isinf(unbounded.allowance_upper_bounds).all()
    bounded = build_stylized_problem(price_upper_bound=50, allowance_upper_bound=64)
    with pytest.raises(ValueError, match="design upper bounds"):
        solve_primal(bounded, initial_stylized_menu(), ScipyRegionalSolver())
