"""General provision costs across assignment, differentiation, and optimization."""

from dataclasses import replace

import numpy as np
import pytest

from maas_bundle import (
    BundleDesignProblem,
    BundleMenu,
    CallableProvisionCost,
    ChoiceConstraintIndex,
    CriticalConstraintSet,
    LinearProvisionCost,
    RegionalSolveStatus,
    RestrictedRegionalProblem,
    RouteValueOracle,
    ScipyRegionalSolver,
    UniformDemandDistribution,
    solve_primal,
)
from maas_bundle.benchmarks.initialization import sample_profitable_menus


def _problem(cost, *, subscriptions=1):
    # With mu in [1, 2], savings are epsilon for epsilon <= 1, and <= epsilon
    # everywhere. K=epsilon**2 therefore gives global profit <= 1/4, attained
    # at C=epsilon=1/2. This checks the nonlinear solver against an external bound.
    oracle = RouteValueOracle([[1.0, 0.0], [2.0, 1.0]])
    return BundleDesignProblem(
        traveler_weights=np.array([1.0]),
        provision_cost=cost,
        route_oracles=((oracle,) * (subscriptions + 1),),
        demand_distributions=(UniformDemandDistribution(1.0, 2.0),),
    )


def _quadratic(*, analytic=True):
    return CallableProvisionCost(
        1,
        value=lambda e: np.sum(e**2, axis=1),
        gradient=(lambda e: 2 * e) if analytic else None,
    )


def _menu(price=0.2, allowance=0.25):
    return BundleMenu(np.array([0.0, price]), np.array([[0.0], [allowance]]))


def _regional(problem):
    pool = CriticalConstraintSet([1], bundle_count=2)
    pool.add(ChoiceConstraintIndex(0, 1, 0))
    return RestrictedRegionalProblem(problem, pool)


def test_legacy_linear_constructor_and_general_model_are_equivalent():
    general = _problem(LinearProvisionCost(np.array([0.2])))
    legacy = BundleDesignProblem(
        general.traveler_weights,
        np.array([0.2]),
        general.route_oracles,
        general.demand_distributions,
    )
    for problem in (general, legacy):
        result = problem.evaluate_assignment(_menu())
        assert result.total_profit == pytest.approx(0.15)
        np.testing.assert_array_equal(result.bundle_indices, [1])
        np.testing.assert_array_equal(problem.unit_allowance_cost, [0.2])
        assert not problem.unit_allowance_cost.flags.writeable
    nonlinear = replace(legacy, provision_cost=_quadratic())
    assert nonlinear.unit_allowance_cost is None
    assert nonlinear.evaluate_assignment(_menu()).total_profit == pytest.approx(0.1375)


def test_bundle_specific_cost_changes_optimistic_tie_break():
    cost = CallableProvisionCost(
        1, lambda e: np.array([2.0, 0.5]) * e[:, 0] ** 2,
        lambda e: np.array([[4.0], [1.0]]) * e,
    )
    problem = _problem(cost, subscriptions=2)
    menu = BundleMenu(np.array([0.0, 0.5, 0.5]), np.array([[0.0], [0.5], [0.5]]))
    result = problem.evaluate_assignment(menu)
    # Both subscriptions and PAYG tie in traveler cost, but margins differ.
    np.testing.assert_allclose(result.generalized_costs, [[3.0, 3.0, 3.0]])
    np.testing.assert_allclose(result.bundle_margins, [0.0, 0.0, 0.375])
    np.testing.assert_array_equal(result.bundle_indices, [2])
    assert result.total_profit == pytest.approx(0.375)


def test_numerical_gradient_couples_modes_and_stays_in_nonnegative_domain():
    def value(e):
        assert np.all(e >= 0)
        assert not e.flags.writeable
        return np.array([1.0, 3.0]) * (e[:, 0] + 2 * e[:, 1]) ** 2

    cost = CallableProvisionCost(2, value)
    e = np.array([[0.0, 0.8], [0.4, 0.0]])
    e.setflags(write=False)
    expected = 2 * np.array([[1.0], [3.0]]) * (e[:, :1] + 2 * e[:, 1:]) * [[1, 2]]
    np.testing.assert_allclose(cost.gradients(e), expected, rtol=1e-7, atol=1e-8)


def test_nonconvex_nonmonotone_cost_allows_negative_gradient():
    cost = CallableProvisionCost(
        1, lambda e: np.sin(e[:, 0]) ** 2,
        lambda e: np.sin(2 * e),
    )
    problem = _problem(cost)
    menu = _menu(1.0, 2.0)
    assert problem.provision_cost_gradients(menu)[0, 0] < 0
    assert _regional(problem).profit_gradient([1.0, 2.0])[1] > 0


def test_numerical_gradient_supports_callbacks_reusing_output_buffers():
    buffer = np.empty(2)

    def value(e):
        buffer[:] = np.sum(e**2, axis=1)
        return buffer

    e = np.array([[0.0, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(CallableProvisionCost(2, value).gradients(e), 2 * e, atol=1e-8)


@pytest.mark.parametrize("analytic", [True, False])
def test_regional_gradient_matches_finite_difference(analytic):
    regional = _regional(_problem(_quadratic(analytic=analytic)))
    x = np.array([0.2, 0.25])
    h = 1e-6
    expected = [
        (regional.profit(x + d) - regional.profit(x - d)) / (2 * h)
        for d in h * np.eye(2)
    ]
    np.testing.assert_allclose(regional.profit_gradient(x), expected, atol=1e-8)


@pytest.mark.parametrize("method", ["SLSQP", "trust-constr"])
@pytest.mark.parametrize("analytic", [True, False])
def test_nonlinear_solvers_reach_known_optimum(method, analytic):
    problem = _problem(_quadratic(analytic=analytic))
    solver = ScipyRegionalSolver(method, options={"maxiter": 500})
    result = solver.solve(_regional(problem), _menu())
    assert result.status is RegionalSolveStatus.SUCCESS
    assert result.objective_value == pytest.approx(0.25, abs=2e-5)
    assert result.menu.prices[1] == pytest.approx(0.5, abs=2e-4)
    assert result.menu.allowances[1, 0] == pytest.approx(0.5, abs=2e-4)


def test_nonlinear_cost_used_in_initialization_and_primal_profit_protection():
    problem = _problem(_quadratic())
    loss = _menu(0.1, 0.5)
    initial = _menu()
    proposals = iter([loss, initial])
    assert sample_profitable_menus(problem, lambda: next(proposals), 1) == (initial,)
    with pytest.raises(ValueError, match="positive"):
        solve_primal(problem, loss, ScipyRegionalSolver())
    result = solve_primal(problem, initial, ScipyRegionalSolver())
    assert result.valid
    assert result.candidate.true_profit == pytest.approx(0.25, abs=2e-5)
    profits = [problem.evaluate_assignment(initial).total_profit]
    profits += [step.true_profit for step in result.candidate.iterations if step.accepted]
    assert np.all(np.diff(profits) >= 0)


def test_cost_only_evaluation_avoids_route_oracles_and_reuses_callback_results(monkeypatch):
    calls = {"value": 0, "gradient": 0}

    def value(e):
        calls["value"] += 1
        assert not e.flags.writeable
        return np.sum(e**2, axis=1)

    def gradient(e):
        calls["gradient"] += 1
        assert not e.flags.writeable
        return 2 * e

    problem = _problem(CallableProvisionCost(1, value, gradient))
    calls["value"] = 0  # Ignore K(0) validation during construction.

    def forbidden(*args):
        pytest.fail("cost-only evaluation must not evaluate route oracles")

    monkeypatch.setattr(BundleDesignProblem, "evaluate_valuations", forbidden)
    regional = _regional(problem)
    for _ in range(3):
        assert regional.profit([0.2, 0.25]) == pytest.approx(0.1375)
        np.testing.assert_allclose(regional.profit_gradient([0.2, 0.25]), [1, -0.5])
    assert calls == {"value": 1, "gradient": 1}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (lambda e: np.zeros_like(e), "shape"),
        (lambda e: np.full(len(e), np.nan), "finite and nonnegative"),
        (lambda e: -np.ones(len(e)), "finite and nonnegative"),
        (lambda e: np.ones(len(e)), r"K_b\(0\) = 0"),
    ],
)
def test_rejects_invalid_cost_values(value, message):
    with pytest.raises(ValueError, match=message):
        _problem(CallableProvisionCost(1, value))


@pytest.mark.parametrize(
    ("gradient", "message"),
    [(lambda e: np.zeros(len(e)), "shape"), (lambda e: np.full_like(e, np.inf), "finite")],
)
def test_rejects_invalid_gradients(gradient, message):
    problem = _problem(CallableProvisionCost(1, lambda e: np.sum(e**2, axis=1), gradient))
    with pytest.raises(ValueError, match=message):
        problem.provision_cost_gradients(_menu())


def test_rejects_ambiguous_cost_definition():
    with pytest.raises(ValueError, match="exactly one"):
        BundleDesignProblem(np.ones(1), np.ones(1), provision_cost=_quadratic())
    with pytest.raises(ValueError, match="exactly one"):
        BundleDesignProblem(np.ones(1))
