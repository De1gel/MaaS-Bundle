"""Check full-domain bound logic against an analytically solvable NLP."""

import importlib.util
from pathlib import Path

import gurobipy as gp
import numpy as np
import pytest

from maas_bundle.design.boundary import ChoiceConstraintIndex, CriticalConstraintSet
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.problem import BundleDesignProblem
from maas_bundle.design.regional import RestrictedRegionalProblem
from maas_bundle.oracle.demand_distribution import UniformDemandDistribution
from maas_bundle.oracle.route_value import RouteValueOracle

_SPEC = importlib.util.spec_from_file_location(
    "chain_global", Path(__file__).parents[1] / "scripts/benchmark-chain-global.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.fixture
def env():
    with gp.Env(empty=True) as environment:
        environment.setParam("OutputFlag", 0)
        environment.start()
        yield environment


def quadratic_problem(*, baseline=False):
    # G(e)=E[max(mu-e,0)]=(2-e)^2/4 for e in [0,2], mu~U[0,2].
    # Profit max is attained at both allowances=1.6, both prices=.96:
    # max_e 1-G(e)-.2e = .64. Larger e>=2 only increases provision cost.
    vertices = [[0, 0], [2, 0], [3, 1]] if baseline else [[0, 0], [1, 1]]
    oracle = RouteValueOracle(np.array(vertices, dtype=float))
    p = BundleDesignProblem(
        traveler_weights=np.array([0.5, 0.5]),
        unit_allowance_cost=np.array([0.2]),
        route_oracles=((oracle,) * 3,) * 2,
        demand_distributions=(UniformDemandDistribution(0, 2),) * 2,
    )
    critical = CriticalConstraintSet(np.array([1, 2]), 3)
    critical.add_all(
        tuple(
            ChoiceConstraintIndex(i, b, a) for i, b in enumerate([1, 2]) for a in range(3) if a != b
        )
    )
    return RestrictedRegionalProblem(p, critical)


def test_global_gap_closes_around_known_full_domain_optimum(env):
    regional = quadratic_problem()
    seed = BundleMenu(np.array([0, 0.75, 0.75]), np.array([[0], [1], [1]], dtype=float))
    search = _MODULE.SpatialSearch(regional, env, [seed], seconds=10, gap=1e-4)
    result = search.run()
    assert result["certified_to_requested_gap"]
    assert result["lower_bound"] <= 0.64 + 1e-7
    assert result["upper_bound"] >= 0.64 - 1e-7
    assert result["absolute_gap"] <= 1e-4
    assert result["best_retained_violation"] <= 1e-7


def test_time_limit_preserves_unsearched_domain_upper_bound(env):
    regional = quadratic_problem()
    seed = BundleMenu(np.array([0, 0.75, 0.75]), np.array([[0], [1], [1]], dtype=float))
    search = _MODULE.SpatialSearch(regional, env, [seed], seconds=0, gap=1e-4)
    result = search.run()
    assert result["status"] == "time_limit"
    assert not result["certified_to_requested_gap"]
    assert result["upper_bound"] >= 0.64
    assert result["open_nodes"] == 1


def test_lp_dual_residual_correction_is_valid_even_with_zero_duals(env):
    from types import SimpleNamespace

    with gp.Model(env=env) as model:
        x = model.addMVar(2, lb=0, ub=2)
        model.addConstr(x.sum() <= 1)
        model.setObjective(np.array([3.0, 2.0]) @ x, gp.GRB.MAXIMIZE)
        model.optimize()
        assert _MODULE.lp_dual_upper_bound(model) >= 3.0
        # Without residual correction these zero duals would incorrectly give
        # objective upper bound zero. The variable bounds safely absorb r'x.
        inexact = SimpleNamespace(
            getVars=model.getVars,
            getConstrs=model.getConstrs,
            getAttr=lambda attr, objects: (
                [0.0] * len(objects) if attr == "Pi" else model.getAttr(attr, objects)
            ),
            getA=model.getA,
            ObjCon=model.ObjCon,
            ObjVal=model.ObjVal,
        )
        assert _MODULE.lp_dual_upper_bound(inexact) >= 10.0


def test_positive_route_baseline_tightens_caps_without_losing_optimum(env):
    # Adding 2*E[mu] to all bundle valuations leaves every price difference
    # unchanged. The safe price cap must reduce from 3 to 1, with the same .64 optimum.
    regional = quadratic_problem(baseline=True)
    seed = BundleMenu(np.array([0, 0.75, 0.75]), np.array([[0], [1], [1]], dtype=float))
    search = _MODULE.SpatialSearch(regional, env, [seed], seconds=10, gap=1e-4)
    assert np.all(search.caps[1:] <= 1 + 1e-6)
    assert np.all(search.root_upper >= 1.6)
    result = search.run()
    assert result["certified_to_requested_gap"]
    assert result["lower_bound"] <= 0.64 + 1e-7 <= result["upper_bound"] + 1e-7
