"""Independent checks of the price profiler and convexity bound construction."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

from maas_bundle.oracle.demand_distribution import (
    TriangularDemandDistribution,
    UniformDemandDistribution,
)
from maas_bundle.oracle.route_value import RouteValueOracle

_PATH = Path(__file__).parents[1] / "scripts/verify-selected-nlps.py"
_SPEC = importlib.util.spec_from_file_location("selected_nlp_verification", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_vectorized_price_profiler_matches_independent_highs():
    rng = np.random.default_rng(42)
    choices = [(0, 1, 0), (1, 2, 0), (2, 0, 1), (3, 0, 2), (4, 1, 2), (5, 2, 1)]
    g0 = rng.uniform(1, 15, 6)
    g1 = rng.uniform(0, 10, (6, 9))
    g2 = rng.uniform(0, 10, (6, 11))
    # Include a known feasible price pair for one grid entry.
    g0[:] = [10, 15, 3, 3, 10, 10]
    g1[:, 0] = [4, 5, 1, 1, 4, 7]
    g2[:, 0] = [1, 3, 0, 0, 2, 2]
    caps = np.array([12.0, 20.0])
    p1, p2, feasible = _MODULE.profile_two_prices(g0, g1, g2, choices, caps)
    assert feasible[0, 0]
    weights = np.array([0.25, 0.75])
    for row in range(g1.shape[1]):
        for col in range(g2.shape[1]):
            matrix, rhs = [], []
            for i, b, a in choices:
                line = np.zeros(2)
                if b:
                    line[b - 1] += 1
                if a:
                    line[a - 1] -= 1
                values = [g0[i], g1[i, row], g2[i, col]]
                matrix.append(line)
                rhs.append(values[a] - values[b])
            matrix.append([1, -1])
            rhs.append(0)
            lp = linprog(
                -weights, A_ub=matrix, b_ub=rhs, bounds=[(0, caps[0]), (0, caps[1])], method="highs"
            )
            assert feasible[row, col] == lp.success
            if lp.success:
                value = weights @ [p1[row, col], p2[row, col]]
                assert value == pytest.approx(-lp.fun, abs=1e-9)


@pytest.mark.parametrize(
    "distribution",
    [
        UniformDemandDistribution(1, 5),
        TriangularDemandDistribution(1, 2, 5),
    ],
)
def test_affine_lower_bound_and_corner_upper_bound(distribution):
    # Contains a parallel dominated face, hence an ambiguous zero-boundary
    # subgradient. Valid lower bounds must not require smoothness there.
    oracle = RouteValueOracle(
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 1, 0],
                [2, 1, 2],
                [4, 2, 1],
            ],
            dtype=float,
        )
    )
    corners = np.array([[0, 0], [0, 4], [6, 0], [6, 4]], dtype=float)
    bounds = [
        _MODULE.affine_lower_and_corner_upper(oracle, corner, distribution) for corner in corners
    ]
    for corner, (_, _, upper) in zip(corners, bounds, strict=True):
        assert oracle.expected_value(corner, distribution) <= upper + 1e-10
    rng = np.random.default_rng(8)
    for _ in range(50):
        weights = rng.dirichlet(np.ones(4))
        point = weights @ corners
        exact = oracle.expected_value(point, distribution)
        assert max(c + g @ point for c, g, _ in bounds) <= exact + 1e-9
        assert exact <= weights @ [b[2] for b in bounds] + 1e-9


def test_summary_excludes_infeasible_objective_gains_and_separates_full_region():
    original = {"profit": 10.0, "restricted_violation": 0.0, "full_region_violation": 0.0}
    restricted_only = {
        "profit": 11.0,
        "restricted_violation": 0.0,
        "full_region_violation": 0.1,
        "start": "restricted_only",
    }
    infeasible = {
        "profit": 20.0,
        "restricted_violation": 0.01,
        "full_region_violation": 0.01,
        "start": "infeasible",
    }
    row = _MODULE.summarize_run(
        {
            "call": 1,
            "region": 1,
            "cg_round": 1,
            "is_region_endpoint": False,
            "retained_choice_constraints": 2,
            "original": original,
            "price_lp": original,
            "grids": [],
            "restarts": [restricted_only, infeasible],
            "box_bounds": [],
        }
    )
    assert row["best_gain"] == pytest.approx(1.0)
    assert row["best_full_gain"] == 0.0
    assert row["best_source"] == "restart_restricted_only"
    assert row["restart_returns_infeasible"] == 1
