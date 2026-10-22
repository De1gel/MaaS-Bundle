"""Independent closed-form checks for the diagnostic envelope curvature."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from maas_bundle.oracle.demand_distribution import (
    TriangularDemandDistribution,
    UniformDemandDistribution,
)
from maas_bundle.oracle.route_value import RouteValueOracle

_PATH = Path(__file__).parents[1] / "scripts" / "inspect-returned-solutions.py"
_SPEC = importlib.util.spec_from_file_location("solution_inspection", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize(
    ("distribution", "allowance", "expected"),
    [
        (UniformDemandDistribution(0, 2), 0.5, 0.5),
        (UniformDemandDistribution(0, 2), 1.5, 0.5),
        (TriangularDemandDistribution(0, 1, 2), 0.5, 0.5),
        (TriangularDemandDistribution(0, 1, 2), 1.25, 0.75),
    ],
)
def test_stop_loss_curvature_matches_closed_form(distribution, allowance, expected):
    # G(epsilon)=E[(mu-epsilon)_+], so G'' is the demand density.
    oracle = RouteValueOracle(np.array([[0.0, 0.0], [1.0, 1.0]]))
    hessian, _ = _MODULE.envelope_curvature(oracle, np.array([allowance]), distribution)
    assert hessian[0, 0] == pytest.approx(expected, abs=1e-12)
    step = 1e-5
    high = oracle.expected_valuation([allowance + step], distribution).allowance_subgradient
    low = oracle.expected_valuation([allowance - step], distribution).allowance_subgradient
    assert ((high - low) / (2 * step))[0] == pytest.approx(expected, abs=1e-8)


def test_parallel_maximizing_faces_are_flagged():
    # Phi=max(0,mu-epsilon_1,mu-epsilon_2) is nonsmooth at epsilon_1=epsilon_2.
    oracle = RouteValueOracle(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
    _, flags = _MODULE.envelope_curvature(
        oracle, np.array([0.5, 0.5]), UniformDemandDistribution(0, 2)
    )
    assert flags["parallel_switch_segments_with_positive_probability"] > 0
