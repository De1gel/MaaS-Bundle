"""Check that the SOSC screen does not incorrectly discard critical directions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "scripts" / "check-longest-trajectory-sosc.py"
_SPEC = importlib.util.spec_from_file_location("trajectory_sosc", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_weakly_active_constraint_is_not_forced_to_equality():
    # min -x^2 on x>=0: origin is KKT, but has feasible negative curvature.
    # Treating the weak constraint as an equality would incorrectly remove it.
    result = _MODULE.curvature_upper_space(
        np.array([[-2.0]]), np.array([[-1.0]]), np.array([0.0]), np.array([0.0])
    )
    assert result["containing_subspace_dimension"] == 1
    assert result["minimum_eigenvalue"] == pytest.approx(-2.0)
    assert not result["positive_on_containing_subspace"]


def test_normal_curvature_need_not_be_positive():
    # The positive-multiplier constraint fixes d_1=0 in the critical cone.
    result = _MODULE.curvature_upper_space(
        np.diag([-100.0, 3.0]), np.array([[1.0, 0.0]]), np.array([2.0]), np.zeros(2)
    )
    assert result["containing_subspace_dimension"] == 1
    assert result["minimum_eigenvalue"] == pytest.approx(3.0)
    assert result["positive_on_containing_subspace"]


def test_strong_lower_bound_can_make_critical_cone_trivial():
    # min x on x>=0 at the origin: every nonzero feasible direction increases f.
    result = _MODULE.curvature_upper_space(
        np.zeros((1, 1)), np.empty((0, 1)), np.array([]), np.array([1.0])
    )
    assert result["containing_subspace_dimension"] == 0
    assert result["positive_on_containing_subspace"]
