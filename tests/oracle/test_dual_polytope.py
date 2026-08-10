"""Tests for the route-choice dual feasible region."""

import numpy as np
import pytest

from maas_bundle.oracle import (
    DualPolytope,
    UnboundedPolytopeError,
    build_dual_polytope,
    enumerate_vertices,
)


@pytest.fixture
def example_polytope():
    return build_dual_polytope(
        path_cost=np.array([10.0, 2.0]),
        mode_usage=np.array([[0.0, 1.0]]),
        payg_tariff=np.array([5.0]),
    )


def test_builds_expected_halfspace_representation() -> None:
    polytope = build_dual_polytope(
        path_cost=np.array([10.0, 2.0]),
        mode_usage=np.array([[0.0, 1.0]]),
        payg_tariff=np.array([5.0]),
    )

    np.testing.assert_allclose(
        polytope.matrix,
        np.array(
            [
                [1.0, 0.0],  # pi <= 10
                [1.0, -1.0],  # pi - omega <= 2
                [0.0, 1.0],  # omega <= 5
                [0.0, -1.0],  # omega >= 0
                [-1.0, 0.0],  # pi >= 0
            ]
        ),
    )
    np.testing.assert_allclose(polytope.bounds, np.array([10.0, 2.0, 5.0, 0.0, 0.0]))

    assert polytope.contains([2.0, 0.0])
    assert polytope.contains([7.0, 5.0])
    assert not polytope.contains([8.0, 5.0])


def test_contains_accepts_small_numerical_violation(example_polytope) -> None:
    point = [7.0 + 5e-10, 5.0]

    assert example_polytope.contains(point)
    assert not example_polytope.contains(point, tolerance=0.0)


def test_contains_rejects_wrong_point_shape(example_polytope) -> None:
    with pytest.raises(ValueError, match=r"point must have shape \(2,\)"):
        example_polytope.contains([2.0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_contains_rejects_nonfinite_point(example_polytope, value) -> None:
    with pytest.raises(ValueError, match="point must contain only finite values"):
        example_polytope.contains([2.0, value])


def test_contains_rejects_negative_tolerance(example_polytope) -> None:
    with pytest.raises(ValueError, match="tolerance must be nonnegative"):
        example_polytope.contains([2.0, 0.0], tolerance=-1e-9)


@pytest.mark.parametrize("field", ["matrix", "bounds"])
def test_polytope_arrays_are_read_only(example_polytope, field) -> None:
    array = getattr(example_polytope, field)

    with pytest.raises(ValueError, match="read-only"):
        array.flat[0] = 0.0


@pytest.mark.parametrize(
    ("path_cost", "mode_usage", "payg_tariff", "message"),
    [
        ([1.0, 2.0], [[1.0]], [3.0], "one column per path_cost"),
        ([1.0], [[1.0], [2.0]], [3.0], "one row per payg_tariff"),
        ([-1.0], [[1.0]], [3.0], "path_cost must be nonnegative"),
        ([1.0], [[-1.0]], [3.0], "mode_usage must be nonnegative"),
        ([1.0], [[1.0]], [-3.0], "payg_tariff must be nonnegative"),
    ],
)
def test_rejects_invalid_input(path_cost, mode_usage, payg_tariff, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_dual_polytope(path_cost, mode_usage, payg_tariff)


@pytest.mark.parametrize(
    ("path_cost", "mode_usage", "payg_tariff", "message"),
    [
        ([[1.0]], [[1.0]], [3.0], "path_cost must be one-dimensional"),
        ([], [[]], [3.0], "path_cost must not be empty"),
        ([np.nan], [[1.0]], [3.0], "path_cost must contain only finite values"),
        ([1.0], [1.0], [3.0], "mode_usage must be two-dimensional"),
        ([1.0], [[np.inf]], [3.0], "mode_usage must contain only finite values"),
        ([1.0], [[1.0]], [[3.0]], "payg_tariff must be one-dimensional"),
        ([1.0], [[1.0]], [np.inf], "payg_tariff must contain only finite values"),
    ],
)
def test_rejects_invalid_shape_or_nonfinite_input(
    path_cost,
    mode_usage,
    payg_tariff,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_dual_polytope(path_cost, mode_usage, payg_tariff)


def test_enumerates_expected_vertices() -> None:
    polytope = build_dual_polytope(
        path_cost=[10.0, 2.0],
        mode_usage=[[0.0, 1.0]],
        payg_tariff=[5.0],
    )

    vertices = enumerate_vertices(polytope)

    np.testing.assert_allclose(
        vertices,
        np.array(
            [
                [0.0, 0.0],
                [0.0, 5.0],
                [2.0, 0.0],
                [7.0, 5.0],
            ]
        ),
    )
    assert all(polytope.contains(vertex) for vertex in vertices)


def test_deduplicates_vertex_with_multiple_active_sets() -> None:
    # The final inequality is also active at (1, 1), so several combinations
    # of active constraints produce the same vertex.
    polytope = DualPolytope(
        matrix=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, 1.0],
            ]
        ),
        bounds=np.array([1.0, 1.0, 0.0, 0.0, 2.0]),
    )

    vertices = enumerate_vertices(polytope)

    np.testing.assert_allclose(
        vertices,
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )


def test_rejects_unbounded_polytope() -> None:
    polytope = DualPolytope(
        matrix=np.array([[1.0, 0.0]]),
        bounds=np.array([1.0]),
    )

    with pytest.raises(UnboundedPolytopeError, match="cddlib returned rays"):
        enumerate_vertices(polytope)


def test_returns_empty_array_for_empty_polytope() -> None:
    # x <= 0 and x >= 1 cannot hold simultaneously.
    polytope = DualPolytope(
        matrix=np.array([[1.0], [-1.0]]),
        bounds=np.array([0.0, -1.0]),
    )

    vertices = enumerate_vertices(polytope)

    assert vertices.shape == (0, 1)
