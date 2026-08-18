"""Tests for piecewise-linear route-value functions."""

import numpy as np
import pytest

from maas_bundle.oracle import build_piecewise_phi, sort_dual_vertices


@pytest.fixture
def sorted_vertices():
    return sort_dual_vertices(
        [
            [7.0, 5.0],
            [0.0, 5.0],
            [2.0, 0.0],
            [0.0, 0.0],
        ]
    )


def test_sorts_vertices_once_by_pi(sorted_vertices) -> None:
    assert sorted_vertices.vertex_count == 4
    np.testing.assert_allclose(sorted_vertices.slopes, [0.0, 0.0, 2.0, 7.0])
    np.testing.assert_allclose(
        sorted_vertices.omega,
        [[5.0], [0.0], [0.0], [5.0]],
    )
    np.testing.assert_array_equal(sorted_vertices.original_indices, [1, 3, 2, 0])


def test_builds_expected_upper_envelope(sorted_vertices) -> None:
    phi = build_piecewise_phi(sorted_vertices, allowance=[3.0])

    np.testing.assert_allclose(phi.breakpoints, [0.0, 3.0])
    np.testing.assert_allclose(phi.slopes, [2.0, 7.0])
    np.testing.assert_allclose(phi.intercepts, [0.0, -15.0])
    np.testing.assert_array_equal(phi.active_vertex_indices, [2, 0])
    np.testing.assert_allclose(phi.cumulative_integrals, [0.0, 9.0])
    np.testing.assert_allclose(phi([0.0, 1.0, 3.0, 5.0]), [0.0, 2.0, 6.0, 20.0])
    assert phi(5.0) == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (0.0, 5.0, 35.0),
        (1.0, 2.0, 3.0),
        (2.0, 4.0, 14.5),
        (3.0, 3.0, 0.0),
    ],
)
def test_integrates_exactly_across_segments(sorted_vertices, lower, upper, expected) -> None:
    phi = build_piecewise_phi(sorted_vertices, allowance=[3.0])

    assert phi.integral(lower, upper) == pytest.approx(expected)


def test_rebuilds_intercepts_without_resorting(sorted_vertices) -> None:
    phi = build_piecewise_phi(sorted_vertices, allowance=[1.0])

    np.testing.assert_allclose(phi.breakpoints, [0.0, 1.0])
    np.testing.assert_allclose(phi.intercepts, [0.0, -5.0])


def test_discards_lower_line_with_same_slope() -> None:
    vertices = sort_dual_vertices([[0.0, 0.0], [0.0, 5.0], [2.0, 0.0]])

    phi = build_piecewise_phi(vertices, allowance=[3.0])

    np.testing.assert_allclose(phi.breakpoints, [0.0])
    np.testing.assert_allclose(phi.slopes, [2.0])


def test_discards_line_below_upper_hull() -> None:
    vertices = sort_dual_vertices([[1.0, 0.0], [2.0, 10.0], [3.0, 0.0]])

    phi = build_piecewise_phi(vertices, allowance=[1.0])

    np.testing.assert_allclose(phi.breakpoints, [0.0])
    np.testing.assert_allclose(phi.slopes, [3.0])


@pytest.mark.parametrize(
    ("vertices", "message"),
    [
        ([1.0, 2.0], "vertices must be two-dimensional"),
        (np.empty((0, 2)), "vertices must contain at least one vertex"),
        (np.empty((1, 0)), "vertices must contain a pi coordinate"),
        ([[1.0, np.inf]], "vertices must contain only finite values"),
    ],
)
def test_sort_rejects_invalid_vertices(vertices, message) -> None:
    with pytest.raises(ValueError, match=message):
        sort_dual_vertices(vertices)


@pytest.mark.parametrize(
    ("allowance", "arguments", "message"),
    [
        ([1.0, 2.0], {}, r"allowance must have shape \(1,\)"),
        ([np.nan], {}, "allowance must contain only finite values"),
        ([-1.0], {}, "allowance must be nonnegative"),
        ([1.0], {"domain_start": np.inf}, "domain_start must be finite"),
        ([1.0], {"domain_start": -1.0}, "domain_start must be nonnegative"),
        ([1.0], {"tolerance": -1.0}, "tolerance must be nonnegative"),
    ],
)
def test_build_rejects_invalid_input(sorted_vertices, allowance, arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_piecewise_phi(sorted_vertices, allowance, **arguments)


@pytest.mark.parametrize("mu", [-1.0, np.nan, np.inf])
def test_evaluation_rejects_invalid_mu(sorted_vertices, mu) -> None:
    phi = build_piecewise_phi(sorted_vertices, allowance=[3.0])

    with pytest.raises(ValueError):
        phi(mu)


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (0.0, np.inf, "integration bounds must be finite"),
        (np.nan, 1.0, "integration bounds must be finite"),
        (-1.0, 1.0, "lower must be at least"),
        (2.0, 1.0, "upper must be greater than or equal to lower"),
    ],
)
def test_integral_rejects_invalid_bounds(sorted_vertices, lower, upper, message) -> None:
    phi = build_piecewise_phi(sorted_vertices, allowance=[3.0])

    with pytest.raises(ValueError, match=message):
        phi.integral(lower, upper)
