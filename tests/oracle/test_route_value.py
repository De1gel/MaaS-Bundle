"""End-to-end tests for the high-level route-value oracle."""

import numpy as np
import pytest

from maas_bundle import build_route_value_oracle


@pytest.fixture
def oracle():
    return build_route_value_oracle(
        path_cost=[10.0, 2.0],
        mode_usage=[[0.0, 1.0]],
        payg_tariff=[5.0],
    )


def test_builds_and_preprocesses_complete_oracle(oracle) -> None:
    assert oracle.vertex_count == 4
    assert oracle.mode_count == 1
    assert oracle.polytope.dimension == 2
    np.testing.assert_allclose(
        oracle.vertices,
        [
            [0.0, 0.0],
            [0.0, 5.0],
            [2.0, 0.0],
            [7.0, 5.0],
        ],
    )


def test_returns_complete_function_for_each_allowance(oracle) -> None:
    phi = oracle.for_allowance([3.0])

    np.testing.assert_allclose(phi.breakpoints, [0.0, 3.0])
    np.testing.assert_allclose(phi.slopes, [2.0, 7.0])
    np.testing.assert_allclose(phi.intercepts, [0.0, -15.0])


def test_evaluates_value_through_convenience_interface(oracle) -> None:
    np.testing.assert_allclose(
        oracle.value([0.0, 1.0, 3.0, 5.0], allowance=[3.0]),
        [0.0, 2.0, 6.0, 20.0],
    )
    assert oracle.value(5.0, allowance=[3.0]) == pytest.approx(20.0)


def test_integrates_through_convenience_interface(oracle) -> None:
    assert oracle.integral(0.0, 5.0, allowance=[3.0]) == pytest.approx(35.0)


def test_reuses_sorted_vertices_across_allowances(oracle) -> None:
    cached_vertices = oracle.sorted_vertices

    first = oracle.for_allowance([1.0])
    second = oracle.for_allowance([3.0])

    assert oracle.sorted_vertices is cached_vertices
    np.testing.assert_allclose(first.breakpoints, [0.0, 1.0])
    np.testing.assert_allclose(second.breakpoints, [0.0, 3.0])
