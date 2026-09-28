"""End-to-end tests for the high-level route-value oracle."""

import numpy as np
import pytest

from maas_bundle import (
    RouteValuation,
    RouteValueOracle,
    UniformDemandDistribution,
    build_route_value_oracle,
)


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
    np.testing.assert_allclose(
        oracle.vertices,
        [
            [0.0, 0.0],
            [0.0, 5.0],
            [2.0, 0.0],
            [7.0, 5.0],
        ],
    )
    np.testing.assert_allclose(oracle.slopes, [0.0, 0.0, 2.0, 7.0])
    np.testing.assert_allclose(oracle.omega, [[0.0], [5.0], [0.0], [5.0]])
    assert not oracle.vertices.flags.writeable


def test_direct_construction_enforces_sorted_read_only_vertices() -> None:
    oracle = RouteValueOracle(vertices=np.array([[2.0, 1.0], [1.0, 0.0]]))

    np.testing.assert_allclose(oracle.vertices, [[1.0, 0.0], [2.0, 1.0]])
    assert not oracle.vertices.flags.writeable


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
    cached_vertices = oracle.vertices

    first = oracle.for_allowance([1.0])
    second = oracle.for_allowance([3.0])

    assert oracle.vertices is cached_vertices
    np.testing.assert_allclose(first.breakpoints, [0.0, 1.0])
    np.testing.assert_allclose(second.breakpoints, [0.0, 3.0])


def test_computes_expected_value_and_subgradient_across_segments(oracle) -> None:
    distribution = UniformDemandDistribution(lower=0.0, upper=5.0)

    valuation = oracle.expected_valuation([3.0], distribution)

    assert valuation.value == pytest.approx(7.0)
    np.testing.assert_allclose(valuation.expected_omega, [2.0])
    np.testing.assert_allclose(valuation.allowance_subgradient, [-2.0])
    assert not valuation.expected_omega.flags.writeable
    assert not valuation.allowance_subgradient.flags.writeable


def test_computes_expectation_when_distribution_occupies_one_segment(oracle) -> None:
    distribution = UniformDemandDistribution(lower=0.0, upper=1.0)

    valuation = oracle.expected_valuation([3.0], distribution)

    assert valuation.value == pytest.approx(1.0)
    np.testing.assert_allclose(valuation.expected_omega, [0.0])


def test_expected_value_convenience_method(oracle) -> None:
    distribution = UniformDemandDistribution(lower=0.0, upper=5.0)

    assert oracle.expected_value([3.0], distribution) == pytest.approx(7.0)


def test_expected_valuation_supports_multiple_allowance_dimensions() -> None:
    oracle = RouteValueOracle(vertices=np.array([[1.0, 0.0, 0.0], [3.0, 2.0, 4.0]]))
    distribution = UniformDemandDistribution(lower=0.0, upper=4.0)

    valuation = oracle.expected_valuation([1.0, 0.5], distribution)

    assert valuation.value == pytest.approx(3.0)
    np.testing.assert_allclose(valuation.expected_omega, [1.0, 2.0])
    np.testing.assert_allclose(valuation.allowance_subgradient, [-1.0, -2.0])


@pytest.mark.parametrize(
    ("value", "expected_omega", "message"),
    [
        (np.inf, [1.0], "expected route value must be finite"),
        (1.0, [[1.0]], "expected_omega must be one-dimensional"),
        (1.0, [np.nan], "expected_omega must contain only finite values"),
        (1.0, [-1.0], "expected_omega must be nonnegative"),
    ],
)
def test_route_valuation_rejects_invalid_result(value, expected_omega, message) -> None:
    with pytest.raises(ValueError, match=message):
        RouteValuation(value=value, expected_omega=np.asarray(expected_omega))


class _InvalidDistribution:
    def __init__(self, probability: float, first_moment: float) -> None:
        self._probability = probability
        self._first_moment = first_moment

    def probability(self, lower: float, upper: float) -> float:
        return self._probability

    def first_moment(self, lower: float, upper: float) -> float:
        return self._first_moment


@pytest.mark.parametrize(
    ("distribution", "message"),
    [
        (_InvalidDistribution(-1.0, 0.0), "invalid interval probability"),
        (_InvalidDistribution(2.0, 0.0), "invalid interval probability"),
        (_InvalidDistribution(np.nan, 0.0), "invalid interval probability"),
        (_InvalidDistribution(1.0, -1.0), "invalid interval first moment"),
        (_InvalidDistribution(1.0, np.inf), "invalid interval first moment"),
    ],
)
def test_expected_valuation_rejects_invalid_distribution_output(
    oracle,
    distribution,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        oracle.expected_valuation([3.0], distribution)


def test_expected_valuation_requires_full_probability_on_domain(oracle) -> None:
    distribution = _InvalidDistribution(probability=0.2, first_moment=0.0)

    with pytest.raises(ValueError, match="total probability one"):
        oracle.expected_valuation([3.0], distribution)


def test_expected_valuation_rejects_negative_probability_tolerance(oracle) -> None:
    distribution = UniformDemandDistribution(lower=0.0, upper=5.0)

    with pytest.raises(ValueError, match="probability_tolerance must be nonnegative"):
        oracle.expected_valuation([3.0], distribution, probability_tolerance=-1.0)
