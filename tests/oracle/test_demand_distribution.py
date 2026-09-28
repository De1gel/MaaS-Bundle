"""Tests for trip-frequency distributions."""

import numpy as np
import pytest

from maas_bundle.oracle import (
    DiscreteDemandDistribution,
    TriangularDemandDistribution,
    UniformDemandDistribution,
)


def test_uniform_distribution_returns_exact_interval_statistics() -> None:
    distribution = UniformDemandDistribution(lower=2.0, upper=6.0)

    assert distribution.probability(-np.inf, np.inf) == pytest.approx(1.0)
    assert distribution.probability(3.0, 5.0) == pytest.approx(0.5)
    assert distribution.probability(0.0, 1.0) == pytest.approx(0.0)
    assert distribution.first_moment(-np.inf, np.inf) == pytest.approx(4.0)
    assert distribution.first_moment(3.0, 5.0) == pytest.approx(2.0)
    assert distribution.first_moment(0.0, 1.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (np.nan, 1.0, "support bounds must be finite"),
        (0.0, np.inf, "support bounds must be finite"),
        (-1.0, 1.0, "lower support must be nonnegative"),
        (1.0, 1.0, "upper support must be greater"),
        (2.0, 1.0, "upper support must be greater"),
    ],
)
def test_uniform_distribution_rejects_invalid_support(lower, upper, message) -> None:
    with pytest.raises(ValueError, match=message):
        UniformDemandDistribution(lower=lower, upper=upper)


@pytest.mark.parametrize(
    ("method", "lower", "upper", "message"),
    [
        ("probability", np.nan, 1.0, "must not be NaN"),
        ("first_moment", 0.0, np.nan, "must not be NaN"),
        ("probability", 2.0, 1.0, "upper bound must be at least"),
        ("first_moment", 2.0, 1.0, "upper bound must be at least"),
    ],
)
def test_uniform_distribution_rejects_invalid_interval(
    method,
    lower,
    upper,
    message,
) -> None:
    distribution = UniformDemandDistribution(lower=0.0, upper=2.0)

    with pytest.raises(ValueError, match=message):
        getattr(distribution, method)(lower, upper)


def test_discrete_distribution_returns_exact_half_open_interval_statistics() -> None:
    distribution = DiscreteDemandDistribution(
        support=np.array([4.0, 1.0, 2.0]),
        probabilities=np.array([0.5, 0.2, 0.3]),
    )

    assert distribution.support.tolist() == [1.0, 2.0, 4.0]
    assert distribution.probability(-np.inf, np.inf) == pytest.approx(1.0)
    assert distribution.probability(2.0, 4.0) == pytest.approx(0.3)
    assert distribution.first_moment(-np.inf, np.inf) == pytest.approx(2.8)
    assert distribution.first_moment(2.0, 4.0) == pytest.approx(0.6)
    assert not distribution.support.flags.writeable
    assert not distribution.probabilities.flags.writeable


@pytest.mark.parametrize(
    ("support", "probabilities", "message"),
    [
        ([], [], "support must be a nonempty vector"),
        ([[1.0]], [1.0], "support must be a nonempty vector"),
        ([1.0, 2.0], [1.0], "one entry per support"),
        ([np.inf], [1.0], "support must be finite"),
        ([-1.0], [1.0], "support must be finite"),
        ([1.0], [np.nan], "probabilities must be finite"),
        ([1.0], [-1.0], "probabilities must be finite"),
        ([1.0, 2.0], [0.2, 0.2], "sum to one"),
    ],
)
def test_discrete_distribution_rejects_invalid_data(support, probabilities, message) -> None:
    with pytest.raises(ValueError, match=message):
        DiscreteDemandDistribution(
            support=np.asarray(support),
            probabilities=np.asarray(probabilities),
        )


def test_discrete_distribution_uses_common_interval_validation() -> None:
    distribution = DiscreteDemandDistribution(
        support=np.array([1.0]),
        probabilities=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="upper bound must be at least"):
        distribution.probability(2.0, 1.0)


def test_triangular_distribution_returns_exact_interval_statistics() -> None:
    distribution = TriangularDemandDistribution(lower=0.0, mode=1.0, upper=2.0)

    assert distribution.probability(-np.inf, np.inf) == pytest.approx(1.0)
    assert distribution.probability(0.0, 1.0) == pytest.approx(0.5)
    assert distribution.probability(3.0, 4.0) == pytest.approx(0.0)
    assert distribution.first_moment(-np.inf, np.inf) == pytest.approx(1.0)
    assert distribution.first_moment(0.0, 1.0) == pytest.approx(1.0 / 3.0)
    assert distribution.first_moment(3.0, 4.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("lower", "mode", "upper", "message"),
    [
        (np.nan, 1.0, 2.0, "parameters must be finite"),
        (0.0, np.inf, 2.0, "parameters must be finite"),
        (-1.0, 0.0, 1.0, "lower support must be nonnegative"),
        (1.0, 1.5, 1.0, "upper support must be greater"),
        (0.0, 0.0, 2.0, "mode must lie strictly inside"),
        (0.0, 2.0, 2.0, "mode must lie strictly inside"),
    ],
)
def test_triangular_distribution_rejects_invalid_parameters(
    lower,
    mode,
    upper,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        TriangularDemandDistribution(lower=lower, mode=mode, upper=upper)


def test_triangular_distribution_uses_common_interval_validation() -> None:
    distribution = TriangularDemandDistribution(lower=0.0, mode=1.0, upper=2.0)

    with pytest.raises(ValueError, match="must not be NaN"):
        distribution.first_moment(np.nan, 1.0)
