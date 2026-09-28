"""Tests for problem-level valuation tables."""

import numpy as np
import pytest

from maas_bundle import ValuationTable


def test_valuation_table_exposes_dimensions_and_immutable_arrays() -> None:
    table = ValuationTable(
        route_values=np.ones((2, 3)),
        expected_omega=np.zeros((2, 3, 4)),
    )

    assert table.traveler_count == 2
    assert table.bundle_count == 3
    assert table.mode_count == 4
    assert not table.route_values.flags.writeable
    assert not table.expected_omega.flags.writeable


@pytest.mark.parametrize(
    ("route_values", "expected_omega", "message"),
    [
        ([1.0], np.zeros((1, 1, 1)), "route_values must be two-dimensional"),
        ([[1.0]], [[0.0]], "one vector per traveler and bundle"),
        ([[1.0]], np.zeros((2, 1, 1)), "one vector per traveler and bundle"),
        ([[np.nan]], np.zeros((1, 1, 1)), "only finite values"),
        ([[1.0]], [[[np.inf]]], "only finite values"),
        ([[1.0]], [[[-1.0]]], "expected_omega must be nonnegative"),
    ],
)
def test_valuation_table_rejects_invalid_input(route_values, expected_omega, message) -> None:
    with pytest.raises(ValueError, match=message):
        ValuationTable(
            route_values=np.asarray(route_values),
            expected_omega=np.asarray(expected_omega),
        )
