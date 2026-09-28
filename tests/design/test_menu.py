"""Tests for immutable bundle-menu snapshots."""

import numpy as np
import pytest

from maas_bundle import BundleMenu


def test_menu_exposes_dimensions_and_immutable_arrays() -> None:
    menu = BundleMenu(prices=np.array([0.0, 1.0]), allowances=np.zeros((2, 3)))

    assert menu.bundle_count == 2
    assert menu.mode_count == 3
    assert not menu.prices.flags.writeable
    assert not menu.allowances.flags.writeable


@pytest.mark.parametrize(
    ("prices", "allowances", "message"),
    [
        ([[1.0]], [[0.0]], "prices must be one-dimensional"),
        ([], np.empty((0, 1)), "at least one bundle"),
        ([1.0], [0.0], "allowances must be two-dimensional"),
        ([1.0, 2.0], [[0.0]], "one row per bundle"),
        ([np.inf], [[0.0]], "only finite values"),
        ([1.0], [[np.nan]], "only finite values"),
        ([-1.0], [[0.0]], "must be nonnegative"),
        ([1.0], [[-1.0]], "must be nonnegative"),
    ],
)
def test_menu_rejects_invalid_input(prices, allowances, message) -> None:
    with pytest.raises(ValueError, match=message):
        BundleMenu(prices=np.asarray(prices), allowances=np.asarray(allowances))
