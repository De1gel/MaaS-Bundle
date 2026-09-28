"""Opt-in full enumeration comparison on the bundled open-network example."""

import pytest

from maas_bundle.cli import _compare_two_route


@pytest.mark.enumeration_scale
def test_two_route_enumeration_finds_better_consistent_candidate() -> None:
    result = _compare_two_route()

    assert result["primal"]["region_iterations"] == 2
    assert result["enumeration"]["assignments"] == 81
    assert result["enumeration"]["best_assignment"] == [1, 1, 2, 2]
    assert result["enumeration"]["best_true_profit"] == pytest.approx(4.03589, abs=1e-5)
    assert result["enumerated_minus_primal_profit"] > 0
