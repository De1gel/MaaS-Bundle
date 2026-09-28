"""Opt-in multi-start experiment on the bundled two-route example."""

import pytest

from maas_bundle.examples.two_route import run_two_route_multistart


@pytest.mark.multistart_scale
def test_two_route_multistart_exposes_multiple_final_regions() -> None:
    result = run_two_route_multistart(30, seed=20260901)

    assert len(result.consistent_runs) == 30
    assert result.distinct_region_count == 14
    assert result.best_run is not None
    assert result.best_run.final_assignment == (1, 1, 2, 2)
    assert result.best_run.true_profit == pytest.approx(4.03589, abs=1e-5)

    for run in result.runs:
        initial_profit = run.diagnostics.candidate.initial_assignment.total_profit
        assert run.diagnostics.candidate.boundary_search_complete
        assert initial_profit > 0
        assert any(run.initial_assignment)
        assert any(run.final_assignment)
        assert run.true_profit >= initial_profit
        profits = [initial_profit] + [
            step.true_profit for step in run.diagnostics.candidate.iterations if step.accepted
        ]
        assert all(after >= before for before, after in zip(profits[:-1], profits[1:], strict=True))
