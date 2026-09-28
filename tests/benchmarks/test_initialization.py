"""Economic starting conditions and rejection sampling safeguards."""

import numpy as np
import pytest

from maas_bundle.benchmarks.initialization import sample_profitable_menus
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.primal import solve_primal
from maas_bundle.examples.two_route import (
    build_two_route_problem,
    generate_two_route_initial_menus,
    initial_two_route_menu,
)


def _bad_menu(price: float) -> BundleMenu:
    return BundleMenu(
        prices=np.array([0.0, price, price]),
        allowances=np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 10.0]]),
    )


@pytest.mark.parametrize("price", [100.0, 0.0, 2.7])
def test_primal_rejects_payg_loss_and_zero_profit_starts(price) -> None:
    # 100: all PAYG; 0: subscribed at a loss; 2.7: subscriptions at zero margin.
    problem = build_two_route_problem()
    with pytest.raises(ValueError, match="positive optimistic profit"):
        solve_primal(problem, _bad_menu(price), None)


def test_sampler_resamples_bad_menus_and_returns_requested_count() -> None:
    problem = build_two_route_problem()
    good = initial_two_route_menu()
    proposals = iter([_bad_menu(100.0), _bad_menu(0.0), good, _bad_menu(2.7), good])
    menus = sample_profitable_menus(problem, lambda: next(proposals), 2, max_attempts=5)
    assert menus == (good, good)


def test_sampler_fails_explicitly_when_no_profitable_start_exists() -> None:
    problem = build_two_route_problem()
    with pytest.raises(ValueError, match="only 0 positive-profit menus in 3 proposals"):
        sample_profitable_menus(problem, lambda: _bad_menu(100.0), 2, max_attempts=3)


def test_two_route_seeded_generator_excludes_plateaus_and_losses() -> None:
    problem = build_two_route_problem()
    menus = generate_two_route_initial_menus(30, seed=20260901, problem=problem)
    assert len(menus) == 30
    for menu in menus:
        assignment = problem.evaluate_assignment(menu)
        assert assignment.total_profit > 1e-9
        assert np.sum(assignment.bundle_demand[1:]) > 0
