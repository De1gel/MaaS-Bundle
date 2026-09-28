"""Bounded rejection sampling of profitable experimental starting menus."""

from collections.abc import Callable

import numpy as np

from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.problem import BundleDesignProblem


def sample_profitable_menus(
    problem: BundleDesignProblem,
    draw_menu: Callable[[], BundleMenu],
    start_count: int,
    *,
    minimum_profit: float = 1e-9,
    max_attempts: int | None = None,
) -> tuple[BundleMenu, ...]:
    """Keep positive-profit starts; reject PAYG plateaus and loss-making menus.

    The requested count is the number of accepted menus, not proposals. The
    attempt limit makes unsuitable sampling ranges fail explicitly instead of
    hanging or silently filling the experiment with zero-profit starts.
    """

    if not isinstance(start_count, (int, np.integer)) or start_count <= 0:
        raise ValueError("start_count must be a positive integer")
    if not np.isfinite(minimum_profit) or minimum_profit < 0:
        raise ValueError("minimum_profit must be finite and nonnegative")
    if max_attempts is None:
        max_attempts = 100 * int(start_count)
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")

    menus = []
    for _ in range(int(max_attempts)):
        menu = draw_menu()
        if np.any(np.diff(menu.prices) < 0):
            raise ValueError("initial menu prices must be nondecreasing")
        assignment = problem.evaluate_assignment(menu)
        if assignment.total_profit <= minimum_profit:
            continue
        menus.append(menu)
        if len(menus) == start_count:
            return tuple(menus)
    raise ValueError(
        f"found only {len(menus)} positive-profit menus in {max_attempts} proposals; "
        "adjust the sampling ranges or increase max_attempts"
    )
