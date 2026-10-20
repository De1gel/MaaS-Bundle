"""Static data and evaluations for one MaaS bundle-design instance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from maas_bundle.design.assignment import (
    AssignmentResult,
    aggregate_bundle_demand,
    select_optimistic_bundles,
)
from maas_bundle.design.cost import (
    LinearProvisionCost,
    ProvisionCost,
    checked_cost_gradients,
    checked_cost_values,
)
from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.tolerances import (
    DEFAULT_COST_TOLERANCE,
    DEFAULT_MARGIN_TOLERANCE,
)
from maas_bundle.design.valuation import ValuationTable
from maas_bundle.oracle.demand_distribution import TripFrequencyDistribution
from maas_bundle.oracle.route_value import RouteValuation, RouteValueOracle


@dataclass(frozen=True, init=False)
class BundleDesignProblem:
    """Immutable traveler, cost, distribution, and route-oracle data.

    Rows of ``route_oracles`` correspond to traveler classes and columns to
    bundle options. Bundle index zero is the fixed PAYG outside option.
    Optional upper bounds cover subscriptions only; scalars broadcast to all
    prices or allowances. Defaults preserve the unbounded design domain.
    """

    traveler_weights: NDArray[np.float64]
    provision_cost: ProvisionCost
    route_oracles: Sequence[Sequence[RouteValueOracle]]
    demand_distributions: Sequence[TripFrequencyDistribution]
    price_upper_bounds: NDArray[np.float64]
    allowance_upper_bounds: NDArray[np.float64]
    _valuation_cache: dict[tuple[int, int], tuple[bytes, RouteValuation]] = field(
        init=False, repr=False, compare=False
    )

    def __init__(
        self,
        traveler_weights: NDArray[np.float64],
        unit_allowance_cost: NDArray[np.float64] | None = None,
        route_oracles: Sequence[Sequence[RouteValueOracle]] = (),
        demand_distributions: Sequence[TripFrequencyDistribution] = (),
        *,
        provision_cost: ProvisionCost | None = None,
        price_upper_bounds: ArrayLike = np.inf,
        allowance_upper_bounds: ArrayLike = np.inf,
    ) -> None:
        """Accept a general cost model or the legacy linear coefficient vector."""

        if (unit_allowance_cost is None) == (provision_cost is None):
            raise ValueError("provide exactly one of provision_cost and unit_allowance_cost")
        if provision_cost is None:
            provision_cost = LinearProvisionCost(unit_allowance_cost)
        object.__setattr__(self, "traveler_weights", traveler_weights)
        object.__setattr__(self, "provision_cost", provision_cost)
        object.__setattr__(self, "route_oracles", route_oracles)
        object.__setattr__(self, "demand_distributions", demand_distributions)
        object.__setattr__(self, "price_upper_bounds", price_upper_bounds)
        object.__setattr__(self, "allowance_upper_bounds", allowance_upper_bounds)
        # At most one exact allowance entry per traveler/bundle. PAYG's entry
        # remains valid across menus because its allowance is fixed at zero.
        object.__setattr__(self, "_valuation_cache", {})
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate all static dimensions once, before iterative optimization."""

        weights = np.asarray(self.traveler_weights, dtype=float)
        cost = self.provision_cost
        oracle_rows = tuple(tuple(row) for row in self.route_oracles)
        distributions = tuple(self.demand_distributions)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("traveler_weights must be a nonempty vector")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("traveler_weights must be finite and nonnegative")
        if not isinstance(cost.mode_count, (int, np.integer)) or cost.mode_count <= 0:
            raise ValueError("provision cost mode_count must be a positive integer")
        if not callable(cost.values) or not callable(cost.gradients):
            raise TypeError("provision cost must provide values and gradients methods")
        if len(distributions) != weights.size:
            raise ValueError("demand_distributions must have one entry per traveler")
        if len(oracle_rows) != weights.size:
            raise ValueError("route_oracles must have one row per traveler")
        if not oracle_rows or not oracle_rows[0]:
            raise ValueError("route_oracles must contain at least one bundle")
        bundle_count = len(oracle_rows[0])
        if any(len(row) != bundle_count for row in oracle_rows):
            raise ValueError("route_oracles must have the same bundles in every row")
        if any(oracle.mode_count != cost.mode_count for row in oracle_rows for oracle in row):
            raise ValueError("every route oracle must use the problem allowance dimensions")
        zero_allowances = np.zeros((bundle_count - 1, cost.mode_count))
        zero_allowances.setflags(write=False)
        if np.any(checked_cost_values(cost, zero_allowances) != 0):
            raise ValueError("provision cost must satisfy K_b(0) = 0")

        immutable_weights = weights.copy()
        immutable_weights.setflags(write=False)
        object.__setattr__(self, "traveler_weights", immutable_weights)
        object.__setattr__(self, "route_oracles", oracle_rows)
        object.__setattr__(self, "demand_distributions", distributions)
        for name, shape in (
            ("price_upper_bounds", (bundle_count - 1,)),
            ("allowance_upper_bounds", (bundle_count - 1, cost.mode_count)),
        ):
            bounds = np.broadcast_to(np.asarray(getattr(self, name), dtype=float), shape).copy()
            if np.any(np.isnan(bounds)) or np.any(bounds < 0):
                raise ValueError(f"{name} must be nonnegative, with no NaN entries")
            bounds.setflags(write=False)
            object.__setattr__(self, name, bounds)

    @property
    def unit_allowance_cost(self) -> NDArray[np.float64] | None:
        """Legacy linear coefficients, or None for a general cost model."""

        if isinstance(self.provision_cost, LinearProvisionCost):
            return self.provision_cost.unit_cost
        return None

    @property
    def traveler_count(self) -> int:
        """Return the number of traveler classes."""

        return self.traveler_weights.size

    @property
    def bundle_count(self) -> int:
        """Return the number of options, including PAYG."""

        return len(self.route_oracles[0])

    @property
    def mode_count(self) -> int:
        """Return the number of allowance dimensions."""

        return self.provision_cost.mode_count

    def bundle_margins(self, menu: BundleMenu) -> NDArray[np.float64]:
        """Evaluate C_b - K_b, with fixed zero margin for PAYG."""

        self._validate_menu(menu)
        margins = menu.prices.copy()
        margins[1:] -= checked_cost_values(self.provision_cost, menu.allowances[1:])
        margins.setflags(write=False)
        return margins

    def provision_cost_gradients(self, menu: BundleMenu) -> NDArray[np.float64]:
        """Evaluate subscription cost derivatives, excluding PAYG."""

        self._validate_menu(menu)
        gradients = checked_cost_gradients(self.provision_cost, menu.allowances[1:]).copy()
        gradients.setflags(write=False)
        return gradients

    def evaluate_valuations(self, menu: BundleMenu) -> ValuationTable:
        """Evaluate all ``G_ib`` and expected shadow prices at a menu."""

        return self._evaluate_selected_valuations(menu, range(self.traveler_count))

    def evaluate_valuations_for_travelers(
        self, menu: BundleMenu, travelers: Sequence[int]
    ) -> ValuationTable:
        """Return compact rows in the requested order, including all bundle columns.

        Custom problem adapters that override the full evaluator retain their
        own semantics: select rows from their implementation instead.
        """

        indices = np.asarray(travelers)
        if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError("travelers must be a one-dimensional integer sequence")
        indices = indices.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= self.traveler_count):
            raise ValueError("traveler index out of range")
        if type(self).evaluate_valuations is not BundleDesignProblem.evaluate_valuations:
            full = self.evaluate_valuations(menu)
            return ValuationTable(full.route_values[indices], full.expected_omega[indices])
        return self._evaluate_selected_valuations(menu, indices)

    def _evaluate_selected_valuations(self, menu: BundleMenu, travelers) -> ValuationTable:
        self._validate_menu(menu)
        route_values = np.empty((len(travelers), self.bundle_count), dtype=float)
        expected_omega = np.empty(
            (len(travelers), self.bundle_count, self.mode_count),
            dtype=float,
        )
        cache: dict[tuple[int, int], tuple[bytes, RouteValuation]] = self._valuation_cache
        for bundle, allowance in enumerate(menu.allowances):
            key = allowance.tobytes()
            for row, traveler in enumerate(travelers):
                pair = (int(traveler), bundle)
                cached = cache.get(pair)
                if cached is None or cached[0] != key:
                    oracle = self.route_oracles[traveler][bundle]
                    valuation = oracle.expected_valuation(
                        allowance, self.demand_distributions[traveler]
                    )
                    cache[pair] = (key, valuation)
                else:
                    valuation = cached[1]
                route_values[row, bundle] = valuation.value
                expected_omega[row, bundle] = valuation.expected_omega
        return ValuationTable(route_values=route_values, expected_omega=expected_omega)

    def evaluate_assignment(
        self,
        menu: BundleMenu,
        *,
        choice_tolerance: float = DEFAULT_COST_TOLERANCE,
        profit_tolerance: float = DEFAULT_MARGIN_TOLERANCE,
    ) -> AssignmentResult:
        """Return the numerical optimistic response and true profit at a menu.

        ``profit_tolerance`` is the legacy name for the per-bundle margin
        tolerance here, not an aggregate-objective improvement threshold.
        """

        valuation = self.evaluate_valuations(menu)
        generalized_costs = valuation.route_values + menu.prices[None, :]
        bundle_margins = self.bundle_margins(menu)
        bundle_indices = select_optimistic_bundles(
            generalized_costs,
            bundle_margins,
            choice_tolerance=choice_tolerance,
            profit_tolerance=profit_tolerance,
        )
        bundle_demand = aggregate_bundle_demand(
            bundle_indices,
            self.traveler_weights,
            self.bundle_count,
        )
        total_profit = float(bundle_demand @ bundle_margins)
        return AssignmentResult(
            menu=menu,
            valuation=valuation,
            bundle_indices=bundle_indices,
            generalized_costs=generalized_costs,
            bundle_margins=bundle_margins,
            bundle_demand=bundle_demand,
            total_profit=total_profit,
            choice_tolerance=choice_tolerance,
            profit_tolerance=profit_tolerance,
        )

    def _validate_menu(self, menu: BundleMenu) -> None:
        if menu.bundle_count != self.bundle_count:
            raise ValueError("menu must have one entry per problem bundle")
        if menu.mode_count != self.mode_count:
            raise ValueError("menu must use the problem allowance dimensions")
        if menu.prices[0] != 0.0 or np.any(menu.allowances[0] != 0.0):
            raise ValueError("bundle zero must be the zero-price, zero-allowance PAYG option")
