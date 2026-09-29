"""Static data and evaluations for one MaaS bundle-design instance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
from maas_bundle.oracle.route_value import RouteValueOracle


@dataclass(frozen=True, init=False)
class BundleDesignProblem:
    """Immutable traveler, cost, distribution, and route-oracle data.

    Rows of ``route_oracles`` correspond to traveler classes and columns to
    bundle options. Bundle index zero is the fixed PAYG outside option.
    """

    traveler_weights: NDArray[np.float64]
    provision_cost: ProvisionCost
    route_oracles: Sequence[Sequence[RouteValueOracle]]
    demand_distributions: Sequence[TripFrequencyDistribution]

    def __init__(
        self,
        traveler_weights: NDArray[np.float64],
        unit_allowance_cost: NDArray[np.float64] | None = None,
        route_oracles: Sequence[Sequence[RouteValueOracle]] = (),
        demand_distributions: Sequence[TripFrequencyDistribution] = (),
        *,
        provision_cost: ProvisionCost | None = None,
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

        self._validate_menu(menu)
        route_values = np.empty((self.traveler_count, self.bundle_count), dtype=float)
        expected_omega = np.empty(
            (self.traveler_count, self.bundle_count, self.mode_count),
            dtype=float,
        )
        for traveler, (distribution, oracle_row) in enumerate(
            zip(self.demand_distributions, self.route_oracles, strict=True)
        ):
            for bundle, oracle in enumerate(oracle_row):
                valuation = oracle.expected_valuation(menu.allowances[bundle], distribution)
                route_values[traveler, bundle] = valuation.value
                expected_omega[traveler, bundle] = valuation.expected_omega
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
