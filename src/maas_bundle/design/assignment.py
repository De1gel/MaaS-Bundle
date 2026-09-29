"""Optimistic bundle selection from an evaluated MaaS menu."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.tolerances import (
    DEFAULT_COST_TOLERANCE,
    DEFAULT_MARGIN_TOLERANCE,
)
from maas_bundle.design.valuation import ValuationTable


@dataclass(frozen=True)
class AssignmentResult:
    """Immutable optimistic response and platform outcome at one menu."""

    menu: BundleMenu
    valuation: ValuationTable
    bundle_indices: NDArray[np.int64]
    generalized_costs: NDArray[np.float64]
    bundle_margins: NDArray[np.float64]
    bundle_demand: NDArray[np.float64]
    total_profit: float
    choice_tolerance: float = DEFAULT_COST_TOLERANCE
    profit_tolerance: float = DEFAULT_MARGIN_TOLERANCE
    minimum_cost_gaps: NDArray[np.float64] = field(init=False, repr=False)
    numerical_tie_mask: NDArray[np.bool_] = field(init=False, repr=False)
    numerical_optimistic_mask: NDArray[np.bool_] = field(init=False, repr=False)
    optimistic_margin_gaps: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate cross-object dimensions and protect result arrays."""

        bundle_indices = np.asarray(self.bundle_indices, dtype=np.int64)
        generalized_costs = np.asarray(self.generalized_costs, dtype=float)
        bundle_margins = np.asarray(self.bundle_margins, dtype=float)
        bundle_demand = np.asarray(self.bundle_demand, dtype=float)
        total_profit = float(self.total_profit)
        _validate_tolerance(self.choice_tolerance, name="choice_tolerance")
        _validate_tolerance(self.profit_tolerance, name="profit_tolerance")
        expected_shape = (self.valuation.traveler_count, self.valuation.bundle_count)

        if self.menu.bundle_count != self.valuation.bundle_count:
            raise ValueError("menu and valuation must use the same bundles")
        if self.menu.mode_count != self.valuation.mode_count:
            raise ValueError("menu and valuation must use the same allowance dimensions")
        if bundle_indices.shape != (self.valuation.traveler_count,):
            raise ValueError("bundle_indices must have one entry per traveler")
        if generalized_costs.shape != expected_shape:
            raise ValueError("generalized_costs must have one entry per traveler and bundle")
        if bundle_margins.shape != (self.menu.bundle_count,):
            raise ValueError("bundle_margins must have one entry per bundle")
        if bundle_demand.shape != (self.menu.bundle_count,):
            raise ValueError("bundle_demand must have one entry per bundle")
        if bundle_indices.size and (
            np.any(bundle_indices < 0) or np.any(bundle_indices >= self.menu.bundle_count)
        ):
            raise ValueError("bundle_indices contains an invalid bundle index")
        floating_arrays = (generalized_costs, bundle_margins, bundle_demand)
        if not all(np.all(np.isfinite(array)) for array in floating_arrays):
            raise ValueError("assignment result must contain only finite values")
        if np.any(bundle_demand < 0):
            raise ValueError("bundle_demand must be nonnegative")
        if not np.isfinite(total_profit):
            raise ValueError("total_profit must be finite")

        if self.valuation.traveler_count:
            minimum_cost_gaps, numerical_tie_mask, numerical_optimistic_mask = _choice_masks(
                generalized_costs, bundle_margins, self.choice_tolerance, self.profit_tolerance
            )
            rows = np.arange(self.valuation.traveler_count)
            if not np.all(numerical_tie_mask[rows, bundle_indices]):
                raise ValueError("bundle_indices must select a numerical minimum-cost bundle")

            if not np.all(numerical_optimistic_mask[rows, bundle_indices]):
                raise ValueError("bundle_indices must satisfy optimistic margin tie-breaking")
            optimistic_margin_gaps = np.full(self.valuation.traveler_count, np.inf)
            for traveler, selected in enumerate(bundle_indices):
                tied_competitors = numerical_tie_mask[traveler].copy()
                tied_competitors[selected] = False
                if np.any(tied_competitors):
                    best_competing_margin = np.max(bundle_margins[tied_competitors])
                    optimistic_margin_gaps[traveler] = (
                        bundle_margins[selected] - best_competing_margin
                    )
                    if optimistic_margin_gaps[traveler] < -self.profit_tolerance:
                        raise ValueError(
                            "bundle_indices must satisfy optimistic margin tie-breaking"
                        )
        else:
            minimum_cost_gaps = np.empty(expected_shape, dtype=float)
            numerical_tie_mask = np.empty(expected_shape, dtype=bool)
            numerical_optimistic_mask = np.empty(expected_shape, dtype=bool)
            optimistic_margin_gaps = np.empty(0, dtype=float)

        immutable_indices = bundle_indices.copy()
        immutable_costs = generalized_costs.copy()
        immutable_margins = bundle_margins.copy()
        immutable_demand = bundle_demand.copy()
        immutable_cost_gaps = minimum_cost_gaps.copy()
        immutable_tie_mask = numerical_tie_mask.copy()
        immutable_margin_gaps = optimistic_margin_gaps.copy()
        numerical_optimistic_mask.setflags(write=False)
        for array in (
            immutable_indices,
            immutable_costs,
            immutable_margins,
            immutable_demand,
            immutable_cost_gaps,
            immutable_tie_mask,
            immutable_margin_gaps,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "bundle_indices", immutable_indices)
        object.__setattr__(self, "generalized_costs", immutable_costs)
        object.__setattr__(self, "bundle_margins", immutable_margins)
        object.__setattr__(self, "bundle_demand", immutable_demand)
        object.__setattr__(self, "total_profit", total_profit)
        object.__setattr__(self, "choice_tolerance", float(self.choice_tolerance))
        object.__setattr__(self, "profit_tolerance", float(self.profit_tolerance))
        object.__setattr__(self, "minimum_cost_gaps", immutable_cost_gaps)
        object.__setattr__(self, "numerical_tie_mask", immutable_tie_mask)
        object.__setattr__(self, "numerical_optimistic_mask", numerical_optimistic_mask)
        object.__setattr__(self, "optimistic_margin_gaps", immutable_margin_gaps)

    @property
    def route_values(self) -> NDArray[np.float64]:
        """Expose the route-value table used for this assignment."""

        return self.valuation.route_values

    @property
    def expected_omega(self) -> NDArray[np.float64]:
        """Expose the expected shadow prices used for this assignment."""

        return self.valuation.expected_omega

    @property
    def has_numerical_cost_ties(self) -> NDArray[np.bool_]:
        """Return which travelers have multiple bundles in their numerical tie set."""

        result = np.count_nonzero(self.numerical_tie_mask, axis=1) > 1
        result.setflags(write=False)
        return result

    @property
    def margin_tolerance(self) -> float:
        """Per-bundle margin tolerance (legacy storage name: profit_tolerance)."""

        return self.profit_tolerance

    @property
    def has_double_ties(self) -> NDArray[np.bool_]:
        """Return tied travelers without a strict optimistic platform margin."""

        result = np.count_nonzero(self.numerical_optimistic_mask, axis=1) > 1
        result.setflags(write=False)
        return result


def select_optimistic_bundles(
    generalized_costs: ArrayLike,
    bundle_margins: ArrayLike,
    *,
    choice_tolerance: float = DEFAULT_COST_TOLERANCE,
    profit_tolerance: float = DEFAULT_MARGIN_TOLERANCE,
) -> NDArray[np.int64]:
    """Select the lowest-index numerical optimistic bundle per traveler.

    ``profit_tolerance`` is a per-bundle margin tolerance (legacy API name).
    """

    costs = np.asarray(generalized_costs, dtype=float)
    margins = np.asarray(bundle_margins, dtype=float)
    _validate_tolerance(choice_tolerance, name="choice_tolerance")
    _validate_tolerance(profit_tolerance, name="profit_tolerance")
    if costs.ndim != 2:
        raise ValueError("generalized_costs must be two-dimensional")
    if costs.shape[1] == 0:
        raise ValueError("generalized_costs must contain at least one bundle")
    if margins.shape != (costs.shape[1],):
        raise ValueError("bundle_margins must have one entry per bundle")
    if not np.all(np.isfinite(costs)) or not np.all(np.isfinite(margins)):
        raise ValueError("costs and margins must contain only finite values")

    _, _, optimistic = _choice_masks(costs, margins, choice_tolerance, profit_tolerance)
    assignment = np.argmax(optimistic, axis=1).astype(np.int64)

    assignment.setflags(write=False)
    return assignment


def aggregate_bundle_demand(
    bundle_indices: ArrayLike,
    traveler_weights: ArrayLike,
    bundle_count: int,
) -> NDArray[np.float64]:
    """Aggregate traveler-class weights by their assigned bundle."""

    raw_indices = np.asarray(bundle_indices)
    weights = np.asarray(traveler_weights, dtype=float)
    if raw_indices.ndim != 1:
        raise ValueError("bundle_indices must be one-dimensional")
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("bundle_indices must contain integers")
    if weights.shape != raw_indices.shape:
        raise ValueError("traveler_weights must match bundle_indices")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("traveler_weights must be finite and nonnegative")
    if not isinstance(bundle_count, (int, np.integer)) or bundle_count <= 0:
        raise ValueError("bundle_count must be a positive integer")

    indices = raw_indices.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= bundle_count):
        raise ValueError("bundle_indices contains an invalid bundle index")

    demand = np.bincount(indices, weights=weights, minlength=int(bundle_count)).astype(float)
    demand.setflags(write=False)
    return demand


def _validate_tolerance(value: float, *, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _choice_masks(
    costs: NDArray[np.float64],
    margins: NDArray[np.float64],
    choice_tolerance: float,
    margin_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Use minima/maxima as common references; approximate equality is not transitive."""

    gaps = costs - np.min(costs, axis=1, keepdims=True)
    tied = gaps <= choice_tolerance
    best_margins = np.max(np.where(tied, margins[None, :], -np.inf), axis=1, keepdims=True)
    optimistic = tied & (best_margins - margins[None, :] <= margin_tolerance)
    return gaps, tied, optimistic
