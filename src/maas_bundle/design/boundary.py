"""Pairwise choice boundaries and critical-traveler separation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from maas_bundle.design.menu import BundleMenu
from maas_bundle.design.valuation import ValuationTable


@dataclass(frozen=True, order=True)
class ChoiceConstraintIndex:
    """Identity of one fixed-assignment traveler choice constraint."""

    traveler: int
    assigned_bundle: int
    alternative_bundle: int


@dataclass(frozen=True)
class ChoiceConstraintViolation:
    """A violated traveler choice constraint and its negative slack."""

    index: ChoiceConstraintIndex
    slack: float

    def __post_init__(self) -> None:
        """Require a finite, strictly negative slack."""

        slack = float(self.slack)
        if not np.isfinite(slack) or slack >= 0:
            raise ValueError("a choice-constraint violation must have finite negative slack")
        object.__setattr__(self, "slack", slack)


@dataclass(frozen=True)
class SeparationResult:
    """Full-separation outcome, with at most one addition per ordered pair."""

    violations: tuple[ChoiceConstraintViolation, ...]
    additions: tuple[ChoiceConstraintViolation, ...]

    @property
    def is_feasible(self) -> bool:
        """Return whether every full traveler constraint is satisfied."""

        return not self.violations

    @property
    def can_add_constraints(self) -> bool:
        """Return whether separation found at least one omitted violation."""

        return bool(self.additions)


class CriticalConstraintSet:
    """Mutable active traveler constraints for one fixed assignment.

    The Boolean state has shape ``(traveler, alternative_bundle)``. The
    assigned bundle is stored once in ``assigned_bundles`` and is therefore
    not repeated as a third dense array axis.
    """

    def __init__(self, assigned_bundles: ArrayLike, bundle_count: int) -> None:
        assignments = _validate_assignments(
            assigned_bundles,
            traveler_count=None,
            bundle_count=bundle_count,
        )
        self._assigned_bundles = assignments
        self._active = np.zeros((assignments.size, bundle_count), dtype=bool)

    @property
    def traveler_count(self) -> int:
        """Return the number of traveler classes in the fixed assignment."""

        return self._assigned_bundles.size

    @property
    def bundle_count(self) -> int:
        """Return the number of bundle options."""

        return self._active.shape[1]

    @property
    def constraint_count(self) -> int:
        """Return the number of currently active traveler constraints."""

        return int(np.count_nonzero(self._active))

    @property
    def assigned_bundles(self) -> NDArray[np.int64]:
        """Return the immutable assignment that owns this active set."""

        return self._assigned_bundles

    @property
    def active_mask(self) -> NDArray[np.bool_]:
        """Return an immutable snapshot of the active constraint mask."""

        result = self._active.copy()
        result.setflags(write=False)
        return result

    def contains(self, index: ChoiceConstraintIndex) -> bool:
        """Return whether a valid choice constraint is already active."""

        self._validate_index(index)
        return bool(self._active[index.traveler, index.alternative_bundle])

    def add(self, index: ChoiceConstraintIndex) -> bool:
        """Activate one constraint and return whether the set changed."""

        self._validate_index(index)
        was_active = self._active[index.traveler, index.alternative_bundle]
        self._active[index.traveler, index.alternative_bundle] = True
        return not bool(was_active)

    def add_all(self, indices: tuple[ChoiceConstraintIndex, ...]) -> int:
        """Activate several constraints and return the number newly added."""

        return sum(self.add(index) for index in indices)

    def active_travelers(self, assigned_bundle: int, alternative_bundle: int) -> NDArray[np.int64]:
        """Return travelers active for one ordered bundle pair."""

        _validate_bundle_pair(assigned_bundle, alternative_bundle, self.bundle_count)
        travelers = np.flatnonzero(
            (self._assigned_bundles == assigned_bundle) & self._active[:, alternative_bundle]
        ).astype(np.int64)
        travelers.setflags(write=False)
        return travelers

    def indices(self) -> tuple[ChoiceConstraintIndex, ...]:
        """Return active constraints in deterministic traveler-major order."""

        travelers, alternatives = np.nonzero(self._active)
        return tuple(
            ChoiceConstraintIndex(
                traveler=int(traveler),
                assigned_bundle=int(self._assigned_bundles[traveler]),
                alternative_bundle=int(alternative),
            )
            for traveler, alternative in zip(travelers, alternatives, strict=True)
        )

    def _validate_index(self, index: ChoiceConstraintIndex) -> None:
        if index.traveler < 0 or index.traveler >= self.traveler_count:
            raise ValueError("constraint traveler index is out of range")
        _validate_bundle_pair(
            index.assigned_bundle,
            index.alternative_bundle,
            self.bundle_count,
        )
        if self._assigned_bundles[index.traveler] != index.assigned_bundle:
            raise ValueError("constraint assigned bundle does not match the fixed assignment")


def assigned_pairwise_values(
    valuation: ValuationTable,
    assigned_bundles: ArrayLike,
) -> NDArray[np.float64]:
    """Return ``delta[i,a] = G[i,a] - G[i,alpha[i]]``."""

    assignments = _validate_assignments(
        assigned_bundles,
        traveler_count=valuation.traveler_count,
        bundle_count=valuation.bundle_count,
    )
    assigned_values = valuation.route_values[np.arange(valuation.traveler_count), assignments]
    result = valuation.route_values - assigned_values[:, None]
    result.setflags(write=False)
    return result


def choice_constraint_slacks(
    menu: BundleMenu,
    valuation: ValuationTable,
    assigned_bundles: ArrayLike,
) -> NDArray[np.float64]:
    """Return slacks for every traveler against every alternative bundle.

    ``slack[i,a] = G[i,a] - G[i,b] - (C[b] - C[a])`` with
    ``b = assigned_bundles[i]``. Feasibility requires nonnegative slack.
    """

    if menu.bundle_count != valuation.bundle_count:
        raise ValueError("menu and valuation must use the same bundles")
    if menu.mode_count != valuation.mode_count:
        raise ValueError("menu and valuation must use the same allowance dimensions")
    assignments = _validate_assignments(
        assigned_bundles,
        traveler_count=valuation.traveler_count,
        bundle_count=valuation.bundle_count,
    )
    differences = assigned_pairwise_values(valuation, assignments)
    assigned_prices = menu.prices[assignments]
    result = differences - (assigned_prices[:, None] - menu.prices[None, :])
    result.setflags(write=False)
    return result


def initialize_critical_constraints(
    menu: BundleMenu,
    valuation: ValuationTable,
    assigned_bundles: ArrayLike,
    *,
    initial_per_pair: int = 1,
    tie_tolerance: float = 1e-12,
) -> CriticalConstraintSet:
    """Initialize each occupied ordered pair with its smallest-slack travelers.

    At least ``initial_per_pair`` travelers are selected when available. All
    additional travelers tied with the last selected slack are included too.
    """

    if not isinstance(initial_per_pair, (int, np.integer)) or initial_per_pair <= 0:
        raise ValueError("initial_per_pair must be a positive integer")
    _validate_tolerance(tie_tolerance, name="tie_tolerance")
    slacks = choice_constraint_slacks(menu, valuation, assigned_bundles)
    pool = CriticalConstraintSet(assigned_bundles, valuation.bundle_count)

    for assigned_bundle in range(valuation.bundle_count):
        travelers = np.flatnonzero(pool.assigned_bundles == assigned_bundle)
        if travelers.size == 0:
            continue
        for alternative_bundle in range(valuation.bundle_count):
            if alternative_bundle == assigned_bundle:
                continue
            pair_slacks = slacks[travelers, alternative_bundle]
            order = np.argsort(pair_slacks, kind="stable")
            selected_count = min(initial_per_pair, travelers.size)
            cutoff = pair_slacks[order[selected_count - 1]] + tie_tolerance
            selected = travelers[pair_slacks <= cutoff]
            pool.add_all(
                tuple(
                    ChoiceConstraintIndex(
                        traveler=int(traveler),
                        assigned_bundle=assigned_bundle,
                        alternative_bundle=alternative_bundle,
                    )
                    for traveler in selected
                )
            )
    return pool


def separate_choice_constraints(
    menu: BundleMenu,
    valuation: ValuationTable,
    critical_constraints: CriticalConstraintSet,
    *,
    violation_tolerance: float = 1e-9,
) -> SeparationResult:
    """Scan all travelers and propose one omitted violation per ordered pair."""

    _validate_tolerance(violation_tolerance, name="violation_tolerance")
    if valuation.traveler_count != critical_constraints.traveler_count:
        raise ValueError("valuation and critical constraints must use the same travelers")
    if valuation.bundle_count != critical_constraints.bundle_count:
        raise ValueError("valuation and critical constraints must use the same bundles")
    slacks = choice_constraint_slacks(
        menu,
        valuation,
        critical_constraints.assigned_bundles,
    )
    active = critical_constraints.active_mask
    violations: list[ChoiceConstraintViolation] = []
    additions: list[ChoiceConstraintViolation] = []

    for assigned_bundle in range(valuation.bundle_count):
        travelers = np.flatnonzero(critical_constraints.assigned_bundles == assigned_bundle)
        if travelers.size == 0:
            continue
        for alternative_bundle in range(valuation.bundle_count):
            if alternative_bundle == assigned_bundle:
                continue
            pair_slacks = slacks[travelers, alternative_bundle]
            most_violated = int(travelers[int(np.argmin(pair_slacks))])
            minimum_slack = float(slacks[most_violated, alternative_bundle])
            if minimum_slack < -violation_tolerance:
                violations.append(
                    _violation(
                        most_violated,
                        assigned_bundle,
                        alternative_bundle,
                        minimum_slack,
                    )
                )

            omitted = travelers[~active[travelers, alternative_bundle]]
            if omitted.size == 0:
                continue
            omitted_slacks = slacks[omitted, alternative_bundle]
            most_violated_omitted = int(omitted[int(np.argmin(omitted_slacks))])
            omitted_slack = float(slacks[most_violated_omitted, alternative_bundle])
            if omitted_slack < -violation_tolerance:
                additions.append(
                    _violation(
                        most_violated_omitted,
                        assigned_bundle,
                        alternative_bundle,
                        omitted_slack,
                    )
                )

    return SeparationResult(violations=tuple(violations), additions=tuple(additions))


def _violation(
    traveler: int,
    assigned_bundle: int,
    alternative_bundle: int,
    slack: float,
) -> ChoiceConstraintViolation:
    return ChoiceConstraintViolation(
        index=ChoiceConstraintIndex(
            traveler=traveler,
            assigned_bundle=assigned_bundle,
            alternative_bundle=alternative_bundle,
        ),
        slack=slack,
    )


def _validate_assignments(
    assigned_bundles: ArrayLike,
    *,
    traveler_count: int | None,
    bundle_count: int,
) -> NDArray[np.int64]:
    raw = np.asarray(assigned_bundles)
    if raw.ndim != 1:
        raise ValueError("assigned_bundles must be one-dimensional")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("assigned_bundles must contain integers")
    if traveler_count is not None and raw.size != traveler_count:
        raise ValueError("assigned_bundles must have one entry per traveler")
    if not isinstance(bundle_count, (int, np.integer)) or bundle_count <= 0:
        raise ValueError("bundle_count must be a positive integer")
    result = raw.astype(np.int64, copy=True)
    if np.any(result < 0) or np.any(result >= bundle_count):
        raise ValueError("assigned_bundles contains an invalid bundle index")
    result.setflags(write=False)
    return result


def _validate_bundle_pair(
    assigned_bundle: int,
    alternative_bundle: int,
    bundle_count: int,
) -> None:
    if not 0 <= assigned_bundle < bundle_count:
        raise ValueError("assigned bundle index is out of range")
    if not 0 <= alternative_bundle < bundle_count:
        raise ValueError("alternative bundle index is out of range")
    if alternative_bundle == assigned_bundle:
        raise ValueError("a choice constraint requires two different bundles")


def _validate_tolerance(value: float, *, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
