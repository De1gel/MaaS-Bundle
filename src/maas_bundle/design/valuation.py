"""Problem-level valuation tables evaluated at one bundle menu."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ValuationTable:
    """Values ``G_ib`` and expected shadow prices at one allowance menu."""

    route_values: NDArray[np.float64]
    expected_omega: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the shared traveler–bundle axes and protect both arrays."""

        route_values = np.asarray(self.route_values, dtype=float)
        expected_omega = np.asarray(self.expected_omega, dtype=float)
        if route_values.ndim != 2:
            raise ValueError("route_values must be two-dimensional")
        if expected_omega.ndim != 3 or expected_omega.shape[:2] != route_values.shape:
            raise ValueError("expected_omega must have one vector per traveler and bundle")
        if not np.all(np.isfinite(route_values)) or not np.all(np.isfinite(expected_omega)):
            raise ValueError("valuation tables must contain only finite values")
        if np.any(expected_omega < 0):
            raise ValueError("expected_omega must be nonnegative")

        immutable_values = route_values.copy()
        immutable_omega = expected_omega.copy()
        immutable_values.setflags(write=False)
        immutable_omega.setflags(write=False)
        object.__setattr__(self, "route_values", immutable_values)
        object.__setattr__(self, "expected_omega", immutable_omega)

    @property
    def traveler_count(self) -> int:
        """Return the number of traveler classes."""

        return self.route_values.shape[0]

    @property
    def bundle_count(self) -> int:
        """Return the number of bundle options."""

        return self.route_values.shape[1]

    @property
    def mode_count(self) -> int:
        """Return the number of allowance dimensions."""

        return self.expected_omega.shape[2]
