"""Menu decision variables for the MaaS bundle-design problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BundleMenu:
    """Immutable menu snapshot with a price and allowance vector per option."""

    prices: NDArray[np.float64]
    allowances: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate dimensions and store independent, read-only arrays."""

        prices = np.asarray(self.prices, dtype=float)
        allowances = np.asarray(self.allowances, dtype=float)
        if prices.ndim != 1:
            raise ValueError("prices must be one-dimensional")
        if prices.size == 0:
            raise ValueError("prices must contain at least one bundle")
        if allowances.ndim != 2:
            raise ValueError("allowances must be two-dimensional")
        if allowances.shape[0] != prices.size:
            raise ValueError("allowances must have one row per bundle price")
        if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(allowances)):
            raise ValueError("menu values must contain only finite values")
        if np.any(prices < 0) or np.any(allowances < 0):
            raise ValueError("menu prices and allowances must be nonnegative")

        immutable_prices = prices.copy()
        immutable_allowances = allowances.copy()
        immutable_prices.setflags(write=False)
        immutable_allowances.setflags(write=False)
        object.__setattr__(self, "prices", immutable_prices)
        object.__setattr__(self, "allowances", immutable_allowances)

    @property
    def bundle_count(self) -> int:
        """Return the number of options, including the outside option."""

        return self.prices.size

    @property
    def mode_count(self) -> int:
        """Return the number of allowance dimensions."""

        return self.allowances.shape[1]
