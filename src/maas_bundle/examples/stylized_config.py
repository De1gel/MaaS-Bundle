"""Synthetic 3-path, 2-mode instance data, independent of solver settings."""

from dataclasses import asdict, dataclass

import numpy as np

from maas_bundle.design.menu import BundleMenu

MODE_NAMES = ("PT", "SM")
BUNDLE_NAMES = ("PAYG", "Basic", "Premium")
PATH_NAMES = ("p1", "p2", "p3")
MODE_USAGE = ((2.0, 0.0, 1.0), (0.0, 2.0, 1.0))


@dataclass(frozen=True)
class StylizedExperimentConfig:
    """Monetary path costs and monthly demand/allowance data.

    Class names denote intuitive baseline preferences. They do not assert
    which path a class actually uses after fares and allowances are included.
    The optional fourth class reuses M's path costs and resource usage.
    """

    version: str = "stylized-3path-2mode-v2"
    include_high_uncertainty: bool = False
    traveler_weights: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    baseline_path_costs: tuple[tuple[float, float, float], ...] = (
        (4.0, 5.0, 4.5),
        (5.0, 4.5, 3.0),
        (5.5, 3.0, 4.5),
    )
    baseline_frequency_supports: tuple[tuple[float, float], ...] = (
        (14.0, 22.0),
        (16.0, 24.0),
        (18.0, 26.0),
    )
    high_frequency_support: tuple[float, float] = (8.0, 32.0)
    payg_tariff: tuple[float, float] = (2.0, 4.0)
    unit_allowance_cost: tuple[float, float] = (0.5, 1.0)
    # These ranges control candidate sampling only, not the feasible menu domain.
    sampled_price_bounds: tuple[float, float] = (0.0, 190.0)
    sampled_allowance_bounds: tuple[float, float] = (0.0, 64.0)
    initial_prices: tuple[float, float, float] = (0.0, 40.0, 75.0)
    initial_allowances: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (24.0, 8.0),
        (32.0, 24.0),
    )

    def __post_init__(self) -> None:
        weights = np.asarray(self.traveler_weights, dtype=float)
        if weights.shape != (len(self.class_names),):
            raise ValueError("traveler_weights must have one entry per active class")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("traveler_weights must be finite and nonnegative")
        if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("traveler_weights must sum to one")
        costs = np.asarray(self.baseline_path_costs, dtype=float)
        if costs.shape != (3, 3) or not np.all(np.isfinite(costs)) or np.any(costs < 0):
            raise ValueError("baseline_path_costs must be a finite nonnegative 3-by-3 table")
        if len(self.baseline_frequency_supports) != 3:
            raise ValueError("baseline_frequency_supports must have one support per baseline class")
        supports = [
            (f"baseline_frequency_supports[{i}]", support)
            for i, support in enumerate(self.baseline_frequency_supports)
        ] + [("high_frequency_support", self.high_frequency_support)]
        for name, support in supports:
            values = np.asarray(support, dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain two finite endpoints")
            if values[0] < 0 or values[1] <= values[0]:
                raise ValueError(f"{name} must satisfy 0 <= lower < upper")
        for name, vector, strictly_positive in (
            ("payg_tariff", self.payg_tariff, True),
            ("unit_allowance_cost", self.unit_allowance_cost, False),
        ):
            values = np.asarray(vector, dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain two finite coefficients")
            invalid_signs = np.any(values <= 0) if strictly_positive else np.any(values < 0)
            if invalid_signs:
                raise ValueError(f"{name} has invalid coefficient signs")
        for name, sampling_range in (
            ("sampled_price_bounds", self.sampled_price_bounds),
            ("sampled_allowance_bounds", self.sampled_allowance_bounds),
        ):
            values = np.asarray(sampling_range, dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain two finite endpoints")
            if values[0] < 0 or values[1] <= values[0]:
                raise ValueError(f"{name} must satisfy 0 <= lower < upper")
        self.validate_menu(
            BundleMenu(
                prices=np.asarray(self.initial_prices, dtype=float),
                allowances=np.asarray(self.initial_allowances, dtype=float),
            )
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return ("P", "M", "S", "M_H") if self.include_high_uncertainty else ("P", "M", "S")

    @property
    def path_costs(self) -> tuple[tuple[float, float, float], ...]:
        """Append an exact network-preference copy of M when enabled."""

        if self.include_high_uncertainty:
            return self.baseline_path_costs + (self.baseline_path_costs[1],)
        return self.baseline_path_costs

    @property
    def frequency_supports(self) -> tuple[tuple[float, float], ...]:
        supports = self.baseline_frequency_supports
        if self.include_high_uncertainty:
            return supports + (self.high_frequency_support,)
        return supports

    @property
    def payg_expected_costs(self) -> tuple[float, ...]:
        """Compute G_i(0) analytically: mean demand times cheapest PAYG path."""

        fares = np.asarray(self.payg_tariff) @ np.asarray(MODE_USAGE)
        return tuple(
            float(np.min(np.asarray(costs) + fares) * ((lower + upper) / 2))
            for costs, (lower, upper) in zip(self.path_costs, self.frequency_supports, strict=True)
        )

    def validate_menu(self, menu: BundleMenu) -> None:
        """Check dimensions, fixed PAYG and price ordering without upper bounds.

        BundleMenu already enforces finite, nonnegative prices and allowances.
        Sampling ranges do not constrain supplied or optimized menus.
        """

        if menu.bundle_count != 3 or menu.mode_count != 2:
            raise ValueError("the stylized menu must have three options and two modes")
        if menu.prices[0] != 0 or np.any(menu.allowances[0] != 0):
            raise ValueError("PAYG must have zero price and allowances")
        if np.any(np.diff(menu.prices) < 0):
            raise ValueError("subscription prices must be nondecreasing")

    def metadata(self) -> dict[str, object]:
        """Return JSON-compatible parameters, ordering, units and design domain."""

        return {
            "parameters": asdict(self),
            "synthetic_parameters": True,
            "billing_period": "month",
            "currency": "illustrative monetary unit (MU)",
            "class_order": self.class_names,
            "mode_order": MODE_NAMES,
            "path_order": PATH_NAMES,
            "bundle_order": BUNDLE_NAMES,
            "mode_usage": MODE_USAGE,
            "path_costs": self.path_costs,
            "frequency_supports": self.frequency_supports,
            "payg_expected_costs": self.payg_expected_costs,
            "design_domain": {
                "nonnegative_prices": True,
                "nonnegative_allowances": True,
                "nondecreasing_prices": True,
                "fixed_zero_payg": True,
                "finite_upper_bounds": False,
            },
            "choice_rule": "optimistic",
            "congestion": False,
            "units": {
                "path_costs": "MU/OD trip; already monetized, beta = 1",
                "trip_frequency": "OD trips/person/month",
                "mode_usage": "mode-specific resource units/OD trip",
                "payg_tariff": "MU/mode-specific resource unit",
                "unit_allowance_cost": "MU/mode-specific resource unit",
                "allowances": "mode-specific resource units/person/month",
                "prices": "MU/person/month",
                "normalized_profit": "MU/person/month",
                "traveler_weights": "dimensionless population share",
            },
        }


DEFAULT_STYLIZED_CONFIG = StylizedExperimentConfig()
HIGH_UNCERTAINTY_STYLIZED_CONFIG = StylizedExperimentConfig(
    include_high_uncertainty=True,
    traveler_weights=(0.8 / 3, 0.8 / 3, 0.8 / 3, 0.2),
)
