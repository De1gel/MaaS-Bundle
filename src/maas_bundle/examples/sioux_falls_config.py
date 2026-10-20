"""Dimensional, synthetic monthly settings for the Sioux Falls experiment.

The TNTP time unit is 0.01 hour, hence 0.6 minute. Monetary parameters below
are illustrative USD amounts, not estimates of actual Sioux Falls services.
One allowance unit pays for one passenger traversing one link of that mode.
"""

from dataclasses import asdict, dataclass
from math import isfinite

TNTP_TIME_TO_MINUTES = 0.6
MODE_NAMES = ("bus", "metro", "bike", "ride_hailing", "car_sharing")


@dataclass(frozen=True)
class SiouxFallsExperimentConfig:
    """Immutable instance parameters; physical units are part of the names.

    Editing route costs, tariffs, or modal travel times changes the oracle
    cache key. Prices and allowances are totals for one person's billing month.
    Source OD/capacity numbers retain TNTP's common flow scale, not persons/month.
    """

    version: str = "sioux-falls-monthly-usd-overlapping-five-mode-v3"
    segment_shares: tuple[float, float] = (0.45, 0.55)
    value_of_time_usd_per_minute: tuple[float, float] = (0.18, 0.32)
    preference_usd_per_link_use: tuple[tuple[float, ...], ...] = (
        (0.02, 0.04, 0.06, 0.08, 0.10),
        (0.08, 0.065, 0.05, 0.04, 0.03),
    )
    travel_time_factors: tuple[float, ...] = (1.5, 1.0, 2.5, 0.65, 0.85)
    payg_usd_per_link_use: tuple[float, ...] = (0.30, 0.60, 0.10, 1.20, 0.90)
    provision_usd_per_link_use: tuple[float, ...] = (0.05, 0.08, 0.02, 0.15, 0.10)
    low_frequency_trips_per_month: tuple[float, float] = (8.0, 20.0)
    high_frequency_trips_per_month: tuple[float, float, float] = (20.0, 32.0, 44.0)
    initial_prices_usd_per_month: tuple[float, float, float] = (0.0, 20.0, 40.0)
    initial_allowances_link_uses_per_month: tuple[tuple[float, ...], ...] = (
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (16.0, 16.0, 4.0, 6.0, 6.0),
        (40.0, 40.0, 12.0, 18.0, 18.0),
    )
    sampled_price_bounds_usd_per_month: tuple[float, float] = (4.0, 64.0)
    sampled_basic_allowance_bounds_link_uses_per_month: tuple[float, float] = (0.0, 48.0)
    sampled_premium_increment_bounds_link_uses_per_month: tuple[float, float] = (0.0, 72.0)

    def __post_init__(self) -> None:
        self._check_mode_vector(self.travel_time_factors, "travel_time_factors")
        if any(value <= 0 for value in self.travel_time_factors):
            raise ValueError("travel_time_factors must be positive")
        for name in ("payg_usd_per_link_use", "provision_usd_per_link_use"):
            self._check_mode_vector(getattr(self, name), name)
        if len(self.preference_usd_per_link_use) != len(self.segment_shares):
            raise ValueError("preference rows must match traveler segments")
        for row in self.preference_usd_per_link_use:
            self._check_mode_vector(row, "preference_usd_per_link_use")
        if len(self.initial_allowances_link_uses_per_month) != len(
            self.initial_prices_usd_per_month
        ):
            raise ValueError("initial allowance rows must match initial prices")
        for row in self.initial_allowances_link_uses_per_month:
            self._check_mode_vector(row, "initial_allowances_link_uses_per_month")

    @staticmethod
    def _check_mode_vector(values: tuple[float, ...], name: str) -> None:
        if len(values) != len(MODE_NAMES) or any(
            not isfinite(value) or value < 0 for value in values
        ):
            raise ValueError(f"{name} must contain five finite nonnegative values")

    def metadata(self) -> dict[str, object]:
        """Return a fresh JSON-compatible record with units and all settings."""

        return {
            "parameters": asdict(self),
            "billing_period": "month",
            "currency": "USD",
            "units": {
                "free_flow_time": "minute",
                "source_free_flow_time": "0.01 hour",
                "prices": "USD/person/month",
                "allowances": "passenger-link traversals/person/month",
                "trip_frequency": "one-way OD trips/person/month",
                "mode_usage": "passenger-link traversals/OD trip",
                "normalized_profit": "USD/person/month",
                "true_profit": "USD/person/month",
                "traveler_weights": "dimensionless population-share proxy",
                "source_od_flow_and_capacity": "TNTP scaled flow units",
                "solve_time_seconds": "second",
            },
            "source_time_to_minutes": TNTP_TIME_TO_MINUTES,
            "mode_order": list(MODE_NAMES),
            "mode_availability": (
                "all five modes share every physical link; synthetic travel times and costs"
            ),
            "od_weighting": "segment share times normalized source OD flow; not monthly population",
            "synthetic_parameters": True,
            "path_scope": "all multimodal routes; complete vertex separation by shortest paths",
        }


DEFAULT_SIOUX_FALLS_CONFIG = SiouxFallsExperimentConfig()
