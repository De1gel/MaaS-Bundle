"""Dimensional, synthetic monthly settings for the Sioux Falls experiment.

The TNTP time unit is 0.01 hour, hence 0.6 minute. Monetary parameters below
are illustrative USD amounts, not estimates of actual Sioux Falls services.
One allowance unit pays for one passenger traversing one link of that mode.
"""

from dataclasses import asdict, dataclass

TNTP_TIME_TO_MINUTES = 0.6


@dataclass(frozen=True)
class SiouxFallsExperimentConfig:
    """Immutable settings; physical units are part of the parameter names.

    Editing route costs, tariffs, or mode classification changes the oracle
    cache key. Prices and allowances are totals for one person's billing month.
    Source OD/capacity numbers retain TNTP's common flow scale, not persons/month.
    """

    version: str = "sioux-falls-monthly-usd-v1"
    segment_shares: tuple[float, float] = (0.45, 0.55)
    value_of_time_usd_per_minute: tuple[float, float] = (0.18, 0.32)
    preference_usd_per_link_use: tuple[tuple[float, float], ...] = (
        (0.02, 0.10),
        (0.08, 0.03),
    )
    payg_usd_per_link_use: tuple[float, float] = (0.45, 0.75)
    provision_usd_per_link_use: tuple[float, float] = (0.05, 0.08)
    low_frequency_trips_per_month: tuple[float, float] = (8.0, 20.0)
    high_frequency_trips_per_month: tuple[float, float, float] = (20.0, 32.0, 44.0)
    initial_prices_usd_per_month: tuple[float, float, float] = (0.0, 20.0, 40.0)
    initial_allowances_link_uses_per_month: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (32.0, 16.0),
        (80.0, 48.0),
    )
    sampled_price_bounds_usd_per_month: tuple[float, float] = (4.0, 64.0)
    sampled_basic_allowance_bounds_link_uses_per_month: tuple[float, float] = (0.0, 48.0)
    sampled_premium_increment_bounds_link_uses_per_month: tuple[float, float] = (0.0, 72.0)
    trunk_capacity_threshold_source_flow_units: float = 10000.0
    default_starts: int = 30
    default_seed: int = 20260901
    max_region_iterations: int = 30
    nlp_max_iterations: int = 2000
    nlp_ftol: float = 1e-9
    minimum_profit_usd_per_person_month: float = 1e-9

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
            "mode_order": ["trunk", "feeder"],
            "od_weighting": "segment share times normalized source OD flow; not monthly population",
            "synthetic_parameters": True,
            "path_scope": "all simple OD paths; all dual vertices cached",
        }


DEFAULT_SIOUX_FALLS_CONFIG = SiouxFallsExperimentConfig()
