"""Run budgets and numerical settings shared by benchmark entry points."""

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral

from maas_bundle.design.tolerances import DEFAULT_PROFIT_TOLERANCE

DEFAULT_START_COUNT = 30
DEFAULT_RANDOM_SEED = 20260901


@dataclass(frozen=True)
class BenchmarkSettings:
    """Solver budgets; profit_tolerance has the problem objective's units."""

    max_region_iterations: int = 30
    nlp_max_iterations: int = 2000
    nlp_ftol: float = 1e-9
    profit_tolerance: float = DEFAULT_PROFIT_TOLERANCE
    check_kkt: bool = True
    slsqp_restarts: int = 1
    solver_fallback: bool = True
    numerical_constraint_recovery: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.check_kkt, bool):
            raise ValueError("check_kkt must be Boolean")
        if not isinstance(self.solver_fallback, bool):
            raise ValueError("solver_fallback must be Boolean")
        if not isinstance(self.numerical_constraint_recovery, bool):
            raise ValueError("numerical_constraint_recovery must be Boolean")
        if (
            isinstance(self.slsqp_restarts, bool)
            or not isinstance(self.slsqp_restarts, Integral)
            or self.slsqp_restarts < 0
        ):
            raise ValueError("slsqp_restarts must be a nonnegative integer")
        for name in ("max_region_iterations", "nlp_max_iterations"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isfinite(self.nlp_ftol) or self.nlp_ftol <= 0:
            raise ValueError("nlp_ftol must be finite and positive")
        if not isfinite(self.profit_tolerance) or self.profit_tolerance < 0:
            raise ValueError("profit_tolerance must be finite and nonnegative")

    def metadata(self) -> dict[str, int | float]:
        """Return JSON-compatible run settings, separate from instance parameters."""

        return asdict(self)


DEFAULT_SIOUX_FALLS_SETTINGS = BenchmarkSettings(
    check_kkt=False, slsqp_restarts=4, solver_fallback=False,
    numerical_constraint_recovery=True,
)
DEFAULT_TWO_ROUTE_SETTINGS = BenchmarkSettings(max_region_iterations=20, nlp_ftol=1e-10)
