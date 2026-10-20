"""Benchmark settings must reach both positive-start screening and the solver."""

from dataclasses import replace

import pytest

import maas_bundle.benchmarks.sioux_falls as sioux
import maas_bundle.benchmarks.two_route as two_route
from maas_bundle.benchmarks import BenchmarkSettings
from maas_bundle.examples import DEFAULT_SIOUX_FALLS_CONFIG


@pytest.mark.parametrize("name,module", [("sioux_falls", sioux), ("two_route", two_route)])
def test_custom_run_settings_are_separate_from_instance_config(monkeypatch, name, module):
    settings = BenchmarkSettings(
        max_region_iterations=61,
        nlp_max_iterations=321,
        nlp_ftol=2e-8,
        profit_tolerance=3e-8,
        slsqp_restarts=3,
        solver_fallback=False,
    )
    config = replace(DEFAULT_SIOUX_FALLS_CONFIG, initial_prices_usd_per_month=(0, 22, 44))
    kwargs = {"settings": settings}
    if name == "sioux_falls":
        kwargs["config"] = config
    constructions, solvers, generations, runs = [], [], [], []
    problem, menu, solver, result = object(), object(), object(), object()

    def build(**kw):
        constructions.append(kw)
        return problem

    def make_solver(method, *, options, check_kkt=True, slsqp_restarts=1,
                    enable_fallback=True, numerical_constraint_recovery=False):
        solvers.append((method, options, check_kkt, slsqp_restarts, enable_fallback,
                        numerical_constraint_recovery))
        return solver

    def generate(count, **kw):
        generations.append((count, kw))
        return (menu,) * count

    def solve(*args, **kw):
        runs.append((args, kw))
        return result

    monkeypatch.setattr(module, f"build_{name}_problem", build)
    monkeypatch.setattr(module, f"initial_{name}_menu", lambda **kw: menu)
    monkeypatch.setattr(module, f"generate_{name}_initial_menus", generate)
    monkeypatch.setattr(module, "ScipyRegionalSolver", make_solver)
    monkeypatch.setattr(module, "solve_primal", solve)
    monkeypatch.setattr(module, "run_multistart", solve)

    assert getattr(module, f"run_{name}_example")(**kwargs) is result
    assert getattr(module, f"run_{name}_multistart")(2, seed=7, **kwargs) is result
    expected_config = {"config": config} if name == "sioux_falls" else {}
    assert constructions == [expected_config, expected_config]
    assert solvers == [("SLSQP", {"maxiter": 321, "ftol": 2e-8}, True, 3, False, False)] * 2
    assert generations == [
        (2, dict(seed=7, problem=problem, minimum_profit=3e-8, **expected_config))
    ]
    assert runs == [
        ((problem, menu, solver), {"max_region_iterations": 61, "profit_tolerance": 3e-8}),
        ((problem, (menu, menu), solver), {"max_region_iterations": 61, "profit_tolerance": 3e-8}),
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_region_iterations", 0),
        ("max_region_iterations", 1.5),
        ("nlp_max_iterations", -1),
        ("nlp_max_iterations", True),
        ("nlp_ftol", 0),
        ("nlp_ftol", float("nan")),
        ("profit_tolerance", -1),
        ("profit_tolerance", float("inf")),
        ("check_kkt", 1),
        ("slsqp_restarts", -1),
        ("slsqp_restarts", True),
        ("slsqp_restarts", 1.5),
        ("solver_fallback", 1),
        ("numerical_constraint_recovery", 1),
    ],
)
def test_invalid_benchmark_settings_fail_before_building_an_instance(field, value):
    with pytest.raises(ValueError, match=field):
        BenchmarkSettings(**{field: value})
