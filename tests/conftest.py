"""Shared pytest reporting for solver integration metrics."""

from __future__ import annotations

import pytest


def pytest_addoption(parser) -> None:
    """Register the opt-in regional NLP scale suite."""

    parser.addoption(
        "--run-regional-scale",
        action="store_true",
        default=False,
        help="run 100/1000-critical-constraint regional NLP scale tests",
    )
    parser.addoption(
        "--run-enumeration",
        action="store_true",
        default=False,
        help="run exponential full-assignment comparison tests",
    )
    parser.addoption(
        "--run-multistart",
        action="store_true",
        default=False,
        help="run the full reproducible multi-start example",
    )
    parser.addoption(
        "--run-sioux-falls",
        action="store_true",
        default=False,
        help="run the full 1,056-traveler Sioux Falls example",
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Keep solver scale experiments out of the fast default unit-test run."""

    for item in items:
        if "regional_scale" in item.keywords and not config.getoption("--run-regional-scale"):
            item.add_marker(
                pytest.mark.skip(reason="use --run-regional-scale to run solver scale tests")
            )
        if "enumeration_scale" in item.keywords and not config.getoption("--run-enumeration"):
            item.add_marker(
                pytest.mark.skip(reason="use --run-enumeration to run full enumeration tests")
            )
        if "multistart_scale" in item.keywords and not config.getoption("--run-multistart"):
            item.add_marker(
                pytest.mark.skip(reason="use --run-multistart to run multi-start tests")
            )
        if "sioux_falls_scale" in item.keywords and not config.getoption("--run-sioux-falls"):
            item.add_marker(
                pytest.mark.skip(reason="use --run-sioux-falls to run Sioux Falls tests")
            )


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Show recorded solver and generation metrics unless pytest is quiet."""

    if config.option.verbose < 0:
        return
    regional_records = _recorded_metrics(terminalreporter, "regional_metric")
    generation_records = _recorded_metrics(terminalreporter, "generation_metric")
    assignment_records = _recorded_metrics(terminalreporter, "assignment_metric")
    validation_records = _recorded_metrics(terminalreporter, "validation_metric")

    if regional_records:
        _write_regional_metrics(terminalreporter, regional_records)
    if generation_records:
        _write_generation_metrics(terminalreporter, generation_records)
    if assignment_records:
        _write_assignment_metrics(terminalreporter, assignment_records)
    if validation_records:
        _write_validation_metrics(terminalreporter, validation_records)


def _recorded_metrics(terminalreporter, property_name: str) -> list[dict]:
    records = []
    for outcome in ("passed", "failed", "xfailed", "xpassed"):
        for report in terminalreporter.stats.get(outcome, []):
            if report.when != "call":
                continue
            records.extend(
                value for name, value in report.user_properties if name == property_name
            )
    return records


def _write_regional_metrics(terminalreporter, records: list[dict]) -> None:
    """Write one compact table for direct regional NLP solves."""

    terminalreporter.write_sep("=", "Regional NLP metrics")
    terminalreporter.write_line(
        "solver                  status  try  critical  vars  iter   objective   "
        "max violation  solver opt     KKT res  KKT  solution check    time"
    )
    for metric in records:
        terminalreporter.write_line(
            f"{metric['solver']:<22} "
            f"{metric['status']:>8}  "
            f"{metric['attempts']:>3}  "
            f"{metric['critical_constraints']:>8}  "
            f"{metric['variables']:>4}  "
            f"{metric['iterations']:>4}  "
            f"{metric['objective']:>10.4f}  "
            f"{metric['max_violation']:>13.3e}  "
            f"{_format_optional(metric['optimality']):>10}  "
            f"{metric['kkt_residual']:>8.2e}  "
            f"{_format_bool(metric['kkt_satisfied']):>3}  "
            f"{metric.get('known_optimum', '-'):>14}  "
            f"{metric['time_seconds']:>7.4f}s"
        )


def _write_generation_metrics(terminalreporter, records: list[dict]) -> None:
    """Write small/large fixed-assignment generation diagnostics."""

    terminalreporter.write_sep("=", "Fixed-assignment generation metrics")
    terminalreporter.write_line(
        "travelers  gen iter  NLP iter  critical(initial/final/+added)  "
        "objective   price  allowance  max violation  KKT     time"
    )
    for metric in records:
        terminalreporter.write_line(
            f"{metric['travelers']:>9}  "
            f"{metric['iterations']:>8}  "
            f"{metric['nlp_iterations']:>8}  "
            f"{metric['initial_critical']:>8}/{metric['final_critical']:<5}"
            f"/+{metric['added']:<7}  "
            f"{metric['objective']:>9.4f}  "
            f"{metric['price']:>6.4f}  "
            f"{metric['allowance']:>9.4f}  "
            f"{metric['max_violation']:>13.3e}  "
            f"{_format_bool(metric['kkt']):>3}  "
            f"{metric['time_seconds']:>7.4f}s"
        )


def _write_assignment_metrics(terminalreporter, records: list[dict]) -> None:
    """Write outer assignment-consistency search diagnostics."""

    terminalreporter.write_sep("=", "Assignment-consistency metrics")
    terminalreporter.write_line(
        "status      regions  changes  revisits  changed travelers  generation/NLP solves  "
        "regional obj  true profit    time"
    )
    for metric in records:
        terminalreporter.write_line(
            f"{metric['status']:<11} "
            f"{metric['regions']:>7}  "
            f"{metric['assignment_changes']:>7}  "
            f"{metric['assignment_revisits']:>8}  "
            f"{metric['changed_travelers']:>17}  "
            f"{metric['generation_iterations']:>10}/{metric['nlp_solves']:<10}  "
            f"{metric['regional_objective']:>12.4f}  "
            f"{_format_optional(metric['true_profit']):>11}  "
            f"{metric['time_seconds']:>7.4f}s"
        )


def _write_validation_metrics(terminalreporter, records: list[dict]) -> None:
    """Write final algorithm-level candidate diagnostics."""

    terminalreporter.write_sep("=", "Primal candidate validation metrics")
    terminalreporter.write_line(
        "status                 valid  assignment  separation  regional  KKT  ties/double"
    )
    for metric in records:
        terminalreporter.write_line(
            f"{metric['status']:<22} "
            f"{_format_bool(metric['valid']):>5}  "
            f"{_format_bool(metric['assignment_consistent']):>10}  "
            f"{_format_bool(metric['full_separation']):>10}  "
            f"{_format_optional_bool(metric['regional_success']):>8}  "
            f"{_format_optional_bool(metric['regional_kkt']):>3}  "
            f"{metric['user_ties']:>4}/{metric['double_ties']:<6}"
        )


def _format_optional(value) -> str:
    return "-" if value is None else f"{value:.3e}"


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _format_optional_bool(value: bool | None) -> str:
    return "-" if value is None else _format_bool(value)
