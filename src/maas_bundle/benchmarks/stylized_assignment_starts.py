"""One solver-owned numerical seed per assignment, followed by existing search."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from maas_bundle.benchmarks.stylized_regional import (
    CLASSES,
    DEFAULT_STYLIZED_REGIONAL_SETTINGS,
    OPTIONS,
    StylizedRegionalSettings,
    _enumerate,
    _full_region,
    _menu,
    _restore_menu,
    _search,
)
from maas_bundle.benchmarks.stylized_regional_multistart import _audit_run
from maas_bundle.examples import DEFAULT_STYLIZED_CONFIG, build_stylized_problem


def run_stylized_assignment_starts(
    settings: StylizedRegionalSettings = DEFAULT_STYLIZED_REGIONAL_SETTINGS,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Initialize 26 non-PAYG regions once; continue certified positive menus.

    The initial fixed-region solve precedes production search. Its requested
    assignment need not be the optimistic assignment at the numerical seed.
    Initialization response changes are therefore counted separately from
    accepted changes inside the existing positive-menu regional search.
    """
    problem = build_stylized_problem(price_upper_bound=190, allowance_upper_bound=64)
    solver = settings.solver()
    seed = solver.default_initial_menu(_full_region(problem, [1, 1, 1]))
    enumeration = _enumerate(problem, [seed], settings, progress)
    rows = []
    for region in enumeration["records"]:
        initial = region["best"]
        continuation, error = None, None
        path = [region["assignment"]]
        switches = []
        certified = initial is not None and initial["certified_regional"]
        if certified:
            response = initial["response"]
            if response["assignment"] != path[-1]:
                path.append(response["assignment"])
            for i, (a, b) in enumerate(
                zip(region["assignment"], response["assignment"], strict=True)
            ):
                if a != b:
                    switches.append(
                        {
                            "class": CLASSES[i],
                            "from": a,
                            "to": b,
                            "from_name": OPTIONS[a],
                            "to_name": OPTIONS[b],
                            "cost_gap": response["generalized_costs"][i][b]
                            - response["generalized_costs"][i][a],
                            "margin_gap": response["bundle_margins"][b]
                            - response["bundle_margins"][a],
                        }
                    )
            if initial["true_profit"] > settings.profit_tolerance:
                menu = _restore_menu(initial["menu"])
                # solve_primal requires exact price ordering on entry. Preserve
                # its rule instead of rounding a numerically unordered solution.
                if np.any(np.diff(menu.prices) < 0):
                    error = "regional menu violates the production entry price-order rule"
                else:
                    continuation = _search(problem, menu, settings, all_constraints=False)
                    continuation["audit"] = _audit_run(problem, continuation, settings)
                    path.extend(continuation["accepted_assignment_path"][1:])
        rows.append(
            {
                "initial_assignment": region["assignment"],
                "initial_region_certified": bool(certified),
                "initial_region_result": region,
                "initial_response_switches": switches,
                "initial_response_profit_gain": None
                if not certified
                else initial["true_profit"] - initial["regional_profit"],
                "path_including_initial_region_response": path,
                "initialization_error": error,
                "continuation": continuation,
            }
        )

    valid = [row for row in rows if row["continuation"] and row["continuation"]["audit"]["passed"]]
    representative = max(
        valid,
        key=lambda row: (
            len(row["path_including_initial_region_response"]),
            row["initial_response_profit_gain"] > settings.profit_tolerance,
            round(row["continuation"]["final_response"]["true_profit"], 9),
            tuple(-a for a in row["initial_assignment"]),
        ),
        default=None,
    )
    full = None
    if representative:
        full = _search(
            problem,
            _restore_menu(representative["initial_region_result"]["best"]["menu"]),
            settings,
            all_constraints=True,
        )
        full["audit"] = _audit_run(problem, full, settings)
    metadata = DEFAULT_STYLIZED_CONFIG.metadata()
    metadata["design_domain"].update(
        finite_upper_bounds=True,
        price_upper_bounds=problem.price_upper_bounds.tolist(),
        allowance_upper_bounds=problem.allowance_upper_bounds.tolist(),
    )
    search_settings = asdict(settings)
    for field in ("starts", "seed", "trajectory_start"):
        search_settings.pop(field)
    return {
        "experiment": "A2 — One numerical start per initial assignment",
        "model": metadata,
        "settings": search_settings,
        "initialization": {
            "owner": "ScipyRegionalSolver.default_initial_menu",
            "policy": "deterministic zero prices and allowances; no random sampling",
            "solver_initial_menu": _menu(seed),
            "starts_per_assignment": 1,
            "recovery_enabled": solver.enable_recovery,
            "numerical_constraint_recovery": solver.numerical_constraint_recovery,
            "recovery_policy": "restart from local iterates; no independently sampled menus",
            "all_initial_regions_use_full_constraints": True,
            "scope": (
                "initialize each requested assignment by a fixed-region solve; then invoke "
                "production solve_primal from the resulting positive optimistic menu"
            ),
        },
        "summary": {
            "assignment_combinations": 27,
            "excluded_initial_assignments": enumeration["excluded_assignments"],
            "initial_assignments": enumeration["assignment_count"],
            "initialization_regional_solves": enumeration["assignment_count"],
            "starts_per_assignment": 1,
            "initialization_nlp_solves": enumeration["nlp_solves"],
            "certified_initial_regions": enumeration["successful_starts"],
            "failed_initial_regions": enumeration["failed_starts"],
            "continuations_attempted": sum(row["continuation"] is not None for row in rows),
            "audited_valid_continuations": len(valid),
            "continuation_status_counts": dict(
                Counter(row["continuation"]["status"] for row in rows if row["continuation"])
            ),
            "continuation_entry_errors": [
                row["initial_assignment"] for row in rows if row["initialization_error"]
            ],
            "total_nlp_solves_excluding_reference_all_constraints": enumeration["nlp_solves"]
            + sum(row["continuation"]["nlp_solves"] for row in rows if row["continuation"]),
            "multiple_changes_including_initial_response": [
                row["initial_assignment"]
                for row in valid
                if len(row["path_including_initial_region_response"]) >= 3
            ],
            "multiple_accepted_changes_inside_production_search": [
                row["initial_assignment"]
                for row in valid
                if row["continuation"]["accepted_assignment_transitions"] >= 2
            ],
            "representative_initial_assignment": None
            if not representative
            else representative["initial_assignment"],
            "reference_all_constraints_agreement": None
            if not representative
            else {
                "audited_valid": full["audit"]["passed"],
                "same_accepted_search_path": full["accepted_assignment_path"]
                == representative["continuation"]["accepted_assignment_path"],
                "profit_difference": full["final_response"]["true_profit"]
                - representative["continuation"]["final_response"]["true_profit"],
            },
        },
        "initial_region_enumeration": enumeration,
        "runs": rows,
        "representative_all_constraints": full,
        "interpretation": (
            "initial fixed-region response and subsequent accepted search changes are "
            "separate events; failed single solves do not prove region infeasibility; "
            "local solves are not global optimality certificates"
        ),
    }


def _plot_assignment_starts(result: dict, output: Path) -> None:
    representative = next(
        (
            row
            for row in result["runs"]
            if row["initial_assignment"] == result["summary"]["representative_initial_assignment"]
        ),
        None,
    )
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    try:
        if representative:
            initial = representative["initial_region_result"]["best"]
            continuation = representative["continuation"]
            points = [(1, initial["true_profit"], initial["response"]["assignment"])]
            points += [
                (
                    row["iteration"] + 1,
                    row["response"]["true_profit"],
                    row["response"]["assignment"],
                )
                for row in continuation["iterations"]
                if row["accepted"]
            ]
            axis.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                "o-",
                color="#2563a6",
                label="True optimistic outcome",
            )
            axis.scatter(
                1,
                initial["regional_profit"],
                marker="D",
                facecolor="white",
                edgecolor="#bc681e",
                label="Initialization: fixed-region objective",
            )
            axis.annotate(
                "",
                (1, initial["true_profit"]),
                xytext=(1, initial["regional_profit"]),
                arrowprops={"arrowstyle": "->", "color": "#bc681e"},
            )
            for k, profit, assignment in points:
                axis.annotate(
                    str(tuple(assignment)) + f"\n{profit:.4f}",
                    (k, profit),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
            switch_text = "; ".join(
                f"{switch['class']}: {switch['from_name']} → {switch['to_name']}"
                for switch in representative["initial_response_switches"]
            )
            axis.annotate(
                f"Selected initial region: {tuple(representative['initial_assignment'])}\n"
                f"Regional objective: {initial['regional_profit']:.4f}\n"
                f"Initialization response: {switch_text}",
                (1, initial["regional_profit"]),
                xytext=(25, -35),
                textcoords="offset points",
                fontsize=9,
                color="#875021",
            )
            mechanisms = {
                "boundary_neighbor_optimization": "double-tie neighbor optimization",
                "double_tie_resolution": "double-tie resolution",
                "optimistic_reassignment": "optimistic reassignment",
            }
            notes = [
                f"Search k={row['iteration']}: {mechanisms[row['assignment_change_mechanism']]}"
                for row in continuation["iterations"]
                if row["retained_assignment_changed"]
            ]
            axis.text(
                0.98,
                0.14,
                "\n".join(notes),
                transform=axis.transAxes,
                ha="right",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.5", "fc": "#fff7ed", "ec": "#dfb78f"},
            )
            axis.margins(x=0.15, y=0.26)
        else:
            axis.text(
                0.5, 0.5, "No audited positive continuation", transform=axis.transAxes, ha="center"
            )
        axis.set_title("26 candidate initial assignments; one solver-owned numerical seed each")
        axis.set_xlabel("Regional solves (1 = initial fixed-region solve)")
        axis.set_ylabel("Platform profit [MU / person / month]")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="upper left", frameon=False, fontsize=8)
        figure.savefig(output / "a2_assignment_starts.png", dpi=220)
    finally:
        plt.close(figure)


def save_stylized_assignment_starts(result: dict, output_dir: str | Path) -> None:
    """Write one PNG, a 26-row CSV and complete initialization/search traces."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "a2_assignment_starts.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = (
        "initial_assignment",
        "initial_region_certified",
        "initial_response_assignment",
        "path_including_initial_response",
        "initial_response_profit_gain",
        "continuation_status",
        "accepted_search_changes",
        "final_profit",
        "entry_error",
    )
    with (output / "a2_assignment_starts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["runs"]:
            continuation = row["continuation"]
            best = row["initial_region_result"]["best"]
            writer.writerow(
                {
                    "initial_assignment": json.dumps(row["initial_assignment"]),
                    "initial_region_certified": row["initial_region_certified"],
                    "initial_response_assignment": None
                    if best is None
                    else json.dumps(best["response"]["assignment"]),
                    "path_including_initial_response": json.dumps(
                        row["path_including_initial_region_response"]
                    ),
                    "initial_response_profit_gain": row["initial_response_profit_gain"],
                    "continuation_status": None if continuation is None else continuation["status"],
                    "accepted_search_changes": None
                    if continuation is None
                    else continuation["accepted_assignment_transitions"],
                    "final_profit": None
                    if continuation is None
                    else continuation["final_response"]["true_profit"],
                    "entry_error": row["initialization_error"],
                }
            )
    _plot_assignment_starts(result, output)
