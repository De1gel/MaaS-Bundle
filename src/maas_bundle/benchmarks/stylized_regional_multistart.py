"""Repeated complete regional searches on the heterogeneous-demand A2 instance."""

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
    DEFAULT_STYLIZED_REGIONAL_SETTINGS,
    StylizedRegionalSettings,
    _check_menu,
    _response,
    _restore_menu,
    _search,
    regional_illustration_menu,
)
from maas_bundle.examples.stylized import (
    build_stylized_problem,
    draw_stylized_menu,
    initial_stylized_menu,
)
from maas_bundle.examples.stylized_config import DEFAULT_STYLIZED_CONFIG


def _audit_run(problem, run, settings) -> dict:
    """Recompute accepted choices, profits and boundary-entry double ties."""
    previous = _response(problem, _restore_menu(run["initial_menu"]), settings)
    path = [previous.bundle_indices.tolist()]
    choice_checks, profit_checks, boundary_checks = [], [], []
    for row in run["iterations"]:
        if not row["accepted"]:
            continue
        menu = _restore_menu(row["optimized_menu"])
        response = _response(problem, menu, settings)
        check = _check_menu(problem, menu, row["starting_assignment"], settings)
        choice_checks.append(
            check["maximum_violation"] <= settings.feasibility_tolerance
            and response.bundle_indices.tolist() == row["response"]["assignment"]
        )
        profit_checks.append(
            response.total_profit >= previous.total_profit - settings.profit_tolerance
            and abs(response.total_profit - row["response"]["true_profit"])
            <= settings.profit_tolerance
        )
        changed = not np.array_equal(previous.bundle_indices, response.bundle_indices)
        if changed and row["boundary_origin"] is not None:
            entry = _response(problem, _restore_menu(row["starting_menu"]), settings)
            for i, (a, b) in enumerate(
                zip(previous.bundle_indices, row["starting_assignment"], strict=True)
            ):
                if a != b:
                    boundary_checks.append(
                        abs(entry.generalized_costs[i, a] - entry.generalized_costs[i, b])
                        <= settings.feasibility_tolerance
                        and abs(entry.bundle_margins[a] - entry.bundle_margins[b])
                        <= settings.margin_tolerance
                    )
        if changed:
            path.append(response.bundle_indices.tolist())
        previous = response
    final = _response(problem, _restore_menu(run["final_menu"]), settings)
    final_check = _check_menu(problem, final.menu, final.bundle_indices, settings)
    checks = {
        "accepted_region_checks_passed": all(choice_checks),
        "accepted_profits_nondecreasing_and_recomputed": all(profit_checks),
        "boundary_transitions_start_at_double_ties": all(boundary_checks),
        "transition_path_matches_recomputed_choices": path == run["accepted_assignment_path"],
        "final_matches_last_retained_outcome": final.bundle_indices.tolist() == path[-1]
        and abs(final.total_profit - previous.total_profit) <= settings.profit_tolerance,
        "final_menu_valid": final_check["maximum_violation"] <= settings.feasibility_tolerance,
        "algorithm_validation_passed": bool(run["diagnostics"]["valid"]),
    }
    checks["passed"] = all(checks.values())
    return checks


def run_stylized_regional_multistart(
    start_count: int = 500,
    *,
    seed: int = 20261004,
    settings: StylizedRegionalSettings = DEFAULT_STYLIZED_REGIONAL_SETTINGS,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Use two reference menus and positive-profit, uniformly sampled menus.

    Every start calls the existing complete regional search. Assignments are
    inferred from each menu; NLP initialization and algorithm rules are unchanged.
    """
    if not isinstance(start_count, int) or isinstance(start_count, bool) or start_count < 2:
        raise ValueError("start_count must be an integer of at least 2")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    problem = build_stylized_problem(price_upper_bound=190, allowance_upper_bound=64)
    random = np.random.default_rng(seed)
    runs, draws = [], 0
    anchors = [initial_stylized_menu(), regional_illustration_menu()]
    for number in range(1, start_count + 1):
        if number <= len(anchors):
            menu = anchors[number - 1]
        else:
            while True:
                if draws >= 100 * start_count:
                    raise RuntimeError("positive-profit sampling exhausted its draw budget")
                menu = draw_stylized_menu(random)
                draws += 1
                if _response(problem, menu, settings).total_profit > settings.profit_tolerance:
                    break
        run = _search(problem, menu, settings, all_constraints=False)
        run["start_number"] = number
        run["audit"] = _audit_run(problem, run, settings)
        runs.append(run)
        if progress and (number % 50 == 0 or number == start_count):
            histogram = dict(
                sorted(Counter(r["accepted_assignment_transitions"] for r in runs).items())
            )
            progress(f"{number}/{start_count} starts; accepted transition counts: {histogram}")

    valid = [run for run in runs if run["audit"]["passed"]]
    histogram = Counter(run["accepted_assignment_transitions"] for run in runs)
    representative = max(
        valid,
        key=lambda run: (
            run["accepted_assignment_transitions"],
            run["accepted_strict_margin_reassignments"],
            round(run["final_response"]["true_profit"], 9),
            -run["start_number"],
        ),
        default=None,
    )
    reference_all = None
    if representative is not None:
        reference_all = _search(
            problem, _restore_menu(representative["initial_menu"]), settings, all_constraints=True
        )
        reference_all["audit"] = _audit_run(problem, reference_all, settings)
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
        "experiment": "A2 — Multi-start complete regional searches",
        "model": metadata,
        "settings": {"start_count": start_count, "seed": seed, "regional_solver": search_settings},
        "sampling": {
            "reference_starts": 2,
            "random_positive_profit_starts": start_count - 2,
            "random_draws": draws,
            "scheme": (
                "ordered Uniform[0,190] prices; independent Uniform[0,64] allowances; "
                "reject nonpositive true profit"
            ),
        },
        "summary": {
            "starts": start_count,
            "audited_valid_starts": len(valid),
            "status_counts": dict(Counter(run["status"] for run in runs)),
            "accepted_transition_histogram": [
                {"transitions": count, "starts": histogram[count]} for count in sorted(histogram)
            ],
            "starts_with_multiple_transitions": [
                run["start_number"] for run in valid if run["accepted_assignment_transitions"] >= 2
            ],
            "starts_with_multiple_strict_margin_reassignments": [
                run["start_number"]
                for run in valid
                if run["accepted_strict_margin_reassignments"] >= 2
            ],
            "initial_assignments_covered": len(
                {tuple(run["initial_response"]["assignment"]) for run in runs}
            ),
            "valid_terminal_assignments_covered": len(
                {tuple(run["final_response"]["assignment"]) for run in valid}
            ),
            "best_valid_profit": max(
                (run["final_response"]["true_profit"] for run in valid), default=None
            ),
            "regional_nlp_solves": sum(run["nlp_solves"] for run in runs),
            "representative_start": None
            if representative is None
            else representative["start_number"],
            "representative_all_constraints_agreement": None
            if representative is None
            else {
                "audited_valid": reference_all["audit"]["passed"],
                "same_accepted_assignment_path": reference_all["accepted_assignment_path"]
                == representative["accepted_assignment_path"],
                "profit_difference": reference_all["final_response"]["true_profit"]
                - representative["final_response"]["true_profit"],
            },
        },
        "transition_definition": (
            "a change in the retained optimistic assignment after an accepted regional solve; "
            "rejected trials and unchanged assignments are excluded"
        ),
        "scope": (
            "outer initial-menu diagnostic; no change to production regional search, "
            "NLP initialization, model parameters or tie-breaking; "
            "sampling is not an exhaustive proof"
        ),
        "runs": runs,
        "representative_all_constraints": reference_all,
    }


def _plot_multistart(result: dict, output: Path) -> None:
    figure, (counts_axis, path_axis) = plt.subplots(
        1, 2, figsize=(11, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.8]}
    )
    try:
        histogram = result["summary"]["accepted_transition_histogram"]
        xs, counts = [r["transitions"] for r in histogram], [r["starts"] for r in histogram]
        bars = counts_axis.bar(xs, counts, color="#2563a6", width=0.65)
        counts_axis.bar_label(bars, padding=4)
        counts_axis.set_ylim(0, max(counts) * 1.15)
        counts_axis.set_xlabel("Accepted assignment transitions per start")
        counts_axis.set_ylabel("Number of starts")
        counts_axis.set_xticks(xs)
        counts_axis.set_title(f"{result['summary']['starts']} complete regional searches")
        number = result["summary"]["representative_start"]
        if number is not None:
            run = result["runs"][number - 1]
            initial = run["initial_response"]
            points = [(0, initial["true_profit"], initial["assignment"])]
            points.extend(
                (r["iteration"], r["response"]["true_profit"], r["response"]["assignment"])
                for r in run["iterations"]
                if r["accepted"]
            )
            path_axis.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                "o-",
                color="#2563a6",
                label="Accepted outcome",
            )
            for row in run["iterations"]:
                if not row["accepted"]:
                    path_axis.scatter(
                        row["iteration"],
                        row["response"]["true_profit"],
                        marker="x",
                        color="#b34337",
                        label="Rejected boundary return",
                        zorder=4,
                    )
            previous_assignment = None
            for k, profit, assignment in points:
                if assignment == previous_assignment:
                    continue
                text = "(" + ",".join(map(str, assignment)) + f")\n{profit:.4f}"
                path_axis.annotate(
                    text,
                    (k, profit),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
                previous_assignment = assignment
            descriptions = {
                "double_tie_resolution": "cost + margin double tie",
                "boundary_neighbor_optimization": "boundary-neighbor optimization",
                "optimistic_reassignment": "optimistic reassignment",
            }
            changes = [
                f"k={row['iteration']}: {descriptions[row['assignment_change_mechanism']]}"
                for row in run["iterations"]
                if row["retained_assignment_changed"]
            ]
            path_axis.text(
                0.97,
                0.12,
                "\n".join(changes) if changes else "No retained assignment change",
                transform=path_axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.5", "fc": "#fff7ed", "ec": "#dfb78f"},
            )
            path_axis.set_title(
                f"Start {number}: {run['accepted_assignment_transitions']} accepted changes"
            )
            path_axis.margins(x=0.12, y=0.24)
            path_axis.legend(loc="upper left", frameon=False, fontsize=8)
        path_axis.set_xlabel("Regional-search iteration (includes boundary trials)")
        path_axis.set_ylabel("True platform profit [MU / person / month]")
        path_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        for axis in (counts_axis, path_axis):
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        figure.savefig(output / "a2_multistart.png", dpi=220)
    finally:
        plt.close(figure)


def save_stylized_regional_multistart(result: dict, output_dir: str | Path) -> None:
    """Save one PNG, one per-start CSV and the complete trace JSON."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "a2_multistart_diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = (
        "start",
        "status",
        "valid",
        "initial_assignment",
        "accepted_assignment_path",
        "final_assignment",
        "initial_profit",
        "final_profit",
        "accepted_transitions",
        "strict_margin_reassignments",
        "boundary_transitions",
        "nlp_solves",
    )
    with (output / "a2_multistart_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in result["runs"]:
            writer.writerow(
                {
                    "start": run["start_number"],
                    "status": run["status"],
                    "valid": run["audit"]["passed"],
                    "initial_assignment": json.dumps(run["initial_response"]["assignment"]),
                    "accepted_assignment_path": json.dumps(run["accepted_assignment_path"]),
                    "final_assignment": json.dumps(run["final_response"]["assignment"]),
                    "initial_profit": run["initial_response"]["true_profit"],
                    "final_profit": run["final_response"]["true_profit"],
                    "accepted_transitions": run["accepted_assignment_transitions"],
                    "strict_margin_reassignments": run["accepted_strict_margin_reassignments"],
                    "boundary_transitions": run["accepted_boundary_transitions"],
                    "nlp_solves": run["nlp_solves"],
                }
            )
    _plot_multistart(result, output)
