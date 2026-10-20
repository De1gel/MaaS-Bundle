"""Summarize the saved 100-start batch; no optimization is rerun."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def describe(values):
    return {
        "n": len(values),
        "total": int(sum(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": int(max(values)),
        "counts": dict(sorted(Counter(values).items())),
    }


def save_figure(fig, name, output, paper):
    for suffix in ("png", "pdf"):
        destination = output / f"{name}.{suffix}"
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        shutil.copyfile(destination, paper / destination.name)
    plt.close(fig)


def distribution(ax, values, color, xlabel, ylabel, title):
    counts = Counter(values)
    ticks = np.arange(min(values), max(values) + 1)
    heights = [counts[int(tick)] for tick in ticks]
    bars = ax.bar(ticks, heights, width=0.72, color=color)
    ax.bar_label(bars, labels=[str(n) if n else "" for n in heights], fontsize=9, padding=3)
    ax.set(xticks=ticks, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_ylim(0, max(heights) * 1.22)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results/sioux_falls_five_modes/multistart_100_20261019",
    )
    parser.add_argument(
        "--paper-figures", type=Path, default=root / "Paper/MaaS bundle design/figures"
    )
    parser.add_argument(
        "--additional-results",
        type=Path,
        default=root / "results/sioux_falls_five_modes/slsqp_retries_20261020",
    )
    args = parser.parse_args()
    output = args.results_dir.resolve()
    paper = args.paper_figures.resolve()
    paper.mkdir(parents=True, exist_ok=True)
    runs = [
        (int(path.stem.split("_")[1]), json.loads(path.read_text()))
        for path in sorted(output.glob("start_[0-9][0-9][0-9].json"))
    ]
    completed = []
    elapsed = {}
    sources = {}
    for start, original in runs:
        elapsed[start] = original["elapsed_seconds"]
        run = original
        source = output / f"start_{start:03d}.json"
        if not original["valid"]:
            source = args.additional_results / f"start_{start}.json"
            run = json.loads(source.read_text())
            assert run["initial_assignment_sha256"] == original["initial_assignment_sha256"]
            assert np.allclose(run["initial_prices"], original["initial_prices"])
            assert np.allclose(run["initial_allowances"], original["initial_allowances"])
            elapsed[start] += run["elapsed_seconds"]
        assert run["valid"] and run["diagnostics"]["full_separation"]
        completed.append((start, run))
        sources[start] = str(source.relative_to(root))
    if len(completed) != 100:
        raise ValueError("Expected 100 completed starts")

    regional_rows = []
    for start, run in completed:
        history = run["candidate_region_history"]
        metrics = run["candidate_metrics"]
        assert len(history) == metrics["regions"]
        assert len(history) - 1 == metrics["assignment_changes"]
        assert len({row["assumed_assignment_sha256"] for row in history}) == len(history)
        assert (
            sum(row["generation_iterations"] for row in history) == metrics["generation_iterations"]
        )
        stage = run.get("search_attempts", [{}])[-1]
        offset = stage.get("first_nlp_index", 0)
        assert offset + metrics["generation_iterations"] == len(run["nlp_attempts"])
        for row in history:
            assert row["accepted"] and row["true_profit"] is not None
            rounds = row["generation_iterations"]
            attempts = run["nlp_attempts"][offset : offset + rounds]
            regional_rows.append(
                {
                    "start": start,
                    "region": row["region"],
                    "cg_rounds_including_initial": rounds,
                    "additional_constraint_rounds": rounds - 1,
                    "initial_active_constraints": attempts[0]["critical_constraints"],
                    "final_active_constraints": attempts[-1]["critical_constraints"],
                    "backend_attempts": sum(attempt["attempts"] for attempt in attempts),
                    "profit_after_response": row["true_profit"],
                    "changed_traveler_classes": row["changed_travelers"],
                }
            )
            offset += rounds

    transitions = [run["candidate_metrics"]["assignment_changes"] for _, run in completed]
    rounds = [row["cg_rounds_including_initial"] for row in regional_rows]
    selected_id, selected = max(
        completed, key=lambda pair: pair[1]["candidate_metrics"]["assignment_changes"]
    )
    history = selected["candidate_region_history"]
    selected_rows = [row for row in regional_rows if row["start"] == selected_id]
    initial_constraints = np.array([row["initial_active_constraints"] for row in selected_rows])
    final_constraints = np.array([row["final_active_constraints"] for row in selected_rows])
    added_constraints = final_constraints - initial_constraints
    assert np.all(added_constraints >= 0)
    stats = {
        "source": output.name,
        "scope": "Final completed search trajectory for each of 100 initial menus",
        "sources_by_start": sources,
        "time_definition": "Original recorded elapsed time plus additional solving time, if any",
        "cumulative_elapsed_seconds_by_start": elapsed,
        "runtime_seconds": {
            "mean": float(np.mean(list(elapsed.values()))),
            "median": float(np.median(list(elapsed.values()))),
            "p95": float(np.percentile(list(elapsed.values()), 95)),
            "max": float(max(elapsed.values())),
        },
        "final_profit": {
            "mean": float(np.mean([r["candidate_metrics"]["true_profit"] for _, r in completed])),
            "median": float(
                np.median([r["candidate_metrics"]["true_profit"] for _, r in completed])
            ),
            "max": float(max(r["candidate_metrics"]["true_profit"] for _, r in completed)),
        },
        "round_definition": (
            "One restricted NLP solve plus full separation; includes initial solve. "
            "Backend retries do not count as new constraint-generation rounds."
        ),
        "transitions_per_start": describe(transitions),
        "regions_per_start": describe(
            [len(run["candidate_region_history"]) for _, run in completed]
        ),
        "cg_rounds_per_start": describe(
            [run["candidate_metrics"]["generation_iterations"] for _, run in completed]
        ),
        "cg_rounds_per_region": describe(rounds),
        "additional_rounds_per_region": describe([value - 1 for value in rounds]),
        "regions_completed_within_three_rounds": sum(value <= 3 for value in rounds),
        "regions_completed_within_four_rounds": sum(value <= 4 for value in rounds),
        "all_within_start_assignments_distinct": True,
        "selected_trajectory": {
            "start": selected_id,
            "regions": len(history),
            "transitions": len(history) - 1,
            "cg_rounds": [row["generation_iterations"] for row in history],
            "initial_constraints": initial_constraints.tolist(),
            "final_constraints": final_constraints.tolist(),
            "added_constraints": added_constraints.tolist(),
            "initial_profit": selected["initial_profit"],
            "profits_after_response": [row["true_profit"] for row in history],
            "elapsed_seconds": elapsed[selected_id],
        },
    }
    (output / "regional_workload_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    with (output / "regional_workload.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(regional_rows[0]))
        writer.writeheader()
        writer.writerows(regional_rows)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
        }
    )
    blue, orange = "#28638F", "#C97B32"
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), layout="constrained")
    distribution(
        axes[0],
        transitions,
        blue,
        "Region transitions per start",
        "Number of starts",
        f"(a) {len(completed)} completed searches",
    )
    distribution(
        axes[1],
        rounds,
        orange,
        "CG rounds per region (initial solve included)",
        "Number of regions",
        f"(b) {len(regional_rows)} completed regional solves",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "sioux_falls_regional_workload", output, paper)

    region_ids = np.arange(1, len(history) + 1)
    profits = [selected["initial_profit"]] + [row["true_profit"] for row in history]
    assert np.all(np.diff(profits) >= -1e-7)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), layout="constrained")
    ax = axes[0]
    ax.plot(np.arange(len(profits)), profits, "o-", color=blue, ms=3.5, lw=1.5)
    ax.set(
        xlabel="Regional solve (0 = initial menu)",
        ylabel="Profit (USD/person/month)",
        title=f"(a) Start {selected_id}: {len(history) - 1} transitions",
        xticks=[0, 2, 4, 6, 8, 10, 12, 14],
        xlim=(-0.5, len(history) + 0.5),
    )
    ax.set_ylim(min(profits) - 0.6, max(profits) + 1.2)
    ax.annotate(
        f"Initial: {profits[0]:.4f}",
        (0, profits[0]),
        xytext=(8, 1),
        textcoords="offset points",
        fontsize=9,
    )
    ax.annotate(
        f"Final: {profits[-1]:.4f}",
        (len(history), profits[-1]),
        xytext=(-2, 13),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.2)
    ax = axes[1]
    ax.bar(region_ids, initial_constraints, color=blue, width=0.72, label="Initial constraints")
    bars = ax.bar(
        region_ids,
        added_constraints,
        bottom=initial_constraints,
        color=orange,
        width=0.72,
        label="Added by CG",
    )
    ax.bar_label(bars, labels=[str(n) for n in final_constraints], padding=3, fontsize=9)
    ax.set(
        xlabel="Region visited",
        ylabel="Number of constraints",
        xticks=region_ids,
        title="(b) Constraints per region",
        ylim=(0, max(final_constraints) * 1.38),
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "sioux_falls_longest_trajectory", output, paper)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
