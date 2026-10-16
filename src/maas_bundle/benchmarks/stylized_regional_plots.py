"""One profit-versus-iteration figure for the observed A2 CG search."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def build_regional_trajectory(result: dict):
    """Label all plotted outcomes with the post-optimistic-response assignment."""
    run = result["regional_search_cg"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    initial = run["initial_response"]
    points = [(0, initial["true_profit"], initial["assignment"])]
    rejected = []
    for row in run["iterations"]:
        response = row["response"]
        point = (row["iteration"], response["true_profit"], response["assignment"])
        (points if row["accepted"] else rejected).append(point)
    ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        "o-",
        color="#2563a6",
        linewidth=1.8,
        markersize=6,
        label="Accepted optimistic outcome",
    )
    ax.scatter(
        [0],
        [initial["true_profit"]],
        s=85,
        facecolors="white",
        edgecolors="#2563a6",
        linewidth=1.8,
        zorder=4,
        label="Initial menu",
    )
    if rejected:
        ax.scatter(
            [p[0] for p in rejected],
            [p[1] for p in rejected],
            marker="x",
            color="#b34337",
            s=45,
            label="Rejected trial",
            zorder=4,
        )
    switch_rows = [row for row in run["iterations"] if row["accepted"] and row["switches"]]
    for index, row in enumerate(switch_rows):
        iteration = row["iteration"]
        regional_profit = row["regional_profit"]
        after_profit = row["response"]["true_profit"]
        ax.scatter(
            [iteration],
            [regional_profit],
            marker="D",
            s=28,
            facecolors="white",
            edgecolors="#bc681e",
            zorder=5,
            label="Regional objective before reassignment" if index == 0 else None,
        )
        ax.annotate(
            "",
            (iteration, after_profit),
            xytext=(iteration, regional_profit),
            arrowprops={"arrowstyle": "->", "color": "#bc681e", "lw": 1.4},
        )
        ax.annotate(
            f"Regional: {regional_profit:.4f}",
            (iteration, regional_profit),
            xytext=(24, -24),
            textcoords="offset points",
            fontsize=8,
            color="#875021",
        )
        switch = row["switches"][0]
        gain = row["same_menu_reassignment_profit_gain"]
        annotation = (
            f"{switch['class']}: {switch['from_name']} → {switch['to_name']}\n"
            "Traveler costs tied\n"
            f"Margin advantage: {switch['margin_gap']:.4f}\n"
            f"Same-menu profit gain: {gain:.5f}"
        )
        ax.annotate(
            annotation,
            (iteration, after_profit),
            xytext=(0.49, 0.18),
            textcoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#fff7ed", "ec": "#dfb78f"},
            arrowprops={
                "arrowstyle": "->",
                "color": "#bc681e",
                "lw": 1.0,
                "connectionstyle": "arc3,rad=-0.12",
            },
        )
    for iteration, profit, assignment in points + rejected:
        label = "(" + ",".join(map(str, assignment)) + ")"
        if iteration == 0:
            label += f"\nInitial: {profit:.4f}"
        elif iteration == run["terminal_region"]:
            label += f"\nFinal: {profit:.4f}"
        elif any(row["iteration"] == iteration for row in switch_rows):
            label += f"\nAfter switch: {profit:.4f}"
        ax.annotate(
            label,
            (iteration, profit),
            xytext=(0, -32 if iteration == run["terminal_region"] else 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    if not run["accepted_assignment_transitions"]:
        ax.text(
            0.98,
            0.03,
            "No assignment switch observed",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#555555",
            fontsize=9,
        )
    ax.set_title("Regional-search trajectory on the stylized instance", fontsize=12, pad=12)
    ax.set_xlabel("Regional-search iteration $k$ (0 = initial menu)")
    ax.set_ylabel("Platform profit [MU / person / month]")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.18, max(1, len(run["iterations"])) + 0.18)
    values = [p[1] for p in points + rejected]
    span = max(1, max(values) - min(values))
    ax.set_ylim(min(values) - 0.23 * span, max(values) + 0.27 * span)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return fig


def plot_regional_trajectory(result: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    figure = build_regional_trajectory(result)
    try:
        for extension in ("pdf", "png"):
            figure.savefig(output / f"a2_regional_search_trajectory.{extension}", dpi=320)
    finally:
        plt.close(figure)
