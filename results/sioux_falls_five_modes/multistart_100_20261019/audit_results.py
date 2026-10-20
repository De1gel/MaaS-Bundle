"""Recompute menus, summarize 100 starts, and plot profit against start count."""

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from maas_bundle.design.menu import BundleMenu
from maas_bundle.examples.sioux_falls import build_sioux_falls_problem


def signature(assignment):
    return hashlib.sha256(np.asarray(assignment, dtype="<i8").tobytes()).hexdigest()


def main():
    output = Path(__file__).resolve().parent
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_starts"] == summary["settings"]["starts"] == 100
    problem = build_sioux_falls_problem(oracle_cache_path=output.parent / "dual_vertices.npz")
    records = []
    payloads = {}
    for row in summary["runs"]:
        payload = json.loads((output / row["result_file"]).read_text())
        payloads[row["start"]] = payload
        initial = problem.evaluate_assignment(BundleMenu(
            payload["initial_prices"], payload["initial_allowances"]
        ))
        initial_error = abs(initial.total_profit - row["initial_profit"])
        assert initial_error < 1e-9
        assert signature(initial.bundle_indices) == payload["initial_assignment_sha256"]
        record = {
            "start": row["start"], "valid": row["valid"], "status": row["status"],
            "initial_assignment_sha256": signature(initial.bundle_indices),
            "initial_profit_error": initial_error,
        }
        if "candidate_prices" in payload:
            response = problem.evaluate_assignment(BundleMenu(
                payload["candidate_prices"], payload["candidate_allowances"]
            ))
            error = abs(response.total_profit - row["final_profit"])
            counts = np.bincount(response.bundle_indices, minlength=3).tolist()
            record.update(
                final_profit_error=error,
                final_assignment_sha256=signature(response.bundle_indices),
                assignment_counts_match=counts == payload["candidate_assignment_counts"],
                recorded_assignment_hash_match=(
                    signature(response.bundle_indices) == payload["candidate_assignment_sha256"]
                ),
                full_separation=payload["diagnostics"]["full_separation"],
            )
            if row["valid"]:
                assert error < 1e-9, record
                assert record["assignment_counts_match"], record
                assert record["recorded_assignment_hash_match"], record
                assert record["full_separation"], record
        observed = problem.evaluate_assignment(BundleMenu(
            payload["best_observed_prices"], payload["best_observed_allowances"]
        ))
        record["best_observed_profit_error"] = abs(
            observed.total_profit - row["best_observed_profit"]
        )
        assert record["best_observed_profit_error"] < 1e-9, record
        history = payload.get("candidate_region_history", [])
        hashes = [step["assumed_assignment_sha256"] for step in history]
        record["solved_regions"] = len(hashes)
        record["distinct_solved_regions"] = len(set(hashes))
        record["maximum_endpoint_double_ties"] = max(
            (step["double_ties"] or 0 for step in history), default=0
        )
        previous = row["initial_profit"]
        for step in history:
            if step["accepted"] and step["true_profit"] is not None:
                assert step["true_profit"] >= previous
                previous = step["true_profit"]
        records.append(record)
        print(json.dumps({"audited": row["start"], "valid": row["valid"]}), flush=True)

    # The first ten initial menus and final results reproduce the earlier batch.
    previous_batch = output.parent / "multistart_20261019"
    for number in range(1, 11):
        old = json.loads((previous_batch / f"start_{number:02d}.json").read_text())
        current = payloads[number]
        assert old["initial_profit"] == current["initial_profit"]
        assert (
            old["candidate_metrics"]["true_profit"] == current["candidate_metrics"]["true_profit"]
        )
        assert np.array_equal(old["candidate_prices"], current["candidate_prices"])
        assert np.array_equal(old["candidate_allowances"], current["candidate_allowances"])

    valid_rows = [row for row in summary["runs"] if row["valid"]]
    profits = [row["final_profit"] for row in valid_rows]
    times = [row["elapsed_seconds"] for row in summary["runs"]]
    best = max(valid_rows, key=lambda row: row["final_profit"])
    old_best = json.loads((previous_batch / "summary.json").read_text())["best_profit"]
    audit = {
        "starts": len(records), "valid_starts": len(valid_rows),
        "first_ten_reproduce_previous_batch_exactly": True,
        "status_counts": dict(Counter(row["status"] for row in summary["runs"])),
        "distinct_initial_assignments": len({r["initial_assignment_sha256"] for r in records}),
        "distinct_valid_final_assignments": len({
            r["final_assignment_sha256"] for r in records if r["valid"]
        }),
        "maximum_final_profit_error": max(r.get("final_profit_error", 0) for r in records),
        "maximum_best_observed_profit_error": max(r["best_observed_profit_error"] for r in records),
        "per_start_seconds": {
            "minimum": min(times), "median": statistics.median(times),
            "maximum": max(times), "sum": sum(times),
        },
        "profit_quantiles": dict(zip(
            ["min", "q25", "median", "q75", "max"],
            np.quantile(profits, [0, .25, .5, .75, 1]).tolist(), strict=True,
        )),
        "valid_starts_within_1pct_of_best": sum(p >= .99 * best["final_profit"] for p in profits),
        "valid_starts_within_5pct_of_best": sum(p >= .95 * best["final_profit"] for p in profits),
        "total_nlp_calls": sum(row["nlp_calls"] for row in summary["runs"]),
        "total_solver_attempts": sum(row["solver_attempts"] for row in summary["runs"]),
        "starts_using_recovery": sum(row["solver_attempts"] > row["nlp_calls"]
                                     for row in summary["runs"]),
        "best_start": best["start"], "best_profit": best["final_profit"],
        "improvement_over_previous_10_start_best_percent": (
            best["final_profit"] / old_best - 1
        ) * 100,
        "improvement_over_original_final_percent": (
            best["final_profit"] / summary["runs"][0]["final_profit"] - 1
        ) * 100,
        "improvement_over_original_initial_percent": (
            best["final_profit"] / summary["runs"][0]["initial_profit"] - 1
        ) * 100,
        "improvement_over_best_start_initial_percent": (
            best["final_profit"] / best["initial_profit"] - 1
        ) * 100,
        "best_observed_start": summary["best_observed_start"],
        "best_observed_profit": summary["best_observed_profit"],
        "records": records,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    best_payload = payloads[best["start"]]
    (output / "best_candidate.json").write_text(json.dumps({
        "start": best["start"], "seed": best["seed"], "source": best["result_file"],
        "profit": best["final_profit"], "check_kkt": False,
        "mode_order": best_payload["experiment"]["mode_order"],
        "prices": best_payload["candidate_prices"],
        "allowances": best_payload["candidate_allowances"],
        "metrics": best_payload["candidate_metrics"],
        "diagnostics": best_payload["diagnostics"],
    }, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "records"}, indent=2))

    import matplotlib.pyplot as plt

    best_so_far = []
    incumbent = -np.inf
    for row in summary["runs"]:
        if row["valid"]:
            incumbent = max(incumbent, row["final_profit"])
        best_so_far.append(incumbent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].step(range(1, 101), best_so_far, where="post", color="#176b87")
    axes[0].axhline(old_best, color="#999999", ls="--", label="Previous 10-start best")
    axes[0].set(xlabel="Number of initial menus", ylabel="Best final profit",
                title="Best result after each prefix of starts", xlim=(1, 100))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].scatter([row["initial_profit"] for row in valid_rows], profits,
                    s=20, alpha=.7, color="#176b87")
    axes[1].scatter(best["initial_profit"], best["final_profit"], color="#c34a36",
                    s=50, label=f"Best: start {best['start']}")
    axes[1].set(xlabel="Initial-menu profit", ylabel="Final profit",
                title="Results from completed valid starts")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "profit_convergence.png", dpi=180)
    fig.savefig(output / "profit_convergence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
