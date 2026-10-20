"""Run independent budgeted Sioux Falls starts and save results as they finish."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--wall-seconds", type=int, default=120)
    parser.add_argument("--nlp-iterations", type=int, default=2000)
    parser.add_argument("--max-regions", type=int, default=30)
    parser.add_argument("--slsqp-restarts", type=int, default=4)
    parser.add_argument("--search-restarts", type=int, default=3)
    parser.add_argument(
        "--numerical-constraint-recovery", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--solver-fallback", action="store_true")
    parser.add_argument(
        "--oracle-cache", default="results/sioux_falls_five_modes/dual_vertices.npz"
    )
    parser.add_argument(
        "--output-dir", default="results/sioux_falls_five_modes/multistart_100_20261019"
    )
    args = parser.parse_args()
    budgets = (args.starts, args.workers, args.wall_seconds, args.nlp_iterations, args.max_regions)
    if min(budgets) <= 0:
        parser.error("run counts and budgets must be positive")
    if min(args.slsqp_restarts, args.search_restarts) < 0:
        parser.error("restart counts must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        parser.error("output directory already has a summary; select a new directory")
    started = perf_counter()
    rows = []
    summary = {
        "settings": {
            **vars(args),
            "initialization": "original fixed menu, then random positive-profit menus",
            "random_seeds": list(range(args.seed, args.seed + args.starts - 1)),
            "check_kkt": False,
            "recovery": True,
            "execution": "independent child processes",
            "blas_threads_per_process": 1,
        },
        "runs": rows,
    }

    def persist():
        rows.sort(key=lambda row: row["start"])
        valid = [row for row in rows if row["valid"] and row["final_profit"] is not None]
        profits = [row["final_profit"] for row in valid]
        best = max(valid, key=lambda row: row["final_profit"]) if valid else None
        observed = [row for row in rows if row["best_observed_profit"] is not None]
        best_observed = (
            max(observed, key=lambda row: row["best_observed_profit"]) if observed else None
        )
        summary.update(
            completed_starts=len(rows),
            valid_starts=len(valid),
            status_counts=dict(Counter(row["status"] for row in rows)),
            total_elapsed_seconds=perf_counter() - started,
            best_start=best["start"] if best else None,
            best_profit=best["final_profit"] if best else None,
            median_profit=statistics.median(profits) if profits else None,
            worst_profit=min(profits) if profits else None,
            best_observed_start=best_observed["start"] if best_observed else None,
            best_observed_profit=best_observed["best_observed_profit"] if best_observed else None,
        )
        temporary = output / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(output / "summary.json")
        if rows:
            with (output / "runs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(root / "src"),
        MPLCONFIGDIR="/tmp/maas-mpl-cache",
        XDG_CACHE_HOME="/tmp/maas-xdg-cache",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )

    def run_start(number):
        seed = None if number == 1 else args.seed + number - 2
        destination = output / f"start_{number:03d}.json"
        command = [
            sys.executable, "-u", str(root / "scripts/run-sioux-falls-five-mode.py"),
            "--output", str(destination), "--oracle-cache", args.oracle_cache,
            "--wall-seconds", str(args.wall_seconds),
            "--nlp-iterations", str(args.nlp_iterations), "--max-regions", str(args.max_regions),
            "--slsqp-restarts", str(args.slsqp_restarts),
            "--search-restarts", str(args.search_restarts),
        ]
        if args.solver_fallback:
            command.append("--solver-fallback")
        if not args.numerical_constraint_recovery:
            command.append("--no-numerical-constraint-recovery")
        if seed is not None:
            command.extend(["--seed", str(seed)])
        launched = perf_counter()
        process_error = None
        returncode = None
        with destination.with_suffix(".log").open("w") as log:
            try:
                process = subprocess.run(
                    command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    check=False, timeout=args.wall_seconds + 60,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                process_error = "child process exceeded hard wall-time limit"
        payload = json.loads(destination.read_text()) if destination.exists() else {}
        status = payload.get("status", "process_error")
        if process_error or status == "running":
            status = "process_error"
        metrics = payload.get("candidate_metrics", {})
        calls = payload.get("nlp_attempts", [])
        return {
            "start": number,
            "seed": seed,
            "status": status,
            "valid": bool(payload.get("valid", False) and returncode == 0),
            "initial_profit": payload.get("initial_profit"),
            "final_profit": metrics.get("true_profit"),
            "best_observed_profit": payload.get("best_observed_profit"),
            "elapsed_seconds": payload.get("elapsed_seconds", perf_counter() - launched),
            "process_elapsed_seconds": perf_counter() - launched,
            "regions": metrics.get("regions"),
            "search_attempts": len(payload.get("search_attempts", [])),
            "total_completed_regions": sum(
                attempt["candidate_metrics"]["regions"]
                for attempt in payload.get("search_attempts", [])
            ),
            "assignment_changes": metrics.get("assignment_changes"),
            "assignment_revisits": metrics.get("assignment_revisits"),
            "boundary_trials": metrics.get("boundary_trials"),
            "boundary_returns": metrics.get("boundary_returns"),
            "final_double_ties": payload.get("diagnostics", {}).get("double_ties"),
            "nlp_calls": len(calls),
            "solver_attempts": sum(call["attempts"] for call in calls),
            "returncode": returncode,
            "process_error": process_error,
            "result_file": destination.name,
        }

    persist()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run_start, number) for number in range(1, args.starts + 1)]
        for completed in as_completed(pending):
            row = completed.result()
            rows.append(row)
            persist()
            print(json.dumps({"completed": len(rows), **row}), flush=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}), flush=True)


if __name__ == "__main__":
    main()
