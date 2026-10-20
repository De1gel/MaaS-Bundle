"""Run ten reproducible starts using the existing budgeted experiment runner."""

import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
BASE_SEED = 20261019
START_COUNT = 10
WALL_SECONDS = 120
NLP_ITERATIONS = 2000
MAX_REGIONS = 30


def main():
    started = perf_counter()
    rows = []
    summary = {
        "settings": {
            "starts": START_COUNT,
            "initialization": "original menu, then nine random positive-profit menus",
            "random_seeds": list(range(BASE_SEED, BASE_SEED + START_COUNT - 1)),
            "wall_seconds_per_start": WALL_SECONDS,
            "nlp_max_iterations": NLP_ITERATIONS,
            "max_regions": MAX_REGIONS,
            "check_kkt": False,
            "recovery": True,
            "oracle_cache": "results/sioux_falls_five_modes/dual_vertices.npz",
            "execution": "sequential independent processes",
        },
        "runs": rows,
    }

    def persist():
        valid = [row for row in rows if row["valid"]]
        profits = [row["final_profit"] for row in valid]
        summary.update(
            completed_starts=len(rows),
            valid_starts=len(valid),
            total_elapsed_seconds=perf_counter() - started,
            best_start=max(valid, key=lambda row: row["final_profit"])["start"] if valid else None,
            best_profit=max(profits) if profits else None,
            median_profit=statistics.median(profits) if profits else None,
            worst_profit=min(profits) if profits else None,
        )
        temporary = OUTPUT / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(OUTPUT / "summary.json")
        if rows:
            with (OUTPUT / "runs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(ROOT / "src"), MPLCONFIGDIR="/tmp/maas-mpl-cache",
                       XDG_CACHE_HOME="/tmp/maas-xdg-cache")
    persist()
    for number in range(1, START_COUNT + 1):
        seed = None if number == 1 else BASE_SEED + number - 2
        output = OUTPUT / f"start_{number:02d}.json"
        command = [
            sys.executable, "-u", "scripts/run-sioux-falls-five-mode.py",
            "--output", str(output), "--wall-seconds", str(WALL_SECONDS),
            "--nlp-iterations", str(NLP_ITERATIONS), "--max-regions", str(MAX_REGIONS),
        ]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        print(json.dumps({"starting": number, "seed": seed}), flush=True)
        with output.with_suffix(".log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        payload = json.loads(output.read_text())
        metrics = payload.get("candidate_metrics", {})
        row = {
            "start": number,
            "seed": seed,
            "status": payload["status"],
            "valid": payload["valid"],
            "initial_profit": payload["initial_profit"],
            "final_profit": metrics.get("true_profit"),
            "best_observed_profit": payload["best_observed_profit"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "regions": metrics.get("regions"),
            "assignment_changes": metrics.get("assignment_changes"),
            "assignment_revisits": metrics.get("assignment_revisits"),
            "nlp_calls": len(payload["nlp_attempts"]),
            "solver_attempts": sum(call["attempts"] for call in payload["nlp_attempts"]),
            "returncode": completed.returncode,
            "result_file": output.name,
        }
        rows.append(row)
        persist()
        print(json.dumps(row), flush=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}), flush=True)


if __name__ == "__main__":
    main()
