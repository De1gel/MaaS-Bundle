"""Recompute final responses and profit independently of the saved search state."""

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from maas_bundle.design.menu import BundleMenu
from maas_bundle.examples.sioux_falls import build_sioux_falls_problem


def main():
    output = Path(__file__).resolve().parent
    root = output.parents[2]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_starts"] == summary["settings"]["starts"]
    problem = build_sioux_falls_problem(
        oracle_cache_path=root / summary["settings"]["oracle_cache"]
    )
    records = []
    for run in summary["runs"]:
        payload = json.loads((output / run["result_file"]).read_text())
        if "candidate_prices" not in payload:
            continue
        menu = BundleMenu(payload["candidate_prices"], payload["candidate_allowances"])
        response = problem.evaluate_assignment(menu)
        error = abs(response.total_profit - run["final_profit"])
        counts = np.bincount(response.bundle_indices, minlength=3).tolist()
        assignment_hash = hashlib.sha256(
            np.asarray(response.bundle_indices, dtype="<i8").tobytes()
        ).hexdigest()
        record = {
            "start": run["start"],
            "valid": run["valid"],
            "recomputed_profit": response.total_profit,
            "recorded_profit_absolute_error": error,
            "assignment_counts_match": counts == payload["candidate_assignment_counts"],
            "assignment_counts": counts,
            "assignment_sha256": assignment_hash,
            "full_separation": payload["diagnostics"]["full_separation"],
            "final_assignment": response.bundle_indices.tolist(),
        }
        if run["valid"]:
            assert error < 1e-9, record
            assert record["assignment_counts_match"], record
            assert record["full_separation"], record
        records.append(record)
    valid = [record for record in records if record["valid"]]
    times = [run["elapsed_seconds"] for run in summary["runs"]]
    audit = {
        "runs_recomputed": len(records),
        "valid_starts": len(valid),
        "distinct_valid_final_assignments": len({record["assignment_sha256"] for record in valid}),
        "maximum_profit_recomputation_error": max(
            record["recorded_profit_absolute_error"] for record in records
        ),
        "status_counts": dict(Counter(run["status"] for run in summary["runs"])),
        "per_start_seconds": {"minimum": min(times), "median": statistics.median(times),
                              "maximum": max(times), "sum": sum(times)},
        "total_nlp_calls": sum(run["nlp_calls"] for run in summary["runs"]),
        "total_solver_attempts": sum(run["solver_attempts"] for run in summary["runs"]),
        "best_final_profit_improvement_over_original_percent": (
            summary["best_profit"] / summary["runs"][0]["final_profit"] - 1
        ) * 100,
        "records": records,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
