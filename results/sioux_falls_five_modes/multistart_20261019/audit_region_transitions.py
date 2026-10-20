"""Audit exact assignment identities and both kinds of ties in the best run."""

import hashlib
import json
from pathlib import Path

import numpy as np

from maas_bundle.design.menu import BundleMenu
from maas_bundle.examples.sioux_falls import build_sioux_falls_problem


def signature(assignment):
    return hashlib.sha256(np.asarray(assignment, dtype="<i8").tobytes()).hexdigest()


def main():
    folder = Path(__file__).resolve().parent
    trace = json.loads((folder / "start_06_trace.json").read_text())
    problem = build_sioux_falls_problem(oracle_cache_path=folder.parent / "dual_vertices.npz")
    initial_menu = BundleMenu(trace["initial_prices"], trace["initial_allowances"])
    initial = problem.evaluate_assignment(initial_menu)
    assumed = initial.bundle_indices
    assignments = [assumed.copy()]
    visited = set()
    records = []
    for step in trace["candidate_region_history"]:
        # No equal-margin neighboring-region exploration occurred in this run,
        # so each next region is exactly the preceding optimistic response.
        assert step["boundary_origin"] is None
        assert not step["boundary_returned"]
        key = signature(assumed)
        repeated = key in visited
        visited.add(key)
        menu = BundleMenu(step["prices"], step["allowances"])
        response = problem.evaluate_assignment(menu)
        changed = np.flatnonzero(response.bundle_indices != assumed)
        assert changed.size == step["changed_travelers"]
        assert abs(response.total_profit - step["true_profit"]) < 1e-10
        differences = []
        for traveler in changed:
            old, new = assumed[traveler], response.bundle_indices[traveler]
            differences.append({
                "traveler": int(traveler),
                "from_bundle": int(old),
                "to_bundle": int(new),
                "new_minus_old_user_cost": float(
                    response.generalized_costs[traveler, new]
                    - response.generalized_costs[traveler, old]
                ),
                "platform_margin_increase": float(
                    response.bundle_margins[new] - response.bundle_margins[old]
                ),
                "both_numerical_minimum_cost": bool(
                    response.numerical_tie_mask[traveler, old]
                    and response.numerical_tie_mask[traveler, new]
                ),
                "double_tie": bool(response.has_double_ties[traveler]),
            })
        rows = np.arange(problem.traveler_count)
        slacks = (response.generalized_costs
                  - response.generalized_costs[rows, assumed, None])
        records.append({
            "region": step["region"],
            "assumed_assignment_sha256": key,
            "repeated_assumed_assignment": repeated,
            "response_assignment_sha256": signature(response.bundle_indices),
            "true_profit": response.total_profit,
            "ordinary_user_ties": int(np.count_nonzero(response.has_numerical_cost_ties)),
            "double_ties": int(np.count_nonzero(response.has_double_ties)),
            "changed_travelers": int(changed.size),
            "full_region_choice_violation": float(max(0, -np.min(slacks))),
            "changes": differences,
            "assumed_assignment": assumed.tolist(),
        })
        assignments.append(response.bundle_indices.copy())
        assumed = response.bundle_indices
    changes = [change for record in records for change in record["changes"]]
    repeated_individuals = []
    for traveler in range(problem.traveler_count):
        sequence = [int(assignments[0][traveler])]
        for assignment in assignments[1:]:
            if int(assignment[traveler]) != sequence[-1]:
                sequence.append(int(assignment[traveler]))
        if len(sequence) > len(set(sequence)):
            repeated_individuals.append({"traveler": traveler, "bundle_sequence": sequence})
    audit = {
        "regions": len(records),
        "distinct_assumed_assignments": len(visited),
        "repeated_regions": sum(record["repeated_assumed_assignment"] for record in records),
        "region_transitions": sum(record["changed_travelers"] > 0 for record in records),
        "boundary_trials": trace["candidate_metrics"]["boundary_trials"],
        "boundary_returns": trace["candidate_metrics"]["boundary_returns"],
        "maximum_endpoint_double_ties": max(record["double_ties"] for record in records),
        "initial_double_ties": int(np.count_nonzero(initial.has_double_ties)),
        "total_traveler_changes": len(changes),
        "all_changed_travelers_have_cost_ties": all(
            change["both_numerical_minimum_cost"] for change in changes
        ),
        "minimum_margin_gain_of_changed_travelers": min(
            change["platform_margin_increase"] for change in changes
        ),
        "maximum_full_region_choice_violation": max(
            record["full_region_choice_violation"] for record in records
        ),
        "choice_tolerance": initial.choice_tolerance,
        "margin_tolerance": initial.margin_tolerance,
        "individual_travelers_returning_to_prior_bundle": repeated_individuals,
        "records": records,
    }
    assert audit["distinct_assumed_assignments"] == 12
    assert audit["repeated_regions"] == 0
    assert audit["region_transitions"] == 11
    (folder / "start_06_transition_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if key != "records"}, indent=2))
    print("region, user ties, double ties, changed travelers")
    for record in records:
        print(record["region"], record["ordinary_user_ties"], record["double_ties"],
              record["changed_travelers"])


if __name__ == "__main__":
    main()
