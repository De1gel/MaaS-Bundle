# SLSQP-only recovery of the three failed starts (2026-10-20)

The chosen recovery method keeps the best independently evaluated feasible menu.
It first retries the failed restricted NLP, then re-enters regional search at the
best observed menu if those retries fail. The assignment is recomputed from that
menu; no assignment is randomly prescribed. All attempts use SLSQP.

## Current policy

1. Solve the restricted NLP normally, returning immediately on success.
2. On failure, allow four additional SLSQP attempts. The first uses the repaired
   endpoint; later attempts perturb the best known restricted-feasible point and
   backtrack to feasibility. The final retry uses numerical constraint derivatives
   to handle active-constraint degeneracy; it is still SLSQP and counts within five.
3. If a regional search still fails, the experiment runner starts another search
   at its best fully evaluated feasible menu, with a freshly computed optimistic
   assignment. The default limit is three such search restarts, all sharing the
   original wall-time budget. Successful candidates stop immediately.
4. Keep feasible menus separately from completed candidates. Failed searches do
   not receive a success or local-optimality label.

Backend switching is disabled by default for Sioux Falls. `--solver-fallback`
explicitly enables trust-constr; `--slsqp-restarts` and `--search-restarts` control
the two restart budgets. `--no-numerical-constraint-recovery` disables the final
numerical-Jacobian retry. `--no-recovery` disables all recovery in the runner.
The generic adapter keeps its compatibility defaults; these Sioux Falls defaults
are set in `DEFAULT_SIOUX_FALLS_SETTINGS` and the experiment scripts.

## Replays

The original seed, initial profit and cached geometry were preserved. Each replay
used 2,000 iterations per NLP attempt, at most 30 regions per search stage, one
BLAS thread and 180 seconds for the entire start, including all restart stages.
The three replays ran concurrently. All finished within the former 120-second
budget, although 180 seconds was allowed. Independent KKT checks stayed disabled.

| Original start | Seed | Final status | Final profit | Seconds | Search stages | Completed NLP calls |
|---|---:|---|---:|---:|---:|---:|
| 78 | 20261095 | consistent_candidate | 14.200517 | 70.336 | 2 | 5 |
| 81 | 20261098 | consistent_candidate | 16.761017 | 28.093 | 2 | 11 |
| 85 | 20261102 | consistent_candidate | 15.323381 | 28.055 | 1 | 41 |

Starts 78 and 81 needed one re-entry into regional search; start 85 completed its
original search after an inner retry. Across stages, starts 78, 81 and 85 visited
2, 5 and 14 regions respectively. The original 100-start results remain unchanged;
this is a separate three-start replay, not a new 100-start experiment.

The best observed feasible menu for start 85 has profit 15.915159, exceeding its
final candidate profit. These values remain separately recorded. The recovery
policy protects the best observed menu; completion of a search does not imply
every intermediate observed menu was optimized in its own assignment region.

`summary.json` records independent full-response and profit checks. All final and
best-observed profits reproduce with zero discrepancy; all completed candidates
passed full separation. Per-attempt names confirm that no trust-constr call occurred.

`start_XX.json` and `.log` contain the final policy replays. `search_attempts`
separates restart stages and their regional trajectories; `attempt_history` stores
each NLP attempt's native status, residual, message and starting menu. The retained
`*_analytic` and `*_inner_only` files document the preliminary checks: analytical
retries alone and inner retries with a final numerical Jacobian rescued start 85
but did not complete starts 78 or 81. This is why re-entry from the best feasible
menu was added instead of relying exclusively on more inner attempts.

Tests: 335 passed, 3 opt-in scale tests skipped; targeted Ruff checks passed.
The tests cover bounded retries, feasible starting points, numerical derivatives
without changing backend, policy forwarding, re-entry at the best feasible menu,
and preserving failure status if the recovery budget is exhausted. These are
exploratory candidates, not independent local-optimality certificates.

Example from the repository root:

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/maas-mpl-cache XDG_CACHE_HOME=/tmp/maas-xdg-cache \
  python scripts/run-sioux-falls-five-mode.py --seed 20261098 \
  --wall-seconds 180 --nlp-iterations 2000 --max-regions 30 \
  --slsqp-restarts 4 --search-restarts 3 \
  --output results/sioux_falls_five_modes/retry_81_rerun.json
```
