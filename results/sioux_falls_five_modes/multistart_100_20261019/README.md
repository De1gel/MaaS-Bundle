# 100-start five-mode Sioux Falls experiment

The batch ran one original fixed menu and 99 random positive-profit menus. Seeds
are 20261019–20261117. The first ten initial profits, final menus and final profits
reproduce the previous ten-start experiment exactly. All runs used the saved
`dual_vertices.npz`; the preprocessing was not repeated.

Four independent child processes ran concurrently, each with one BLAS thread,
a 120-second solve budget, at most 30 regions, 2,000 iterations per NLP attempt,
SLSQP recovery and the trust-constr fallback enabled. Independent KKT acceptance
was disabled, as in the previous experiment. Batch elapsed time was 286.363 seconds
(4 minutes 46 seconds), including process launch and cache loading but excluding
the later independent audit. Per-start elapsed times below were measured during
this parallel execution and should not be interpreted as isolated benchmarks.

| Metric | Result |
|---|---:|
| Starts attempted | 100 |
| Completed valid exploratory candidates | 97 |
| Regional solver failures, with feasible menus retained | 3 |
| Timeouts | 0 |
| Distinct initial assignments | 96 |
| Distinct valid final assignments | 92 |
| Best final profit | 17.567400 |
| Median valid final profit | 14.322484 |
| Worst valid final profit | 6.556442 |
| Valid results within 1% of best | 1 |
| Valid results within 5% of best | 11 |
| Per-start elapsed time: minimum / median / maximum (s) | 0.547 / 6.760 / 109.059 |
| Sum of per-start elapsed times (s) | 938.614 |
| Completed logged NLP calls / backend attempts | 769 / 778 |
| Starts invoking recovery | 4 |

Profits use the model's normalized USD/person/month units. A valid exploratory
candidate passed full separation, assignment consistency, native solver acceptance
and the implemented boundary search under the selected policy; it is not an
independent KKT, second-order or global optimality certificate.

## Profit comparisons

| Reference | Reference profit | Best final profit | Increase |
|---|---:|---:|---:|
| Previous ten-start best final result | 16.975517 | 17.567400 | 3.48669% |
| Original fixed start's final result | 15.424579 | 17.567400 | 13.89225% |
| Original fixed initial menu | 13.726681 | 17.567400 | 27.97995% |
| Best start's own initial menu | 6.967458 | 17.567400 | 152.13498% |

Start 14, seed 20261031, obtained the best final candidate. It took 10.099 seconds,
visited four distinct assignment regions, made three transitions and solved eleven
restricted NLPs. No assignment was revisited, no double tie occurred at its region
endpoints, and no equal-margin boundary trial was needed. Its accepted true profits
were `6.967458 (initial) -> 17.066223 -> 17.332646 -> 17.559674 -> 17.567400`.
The remaining starts 15–100 did not improve on this best final profit.

The highest independently recomputed **intermediate feasible menu** had profit
17.853057, also observed during start 14. This menu differs from the final candidate
and has not itself completed the regional search. It is saved separately in
`best_observed_menu.json`; it is not used in the final-candidate ranking or plot.

## Highest-profit completed candidates

| Start | Seed | Final profit | Seconds | Regions | Transitions | NLP calls |
|---|---|---:|---:|---:|---:|---:|
| 14 | 20261031 | 17.567400 | 10.099 | 4 | 3 | 11 |
| 48 | 20261065 | 17.277035 | 16.613 | 4 | 3 | 12 |
| 80 | 20261097 | 17.107992 | 16.073 | 7 | 6 | 20 |
| 98 | 20261115 | 17.073570 | 13.229 | 3 | 2 | 9 |
| 47 | 20261064 | 17.053722 | 10.337 | 4 | 3 | 9 |
| 6 | 20261023 | 16.975517 | 35.050 | 12 | 11 | 38 |
| 58 | 20261075 | 16.952730 | 6.708 | 4 | 3 | 7 |
| 72 | 20261089 | 16.941254 | 10.820 | 4 | 3 | 8 |
| 44 | 20261061 | 16.874904 | 1.183 | 1 | 0 | 1 |
| 99 | 20261116 | 16.846002 | 13.545 | 5 | 4 | 12 |

## Incomplete searches

Starts 78, 81 and 85 exhausted recovery without native solver success. Start 78's
trust-constr fallback reached its iteration/evaluation limit; starts 81 and 85
ended with native status 4, `Constraint violation exceeds 'gtol'`. The latter two
reported residuals around 5.71e-8 and 9.98e-8: their final rejection came from the
solver's native stopping rule, not the disabled independent KKT check. Each search
retained its previously accepted feasible menu. All 100 retained/final menu profits
and all 100 best-observed menu profits were independently recomputed, with zero
recorded discrepancy. The 97 valid final assignments also match their saved hashes,
counts and completed full-separation diagnostics.

The fallback's `gtol=1e-8` is explicitly configured by our adapter; it is stricter
than the outer feasibility tolerance of 1e-7. Thus starts 81 and 85 satisfy the
outer residual tolerance but not the fallback's native feasibility threshold.
Their native optimality measures are approximately 0.0543 and 0.0964, so these
are stalled near-feasible points, not independently verified local optima rejected
solely by an extra KKT check. In the installed SciPy implementation, status 4 here
follows trust-radius (`xtol`) termination with constraint violation above `gtol`.
Start 78 reaches the fallback's 2,000-iteration limit with zero recorded constraint
violation and native optimality approximately 2.80e-5. Its combined final NLP
record includes 2,215 iterations across both SLSQP attempts and the fallback.
The stored `infeasible` SLSQP statuses describe returned-iterate residuals, not a
proof that the regional problem has no feasible solution. Individual rejected
SLSQP attempts' raw messages and residuals are not separately preserved in the
combined records, so their precise internal failure causes require a replay.

## Artifacts and reproduction

- `summary.json` and `runs.csv`: all starts, statuses, times and profit comparisons.
- `start_XXX.json` and `.log`: individual runs, menus, NLP metrics and region history.
- `best_candidate.json`: the best completed candidate's menu and diagnostics.
- `best_observed_menu.json`: the separately tracked intermediate feasible menu.
- `audit.json`: fresh evaluation of the initial, final and best-observed menus.
- `profit_convergence.png` and `.pdf`: final-profit convergence and initial/final scatter.
- `regional_workload_summary.json` and `regional_workload.csv`: recomputed workload
  statistics and one row per completed regional solve, including CG rounds,
  active constraints and backend attempts.
- `sioux_falls_regional_workload.png` and `.pdf`: distributions of region transitions
  and CG rounds.
- `sioux_falls_longest_trajectory.png` and `.pdf`: profit and initial/added
  traveler-choice constraint counts per region for start 85 (14 distinct regions,
  13 transitions). Bar totals exclude variable bounds and menu-ordering constraints.

The workload figures use the final completed trajectories for all 100 initial
menus: 304 regions and 795 CG rounds, including 304 initial restricted solves and
491 subsequent rounds. The exact source paths are saved in
`regional_workload_summary.json`; the original raw batch is preserved.
A CG round is one restricted NLP solve followed by full separation; solver
retries within that call do not add CG rounds. Mean rounds per region are 2.6151
(median 3, maximum 6); 250/304 regions (82.2%) finish within three rounds.
All region assignments within each completed trajectory are distinct. Runtime
statistics include all recorded solving attempts for each initial menu (mean
10.6510 s, median 6.7597 s); trajectory counts describe the final completed search.

Regenerate these figures, their manuscript copies and workload statistics without
rerunning optimization:

```sh
MPLCONFIGDIR=/tmp/maas-mpl-cache XDG_CACHE_HOME=/tmp/maas-xdg-cache \
  python scripts/plot-sioux-falls-large-scale.py
```

From the repository root in the maas-bundle environment:

```sh
python -u scripts/run-sioux-falls-multistart.py --starts 100 --workers 4 \
  --seed 20261019 --wall-seconds 120 --nlp-iterations 2000 --max-regions 30 \
  --slsqp-restarts 1 --solver-fallback --no-numerical-constraint-recovery \
  --search-restarts 0 \
  --output-dir results/sioux_falls_five_modes/multistart_100_rerun
```

The runner refuses to overwrite an existing experiment summary. To repeat the
independent audit of this saved batch:

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/maas-mpl-cache XDG_CACHE_HOME=/tmp/maas-xdg-cache \
  python results/sioux_falls_five_modes/multistart_100_20261019/audit_results.py
```
