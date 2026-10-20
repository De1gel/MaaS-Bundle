# Five-mode Sioux Falls multi-start experiment

10 starts completed: the original fixed menu and nine positive-profit random menus
with seeds 20261019–20261027. All starts used the same preprocessed vertex cache.
They ran sequentially, with 120 seconds per start, at most 30 regions and 2,000
iterations per NLP. SLSQP recovery was enabled; independent KKT checks were disabled.

All 10 starts returned `consistent_candidate`, with full separation and completed
boundary search under the current exploratory policy. All 148 NLP calls were
accepted on the first solver attempt. These results are not independent KKT or
second-order certificates.

| Start | Seed | Final profit | Seconds | Regions | Region changes | NLP calls |
|---|---|---:|---:|---:|---:|---:|
| 1 | Original menu | 15.424579 | 12.886 | 3 | 2 | 13 |
| 2 | 20261019 | 11.491632 | 8.290 | 3 | 2 | 8 |
| 3 | 20261020 | 11.308371 | 10.229 | 4 | 3 | 10 |
| 4 | 20261021 | 15.286458 | 5.747 | 1 | 0 | 3 |
| 5 | 20261022 | 15.072599 | 7.237 | 3 | 2 | 5 |
| **6** | **20261023** | **16.975517** | **32.541** | **12** | **11** | **38** |
| 7 | 20261024 | 15.060879 | 17.296 | 6 | 5 | 20 |
| 8 | 20261025 | 15.448394 | 37.935 | 13 | 12 | 38 |
| 9 | 20261026 | 15.363418 | 12.511 | 3 | 2 | 11 |
| 10 | 20261027 | 11.829921 | 0.599 | 1 | 0 | 2 |

The best final profit is 10.05497% above the original-menu result. Final profits
range from 11.308371 to 16.975517, with median 15.179528. The median per-start
elapsed time is 11.370 seconds. Total batch time is 149.876 seconds, including
process startup and cache loading; it excludes the previously completed vertex
preprocessing and the subsequent independent audit.

A fresh problem instance independently recomputed all final optimistic responses
and profits. All ten profit differences are exactly zero at stored precision;
all assignment counts agree. The ten final assignments are distinct under the
implemented numerical tie policy; this does not establish ten distinct local optima.

`summary.json` and `runs.csv` contain aggregate results. Each `start_XX.json` stores
the final menu, diagnostics and individual NLP metrics; its `.log` stores progress.
`best_observed_profit` records intermediate menus separately from `final_profit`.
The best run here is start 6 by both measures. `audit.json` contains the independent
recomputation, including the full reconstructed assignments.

From the repository root, using the maas-bundle environment:

```sh
python results/sioux_falls_five_modes/multistart_20261019/run_experiment.py
PYTHONPATH=src python results/sioux_falls_five_modes/multistart_20261019/audit_results.py
```

The runner uses `scripts/run-sioux-falls-five-mode.py` without modifying the solver
or preprocessing implementation.

## Best-start trajectory

A supplemental replay of seed 20261023, with per-region logging added to the
runner, reproduced the original final prices, allowances and profit exactly.
Its timing is not included in the original ten-start batch totals above.

The 10.05497% improvement compares the best final candidate (16.975517) with the
original start's final candidate (15.424579). Compared with the original fixed
initial menu (13.726681), the best final candidate improves profit by 23.66804%.
Compared with its own random initial menu (8.674327), it improves profit by
95.69837%.

| Region | Accepted true profit | Increase from previous accepted menu |
|---|---:|---:|
| Initial menu | 8.674327 | — |
| 1 | 15.395103 | 6.720776 |
| 2 | 15.575358 | 0.180255 |
| 3 | 15.708147 | 0.132790 |
| 4 | 15.828000 | 0.119853 |
| 5 | 16.022287 | 0.194288 |
| 6 | 16.143622 | 0.121335 |
| 7 | 16.233893 | 0.090271 |
| 8 | 16.304231 | 0.070338 |
| 9 | 16.827870 | 0.523639 |
| 10 | 16.885612 | 0.057742 |
| 11 | 16.973867 | 0.088255 |
| 12 | 16.975517 | 0.001650 |

All twelve region endpoints were accepted and strictly improved true profit.
The intermediate restricted-NLP objective can decrease as omitted constraints
are added; those intermediate values are not accepted full-region profits.
See `start_06_trace.json`, `start_06_trajectory.csv` and the independent profit
recomputation in `start_06_trajectory_audit.json`.

## Region identity and tie audit

`audit_region_transitions.py` reconstructed each region's full 1,056-entry assumed
assignment from the initial menu and successive optimistic responses. All twelve
assignment vectors are distinct, with eleven transitions and zero revisits. The
per-region numbers of travelers switching afterward are
`2, 2, 6, 8, 2, 4, 2, 2, 16, 2, 10, 0` (56 changes total).

All twelve endpoints and the initial menu have zero double ties. Every switching
traveler has both old and new bundles in the numerical minimum-cost set, but the
platform strictly prefers the selected bundle: the smallest margin gain across
the 56 switches is 14.465985, compared with margin tolerance 1e-9. Thus these are
ordinary user-cost ties with strict platform tie-breaking. There are zero
equal-margin boundary trials or returns, and no individual traveler returns to
a previously selected bundle anywhere in this trajectory. The largest full-region
choice violation is 3.18366e-10, below cost tolerance 1e-7.

Full assignment vectors, hashes and individual switching cost/margin differences
are saved in `start_06_transition_audit.json`.
