# Analysis of the 100-start batch

This analysis uses the original 100 independent initial menus and the old recovery
policy recorded in this directory. The newer recovery policy was tested separately
only on starts 78, 81 and 85. Replacing those rows does not constitute a fresh
100-start benchmark under the new policy.

## Runtime

| Metric | All 100 starts | 97 completed valid starts |
|---|---:|---:|
| Mean per-start runtime (s) | 9.386 | 8.303 |
| Median (s) | 6.760 | 6.415 |
| 90th percentile (s) | 16.232 | 16.120 |
| 95th percentile (s) | 27.432 | 25.400 |
| Maximum (s) | 109.059 | 40.807 |
| Sum of per-start runtimes (s) | 938.614 | 805.398 |

The four-process batch took 286.363 seconds of wall time. Dividing this by 100
gives a throughput measure, not the latency of one start. The mean launch-to-exit
duration, including Python process startup, was 9.955 seconds. Per-start timings
were collected during parallel execution, rather than isolated single-process runs.
Preprocessing of vertices is excluded; loading the saved cache is included.

66 starts finished within 10 seconds, 93 within 20 seconds and 99 within 60 seconds.
The three failed searches consumed 133.216 seconds, or 14.19% of summed per-start
time. Start 78 alone consumed 11.62% of that sum, making the runtime distribution
strongly skewed. The median therefore describes a typical start better than the mean.

## Search workload

The original batch visited 296 regions and logged 769 completed restricted-NLP
calls, with 778 backend attempts including recovery. Among the 97 successful
searches, the averages were 2.959 regions, 1.959 transitions and 7.691 NLP calls
per start. 35 successful starts finished in their initial assignment region.
No assignment revisit or equal-margin boundary trial was recorded.

The Pearson correlation between runtime and NLP-call count among valid searches
was 0.960; runtime versus region count was 0.933. These associations support the
observed workload explanation: most variation comes from how many regional/CG
solves a trajectory needs, while the outlier start 78 also needed many iterations
inside just two NLP calls.

Across all logged NLP calls, mean solver time was 0.904 seconds and median time
0.684 seconds. Excluding start 78, the mean was 0.766 seconds. Solver-call timing
accounted for about 74.1% of summed end-to-end time; the remainder also includes
full valuation/separation, restart preparation, logging and initialization, so it
cannot all be attributed to solver internals.

## Profit and dependence on the initial menu

All values below refer to final candidates from the 97 successful searches and
use normalized USD per representative person per month, including PAYG users
with zero platform margin.

| Statistic | Final profit |
|---|---:|
| Minimum | 6.556442 |
| 25th percentile | 12.327771 |
| Mean | 14.023548 |
| Median | 14.322484 |
| 75th percentile | 15.448394 |
| Maximum | 17.567400 |

For these same 97 starts, mean initial profit was 7.989751 and mean gain was
6.033797. The ratio of mean final to mean initial profit increased by 75.52%;
this is not the average of the 97 individual percentage improvements, which can
be distorted by starts with initial profits close to zero.

Only one successful result was within 1% of the best, and eleven were within 5%.
The 100 menus induced 96 distinct initial assignments; the 97 valid results had
92 distinct final assignments under the current numerical choice policy. This
demonstrates diverse final outcomes, not 92 independently certified local optima.
Initial and final profit had Pearson correlation 0.566: a better initial profit
helps on average but does not determine the best eventual result. The winning
start had initial profit 6.967458, well below the largest initial profit 14.167291.

## Value of additional starts

The best final result changed only at these initial-menu indices:

| Number of starts considered | Best final profit |
|---|---:|
| 1 | 15.424579 |
| 6 | 16.975517 |
| 14 | 17.567400 |
| 100 | 17.567400 |

Thus 100 starts improved on the original fixed start's final result by 13.89%,
and on the previous ten-start best by 3.49%. Starts 15–100 did not improve the
best final result in this sample. This observed plateau is not a proof that
fourteen starts suffice or that the best result is global.

An intermediate feasible menu already achieved 17.853057, 1.626% above the best
completed candidate, but it was not itself followed through a complete regional
search. It remains separately saved. Continuing from this known good menu and
understanding difficult trajectories is a more targeted next experiment than
assuming that many more blind random starts will necessarily improve the result.

## Later recovery of three failures

The newer SLSQP-only policy recovered all three old failures in separate runs:

| Start | Final profit | Runtime (s) | Search stages |
|---|---:|---:|---:|
| 78 | 14.200517 | 70.336 | 2 |
| 81 | 16.761017 | 28.093 | 2 |
| 85 | 15.323381 | 28.055 | 1 |

Together with the old 97 successful runs, these provide completed candidates for
all original 100 initial menus. If one merely replaces the old three failed rows
with their new replay rows, the mixed collection has mean runtime 9.319 seconds
and mean final profit 14.065691. Those are derived mixed-policy statistics,
not a measured full-batch runtime or success rate for the new policy. A unified
100-start run under that policy has not been performed.

The original full batch plus these three final replays consumed 1,065.098 seconds
of summed per-start time, excluding preliminary recovery-method diagnostics.
This cumulative work must not be confused with the 931.882-second sum obtained
by replacing rows.

Independent profit audits passed. All results use exploratory acceptance with
independent KKT checks disabled and do not certify local/global optimality.

Machine-readable statistics: `statistical_analysis_20261020.json`.
Figures: `runtime_profit_distribution.png/.pdf` and the existing
`profit_convergence.png/.pdf`. Original menu-by-menu details remain in `runs.csv`.
