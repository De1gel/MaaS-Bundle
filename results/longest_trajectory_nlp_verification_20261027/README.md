# NLP verification along the longest Sioux Falls path

Source: start 85, 14 regions, 13 transitions. The path was selected by the maximum transition count across the same 100 completed source trajectories. Deterministic replay matched the saved objectives, menus, assignments and constraint counts.

Checked 41 restricted NLP return points, including 14 final regional endpoints. The source contains 44 backend attempts; the per-call rows refer to each combined call's final return. An additional 3 internal retry returns were checked for feasibility; 3 were infeasible for their retained NLP. These are not alternative feasible optima.

## Method and acceptance

Each comparison uses the exact original fixed assignment and retained choice constraints. No reassignment, rounding of dual vertices, or modification of the production solver is performed. Original and alternative menus are also checked against every choice constraint in that region. Feasibility threshold: 1e-7. Infeasible restarts are retained in the raw record and excluded from objective-gain calculations.

1. Gurobi reoptimizes both prices at the original allowance vector.
2. Exhaust three allowance grids, with r=0.1, 0.03, 0.01. Coordinate e takes max(0,e-r max(1,e)), e, e+r max(1,e). The two-price LP is optimized at every combination, using the independently tested exact two-price profiler. Grid winners are checked with Gurobi and the original oracle constraints.
3. Restart SLSQP from the original return and the best noncentral feasible grid point, using numerical constraint derivatives (eps=1e-6), maxiter=500, ftol=1e-11, with recovery and fallback disabled.
4. Solve Gurobi LP outer relaxations in allowance boxes at r=0.1, 0.01, 0.001, 0.0001; prices retain their original domain. For convex G, a convex combination of corner values is an upper bound at the same combination of corners. Integrated feasible dual selections yield affine lower bounds. In each h=C_b-C_a+G_ib-G_ia<=0, replace +G_ib by a lower bound and -G_ia by the negative upper bound. This enlarges the feasible set, so maximizing the relaxed profit supplies an objective upper bound in that box.

Corner upper values integrate chords of the max over all cached dual vertices at support endpoints and envelope breakpoints. These chords dominate the convex integrand in trip frequency. Lower constants receive -1e-9 padding and upper values +1e-9; Gurobi LP feasibility and optimality tolerances are 1e-9. These are floating-point bounds, not interval proofs.

## Results

Largest observed feasible gain: 1.13004019829 at call 15 (region 5, CG round 1). This is 6.3483% of its original profit.

The largest remaining gain is 1.210629877235192e-06; the largest gain among full-region-feasible final endpoints is 8.948060443714212e-07. A large-gain intermediate call is assessed as its retained NLP, not as a feasible solution of its complete region.

Grid combinations: 5,011,875. Total verification time: 178.2 seconds after cached-problem construction.

Number of calls with best feasible gain above each absolute threshold:

| Gain threshold | Calls |
| --- | ---: |
| 1e-07 | 10 |
| 1e-06 | 2 |
| 1e-05 | 1 |
| 0.0001 | 1 |
| 0.001 | 1 |
| 0.01 | 1 |

The following table includes all checked restricted NLP returns. An asterisk denotes the last CG call in its region. A large original full-region violation at an intermediate CG call does not invalidate its feasibility for the restricted NLP actually solved.

| Call | Region | Original profit | Best feasible gain | Best source | Original full-region violation | Best full-region violation |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | 1 | 16.312845949355 | 4.91199e-08 | restart_best_noncentral_grid | 16.7 | 16.7 |
| 2 | 1 | 15.039405216665 | 3.37508e-14 | price_lp | 2.551 | 2.551 |
| 3 | 1 | 14.500352841603 | 0 | original | 0.3147 | 0.3147 |
| 4* | 1 | 14.299151270448 | 0 | original | 1.225e-10 | 1.225e-10 |
| 5 | 2 | 15.227028808002 | 0 | original | 3.588 | 3.588 |
| 6 | 2 | 14.723526643028 | 6.26479e-07 | restart_best_noncentral_grid | 2.405 | 2.405 |
| 7* | 2 | 14.663558513753 | 0 | original | 6.53e-11 | 6.53e-11 |
| 8 | 3 | 18.827736768756 | 0 | original | 16.69 | 16.69 |
| 9 | 3 | 14.755847884647 | 2.05681e-09 | restart_best_noncentral_grid | 0.1565 | 0.1565 |
| 10* | 3 | 14.752665250209 | 2.26162e-07 | restart_best_noncentral_grid | 6.966e-11 | 1.421e-14 |
| 11 | 4 | 15.774114150815 | 1.10823e-08 | restart_best_noncentral_grid | 8.093 | 8.093 |
| 12 | 4 | 15.223758465188 | 0 | original | 2.297 | 2.297 |
| 13 | 4 | 14.775211934163 | 0 | original | 0.3092 | 0.3092 |
| 14* | 4 | 14.769385437687 | 8.94806e-07 | restart_best_noncentral_grid | 1.139e-10 | 2.274e-13 |
| 15 | 5 | 17.800746854157 | 1.13004 | restart_best_noncentral_grid | 15.63 | 16.68 |
| 16 | 5 | 14.880312209136 | 0 | original | 1.592 | 1.592 |
| 17* | 5 | 14.832416969487 | 1.59936e-10 | restart_original_return | 1.24e-10 | 1.421e-14 |
| 18 | 6 | 15.352465441899 | 1.21063e-06 | restart_best_noncentral_grid | 4.94 | 4.94 |
| 19 | 6 | 15.166395971005 | 0 | original | 2.555 | 2.555 |
| 20* | 6 | 14.950377055425 | 0 | original | 6.811e-11 | 6.811e-11 |
| 21 | 7 | 15.176212021343 | 0 | original | 2.537 | 2.537 |
| 22 | 7 | 15.083505212736 | 9.28398e-08 | restart_best_noncentral_grid | 1.125 | 1.125 |
| 23* | 7 | 15.013305584297 | 1.20166e-08 | restart_original_return | 5.599e-12 | 2.842e-13 |
| 24 | 8 | 15.117881936165 | 0 | original | 1.833 | 1.833 |
| 25* | 8 | 15.063037941798 | 0 | original | 6.395e-13 | 6.395e-13 |
| 26 | 9 | 15.105610616970 | 0 | original | 0.04919 | 0.04919 |
| 27* | 9 | 15.099332332349 | 0 | original | 9.948e-13 | 9.948e-13 |
| 28* | 10 | 15.138494876054 | 2.40592e-07 | restart_best_noncentral_grid | 5.245e-11 | 8.527e-14 |
| 29 | 11 | 15.969178465152 | 2.02125e-09 | restart_best_noncentral_grid | 8.005 | 8.005 |
| 30 | 11 | 15.419105875992 | 0 | original | 2.307 | 2.307 |
| 31 | 11 | 15.218602561521 | 0 | original | 0.9829 | 0.9829 |
| 32* | 11 | 15.187250564764 | 0 | original | 2.629e-12 | 2.629e-12 |
| 33 | 12 | 15.368255122447 | 7.84948e-09 | restart_original_return | 3.809 | 3.809 |
| 34* | 12 | 15.239468319083 | 1.35752e-07 | restart_best_noncentral_grid | 4.143e-11 | 1.705e-13 |
| 35 | 13 | 16.065693964233 | 1.58e-07 | restart_best_noncentral_grid | 7.961 | 7.961 |
| 36 | 13 | 15.515837768006 | 2.5113e-07 | restart_original_return | 2.311 | 2.312 |
| 37 | 13 | 15.327973362301 | 1.10581e-08 | restart_best_noncentral_grid | 1.35 | 1.35 |
| 38* | 13 | 15.286494627558 | 3.55032e-07 | restart_best_noncentral_grid | 1.387e-10 | 2.842e-14 |
| 39 | 14 | 19.442048628463 | 1.49214e-13 | restart_best_noncentral_grid | 16.63 | 16.63 |
| 40 | 14 | 15.352671318537 | 8.07487e-10 | restart_best_noncentral_grid | 0.7342 | 0.7342 |
| 41* | 14 | 15.323380503977 | 4.56297e-08 | restart_best_noncentral_grid | 3.555e-10 | 5.684e-14 |

## Scope of the conclusion

A positive gain is an observed improvement of the same retained NLP, not by itself a proof that the original return was not locally optimal. No gain found is numerical evidence, not a local or global certificate. Full-region improvements are reported separately and require both the original and alternative points to be full-region feasible. Box upper bounds apply only to the stated boxes, not the entire original unbounded allowance domain. SLSQP success is not used as a certificate of KKT, SOSC or global optimality.

See nlp_calls.csv for every gain and box bound; region_endpoints.csv for the final CG calls; verification.json for all original and alternative menus, grid counts, initial points, constraints and statuses.
For calls with gain above 1e-5, call_N_independent_check.json contains a separate direct-interval-integration and HiGHS price-LP check, plus a shrinking sequence of feasible improving allowance perturbations. These numerical witnesses are not an exact local-optimality theorem.

## Reproduce

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /opt/homebrew/Caskroom/miniforge/base/envs/maas-bundle/bin/python \
  scripts/verify-selected-nlps.py --all-calls \
  --output results/longest_trajectory_nlp_verification_20261027
```
