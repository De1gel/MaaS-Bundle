# Full-domain spatial B&B versus one-shot SLSQP

This comparison freezes the 41 original restricted NLP inputs on start 85's longest regional path. It does not rerun the outer regional algorithm, whose trajectory could change if intermediate NLP solutions change.

SLSQP is called once per saved input, maxiter=2000, ftol=1e-9, with recovery, restarts, fallback and KKT acceptance disabled. Its returned menu is checked for feasibility. No SLSQP success or objective is called a global certificate.

The global method is a custom spatial branch-and-bound over the exact cached valuation oracle, using Gurobi LP node relaxations. It is NOT the native Gurobi nonlinear spatial branch-and-bound engine, and these timings must not be attributed to that engine. All Gurobi node LPs use one thread; BLAS thread counts are also one. Calls are executed sequentially.

Target absolute profit gap: 0.001; per-call tree search budget: 20 seconds. The same one-shot SLSQP result may initialize the global incumbent. Its elapsed time is included in the global total; no prior recovered or refined solutions are used as global warm starts. The zero menu and the saved input also supply feasible incumbent candidates through exact conditional price LPs.

## Measured results

- Checked calls: 41.
- One-shot SLSQP total: 14.274643 seconds; 40/41 feasible returns at 1e-7.
- Global measured total, including local seeds and setup: 787.066665 seconds.
- Global gaps closed to 0.001: 3/41.
- Measured aggregate cost ratio: 55.137.

**Full certification of the whole batch did not finish.** Consequently the measured global total is a cost LOWER BOUND, not a completion time. A time-limit return is not a global certificate. No completion time is extrapolated from the unfinished runs.

| NLP | SLSQP s | Global measured s | Closed? | Global lower | Global upper | Gap |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 0.5058 | 5.3703 | Yes | 16.31284595 | 16.31376707 | 0.000921124 |
| 2 | 0.3554 | 20.3933 | No | 15.03940522 | 15.15284521 | 0.11344 |
| 3 | 0.5485 | 20.5868 | No | 14.50035284 | 14.98061937 | 0.480267 |
| 4 | 0.4001 | 20.4606 | No | 14.29915127 | 14.98076269 | 0.681611 |
| 5 | 0.4215 | 20.4671 | No | 15.22702881 | 15.35899294 | 0.131964 |
| 6 | 0.3115 | 20.4026 | No | 14.72352664 | 15.28694307 | 0.563416 |
| 7 | 0.2283 | 20.2757 | No | 14.66355851 | 15.15427475 | 0.490716 |
| 8 | 0.3380 | 3.6593 | Yes | 18.82773677 | 18.82793446 | 0.000197695 |
| 9 | 0.2903 | 20.3749 | No | 14.75584788 | 15.14314190 | 0.387294 |
| 10 | 0.3835 | 20.5082 | No | 14.75266525 | 15.13589678 | 0.383232 |
| 11 | 0.6406 | 20.7548 | No | 15.77411415 | 15.79158648 | 0.0174723 |
| 12 | 0.2537 | 20.2978 | No | 15.22375847 | 15.26946529 | 0.0457068 |
| 13 | 0.1333 | 20.2407 | No | 14.77521193 | 15.24578620 | 0.470574 |
| 14 | 0.0596 | 20.2048 | No | 14.76938544 | 15.22156233 | 0.452177 |
| 15 | 0.9307 | 20.9954 | No | 18.92779870 | 18.95060462 | 0.0228059 |
| 16 | 0.6277 | 20.7429 | No | 14.88031221 | 15.26181053 | 0.381498 |
| 17 | 0.5599 | 20.6680 | No | 14.83241697 | 15.18912873 | 0.356712 |
| 18 | 0.2508 | 20.3078 | No | 15.35246544 | 15.40134252 | 0.0488771 |
| 19 | 0.2067 | 20.2551 | No | 15.16639597 | 15.35575114 | 0.189355 |
| 20 | 0.1928 | 20.2511 | No | 14.95037706 | 15.27911698 | 0.32874 |
| 21 | 0.1841 | 20.2254 | No | 15.17621202 | 15.27954576 | 0.103334 |
| 22 | 0.3940 | 20.4580 | No | 15.08350521 | 15.19492886 | 0.111424 |
| 23 | 0.3143 | 20.4927 | No | 15.01330558 | 15.15681121 | 0.143506 |
| 24 | 0.4474 | 20.4905 | No | 15.11788194 | 15.23838857 | 0.120507 |
| 25 | 0.1551 | 20.2078 | No | 15.06303794 | 15.19760374 | 0.134566 |
| 26 | 0.2563 | 20.3521 | No | 15.10561062 | 15.17784653 | 0.0722359 |
| 27 | 0.1683 | 20.2815 | No | 15.09933233 | 15.22448989 | 0.125158 |
| 28 | 0.2430 | 20.4350 | No | 15.13849488 | 15.26849095 | 0.129996 |
| 29 | 0.5187 | 20.6140 | No | 15.96917847 | 15.98283019 | 0.0136517 |
| 30 | 0.1906 | 20.2268 | No | 15.41910588 | 15.45715678 | 0.0380509 |
| 31 | 0.3256 | 20.3735 | No | 15.21860256 | 15.41879867 | 0.200196 |
| 32 | 0.2431 | 20.4237 | No | 15.18725056 | 15.27458737 | 0.0873368 |
| 33 | 0.5831 | 20.6638 | No | 15.36825512 | 15.58100401 | 0.212749 |
| 34 | 0.4546 | 20.5441 | No | 15.23946832 | 15.36381008 | 0.124342 |
| 35 | 0.5732 | 20.6739 | No | 16.06569396 | 16.08234052 | 0.0166466 |
| 36 | 0.1460 | 20.2382 | No | 15.51583777 | 15.53851803 | 0.0226803 |
| 37 | 0.3870 | 20.4235 | No | 15.32797336 | 15.43367066 | 0.105697 |
| 38 | 0.0933 | 20.1402 | No | 15.28649463 | 15.38444147 | 0.0979468 |
| 39 | 0.4229 | 1.9454 | Yes | 19.44204863 | 19.44215561 | 0.000106984 |
| 40 | 0.3016 | 20.3683 | No | 15.35267132 | 15.45005511 | 0.0973838 |
| 41 | 0.2333 | 20.2709 | No | 15.32338050 | 15.37625681 | 0.0528763 |

## Full-domain bound construction

For h_iba=C_b-C_a+G_ib-G_ia<=0 and 0<=G_ib(e)<=G_ib(0), C_b<=C_a+G_ia(0). Shortest-path propagation of these difference bounds, plus price order, gives finite price caps. For an incumbent profit L, every potentially improving point satisfies sum_b,m D_b*c_m*e_bm <= sum_b D_b*Ccap_b-L. Positive demands and unit costs therefore give finite coordinate bounds on every potentially improving allowance. No arbitrary search box is imposed.

Monotonic valuation lower bounds at the initial upper allowance corner tighten price caps and cost bounds further. This propagation only excludes points below the feasible incumbent. The full global optimum remains represented in the search or is already covered by the incumbent.

Within a node, integrated feasible dual selections are affine lower bounds for +G. Jensen convex combinations of corner upper values are upper bounds for the subtracted G. Corner upper values integrate chords of the convex trip-frequency envelope. These choices enlarge the original feasible set, so node LPs upper-bound profit. Bisection children cover their complete parent box; unresolved nodes retain their valid inherited upper bound.

Each LP upper bound uses a sign-feasible dual vector and the finite variable bounds to correct all residuals: with r=c-A'pi, U=pi'b+sum_j max(r_j*l_j,r_j*u_j). Dot products use extended precision and an extra 1e-8 padding. These are floating-point numerical certificates, not interval-arithmetic proofs.

Calibration/pilot files are retained separately and excluded from the reported comparison totals. Cache construction and common problem loading are excluded from both solver totals; global setup is included.

## Reproduce

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /opt/homebrew/Caskroom/miniforge/base/envs/maas-bundle/bin/python \
  scripts/benchmark-chain-global.py --seconds 20 --gap 0.001
```
