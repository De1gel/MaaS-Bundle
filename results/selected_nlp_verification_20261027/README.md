# Three saved NLP checks

The three checked endpoints are calls **4, 28, 41** on the deterministically
reconstructed start-85 trajectory, in regions 1, 10, 14. They use respectively
24, 24, 19 retained choice constraints. The actual assignment, retained
constraints, costs, distributions and cached dual vertices are unchanged.
These are complete-region endpoints, but the optimization comparisons retain
the exact restricted NLP each solver call originally solved. All reported
returned candidate menus are also checked against all 2,112 full-region choice
constraints. No optimistic reassignment is performed.

## Findings

| Call | Original profit | Best feasible profit found, including original | Improvement |
| --- | ---: | ---: | ---: |
| 4 | 14.299151270448 | 14.299151270448 | 0 |
| 28 | 15.138494876054 | 15.138495116646 | 2.406e-7 |
| 41 | 15.323380503977 | 15.323380549607 | 4.563e-8 |

Fixed-allowance Gurobi price LPs found no improvement; objective differences
from the saved menus were below 2e-10. The exhaustive coarse grids found no
better point. Numerical-Jacobian SLSQP restarts found the very small gains
above. Full-region constraint violations of the improved endpoints were below
9e-14. Call 4's original full-region violation is 1.23e-10, within the stated
1e-7 numerical feasibility threshold.

The original first-order audit had classified call 28 as not verified. The
present objective comparisons show that this does not imply a substantial
objective loss. They do not establish KKT conditions or SOSC.

## Independent checks

1. **Conditional prices.** Fix the saved allowances, and solve the retained
   price problem as a Gurobi LP. No approximation of G is used.
2. **Allowance grids.** For r in {0.1, 0.03, 0.01}, each coordinate takes
   max(0, e-r max(1,e)), e, e+r max(1,e), with exact duplicates removed.
   Exhaust the Cartesian product of all ten coordinates and optimize both
   prices at every point. Separability permits precomputing the two five-mode
   valuation tables. The two-price LP is solved by its componentwise greatest
   feasible price pair; the winning pair at every scale is checked independently
   with Gurobi. There are 26,244, 59,049, and 39,366 combinations per scale for
   calls 4, 28, and 41, respectively: **373,977 combinations** over the nine
   grids, including repeated centers across scales. There are thousands of
   feasible noncentral points in each grid; full counts are in verification.json.
3. **Two SLSQP starts per NLP.** Start from the original return and the best
   noncentral feasible grid point. Noncentral means a normalized allowance
   displacement of at least r/2 at its grid scale. Use numerical constraint
   derivatives (eps=1e-6), maxiter=500, ftol=1e-11. Recovery and fallback are
   disabled. This bypasses arbitrary boundary subgradient selection, but does
   not turn the optimizer into a global solver. Initial and returned menus,
   native statuses, and independent feasibility checks are preserved.
4. **Gurobi LP bounds in explicit boxes.** Construct convexity-based outer
   relaxations in boxes around the original allowance vector. Prices remain
   unrestricted except for the original nonnegativity and price order.

## Objective bounds and their scope

The boxes use max(0,e-r max(1,e)) <= allowance <= e+r max(1,e).
The table gives U minus the original profit, where U is the optimized LP
relaxation objective, an upper bound for the original retained NLP **restricted
to that box**.

| Box radius r | Call 4 | Call 28 | Call 41 |
| --- | ---: | ---: | ---: |
| 0.1 | 0.203554224 | 0.061438789 | 0.058591884 |
| 0.01 | 0.002409849 | 0.000903635 | 0.000636553 |
| 0.001 | 0.000024547 | 0.000009273 | 0.000006443 |
| 0.0001 | 0.000000249 | 0.000000104 | 0.000000077 |

These are floating-point numerical bounds, not interval-arithmetic proofs.
The 1e-4 box changes a coordinate with allowance e=80 by at most 0.008;
coordinates below 1 change by at most 0.0001. Restart points can lie outside
these small boxes, so their profits are not substituted as box lower bounds.
The original return is the common reference point in every box.

For each convex valuation G and box corners v, introduce lambda >= 0,
sum lambda=1, and e=sum lambda*v. Jensen's inequality gives
G(e) <= sum lambda*G(v). Globally valid affine lower bounds come from fixed
feasible dual selections integrated over demand. In each retained constraint
C_b-C_a+G_ib-G_ia<=0, replace +G_ib by its lower envelope and -G_ia by the
negative corner upper bound. Every original feasible point in the box has a
feasible representation in this relaxed LP, so its optimum is an objective
upper bound. Shared lambda per bundle is valid because the same convex
combination yields Jensen upper bounds for every traveler.

Corner upper values are computed by integrating chords between actual
max-of-all-dual-vertices values at demand-support endpoints and envelope
breakpoints. Convexity in trip frequency ensures these chords dominate Phi,
even if the cached envelope omitted a tiny segment. Lower planes use integrated
feasible dual selections and require no differentiability. Lower constants
receive -1e-9 padding and corner upper values +1e-9. Gurobi LP feasibility and
optimality tolerances are 1e-9. No rounding or pruning of the original dual
vertices is used.

**There is no full-domain global certificate here.** The coarse grids do not
cover every point in their boxes; the LP bounds cover the stated boxes, but
not the entire original unbounded allowance domain. Positive local curvature,
KKT, SOSC and exact local optimality have not been established by this experiment.

## Reproduce

Run from the project root with the maas-bundle environment and a Gurobi license:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /opt/homebrew/Caskroom/miniforge/base/envs/maas-bundle/bin/python \
  scripts/verify-selected-nlps.py
```

Gurobi version: 13.0.3. The last full run took about 13.1 seconds after building
the cached problem. Full source menus, alternative menus, initial points,
grid counts, bounds and solver metrics are in verification.json.
