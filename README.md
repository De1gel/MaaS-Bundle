## Running the bundled example

The algorithm entry point is `maas_bundle.solve_primal(...)`. The installed
command-line entry point currently exposes a small end-to-end example:

```bash
maas-bundle example two-route
```

The example is derived from TAPLab's four-node
[CC0 two-route diagnostic network](https://github.com/asu-trans-ai-lab/TAPLab/tree/main/tapbench/diagnostic/two_route).
It runs network path construction, route-value oracles, regional NLP,
critical-traveler separation, optimistic reassignment, and candidate
validation, then prints a JSON result. Four traveler classes share the same
OD pair: two use continuous uniform trip-frequency distributions and two use
continuous triangular distributions. The warm start deliberately changes
assignment after its first regional solve, so the example exercises two outer
assignment-region iterations.

For this small instance, all traveler-to-bundle assignments can also be
enumerated and compared with the single primal run:

```bash
maas-bundle compare two-route
```

The enumeration reuses the production fixed-assignment solver. It is an
exponential diagnostic baseline and does not turn the local regional NLP
solver into a continuous global optimizer.

To test sensitivity to the initial menu, run the reproducible multi-start
benchmark:

```bash
maas-bundle multistart two-route --starts 30 --seed 20260901
```

It samples menus with nondecreasing prices and componentwise nondecreasing
allowances, retaining only those with optimistic profit above `1e-9`. All-PAYG,
zero-profit and loss-making proposals are resampled until the requested number
of starts is reached (at most 100 proposals per requested start). It runs the
primal pipeline from each accepted menu and groups consistent results by final
assignment. This diagnoses local attraction
regions; it is not a global-optimality proof.

The corresponding reproducibility test is opt-in:

```bash
pytest tests/benchmarks/test_two_route_multistart.py --run-multistart
```

The full 81-assignment comparison is also available as an opt-in test:

```bash
pytest tests/benchmarks/test_two_route_comparison.py --run-enumeration
```

## Running the larger Sioux Falls example

The repository also includes the standard 24-node, 76-directed-link Sioux
Falls network and its 360,600-trip OD table from
[Transportation Networks for Research](https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls).
The source network is unimodal; the two credit modes, two traveler segments
per positive OD pair, trip-frequency distributions, preferences, and MaaS
costs are explicitly synthetic extensions. The resulting instance has 528
positive OD pairs and 1,056 traveler classes.

The current configuration is `sioux-falls-monthly-usd-v1`: source travel times
are converted from 0.01 hour to minutes (multiply by 0.6), trip frequencies are
monthly, prices and normalized profit are USD/person/month, and allowances
count mode-specific passenger-link traversals per person per month. A link
traversal is not a whole OD trip or a boarding. All economic parameters are
synthetic. See the [complete parameters and units](docs/sioux_falls_experiment.md)
and [configuration object](src/maas_bundle/examples/sioux_falls_config.py).
CLI results include the configuration and unit metadata in `experiment`.

Every simple OD path contributes a constraint to the route-choice dual
polytope; there is no fixed path-count cutoff. All dual vertices are enumerated
once per OD/segment and cached within the process, shared by all three bundle
options and repeated starts. Positive edge costs and nonnegative mode usage
make cyclic routes unnecessary. Menu evaluations only update the cached
vertex envelope and integrate it against the demand distribution.

```bash
maas-bundle example sioux-falls
# The short alias requested for experiments also works:
maas-bundle example souxf
```

Ordinary ties follow optimistic reassignment. Equal-margin neighbors are checked
at the current menu, and a return at a different menu requires fresh regional
optimization. The full single-start regression requires successful completion;
the earlier 30-region limit exit was a boundary-search implementation error.
With the monthly configuration and complete path geometry, the default start
finishes in 5 regional solves (9 NLP calls), with profit about
`10.16182917 USD/person/month`. Earlier results `3.27138465` (complete paths,
old parameters) and `2.60624935` (five paths per OD) belong to superseded setups.

The complete 30-start experiment is substantially slower than the two-route
case and is invoked explicitly:

```bash
maas-bundle multistart souxf --starts 30 --seed 20260901
```

The earlier unfiltered 30-start run (about 590 seconds, 26 consistent starts,
25 validated starts, 20 final assignments) predates the positive-profit
initialization policy and is retained only as a historical observation. It
must not be compared as if it used the current sampling protocol.

The full single-start integration test is opt-in:

```bash
pytest tests/examples/test_sioux_falls.py --run-sioux-falls
```

## Positive-profit starts and incumbent protection

Experimental starts must have positive optimistic platform profit; with
`M_0 = 0`, this excludes all-PAYG flat regions. Merely having subscribers is
insufficient: an allowance-heavy menu can attract them at a loss. The public
`solve_primal` entry rejects nonpositive starts and requires profit above its
`profit_tolerance` (default `1e-9`). It also checks initial price ordering.

Cost ties and choice feasibility share a run-wide threshold (default `1e-7`)
from initialization onward. `margin_tolerance` (default `1e-9`) controls both
boundary exploration and final tie diagnostics, independently of aggregate
`profit_tolerance`. Price and allowance matching have separate absolute
tolerances. See [numerical tolerances](docs/numerical_tolerances.md).

After a successful fixed-assignment solve, the outer search checks both the
regional objective and recomputed optimistic profit against the current
incumbent in ordinary ascent. Boundary exploration instead compares each trial
against its own fixed-assignment entry objective while retaining the incumbent
separately. Equal-profit moves to different menus continue optimization. Solver or
separation failures retain that incumbent and report failure. Accepted profits
never decrease, so a positive-profit run cannot return an all-PAYG menu, even
on failure. Internal boundary trials may visit all-PAYG assignments.

The updated two-route 30-start run with seed `20260901` produced 30 validated
results, 14 final assignments, and no all-PAYG starts or returned menus. The
best profit was `4.03589`. These results use positive-profit sampling and the
new boundary search, so they differ from earlier experiments with the same seed.

Fixed-assignment enumeration and low-level diagnostic tests can still inspect
zero-profit regions. Positive-profit initialization does not require every
subscription bundle to be occupied, nor does it guarantee global optimality.

## Manuscript and implementation scope

The latest [manuscript](Paper/MaaS_bundle_design.tex) allows a nonnegative,
continuous per-subscriber provision cost `K_b(allowance)`, with `K_b(0) = 0`
and continuous differentiability where derivatives are used. The implementation
now supports `ProvisionCost` models and `CallableProvisionCost` callbacks with
analytic or numerical gradients. Legacy `unit_allowance_cost=k` remains supported
through `LinearProvisionCost`; the default experiments still use that special case.
See the [cost interface](docs/provision_cost.md) for examples and solver limitations.

The updated convergence proof bounds accepted profit by the finite quantity
`sum_i Q_i * e_i0`, using PAYG availability and nonnegative route/provision
costs. It no longer requires a compact menu domain. This is objective-value
convergence, not a guarantee of finite termination or convergence of menus.
Ordinary ties follow the optimistic rule; double ties generate extra neighbor
solves. Equal-profit moves to different menus continue optimization, including
returns to an earlier assignment. A same-point return completes a check only
when both regions have passed their checks at the retained menu. See [boundary search](docs/boundary_assignment_search.md).
This numerical stopping rule is not an independent local-optimality certificate.
Second-order neighbor certification and direct PBD optimization remain pending;
see the [architecture](docs/software_architecture.md) for implementation gaps.
