# Call 15: KKT acceptance and fallback experiment

The original assignment, 18 retained choice constraints, demand distributions,
costs, cached dual vertices, initial menu and SLSQP settings were restored from
the exact start-85 replay. Only the flags `check_kkt=True` and
`enable_fallback=True` differ from the original experiment. Production defaults,
the original saved trajectory and the manuscript were not modified.

Two runs were performed: one starts from the original NLP input, and one starts
from its previously accepted bad return. Both reach the same new menu through
the existing recovery policy. Each run has five SLSQP attempts followed by one
trust-constr fallback attempt.

| Check | Original accepted return | New fallback return |
| --- | ---: | ---: |
| Fixed-assignment profit | 17.800746854157 | 18.930786500397 |
| Retained-NLP primal violation | 0 | 0 |
| Normalized stationarity, active tolerance 1e-4 | 0.02803844035 | 2.5282477033e-7 |
| Absolute complementarity, active tolerance 1e-4 | 7.0467625773e-4 | 1.8157773187e-7 |
| Independent first-order screen | Fail | Pass |

The independent screen uses feasibility <=1e-7, normalized stationarity <=1e-5,
and absolute complementarity <=1e-6. Multipliers are nonnegative from the NNLS
fit. The adapter's built-in `satisfies_kkt` checks feasibility and stationarity;
the extra complementarity check is computed separately in this experiment.

The fallback policy internally uses active tolerance 1e-3. Its reported
stationarity residual is 2.8545882568e-8, and its native trust-constr optimality
measure is 1.4893488907e-9. Crucially, recomputing multipliers at the original
1e-4 active tolerance also passes, as shown above. Acceptance is therefore not
solely a consequence of the fallback's larger active tolerance.

## Actual recovery chain from the original NLP input

| Attempt | Backend | Native success | Adapter status | Profit | Stationarity residual |
| --- | --- | --- | --- | ---: | ---: |
| 1 | SLSQP | False | Infeasible | 17.839699154161 | 0.027999 |
| 2 | SLSQP | False | Infeasible | 17.780632431313 | 0.027891 |
| 3 | SLSQP | False | Infeasible | 17.785831525347 | 0.027868 |
| 4 | SLSQP | True | Failed first-order acceptance | 17.800746854157 | 0.028038 |
| 5 | SLSQP, numerical constraint Jacobian | True | Failed first-order acceptance | 17.856532275482 | 0.062536 |
| 6 | trust-constr | True | Success | 18.930786500397 | 2.8546e-8 |

The original bad menu is reproduced exactly at attempt 4 and rejected. Attempt
5 is also rejected despite native success. The existing trust-constr fallback
then returns a menu that passes both adapter acceptance and the independent
screen at active tolerance 1e-4. The first run took about 10.5 seconds including
per-attempt diagnostics; the second about 10.2 seconds.

Fresh interval integration and a HiGHS conditional price LP show a remaining
price improvement of only 2.2111e-7. The fallback profit is about 5.5205e-7 below
the previously found best feasible retained-NLP witness, 18.930787052444.

## Limits

This is successful numerical first-order recovery of the retained NLP, not a
classical SOSC or global-optimality certificate. At tighter activity thresholds
1e-6 and 1e-8, the fallback menu's stationarity residuals are 0.01286 and 0.18820:
barrier-returned slacks are not all below those thresholds. The configured
acceptance is explicitly at 1e-4, with complementarity checked independently.

The new menu's complete-region violation remains about 16.6835. This call is an
intermediate constraint-generation round; subsequent constraint separation is
still necessary. No claim is made that rerunning the whole regional path with
these flags would leave its trajectory unchanged, or that fallback always
succeeds on other NLPs.

## Reproduce

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /opt/homebrew/Caskroom/miniforge/base/envs/maas-bundle/bin/python \
  scripts/retry-nlp-with-kkt.py
```

`retry.json` preserves both initial points, every SLSQP attempt, the full
recovery-chain metrics, returned menus, independent price checks and
first-order sensitivity checks at 1e-8, 1e-6, 1e-4 and 1e-3.
