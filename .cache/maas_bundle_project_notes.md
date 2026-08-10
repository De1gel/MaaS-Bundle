# MaaS Bundle Design Project Notes

Date: 2026-08-07

## 1. Project purpose

This repository is about MaaS bundle design: choosing subscription bundles with bundle prices and mode allowances so that travelers select among bundles and the platform maximizes subscription profit.

From the paper, the model has three layers:

1. Lower-level route choice for a realized trip frequency `mu` under a fixed bundle `b`.
2. Upper-level bundle choice based on expected generalized bundle cost.
3. Bundle design over prices and allowances, with optional network loading / equilibrium analysis.

The key economic idea is service personalization: different traveler classes have different trip-frequency distributions, feasible paths, and bundle preferences.

## 2. Core mathematical structure from the paper

### Lower-level route-choice oracle

For a traveler class `i`, bundle `b`, and frequency realization `mu`, the lower-level problem is a parametric LP of the form:

- minimize baseline travel cost plus PAYG excess charges
- subject to total path flow summing to `mu`
- subject to mode-usage capacity / allowance constraints

The paper defines the realized route-choice value as `Phi_ib(mu; epsilon_b, rho_b)`.

The associated shadow prices enter the generalized path cost. This is the main object that should be exposed by an oracle interface, not the raw LP implementation.

### Upper-level bundle choice

The generalized bundle cost is

`e_ib = C_b + E_{mu ~ F_i}[Phi_ib(mu; epsilon_b, rho_b)]`

This is the quantity used by the upper level. The subscription price `C_b` belongs here, not in the lower-level oracle.

The paper also discusses a logit version, but the same lower-level value oracle still sits underneath it.

### Bundle design

The platform chooses bundle prices and allowances to maximize expected subscription profit, subject to monotonicity constraints across bundle tiers.

The paper also notes possible extensions:

- hard-constraint / chance-constraint interpretation
- network loading with expected path flow aggregation
- equilibrium formulations
- dual / convex-hull / shortest-path style acceleration

## 3. Current work implied by the paper

The current implementation work should focus on a clean oracle stack:

1. Define fixed instance data for one class-bundle pair.
2. Define a query object for realization-specific parameters such as `mu`, allowances, and overage price.
3. Define a result object that returns the value and subgradient / shadow-price information.
4. Implement a primal oracle first.
5. Add an expectation wrapper over the frequency distribution.
6. Add dual / separation-based variants later without changing upper-level code.

This keeps the upper level independent of whether the lower level is solved by primal LP, dual LP, or a shortest-path separation method.

## 4. Repository status right now

The workspace is still at a very early stage:

- `README.md` is empty.
- `src/maas_bundle/` currently only contains an empty `__init__.py`.
- The package metadata says the project depends on `numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `PyYAML`, and `gurobipy`.
- The paper has been parsed into `Paper/full.md`, which is the best source for the model and notation.

So the main near-term work is not analysis of existing code, but building the package structure and the oracle interface that the paper motivates.

## 5. Practical implementation direction

Recommended first milestone:

- `src/maas_bundle/oracle/types.py`
- `src/maas_bundle/oracle/base.py`
- `src/maas_bundle/oracle/primal.py`

Then add:

- `src/maas_bundle/oracle/expectation.py`
- `src/maas_bundle/oracle/dual.py`

This matches the paper's decomposition and keeps the upper-level code stable.

## 6. Short takeaway

The paper's central computational object is an oracle for `Phi_ib(mu; epsilon_b, rho_b)` and its sensitivity with respect to allowances. Everything else should wrap around that oracle, not duplicate its logic.
