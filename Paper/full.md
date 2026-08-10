# MaaS bundle design

# immediate

# July 27, 2026

## 1 Model

## 1.1 Main ideas

• Finite number of user classes, while each class features a continuous trip frequency distribution, e.g., daily commuters vs flexible working schedules

• Travelers in each class make bundle choice based on some statistical metric of generalized path cost (e.g., expectation).

• Expected travel flows are constrained by bundle-specific link capacity constraints, while the shadow price of each constraint is included in the generalized path cost.

• Design variables of each bundle are the bundle price and maximum usage of each mode, $\mathrm { i . e . , }$ the set of link capacities.

• Expected outcomes: travelers with different trip frequency distributions and feasible paths vary in their bundle preferences. Hence, bundle design introduces a form of service personalization instead of simply price discrimination.

## 1.2 Notations

• Traveler class i, traveler class set I, traveler class number $I = | \mathcal { I } |$

• Path $k ,$ path set $\kappa ,$ path number $K = | { \cal { K } } |$ , (optimal) path flow $f \left( f ^ { \ast } \right)$ , (min) path cost $c \left( \pi ^ { * } \right)$

• Link-path incidence matrix ∆

• Link $^ { a , }$ link set A, link number $A = | { \mathcal { A } } |$ , (optimal) link flow $x \left( x ^ { * } \right)$ , link cost t

• Bundle-specific link capacity ε and shadow price ω

• Bundle $b ,$ bundle set $B ,$ bundle number $B = | \boldsymbol { B } |$ , bundle price $C ,$ (min) generalized bundle cost e (Π<sup>∗</sup>)

• Path choice probability p, (optimal) bundle choice probability $q \ ( q ^ { * } )$

• Value of time $\beta$

• Bundle horizon T, total demand Q

## 1.3 Route choice with soft constraints

Suppose travelers of class $i \in I$ share a trip-frequency distribution $F _ { i } .$ For a given trip-frequency realization $\mu \sim F _ { i }$ and bundle $b \in B ,$ , let $P _ { i b }$ denote the set of feasible paths. Define

• path choice probability vector $p _ { i b } ( \mu ) \in \mathbb { R } _ { + } ^ { | P _ { i b } | }$ with

$$
\mathbb {1} ^ {\top} p _ {i b} (\mu) = 1,
$$

• path usage vector

$$
f _ {i b} (\mu) = \mu p _ {i b} (\mu),
$$

• link usage vector

$$
x _ {i b} (\mu) = \Delta f _ {i b} (\mu),
$$

• mode usage vector

$$
M _ {i b} (\mu) = N _ {i b} f _ {i b} (\mu),
$$

where $N _ { i b } \in \mathbb { R } _ { + } ^ { M \times | P _ { i b } | }$ is the mode-usage matrix, and its $( m , r )$ -entry records the number of times mode m is used on path $r ,$

• excess-usage vector

$$
z _ {i b} (\mu) \in \mathbb {R} _ {+} ^ {M},
$$

where $z _ { i b , m } ( \mu )$ denotes the number of pay-as-you-go trips exceeding the bundle allowance on mode m.

Let $t \in \mathbb { R } _ { + } ^ { | A | }$ be the vector of link travel times. Let $ { \varepsilon } _ { b } \in \mathbb { R } _ { + } ^ { M }$ denote the allowance vector of bundle $b ,$ and let $\rho _ { b } \in \mathbb { R } _ { + } ^ { M }$ denote the pay-as-you-go tariff vector, where $\rho _ { b , m }$ is the unit charge for excess usage of mode m.

Define the baseline path travel time vector

$$
\bar {c} _ {i b} := \Delta^ {\top} t \in \mathbb {R} _ {+} ^ {| P _ {i b} |}.
$$

For a given bundle b and realization $\mu ,$ the lower-level route-choice problem with fixed link travel time is

$$
\Phi_ {i b} (\mu ; \varepsilon_ {b}, \rho_ {b}) := \min _ {f \in \mathbb {R} _ {+} ^ {| P _ {i b} |}, z \in \mathbb {R} _ {+} ^ {M}} \beta   \bar {c} _ {i b} ^ {\top} f + \rho_ {b} ^ {\top} z\tag{1}
$$

$$
\mathrm{s.t.} \quad \mathbb {1} ^ {\top} f = \mu ,\tag{2}
$$

$$
N _ {i b} f \leq \varepsilon_ {b} + z.\tag{3}
$$

where $\beta > 0$ is the weight placed on expected travel expenditure.

Problem (1)–(3) minimizes the total realized travel cost, consisting of baseline travel time cost and pay-as-you-go charges for excess mode usage.

Let $\overline { { \omega _ { i b } ( \mu ) } } \in \mathbb { R } _ { + } ^ { M }$ be the shadow price vector associated with (3), and let $\pi _ { i b } ( \mu ) \in \mathbb { R }$ be the dual variable associated with (2). The generalized path-cost vector is

$$
c _ {i b} (\mu) = \beta \bar {c} _ {i b} + N _ {i b} ^ {\top} \omega_ {i b} (\mu).\tag{4}
$$

Accordingly, the lower-level equilibrium for a given bundle b and realization $\mu$ can be characterized by the KKT system: find $( f _ { i b } ^ { * } ( \mu ) , z _ { i b } ^ { * } ( \bar { \mu } ) , \omega _ { i b } ^ { * } ( \mu ) , \pi _ { i b } ^ { * } ( \mu ) )$ such that

$$
0 \leq \omega_ {i b} ^ {*} (\mu) \perp \varepsilon_ {b} + z _ {i b} ^ {*} (\mu) - N _ {i b} f _ {i b} ^ {*} (\mu) \geq 0,\tag{5}
$$

$$
0 \leq f _ {i b} ^ {*} (\mu) \perp \beta \bar {c} _ {i b} + N _ {i b} ^ {\top} \omega_ {i b} ^ {*} (\mu) - \pi_ {i b} ^ {*} (\mu) \mathbb {1} \geq 0,\tag{6}
$$

$$
0 \leq z _ {i b} ^ {*} (\mu) \perp \rho_ {b} - \omega_ {i b} ^ {*} (\mu) \geq 0,\tag{7}
$$

subject to

$$
\mathbb {1} ^ {\top} f _ {i b} ^ {*} (\mu) = \mu .\tag{8}
$$

RY:Or just write everything as complementarity

$$
\pi_ {i b} ^ {*} (\mu) \perp \mu - \mathbb {1} ^ {\top} f _ {i b} ^ {*} (\mu) = 0\tag{9}
$$

Here $\pi _ { i b } ^ { * } ( \mu )$ denotes the marginal generalized path cost associated with the realized trip frequency µ under bundle b. Condition (7) implies that the shadow price of any mode allowance is capped by its pay-as-you-go tariff. In particular, whenever excess usage of mode m is strictly positive, the corresponding shadow price equals the tariff, i.e., $\omega _ { i b , m } ^ { * } ( \mu ) = \rho _ { b , m }$

JL: I think soft capacity is good because it is compatible with PAYG, or with hard capacity the PAYG passenger (or passengers that have more trips) will be forbidden to travel unless you set a different route choice rule.

RY:To me, this can be considered as hard bundle usage constraint. Since the routing objective is linear, this is separable: one part corresponds to only with bundle and another for PAYG. To see this, notice that the objective can be written as

$$
\Phi_ {i b} (\mu ; \varepsilon_ {b}, \rho_ {b}) := \min _ {f \in \mathbb {R} _ {+} ^ {| P _ {i b} |}, z \in \mathbb {R} _ {+} ^ {M}} \quad \beta   \bar {c} _ {i b} ^ {\top} (f - z) + [ \beta   \bar {c} _ {i b} ^ {\top} + \rho_ {b} ^ {\top} ] z,\tag{10}
$$

where $f - z$ is the bundle usage and it is alway $\dot { \mathbf { \rho } } \leq \varepsilon _ { b }$

KZ:An different interpretation of the same problem and decision variable: $f - z$ is the bundle trip flow and z is the PAYG trip flow.

## 1.4 Bundle choice

At the upper level, travelers choose among bundles on the basis of the expected realized travel expenditure induced by the lower-level route-choice problem RY:(Self notes: for now, we use expectations, but later we can incorporate measures other than expectation (risk neutral)). For example, different traveler classes can have different risk preferences.. Under the pay-as-you-go scheme, the lower-level problem yields the realized total cost function $\Phi _ { i b } \big ( \mu ; \varepsilon _ { b } , \rho _ { b } \big )$ for traveler class i under bundle b and trip-frequency realization $\mu .$ Accordingly, for traveler class $i \in I$ and bundle $b \in B .$ , define the generalized bundle cost as

$$
e _ {i b} = C _ {b} + \mathbb {E} _ {\mu \sim F _ {i}} [ \Phi_ {i b} (\mu ; \varepsilon_ {b}, \rho_ {b}) ],\tag{11}
$$

where $C _ { b }$ is the bundle price.

JL:Here I change the shortest path cost to the cumulated path cost, because the user should compare the whole trip cost over the bundle cycle rather than the cost of one single trip.

RY:Side note: it seems Φ is convex in $\mu .$ This can become handy when we try to optimize the whole bundle choice problem. We might be able to exploit convex hull to identify the vertex and solve it. JL:

## For hard constraints version

At the same time, if we really want a hard constraint at the route level, I would suggest using the idea of chance constraints (Das and Lee (2021)), then we may get a different interpretation of traveler behavior.

To be specific, the bundle choice can follow a rule like

$$
\tilde {e} _ {i b} = \left\{ \begin{array}{l l} C _ {b} + \beta   \Gamma_ {i b}, & \text { if } R _ {i b} (\varepsilon_ {b}) \geq \alpha_ {i}, \\ + \infty , & \text { otherwise }. \end{array} \right.
$$

where $R _ { i b } ( \varepsilon _ { b } )$ is the coverage probability of bundle b for class i:

$$
R _ {i b} (\varepsilon_ {b}) := \mathbb {P} _ {\mu \sim F _ {i}} \left(\exists f \geq 0: \mathbf {1} ^ {\top} f = \mu , N _ {i b} f \leq \varepsilon_ {b}\right).
$$

Here $\Gamma _ { i b }$ is a representative route-side cost metric. For example, it can be taken as the minimum feasible route cost (cumulated) evaluated at the class-specific quantile demand level

$$
\mu_ {i} ^ {\alpha} := F _ {i} ^ {- 1} (\alpha_ {i}).
$$

This has some spirit of robustness mentioned by Prof. Zhang, although it is not standard robust optimization. The interpretation is that travelers choose bundles that have a high enough probability of covering their trips. In this way, the hard constraint is reasonable and we may avoid working with the full distribution explicitly in the bundle-choice metric.

Let $q _ { i b }$ denote the probability that a traveler of class i chooses bundle $b ,$ and let $\Pi _ { i } ^ { * }$ denote the minimum generalized bundle cost attained by class i. Then the bundle-choice equilibrium is characterized by

$$
q _ {i} \geq 0, \qquad \mathbb {1} ^ {\top} q _ {i} = 1, \qquad \forall i \in I,\tag{12}
$$

and

$$
0 \leq q _ {i b} ^ {*} \perp e _ {i b} - \Pi_ {i} ^ {*} \geq 0, \quad \forall i \in I, b \in \mathcal {B}.\tag{13}
$$

Condition (13) implies that a bundle can be chosen with positive probability only if it attains the minimum generalized bundle cost for the corresponding traveler class. Conversely, any bundle with strictly higher generalized cost is assigned zero choice probability.

Together, (11)–(13) define the upper-level bundle-choice equilibrium induced by the lower-level payas-you-go route-choice model.

## 1.5 Flow-independent case

RY:I would suggest first to focus on the flow-independent case. If we can solve this efficiently, and at scale, we can extend this to equilibrium or bundle design. See some initial ideas below:

For each class of the traveler i, we assume they solve the following choice problem: (or maybe make it a logit will have better properties?)

$$
\min _ {q _ {i} \in \Delta} q _ {i} ^ {\top} [ C _ {b} + \mathbb {E} _ {\mu \sim F _ {i}} [ \Phi_ {i} (\mu ; \varepsilon_ {b}, \rho) ] ],\tag{14}
$$

where ∆ denote the probability simplex.

Observe that the objective is linear in $q .$ Hence, the optimal choice occurs at a vertex of ∆ (one of the bundles). This means, we need only to evaluate the terms in the bracket for each bundle in parallel.

We now need to deal with $\Phi _ { i b } \big ( \mu ; \varepsilon _ { b } , \rho _ { b } \big )$ over a continuous distribution $\mu \sim F$ . Recall this is a parametric LP, the issue now is that $\mu$ enters into the constraint. If it is in the objective, we can simply apply convex hull and check all vertex of the feasible region easily. One trick we used (for another problem I worked on, and perhaps also well-known in outer approximation literature) is to use the dual LP (and strong duality holds trivially). In our case, this becomes:

$$
\Phi_ {i} (\mu ; \varepsilon , \rho) = \max _ {\pi , \omega} \pi \mu - \omega^ {\top} \varepsilon\tag{15}
$$

$$
s. t. \pi \mathbf {1} - N _ {i} ^ {\top} \omega \leq \beta \bar {c} _ {i}\tag{16}
$$

$$
0 \leq \omega \leq \rho .\tag{17}
$$

$$
\pi \geq 0\tag{18}
$$

Importantly, the feasible region is a polytope that is independent of $\mu$ (we will need to show $\pi \geq 0$ without loss of generality; it should, so that it is bounded and form a polytope)! It is exactly the H-representation of a convex hull. This means we can run convex hull algorithm (e.g., we used cdd) to transforxm the H-representation of the feasible region into V-representation (all the vertex), namely $( \pi _ { j } , \omega _ { j } ) _ { j }$ . Then, the optimal value is attained at one of the vertex (may have tie, but it is fine to break tie arbitrarily); so we just check over all vertex

$$
\Phi_ {i} (\mu ; \varepsilon_ {b}, \rho) = \max _ {j = 1, \dots} (\pi_ {j} \mu - \omega_ {j} \varepsilon_ {b}).\tag{19}
$$

This requires only a simple sorting for any $\mu ,$ and the vertex needs only computed once.

For each vertex $j ,$ the function $\left( \pi _ { j } \mu - \omega _ { j } \varepsilon _ { b } \right)$ is affine in $\mu , { \mathrm { s o } } , \Phi _ { i } ( \mu ; \varepsilon _ { b } , \rho _ { b } )$ as the max over a set of affine functions is piece-wise linear in $\mu ;$ and also convex.

Then, the expectaction $\mathbb { E } _ { \mu \sim F _ { i } } [ \Phi _ { i } ( \mu ; \varepsilon _ { b } , \rho ) ]$ is the integral over a piece-wise linear function, which is also convex (expectation of convex function). This means, we need only to find the break points, and do integral for each segment. But the break points are exactly when the optimal value is equal for both vertex. (Similar to your idea in the bundle paper).

Recall that ω are the shadow prices of the capacity $\varepsilon _ { b }$ .

## 1.6 Network loading

Suppose each traveler class is sufficiently large, then we can use the expected path flow to represent the traffic flow aggregated over a bundle horizon. The traffic averaged on each day is thus

$$
x = \frac {Q}{T} \sum_ {i \in \mathcal {I}} \sum_ {b \in \mathcal {B}} q _ {i b} \mathbb {E} _ {\mu} [ \Delta f _ {i b} ^ {*} (\mu) ]\tag{20}
$$

JL: If congestion and mode-specific capacity constraints are considered, a path-based formulation becomes necessary, but the resulting path-flow solution is generally not unique. If link flows are defined as the expectation over a stochastic distribution of such non-unique path flows, are the resulting link flows unique, and how can they be computed?

## 1.7 Equilibrium

JL:Should the goal of this section be a coupled equilibrium definition in its original form, or a further reduced single-object formulation such as a VI/fixed-point representation? What is the minimum level of theoretical properties that needs to be established before moving on to the bundle design problem?

## 1.8 Bundle design

Assume that the subscription bundles are indexed in increasing order of service tier, i.e., $b = 1 , \dots , B ,$ where a larger index corresponds to a weakly higher-tier bundle. We also introduce a pay-as-you-go option, indexed by $b = 0$ , as the outside option of the subscription menu. Let

$$
\mathcal {B} _ {0} := \{0 \} \cup \mathcal {B}.
$$

The pay-as-you-go option has no subscription fee and no included allowance:

$$
C _ {0} = 0, \qquad \varepsilon_ {0} = 0.
$$

It provides a benchmark against which travelers compare the subscription bundles. In this way, travelers are not forced to subscribe to a bundle.

Traveler class i chooses an option with the minimum generalized cost; if multiple options attain the same minimum cost, the highest-index option is selected. Formally,

$$
\mathcal {B} _ {i} ^ {*} (C, \varepsilon) := \arg \min _ {b \in \mathcal {B} _ {0}} e _ {i b} (C, \varepsilon),\tag{21}
$$

$$
b _ {i} ^ {*} (C, \varepsilon) := \max \mathcal {B} _ {i} ^ {*} (C, \varepsilon).\tag{22}
$$

Define the induced choice indicator

$$
y _ {i b} (C, \varepsilon) = \left\{ \begin{array}{l l} 1, & \text { if } b = b _ {i} ^ {*} (C, \varepsilon), \\ 0, & \text { otherwise }, \end{array} \right. \qquad i \in \mathcal {I},   b \in \mathcal {B} _ {0}.\tag{23}
$$

The platform earns revenue only from subscription fees. The expected subscription profit is

$$
\Pi (C, \varepsilon) = \sum_ {i \in \mathcal {I}} Q _ {i} \sum_ {b \in \mathcal {B}} y _ {i b} (C, \varepsilon) \left(C _ {b} - k ^ {\top} \varepsilon_ {b}\right).\tag{24}
$$

The bundle design problem is

$$
\max _ {C, \varepsilon} \Pi (C, \varepsilon)\tag{25}
$$

$$
\begin{array}{l l} \text {s.t.} & C _ {b} \geq 0, \quad \varepsilon_ {b} \geq 0, \end{array} \qquad \qquad b \in \mathcal {B},\tag{26}
$$

$$
C _ {b + 1} \geq C _ {b}, \quad b = 1, \dots , B - 1,\tag{27}
$$

$$
\varepsilon_ {b + 1, m} \geq \varepsilon_ {b, m}, \quad b = 1, \ldots , B - 1, m = 1, \ldots , M,\tag{28}
$$

$$
C _ {0} = 0, \qquad \varepsilon_ {0} = 0.\tag{29}
$$

The monotonicity constraints impose a weak tier structure on the subscription menu: a higher-tier bundle has a weakly higher subscription price and weakly larger allowances in every mode. The pay-as-you-go option plays the role of an outside option and prevents the platform from increasing all subscription prices without changing travelers’ choices.

## Bundle choice with logit

Here we treat the zero-allowance bundle as the outside option, so there is no need to introduce an additional alternative outside B.

For each traveler class $i \in \mathcal { T }$ , define the generalized bundle cost of bundle $b \in B$ by

$$
e _ {i b} = C _ {b} + \mathbb {E} _ {\mu \sim F _ {i}} [ \Phi_ {i b} (\mu ; \varepsilon_ {b}, \rho_ {b}) ].\tag{30}
$$

Instead of the deterministic minimum-cost rule, we consider an entropy-regularized bundle-choice problem:

$$
\min _ {q _ {i} \in \Delta (\mathcal {B})} \quad \sum_ {b \in \mathcal {B}} e _ {i b} q _ {i b} + \frac {1}{\theta_ {i}} \sum_ {b \in \mathcal {B}} q _ {i b} \ln q _ {i b},\tag{31}
$$

where

$$
\Delta (\mathcal {B}) := \left\{q _ {i} \geq 0: \sum_ {b \in \mathcal {B}} q _ {i b} = 1 \right\},
$$

and $\theta _ { i } > 0$ is the bundle-choice sensitivity parameter.

The first-order condition yields the logit bundle shares

$$
q _ {i b} = \frac {\exp (- \theta_ {i} e _ {i b})}{\sum_ {b ^ {\prime} \in \mathcal {B}} \exp (- \theta_ {i} e _ {i b ^ {\prime}})}, \quad \forall i \in \mathcal {I}, b \in \mathcal {B}.\tag{32}
$$

So the lower-level route-choice LP is unchanged for each bundle, and the bundle-choice layer is smoothed by the entropy term. In this way, bundle choice becomes unique and no separate tie-breaking rule is needed.

The bundle design problem with logit bundle choice is therefore

$$
\max _ {C, \varepsilon} \quad \Pi (C, \varepsilon)\tag{33}
$$

$$
\text { s.t. } \quad e _ {i b} = C _ {b} + \mathbb {E} _ {\mu \sim F _ {i}} [ \Phi_ {i b} (\mu ; \varepsilon_ {b}, \rho_ {b}) ]  , \qquad \forall i \in \mathcal {I},   b \in \mathcal {B},\tag{34}
$$

$$
q _ {i b} = \frac {\exp (- \theta_ {i} e _ {i b})}{\sum_ {b ^ {\prime} \in \mathcal {B}} \exp (- \theta_ {i} e _ {i b ^ {\prime}})}, \quad \forall i \in \mathcal {I}, b \in \mathcal {B},\tag{35}
$$

$$
C _ {b} \geq 0, \quad \varepsilon_ {b} \geq 0, \quad \forall b \in \mathcal {B},\tag{36}
$$

$$
C _ {b + 1} \geq C _ {b}, \quad \forall b = 1, \dots , B - 1,\tag{37}
$$

$$
\varepsilon_ {b + 1, m} \geq \varepsilon_ {b, m}, \quad \forall b = 1, \dots , B - 1, \forall m = 1, \dots , M.\tag{38}
$$

## 1.9 Solution Algorithm

## For lexicographic version

## 1.9.1 Branch-and-Bound over Allowance Menus with Region-Enumeration Pricing

For the lexicographic version, we discretize each allowance component and restrict

$$
\varepsilon_ {b, m} \in \mathcal {E} _ {m}, \quad \forall b \in \mathcal {B}, m = 1, \ldots , M,
$$

where each ${ \mathcal { E } } _ { m }$ is a finite candidate set. Together with the monotonicity constraints (27)–(28), the design problem (25) becomes a finite search over feasible allowance menus.

For a fixed allowance menu ε, the route-choice problem does not depend on the subscription price vector C. Hence, for each traveler class $i \in \mathcal { Z }$ and option $b \in B _ { 0 }$ , define

$$
\bar {\Phi} _ {i b} ^ {\varepsilon} := \mathbb {E} _ {\mu \sim F _ {i}} [ \Phi_ {i b} (\mu ; \varepsilon_ {b}, \rho_ {b}) ].
$$

This quantity is a constant in the pricing problem. By (11),

$$
e _ {i b} (C; \varepsilon) = C _ {b} + \bar {\Phi} _ {i b} ^ {\varepsilon}, \qquad i \in \mathcal {I}, b \in \mathcal {B} _ {0}.
$$

Hence, the generalized bundle cost is affine in the price vector C. Under the lexicographic choice rule (21)–(22), traveler class i switches between two options b and $b ^ { \prime }$ only when

$$
e _ {i b} (C; \varepsilon) = e _ {i b ^ {\prime}} (C; \varepsilon) \iff C _ {b} - C _ {b ^ {\prime}} = \bar {\Phi} _ {i b ^ {\prime}} ^ {\varepsilon} - \bar {\Phi} _ {i b} ^ {\varepsilon}.
$$

Therefore, all pairwise bundle-indifference conditions induce a finite collection of hyperplanes in the feasible price region. These hyperplanes partition the price space into polyhedral faces, and on the relative interior of each face the bundle-selection indicator $y _ { i b } ( C , \varepsilon )$ in (23) is fixed.

Once the bundle assignment is fixed, say $b _ { i } ^ { * } ( C , \varepsilon ) = \alpha _ { i }$ on a given face, the profit function (24) reduces to a linear function of C. The corresponding regional pricing problem can be written, in its closed form, as

$$
\begin{array}{l l l} V _ {\alpha} (\varepsilon) := \max _ {C} & \sum_ {i \in \mathcal {I}: \alpha_ {i} \in \mathcal {B}} Q _ {i} \left(C _ {\alpha_ {i}} - k ^ {\top} \varepsilon_ {\alpha_ {i}}\right) \\ \text {s.t.} & C _ {\alpha_ {i}} + \bar {\Phi} _ {i \alpha_ {i}} ^ {\varepsilon} \leq C _ {b} + \bar {\Phi} _ {i b} ^ {\varepsilon}, & i \in \mathcal {I},   b \in \mathcal {B} _ {0}, \\ & C _ {b} \geq 0, & b \in \mathcal {B}, \\ & C _ {b + 1} \geq C _ {b}, & b = 1, \ldots , B - 1, \\ & C _ {0} = 0. \end{array}\tag{39}
$$

Thus, for each supportable assignment pattern α, the regional pricing problem is a linear program.

For the lexicographic rule, the strict assignment region is obtained by allowing ties with lower-index options but not with higher-index options. Equivalently, after solving the closed LP (39), the induced assignment is verified using the tie-breaking rule in (22). Candidate solutions that do not induce the prescribed assignment are assigned to the corresponding lexicographic face.

This yields an exact pricing oracle for any fixed allowance menu: enumerate all relevant faces of the induced price partition, or equivalently all supportable bundle-assignment patterns $\alpha ,$ solve (39) for each of them, verify lexicographic consistency when needed, and select the best feasible solution. Since each fixed-menu pricing problem is solved globally in this way, the overall discretized lexicographic design problem can be solved globally by combining the pricing oracle with either exhaustive menu enumeration or branch-and-bound over feasible menus.

The role of branch-and-bound is to avoid enumerating all allowance menus explicitly. Each node of the search tree corresponds to a partially specified allowance menu satisfying the monotonicity conditions in (28). Bounds are computed from valid relaxations of the corresponding pricing value, while a complete menu is evaluated exactly by the fixed-menu pricing oracle above. Since the discretized menu space is finite and every complete menu can be evaluated globally, the resulting algorithm returns a globally optimal solution to the discretized lexicographic bundle design problem. JL:

## Ordered structure

If, in addition, traveler classes can be ordered by first-order stochastic dominance of trip-frequency distributions, and this ordering implies an ordered segmentation structure in bundle choice, then the inner pricing oracle can be accelerated further. In that case, it suffices to enumerate ordered assignment patterns rather than arbitrary ones, which can substantially reduce the number of candidate faces while preserving exactness under the ordering assumptions.

## Problem of tie-breaking

The regional LP analysis also reveals a technical issue with the current lexicographic tie-breaking rule. Although the rule makes bundle choice single-valued, some assignment regions become half-open. Hence, the supremum of the upper-level profit over one assignment region may be attained only on a boundary hyperplane, while that boundary is assigned to another assignment under the tie-breaking rule. As a result, the upper-level profit may fail to attain its supremum.

To ensure existence of an optimal solution without changing the tie-breaking rule, one may impose compactness of the menu-design set together with a profit-compatibility condition for the lexicographic rule, requiring that the bundle selected on any tie boundary yields a weakly larger platform margin than the other tied bundles.

Or we can replace the lexicographic rule with a platform-favorable tie-breaking rule. Under this rule, when a traveler class is indifferent among several minimum-cost options, the platform selects the one that yields the largest platform margin. This can be modeled by keeping the bundle-choice variables in the upper-level problem and imposing only the minimum-cost support condition:

$$
\max _ {C, \varepsilon , q, \Pi} \sum_ {i \in \mathcal {I}} Q _ {i} \sum_ {b \in \mathcal {B}} q _ {i b} \left(C _ {b} - k ^ {\top} \varepsilon_ {b}\right)\tag{40}
$$

$$
\begin{array}{l l} \text { s   .   t   . } & q _ {i} \in \Delta (\mathcal {B}), \end{array}
$$

$$
\forall i \in \mathcal {I},\tag{41}
$$

$$
e _ {i b} - \Pi_ {i} \geq 0,
$$

$$
\forall i \in \mathcal {I}, b \in \mathcal {B},\tag{42}
$$

$$
q _ {i b} \left(e _ {i b} - \Pi_ {i}\right) = 0,
$$

$$
\forall i \in \mathcal {I}, b \in \mathcal {B},\tag{43}
$$

$$
\text { menu   feasibility   constraints. }\tag{44}
$$

The complementarity condition ensures that a class can be assigned only to bundles attaining its minimum generalized cost. Since the assignment variables are chosen in the platform’s objective, any tie among minimum-cost bundles is resolved in favor of the bundle with the largest platform margin.

For a fixed allowance menu, this rule is also compatible with the regional LP oracle: one may enumerate class-to-bundle assignments and impose only that the assigned bundle is minimum-cost. Each candidate assignment then yields a linear pricing problem.

## 1.9.2 Alternating Optimization

## JL: For logit version

## Multi-Start Projected Gradient Method

## Epigraph Outer Approximation with Sequential Nonlinear Programming

## 1.10 Notes

• Although path and bundle choice probabilities are introduced, the optimality conditions are based on the simple setting of mixed strategies, without considering random utility.

• The statistical metric in bundle choice is set to expectation for the baseline case, while it’s more interesting to explore metrics reflecting robustness/reliability, e.g., ensuring that the bundle is sufficient to cover most possibly realized trips.

• Aggregate link flow to be double-checked.

• The equilibrium condition to be derived. Not sure if it corresponds to or can be transformed into any existing one.

## References

A. Das and G. M. Lee. A multi-objective stochastic solid transportation problem with the supply, demand, and conveyance capacity following the weibull distribution. Mathematics, 9(15), 2021. ISSN 2227-7390. doi: 10.3390/ math9151757. URL https://www.mdpi.com/2227-7390/9/15/1757.