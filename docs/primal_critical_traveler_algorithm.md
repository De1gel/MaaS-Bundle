# Optimistic Regional Search：算法与实现差距

本文档对应 [2026-09-24 最新修订稿](../Paper/MaaS_bundle_design.tex) 第 4.1–4.3 节：valuation oracle、乐观区域搜索、收敛与局部最优性。本次修订推广至一般供给成本，明确正利润初值与贯穿约束生成的 entry incumbent，并用 PAYG 利润上界替代紧性假设。当前代码在 assignment-consistency finder 上已加入正利润初值检查和外层 incumbent 保护，普通 tie / double tie 的逐边界往返搜索已加入，独立邻区最优性认证仍待实现；下文区分现有行为与完整算法要求。

论文保留 PBD 基于强对偶的单层非凸重构，并明确其适用于非线性供给成本；其原始—对偶约束见 [论文阅读版第 3.3 节](../Paper/full.md#33-两种选择规则的单层重构)。PBD 不使用本文件的 optimistic regional search，目前也没有对应求解入口。

## 1. 输入、输出与核心记号

设：

- \(i\in\mathcal I\)：旅行者类型，权重为 \(Q_i\)；
- \(b,a\in\mathcal B_0=\{0\}\cup\mathcal B\)：包含 PAYG 的选项，\(B=|\mathcal B|\) 为订阅套餐数；
- \(M\)：交通方式数；
- \(x=(C,\varepsilon)\)：菜单，\(C_b\) 为价格，\(\varepsilon_b\in\mathbb R^M\) 为额度；
- \(G_{ib}(\varepsilon_b)\)：旅行者对套餐的期望出行成本；
- \(e_{ib}=C_b+G_{ib}\)：感知总成本；
- \(M_b=C_b-K_b(\varepsilon_b)\)：平台单位利润。

所有选项共享外生超额费率 \(\rho\)。供给成本 \(K_b\) 非负、连续，\(K_b(0)=0\)，在使用导数的地方连续可微；PAYG 满足 \(C_0=0,\varepsilon_0=0,K_0(0)=0\)。当前代码通过 `ProvisionCost` 支持一般成本，保留共享线性参数作为兼容特例，见 [成本接口](./provision_cost.md)。

算法输出应包含：

- 最终菜单 \(x^*\)；
- optimistic assignment \(\alpha^*\)；
- 平台利润；
- regional NLP 与 separation 的诊断信息；
- assignment consistency、regional solve、separation 和 double-tie 诊断。

当前代码的算法级诊断状态如下；边界完成状态另由 `boundary_search_complete` 记录：

```text
CONSISTENT_CANDIDATE
NOT_ASSIGNMENT_CONSISTENT
SEPARATION_FAILED
REGIONAL_SOLVE_FAILED
DOUBLE_TIE
PROFIT_DECREASE
BOUNDARY_SEARCH_INCOMPLETE
```

## 2. 已有 valuation oracle 的职责

底层 oracle 对每个 traveler–bundle 组合提供：

\[
G_{ib}(\varepsilon_b)
\]

以及一个选定的梯度或次梯度

\[
g_{ib}(\varepsilon_b)=-\bar\omega_{ib}.
\]

已有对偶顶点和分段线性模块负责：

1. 使用 `pycddlib` double-description 将对偶约束的 H-representation 转为顶点，构造 oracle 时只做一次；
2. 按 \(\pi\) 排序候选直线；
3. 对给定额度构造 \(\Phi(\mu,\varepsilon)\) 的一维上包络；
4. 快速求值、积分并计算 \(G_{ib}\)；
5. 返回用于 NLP 的 \(-\bar\omega_{ib}\)。

对偶可行域包含 \(\pi\ge0\)、\(0\le\omega\le\rho\) 和路径约束，与额度、实现频率无关。顶点按 \(\pi\) 预排序；给定 \(J\) 个顶点和 \(M\) 个方式，截距更新为 \(O(JM)\)，包络扫描为 \(O(J)\)。期望计算使用分布提供的区间概率和未归一化一阶矩。

有限均值下，\(G\) 有限、凸、逐分量非增，并满足 \(|G(\varepsilon')-G(\varepsilon)|\le\rho^\top|\varepsilon'-\varepsilon|\)。新稿允许任意可测的最优影子价格选择产生次梯度，不要求最优对偶解处处唯一。

要注意：当多个对偶面同时最优时，\(G\) 可能不可微。此时 oracle 返回的是次梯度。若 NLP solver 强制要求光滑导数，必须采用平滑、bundle/SQP 型处理或显式记录非光滑点，不能把任意次梯度误称为经典梯度。

## 3. 精确 optimistic assignment

给定菜单，先计算全部 \(e_{ib}\)，再执行两级选择：

1. 找感知成本最小的套餐集合；
2. 在该集合中选择平台单位利润最大的套餐。

数值实现需要两个不同的容差：

- `choice_tolerance`：判断 \(e_{ib}\) 是否并列；
- `profit_tolerance`：判断 \(M_b\) 是否并列。

最终仍并列时必须使用确定性的索引规则，保证相同输入得到相同 assignment。

目标模型伪代码（实际批量 API 为 `problem.bundle_margins(menu)`，PAYG margin 固定为零）：

```python
def optimistic_assignment(menu, travelers, oracle, provision_cost):
    margins = [menu.price[b] - provision_cost(b, menu.allowance[b]) for b in bundles]
    assignment = empty_integer_array(len(travelers))

    for i in travelers:
        perceived = [
            menu.price[b] + oracle.expected_value(i, b, menu.allowance[b])
            for b in bundles
        ]
        cheapest = indices_within_min_tolerance(perceived)
        assignment[i] = deterministic_argmax(margins, among=cheapest)

    return assignment
```

## 4. Fixed-assignment region 与精确边界

固定 \(\alpha\)，令

\[
\mathcal I_b(\alpha)=\{i:\alpha_i=b\},
\qquad
D_b=\sum_{i\in\mathcal I_b}Q_i.
\]

定义 pairwise valuation difference

\[
\delta_{iba}(\varepsilon)
=G_{ia}(\varepsilon_a)-G_{ib}(\varepsilon_b).
\]

保持 \(i\) 选择 \(b\) 的约束为

\[
C_b-C_a\le\delta_{iba}(\varepsilon).
\]

对每个有序套餐对，全部约束可精确写成

\[
C_b-C_a\le
\Gamma_{ba}(\varepsilon;\alpha),
\qquad
\Gamma_{ba}
=\min_{i\in\mathcal I_b(\alpha)}\delta_{iba}.
\]

因此 critical traveler 不是预先固定的旅行者排序，而是当前 \((b,a,\varepsilon,\alpha)\) 下达到该最小值的人。若唯一 critical traveler 为 \(i^*\)，则

\[
\nabla_{\varepsilon_a}\Gamma_{ba}=-\bar\omega_{i^*a},
\qquad
\nabla_{\varepsilon_b}\Gamma_{ba}=+\bar\omega_{i^*b}.
\]

多个 traveler 同时 critical 时，\(\Gamma\) 不光滑。工程实现使用逐步加入原始 traveler constraints 的方式，比把 `min` 直接交给光滑 NLP solver 更稳健。

## 5. Restricted regional NLP

对每个 ordered pair \((b,a)\) 维护集合

\[
S_{ba}\subseteq\mathcal I_b(\alpha).
\]

受限问题为

\[
\max_{C,\varepsilon}
\sum_bD_b\bigl(C_b-K_b(\varepsilon_b)\bigr)
\]

subject to

\[
C_b-C_a+G_{ib}(\varepsilon_b)-G_{ia}(\varepsilon_a)\le0,
\qquad i\in S_{ba},
\]

以及论文中给出的菜单约束：\(C_b\ge0\)、\(\varepsilon_b\ge0\)、\(C_{b+1}\ge C_b\)，并固定 \(C_0=0,\varepsilon_0=0\)。当前实现不增加有限上界或额度单调性。

一般 \(K_b\) 下区域目标不再一定线性。在可微处，其最大化目标的导数为

\[
\frac{\partial\Pi_\alpha}{\partial C_b}=D_b,\qquad
\nabla_{\varepsilon_b}\Pi_\alpha=-D_b\nabla K_b(\varepsilon_b).
\]

当前 `regional.py` 已通过成本接口计算上述目标和梯度；assignment margin、初值利润评估及 incumbent 验收使用同一成本模型。`trust-constr` 对一般成本使用 BFGS 曲率近似，线性成本保留零目标 Hessian。choice constraints 与 valuation oracle 不受供给成本形式影响。

变量数只有 \(B(M+1)\)，与旅行者数量无关；旅行者数量主要影响 separation 扫描成本。

初始化 \(S_{ba}\) 时，对每个非空 \(\mathcal I_b\) 计算当前 slack，并选取一个或若干最小 slack 的旅行者。空 assignment group 不生成该 pair 的 choice constraints。按照本次 Algorithm 2，初始化发生在每次进入区域时；完整区域可行的 entry menu 在整个内层过程中保留，其利润是接受基准。加入约束后 restricted 最优值可以下降，不要求尚未完整可行的 successive trial objectives 单调上升。

regional adapter 的正常路径只调用一次 SLSQP。只有返回点未通过可行性/KKT
验收时才启动恢复链：可行但非驻点的返回点直接用于一次 SLSQP restart；不可行
返回点先沿已知可行锚点与返回点的线段回溯；restart 仍失败时，最后调用一次
`trust-constr`。fallback 使用约束拉格朗日项的数值 Hessian，只发生在异常路径，
因此不增加正常求解成本。所有尝试仍必须通过统一的一阶验收；非凸 NLP 不因
fallback 而获得数学上的全局或局部最优保证。

## 6. Full traveler separation

NLP 得到候选菜单 \(x^t\) 后，计算

\[
s_{iba}(x^t)
=G_{ia}(\varepsilon_a^t)-G_{ib}(\varepsilon_b^t)
-(C_b^t-C_a^t).
\]

约束可行当且仅当 \(s_{iba}\ge0\)。对每个 \((b,a)\)：

1. 扫描所有 \(i\in\mathcal I_b\)；
2. 找到最小 slack；
3. 若小于 `separation_tolerance` 的相反数，则加入最违反且尚未加入的 traveler；
4. 已有约束不删除；
5. warm start 重新求解 NLP。

固定 assignment 下，旅行者约束总数有限，critical sets 单调增加，每次未停止至少加入一个新约束，因此 loop 最多加入

\[
\sum_{b\in\mathcal B_0}|\mathcal I_b|(|\mathcal B_0|-1)=|\mathcal I|B
\]

条不同的 choice constraints（实际新增数需扣除初始化集合）。只要每次成功求解并继续时都加入新约束，就不会发生无限次约束添加。solver 失败、重复违反或达到工程迭代上限应单独报告；它们不属于定理中的成功终止。最终 restricted NLP 的局部极大点若对完整区域可行，也是完整区域的局部极大点，因为补回约束只缩小可行集。

## 7. Region transition：新版算法要求

### 7.1 普通乐观切换与 incumbent 保护

full separation 通过后，在 \(x^*\) 上重新计算 optimistic response。旧 assignment 在该点仍为最低成本响应，故 boundary-dominance 命题给出

\[
\Pi_{\alpha^+}(x^*)\ge\Pi_\alpha(x^*).
\]

若正权重旅客切换至严格更高 margin 的最低成本选项，利润严格增加。应同时更新新 assignment、重新计算 \(D_b\)、重建 assignment-dependent critical sets，并以 \(x^*\) 进入新区域。

新版还要求**保留可行 incumbent 的目标值**：若 \(x^*\in\mathcal R_{\alpha^+}\)，下一次 regional solve 接受的点不得比 \(\Pi_{\alpha^+}(x^*)\) 更差。warm start 本身不能保证这一性质；可行 incumbent 也不能仅因被保留就自动成为区域局部最优解。

工程实现需分别记录同菜单 reassignment 的利润变化、下一次区域求解相对于 incumbent 的变化，以及接受后的真实 optimistic profit。`choice_tolerance` 与 `profit_tolerance` 只定义数值近似，不能将近似 cost tie 无条件当作精确边界支配的证据。

### 7.2 Double-tie 相邻区域

在 \(x^*\) 上，若

\[
e_{ia}(x^*)=e_{ib}(x^*),\qquad M_a(x^*)=M_b(x^*),
\]

则更换该旅客的 assignment 不改变当前利润，且同一菜单对邻区仍可行。新版要求以 \(x^*\) 为 incumbent，对该邻区执行完整 regional solve（包括 constraint generation 和 full separation）：

- 找到严格改善：接受新菜单，继续区域搜索并重新评估 optimistic response。
- 已有证据表明 \(x^*\) 是该邻区的局部极大点：把这个邻区标为在 \(x^*\) 处 resolved。
- solver 失败、达到限制或没有找到改善但缺少局部最优证据：保留 unresolved，不能进入成功终止。

多个 double ties 同时出现时，需要检查它们生成的全部局部相邻 assignment 组合，而不只是逐旅客的单次切换。单个一般边界上的 double tie 只需一个额外邻区求解。

Resolved 状态必须绑定 incumbent 菜单与邻区 assignment；接受新菜单后，应重新建立相邻集合，不能沿用旧菜单处的认证。确定性索引 tie-break 只选出一个代表 assignment，不能替代邻区检查。

### 7.3 当前代码行为

`find_assignment_consistent_menu` 对普通 tie 遵循 optimistic rule；assignment 一致时额外
检查 double-tie 邻区。不展开普通 tie 的递归分支。菜单不同就进入新点继续优化，即使利润
相等、assignment 返回 A，也须从新点求解 A。同点且两侧区域检查完成才结束这次往返；
数值近点先在保留菜单复核邻区 full separation 和 KKT。新点的检查记录重新建立。

边界试探与 incumbent 分开保存，只额外检查 double-tie 邻区。试探目标比较自身的
entry objective；保留菜单的真实利润始终不降。求解失败或达到次数上限不算已完成检查。
内层仍保留 entry 可行锚点，未逐次对 restricted NLP 施加目标下限。

`solve_primal` 继续要求正利润初值。所有待检查边界耗尽后，`boundary_search_complete=True`，
已搜索的 double-tie 候选可通过 validation；缺少搜索记录的 double tie 仍报告 `DOUBLE_TIE`。
`terminal_iteration` 指向支持保留菜单的区域记录，避免使用最后一次试探的 KKT 记录。
完整参数与状态见 [边界搜索](./boundary_assignment_search.md)。这是数值搜索规则，
并不自动完成论文所要求的独立局部最优认证。

## 8. 收敛结论与工程终止

### 8.1 内层有限约束生成

固定 assignment、有限候选约束、每次添加新违反约束且不删除，保证只发生有限次添加。成功完成时全区域可行；局部最优性还依赖最后一次 restricted NLP 的局部极大性质。

### 8.2 外层利润单调收敛

最新第 4.3 节假设每个 regional NLP 求得局部极大点，目标不低于可行 region-entry incumbent；double-tie 邻区只能在 incumbent 的局部极大性得到认证后标为 resolved。各类型的 PAYG 成本 \(e_{i0}\) 有限。正利润初值已写入算法输入。

已接受利润满足 \(P^{k+1}\ge P^k\)。其有限上界由 outside option 给出：对 \(b=\alpha_i^k\)，完整 assignment 可行性及 \(G_{ib},K_b\ge0\) 推出

\[
C_b+G_{ib}(\varepsilon_b)\le e_{i0},\qquad
M_b=C_b-K_b(\varepsilon_b)\le C_b\le e_{i0},
\]

从而

\[
0<P^0\le P^k\le\sum_iQ_i e_{i0}<\infty.
\]

因此利润单调收敛且已接受迭代不会进入全 PAYG 区域，**不再需要菜单域紧性假设**。普通严格 margin 切换与产生改善的 double-tie transition 都使已接受利润严格增加。

上界适用于完整 choice constraints 可行的响应；约束尚未补齐的 restricted trial 可能不满足它。利润有界不意味着未被选择的套餐价格/额度有界，也不证明所有 regional NLP 都存在局部极大点。

这不是外层有限终止定理，也不保证菜单序列收敛。有限 assignment 数量不能单独推出有限终止，因为实际状态为 \((\alpha,x)\)，同一 assignment 可以对应不同菜单与局部解。旧 Numerical Results 中的 no-double-tie 聚点与有限切换讨论已从稿件删除。

### 8.3 当前实现的终止保护

- assignment 重复时默认记录 `revisited_assignment=True` 并继续。
- `stop_on_assignment_revisit=True` 只对未确认且无实质改善的同菜单重访作保守诊断；异点重访继续优化，已检查的同点往返处理剩余邻区。
- `max_region_iterations` 限制程序运行；达到限制时返回 `REGION_ITERATION_LIMIT`。
- 无需为了利润收敛额外增加菜单上界。虽然已加入外层不降利润检查，完整理论仍需要区域局部极大解和邻区认证等条件，不能将失败时保留 incumbent 解释为成功终止。

## 9. Resolved terminal point 与候选诊断

新版局部最优定理的终止条件是：当前点为自身区域的局部极大点、assignment 乐观一致，并且活跃 double ties 生成的所有局部邻区均已在同一点 resolved。严格更低 margin 的相邻 assignment 在足够小邻域内的利润低于终点值 \(\Pi_{\alpha^*}(x^*)\)，而不只是低于同一扰动菜单下的当前区域目标；同 margin 的邻区由额外求解处理。因此 no-double-tie 是无需额外邻区检查的特殊情况，不再是新版方法的唯一终止路径。

当前程序确认的内容较少：

1. full separation 无违反；
2. numerical optimistic reassignment 与区域 assignment 一致；
3. regional adapter 的可行性和一阶 KKT 验收通过；
4. numerical active ties 的 margin gap 与 double-tie 数量。

`CONSISTENT_CANDIDATE` 不等价于论文认证意义的 resolved terminal point。当前已完成逐 tie 往返搜索的 double tie 可通过数值候选检查；缺少搜索记录时仍返回 `DOUBLE_TIE`。一次往返或未找到改善不能直接视为独立二阶最优证明。

程序不计算 LICQ/MFCQ、Lagrangian Hessian 或 reduced-space SOSC。一阶 KKT 是必要条件检查，不能单独推出一般非凸 NLP 的局部极大性；论文的区域局部解假设与工程证据必须分别记录。

## 10. 推荐模块划分

```text
src/maas_bundle/
├── oracle/
│   ├── dual_polytope.py       # H-to-V 与对偶可行域
│   ├── piecewise_linear.py    # 一维上包络、求值和积分
│   ├── demand_distribution.py # F_i 的区间概率和一阶矩
│   └── route_value.py         # G 与 omega-bar 的高层接口
└── design/
    ├── menu.py                # 动态菜单快照
    ├── valuation.py           # G 与 omega-bar 批量表
    ├── assignment.py          # optimistic rule、结果和需求聚合
    ├── problem.py             # 静态问题数据与统一评估入口
    ├── boundary.py            # delta、slack、critical set 和 separation
    ├── regional.py            # regional NLP 与 solver adapter
    ├── primal.py              # generation、外层状态机和公共入口
    └── validation.py          # 已有状态汇总与 double-tie 诊断
```

`primal.py` 负责流程编排，不应重新实现 valuation、assignment 或 solver 细节。NLP solver 应通过 adapter 注入，以便在 IPOPT、KNITRO 或 SQP 后端间切换并进行单元测试。

完整的数据所有权、数组形状和依赖方向见 `docs/software_architecture.md`。

## 11. 新版目标流程伪代码（尚未完整实现）

下列名称描述目标算法，不是现有 Python API。`solve_region` 包含完整的内层约束生成；`preserve_incumbent` 表示只接受不劣于已有可行解的区域解，同时保留是否取得局部最优证据的独立状态。

```text
x ← 满足 Pi_opt(x) > 0 的初始可行菜单
alpha ← optimistic_assignment(x)

循环：
    初始化该 assignment 的 critical sets
    保留完整区域可行的 entry menu x 及其利润，贯穿整个 constraint generation
    result ← solve_region(alpha, warm_start=x, preserve_incumbent=x)
    若求解/完整 separation 失败：返回相应失败或未完成状态
    x_star ← result.menu
    alpha_plus ← optimistic_assignment(x_star)

    若 Pi(alpha_plus, x_star) > Pi(alpha, x_star)：
        alpha ← alpha_plus
        x ← x_star
        继续

    pending ← x_star 上的全部 double-tie 相邻 assignment 组合
    resolved ← 空集合（记录仅对当前 x_star 有效）
    improved ← false
    对 pending 中的每个 alpha_neighbor：
        neighbor ← solve_region(alpha_neighbor,
                                warm_start=x_star,
                                preserve_incumbent=x_star)
        若邻区完整可行，且利润严格改善：
            alpha ← optimistic_assignment(neighbor.menu)
            x ← neighbor.menu
            improved ← true
            退出邻区循环
        否则若有证据证明 x_star 在此邻区局部最优：
            将 alpha_neighbor 加入 resolved
        否则：
            返回 unresolved 诊断

    若 improved：继续
    若当前区域局部最优证据成立，且 pending 全部 resolved：
        返回满足论文终止条件的菜单与 optimistic assignment
    否则：返回未认证候选
```

现有 `solve_primal` 执行正利润初值检查、带 incumbent 保护的逐 tie 边界搜索和 candidate validation；其数值往返规则见第 7.3 节，不等于上述带独立局部最优认证的流程。具体职责见 [软件架构](./software_architecture.md)。

## 12. 测试计划

至少覆盖以下分支：

- 唯一最便宜套餐；
- 感知成本 tie 下的 optimistic profit tie-break；
- 完全 tie 下的确定性索引规则；
- 空 \(\mathcal I_b\)；
- 每个 pair 唯一与多个 critical travelers；
- separation 无违反、单个违反、多个 pair 同时违反；
- 重复违反不会重复加入约束；
- fixed-assignment loop 有限结束；
- reassignment 不变与发生变化；
- transition 后 critical sets 正确重建；
- strict tie margin 通过；
- 已完成往返搜索的 double tie 通过候选检查；未检查的 near double tie 报告 `DOUBLE_TIE`；
- 首次 solver/KKT failure 触发可行恢复、一次 SLSQP restart 和按需
  `trust-constr` fallback，全部失败后才由 fixed-assignment 阶段拒绝；
- 数值容差边界和不可微 valuation 点。

待补充：使用小型人工实例一次性加入全部 traveler constraints 解完整 regional NLP，与 constraint-generation 结果比较可行性、目标值和最终菜单。现有 81-assignment 穷举比较是不同 assignment 间的对照，不等价于这一约束生成对照。

### Regional NLP 专项测试矩阵

`regional.py` 不能只做代码分支覆盖，还必须按 critical constraints 数量和 solver 后端做参数化测试。

按 active constraint 规模至少覆盖：

- 0 条 choice constraint（仅菜单约束）；
- 每个 occupied ordered pair 1 条 critical constraint；
- 部分 pair 含多个 critical constraints；
- 全部 traveler constraints 激活；
- 空 assignment group 与多个 occupied bundles；
- traveler、bundle 和 mode 数量不同的若干小型实例。

按 solver 至少覆盖两个独立后端或算法。基础 CI 使用项目环境中稳定可用的 solver adapter；IPOPT、KNITRO 等可选后端使用相同的 contract test，并在依赖未安装时显式 skip。不同 solver 的测试比较：

- termination/status 是否被 adapter 正确映射；
- choice constraints、非负性、价格单调性和 PAYG equality 的最大违反量；
- regional objective 是否达到已知值或与基准解在容差内一致；
- 用户提供的 objective gradient 和 constraint Jacobian 是否通过有限差分检查；
- warm start 是否接受相同变量布局；
- 不要求多个 local NLP solver 返回逐元素完全相同的菜单。

`RegionalSolveResult` 统一记录以下 metrics：

```text
solver_name
status
solve_time_seconds
iteration_count
objective_value
maximum_constraint_violation
optimality_measure
first_order_certificate.normalized_stationarity_residual
first_order_certificate.satisfies_kkt
active_choice_constraint_count
variable_count
attempt_count
attempted_solvers
```

其中 `optimality_measure` 是 solver 原生诊断，独立 KKT residual 是 adapter 接受结果前执行的一阶必要条件检查。KKT 成立不能单独推出局部最优；一般 NLP 的二阶与正则性条件留在论文假设和 solver 责任中。测试终端报告遵循 pytest verbosity：`pytest -q` 不输出逐实例 metrics；普通或详细模式在测试结束时按 solver 和规模输出汇总表。计时使用单调高精度时钟，普通正确性测试只展示时间而不设置机器相关的通过阈值。需要定位整体测试耗时时，可额外运行 `pytest --durations=10`。

100/1000 条 critical constraints 的规模测试是显式 opt-in 套件：

```bash
pytest tests/design/test_regional.py --run-regional-scale \
  -k solver_scale_and_known_global_optimum
```

该套件使用具有可解析全局最优解的仿射 regional problem。终端的 `solution check=global+local` 表示数值解同时通过可行性、独立 KKT 检查并匹配解析全局最优解；`not verified` 表示不能声称局部最优。运行时间与内存回归仍应单列 benchmark，避免把机器相关的性能阈值混入普通单元测试。

## 13. 实现进度与后续顺序

1. 定义静态 `BundleDesignProblem`、动态 `BundleMenu` 和评估结果；（已完成）
2. 实现并测试精确 optimistic assignment；（已完成）
3. 实现 \(\delta\)、slack、critical-set 初始化和 separation；（已完成）
4. 定义菜单变量布局和 solver adapter，并接入 restricted regional NLP；（已完成）
5. 完成 fixed-assignment constraint-generation loop；（已完成基本分支及 10/1000-traveler 已知最优收敛测试）
6. 加入 assignment-consistency region transition，并分别记录 regional objective 与 true profit；（已完成）
7. 加入已有状态汇总与 tie-margin/double-tie 诊断；（已完成）
8. 最后组装公共入口 `solve_primal`。（已完成）

上述 1–8 项构成 consistency finder。新版扩展进度为：

9. 已实现实验正利润初值采样、`solve_primal` 初值检查、外层 incumbent 目标保护及接受记录；回归测试覆盖区域利润下降、真实利润下降、微小下降和后续求解失败。拒绝后保留可行解并报告未收敛，不伪造局部最优证据。
10. **已实现**逐 tie 构造普通/ double-tie 的另一侧 assignment，并测试多个 tie 的轮流尝试。多重交点的全部联合方向认证仍未实现。
11. 增加绑定当前菜单的 resolved/unresolved 记录；测试求解失败、只有 KKT、无改善但无证据均不能误标 resolved，菜单变化时旧记录失效。
12. **已实现**边界往返搜索及诊断，覆盖普通切换、double-tie 返回、改善、其他 tie、失败和迭代限制；独立二阶 resolved 认证仍未实现。
13. 独立增加 PBD 单层 NLP（以及需要时的直接 OBD MPEC）；用唯一响应及不同 margin 的成本 tie 验证两种规则，检查强对偶等式。不能只把 assignment 中的 `argmax` 改为 `argmin` 就视为完成 PBD 设计算法。
14. 在 PAYG 成本有限的实例上记录已接受利润序列，核对正利润下界与 PAYG 利润上界，再用 two-route/Sioux Falls 对照完整区域搜索；不要求为利润收敛额外施加紧性。既有实验数据不自动升级为新版算法结果。
15. **已实现**一般 `K_b` 的值/梯度接口，并验证线性兼容性、非线性 objective gradient、optimistic tie-break、初值筛选与利润验收，以及两种求解器的已知非线性最优解。网络规模的非线性成本实验仍待补充。


## 14. 实验初始化约定

正利润初值现已是 Algorithm 2 的明确前提。所有常规单次/多初值实验从实际 optimistic profit 严格为正的菜单开始。two-route 与 Sioux Falls 的随机生成器保持原有价格/额度采样范围和单调性，执行有上限的拒绝采样，直到取得请求数量的有效初值。默认阈值为 `1e-9`，最多尝试 `100 * starts` 次；找不到足够初值时明确报错，不补入 PAYG 菜单。

正利润条件根据完整实例的 valuation、assignment 和旅客权重判断，不能只检查价格高于零或至少一个旅客订阅。若所有订阅选项无人选择，或订阅者全部为零权重，平台利润为零，会被排除。未使用的部分套餐仍允许存在。

外层接受的利润满足 `P_next >= P_incumbent >= P_initial > 0`，因此返回的菜单不会落入全 PAYG 区域。该不变量覆盖求解失败后的保留结果，不要求 NLP 的每一个内部试探点都满足正利润。穷举基线和低层测试仍可检查全 PAYG 区域；它们不是常规实验初值。

拒绝采样改变了同一个随机种子所对应的初值序列，旧的 30-start 统计不能与新协议混用。
