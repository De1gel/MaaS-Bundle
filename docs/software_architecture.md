# MaaS Bundle Design 软件架构

本文档对应 [2026-09-24 最新修订稿](../Paper/MaaS_bundle_design.tex)，记录当前代码的数据所有权和依赖，并列出新版 OBD/PBD 的扩展边界。现有 `solve_primal` 使用正利润初值检查和带 incumbent 保护的 optimistic assignment-consistency finder；一般供给成本 `K_b` 已接入；普通 tie / double tie 的逐边界往返搜索已加入，独立二阶邻区认证及第 3.3 节的 PBD 求解仍未实现。目标算法见 [区域搜索文档](./primal_critical_traveler_algorithm.md)。

| 论文部分 | 当前实现 | 新版待补充 |
|---|---|---|
| 2.4：一般供给成本 | 已实现批量 `ProvisionCost` 值/梯度接口，兼容线性参数 | 待补充网络规模非线性成本实验 |
| 3.1 / 4.1：固定对偶几何与 valuation oracle | `oracle/` 已实现 | 保持通用 oracle 供两种选择模型复用 |
| 3.3：OBD MPEC / PBD 强对偶非凸重构 | 尚无直接模型求解器 | 独立建模与求解入口 |
| 3.4 / 4.2：区域 NLP 与约束生成 | `boundary.py`、`regional.py`、`solve_fixed_assignment` | 复用已有模块 |
| 4.2：乐观区域搜索 | 已有 optimistic 切换、double-tie 邻区、同点/异点返回检查及 incumbent 保护 | 独立二阶认证、多重交点联合方向认证 |
| 4.3：resolved terminal point | 当前仅候选诊断和一阶 KKT | 与当前菜单绑定的邻区证据及终止判断 |

## 1. 设计原则

1. **静态问题与动态决策分离**：旅行者、分布、供给成本模型和 route oracle 在整个算法中不变；菜单和 assignment 随迭代变化。
2. **精确 valuation 与设计算法分离**：`oracle/` 不知道套餐 assignment 或 NLP；`design/` 只通过公开 oracle 接口取得 \(G\) 和 \(\bar\omega\)。
3. **数值表与业务结果分离**：单次 \(G\) 评估使用 `ValuationTable`；只有执行 optimistic rule 后才产生 `AssignmentResult`。
4. **数据对象不可变，迭代状态显式可变**：菜单、估值表和求解结果是只读快照；critical constraint pool 是算法内部明确管理的可变状态。
5. **公式用纯函数，外部求解器用 Protocol**：不为无状态计算建立服务类，也不通过继承绑定 IPOPT、KNITRO 或 SQP。
6. **数组轴全局统一**：traveler 轴为 \(i\)，bundle 轴为 \(b\)，mode 轴为 \(m\)；bundle `0` 始终是 PAYG outside option。

## 2. 已固定的核心对象

### `RouteValueOracle`

每个 \((i,b)\) 一个 oracle，保存该 traveler–bundle 路径问题的对偶顶点。它只负责：

- 构造 \(\Phi_{ib}(\mu,\varepsilon_b)\)；
- 计算 \(G_{ib}(\varepsilon_b)\)；
- 返回 \(\bar\omega_{ib}\) 和次梯度 \(-\bar\omega_{ib}\)。

它不保存旅行者权重、菜单价格、assignment 或平台成本。

`dual_polytope.py` 以 \((\pi,\omega)\) 排列对偶变量，包含路径约束、\(\pi\ge0\) 和 \(0\le\omega\le\rho\)，用 `pycddlib` double-description 枚举顶点。`RouteValueOracle` 在构造时排序并保存顶点；额度变化只更新截距、构造上包络和积分。\(J\) 个顶点、\(M\) 个方式下，截距更新为 \(O(JM)\)，包络扫描为 \(O(J)\)。

新稿模型采用所有套餐共享的外生严格正费率 \(\rho\)。底层 `build_dual_polytope` 接口仍接受非负费率，静态 problem 也未跨 oracle 强制验证共同费率；构造论文实验时需遵守模型设定，不能把更宽的通用接口当作论文改变了假设。

### `BundleDesignProblem`

当前代码中跨迭代复用的静态数据（成本模型参数在求解期间必须保持不变）：

```text
traveler_weights          shape (I,)
provision_cost            ProvisionCost: mode_count, values, gradients
route_oracles             logical shape (I, B0)
demand_distributions      length I
```

该对象在构造时一次性验证所有静态维度，并提供：

```python
problem.evaluate_valuations(menu)
problem.evaluate_assignment(menu)
```

regional NLP 使用前者；外层真实 optimistic reassignment 使用后者。

`design/cost.py` 提供独立成本值/梯度接口 `ProvisionCost`；`CallableProvisionCost` 接受一般函数，`LinearProvisionCost` 封装原共享线性参数。成本模型是静态数据，成本值随菜单更新；平台成本不进入路径 oracle 或 `ValuationTable`。

### `BundleMenu`

一次菜单决策快照：

```text
prices                    shape (B0,)
allowances                shape (B0, M)
```

其中第 0 行必须满足 \(C_0=0,\varepsilon_0=0\)。菜单只保存决策值，不拥有 traveler 或 solver 数据。

### `ValuationTable`

在某个 allowance menu 上批量计算得到：

```text
route_values              shape (I, B0)       # G_ib
expected_omega            shape (I, B0, M)    # omega-bar_ibm
```

它是 regional objective/constraint evaluation 与 assignment evaluation 共享的中间层，避免 NLP 为了取得 \(G\) 被迫执行 tie-breaking。

### `AssignmentResult`

在一个菜单上执行 optimistic rule 后的完整只读快照：

```text
menu
valuation
bundle_indices            shape (I,)          # alpha_i
generalized_costs         shape (I, B0)       # e_ib
bundle_margins            shape (B0,)         # M_b
bundle_demand             shape (B0,)         # D_b
total_profit
minimum_cost_gaps         shape (I, B0)       # e_ib - min_a e_ia
numerical_tie_mask        shape (I, B0)       # gap <= tau_choice
numerical_optimistic_mask shape (I, B0)       # cost tie and within tau_margin of best margin
optimistic_margin_gaps    shape (I,)           # selected margin - best tied competitor
```

`route_values` 和 `expected_omega` 通过 `valuation` 暴露，不再复制保存。assignment evaluator 显式保存成本与 margin 容差；低层 `profit_tolerance` 是 margin 容差的兼容名称。数值 optimistic 集合相对最低成本和最高 margin 统一构造，集合中多于一个选项即为 double tie，搜索与诊断复用同一集合。详见[数值容差](numerical_tolerances.md)。

## 3. 固定的依赖方向

箭头表示调用或依赖：

```text
cli.py → benchmarks/具体实验入口 → examples/数据与构造
                   ↓                    ↓
          benchmarks/通用实验方法       problem / menu / oracle
                   ↓
              design/primal
               ├─ boundary（约束与分离）
               ├─ regional → problem → oracle / cost / assignment
               └─ validation（读取结果）
```

`examples/` 不导入 `benchmarks/`、`primal.py` 或区域求解器。
通用 benchmark 接收 `BundleDesignProblem`、`BundleMenu` 和 `RegionalNlpSolver`，
具体 benchmark 负责选择网络、初值和求解预算。`design/` 与 `oracle/` 不依赖实验模块。
依赖方向由 `tests/test_architecture.py` 检查。

网络数据与经济参数在 `examples/`，运行预算在 `benchmarks/settings.py`。
具体入口为 `benchmarks/two_route.py` 和 `benchmarks/sioux_falls.py`；CLI 负责命令解析与输出。
Python 导入迁移和调用示例见[实验分工](benchmark_experiments.md)。

## 4. 现有模块职责与扩展边界

### `boundary.py`（已实现）

负责同一组 pairwise choice constraints：

- 计算 \(\delta_{iba}=G_{ia}-G_{ib}\)；
- 计算 slack \(s_{iba}=\delta_{iba}-(C_b-C_a)\)；
- 找每个 occupied ordered pair 的 critical travelers；
- 初始化和更新 critical constraint mask；
- 执行 full traveler separation。

固定 assignment 下 active constraints 使用布尔矩阵：

```text
active[i, a]              shape (I, B0)
```

其中 assigned bundle 由 \(b=\alpha_i\) 唯一确定，因此不额外存第三个 bundle 轴。`active[i, a]` 表示加入 traveler \(i\) 从当前 assigned bundle \(b\) 到 alternative \(a\) 的约束。

### `regional.py`（已实现）

负责固定 assignment 的连续问题：

- solver vector 与 `BundleMenu` 之间的 pack/unpack；
- regional objective 和 gradient（统一使用一般供给成本接口）；
- active choice constraints 和 Jacobian；
- 价格和额度非负、outside-option equality 和 subscription price monotonicity；
- `RegionalNlpSolver` Protocol；
- `RegionalSolveResult`。

区域目标使用 \(\Pi_\alpha=\sum_bD_b(C_b-K_b(\varepsilon_b))\)，并在可微处返回 \(\partial\Pi_\alpha/\partial C_b=D_b\)、\(\nabla_{\varepsilon_b}\Pi_\alpha=-D_b\nabla K_b\)。`profit_gradient` 通过成本接口计算额度导数；求解器最小化负利润时统一反号。菜单、利润、梯度和路径估值按需缓存，单独计算目标不触发路径估值。

solver vector 只包含设计套餐 \(b=1,\ldots,B\)，不包含固定 PAYG 变量。固定顺序为：

```text
[C_1, ..., C_B, epsilon_1,1, ..., epsilon_B,M]
```

测试分为两层。默认单元测试覆盖 0/1/多条 active critical constraints、导数、状态映射和至少两个 solver 算法；显式规模测试覆盖 100/1000 条 critical constraints。所有 solver 后端遵循同一个 contract：返回统一状态、菜单、目标值、最大约束违反、运行时间、迭代次数、solver 原生最优性指标和独立的一阶 KKT 残差。跨 solver 比较以可行性和目标值容差为准，不要求局部解的变量逐项完全相同。

不能把 solver 返回的原生 `success` 或 `optimality` 直接解释为数学上的局部最优证明。regional adapter 会根据 active regional inequalities 与 active lower bounds，通过非负最小二乘拟合 KKT multipliers 并检查 stationarity；只有独立可行性/KKT 检查通过时才生成统一的 `SUCCESS` 状态，solver 原生状态保留为诊断。首次 SLSQP 验收失败时，adapter 才依次执行可行回溯、一次 SLSQP restart 和 `trust-constr` fallback；首次成功时不产生额外 NLP 成本。后续流程直接信任这个 contract，不重复检查。一般非线性 regional problem 的局部最优性仍依赖 solver 能力及论文中的 regularity/SOSC 假设，主程序不重复实现二阶 NLP 验证器。规模测试采用可解析的仿射实例，其目标存在独立全局上界，因此匹配已知解时可标记为 `global`（自然也意味着 local optimum）。

普通/详细 pytest 模式在终端汇总 metrics，`-q` 模式保持安静。默认测试保持快速；运行规模套件使用：

```bash
pytest tests/design/test_regional.py --run-regional-scale \
  -k solver_scale_and_known_global_optimum
```

运行时间只作为观测指标，不设置依赖机器性能的通过阈值。

### `validation.py`（已实现）

负责把已有终止状态整理成算法级诊断：

- outer assignment 是否一致；
- fixed-assignment generation 是 solver 失败还是 separation 未完成；
- 已由 regional adapter 保存的 solver/KKT 状态；
- numerical active tie 和 double-tie 诊断。

`validate_candidate(...)` 不重建 regional problem，也不重复执行前序状态机已经保证的检查。若 outer 状态为 `CONSISTENT`，则根据控制流不变量，最终 regional solve、KKT 门槛与 full separation 已经通过；validation 读取 `terminal_iteration` 的支持记录与 `boundary_search_complete`。数值近点复用之前，搜索层会在保留菜单上额外检查邻区 full separation 和 KKT。已完成搜索的 double tie 可以通过数值候选检查，缺少搜索记录时仍报告 `DOUBLE_TIE`。

返回类型为 `CandidateDiagnostics`，成功状态为 `CONSISTENT_CANDIDATE`。这个名字有意不使用 `CERTIFIED_LOCAL_OPTIMUM`：程序负责返回 assignment-consistent regional candidate；新版局部最优定理要求当前点是自身区域局部极大点，并且所有活跃 double ties 生成的局部相邻区域也在该点 resolved。无 double tie 是无需邻区检查的特殊情况。当前往返检查是数值停止规则，不能自动替代该定理要求的局部最优证据。

### `primal.py`（内层 generation 与外层 consistency finder 已实现）

当前外层公共入口：

```python
find_assignment_consistent_menu(problem, initial_menu, solver, ...)
```

它先优化 assignment region，按 optimistic rule 切换，响应一致后再试探 double tie 替代项；搜索维护边界栈、未尝试的切换和返回记录，复用原有区域求解器，详见 [边界搜索](./boundary_assignment_search.md)。

当前公共函数 `solve_fixed_assignment(...)` 已实现内层 constraint-generation loop：初始化 critical constraints、求解 restricted regional NLP、full traveler separation、按 ordered pair 单调加入最违反约束，并区分 `converged`、`solver_failed`、`iteration_limit` 和 `stalled`。测试包括纯合成的基本控制流分支，以及调用 SLSQP 的 10/1000-traveler 已知最优算例；后者检查最终价格、额度、固定 assignment 利润、full separation 和 KKT，并在非静默 pytest 中输出 generation metrics。与一次性加入全部 traveler constraints 的数值对照测试暂未加入。

`find_assignment_consistent_menu(...)` 在每个候选菜单重新执行 numerical optimistic assignment；assignment 改变时用当前菜单 warm start，并通过新的 `solve_fixed_assignment` 调用完全重建 assignment-dependent critical set。每轮分别记录旧 assignment 下的 regional objective 和真实 optimistic assignment 下的 true profit。普通上升步骤要求两个值均不低于 incumbent；边界试探只要求区域目标不低于该 assignment 的 entry objective，并独立保留最高真实利润菜单。普通 tie 不再被强制反向试探；等利润异点也继续优化。

边界试探同点且双方检查完成时记录 `boundary_returned`，再处理其他待查邻区；不同菜单则继续优化，即使 assignment 返回且利润相等。`stop_on_assignment_revisit=True` 仅对未确认、无实质改善的同菜单重访作保守诊断；异点或有改善的 assignment 重访继续优化。这不是数学 cycle 声明，因为 warm-started local NLP 的状态是 `(assignment, menu)`，相同 assignment 配不同 menu 可能通向不同局部解。默认模式由 `max_region_iterations` 兜底。

最终入口 `solve_primal(...)` 检查初值价格顺序与严格正利润，再调用 consistency finder 和 `validate_candidate(...)`，返回 `CandidateDiagnostics`。它不在 production path 中计算 LICQ/MFCQ、有限差分 Hessian或 reduced-space SOSC，也不单独声称已经给出数学局部最优证书。

## 5. 不采用的结构

- 不创建 `ExpectedRouteValueOracle`：期望仍是 route oracle 的职责。
- 不让 `RouteValueOracle` 保存价格、权重或 assignment。
- 不使用一个不断膨胀的 `models.py` 存放所有类。
- 不把完整 primal 算法写成 `BundleDesignProblem` 的巨大方法。
- 不为每个 traveler 或 bundle 创建带可变状态的对象；大规模数值数据继续使用规则 NumPy 数组。
- 不让 NLP solver 直接决定 optimistic assignment。

## 6. 菜单约束的实现口径

regional NLP 严格实现当前 LaTeX 已定义的约束：

\[
C_b\ge0,\qquad
\varepsilon_b\ge0,\qquad
C_{b+1}\ge C_b,\qquad
C_0=0,\quad\varepsilon_0=0.
\]

当前不增加有限 price/allowance upper bounds，也不增加 allowance monotonicity。最新利润收敛证明不再依赖紧性，无需为它额外添加 bounds。若以后引入 `MenuDesignSpace`，有限 bounds 应作为明确的业务或实验约束说明；这个对象目前尚不存在。算例生成器的额度单调初值不等于区域 NLP 施加额度单调约束。

最新第 4.3 节由完整 assignment 可行性得到 \(C_b+G_{ib}\le e_{i0}\)。由于 \(G_{ib}\ge0\)、\(K_b\ge0\)，有 \(M_b\le C_b\le e_{i0}\)，从而 \(P^k\le\sum_iQ_i e_{i0}<\infty\)。结合正利润初值与单调接受，可得 \(0<P^0\le P^k\)，无需菜单域紧性。

这个上界不是对所有菜单变量的上界，也不能直接约束尚未 full separation 的试探点。代码已有外层 incumbent 保护，但失败时保留菜单不等于求得区域局部极大点。实现上保留外层 iteration limit；触及限制时返回 `REGION_ITERATION_LIMIT`。利润值收敛不等价于有限次区域切换，不能据此移除迭代限制。

## 7. 稳定的公共调用方式

```python
problem = BundleDesignProblem(
    traveler_weights=weights,
    unit_allowance_cost=k,
    route_oracles=oracles,
    demand_distributions=distributions,
)

menu = BundleMenu(prices=prices, allowances=allowances)

# 给 regional constraints 单独使用
valuation = problem.evaluate_valuations(menu)

# 给初始化、region transition 和最终 consistency check 使用
assignment = problem.evaluate_assignment(menu)
```

上述兼容写法可改为 `provision_cost=CallableProvisionCost(...)`，详见 [成本接口](./provision_cost.md)。后续模块应接收 `problem`、`menu` 和明确的 result/state 对象，不重新展开成长参数列表。


## 8. 边界搜索的状态与职责

- **incumbent**：保存不降利的 incumbent；等利润异点可更新菜单，同点复核通过则保留原菜单。
- **边界栈**：保存已优化的菜单、原 assignment、来源区域编号和尚未尝试的 tie。
  普通 tie 按 optimistic rule 切换；从 `numerical_optimistic_mask` 中选取替代项作额外邻区求解。当前一次只切换一个旅客；多旅客同时切换组合的按需扩展尚未实现。
- **邻区求解**：每次复用 `solve_fixed_assignment`，重建 critical sets 并执行 full separation。
- **返回记录**：同点、无实质利润变化且双方区域检查完成时才结束往返；数值近点须在保留点复核邻区。
  不同菜单（包括等利润）清空旧点检查记录并继续优化；返回旧 assignment 也须从新菜单重新求解。
- **结果与限制**：`terminal_iteration` 指向支持最终菜单的原区域求解，避免误用最后一个试探区域的证书。
  `boundary_search_complete` 表示逐 tie 数值搜索完成。失败或触及次数上限时保留 incumbent 并报告未完成。

具体参数、记录和数值停止条件见 [边界搜索](./boundary_assignment_search.md)。往返和一阶 KKT
不等价于独立二阶局部最优证明，也没有认证多重交点的全部联合方向；这些仍是论文认证要求的差距。
validation 汇总已有记录，不隐式启动优化。

## 9. 直接 OBD/PBD 模型的扩展边界

新稿第 3.3 节给出的两个单层模型共用 `BundleDesignProblem`、`BundleMenu` 和 `ValuationTable`，但拥有独立的 response 变量及求解状态：

| 变量 | 维度 | 用途 |
|---|---|---|
| \(q\) | \((I,B+1)\) | 原始响应份额 |
| \(\Pi\) | \((I,)\) | 旅客最低广义成本 |
| \(\lambda\) | \((I,)\) | PBD 内层 simplex 的自由对偶变量 |
| \(\eta\ge0\) | \((I,)\) | PBD 最优响应支持约束的对偶变量 |

OBD 施加 simplex、\(e_{ib}-\Pi_i\ge0\) 和 \(q_{ib}(e_{ib}-\Pi_i)=0\)，由上层最大化选择最有利响应。

PBD 施加 simplex、\(\sum_be_{ib}q_{ib}\le\Pi_i\)、\(\Pi_i\le e_{ib}\)，再增加

\[
\lambda_i-\eta_i e_{ib}\le M_b,\qquad
\sum_bq_{ib}M_b=\lambda_i-\eta_i\Pi_i.
\]

原始可行、对偶可行和强对偶等式共同强制最不利的最低成本响应。即使 \(K_b\) 非线性，固定菜单后 \(M_b=C_b-K_b(\varepsilon_b)\) 仍是响应 LP 的固定系数，单层重构继续成立。新稿将其称为基于强对偶的非凸重构；valuation 也可能不可微，接入 NLP solver 时不能默认光滑性或凸性。

`AssignmentResult` 当前专门表示 optimistic selection，不应将 PBD 的原始响应变量或最不利响应静默塞入此类型。直接模型的 pack/unpack、约束和 solver adapter 应独立于固定 assignment 的 `regional.py`；PBD 不复用 optimistic boundary-dominance 或区域搜索收敛论证。


## 10. 初值采样职责

`examples/` 的 `draw_*_menu` 提供各网络的单个候选菜单；`benchmarks/` 的 `generate_*_initial_menus` 管理随机种子、数量及问题复用，并调用 `benchmarks/initialization.py` 的 `sample_profitable_menus` 进行有上限的正利润拒绝采样。`solve_primal` 再检查实际初始 profit 大于 `profit_tolerance`，覆盖用户直接传入的菜单。

初值筛选和外层 incumbent 保护共同保证返回利润不低于严格正的初始利润，全 PAYG 因利润为零而不可被接受。`PROFIT_DECREASE` 对应失败诊断，不得计入 validated 或 consistent runs。逐 tie 边界搜索已实现，独立局部最优认证与 PBD 模型仍属于待实现部分。


## 11. 一般供给成本接口

新稿要求 \(K_b\ge0\)、连续、\(K_b(0)=0\)，并在使用导数处连续可微；未要求其凸或单调。当前接口位于 `design/cost.py`，完整用法见 [成本接口](./provision_cost.md)：

- 按套餐计算成本 \(K_b(\varepsilon_b)\)，并在需要导数时提供 \(\nabla K_b(\varepsilon_b)\)。PAYG 的固定零额度对应零成本和零 margin。
- 将共享线性 \(k^\top\varepsilon_b\) 保留为兼容特例；一般成本值批量返回 `(B,)`，梯度返回 `(B, M)`，两者均不含 PAYG。
- `BundleDesignProblem.evaluate_assignment`、`RestrictedRegionalProblem.profit/profit_gradient`、初值筛选、margin tie-break 和 incumbent 接受已统一使用同一成本定义；未来 PBD 重构也应复用。
- 路径 LP、对偶多面体、\(G\)、choice constraints 和 full separation 不因供给成本推广而改变。
- `tests/design/test_cost.py` 已覆盖线性兼容、非线性梯度差分、套餐特定成本、初值与利润验收及已知最优解。现有 two-route/Sioux Falls 默认仍验证线性特例。
- 一般成本的 `trust-constr` 目标 Hessian 使用 BFGS，线性成本保留零 Hessian；数值差分和目标非凸性可能增加求解时间，接口推广不等于性能或全局最优性保证。
