# 三路径、两模式的小算例

本算例为人工设定的机制说明实例，版本 `stylized-3path-2mode-v2`。
基础配置使用 P、M、S 三类；扩展配置加入高不确定性类 M_H。
类名仅代表直觉上的基础偏好，实际路径使用由下层优化决定。

## 网络、路径与额度单位

节点为 O、H、D，一个 OD 对 O→D。网络使用 MultiDiGraph，O→H 的 PT 与 SM
服务是两条不同的平行有向边。共四条服务边、三条可行路径：

| 路径 | 服务边 | 单次 PT 用量 | 单次 SM 用量 |
| --- | --- | ---: | ---: |
| p1 | O→H（PT），H→D（PT） | 2 | 0 |
| p2 | O→D（SM） | 0 | 2 |
| p3 | O→H（SM），H→D（PT） | 1 | 1 |

模式顺序为 `(PT, SM)`，路径顺序为 `(p1, p2, p3)`：

\[
N=\begin{pmatrix}2&0&1\\0&2&1\end{pmatrix}.
\]

额度计量单位为模式专属的资源单位。两条 PT 边及 O→H 的 SM 边每次使用消耗
1 单位，直达 SM 服务每次使用消耗 2 单位。因此，资源用量不等于边数。
网络无拥堵，路径成本与人口、流量无关。各套餐提供相同的路径集合、资源矩阵和超额费率。

## 旅客、分布与人口

成本已货币化，以人工货币单位 MU/次 OD 出行计，等价于 beta=1。
周期固定为一个月，频率为每人每月 OD 出行次数，采用连续均匀分布。

| 类别 | 直觉标签 | 路径成本 (p1,p2,p3) | 频率分布 | 均值 | 方差 |
| --- | --- | --- | --- | ---: | ---: |
| P | PT-oriented | (4,5,4.5) | U[14,22] | 18 | 16/3 |
| M（亦称 M_L） | multimodal-oriented | (5,4.5,3) | U[16,24] | 20 | 16/3 |
| S | SM-oriented | (5.5,3,4.5) | U[18,26] | 22 | 16/3 |
| M_H（可选） | 高不确定性的 M | (5,4.5,3) | U[8,32] | 20 | 48 |

基础配置类顺序为 `(P,M,S)`，权重均为 1/3。
`baseline_frequency_supports` 按此顺序逐类配置分布，构造器不会复制同一个分布给三类。
同 OD、同频率分布但路径成本偏好不同，仍可作为不同的偏好类；本实例进一步引入
频率异质性，避免三类对部分额度的价值完全对齐。三个基础类保持相同的方差，
均值分别为 18、20、22，因此这里展示的是需求水平差异；需求不确定性差异仍由
M/M_H 这对同均值类别隔离展示。

例如，纯 PT 额度 `(44,0)` 相对于 PAYG 的期望成本节省为 `(72,79,84)`。
v1 三类均使用 `U[16,24]` 时，这三个数均为 79。M 的分布和成本保持原值，
因此 A1 的数值和两面板图保持不变。网络、路径成本、费率、供给成本及算法规则未调整。

四类扩展默认采用 lambda=0.20：

\[
(Q_P,Q_M,Q_S,Q_{M_H})=(4/15,4/15,4/15,1/5).
\]

人口归一化为 1。M_H 与 M 共用同一个路线 oracle，仅需求分布不同。
其他人口构成可通过替换配置的 `traveler_weights` 设置；权重必须非负且和为 1。

## 套餐、成本与设计域

套餐顺序为 `(PAYG,Basic,Premium)`，PAYG 的价格和两种额度均固定为零。
Basic 与 Premium 各有一个价格和两个额度，合计六个连续决策变量。

\[
\rho=(2,4),\qquad k=(0.5,1),
\]

分别以 MU/资源单位计。套餐供给成本按每位订阅者的承诺额度计：

\[
K_b(\varepsilon_b)=0.5\varepsilon_{b,PT}+\varepsilon_{b,SM},
\qquad M_b=C_b-K_b(\varepsilon_b),\quad M_0=0.
\]

用户以 `C_b + E[Phi_ib(mu;epsilon_b,rho)]` 比较套餐，需求实现后再决定路径分配
与超额使用。路径分配为连续非负变量。用户按期望成本决策；成本平局采用平台利润
更高的 optimistic 选择规则。平台目标为 `sum_i Q_i M_{alpha_i}`。
平台利润不另外计入用户向服务商支付的 PAYG 超额费用。

基础算例的默认构造仅保留非负性、价格排序和固定 PAYG：

\[
0\le C_1\le C_2,\qquad
\varepsilon_{b,m}\ge0,\qquad C_0=0,\quad\varepsilon_0=(0,0).
\]

价格和额度不设置有限优化上界，也不施加 Premium 各模式额度均大于 Basic 的约束。
PAYG 下各类的期望成本为 `(144,180,209)`，加入 M_H 后为 `(144,180,209,180)`。

候选初值抽样单独使用 `sampled_price_bounds=(0,190)` 和
`sampled_allowance_bounds=(0,64)`。190 保留为既有抽样设置，已不等于 v2 的最大
PAYG 期望成本；64 参考可选 M_H 的最大频率
32 与单次最大模式用量 2 的乘积。这两个范围仅控制随机抽样，手动提供的菜单及
后续区域 NLP 解可以超过它们。`validate_menu` 不再检查价格或额度的有限上界。
配置元数据以 `design_domain.finite_upper_bounds=false` 明确记录这一设计域。

A2 按该实验的要求显式恢复价格 190、额度 64 的有限上界，供三种方法共同使用。
这通过构造器的 `price_upper_bound` 和 `allowance_upper_bound` 接入，默认构造与 A1
保持原配置。A2 导出的模型元数据和设计域均记录实际采用的有限上界。

## 初始菜单

以下菜单用于后续算法的启动，不代表优化后的结果：

| 套餐 | 月价格 | PT 额度 | SM 额度 | 每位订阅者 margin |
| --- | ---: | ---: | ---: | ---: |
| PAYG | 0 | 0 | 0 | 0 |
| Basic | 40 | 24 | 8 | 20 |
| Premium | 75 | 32 | 24 | 35 |

oracle 已验证：基础配置的 optimistic assignment 为 `(Basic,Premium,Premium)`，
归一化利润为 30 MU/人·月；四类扩展为 `(Basic,Premium,Premium,Premium)`，
归一化利润为 31 MU/人·月。两者均提供正利润初值。

## Python 入口

```python
from dataclasses import replace
from maas_bundle.examples import (
    DEFAULT_STYLIZED_CONFIG,
    HIGH_UNCERTAINTY_STYLIZED_CONFIG,
    build_stylized_graph,
    build_stylized_problem,
    initial_stylized_menu,
)

config = DEFAULT_STYLIZED_CONFIG
graph = build_stylized_graph()
problem = build_stylized_problem(config)
menu = initial_stylized_menu(config)
result = problem.evaluate_assignment(menu)
metadata = config.metadata()

# 加入 M_H，默认占比为 20%。
extended_problem = build_stylized_problem(HIGH_UNCERTAINTY_STYLIZED_CONFIG)

# 例如：固定 multimodal 总占比 30%，分给 M_L 和 M_H 各 15%。
replacement_config = replace(
    HIGH_UNCERTAINTY_STYLIZED_CONFIG,
    traveler_weights=(0.35, 0.15, 0.35, 0.15),
)
```

`draw_stylized_menu(random, config)` 在上述初值抽样范围内抽样，价格有序，额度可以交叉。
它不保证候选菜单利润为正；后续多初值实验可复用 benchmark 中的正利润筛选工具。
算例数据位于 `examples/`。A1 的实验运行器与 CLI 子命令见下文。

## 第一组实验 A1：Oracle structure

固定基础用户 M，使用路径成本 `(5,4.5,3)`、需求 `U[16,24]` 和超额费率 `(2,4)`。
在原配置的 `[0,40]^2` 额度窗口内，使用 161×161 网格（间隔 0.25）计算
`G_M`、`omega_PT` 和 `omega_SM`。本实验不求解上层菜单设计。
绘图窗口与初值抽样范围均不构成优化上界。

```bash
maas-bundle oracle stylized --output-dir results/stylized_a1
```

也可以运行模块入口：

```bash
python -m maas_bundle.cli oracle stylized --output-dir results/stylized_a1
```

使用 `--grid-points` 调整网格分辨率，`--allowance-max` 调整扫描窗口。
Python 入口为 `run_stylized_oracle(config=..., settings=...)` 与
`save_stylized_oracle_results(result, output_dir)`，均从 `maas_bundle.benchmarks` 导入。
数值计算与绘图分别位于 `benchmarks/stylized_oracle.py` 和
`benchmarks/stylized_oracle_plots.py`。

正式输出仅为一张 1×2 figure 的两个格式文件，分辨率 320 dpi：

- `a1_valuation_structure.pdf`
- `a1_valuation_structure.png`

Panel (a) 绘制期望估值 `G_M` 的二维热力图与少量细等值线。
Panel (b) 在同一额度平面叠加 `mu=16,20,24` 的实现需求对偶区域边界，
分别采用蓝色点线、灰色实线、橙色长虚线；两幅图均使用 `[0,40]^2` 坐标范围和等比例轴。
不再输出影子价格热力图、额度截面、实现成本热力图、单一需求的彩色区域图或三维曲面。

三个需求的活跃顶点图分别直接通过同一个 oracle 的缓存顶点求最大值获得，
未缩放 `mu=20` 的边界。先由网格识别全维活跃区域，再将候选顶点对的成本相等线
与扫描窗口及全部对偶支配不等式求交，仅绘制属于上包络的真实区段。
多顶点平局通过边界线表示。只有验证多区域交汇且达到实现需求的充分覆盖成本后，
才标记 `(16,16)`、`(20,20)`、`(24,24)`。

期望值由均匀分布的区间概率和一阶矩精确积分，不使用 Monte Carlo。
算法同时通过独立路径分配 primal LP 核对实现成本，通过 primal LP 的数值积分核对
期望成本，并在 64 个固定随机种子抽取的内部点核对有限差分与负期望影子价格。
导数检查使用种子 20261002；两侧导数差异明显的点会被标记并从光滑点误差统计中排除。
另外检查 G_M 随两种额度均不增加，期望影子价格位于 `[0,rho]`。

上述模型、参数、oracle 与数值求值流程未改变。额外的绘图检查包括 `G_M(0,0)=180`、
大额度区成本为 60、新旧 `mu=20` 的活跃顶点图逐点一致，以及提取的边界确实属于对偶上包络。
终端报告所用需求、额度窗口、缓存顶点数、每个需求的活跃顶点数和验证状态。
数值数组仍由 `run_stylized_oracle` 返回；正式导出不再写 CSV、NPZ、JSON 或结果 README。
成功生成新图后，导出器清理自身旧版本的 A1 图与数据文件。

这张图解释 valuation oracle 的数学结构：实现需求变化使参数 LP 的区域边界移动，
期望估值对这族移动的分段仿射值函数取平均。它不承担套餐设计的 managerial insights。
连续需求平均缓和了实现需求下尖锐的区域转换，但不意味着 `G_M` 处处可微；
不可微处仍采用次梯度表述。

建议论文 caption：

> Valuation structure for traveler class M. (a) Expected route-side cost
> $G_M(\boldsymbol{\varepsilon})=\mathbb{E}_{\mu}[\Phi_M(\mu;\boldsymbol{\varepsilon})]$
> under $\mu\sim U[16,24]$. (b) Realized-demand dual-regime boundaries for selected
> demand realizations $\mu=16,20,24$. For each realized demand, the allowance space
> is partitioned according to the active dual extreme point. As demand varies,
> these boundaries shift, so the expected valuation in panel (a) averages over
> a moving family of polyhedral regime partitions.

## 第二组实验 A2：Regional search 与 assignment benchmark

```bash
maas-bundle regional stylized --output-dir results/stylized_a2
# 同等模块入口：python -m maas_bundle.cli regional stylized ...
```

代码位于 `benchmarks/stylized_regional.py`，绘图位于
`benchmarks/stylized_regional_plots.py`。Python 入口为
`run_stylized_regional(settings=...)`、`save_stylized_regional_results(result, output_dir)`，
设置类为 `StylizedRegionalSettings`，均从 `maas_bundle.benchmarks` 导入。

三种方法共用同一个问题实例，保持 P/M/S、需求、权重、oracle、供给成本和 optimistic
规则不变；统一使用 `0 <= C1 <= C2 <= 190`、`0 <= epsilon_bm <= 64`。
通用 `BundleDesignProblem` 增加可选订阅价格/额度上界，默认仍为无穷。
`RestrictedRegionalProblem` 将上界传入 SciPy `Bounds`，同时检查上界违反量和
活跃上界的 KKT 乘子。重启与 trust-constr 备用求解也使用同一组 Bounds。
区域搜索拒绝超出问题上界的初始 menu。

正式运行固定如下数值设置：

| 项目 | 设置 |
| --- | --- |
| NLP backend | SLSQP |
| 每次 NLP 最大迭代 | 500 |
| SLSQP ftol | 1e-10 |
| choice / feasibility 容差 | 1e-7 |
| margin 容差 | 1e-9 |
| weighted profit 容差 | 1e-9 |
| normalized KKT stationarity 容差 | 1e-5 |
| KKT active-set 容差 | 1e-4 |
| 最多 regional-search iterations | 100 |
| 每个 assignment 的起点数 | 1（求解器适配层提供） |
| 随机种子 | 20261004 |

A2 三种方法均启用求解器内部恢复：SLSQP 失败后从已有局部迭代点重启，必要时改用
数值约束 Jacobian，仍失败则调用 trust-constr。所有尝试继承设计上界，并使用原有
可行性/KKT 容差认证，不增加独立抽样的实验起点。区域初始化请求数和实际 backend
调用数分别记录。微小价格逆序只在原可行性容差内修正，修正后重新认证；微小利润损失
只有在原菜单通过当前区域 KKT/可行性检查时才保留原菜单，不放宽“不降利润”要求。
CLI 可用 `--starts`、`--seed`、`--max-nlp-iterations` 修改运行预算，实际参数写入 JSON。
默认每个 assignment 使用一个数值初值，由
`ScipyRegionalSolver.default_initial_menu` 生成零价格、零额度。SciPy 的 SLSQP 要求
显式 `x0`，该默认策略属于适配层；给定 warm start 时仍使用既有 entry menu。
需要复现旧多初值对照时，可显式设置 `--starts 20`；该模式至少使用 4 个共同锚点。
默认单初值模式没有随机抽样，`--seed` 不影响它的数值初值。

默认 `--trajectory-start switch` 使用下面的展示初值，仅改变实验的启动 menu：

| 选项 | 价格 | PT 额度 | SM 额度 |
| --- | ---: | ---: | ---: |
| PAYG | 0 | 0 | 0 |
| Basic | 50 | 0 | 30 |
| Premium | 60 | 40 | 12 |

三类旅客的初始最小成本选项均唯一，assignment 为 `(2,2,1)`，利润为约
`25.3333333333`。选择该初值是为了展示一次由真实成本边界与 margin 优势产生的
切换；旅客、需求、路径、成本、oracle 与算法规则完全保持原配置。区域 NLP 的
数值起点仍由原 solver adapter 按 entry menu warm start，不增加外层采样步骤。
`--trajectory-start baseline` 可复现原始 `(1,2,2)` 初值。默认 JSON 另外记录
`baseline_reference` 的完整 baseline 启动轨迹，使用同一组 v2 分布。
v1 同分布下的切换展示六个输出保存在 `results/stylized_a2_same_demand/`，
v1 原始 baseline 输出保存在 `results/stylized_a2_baseline/`，均属于历史对照。

CG 直接调用现有 `solve_primal`。全约束版本调用同一入口，仅令
`initial_per_pair=traveler_count`，保证每个区域开始时全部 6 个 assignment-preserving
旅客约束都已包含。记录器包装 solver，捕获实际进入每次 NLP 的 menu，既不推测边界试探
的起点，也不重写接受/拒绝逻辑。JSON 同时保存各区域的全部 CG 内部迭代。

共有 `3^3=27` 个 assignment 组合；排除固定目标恒为零的 `(0,0,0)` 后，
枚举依字典序遍历 26 个候选 assignment，每个固定区域均从完整约束开始，
且只使用一个由适配层提供的默认数值初值。当前算例各套餐共用路径与 oracle，
零价格、零额度使各选项成本相等，因此这个数值点满足所有固定区域的弱选择约束。
它的真实 optimistic assignment 是 `(0,0,0)`；不能把求解器的数值点误当作已选择
的初始 assignment，也不能直接用它启动要求正利润 menu 的 `solve_primal`。

每个起点分别记录 solver/KKT 成功和完整区域可行性。失败但可行的点也保留为已找到的
可行候选，并明确标记 solver 状态；局部失败不解释成数学不可行。每个区域报告最大固定
assignment 目标及其可行 menu，同时报告该 menu 的真实 optimistic assignment 和利润。
比较表使用全部可行候选中独立重算的最大 optimistic 利润；如果 tie-breaking 改变
assignment，JSON 的 `best_origin_assignment` 保留其来源区域。

切换展示的轨迹为：

| 阶段 | assignment | 平台利润 |
| --- | --- | ---: |
| 初始 menu | `(2,2,1)` | 25.33333333 |
| 第一轮区域求解，尚未 reassignment | `(2,2,1)` | 47.86508583 |
| 同 menu 下 optimistic reassignment | `(2,2,2)` | 49.13671869 |
| 第二轮区域求解并终止 | `(2,2,2)` | 51.00000000 |

S 在第一轮的 Basic/Premium 成本差约 `3.69e-13`，低于 `1e-7` 的成本 tie 容差。
Premium 的 margin 比 Basic 高约 `3.81489859`，因此真实 optimistic response 使
S 从 Basic 切换到 Premium。同一 menu 的加权利润增加约 `3.81489859 / 3 = 1.27163286`。
下一轮使用切换点的同一个 menu 进入 `(2,2,2)`，继续优化到约 51。

CG 共访问两个 assignment，完成一次接受的 assignment transition，进行 2 次 NLP，
新增约束 0 个：本次两个区域的初始 critical constraints 已足够。全约束版本同样发生
S 的切换，进行 2 次 NLP；最终 assignment 和利润与 CG 相同。两个终点的完整 menu
最大坐标差约 `1.48e-11`。JSON 同时保留完整 menu 差和
`max_used_menu_variable_difference`，不强制菜单逐坐标相同。本次轨迹展示区域切换，
没有展示 CG 新增约束或减少 NLP 次数，不能据此声称 CG 比全约束版本节省求解次数。

枚举覆盖 26 个候选固定区域、每个区域一个起点。26 次初始化请求均通过
可行性/KKT 检查，包括内部恢复在内共 30 次 backend NLP 调用。最佳利润约 70.04308973，
对应 `(1,2,2)`；展示轨迹的终点 51 比它低约 19.04308973，相对差约 27.19%。
所以这条轨迹展示了真实、利润不下降的区域切换，也展示了局部搜索可能停在比
枚举最佳解更差的候选点，不能据此声称找到全局最优。

默认结果的 `baseline_reference` 保留原 `(1,2,2)` 初值从利润 30 优化到约 70.04308973
而不发生切换的轨迹。起始 menu 的作用由这两条具体轨迹说明，无需把外层随机
多初值覆盖率作为算法步骤。

在枚举找到的 `(1,2,2)` 最佳候选上，P 的最小成本选项是 `{PAYG,Basic}`，M、S
均为 `{Basic,Premium}`。v1 同分布结果中，M、S 在此候选上的最小成本选项均为
`{PAYG,Basic,Premium}`；这里的同时三重 tie 已被打破。其他区域仍可能出现三重
tie，特别是零 menu 或未被使用的套餐，因此异质需求不等于消除所有 tie。

所有报告 menu 的设计域、区域 full separation、最终 optimistic assignment、独立
利润重算、接受利润单调性和切换成本/margin 条件均记录在 JSON 中。CPU 与 wall time
分别用 `process_time()`、`perf_counter()`，各方法计时写入 CSV/JSON。

结果目录只生成以下六个文件：

- `a2_regional_search_trajectory.pdf`
- `a2_regional_search_trajectory.png`
- `a2_iteration_table.csv`
- `a2_assignment_enumeration.csv`
- `a2_algorithm_comparison.csv`
- `a2_full_results.json`

主图横轴是 regional-search iteration，纵轴是真实平台利润，0 表示原始初值。
所有点使用 optimistic response 之后的 assignment 标注；接受的点用线连接。
本次图显示初始点和两个区域求解终点，所有蓝色主点标注 optimistic response 后的
assignment。第一轮额外用小菱形显示 reassignment 前的固定区域目标，并标注 S 的成本
tie、margin 优势与同 menu 利润增量。该菱形不冒充真实 optimistic 平台利润。

## A2 当前补充：每个 assignment 一个数值初值

```bash
maas-bundle multistart stylized --output-dir results/stylized_a2_assignment_starts
```

共 26 个候选初始 assignment，各执行一次完整约束的初始化区域求解，数值初值均由
求解器适配层生成。得到经过区域检查的正利润 menu 后，再调用既有 `solve_primal`
继续搜索。初始化 fixed-assignment 的请求与该数值点的真实 choice 不必相同；
因此初始化后的 `alpha -> optimistic response` 与正式搜索中被接受的 assignment
变化分别记录，不将两者混为一个算法切换次数。

`(0,0,0)` 在生成实验起始 assignment 时直接排除，因为它的固定区域目标恒为零；
这不表示全 PAYG 的选择区域数学上不存在。零菜单仅是求解器数值初值。
26 次初始化请求全部成功；内部恢复后实际使用 30 次 backend NLP 调用。
`(2,1,2)` 约 `5.1e-12` 的价格逆序在适配层修正并重新认证，入口拒绝为零。
26 个正利润菜单均启动正式搜索，25 条通过最终诊断；`(0,2,0)` 的搜索在探索
邻区 `(1,2,0)` 时未通过可行性/KKT 认证，保留利润 28.5 的原菜单，并记录失败，
不将该条算作有效完成。两条约 `1e-11` 的数值利润下降通过重新认证并保留原 CG
可行菜单解决。初始化、后续搜索、CG 和切换合计 70 次 backend NLP 调用（不含
代表轨迹的全约束对照），这些调用不是 70 个初始点。

代表序列为 `(0,0,2) -> (0,2,2) -> (1,2,2)`：

| 阶段 | 求解的 assignment | 真实 optimistic assignment | 真实利润 |
| --- | --- | --- | ---: |
| 初始化区域求解 | `(0,0,2)` | `(0,2,2)` | 47.00000000 |
| 正式搜索第一轮 | `(0,2,2)` | `(0,2,2)` | 55.04308973 |
| 双重 tie 邻域优化 | `(1,2,2)` | `(1,2,2)` | 70.04308973 |

初始化的固定区域目标为 23.5，M 因 Premium 的 margin 约为 70.5，在同 menu 下从
PAYG 转到 Premium，使真实利润增加 23.5。随后 P 在 PAYG/Basic 的成本和 margin
双重 tie 处触发邻域探索，进入 `(1,2,2)` 优化后提高利润。第二个变化属于邻区优化，
不是第一次固定区域解后的直接 reassignment；因此正式搜索只有一次被接受的
assignment 变化。本次没有找到正式搜索内连续两次变化的轨迹。

全约束版本从同一初始化结果启动，复现正式搜索路径，最终利润差约 1.71e-13。
PNG 用橙色菱形区分初始化的固定区域目标与蓝色真实 optimistic 利润。输出只有
`a2_assignment_starts.png`、26 行的 `a2_assignment_starts.csv` 和完整 JSON。
修复前结果分别保存在 `results/stylized_a2_before_solver_fix/` 和
`results/stylized_a2_assignment_starts_before_solver_fix/`。
此前主 A2 的每区域 20 起点输出保存在 `results/stylized_a2_twenty_starts/`。

## A2 历史补充：500 个随机初始 menu 的探索

以下为此前的探索结果，已被上面的 26 个候选 assignment、每个一个初值配置替代。
历史输出保留在 `results/stylized_a2_multistart/`，不属于当前主实验设置。

本实验使用 v2 的异质需求配置及与 A2 相同的 190/64 设计上界和求解容差。
先使用 baseline 与 switch 两个参考初值，再从既有抽样范围生成 498 个正利润 menu。
随机抽样共尝试 1179 次，非正利润初值被筛除。每个 menu 均单独调用原有完整
`solve_primal`；初始 assignment 由实际 optimistic response 决定，区域 NLP 的
warm start 和算法规则未改变。这是外层初始 menu 的补充诊断，不是新增算法步骤。

切换次数按相邻接受结果的 retained optimistic assignment 是否不同计算。
被拒绝的邻域试探、同 assignment 再求解均不计数。另行报告固定区域求解后发生的
optimistic reassignment，以及严格增加同 menu 利润的 margin reassignment。
只数“假设 assignment 与该区域求解后的 response 不同”会遗漏被接受的邻区优化；
记录器现已修正此统计口径，生产算法未修改。

| 每条完整轨迹接受的 assignment 变化次数 | 起点数 |
| ---: | ---: |
| 0 | 492 |
| 1 | 6 |
| 2 | 2 |

500 条轨迹均通过独立 choice、利润、设计域和算法诊断检查，共执行 797 次区域 NLP。
初始 assignment 覆盖 14 类，终点 assignment 覆盖 12 类；最高找到利润约 70.04308973。
两条有两次变化的轨迹来自第 450 和 488 个起点，均为
`(2,1,1) -> (1,1,1) -> (1,2,1)`。

代表轨迹选用第 450 个起点：

| 套餐 | 价格 | PT 额度 | SM 额度 |
| --- | ---: | ---: | ---: |
| PAYG | 0 | 0 | 0 |
| Basic | 63.1585126058 | 32.6580304237 | 51.2549973675 |
| Premium | 64.5234186458 | 42.4832845089 | 14.0956634479 |

| 区域求解迭代 | 求解的 assignment | retained assignment | 真实利润 | 处理结果 |
| ---: | --- | --- | ---: | --- |
| 0 | — | `(2,1,1)` | 6.77837100 | 初始 menu |
| 1 | `(2,1,1)` | `(1,1,1)` | 51.00000000 | 接受；成本和 margin 双重 tie 下的 P 归并 |
| 2 | `(1,1,1)` | `(1,1,1)` | 51.00000000 | 接受；无 assignment 变化 |
| 3 | `(2,1,1)` | `(1,1,1)` | 51.00000000 | 邻域试探返回同点；拒绝 |
| 4 | `(1,2,1)` | `(1,2,1)` | 61.00000000 | 接受；双重 tie 邻域中找到改善 |

第一次变化时 Basic/Premium 的 margin 都约为 51，P 在两者之间的归并不增加
同 menu 的平台利润。第二次变化来自 M 的双重 tie 邻域：算法进入 `(1,2,1)`
重新优化后，将 retained 利润从 51 提高到 61。该轨迹中 P 也有一次三重成本 tie。
因此这是两次被接受的 assignment 变化，不能描述成连续两次严格 margin 优势触发的
optimistic reassignment。本次 500 个起点未找到后一种轨迹，也未找到三次及以上变化；
有限抽样不证明这些轨迹不存在。

全约束版本从同一个第 450 初值出发，复现了相同的接受路径和拒绝试探，最终利润差
约 2.13e-13。PNG 左侧显示切换次数分布，右侧保留实际区域求解迭代编号，并用叉号
标明被拒绝的试探。输出仅为 `a2_multistart.png`、`a2_multistart_runs.csv` 和
`a2_multistart_diagnostics.json`。旧 v1 的 20 起点诊断保存在
`results/stylized_a2_multistart_same_demand/`，主 A2 的六个输出未被覆盖。
