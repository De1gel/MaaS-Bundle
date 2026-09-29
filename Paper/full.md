# Optimistic and Pessimistic Mobility-as-a-Service Bundle Design

> 本文件是 2026-09-24 最新修订稿（一般供给成本与 PAYG 利润上界版） [MaaS_bundle_design.tex](./MaaS_bundle_design.tex) 的中文 Markdown 阅读版，归纳模型、重构和算法，并非逐段翻译。公式与证明以 LaTeX 为准；草稿待统一内容见 [文件说明](./README.md)，代码完成情况见 [软件架构](../docs/software_architecture.md)。

## 1. 问题与选择规则

平台联合设计订阅价格和多方式额度。旅行者在随机出行频率实现之前选择套餐，在频率实现之后分配路径。旅客类型由各自的频率分布、OD、可行路径及方式使用特征共同决定，一般不存在适用于所有套餐比较的统一类型排序。

新稿同时考虑两种平台设计问题：

- **OBD（optimistic bundle design）**：在旅客成本最低的选项中，选择平台利润最大的响应。
- **PBD（pessimistic bundle design）**：在旅客成本最低的选项中，选择平台利润最小的响应。

两者仅在旅客的最低成本选项并列时产生差别。本文为两种模型给出单层表示，并为 OBD 构造利用 assignment regions 和 critical travelers 的区域搜索。

## 2. MaaS Bundle Design under Uncertainty

### 2.1 套餐和可行菜单

令 \(i\in\mathcal I\) 为旅客类型，权重为 \(Q_i\)；\(b\in\mathcal B\) 为订阅套餐；\(m\in\mathcal M\) 为交通方式。记 \(B=|\mathcal B|\)、\(\mathcal B_0=\{0\}\cup\mathcal B\)，其中 \(b=0\) 是纯 PAYG。

菜单决策为 \(x=(C,\varepsilon)\)。所有套餐共享外生、严格为正的方式超额费率 \(\rho\)；平台只优化订阅价格 \(C_b\) 和额度 \(\varepsilon_b\)。可行域为

\[
\mathcal D=\{(C,\varepsilon):
C_b\ge0,\ \varepsilon_b\ge0\ (b\in\mathcal B),\quad
C_{b+1}\ge C_b\ (b=1,\ldots,B-1),\quad
C_0=0,\ \varepsilon_0=0\}.
\]

该定义没有有限价格/额度上界，也没有额度单调性。最新第 4.3 节利用 PAYG outside option 给出利润上界，不再假设菜单域紧。利润有界并不意味着菜单变量有界。

### 2.2 事后路径分配

类型 \(i\) 的频率 \(\mu\sim F_i\)。对选项 \(b\)，\(P_{ib}\) 为路径集合，\(\bar c_{ib}=\Delta_{ib}^{\top}t\) 为固定路径时间，\(N_{ib}\) 为方式—路径使用矩阵。给定 \(\mu\) 后，解

\[
\begin{aligned}
\Phi_{ib}(\mu;\varepsilon_b,\rho)
=\min_{f,z}\quad&\beta\bar c_{ib}^{\top}f+\rho^{\top}z\\
\text{s.t.}\quad&\mathbf1^{\top}f=\mu,\\
&N_{ib}f\le\varepsilon_b+z,\qquad f,z\ge0.
\end{aligned}
\]

额度约束划分已包含的使用量与另付费的超额使用量，不禁止旅客使用路径，也不是物理路网容量。严格正的 \(\rho\) 使超额量取最小可行值。

### 2.3 事前选择与平台利润

定义

\[
G_{ib}(\varepsilon_b)=\mathbb E_{F_i}[\Phi_{ib}(\mu;\varepsilon_b,\rho)],
\qquad e_{ib}=C_b+G_{ib}(\varepsilon_b),
\]

\[
\Pi_i=\min_{b\in\mathcal B_0}e_{ib},\qquad
\mathcal B_i^*=\arg\min_{b\in\mathcal B_0}e_{ib}.
\]

旅客最优响应集合为

\[
\mathcal S_i(x)=\{q_i\ge0:\ \sum_bq_{ib}=1,\quad
q_{ib}>0\Rightarrow b\in\mathcal B_i^*\}.
\]

等价互补表示是

\[
q_{ib}\ge0,\quad \sum_bq_{ib}=1,\quad
e_{ib}-\Pi_i\ge0,\quad q_{ib}(e_{ib}-\Pi_i)=0.
\]

每个订阅套餐的每位订阅者供给成本为 \(K_b(\varepsilon_b)\)，假设其非负、连续，满足 \(K_b(0)=0\)，并在使用导数的地方连续可微。平台单位利润为

\[
M_b=C_b-K_b(\varepsilon_b),\qquad K_0(0)=0,\quad M_0=0.
\]

共享线性成本 \(K_b(\varepsilon_b)=k^\top\varepsilon_b\)、\(k\ge0\) 是特例，也是当前默认算例使用的形式；代码已支持一般成本接口。新稿未额外要求一般 \(K_b\) 凸或单调。

两种设计目标分别为

\[
\max_{x\in\mathcal D}\sum_iQ_i\max_{q_i\in\mathcal S_i(x)}
\sum_bq_{ib}M_b
\quad\text{(OBD)},
\]

\[
\max_{x\in\mathcal D}\sum_iQ_i\min_{q_i\in\mathcal S_i(x)}
\sum_bq_{ib}M_b
\quad\text{(PBD)}.
\]

对固定菜单，内层目标关于 \(q_i\) 线性，两种规则都可选择极点响应，即把整类旅客分配给一个最低成本选项；无需依赖分数分配。

## 3. Bundle Valuation and Segmentation

### 3.1 固定对偶几何与额度边际价值

路径问题的对偶可行域为

\[
\mathcal V_{ib}=\{(\pi,\omega):
\pi\mathbf1-N_{ib}^{\top}\omega\le\beta\bar c_{ib},\quad
\pi\ge0,\quad 0\le\omega\le\rho\}.
\]

原对偶中的 \(\pi\) 是自由变量；在非负频率、成本和额度下，可不损失最优值地取 \(\pi\ge0\)。有至少一条有限成本路径时，\(\mathcal V_{ib}\) 有界，而且与 \(\mu\)、\(\varepsilon_b\) 无关。令其极点为 \((\pi_{ib}^j,\omega_{ib}^j)\)，则

\[
\Phi_{ib}(\mu;\varepsilon_b,\rho)
=\max_j\{\pi_{ib}^j\mu-(\omega_{ib}^j)^{\top}\varepsilon_b\}.
\]

因此 \(\Phi_{ib}\) 关于 \((\mu,\varepsilon_b)\) 联合凸且分段线性，关于 \(\mu\) 非减，关于额度逐分量非增。

当 \(F_i\) 有有限均值时，\(G_{ib}\) 有限、凸、逐分量非增，并满足

\[
|G_{ib}(\varepsilon_b')-G_{ib}(\varepsilon_b)|
\le\rho^{\top}|\varepsilon_b'-\varepsilon_b|.
\]

对任意可测的最优影子价格选择，令 \(\bar\omega_{ib}=\mathbb E_{F_i}[\omega_{ib}^*(\mu;\varepsilon_b)]\)。则 \(-\bar\omega_{ib}\in\partial G_{ib}\)；在可微点，\(\nabla G_{ib}=-\bar\omega_{ib}\)。期望额度边际价值被对应 PAYG 费率所界定。

### 3.2 无序异质性

对不同选项 \(a,b\in\mathcal B_0\)，定义

\[
\delta_{iba}=G_{ia}(\varepsilon_a)-G_{ib}(\varepsilon_b).
\]

\(\delta_{iba}\) 是旅客从 \(a\) 转向 \(b\) 愿意支付的最高价格溢价。无差异条件为 \(C_b-C_a=\delta_{iba}\)。不同类型的 \(\delta\) 排序可随套餐对变化，因此每个套餐对有自身的标量阈值，但通常不存在共同旅客排序。

### 3.3 两种选择规则的单层重构

**OBD：MPEC。** 将 \(q,\Pi\) 作为上层变量，最大化 \(\sum_iQ_i\sum_bq_{ib}M_b\)，施加第 2.3 节的 simplex、最低成本及互补约束和 \(x\in\mathcal D\)。上层对 \(q\) 的最大化实现乐观选择。

**PBD：内层 LP 的原始—对偶重构。** 对固定菜单，最不利响应解

\[
\min_{q_i\ge0}\sum_bM_bq_{ib}
\quad\text{s.t.}\quad
\sum_bq_{ib}=1,\qquad \sum_be_{ib}q_{ib}\le\Pi_i,
\]

其中同时要求 \(\Pi_i\le e_{ib}\) 对所有 \(b\) 成立。这些条件把 \(q_i\) 限定在最低成本响应集合。

令 \(\lambda_i\in\mathbb R\)、\(\eta_i\ge0\)，对应内层对偶为

\[
\max_{\lambda_i,\eta_i}\lambda_i-\eta_i\Pi_i
\quad\text{s.t.}\quad
\lambda_i-\eta_i e_{ib}\le M_b\quad(\forall b).
\]

PBD 的单层模型最大化 \(\sum_iQ_i\sum_bq_{ib}M_b\)，同时施加菜单可行性、上述原始/对偶可行性及强对偶等式

\[
\sum_bq_{ib}M_b=\lambda_i-\eta_i\Pi_i\qquad(\forall i).
\]

这个表示通过强对偶消去了内层最不利响应优化，得到非凸单层重构。即使供给成本为非线性的，一旦菜单固定，margin 仍是响应变量的固定系数，因此上述 OBD/PBD 重构仍成立。新稿将 PBD 更准确地称为“基于强对偶的非凸重构”，并未增加光滑性或凸性保证。

### 3.4 Assignment regions 与 critical travelers

固定 \(\alpha_i\in\mathcal B_0\)，定义

\[
\mathcal I_b(\alpha)=\{i:\alpha_i=b\},\quad
D_b(\alpha)=\sum_{i\in\mathcal I_b(\alpha)}Q_i,\quad
\Pi_\alpha(x)=\sum_bD_b(\alpha)M_b(x).
\]

弱选择区域为

\[
\mathcal R_\alpha=\{x\in\mathcal D:
C_b-C_a\le\delta_{iba},\quad
i\in\mathcal I_b(\alpha),\ a\ne b\},
\]

区域问题为

\[
\max_{x\in\mathcal R_\alpha}\Pi_\alpha(x)
=\max_{x\in\mathcal R_\alpha}\sum_bD_b(\alpha)
\bigl(C_b-K_b(\varepsilon_b)\bigr).
\]

一般成本下，目标关于额度也可以非线性；只有在线性成本特例中目标才是线性的。旅客选择约束仍由 \(G\) 决定。

在同一个菜单 \(x\in\mathcal R_\alpha\) 上，乐观响应

\[
\alpha_i^+\in\arg\max_{b\in\mathcal B_i^*(x)}M_b(x)
\]

满足 \(\Pi_{\alpha^+}(x)\ge\Pi_\alpha(x)\)。若某个正权重旅客切换至严格更高利润的并列最低成本选项，则严格增利。这是 optimistic boundary dominance；它比较的是同一菜单下的两个 assignment。

对每个非空 \(\mathcal I_b\)，定义

\[
\Gamma_{ba}(\varepsilon;\alpha)
=\min_{i\in\mathcal I_b(\alpha)}\delta_{iba},\qquad
\mathcal I_{ba}^{\mathrm{crit}}=\arg\min_{i\in\mathcal I_b(\alpha)}\delta_{iba}.
\]

该 pair 的全部旅客约束等价于 \(C_b-C_a\le\Gamma_{ba}\)。等号成立时，critical travelers 对 \(a,b\) 无差异，其他严格超过该最小值的类型仍严格偏好 \(b\)。critical 身份会随菜单变化，因此约束应动态生成，不能预先固定一组 critical classes。

## 4. Solution Approach

### 4.1 Bundle valuation oracle

对每个 \((i,b)\)，使用 double-description 方法枚举 \(\mathcal V_{ib}\) 极点一次，按斜率 \(\pi\) 排序并缓存。每次额度更新：

1. 更新直线截距 \(-\omega_j^{\top}\varepsilon_b\)。
2. 删除相同斜率的被支配直线，并扫描构造 \(\mu\ge0\) 上的上包络。
3. 对每个活跃区间读取 \(F_i\) 的区间概率和未归一化一阶矩。
4. 汇总得到 \(G_{ib}\) 和 \(\bar\omega_{ib}\)。

设缓存极点数为 \(J\)、方式数为 \(M\)，更新截距需要 \(O(JM)\)，已按斜率排序时包络扫描需要 \(O(J)\)。重复上层评估无需重新求解路径 LP。

当前 Sioux Falls 实验在初始化阶段使用每个 OD 的全部简单路径构造对偶约束，不再按自由流时间截取前 5 条路径。路段成本为正、方式使用量非负，因此含环路径的约束由去环路径蕴含。全部对偶极点按 OD 和旅客类型在进程内缓存，套餐及多初值共享；此前采用 5 条路径截断的实验结果不代表此完整路径配置。

当前实验参数版本为 `sioux-falls-monthly-usd-v1`：TNTP 原时间单位 0.01 小时转换为分钟，价格与归一化利润单位为 USD/人·月，额度单位为分模式路段通行次数/人·月，出行频次为单向 OD 次数/人·月。两个类型的分布分别为 Uniform(8,20) 和 Triangular(20,32,44)。经济参数为合成设置，OD 源流量仅用于归一化权重，不解释为月度人数。完整参数表与结果见[月度实验配置](../docs/sioux_falls_experiment.md)。

### 4.2 乐观选择下的区域搜索

从满足 \(\Pi^{\mathrm{opt}}(x^0)>0\) 的可行菜单开始。每次进入区域，初始化 critical sets，并在整个约束生成过程中保留该可行 entry menu 作为 incumbent。固定 assignment 时，对每个 ordered pair 维护 \(S_{ba}\subseteq\mathcal I_b(\alpha)\)，只加入这些旅客的选择约束。每轮 restricted NLP 后执行 full separation：

\[
s_{iba}=\delta_{iba}-(C_b-C_a).
\]

对存在负 slack 的 pair，加入最违反且未加入的旅客，再解 NLP。固定 assignment 内约束只增不删，直到完整区域可行。

得到区域解 \(x^*\) 后，外层按以下规则处理：

1. **严格利润改善的乐观切换**：同时更新获得更高 margin 的并列最低成本响应，以 \(x^*\) 为新区域 warm start，并在下一次区域求解中保留该可行 incumbent（当前可行解）及其目标值。
2. **Double tie 邻域检查**：若 \(e_{ia}=e_{ib}\) 且 \(M_a=M_b\)，切换在当前菜单处不改变利润。以同一 \(x^*\) 为 incumbent，用相同的约束生成流程求解相邻 assignment region；找到严格改善则退出邻区循环并继续区域搜索，没有改善时必须先认证 \(x^*\) 在该邻区局部最优，才能标记为 resolved。
3. **组合检查与终止**：多个 double ties 同时活跃时，检查它们生成的所有局部相邻 assignment 组合。当前区域和所需相邻区域全部在 \(x^*\) 处 resolved 后，才能使用局部最优定理。

单个一般边界上的 double tie 只需一个额外邻区求解。仅仅“未找到改善”或 solver 失败，不等价于局部最优证据；若 incumbent 改变，旧点处的 resolved 记录不能直接用于新点。

### 4.3 收敛与局部最优性

新版第 4.3 节区分三项结论：

| 结论 | 条件与含义 |
|---|---|
| 内层有限约束生成 | 固定 assignment、每次加入新的违反约束且不删除；终点完整区域可行。若最终 restricted NLP 的解是局部极大点，则也是完整 regional problem 的局部极大点。 |
| 外层利润单调收敛 | 有限 PAYG 成本、非负供给/出行成本、区域局部求解且不劣于可行 entry incumbent；已接受利润满足 \(P^{k+1}\ge P^k\) 并收敛到有限值。 |
| Resolved terminal point 的局部最优性 | 当前点是自身区域的局部极大点，assignment 乐观一致，所有活跃 double ties 生成的局部邻区也已在该点 resolved；可推出原 OBD 的局部最优性。 |

利润上界来自选择 PAYG 的可能性。对完整 assignment region 中被分配到 \(b=\alpha_i^k\) 的类型 \(i\)，有

\[
C_b+G_{ib}(\varepsilon_b)\le e_{i0},\qquad
M_b=C_b-K_b(\varepsilon_b)\le C_b\le e_{i0},
\]

故

\[
0<P^0\le P^k\le\sum_{i\in\mathcal I}Q_i e_{i0}<\infty.
\]

这给出无需紧性的利润收敛论证，也说明每个已接受迭代均不会落入全 PAYG 区域。该上界适用于完整选择约束可行的响应，不应直接套用到尚未通过 full separation 的 restricted NLP 试探点。

局部最优证明也进一步明确：若相邻 assignment 把某个类型切换至严格更低 margin 的并列选项，其利润在足够小邻域内低于终点利润 \(\Pi_{\alpha^*}(x^*)\)；等 margin 的 double-tie 邻区则需要在同一个 \(x^*\) 上认证局部极大性。

外层结论是**目标值收敛**，不直接保证有限次区域切换、菜单序列收敛或全局最优。仅有一阶 KKT 检查也不足以证明非凸区域问题的局部极大性。

## 5. Numerical Results 与实现状态

最新稿已删除 Numerical Results 中旧的算法批注、no-double-tie 有限切换讨论及重复公式；该节目前为空，待补充实验设计和结果。算法依据为第 4.2–4.3 节。

仓库已有 two-route 单次求解、81-assignment 穷举比较、多初值实验，以及 Sioux Falls 单次/多初值实验，运行方式见 [项目 README](../README.md)。它们目前运行的是线性成本特例下的 optimistic consistency finder，不能作为非线性供给成本扩展的实验结果。

| 部分 | 当前代码状态 |
|---|---|
| 对偶极点缓存、上包络、期望与影子价格 | 已实现，供给成本推广不改变路径 oracle |
| 一般供给成本及其导数 | 已实现 `ProvisionCost` / `CallableProvisionCost`，可提供解析梯度或使用数值差分；兼容共享线性参数，见 [接口说明](../docs/provision_cost.md) |
| Optimistic assignment、区域 NLP、full separation | 已实现 |
| Assignment 更新、warm start、候选诊断 | 已实现 |
| 数值容差 | 成本 tie/选择可行性从初值统一；margin 搜索与诊断共享判定；总利润、价格坐标、额度坐标分别设置误差。见 [数值容差](../docs/numerical_tolerances.md) |
| 正利润初值、外层 incumbent 的利润保护与单调接受 | 已实现；下降或求解失败时保留旧菜单并报告未收敛；内层保留 entry 可行锚点，未逐次设置 restricted NLP 目标下限 |
| 边界 assignment 切换 | 普通 tie 按 optimistic rule 切换，double tie 额外探索；异点继续优化，同点双方检查完成后才结束往返；未独立认证二阶最优性或多重交点联合方向，见 [边界搜索](../docs/boundary_assignment_search.md) |
| PBD 单层 NLP、直接 OBD MPEC 求解器 | 未实现 |
| 通用 regional local-optimality 证书 | 未实现；现有 adapter 验证可行性和一阶 KKT |

因此当前 `CONSISTENT_CANDIDATE` 表示通过已有检查的分配一致候选，并不自动等价于新稿的 resolved terminal point。
