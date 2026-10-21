# 固定分配问题的凸化验证（2026-10-21）

在当前 Sioux Falls 五模式实例上验证了两阶段路线：**可行 CCP（逐次凸内近似）构造改善后的可行初值，然后用原始约束的 SLSQP 做最终优化与一阶验收。** 每个凸子问题用 LP 切平面求解，并用原始积分 oracle 检查全部选择约束。

这是固定 assignment 的求解原型，不是原问题的等价单个凸模型，不提供全局最优或区域局部最优证明。本次没有改写论文或替换生产求解器。

## 实际适用条件

- 当前实例包含 1,056 类旅客、PAYG 加两个付费套餐、五维 allowance，共 12 个上层变量。
- `G_ib` 凸，已有 oracle 同时提供精确分段积分与次梯度 `-E[omega*]`。这里的“精确”指不做需求采样或数值积分近似，浮点计算仍有舍入误差。
- 当前 `LinearProvisionCost` 使用单位成本 `(0.05, 0.08, 0.02, 0.15, 0.10)`，固定 assignment 的利润为线性函数。
- 每个起点先按原模型确定 assignment，该 assignment 在本次内层实验中保持不变。算法从其可行菜单开始。
- 直接读取 `results/sioux_falls_five_modes/dual_vertices.npz`，不重新枚举下层极点。

论文允许一般连续、非凸的 `K_b`。本次验证不把该一般情形称为凸子问题；这条结论适用于已经运行的线性成本实例，也适用于凸成本版本。

## 实际求解的约束

原选择约束为

\[
h_{iba}(C,\varepsilon)
=C_b-C_a+G_{ib}(\varepsilon_b)-G_{ia}(\varepsilon_a)\leq0.
\]

在第 `k` 次更新，使用 oracle 返回的
\(g_{ia}^k\in\partial G_{ia}(\varepsilon_a^k)\)，构造

\[
\ell_{ia}^k(\varepsilon_a)
=G_{ia}(\varepsilon_a^k)
+(g_{ia}^k)^\top(\varepsilon_a-\varepsilon_a^k).
\]

实际凸子问题保留原利润目标，将选择约束替换为

\[
\widehat h_{iba}^k
=C_b-C_a+G_{ib}(\varepsilon_b)-\ell_{ia}^k(\varepsilon_a)\leq0.
\]

价格非负、价格排序和 allowance 非负约束保持线性。原型还限制单步 allowance 变化不超过 48、单步价格变化不超过 100；这些是算法步长盒，不是原模型的新经济约束。

由于 \(\ell_{ia}^k\leq G_{ia}\)，有
\(h_{iba}\leq\widehat h_{iba}^k\)。因此凸子问题的可行点满足原选择约束。在当前点两者相等，且当前点位于步长盒中，因此精确求解时，每步都有可行 incumbent，固定 assignment 利润不下降。使用次梯度也保留该结论。

实际数值验收使用 `1e-8` 的约束容差和 `1e-7` 的利润不下降容差。

## 如何求解含积分函数的凸子问题

不需要 CVXPY 的符号积分表达式。对每个凸约束，oracle 提供其值和次梯度，以此建立支撑线性不等式。初始 LP 包含全部 2,112 个选择约束在当前点的支撑切面，以及价格排序约束。

1. 用 SciPy/HiGHS 求解 LP 外松弛。
2. 用原始积分 oracle 计算全部凸约束。
3. 对违反凸约束的点加入新的支撑切面，重新求解 LP。
4. LP 最优点满足全部凸约束至 `1e-8` 后，检查全部原始选择约束，再接受更新。

LP 外松弛给出凸子问题的利润上界；若 LP 最优点本身满足凸子问题，则在精确算术下也达到凸子问题最优值。本次只声称该关系在记录的浮点及可行性容差下通过检查，不将微小违反量当作严格可行性证明。

标准方法依据：[Lipp and Boyd, Variations and extension of the convex–concave procedure](https://web.stanford.edu/~boyd/papers/pdf/cvx_ccv.pdf)，Algorithm 1.1、Section 1.3。该文明确说明可行性和目标单调性，也说明目标值收敛不保证达到局部最优点。

## 已排除的具体方案

### 仅合并积分后宣称凸性

在当前缓存 oracle 中找到一个实际反例。对于 traveler 427、选择套餐 2 相对套餐 1 的约束，两个端点的价格非负且满足排序，选择约束残差都为零，中点残差为 **24.0468661368 USD/月**。完整 allowance、价格和复算结果保存在实验 JSON 中。

这直接证明该条约束在原变量下的可行集非凸。仅合并积分没有改变这个集合。也不能用“增加连续辅助变量、得到一个凸集合后再投影回原变量”的方式精确表示该集合，因为凸集的投影仍然凸。这不排除其他非线性坐标变换，也不单凭一条约束断言完整 assignment 区域必然非凸。

### 一个共同标量支付意愿坐标

在 allowance `(16.123,16.456,4.789,6.246,6.357)`，跨全部 traveler 的估值 Jacobian 的五个奇异值约为

`(33.50834, 2.46442, 1.11163, 0.72169, 0.25544)`。

中心差分与 oracle 导数最大差异为 `1.91e-10`。该点 Jacobian 的秩为 5；若所有 class 的估值都由同一个可微标量坐标生成，其 Jacobian 秩至多为 1。因此前述共同标量变换不适用于当前实例。

### 将各 class 的支付意愿作为互相独立的自由变量

同一个套餐的所有 class 估值必须由同一个五维 allowance 实现。删除该一致性关系就改变了可行菜单集合；这种写法不能作为等价凸化。此次不采用它。

### 消去价格后直接宣称完整问题凸

固定 allowance 的价格 LP 是成立的，审计程序也独立求解了它。但消去价格本身并未证明外层 allowance 问题凸，因此不能作为完整凸化结论。

## 复现

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/homebrew/Caskroom/miniforge/base/envs/maas-bundle/bin/python \
  scripts/check-regional-convexification.py \
  --output results/regional_convexification_20261021/verification.json \
  --starts 3 --outer-iterations 15

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/homebrew/Caskroom/miniforge/base/envs/maas-bundle/bin/python \
  scripts/audit-regional-convexification.py \
  results/regional_convexification_20261021/verification.json
```

第一个起点是当前默认菜单，另两个来自既有多起点实验 `start_002.json`、`start_003.json` 的初始菜单。它们只用于固定 assignment 验证，不与历史完整区域搜索的最终利润排名混用。

## 独立审计发现的限制

单独的 CCP 原型共接受 37 次更新，所有记录点的原始选择约束违反量低于 `1e-8`，利润不下降。三个起点的固定 assignment 利润分别从 `13.726681`、`4.021766`、`2.078446` 提高到 `14.519743`、`11.124883`、`10.567467`。

但是，三个终点在现有 oracle 次梯度下的一阶 KKT 拟合残差分别为 `0.04117`、`0.004333`、`0.001460`，没有通过现有验收。这不单凭任意次梯度的拟合残差证明非光滑点不是局部最优，但足以排除“仅凭 CCP 小改进终止就接受为区域优化成功”的工程方案。

因此增加最终原约束优化：`SLSQP`、`maxiter=200`、`ftol=1e-11`，恢复与跨后端回退均关闭。最终必须检查原始约束和一阶 KKT，不接受仅有原生 solver success 的点。复现该阶段：

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /opt/homebrew/Caskroom/miniforge/base/envs/maas-bundle/bin/python \
  scripts/audit-regional-convexification.py \
  results/regional_convexification_20261021/verification.json --polish
```

逐次凸化阶段的独立复算见 `results/regional_convexification_20261021/audit.json`，最终原约束优化的单独记录见同目录 `audit_polish.json`。

两阶段方法的三个最终点全部通过现有原约束与 KKT 验收，结果如下。利润单位为 USD/代表性加权旅客/月；耗时为该次运行的阶段一加阶段二墙钟时间，不代表与原求解器的速度比较。

| 起点 | 初始固定分配利润 | 最终固定分配利润 | 原始选择约束最大违反量 | 归一化 KKT 残差 | 合计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 默认菜单 | 13.726681 | 14.519743 | 1.42e-13 | 3.33e-7 | 50.2 s |
| 既有 start 002 | 4.021766 | 11.125098 | 7.96e-13 | 6.00e-10 | 171.8 s |
| 既有 start 003 | 2.078446 | 10.567711 | 1.56e-12 | 2.85e-9 | 184.3 s |

三个最终利润已另行按 class 权重、价格和线性 allowance 成本直接重算。原始积分 oracle 和需求分布相关的 49 项已有测试通过；两个验证脚本通过 Ruff 检查。

## 保证的边界

该原型验证凸子问题可解、接受点的原约束可行性和固定 assignment 利润不下降。小改进终止或一阶 KKT 检查都不能单独证明区域局部最优。两阶段方法的最终验收与现有 NLP 求解器处于同样的一阶数值保证层次，仍不替代论文局部极大点假设所需的二阶或问题专门论证。
