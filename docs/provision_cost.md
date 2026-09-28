# 一般供给成本接口

`BundleDesignProblem` 接受 `provision_cost`，表示每位订阅者的供给成本
\(K_b(\varepsilon_b)\)。允许套餐特定、非线性、非凸、非单调函数，以及同一套餐内
不同模式额度之间的交叉项。每个套餐的成本只能依赖该套餐自身的额度。

## 使用方式

```python
import numpy as np
from maas_bundle import BundleDesignProblem, CallableProvisionCost

# 输入只含订阅套餐，不含 PAYG；e.shape == (B, M)。
cost = CallableProvisionCost(
    mode_count=2,
    value=lambda e: 0.1 * e.sum(axis=1) + 0.02 * (e.sum(axis=1)) ** 2,
    gradient=lambda e: np.broadcast_to(
        0.1 + 0.04 * e.sum(axis=1, keepdims=True), e.shape
    ),
)
problem = BundleDesignProblem(
    traveler_weights=weights,
    provision_cost=cost,
    route_oracles=oracles,
    demand_distributions=distributions,
)
```

`value` 返回 `(B,)`，`gradient` 返回 `(B, M)`。行顺序始终对应套餐 `1..B`，
因此可用按行系数表达不同套餐的成本。回调必须确定性计算且不能修改输入；算法传入
只读额度数组。模型参数在一次求解期间应保持不变。也可直接实现公共 `ProvisionCost`
协议的 `mode_count`、`values(e)` 和 `gradients(e)`，接入自动微分或其他计算后端。

省略 `gradient` 时，`CallableProvisionCost` 使用三点数值差分；在零额度附近使用
前向差分，避免评估负额度。建议能提供解析梯度时显式提供。

旧写法 `BundleDesignProblem(..., unit_allowance_cost=k)` 继续可用，内部转换为
`LinearProvisionCost(k)`；它与 `provision_cost` 二选一。已有问题也可通过
`dataclasses.replace(problem, provision_cost=cost)` 更换成本模型。
兼容属性 `problem.unit_allowance_cost` 在线性模型下返回只读系数，一般模型下返回
`None`。新代码统一使用 `problem.provision_cost`。

## 数学约定与求解范围

与论文一致，成本非负、连续，满足 \(K_b(0)=0\)，并在使用导数的位置连续可微。
初始化检查零额度成本；每次计算检查值/梯度的形状及有限性，并检查成本非负。
连续性、光滑性和套餐间独立性由成本模型提供者保证，运行时不作全域验证。
PAYG 不交给成本回调，成本和 margin 固定为零。

利润、optimistic margin tie-break、正利润初值筛选、区域 NLP 与 incumbent 验收
统一使用这个接口。区域目标及其梯度为

\[
\Pi_\alpha=\sum_bD_b(C_b-K_b(\varepsilon_b)),\qquad
\partial_{C_b}\Pi_\alpha=D_b,\qquad
\nabla_{\varepsilon_b}\Pi_\alpha=-D_b\nabla K_b.
\]

接口不限定函数形状，但当前 SLSQP / `trust-constr` 是局部 NLP 求解器。
不可微折点、阶跃或不连续函数需要相应的求解方式；提供数值差分并不能自动得到
这些情况下的收敛保证。非凸成本也不保证全局最优。

## 性能与验证

成本按套餐批量计算，不按旅客重复计算；同一菜单的区域利润/梯度缓存复用，
仅计算目标时不触发路径估值。路径 oracle、choice constraints 和 full separation
保持原有形式。`trust-constr` 对线性成本保留零目标 Hessian；一般成本采用 BFGS
曲率近似，SLSQP 使用其原有拟牛顿机制。

接口本身不增加变量或选择约束数量，但总耗时仍取决于成本求值、导数和目标曲率。
未提供梯度时，每次梯度约需 `2*M` 次额外成本求值，在边界还需要基准值。
不能由“交给求解器”推出性能一定不变。

`tests/design/test_cost.py` 覆盖线性兼容、套餐特定 tie-break、跨模式差分、
非凸非单调成本、初值筛选和利润验收，以及两种求解器的非线性解析最优解检查：
在选择约束给出 \(C\le\varepsilon\) 的实例中，令 \(K=\varepsilon^2\)，
验证最优 \(C=\varepsilon=1/2\)、利润 \(1/4\)。现有 two-route / Sioux Falls
默认实验参数仍采用共享线性成本，不能当作一般非线性成本的规模实验。
