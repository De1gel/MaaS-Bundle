# 数值容差与统一判定

同一种比较在整个搜索中使用同一标准；不同物理量分别配置。
默认值集中在 `src/maas_bundle/design/tolerances.py`。

| 参数 | 默认值 | 比较对象及单位 |
| --- | --- | --- |
| `choice_tolerance`、`violation_tolerance` | `1e-7` | 用户广义成本差、选择约束残差；每位旅客的货币成本 |
| `margin_tolerance` | `1e-9` | 套餐单位利润差；每位订阅者的货币利润 |
| `profit_tolerance` | `1e-9` | 加权总利润改善、邻区同点利润比较、初始正利润门槛；与目标函数同单位 |
| `boundary_tolerance` | `1e-7` | 同点判定中的价格绝对误差；与价格同单位 |
| `allowance_boundary_tolerance` | `1e-7` | 同点判定中的额度绝对误差；与额度同单位 |
| `boundary_relative_tolerance` | `1e-6` | 价格和额度分量的相对误差；无量纲 |

例如，Sioux Falls 的成本、margin 和归一化目标均以 USD/人·月计，
额度以路段次/人·月计。通用问题的旅客权重未必归一化；改变权重总规模时，
应按目标函数尺度设置 `profit_tolerance`，不能因此改变 `margin_tolerance`。
当前采用显式绝对成本/利润容差，不自动按权重或菜单大小缩放。

## 成本与可行性

在初始 assignment 和正利润检查之前，固定：

```text
effective_cost_tolerance = max(
    choice_tolerance,
    violation_tolerance,
    solver.feasibility_tolerance,
)
```

之后初始/求解后 assignment、入口可行锚点、full separation、约束扩充后的
回溯以及保留点复核，都使用这个有效成本/可行性容差。低层独立调用
`solve_fixed_assignment` 也会先协调求解器精度。枚举基准和多初值运行沿用相同政策。

SciPy adapter 可以采用更严格的验收标准；外层不会在搜索途中因为某次返回的
certificate 更宽松，就临时扩大 tie 集。求解结果报告的约束违反量超过固定阈值时
拒绝该结果，遗漏的选择约束仍由 full separation 检查。
`FixedAssignmentResult.feasibility_tolerance` 记录实际采用的阈值。

自定义 adapter 可提供固定的 `feasibility_tolerance` 属性；未提供时按 `1e-7`
协调。若希望整次运行采用更严格阈值，需要同时配置 choice、violation 和 adapter，
不能只收紧其中一个参数。adapter 的非负坐标和价格顺序残差仍使用其可行性阈值，
KKT 驻点残差、活跃集阈值保留各自意义；这些不作为 margin 判定标准。

## Margin、double tie 与最终验证

对每位旅客先生成：

```text
cost_ties = cost - minimum_cost <= effective_cost_tolerance
optimistic_options = cost_ties & (best_margin_among_cost_ties - margin <= margin_tolerance)
```

optimistic assignment 选第二个集合中编号最小的选项；多于一个选项即为数值 double tie。
额外邻区生成、`AssignmentResult.has_double_ties` 和最终诊断共用
`numerical_optimistic_mask`。所有 margin 比较都相对同一个最高 margin，避免
“A 接近 B、B 接近 C”被错误当成“A、B、C 都在同一容差范围”。

`solve_primal(..., margin_tolerance=...)` 现在控制整个搜索，而非仅控制最终诊断。
`validate_candidate(candidate)` 继承已搜索的标准。若显式传入不一致的
`margin_tolerance`，直接报错，不能沿用原来的“边界搜索完成”标记重新解释结果。

兼容旧接口：低层 `select_optimistic_bundles`、`evaluate_assignment`、
`AssignmentResult` 和枚举基准的 `profit_tolerance` 名称仍表示 **margin** 容差；
`AssignmentResult.margin_tolerance` 提供明确的读取名称。外层
`solve_primal` / `find_assignment_consistent_menu` / `run_multistart` 的
`profit_tolerance` 则只表示总利润门槛，margin 必须通过独立参数设置。

总利润容差不允许 incumbent 持续下降：普通接受仍保持实际利润不下降，
邻区数值同点返回时保留原菜单及其利润。

## 其他阈值

`critical_tie_tolerance=1e-12` 仅用于初始 critical travelers 筛选时的并列截断，
不定义用户选择 tie，也不参与最终可行性验收；遗漏约束仍会被 full separation 找出。
它不需要改成 `1e-7`。

同菜单判定仍使用逐分量的绝对加相对误差：价格和额度分别取自己的绝对容差。
坐标近似相同还不够，必须通过保留点的区域复核与独立总利润比较。

## 回归覆盖

新增回归覆盖初始/最终成本判定一致、数值可行入口仍可作为回溯锚点、
后续较宽 certificate 不改变搜索阈值、margin 搜索与验证一致、验证拒绝换标准、
三选项近似相等不传递、总利润阈值不改变 margin tie 集、价格与额度分别设定误差，
以及非法容差在搜索前被拒绝。

2026-09-29 完整回归：`424 passed`，0 失败、0 跳过，耗时 171.52 秒。
命令为：

```bash
python -m pytest --run-regional-scale --run-enumeration --run-multistart --run-sioux-falls -q
```

包含 81 个 assignment 的小算例枚举、30 初值实验、100/1000 类型区域规模测试、
Sioux Falls 完整网络单初值求解和独立路段流 LP 对照。Sioux Falls 仍为 5 个区域、
利润约 `10.16182917 USD/人·月`。这里的 30 初值优化是小算例；没有新增运行
Sioux Falls 的 30 初值优化。
