# 实验代码与算例数据

`examples/` 定义具体问题；`benchmarks/` 决定如何运行和比较实验。
`design/` 实现优化算法，`oracle/` 实现下层值函数。

| 位置 | 职责 |
| --- | --- |
| `examples/two_route.py`、`examples/sioux_falls.py` | 网络、旅客、问题构造、固定初始菜单、单个候选菜单抽样 |
| `examples/sioux_falls_config.py` | 带单位的经济参数、初始菜单与抽样范围 |
| `benchmarks/settings.py` | 求解预算、NLP 精度、正利润门槛、初值数量和随机种子的默认值 |
| `benchmarks/initialization.py` | 有尝试次数上限的正利润拒绝采样 |
| `benchmarks/multistart.py` | 对通用问题执行多初值搜索并汇总 |
| `benchmarks/assignment_enumeration.py` | 对小问题枚举 assignment，比较区域解 |
| `benchmarks/two_route.py`、`benchmarks/sioux_falls.py` | 将具体算例与实验方法连接，提供单次、多初值及小算例比较入口 |
| `cli.py` | 解析命令、调用 benchmark、格式化 JSON 并设置退出码 |

依赖方向为 `cli → benchmarks → examples`，benchmark 同时调用 `design`。
`examples` 只依赖基础问题对象和 oracle，不导入 benchmark 或区域求解器。
通用多初值、枚举和筛选模块不依赖具体网络。

## 运行 Sioux Falls 多初值实验

命令保持不变：

```bash
maas-bundle multistart sioux-falls --starts 30 --seed 20260901 > sioux_falls_multistart.json
```

Python 入口现在从 `benchmarks` 导入：

```python
from maas_bundle.benchmarks import run_sioux_falls_multistart

result = run_sioux_falls_multistart(30, seed=20260901)
for run in result.runs:
    print(run.number, run.diagnostics.status.value, run.diagnostics.valid, run.true_profit)

best = result.best_run
if best is not None:
    print(best.diagnostics.candidate.menu)
```

多初值运行仍为串行，同一进程内共享问题及 oracle。每次运行包含自己的区域搜索状态。
CLI 的 `experiment` 字段记录算例数据与单位；`benchmark_settings` 字段单独记录运行预算和精度。
多初值 JSON 另外记录实际 `starts` 和 `seed`。进程退出码为零表示至少有一个
assignment-consistent 结果，不代表所有初值均完成；逐次详细状态可从 `result.runs` 读取。

## 分别修改数据与实验设置

```python
from dataclasses import replace

from maas_bundle.examples import DEFAULT_SIOUX_FALLS_CONFIG
from maas_bundle.benchmarks import (
    DEFAULT_SIOUX_FALLS_SETTINGS,
    run_sioux_falls_multistart,
)

config = replace(
    DEFAULT_SIOUX_FALLS_CONFIG,
    sampled_price_bounds_usd_per_month=(6.0, 60.0),
)
settings = replace(
    DEFAULT_SIOUX_FALLS_SETTINGS,
    max_region_iterations=60,
    nlp_max_iterations=3000,
)
result = run_sioux_falls_multistart(
    30, seed=20260901, config=config, settings=settings,
)
```

注意：`initial_prices_usd_per_month` 控制单次实验的固定初值；多初值实验的随机价格
由 `sampled_price_bounds_usd_per_month` 控制。若要修改多初值抽样范围，请修改后者。

`config` 进入问题和 oracle 的构造；`settings` 仅控制运行。
因此修改区域预算或 NLP 精度不再改变 oracle 缓存键。
`settings.profit_tolerance` 同时传给初值正利润筛选与上层正利润检查，单位与目标函数一致。
Sioux Falls 默认是 USD/人·月。

## 通用接口

具体 benchmark 的内部流程是：

```text
build_*_problem(config) → BundleDesignProblem
                      ↓
draw_*_menu(random, config) → 候选 BundleMenu
                      ↓
sample_profitable_menus → 正利润初值序列
                      ↓
run_multistart(problem, menus, solver, ...) → MultiStartResult
```

`draw_*_menu` 只生成候选，不承诺正利润。`generate_*_initial_menus` 位于 benchmark，
负责固定随机种子、复用问题并调用拒绝采样，输出符合利润门槛的菜单序列。
固定初值 `initial_*_menu` 仍由 example 提供，`solve_primal` 检查其实际利润。

## Python 接口迁移

原来从 `maas_bundle.examples` 或其具体子模块导入的以下函数，现在统一从
`maas_bundle.benchmarks` 或对应 benchmark 子模块导入：

- `run_two_route_example`、`run_sioux_falls_example`。
- `run_two_route_multistart`、`run_sioux_falls_multistart`。
- `generate_two_route_initial_menus`、`generate_sioux_falls_initial_menus`。

`compare_two_route` 是公开的 benchmark 比较入口，不再依赖 CLI 私有函数。
`build_*`、`initial_*` 仍从 `examples` 导入；新增 `draw_*_menu` 提供未筛选的单次抽样。

原 `SiouxFallsExperimentConfig` 中的 `max_region_iterations`、`nlp_max_iterations`、
`nlp_ftol` 改由 `BenchmarkSettings` 提供；`minimum_profit_usd_per_person_month` 对应
`settings.profit_tolerance`。初值数量与种子通过入口的 `start_count`、`seed` 参数设置，
默认值定义于 `benchmarks/settings.py`。examples 不保留反向导入 benchmark 的兼容转发。

## 测试分工

`tests/examples/` 检查网络、数据、配置和 oracle 构造；`tests/benchmarks/` 检查初值筛选、
实验组织及端到端求解。`tests/test_architecture.py` 检查依赖方向，防止运行逻辑回流到数据层。
完整 Sioux Falls 求解测试已迁移至：

```bash
pytest tests/benchmarks/test_sioux_falls_experiments.py --run-sioux-falls
```

2026-09-29 架构调整后的完整回归：438 项通过、0 失败、0 跳过，170.85 秒。
包含小算例单次、81 个 assignment 枚举、30 初值、100/1000 区域规模测试、
完整 Sioux Falls 单初值和独立路段流 LP 对照；本次未运行 Sioux Falls 的 30 初值优化。
此外直接运行 CLI 小算例成功，仍为 2 次区域求解、利润约 3.54506977。

```bash
python -m pytest --run-regional-scale --run-enumeration --run-multistart --run-sioux-falls -q
```
