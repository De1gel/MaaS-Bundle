# Paper 文件说明

当前稿件：**Optimistic and Pessimistic Mobility-as-a-Service Bundle Design**。2026-09-24 本次同步采用最新修订稿，关键标识为一般供给成本 `K_b` 与 PAYG 利润上界 `eq:payg-profit-upper-bound`。

- [MaaS_bundle_design.tex](./MaaS_bundle_design.tex)：当前论文规范源文件，与本次提供的稿件逐字节一致，保留原稿作者批注。
- [full.md](./full.md)：中文 Markdown 阅读版，概括模型、单层重构、区域搜索和理论条件；不是逐段翻译。
- `MaaS_bundle_design.pdf`：旧版本此前已删除，仓库当前未提供新 PDF。
- `manifest.json`、`content_list.json`、`_llm_source.json` 和 `images/`：旧 PDF 导入时产生的历史辅助材料，保留原始来源信息，不作为当前模型或 Markdown 的索引。

## 相比上一稿的改动

| 位置 | 本次变化 | 文档同步 |
|---|---|---|
| 2.4：平台供给成本 | 共享线性成本推广为套餐特定的非负连续 `K_b(ε_b)`，零额度成本为零，在使用导数处连续可微；线性形式是特例 | margin、regional objective、导数说明及实现差距统一更新 |
| 3.3 / 4：单层重构 | 明确非线性成本不影响固定菜单下响应 LP 的线性性；PBD 改称基于强对偶的非凸重构 | 保留原始—对偶条件，避免默认目标线性或模型光滑 |
| 3.4：critical travelers | 补充 critical 身份随额度变化，解释动态生成约束的必要性 | 保留动态 critical-set 逻辑，避免预先固定类型 |
| 4.2 / Algorithm 2：区域搜索 | 明确初值利润严格为正、每次进入区域初始化约束集，并在整个约束生成中保留可行 entry incumbent | 更新算法输入、伪代码和内外层保护的实现说明 |
| 4.2 / Algorithm 2：double ties | 遍历未解决邻区；邻区也用相同 constraint generation；有改善则继续搜索，否则须先认证 incumbent 局部最优再标记 resolved | 明确失败或仅未找到改善不能作为认证 |
| 4.3：利润收敛 | 移除菜单域紧性假设，用有限 PAYG 成本及非负成本证明 `P^k <= sum_i Q_i e_i0`；由正初值推出已接受点不进入全 PAYG 区域 | 移除紧性缺口提示，补充利润上界推导及适用条件 |
| 4.3：局部最优证明 | 严格低 margin 邻区与固定终点利润比较，double-tie 邻区必须在该终点认证 | 同步局部最优条件，保留与一阶 KKT 的区别 |
| 5：Numerical Results | 删除旧算法批注、重复公式和 no-double-tie 有限切换讨论，目前仅保留章节标题 | 删除旧段落及符号残留的过期提示，标记实验章节待补充 |

## 当前实现对应关系

代码已实现正利润初值筛选、外层 incumbent 利润保护，以及支持一般供给成本的区域 NLP 和约束生成。内层保留 entry menu 作为可行回溯锚点，目标保护在外层验收时执行；尚未逐次对 restricted NLP 设置 entry 利润下限。

一般 `K_b` 已通过 `ProvisionCost` / `CallableProvisionCost` 接入，支持解析梯度或数值差分，并兼容原线性参数；使用方式与求解限制见 [成本接口](../docs/provision_cost.md)。普通 tie 已按 optimistic rule 切换，double tie 额外探索；异点返回须重新优化，同点且双方已检查才完成往返，见 [边界搜索](../docs/boundary_assignment_search.md)。独立二阶局部最优认证、直接 PBD 求解器尚未实现。现有网络实验默认仍采用线性成本。实现设计见 [软件架构](../docs/software_architecture.md) 和 [区域搜索算法](../docs/primal_critical_traveler_algorithm.md)。

## 原稿中仍待补充的内容

- Introduction 的 Main ideas 仍使用 bundle-specific link capacity / expected flows 的旧措辞；正文第 2.2 节已将额度解释为预付使用量与 PAYG 超额量的划分。
- Introduction 仍有作者批注，Numerical Results 尚无实验设计或结果。
- 文献引用使用 `\bibliography{ref}`，仓库尚缺 `ref.bib`。

紧性假设、Algorithm 2 中未认证便标记 resolved，以及 Numerical Results 中 `\Delta_{iba}` / `\delta_{iba}` 混用的旧提示已随本次稿件修订移除。

## 编译

在具备相关 LaTeX 宏包并补齐 `Paper/ref.bib` 后，可从 `Paper/` 运行：

```bash
latexmk -pdf MaaS_bundle_design.tex
```

未补齐 bibliography 时，LaTeX 可以生成带未解析文献引用的草稿 PDF，不能视为完整出版稿。验证用 PDF 在临时目录生成，不写入仓库。

本次验证：两轮 `pdflatex` 编译成功，内部交叉引用全部解析；Markdown 本地链接、标题锚点、公式分隔符与代码围栏检查通过。仍有缺失参考文献、未设置作者及页面高度溢出等原稿警告。已核对 `src/`、`tests/` 的 Python 文件均未改变。
