# 本地编译论文

使用仓库根目录 `MaaS-Bundle` 打开 VS Code。`Paper/MaaS bundle design/`
是 Overleaf 的本地同步目录；所有本地编译产物（PDF、参考文献、日志、
辅助文件和 SyncTeX）统一写入同级的 `Paper/build/`，位于同步目录之外。
`.gitignore` 只控制 Git，编译产物不会被 Overleaf 同步的原因是它们位于同步目录之外。

本机需要 TeX 发行版中的 `latexmk`、XeLaTeX 和 BibTeX，目前已安装。

## 在 VS Code 中编译

- 打开同步目录中的本地 `.tex` 文件并保存，LaTeX Workshop 会自动编译。
- 默认使用 `latexmk (xelatex)`，自动处理参考文献和多轮编译。
- 按 `Cmd+Shift+B` 可运行 `Build paper locally`，编译主论文。
- 使用 LaTeX Workshop 的 `View LaTeX PDF file` 命令查看当前论文的 PDF。

## 在终端中编译

在仓库根目录执行：

```bash
bash scripts/build-paper.sh
```

输出主论文：
`Paper/build/MaaS bundle design under uncertain trip frequency.pdf`。

编译另一篇独立文档：

```bash
bash scripts/build-paper.sh Soft_constraints_and_chance_constraints.tex
```

脚本会自动创建输出目录，并从论文源文件目录运行，以正确读取 `ref.bib`
和 `figures/`。编辑器配置和脚本都放在 Overleaf 同步目录之外。
