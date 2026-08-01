# MCM-Hwk

数学建模新题目的文件与版本管理仓库。

## 目录结构

```text
MCM-Hwk/
├─ problem/          题目原文与附件说明
├─ data/
│  ├─ raw/           原始数据，只做备份，不直接修改
│  └─ processed/     清洗、转换后的建模数据
├─ code/             数据预处理、建模与检验代码
├─ paper/            论文源文件
├─ figures/          论文图表与流程图
├─ results/          模型输出和结果表格
├─ references/       参考文献
└─ notes/            思路讨论与阶段记录
```

## 建议工作流程

1. 将题目和附件放入 `problem/`，原始数据放入 `data/raw/`。
2. 数据清洗结果保存到 `data/processed/`，不要覆盖原始文件。
3. 各问题代码按 `q1_`、`q2_` 等前缀分别保存到 `code/`。
4. 论文、图片和结果分别放入 `paper/`、`figures/` 和 `results/`。
5. 每完成一个可验证阶段提交一次 Git 版本。

