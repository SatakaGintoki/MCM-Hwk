# 官方结果口径（定稿）

- 统一报告时间步长: dt = 0.025 s
- 问题一预测 / result.csv / 问题二最大速度 / 问题三面积 / 问题四指标均按此口径

## 定稿数值

| 项目 | 数值 |
|------|------|
| 问题二 v_max | 85.68 cm/min |
| 问题三理论最优 A3* | 344.496592 °C·s |
| 问题四最终 A_L | 358.152877 °C·s |
| 问题四最终 J_sym | 0.16291033 |

## 说明

- 第三问官方解已与第四问细网格面积端点 endpoint_area 对齐。
- 工程解（安全余量解）见 results/verification/q3_theoretical_vs_engineering.*
- 参数扰动下工程解可行率 75%，理论解 25%；不宜称工程解为鲁棒最优。
- 若根目录 result.csv 被 Excel 占用，以 results/q1/result.csv 与 result_q1.csv 为准。
