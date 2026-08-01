# 问题一～四代码

```bash
pip install -r code/requirements.txt
python code/q1.py
python code/q2.py
python code/q3.py
python code/q4.py
python code/finalize_results.py           # 统一细网格第三问 + 验证
python code/finalize_results.py --skip-repeats
```

| 文件 | 内容 |
|------|------|
| `q1.py` | 问题一：标定 + 预测 |
| `q2.py` | 问题二：最大速度 |
| `q3.py` | 问题三：最小阴影面积 |
| `q4.py` | 问题四：面积–对称性 Pareto |
| `finalize_results.py` | 官方网格统一 + 收敛/敏感性/工程解/算法重复 |
| `requirements.txt` | 依赖 |

官方报告时间步长：**dt = 0.025 s**（与问题四 `dt_verify` 一致）。验证输出在 `results/verification/`。
