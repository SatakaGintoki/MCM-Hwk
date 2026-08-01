# 问题一～四代码

```bash
pip install -r code/requirements.txt
python code/q1.py
python code/q2.py
python code/q3.py
python code/q4.py
python code/finalize_results.py    # 统一细网格第三问 + 验证
python code/closeout_fix.py       # 收口：一二问 dt=0.025、重跑四问、收敛/敏感性
```

| 文件 | 内容 |
|------|------|
| `q1.py` | 问题一（预测 dt=0.025，result.csv） |
| `q2.py` | 问题二（dt=0.025） |
| `q3.py` | 问题三 |
| `q4.py` | 问题四 |
| `finalize_results.py` / `closeout_fix.py` | 结果统一与收口 |
| `requirements.txt` | 依赖 |

官方报告时间步长：**dt = 0.025 s**。详见 `results/OFFICIAL_RESULTS.md`。
