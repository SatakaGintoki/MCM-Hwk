"""
收口脚本：修复审查指出的文件/口径问题（不写论文）。

1. 用 results/q1/result.csv 覆盖根目录 result.csv
2. 重跑 Q1/Q2（dt=0.025）
3. 修正 Q3 at_bound 并刷新 summary
4. 完整重跑 Q4
5. 统一收敛表（含 0.0125）+ 工程解敏感性
6. 更新 OFFICIAL_RESULTS.md

运行：python -u code/closeout_fix.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q1
import q2
import q3
import q4

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = q1.ROOT
OUT_VER = ROOT / "results" / "verification"
FIG_VER = ROOT / "figures" / "verification"
DT = 0.025
DTS = [0.2, 0.1, 0.05, 0.025, 0.0125]


def copy_result_csv() -> None:
    src = ROOT / "results" / "q1" / "result.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    for dst in (ROOT / "result.csv", ROOT / "result_q1.csv"):
        try:
            shutil.copyfile(src, dst)
            print(f"  已复制 {src.name} → {dst}")
        except PermissionError:
            print(f"  无法写入 {dst}（可能被 Excel 占用），请关闭后手动复制")


def refresh_q3_at_bound() -> None:
    path = ROOT / "results" / "q3" / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    y = np.array(
        [data["best"]["y"][k] for k in ("S1_5", "S6", "S7", "S8_9", "v")], dtype=float
    )
    plate = q3.load_plate()
    ev = q3.evaluate_y(y, plate, dt=DT, keep_curve=True)
    data["best"] = q3.result_to_dict(ev)
    data["official_grid"] = {
        "dt_report": DT,
        "dt_search": 0.1,
        "dt_refine": 0.05,
        "note": "全部论文引用的第三问面积/指标均以 dt_report 计算",
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Q3 at_bound 已刷新: {data['best']['at_bound']}")
    # 同步曲线
    assert ev.t is not None and ev.T is not None
    pd.DataFrame({"t": ev.t, "T": ev.T}).to_csv(
        ROOT / "results" / "q3" / "best_curve.csv", index=False, encoding="utf-8-sig"
    )
    q3.plot_optimal_curve(ev, ROOT / "figures" / "q3" / "optimal_curve.png")
    q3.plot_margins(ev, ROOT / "figures" / "q3" / "constraint_margins.png")


def rebuild_convergence(plate: q1.PlateParams, y3: np.ndarray, y4: np.ndarray | None) -> None:
    OUT_VER.mkdir(parents=True, exist_ok=True)
    FIG_VER.mkdir(parents=True, exist_ok=True)
    rows = []
    for dt in DTS:
        print(f"  convergence dt={dt} ...")
        row: dict = {"dt": dt}
        t, T = q3.simulate_plate_fast(q1.SETPOINTS_Q1, 78.0, plate, dt=dt)
        for name, x in q1.Q1_PROBE_X.items():
            tt = x / (78.0 / 60.0)
            row[f"q1_{name}"] = float(np.interp(tt, t, T))

        scan = []
        for v in np.arange(q2.V_MIN, q2.V_MAX + 0.5, 1.0):
            ev = q2.evaluate_speed(float(v), plate, dt=dt)
            scan.append({k: ev[k] for k in ev if k not in ("t", "T")})
        roots = q2.find_boundary_roots(plate, scan, dt=dt)
        vmax, detail = q2.max_feasible_speed(plate, scan, roots, dt=dt, report_step=0.01)
        row["q2_vmax"] = vmax
        row["q2_vstar"] = detail["v_star_continuous"]
        row["q2_Tpeak"] = detail["at_report"]["T_peak"]

        e3 = q3.evaluate_y(y3, plate, dt=dt)
        row["q3_A"] = float(e3.A)
        row["q3_Tpeak"] = e3.metrics["T_peak"]
        row["q3_feasible"] = bool(e3.feasible)

        if y4 is not None:
            e4 = q4.evaluate_y4(y4, plate, dt=dt)
            row["q4_J_sym"] = float(e4.J_sym)
            row["q4_A_L"] = float(e4.A_L)
            row["q4_feasible"] = bool(e4.feasible_process)
        rows.append(row)
        print(
            f"    vmax={row['q2_vmax']}, A={row['q3_A']:.4f}, "
            f"feas3={row['q3_feasible']}, J={row.get('q4_J_sym')}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_VER / "timestep_convergence.csv", index=False, encoding="utf-8-sig")
    (OUT_VER / "timestep_convergence.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].plot(df["dt"], df["q2_vmax"], "o-", color="#1d3557")
    axes[0].set_xlabel("dt / s")
    axes[0].set_ylabel("v_max / (cm/min)")
    axes[0].set_title("问题二最大速度")
    axes[0].invert_xaxis()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["dt"], df["q3_A"], "o-", color="#e76f51")
    axes[1].set_xlabel("dt / s")
    axes[1].set_ylabel("A / (°C·s)")
    axes[1].set_title("问题三面积（固定最优 y）")
    axes[1].invert_xaxis()
    axes[1].grid(True, alpha=0.3)

    if "q4_J_sym" in df.columns:
        axes[2].plot(df["dt"], df["q4_J_sym"], "o-", color="#2a9d8f")
    axes[2].set_xlabel("dt / s")
    axes[2].set_ylabel("J_sym")
    axes[2].set_title("问题四对称指标（固定 y）")
    axes[2].invert_xaxis()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_VER / "timestep_convergence.png", dpi=150)
    plt.close(fig)


def eng_sensitivity(plate: q1.PlateParams) -> None:
    path = OUT_VER / "q3_theoretical_vs_engineering.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    y_th = np.array([data["theoretical"]["y"][k] for k in ("S1_5", "S6", "S7", "S8_9", "v")])
    y_eng = np.array([data["engineering"]["y"][k] for k in ("S1_5", "S6", "S7", "S8_9", "v")])

    base = {
        "eta_pre": plate.eta_pre,
        "eta_soak": plate.eta_soak,
        "eta_ref": plate.eta_ref,
        "eta_cool": plate.eta_cool,
        "eta_r": plate.eta_r,
        "alpha": plate.alpha,
    }
    names = ["eta_pre", "eta_soak", "eta_ref", "eta_cool"]
    rows = []
    for label, y in (("theoretical", y_th), ("engineering", y_eng)):
        e0 = q3.evaluate_y(y, plate, dt=DT)
        rows.append(
            {
                "solution": label,
                "param": "baseline",
                "delta": 0.0,
                "A": e0.A,
                "feasible": e0.feasible,
                "T_peak": e0.metrics["T_peak"],
                "tau_150_190": e0.metrics["tau_150_190"],
                "tau_217": e0.metrics["tau_217"],
            }
        )
        for name in names:
            for sign in (-0.05, 0.05):
                kw = dict(base)
                kw[name] = base[name] * (1.0 + sign)
                p2 = q1.PlateParams(**kw)
                e = q3.evaluate_y(y, p2, dt=DT)
                rows.append(
                    {
                        "solution": label,
                        "param": name,
                        "delta": sign,
                        "A": e.A,
                        "feasible": e.feasible,
                        "T_peak": e.metrics["T_peak"],
                        "tau_150_190": e.metrics["tau_150_190"],
                        "tau_217": e.metrics["tau_217"],
                        "dA": e.A - e0.A,
                    }
                )
                print(
                    f"  {label} {name} {sign:+.0%}: feas={e.feasible}, "
                    f"A={e.A:.3f}, Tpeak={e.metrics['T_peak']:.3f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_VER / "param_sensitivity_theory_vs_eng.csv", index=False, encoding="utf-8-sig")

    # 可行率汇总
    summary = []
    for sol, g in df[df["param"] != "baseline"].groupby("solution"):
        summary.append(
            {
                "solution": sol,
                "n_perturb": len(g),
                "feasible_count": int(g["feasible"].sum()),
                "feasible_rate": float(g["feasible"].mean()),
            }
        )
    pd.DataFrame(summary).to_csv(
        OUT_VER / "param_sensitivity_feas_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("  可行率:", summary)


def update_official_md(q2_vmax: float, q3_A: float, q4_final: dict | None) -> None:
    text = f"""# 官方结果口径（定稿）

- **统一报告时间步长**: Δt = {DT} s
- 问题一预测 / result.csv / 问题二最大速度 / 问题三面积 / 问题四指标均按此口径
- 标定拟合仍可用稍粗网格，但不影响最终报告数字

## 当前定稿数值

| 项目 | 数值 |
|------|------|
| 问题二 v_max | {q2_vmax:.2f} cm/min |
| 问题三理论最优 A3* | {q3_A:.6f} °C·s |
| 问题四最终 A_L | {q4_final['A_L']:.6f} °C·s |
| 问题四最终 J_sym | {q4_final['J_sym']:.8f} |

## 文件

- 赛题 result.csv：根目录与 results/q1/result.csv 内容一致（若根目录被 Excel 占用，以 results/q1/result.csv 为准）
- 验证表：results/verification/
- 第三问：results/q3/summary.json
- 第四问：results/q4/summary.json（Pareto 已相对统一端点重算）
"""
    (ROOT / "results" / "OFFICIAL_RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    t_all = time.perf_counter()
    print("=" * 60)
    print("收口修正开始")
    print("=" * 60)

    print("\n[0] 先用现有 q1 结果修复根目录 result.csv ...")
    copy_result_csv()

    print("\n[1] 重跑问题一 (dt_report=0.025) ...")
    # 直接调用 q1.main
    q1.main()
    copy_result_csv()

    print("\n[2] 重跑问题二 (dt=0.025) ...")
    q2.main()
    q2s = json.loads((ROOT / "results" / "q2" / "summary.json").read_text(encoding="utf-8"))
    print(f"  v_max = {q2s['v_max_report']}")

    print("\n[3] 刷新问题三 at_bound ...")
    refresh_q3_at_bound()
    q3s = json.loads((ROOT / "results" / "q3" / "summary.json").read_text(encoding="utf-8"))
    y3 = np.array([q3s["best"]["y"][k] for k in ("S1_5", "S6", "S7", "S8_9", "v")])

    print("\n[4] 完整重跑问题四 ...")
    # 默认预算；通过 argv 注入无额外参数
    sys.argv = ["q4.py"]
    q4.main()
    q4s = json.loads((ROOT / "results" / "q4" / "summary.json").read_text(encoding="utf-8"))
    y4 = np.array([q4s["final"]["y"][k] for k in ("S1_5", "S6", "S7", "S8_9", "v")])

    plate = q3.load_plate()
    print("\n[5] 重建收敛表/图（含 dt=0.0125）...")
    rebuild_convergence(plate, y3, y4)

    print("\n[6] 工程解 vs 理论解 参数敏感性 ...")
    eng_sensitivity(plate)

    update_official_md(
        q2s["v_max_report"],
        q3s["best"]["A"],
        {
            "A_L": q4s["final"]["A_L"],
            "J_sym": q4s["final"]["J_sym"],
        },
    )

    # 核对 Pareto 左端
    front = pd.read_csv(ROOT / "results" / "q4" / "pareto_front.csv")
    print("\n核对:")
    print(f"  Q3 A* = {q3s['best']['A']:.6f}")
    print(f"  Q4 Pareto 最小 A_L = {front['A_L'].min():.6f}")
    print(f"  Q4 final A_L = {q4s['final']['A_L']:.6f}, J={q4s['final']['J_sym']:.6f}")
    print(f"  Q2 vmax = {q2s['v_max_report']}")
    n = sum(1 for _ in open(ROOT / "results" / "q1" / "result.csv", encoding="utf-8-sig")) - 1
    print(f"  result.csv 数据行 ≈ {n}")
    print(f"\n总用时 {time.perf_counter()-t_all:.1f}s")


if __name__ == "__main__":
    main()
