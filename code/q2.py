"""
问题二：最大允许过炉速度（单文件版）

流程：
1. 继承问题一标定参数（优先读 results/q1/summary.json，否则现场标定）
2. 固定问题二温区设定，只改变传送带速度 v ∈ [65, 100]
3. 稀疏扫描 + Brent 约束边界寻根 + 区间可行性验证
4. 输出最大速度、有效约束、指标表与曲线

运行：python code/q2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

# 复用问题一仿真器与标定
sys.path.insert(0, str(Path(__file__).resolve().parent))
import q1

ROOT = q1.ROOT
OUT_DIR = ROOT / "results" / "q2"
FIG_DIR = ROOT / "figures" / "q2"

SETPOINTS_Q2 = np.array(
    [182, 182, 182, 182, 182, 203, 237, 254, 254, 25, 25], dtype=float
)
V_MIN, V_MAX = 65.0, 100.0
# 升温段局部微负斜率容差（缝隙离散可能产生数值噪声）
SLOPE_UP_TOL = 1e-3


# ---------------------------------------------------------------------------
# 工艺指标与约束裕量
# ---------------------------------------------------------------------------
def evaluate_speed(v: float, plate: q1.PlateParams, dt: float = 0.1) -> dict:
    """给定速度，仿真并返回指标、裕量与可行性。"""
    t_end = q1.FURNACE_LEN / (v / 60.0)
    t, T = q1.simulate_plate(SETPOINTS_Q2, v, plate, t_end=t_end, dt=dt)

    i_peak = int(np.argmax(T))
    t_peak = float(t[i_peak])
    T_peak = float(T[i_peak])
    dTdt = np.gradient(T, t)

    # 峰值前/后斜率
    if i_peak <= 1:
        r_up_max = r_up_min = float("nan")
    else:
        up = dTdt[1:i_peak]
        r_up_max = float(np.max(up))
        r_up_min = float(np.min(up))
    if i_peak >= len(T) - 2:
        r_dn_min = r_dn_max = float("nan")
    else:
        dn = dTdt[i_peak + 1 : -1]
        r_dn_min = float(np.min(dn))
        r_dn_max = float(np.max(dn))

    def cross_time(level: float, rising: bool) -> float:
        if rising:
            idx = np.where((T[:-1] < level) & (T[1:] >= level))[0]
        else:
            idx = np.where((T[:-1] >= level) & (T[1:] < level))[0]
        if idx.size == 0:
            return float("nan")
        i = int(idx[0 if rising else -1])
        return float(
            t[i] + (level - T[i]) / (T[i + 1] - T[i] + 1e-30) * (t[i + 1] - t[i])
        )

    # 150–190：仅升温段
    t150 = cross_time(150.0, True)
    t190 = cross_time(190.0, True)
    if np.isfinite(t150) and np.isfinite(t190) and t190 > t150 and t190 <= t_peak + 1e-9:
        tau_150_190 = float(t190 - t150)
    else:
        tau_150_190 = float("nan")

    t217_up = cross_time(217.0, True)
    t217_dn = cross_time(217.0, False)
    if np.isfinite(t217_up) and np.isfinite(t217_dn) and t217_dn > t217_up:
        tau_217 = float(t217_dn - t217_up)
    else:
        tau_217 = float("nan")

    # 未穿越阈值 → 直接不可行（裕量记为很大负值）
    def g_or_fail(ok: bool, value: float) -> float:
        return value if ok else -1e3

    margins = {
        "g1_up_max": 3.0 - r_up_max if np.isfinite(r_up_max) else -1e3,
        "g2_up_min": r_up_min + SLOPE_UP_TOL if np.isfinite(r_up_min) else -1e3,
        "g3_dn_min": r_dn_min + 3.0 if np.isfinite(r_dn_min) else -1e3,
        "g4_dn_max": -r_dn_max if np.isfinite(r_dn_max) else -1e3,
        "g5_tau150_lo": g_or_fail(np.isfinite(tau_150_190), tau_150_190 - 60.0),
        "g6_tau150_hi": g_or_fail(np.isfinite(tau_150_190), 120.0 - tau_150_190),
        "g7_tau217_lo": g_or_fail(np.isfinite(tau_217), tau_217 - 40.0),
        "g8_tau217_hi": g_or_fail(np.isfinite(tau_217), 90.0 - tau_217),
        "g9_peak_lo": T_peak - 240.0,
        "g10_peak_hi": 250.0 - T_peak,
    }
    feasible = all(g >= 0.0 for g in margins.values())

    return {
        "v": float(v),
        "feasible": bool(feasible),
        "T_peak": T_peak,
        "t_peak": t_peak,
        "r_up_max": r_up_max,
        "r_up_min": r_up_min,
        "r_dn_min": r_dn_min,
        "r_dn_max": r_dn_max,
        "tau_150_190": tau_150_190,
        "tau_217": tau_217,
        "margins": margins,
        "min_margin": float(min(margins.values())),
        "t": t,
        "T": T,
    }


def active_constraints(margins: dict, roots: dict[str, float], v_max: float, eps: float = 0.15) -> list[str]:
    """
    有效约束：在 v_max 附近裕量很小，且 Brent 根落在 v_max 附近
    （避免把“升温段局部斜率接近 0”误判为限速约束）。
    """
    near_roots = []
    for key, vr in roots.items():
        name = key.split("@")[0]
        if abs(vr - v_max) <= 1.0 and 0.0 <= margins.get(name, 1e9) <= eps:
            near_roots.append(name)
    if near_roots:
        return sorted(set(near_roots))
    # 回退：只看峰值/时间类下界裕量
    keys = ["g5_tau150_lo", "g7_tau217_lo", "g9_peak_lo"]
    return [k for k in keys if 0.0 <= margins.get(k, 1e9) <= eps]


# ---------------------------------------------------------------------------
# 搜索：扫描 + Brent + 区间验证
# ---------------------------------------------------------------------------
def sparse_scan(
    plate: q1.PlateParams, step: float = 1.0, dt: float = 0.1
) -> list[dict]:
    rows = []
    for v in np.arange(V_MIN, V_MAX + 0.5 * step, step):
        ev = evaluate_speed(float(v), plate, dt=dt)
        # 扫描表不存整条曲线，省内存
        rows.append({k: ev[k] for k in ev if k not in ("t", "T")})
        print(
            f"  v={v:6.1f}: feas={ev['feasible']}, "
            f"Tmax={ev['T_peak']:.2f}, tau150={ev['tau_150_190']}, "
            f"tau217={ev['tau_217']}, Gmin={ev['min_margin']:.3f}"
        )
    return rows


def find_boundary_roots(
    plate: q1.PlateParams, scan: list[dict], dt: float = 0.1
) -> dict[str, float]:
    """对每个裕量在符号变化区间用 Brent 求 g_j(v)=0。"""
    roots: dict[str, float] = {}
    names = list(scan[0]["margins"].keys())

    for name in names:
        for a, b in zip(scan[:-1], scan[1:]):
            ga = a["margins"][name]
            gb = b["margins"][name]
            # 需要穿过 0：一端非负一端负
            if ga * gb < 0:

                def f(v: float, n=name) -> float:
                    return evaluate_speed(float(v), plate, dt=dt)["margins"][n]

                try:
                    root = brentq(f, a["v"], b["v"], xtol=1e-4, maxiter=80)
                    key = f"{name}@{root:.4f}"
                    # 同一约束可能多根，都记下；后面用区间法汇总
                    roots[key] = float(root)
                except ValueError:
                    pass
    return roots


def max_feasible_speed(
    plate: q1.PlateParams,
    scan: list[dict],
    roots: dict[str, float],
    dt: float = 0.1,
    report_step: float = 0.01,
) -> tuple[float, dict]:
    """
    将扫描点与约束边界排序，检验子区间，取最高可行区间右端点。
    最终按 report_step（默认 0.01 cm/min）向可行侧取整。
    """
    boundaries = sorted({V_MIN, V_MAX, *[r["v"] for r in scan], *roots.values()})
    # 去重
    cleaned = [boundaries[0]]
    for x in boundaries[1:]:
        if abs(x - cleaned[-1]) > 1e-6:
            cleaned.append(x)

    feasible_intervals: list[tuple[float, float]] = []
    for lo, hi in zip(cleaned[:-1], cleaned[1:]):
        mid = 0.5 * (lo + hi)
        if evaluate_speed(mid, plate, dt=dt)["feasible"]:
            feasible_intervals.append((lo, hi))

    if not feasible_intervals:
        raise RuntimeError("在 [65,100] 内未找到可行速度，请检查标定参数或约束实现")

    # 最高可行区间
    lo, hi = feasible_intervals[-1]
    # 右端点：若 hi 可行则取 hi，否则取略小于边界的可行值
    ev_hi = evaluate_speed(hi, plate, dt=dt)
    if ev_hi["feasible"]:
        v_star = hi
        ev_star = ev_hi
    else:
        # 在 (lo, hi) 上对“可行性边界”细化：找最大可行 v
        # 用二分：左可行右不可行
        left, right = lo, hi
        # 确保 left 可行
        if not evaluate_speed(left, plate, dt=dt)["feasible"]:
            left = mid = 0.5 * (lo + hi)
        for _ in range(40):
            mid = 0.5 * (left + right)
            if evaluate_speed(mid, plate, dt=dt)["feasible"]:
                left = mid
            else:
                right = mid
        v_star = left
        ev_star = evaluate_speed(v_star, plate, dt=dt)

    # 按 report_step 向可行侧取整（默认 0.01 cm/min）
    v_report = np.floor(v_star / report_step + 1e-9) * report_step
    # 再向右探到该精度下仍可行的最大格子点
    cand = round(v_report, 10)
    while cand + report_step <= V_MAX + 1e-12:
        nxt = round(cand + report_step, 10)
        if evaluate_speed(nxt, plate, dt=dt)["feasible"]:
            cand = nxt
        else:
            break
    v_report = cand

    ev_report = evaluate_speed(v_report, plate, dt=dt)
    return float(v_report), {
        "v_star_continuous": float(v_star),
        "v_report": float(v_report),
        "report_step": float(report_step),
        "intervals": feasible_intervals,
        "at_star": {k: ev_star[k] for k in ev_star if k not in ("t", "T")},
        "at_report": {k: ev_report[k] for k in ev_report if k not in ("t", "T")},
        "curve_t": ev_report["t"],
        "curve_T": ev_report["T"],
    }


# ---------------------------------------------------------------------------
# 参数加载
# ---------------------------------------------------------------------------
def load_or_calibrate_plate() -> q1.PlateParams:
    summary_path = ROOT / "results" / "q1" / "summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        p = data["plate_params"]
        print(f"从 {summary_path} 读取问题一标定参数")
        return q1.PlateParams(
            eta_pre=p["eta_pre"],
            eta_soak=p["eta_soak"],
            eta_ref=p["eta_ref"],
            eta_cool=p["eta_cool"],
            eta_r=p.get("eta_r", 0.0),
            alpha=p.get("alpha", q1.ALPHA_FIXED),
        )
    print("未找到 q1 summary，现场用附件重新标定...")
    t_obs, y_obs = q1.load_calibration_data()
    plate, metrics = q1.fit_plate(t_obs, y_obs, fit_radiation=True)
    print(f"标定完成 RMSE={metrics['rmse']:.4f} °C")
    return plate


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("问题二：最大允许过炉速度")
    print("温区设定: 182/203/237/254/25 °C")
    print("速度范围: [65, 100] cm/min")
    print("=" * 60)

    plate = load_or_calibrate_plate()
    print(
        f"参数: eta_pre={plate.eta_pre:.4f}, soak={plate.eta_soak:.4f}, "
        f"ref={plate.eta_ref:.4f}, cool={plate.eta_cool:.4f}"
    )

    print("\n[1] 稀疏扫描 (步长 1 cm/min)...")
    scan = sparse_scan(plate, step=1.0, dt=0.1)
    scan_df = pd.DataFrame(
        [
            {
                "v": r["v"],
                "feasible": r["feasible"],
                "T_peak": r["T_peak"],
                "r_up_max": r["r_up_max"],
                "r_dn_min": r["r_dn_min"],
                "tau_150_190": r["tau_150_190"],
                "tau_217": r["tau_217"],
                "min_margin": r["min_margin"],
                **{f"m_{k}": v for k, v in r["margins"].items()},
            }
            for r in scan
        ]
    )
    scan_df.to_csv(OUT_DIR / "speed_scan.csv", index=False, encoding="utf-8-sig")

    n_feas = int(scan_df["feasible"].sum())
    print(f"扫描可行点数: {n_feas} / {len(scan_df)}")
    if n_feas == 0:
        raise SystemExit("扫描未发现可行速度，停止。")

    print("\n[2] Brent 约束边界寻根...")
    roots = find_boundary_roots(plate, scan, dt=0.1)
    for k, v in sorted(roots.items(), key=lambda kv: kv[1]):
        print(f"  {k}: v*={v:.4f}")
    (OUT_DIR / "constraint_roots.json").write_text(
        json.dumps(roots, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[3] 确定最大允许速度...")
    v_max, detail = max_feasible_speed(plate, scan, roots, dt=0.1, report_step=0.01)
    ev = detail["at_report"]
    active = active_constraints(ev["margins"], roots, v_max, eps=0.15)

    print(f"\n连续边界速度 ≈ {detail['v_star_continuous']:.4f} cm/min")
    print(f"报告最大允许速度 v_max = {v_max:.2f} cm/min（精度 0.01）")
    print("对应工艺指标:")
    print(f"  峰值温度     = {ev['T_peak']:.4f} °C")
    print(f"  最大升温斜率 = {ev['r_up_max']:.4f} °C/s")
    print(f"  最小降温斜率 = {ev['r_dn_min']:.4f} °C/s")
    print(f"  150–190 时间 = {ev['tau_150_190']:.4f} s")
    print(f"  >217 时间    = {ev['tau_217']:.4f} s")
    print(f"  最小裕量     = {ev['min_margin']:.4f}")
    print(f"  接近有效约束 = {active if active else '（均有余量，可能顶到 100）'}")

    # 边界两侧验证
    print("\n[4] 边界两侧验证 (±0.01):")
    for dv in (-0.01, 0.0, 0.01):
        vv = min(V_MAX, max(V_MIN, round(v_max + dv, 10)))
        e = evaluate_speed(vv, plate, dt=0.1)
        print(
            f"  v={vv:.2f}: feas={e['feasible']}, "
            f"Tmax={e['T_peak']:.4f}, tau217={e['tau_217']}, Gmin={e['min_margin']:.4f}"
        )

    # 指标—速度图
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    vv = scan_df["v"].to_numpy()
    axes[0, 0].plot(vv, scan_df["T_peak"], "-o", ms=3)
    axes[0, 0].axhspan(240, 250, color="g", alpha=0.15)
    axes[0, 0].axvline(v_max, color="r", ls="--")
    axes[0, 0].set_ylabel("T_peak / degC")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(vv, scan_df["tau_150_190"], "-o", ms=3)
    axes[0, 1].axhspan(60, 120, color="g", alpha=0.15)
    axes[0, 1].axvline(v_max, color="r", ls="--")
    axes[0, 1].set_ylabel("tau 150-190 / s")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(vv, scan_df["tau_217"], "-o", ms=3)
    axes[1, 0].axhspan(40, 90, color="g", alpha=0.15)
    axes[1, 0].axvline(v_max, color="r", ls="--")
    axes[1, 0].set_ylabel("tau >217 / s")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(vv, scan_df["r_up_max"], "-o", ms=3, label="up max")
    axes[1, 1].plot(vv, scan_df["r_dn_min"], "-o", ms=3, label="dn min")
    axes[1, 1].axhline(3, color="k", ls=":")
    axes[1, 1].axhline(-3, color="k", ls=":")
    axes[1, 1].axvline(v_max, color="r", ls="--")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylabel("slope / degC/s")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(vv, scan_df["min_margin"], "-o", ms=3)
    axes[2, 0].axhline(0, color="k")
    axes[2, 0].axvline(v_max, color="r", ls="--")
    axes[2, 0].set_xlabel("v / (cm/min)")
    axes[2, 0].set_ylabel("min margin")
    axes[2, 0].grid(True, alpha=0.3)

    feas = scan_df["feasible"].to_numpy()
    axes[2, 1].step(vv, feas.astype(float), where="mid")
    axes[2, 1].axvline(v_max, color="r", ls="--")
    axes[2, 1].set_xlabel("v / (cm/min)")
    axes[2, 1].set_ylabel("feasible")
    axes[2, 1].set_ylim(-0.1, 1.1)
    axes[2, 1].grid(True, alpha=0.3)
    fig.suptitle("Problem 2: metrics vs speed")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_vs_speed.png", dpi=150)
    plt.close(fig)

    # 最大速度炉温曲线
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(detail["curve_t"], detail["curve_T"], label=f"v={v_max:.2f} cm/min")
    ax.axhline(217, color="orange", ls="--", lw=0.8)
    ax.axhline(240, color="g", ls="--", lw=0.8)
    ax.axhline(250, color="g", ls="--", lw=0.8)
    ax.set_xlabel("t / s")
    ax.set_ylabel("T / degC")
    ax.set_title("Oven profile at maximum feasible speed")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "profile_at_vmax.png", dpi=150)
    plt.close(fig)

    # 指标表
    table = pd.DataFrame(
        [
            {
                "指标": "最大升温斜率",
                "结果": ev["r_up_max"],
                "范围": "0~3",
                "满足": 0 <= ev["r_up_max"] <= 3,
                "裕量": ev["margins"]["g1_up_max"],
            },
            {
                "指标": "最小降温斜率",
                "结果": ev["r_dn_min"],
                "范围": "-3~0",
                "满足": -3 <= ev["r_dn_min"] <= 0,
                "裕量": ev["margins"]["g3_dn_min"],
            },
            {
                "指标": "150-190时间",
                "结果": ev["tau_150_190"],
                "范围": "60~120",
                "满足": 60 <= ev["tau_150_190"] <= 120
                if np.isfinite(ev["tau_150_190"])
                else False,
                "裕量": min(ev["margins"]["g5_tau150_lo"], ev["margins"]["g6_tau150_hi"]),
            },
            {
                "指标": "高于217时间",
                "结果": ev["tau_217"],
                "范围": "40~90",
                "满足": 40 <= ev["tau_217"] <= 90 if np.isfinite(ev["tau_217"]) else False,
                "裕量": min(ev["margins"]["g7_tau217_lo"], ev["margins"]["g8_tau217_hi"]),
            },
            {
                "指标": "峰值温度",
                "结果": ev["T_peak"],
                "范围": "240~250",
                "满足": 240 <= ev["T_peak"] <= 250,
                "裕量": min(ev["margins"]["g9_peak_lo"], ev["margins"]["g10_peak_hi"]),
            },
        ]
    )
    table.to_csv(OUT_DIR / "metrics_at_vmax.csv", index=False, encoding="utf-8-sig")

    summary = {
        "setpoints": SETPOINTS_Q2.tolist(),
        "v_max_report": v_max,
        "v_star_continuous": detail["v_star_continuous"],
        "report_step": 0.01,
        "active_constraints": active,
        "metrics": {
            "T_peak": ev["T_peak"],
            "r_up_max": ev["r_up_max"],
            "r_up_min": ev["r_up_min"],
            "r_dn_min": ev["r_dn_min"],
            "r_dn_max": ev["r_dn_max"],
            "tau_150_190": ev["tau_150_190"],
            "tau_217": ev["tau_217"],
            "min_margin": ev["min_margin"],
        },
        "margins": ev["margins"],
        "feasible_scan_count": n_feas,
        "constraint_roots": roots,
        "plate_params": {
            "eta_pre": plate.eta_pre,
            "eta_soak": plate.eta_soak,
            "eta_ref": plate.eta_ref,
            "eta_cool": plate.eta_cool,
            "eta_r": plate.eta_r,
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
