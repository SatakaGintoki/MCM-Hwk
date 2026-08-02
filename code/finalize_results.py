"""
论文前结果统一与验证（不写论文）

官方精度口径（与问题四 dt_verify 一致）：
  DT_REPORT = 0.025 s
  DT_SEARCH = 0.1 s
  DT_REFINE = 0.05 s

流程：
1. 细网格统一重解问题三，覆写 results/q3 与 figures/q3
2. 时间步长收敛表
3. 换热参数 ±5% 敏感性表
4. 问题三工程推荐解（约束余量）
5. CDE/GA/PSO 重复实验（默认 5 次）
6. 用统一第三问端点轻量刷新问题四端点衔接信息

运行：
  python code/finalize_results.py
  python code/finalize_results.py --repeats 5
  python code/finalize_results.py --skip-repeats
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
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
OUT_Q3 = ROOT / "results" / "q3"
OUT_Q4 = ROOT / "results" / "q4"
OUT_VER = ROOT / "results" / "verification"
FIG_Q3 = ROOT / "figures" / "q3"
FIG_VER = ROOT / "figures" / "verification"

# 与问题四默认 dt_verify 对齐的官方报告网格
DT_REPORT = 0.025
DT_SEARCH = 0.1
DT_REFINE = 0.05


def y_from_block(block: dict) -> np.ndarray:
    y = block["y"]
    return np.array([y["S1_5"], y["S6"], y["S7"], y["S8_9"], y["v"]], dtype=float)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. 统一问题三
# ---------------------------------------------------------------------------
def unify_q3(plate: q1.PlateParams, seed: int = 2020) -> q3.EvalResult:
    print("\n" + "=" * 60)
    print("[A] 统一问题三（官方 dt=0.025）")
    print("=" * 60)
    OUT_Q3.mkdir(parents=True, exist_ok=True)
    FIG_Q3.mkdir(parents=True, exist_ok=True)

    seeds: list[np.ndarray] = []
    q3_path = OUT_Q3 / "summary.json"
    if q3_path.exists():
        old = json.loads(q3_path.read_text(encoding="utf-8"))
        seeds.append(y_from_block(old["best"]))
        print(f"  种子(旧q3): A@0.025={q3.evaluate_y(seeds[-1], plate, dt=DT_REPORT).A:.4f}")

    q4_path = OUT_Q4 / "summary.json"
    if q4_path.exists():
        q4s = json.loads(q4_path.read_text(encoding="utf-8"))
        for key in ("endpoint_q3", "endpoint_area", "final"):
            if key in q4s and q4s[key]:
                seeds.append(y_from_block(q4s[key]))
                ev = q3.evaluate_y(seeds[-1], plate, dt=DT_REPORT)
                print(f"  种子(q4.{key}): A@0.025={ev.A:.4f}, feas={ev.feasible}")

    # 去重
    uniq: list[np.ndarray] = []
    for y in seeds:
        z = q3.encode(y)
        if not any(np.linalg.norm(z - q3.encode(u)) < 1e-8 for u in uniq):
            uniq.append(y.copy())
    if not uniq:
        uniq = [q3.SEED_Q2.copy()]

    rng = np.random.default_rng(seed)
    print(f"  CDE 搜索 dt={DT_SEARCH}, 预算=1500, 种子数={len(uniq)} ...")
    t0 = time.perf_counter()
    # 手动注入种子到初始种群
    best_cde, pop_cde, hist = q3.run_cde(
        plate, npop=36, max_eval=1500, dt=DT_SEARCH, rng=rng
    )
    # 把种子评估后并入精英比较
    for y in uniq:
        ev = q3.evaluate_y(y, plate, dt=DT_SEARCH)
        pop_cde.append(ev)
        if q3.deb_better(ev, best_cde):
            best_cde = ev
    print(
        f"  搜索最优: A={best_cde.A:.4f}, feas={best_cde.feasible}, "
        f"用时 {time.perf_counter()-t0:.1f}s"
    )

    print(f"  精修 dt={DT_REFINE} ...")
    mid = q3.multi_start_cobyla(
        pop_cde, plate, dt=DT_REFINE, n_elites=5, maxfun=80
    )
    for y in uniq:
        e = q3.evaluate_y(y, plate, dt=DT_REFINE)
        r = q3.refine_cobyla(e, plate, dt=DT_REFINE, maxfun=80)
        if q3.deb_better(r, mid):
            mid = r
    print(f"  中精修: A={mid.A:.4f}, feas={mid.feasible}")

    print(f"  官方精修 dt={DT_REPORT} ...")
    hi_seed = q3.evaluate_y(mid.y, plate, dt=DT_REPORT)
    best = q3.refine_cobyla(hi_seed, plate, dt=DT_REPORT, maxfun=60)
    for y in uniq:
        e = q3.evaluate_y(y, plate, dt=DT_REPORT)
        r = q3.refine_cobyla(e, plate, dt=DT_REPORT, maxfun=40)
        if q3.deb_better(r, best):
            best = r
    best = q3.evaluate_y(best.y, plate, dt=DT_REPORT, keep_curve=True)
    print(
        f"  官方最优: A*={best.A:.6f}, feas={best.feasible}, "
        f"Tpeak={best.metrics['T_peak']:.4f}, "
        f"tau150={best.metrics['tau_150_190']:.4f}, y={best.y}"
    )
    if not best.feasible:
        raise RuntimeError("统一后的问题三最优解在官方网格上不可行")

    # 邻域
    neigh = q3.neighborhood_check(best, plate, dt=DT_REPORT)
    pd.DataFrame(neigh).to_csv(OUT_Q3 / "neighborhood.csv", index=False, encoding="utf-8-sig")

    # 曲线
    assert best.t is not None and best.T is not None
    pd.DataFrame({"t": best.t, "T": best.T}).to_csv(
        OUT_Q3 / "best_curve.csv", index=False, encoding="utf-8-sig"
    )

    # 图
    q3.plot_optimal_curve(best, FIG_Q3 / "optimal_curve.png")
    q3.plot_margins(best, FIG_Q3 / "constraint_margins.png")

    summary = {
        "official_grid": {
            "dt_report": DT_REPORT,
            "dt_search": DT_SEARCH,
            "dt_refine": DT_REFINE,
            "note": "全部论文引用的第三问面积/指标均以 dt_report 计算",
        },
        "budget": {
            "npop": 36,
            "max_eval": 1500,
            "dt_search": DT_SEARCH,
            "dt_refine": DT_REFINE,
            "dt_verify": DT_REPORT,
            "elites": 5,
            "cobyla_maxfun": 80,
        },
        "seed": seed,
        "best": q3.result_to_dict(best),
        "algo_compare": [],  # 稍后由重复实验填充
        "plate_params": {
            "eta_pre": plate.eta_pre,
            "eta_soak": plate.eta_soak,
            "eta_ref": plate.eta_ref,
            "eta_cool": plate.eta_cool,
            "eta_cool_late": plate.eta_cool_late,
            "eta_r": plate.eta_r,
            "alpha": plate.alpha,
        },
    }
    save_json(OUT_Q3 / "summary.json", summary)
    print(f"  已写入 {OUT_Q3 / 'summary.json'}")
    return best


# ---------------------------------------------------------------------------
# 2. 时间步长收敛
# ---------------------------------------------------------------------------
def run_convergence(plate: q1.PlateParams, y3: np.ndarray, y4: np.ndarray | None) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("[B] 时间步长收敛")
    print("=" * 60)
    OUT_VER.mkdir(parents=True, exist_ok=True)
    dts = [0.2, 0.1, 0.05, 0.025]
    rows = []

    # 问题二：每个 dt 做稀疏扫描+Brent（步长 1，再细化）
    for dt in dts:
        print(f"  dt={dt} ...")
        row: dict = {"dt": dt}

        # Q1 probes
        t_end = q1.FURNACE_LEN / (78.0 / 60.0)
        t, T = q3.simulate_plate_fast(q1.SETPOINTS_Q1, 78.0, plate, dt=dt)
        for name, x in q1.Q1_PROBE_X.items():
            tt = x / (78.0 / 60.0)
            row[f"q1_{name}"] = float(np.interp(tt, t, T))

        # Q2 vmax（静默扫描，避免刷屏）
        try:
            scan = []
            for v in np.arange(q2.V_MIN, q2.V_MAX + 0.5, 1.0):
                ev = q2.evaluate_speed(float(v), plate, dt=dt)
                scan.append({k: ev[k] for k in ev if k not in ("t", "T")})
            roots = q2.find_boundary_roots(plate, scan, dt=dt)
            vmax, detail = q2.max_feasible_speed(
                plate, scan, roots, dt=dt, report_step=0.01
            )
            row["q2_vmax"] = vmax
            row["q2_vstar"] = detail["v_star_continuous"]
            row["q2_Tpeak"] = detail["at_report"]["T_peak"]
        except Exception as exc:
            row["q2_vmax"] = None
            row["q2_vstar"] = None
            row["q2_Tpeak"] = None
            print(f"    q2 failed: {exc}")

        # Q3 area at official y
        e3 = q3.evaluate_y(y3, plate, dt=dt)
        row["q3_A"] = e3.A if e3.feasible else None
        row["q3_Tpeak"] = e3.metrics["T_peak"]
        row["q3_feasible"] = e3.feasible

        # Q4 symmetry at final y
        if y4 is not None:
            e4 = q4.evaluate_y4(y4, plate, dt=dt)
            row["q4_J_sym"] = e4.J_sym if e4.feasible_process else None
            row["q4_A_L"] = e4.A_L if e4.feasible_process else None
            row["q4_feasible"] = e4.feasible_process
        rows.append(row)
        print(
            f"    q2_vmax={row.get('q2_vmax')}, q3_A={row.get('q3_A')}, "
            f"q4_J={row.get('q4_J_sym')}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_VER / "timestep_convergence.csv", index=False, encoding="utf-8-sig")
    save_json(OUT_VER / "timestep_convergence.json", {"rows": rows})

    # 简图：q3_A vs dt
    FIG_VER.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
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

    if "q4_J_sym" in df.columns and df["q4_J_sym"].notna().any():
        axes[2].plot(df["dt"], df["q4_J_sym"], "o-", color="#2a9d8f")
    axes[2].set_xlabel("dt / s")
    axes[2].set_ylabel("J_sym")
    axes[2].set_title("问题四对称指标（固定 y）")
    axes[2].invert_xaxis()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_VER / "timestep_convergence.png", dpi=150)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# 3. 参数敏感性
# ---------------------------------------------------------------------------
def run_sensitivity(plate: q1.PlateParams, y3: np.ndarray, y4: np.ndarray | None) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("[C] 参数敏感性 ±5%")
    print("=" * 60)
    OUT_VER.mkdir(parents=True, exist_ok=True)
    names = ["eta_pre", "eta_soak", "eta_ref", "eta_cool", "eta_cool_late"]
    base = {
        "eta_pre": plate.eta_pre,
        "eta_soak": plate.eta_soak,
        "eta_ref": plate.eta_ref,
        "eta_cool": plate.eta_cool,
        "eta_cool_late": plate.eta_cool_late,
        "eta_r": plate.eta_r,
        "alpha": plate.alpha,
    }
    rows = []

    def make_plate(overrides: dict) -> q1.PlateParams:
        kw = dict(base)
        kw.update(overrides)
        return q1.PlateParams(**kw)

    # baseline
    e3b = q3.evaluate_y(y3, plate, dt=DT_REPORT)
    e4b = q4.evaluate_y4(y4, plate, dt=DT_REPORT) if y4 is not None else None
    rows.append(
        {
            "param": "baseline",
            "delta": 0.0,
            "T_peak": e3b.metrics["T_peak"],
            "tau_150_190": e3b.metrics["tau_150_190"],
            "tau_217": e3b.metrics["tau_217"],
            "q3_A": e3b.A,
            "q3_feasible": e3b.feasible,
            "q4_J_sym": None if e4b is None else e4b.J_sym,
            "q4_feasible": None if e4b is None else e4b.feasible_process,
        }
    )

    for name in names:
        for sign, tag in ((-0.05, "-5%"), (0.05, "+5%")):
            ov = {name: base[name] * (1.0 + sign)}
            p2 = make_plate(ov)
            e3 = q3.evaluate_y(y3, p2, dt=DT_REPORT)
            e4 = q4.evaluate_y4(y4, p2, dt=DT_REPORT) if y4 is not None else None
            rows.append(
                {
                    "param": name,
                    "delta": sign,
                    "delta_label": tag,
                    "T_peak": e3.metrics["T_peak"],
                    "tau_150_190": e3.metrics["tau_150_190"],
                    "tau_217": e3.metrics["tau_217"],
                    "q3_A": e3.A,
                    "q3_feasible": e3.feasible,
                    "q4_J_sym": None if e4 is None else e4.J_sym,
                    "q4_feasible": None if e4 is None else e4.feasible_process,
                    "dA": e3.A - e3b.A,
                    "dTpeak": e3.metrics["T_peak"] - e3b.metrics["T_peak"],
                }
            )
            print(
                f"  {name} {tag}: A={e3.A:.3f} (dA={e3.A-e3b.A:+.3f}), "
                f"feas={e3.feasible}, Tpeak={e3.metrics['T_peak']:.3f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_VER / "param_sensitivity.csv", index=False, encoding="utf-8-sig")
    save_json(
        OUT_VER / "param_sensitivity.json",
        {"baseline_A": e3b.A, "rows": rows},
    )

    FIG_VER.mkdir(parents=True, exist_ok=True)
    sub = df[df["param"] != "baseline"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{r.param}{r.delta_label}" for r in sub.itertuples()]
    ax.barh(labels, sub["dA"], color=["#e76f51" if v > 0 else "#2a9d8f" for v in sub["dA"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ΔA / (°C·s) 相对基线")
    ax.set_title("换热参数 ±5% 对第三问面积的影响（固定最优工艺）")
    fig.tight_layout()
    fig.savefig(FIG_VER / "param_sensitivity_dA.png", dpi=150)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# 4. 工程推荐解（约束余量）
# ---------------------------------------------------------------------------
def refine_with_margins(
    elite: q3.EvalResult,
    plate: q1.PlateParams,
    dt: float,
    peak_margin: float = 1.0,
    tau150_margin: float = 2.0,
    tau217_margin: float = 2.0,
    maxfun: int = 80,
) -> q3.EvalResult:
    """在原约束基础上要求额外余量，再最小化面积。"""
    cache: dict[bytes, q3.EvalResult] = {}

    def eval_cached(z: np.ndarray) -> q3.EvalResult:
        key = np.asarray(z, dtype=float).tobytes()
        if key not in cache:
            cache[key] = q3.evaluate_z(z, plate, dt=dt)
        return cache[key]

    def objective(z):
        ev = eval_cached(z)
        return ev.A if ev.feasible else ev.A + 1e3 * (1.0 + ev.V)

    cons = []
    # 原十项
    for j in range(10):

        def make_c(jj):
            def fun(z, j=jj):
                return -eval_cached(z).constraints[j]

            return fun

        cons.append({"type": "ineq", "fun": make_c(j)})

    # 额外余量：Tpeak>=240+m, tau150>=60+m, tau217>=40+m
    def peak_extra(z, m=peak_margin):
        return eval_cached(z).metrics["T_peak"] - (240.0 + m)

    def tau150_extra(z, m=tau150_margin):
        v = eval_cached(z).metrics["tau_150_190"]
        return (-1e3 if not np.isfinite(v) else v - (60.0 + m))

    def tau217_extra(z, m=tau217_margin):
        v = eval_cached(z).metrics["tau_217"]
        return (-1e3 if not np.isfinite(v) else v - (40.0 + m))

    cons.append({"type": "ineq", "fun": peak_extra})
    cons.append({"type": "ineq", "fun": tau150_extra})
    cons.append({"type": "ineq", "fun": tau217_extra})

    for j in range(q3.DIM):

        def lo(z, j=j):
            return float(z[j])

        def hi(z, j=j):
            return float(1.0 - z[j])

        cons.append({"type": "ineq", "fun": lo})
        cons.append({"type": "ineq", "fun": hi})

    from scipy.optimize import minimize

    try:
        res = minimize(
            objective,
            elite.z.copy(),
            method="COBYLA",
            constraints=cons,
            options={"maxiter": maxfun, "rhobeg": 0.05, "tol": 1e-6},
        )
        cand = eval_cached(res.x)
    except Exception:
        return elite

    # 验收额外余量
    ok = (
        cand.feasible
        and cand.metrics["T_peak"] >= 240.0 + peak_margin - 1e-6
        and np.isfinite(cand.metrics["tau_150_190"])
        and cand.metrics["tau_150_190"] >= 60.0 + tau150_margin - 1e-6
        and np.isfinite(cand.metrics["tau_217"])
        and cand.metrics["tau_217"] >= 40.0 + tau217_margin - 1e-6
    )
    return cand if ok else elite


def run_engineering(plate: q1.PlateParams, theoretical: q3.EvalResult) -> q3.EvalResult:
    print("\n" + "=" * 60)
    print("[D] 工程推荐解（峰值余量≥1°C, τ150余量≥2s, τ217余量≥2s）")
    print("=" * 60)
    OUT_VER.mkdir(parents=True, exist_ok=True)

    # 多起点：理论最优邻域扰动
    starts = [theoretical]
    rng = np.random.default_rng(2026)
    for _ in range(8):
        z = theoretical.z + rng.normal(0, 0.05, size=q3.DIM)
        z = q3.reflect_bounds(z)
        starts.append(q3.evaluate_z(z, plate, dt=DT_REFINE))

    def meets_eng(ev: q3.EvalResult) -> bool:
        return (
            ev.feasible
            and ev.metrics["T_peak"] >= 241.0 - 1e-6
            and np.isfinite(ev.metrics["tau_150_190"])
            and ev.metrics["tau_150_190"] >= 62.0 - 1e-6
            and np.isfinite(ev.metrics["tau_217"])
            and ev.metrics["tau_217"] >= 42.0 - 1e-6
        )

    candidates: list[q3.EvalResult] = []
    for s in starts:
        mid = refine_with_margins(s, plate, dt=DT_REFINE, maxfun=70)
        hi = refine_with_margins(
            q3.evaluate_y(mid.y, plate, dt=DT_REPORT),
            plate,
            dt=DT_REPORT,
            maxfun=50,
        )
        hi = q3.evaluate_y(hi.y, plate, dt=DT_REPORT)
        if meets_eng(hi):
            candidates.append(hi)

    if candidates:
        best = min(candidates, key=lambda e: e.A)
    else:
        print("  警告：未找到满足余量的工程解，回退为理论最优邻域可行解")
        best = theoretical

    best = q3.evaluate_y(best.y, plate, dt=DT_REPORT, keep_curve=True)
    print(
        f"  工程解: A={best.A:.4f}, Tpeak={best.metrics['T_peak']:.4f}, "
        f"tau150={best.metrics['tau_150_190']:.4f}, "
        f"tau217={best.metrics['tau_217']:.4f}, feas={best.feasible}"
    )
    print(
        f"  相对理论最优 dA={best.A - theoretical.A:+.4f} "
        f"({100*(best.A/theoretical.A-1):+.2f}%)"
    )

    compare = {
        "theoretical": q3.result_to_dict(theoretical),
        "engineering": q3.result_to_dict(best),
        "margins_required": {
            "peak_above_240": 1.0,
            "tau150_above_60": 2.0,
            "tau217_above_40": 2.0,
        },
        "delta_A": best.A - theoretical.A,
        "delta_A_pct": 100.0 * (best.A / theoretical.A - 1.0),
    }
    save_json(OUT_VER / "q3_theoretical_vs_engineering.json", compare)
    pd.DataFrame(
        [
            {
                "type": "theoretical",
                "A": theoretical.A,
                **{n: float(v) for n, v in zip(q3.VAR_NAMES, theoretical.y)},
                **{k: theoretical.metrics[k] for k in ("T_peak", "tau_150_190", "tau_217")},
                **{f"m_{k}": theoretical.margins[k] for k in theoretical.margins},
            },
            {
                "type": "engineering",
                "A": best.A,
                **{n: float(v) for n, v in zip(q3.VAR_NAMES, best.y)},
                **{k: best.metrics[k] for k in ("T_peak", "tau_150_190", "tau_217")},
                **{f"m_{k}": best.margins[k] for k in best.margins},
            },
        ]
    ).to_csv(OUT_VER / "q3_theoretical_vs_engineering.csv", index=False, encoding="utf-8-sig")

    # 对比图
    FIG_VER.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(theoretical.t, theoretical.T, label=f"理论最优 A={theoretical.A:.2f}", lw=1.8)
    ax.plot(best.t, best.T, label=f"工程推荐 A={best.A:.2f}", lw=1.8)
    ax.axhline(217, color="#c0392b", ls="--", lw=1)
    ax.axhline(240, color="gray", ls=":", lw=1)
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("温度 T / °C")
    ax.set_title("问题三：理论最优 vs 工程推荐")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_VER / "q3_theoretical_vs_engineering.png", dpi=150)
    plt.close(fig)
    return best


# ---------------------------------------------------------------------------
# 5. 算法重复实验
# ---------------------------------------------------------------------------
def run_algo_repeats(
    plate: q1.PlateParams,
    n_repeats: int = 5,
    npop: int = 24,
    max_eval: int = 800,
    seed0: int = 3000,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(f"[E] 算法重复实验 ×{n_repeats}（同预算，最终 A 一律 dt={DT_REPORT}）")
    print("=" * 60)
    OUT_VER.mkdir(parents=True, exist_ok=True)
    OUT_Q3.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    for r in range(n_repeats):
        for name, runner, s_off in (
            ("CDE+COBYLA", q3.run_cde, 0),
            ("GA+COBYLA", q3.run_ga, 100),
            ("PSO+COBYLA", q3.run_pso, 200),
        ):
            rng = np.random.default_rng(seed0 + r * 17 + s_off)
            t0 = time.perf_counter()
            best, pop, _hist = runner(plate, npop, max_eval, DT_SEARCH, rng)
            refined = q3.multi_start_cobyla(
                pop, plate, dt=DT_REFINE, n_elites=3, maxfun=50
            )
            if q3.deb_better(refined, best):
                best = refined
            hi = q3.evaluate_y(best.y, plate, dt=DT_REPORT)
            # 再在官方网格上轻量精修
            hi2 = q3.refine_cobyla(hi, plate, dt=DT_REPORT, maxfun=30)
            if q3.deb_better(hi2, hi):
                hi = hi2
            elapsed = time.perf_counter() - t0
            detail_rows.append(
                {
                    "algo": name,
                    "repeat": r,
                    "feasible": hi.feasible,
                    "A": hi.A if hi.feasible else None,
                    "V": hi.V,
                    "elapsed_s": elapsed,
                    **{n: float(v) for n, v in zip(q3.VAR_NAMES, hi.y)},
                }
            )
            print(
                f"  {name} #{r}: feas={hi.feasible}, A={hi.A:.4f}, "
                f"t={elapsed:.1f}s"
            )

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUT_VER / "algo_repeats_detail.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for algo, g in detail.groupby("algo"):
        feas = g[g["feasible"] == True]  # noqa: E712
        summary_rows.append(
            {
                "algo": algo,
                "n_repeats": len(g),
                "feasible_rate": float(g["feasible"].mean()),
                "best_A": float(feas["A"].min()) if len(feas) else None,
                "mean_A": float(feas["A"].mean()) if len(feas) else None,
                "std_A": float(feas["A"].std(ddof=1)) if len(feas) > 1 else 0.0,
                "mean_elapsed_s": float(g["elapsed_s"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("algo")
    summary.to_csv(OUT_VER / "algo_repeats_summary.csv", index=False, encoding="utf-8-sig")
    # 同步到 q3/algo_compare.csv：用各算法 best_A 对应那一次的 y
    compare_rows = []
    for algo, g in detail.groupby("algo"):
        feas = g[g["feasible"] == True]  # noqa: E712
        if len(feas) == 0:
            continue
        best_row = feas.loc[feas["A"].idxmin()]
        compare_rows.append(
            {
                "algo": algo,
                "feasible": True,
                "A": float(best_row["A"]),
                "V": 0.0,
                "note": f"best over {n_repeats} repeats @ dt={DT_REPORT}",
                **{n: float(best_row[n]) for n in q3.VAR_NAMES},
            }
        )
    pd.DataFrame(compare_rows).to_csv(
        OUT_Q3 / "algo_compare.csv", index=False, encoding="utf-8-sig"
    )

    # 箱线图
    FIG_VER.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data, labels = [], []
    for algo in ["CDE+COBYLA", "GA+COBYLA", "PSO+COBYLA"]:
        vals = detail[(detail["algo"] == algo) & (detail["feasible"] == True)]["A"].dropna()  # noqa: E712
        if len(vals):
            data.append(vals.to_numpy())
            labels.append(algo)
    if data:
        ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.set_ylabel("A / (°C·s)")
        ax.set_title(f"算法重复实验可行面积分布（n={n_repeats}）")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_VER / "algo_repeats_boxplot.png", dpi=150)
    plt.close(fig)

    # 写回 q3 summary 的 algo_compare 字段
    q3s = json.loads((OUT_Q3 / "summary.json").read_text(encoding="utf-8"))
    q3s["algo_compare"] = compare_rows
    q3s["algo_repeats_summary"] = summary.to_dict(orient="records")
    save_json(OUT_Q3 / "summary.json", q3s)
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# 6. 刷新问题四衔接（不重跑全 Pareto，只对齐端点记录）
# ---------------------------------------------------------------------------
def refresh_q4_endpoint(plate: q1.PlateParams, y3: np.ndarray) -> None:
    print("\n" + "=" * 60)
    print("[F] 刷新问题四中的第三问端点记录")
    print("=" * 60)
    path = OUT_Q4 / "summary.json"
    if not path.exists():
        print("  无 q4 summary，跳过")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    ev = q4.evaluate_y4(y3, plate, dt=DT_REPORT, keep_curve=True)
    block = q4.ev_to_dict(ev)
    data["endpoint_q3_input"] = block
    data["endpoint_q3"] = block
    data["endpoint_area"] = block
    data["q3_unified_note"] = (
        f"已与 results/q3 官方网格 dt={DT_REPORT} 对齐；"
        "若需完整重算 Pareto，请再运行 python code/q4.py"
    )
    # 用新的面积端点重算归一化表中的端点信息（不改整条前沿的其他点）
    save_json(path, data)
    q4.plot_curve(
        ev,
        ROOT / "figures" / "q4" / "q3_endpoint.png",
        f"问题三统一端点  A_L={ev.A_L:.2f}, J={ev.J_sym:.4f}",
    )
    print(f"  已更新 q4 endpoint_q3 = A_L={ev.A_L:.6f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一结果并补验证")
    p.add_argument("--repeats", type=int, default=5, help="算法重复次数")
    p.add_argument("--skip-repeats", action="store_true")
    p.add_argument("--skip-convergence", action="store_true")
    p.add_argument("--skip-q2-in-convergence", action="store_true",
                   help="收敛表中跳过问题二全搜索（更快）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_VER.mkdir(parents=True, exist_ok=True)
    FIG_VER.mkdir(parents=True, exist_ok=True)

    plate = q3.load_plate()
    print(
        f"标定参数: pre={plate.eta_pre:.4f}, soak={plate.eta_soak:.4f}, "
        f"ref={plate.eta_ref:.4f}, cool10={plate.eta_cool:.4f}, "
        f"cool11={plate.eta_cool_late:.4f}"
    )
    print(f"官方报告网格 dt={DT_REPORT}")

    # A. 统一 Q3
    best3 = unify_q3(plate)

    # 读取 q4 final y（若有）
    y4 = None
    if (OUT_Q4 / "summary.json").exists():
        q4s = json.loads((OUT_Q4 / "summary.json").read_text(encoding="utf-8"))
        if q4s.get("final"):
            y4 = y_from_block(q4s["final"])

    # B. 收敛
    if not args.skip_convergence:
        if args.skip_q2_in_convergence:
            # 临时打补丁：用更快的 q2 评估——仍跑完整，用户要完整表
            pass
        run_convergence(plate, best3.y, y4)

    # C. 敏感性
    run_sensitivity(plate, best3.y, y4)

    # D. 工程解
    eng = run_engineering(plate, best3)

    # E. 算法重复
    if not args.skip_repeats:
        run_algo_repeats(plate, n_repeats=args.repeats)

    # F. 对齐 q4 端点
    refresh_q4_endpoint(plate, best3.y)

    # 总览
    overview = {
        "official_dt": DT_REPORT,
        "q3_theoretical_A": best3.A,
        "q3_theoretical_y": {n: float(v) for n, v in zip(q3.VAR_NAMES, best3.y)},
        "q3_engineering_A": eng.A,
        "q3_engineering_y": {n: float(v) for n, v in zip(q3.VAR_NAMES, eng.y)},
        "outputs": {
            "q3_summary": str(OUT_Q3 / "summary.json"),
            "verification": str(OUT_VER),
            "figures_verification": str(FIG_VER),
        },
    }
    save_json(OUT_VER / "overview.json", overview)

    print("\n" + "=" * 60)
    print("完成")
    print(f"  统一第三问 A* = {best3.A:.6f} °C·s  (dt={DT_REPORT})")
    print(f"  工程推荐   A  = {eng.A:.6f} °C·s")
    print(f"  验证输出目录: {OUT_VER}")
    print("=" * 60)
    print("说明：问题四完整 Pareto 若需与新端点严格重算，请再运行 python code/q4.py")


if __name__ == "__main__":
    main()
