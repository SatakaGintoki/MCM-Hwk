"""
问题四：阴影面积与炉温曲线对称性（单文件版）

流程：
1. 继承问题一标定参数；决策变量同问题三
2. 对称性拆为归一化形状误差 J_shape 与持续时间误差 J_tau
3. 目标：min (A_L, max(J_shape, J_tau))；主方法为 ε-约束
4. 端点：读取问题三解，并在第四问细网格上重新校准面积端点；再求纯对称最优
5. 对每个面积上限 ε_A，用 CDE+COBYLA 最小化新对称指标
6. 过滤非支配点，按膝点（交叉检查最近理想点）选最终解
7. 可选 NSGA-II 交叉验证

运行：
  python code/q4.py
  python code/q4.py --fast
  python code/q4.py --full
  python code/q4.py --nsga
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q1
import q3

ROOT = q1.ROOT
OUT_DIR = ROOT / "results" / "q4"
FIG_DIR = ROOT / "figures" / "q4"

L_BOUNDS = q3.L_BOUNDS
U_BOUNDS = q3.U_BOUNDS
DIM = q3.DIM
VAR_NAMES = q3.VAR_NAMES
CONSTRAINT_SCALE = q3.CONSTRAINT_SCALE
SLOPE_UP_TOL = q3.SLOPE_UP_TOL
EPS_AREA = 1e-9


# ---------------------------------------------------------------------------
# 评价：工艺约束 + A_L/A_R + 对称指标
# ---------------------------------------------------------------------------
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(y, x))


def shadow_area_interval(t: np.ndarray, T: np.ndarray, t0: float, t1: float) -> float:
    """∫_{t0}^{t1} (T-217) dt，端点插值。"""
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return float("nan")
    mask = (t >= t0) & (t <= t1)
    tt = t[mask]
    TT = T[mask]
    T0 = float(np.interp(t0, t, T))
    T1 = float(np.interp(t1, t, T))
    if tt.size == 0 or tt[0] > t0 + 1e-12:
        tt = np.insert(tt, 0, t0)
        TT = np.insert(TT, 0, T0)
    else:
        tt[0], TT[0] = t0, T0
    if tt[-1] < t1 - 1e-12:
        tt = np.append(tt, t1)
        TT = np.append(TT, T1)
    else:
        tt[-1], TT[-1] = t1, T1
    return _trapz(TT - 217.0, tt)


def symmetry_metrics(
    t: np.ndarray, T: np.ndarray, tu: float, tp: float, td: float, dtau: float | None = None
) -> dict:
    """计算镜像几何量与不受面积分母稀释的新对称指标。

    J_shape 在左右各自的相对时间 s∈[0,1] 上比较归一化超温形状；
    J_tau 衡量左右高于 217 °C 的持续时间差；
    J_sym=max(J_shape,J_tau)，迫使优化器优先改善较差的一项。

    旧指标 E_sym/(A_L+A_R) 保留为 J_overlap，只用于诊断，不再优化。
    """
    tau_L = tp - tu
    tau_R = td - tp
    if tau_L <= 0 or tau_R <= 0:
        return {
            "A_L": float("nan"),
            "A_R": float("nan"),
            "E_sym": float("nan"),
            "J_sym": 1.0,
            "J_shape": 1.0,
            "J_overlap": 1.0,
            "J_A": 1.0,
            "J_tau": 1.0,
            "tau_L": tau_L,
            "tau_R": tau_R,
            "tau": None,
            "qL": None,
            "qR": None,
            "phase": None,
            "thetaL": None,
            "thetaR": None,
        }

    A_L = shadow_area_interval(t, T, tu, tp)
    A_R = shadow_area_interval(t, T, tp, td)
    tau_max = max(tau_L, tau_R)
    if dtau is None:
        dtau = max(float(np.median(np.diff(t))), 0.05)
    n = max(int(np.ceil(tau_max / dtau)), 8)
    tau = np.linspace(0.0, tau_max, n + 1)

    # 左侧：t = tp - tau；右侧：t = tp + tau；超出各自区间补 0
    tL = tp - tau
    tR = tp + tau
    TL = np.interp(tL, t, T, left=T[0], right=T[-1])
    TR = np.interp(tR, t, T, left=T[0], right=T[-1])
    qL = np.where(tau <= tau_L + 1e-12, TL - 217.0, 0.0)
    qR = np.where(tau <= tau_R + 1e-12, TR - 217.0, 0.0)
    qL = np.maximum(qL, 0.0)
    qR = np.maximum(qR, 0.0)

    E_sym = _trapz(np.abs(qL - qR), tau)
    denom = A_L + A_R + EPS_AREA
    J_overlap = float(np.clip(E_sym / denom, 0.0, 1.0))
    J_A = float(abs(A_L - A_R) / denom)
    J_tau = float(abs(tau_L - tau_R) / (tau_L + tau_R + EPS_AREA))

    # 形状误差：先把左右各自的持续时间映射到相同相位 s∈[0,1]，
    # 再以峰值超温归一化。这样不会把“左宽右窄”误当成形状差，
    # 时间宽度差由 J_tau 单独、明确地惩罚。
    phase = np.linspace(0.0, 1.0, n + 1)
    TL_phase = np.interp(tp - phase * tau_L, t, T)
    TR_phase = np.interp(tp + phase * tau_R, t, T)
    peak_excess = max(float(np.interp(tp, t, T)) - 217.0, EPS_AREA)
    thetaL = np.clip((TL_phase - 217.0) / peak_excess, 0.0, 1.0)
    thetaR = np.clip((TR_phase - 217.0) / peak_excess, 0.0, 1.0)
    J_shape = float(np.clip(_trapz(np.abs(thetaL - thetaR), phase), 0.0, 1.0))

    # 最坏分量准则不依赖人为加权：只有形状和持续时间都改善，
    # 主指标才会显著下降。
    J_sym = float(max(J_shape, J_tau))

    return {
        "A_L": float(A_L),
        "A_R": float(A_R),
        "E_sym": float(E_sym),
        "J_sym": J_sym,
        "J_shape": J_shape,
        "J_overlap": J_overlap,
        "J_A": J_A,
        "J_tau": J_tau,
        "tau_L": float(tau_L),
        "tau_R": float(tau_R),
        "tau": tau,
        "qL": qL,
        "qR": qR,
        "phase": phase,
        "thetaL": thetaL,
        "thetaR": thetaR,
    }


@dataclass
class Eval4:
    y: np.ndarray
    z: np.ndarray
    A_L: float
    A_R: float
    J_sym: float
    J_shape: float
    J_overlap: float
    E_sym: float
    J_A: float
    J_tau: float
    V: float
    feasible_process: bool
    feasible: bool  # 含面积上限
    metrics: dict
    constraints: np.ndarray  # c1..c10
    margins: dict
    t: np.ndarray | None = None
    T: np.ndarray | None = None
    tau: np.ndarray | None = None
    qL: np.ndarray | None = None
    qR: np.ndarray | None = None
    phase: np.ndarray | None = None
    thetaL: np.ndarray | None = None
    thetaR: np.ndarray | None = None


@dataclass
class EvalCounter:
    n: int = 0


def evaluate_y4(
    y: np.ndarray,
    plate: q1.PlateParams,
    counter: EvalCounter | None = None,
    dt: float = 0.5,
    eps_A: float | None = None,
    keep_curve: bool = False,
) -> Eval4:
    y = np.asarray(y, dtype=float)
    z = q3.encode(y)
    S = q3.expand_setpoints(y)
    v = float(y[4])
    t, T = q3.simulate_plate_fast(S, v, plate, dt=dt)
    if counter is not None:
        counter.n += 1

    i_peak = int(np.argmax(T))
    tp = float(t[i_peak])
    T_peak = float(T[i_peak])
    dTdt = np.gradient(T, t)

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

    t150 = q3._cross_time(t, T, 150.0, True)
    t190 = q3._cross_time(t, T, 190.0, True)
    if np.isfinite(t150) and np.isfinite(t190) and t190 > t150 and t190 <= tp + 1e-9:
        tau_150_190 = float(t190 - t150)
    else:
        tau_150_190 = float("nan")

    tu = q3._cross_time(t, T, 217.0, True)
    td = q3._cross_time(t, T, 217.0, False)
    if np.isfinite(tu) and np.isfinite(td) and td > tu and tu < tp < td:
        tau_217 = float(td - tu)
        sym = symmetry_metrics(t, T, tu, tp, td)
    else:
        tau_217 = float("nan")
        sym = {
            "A_L": float("nan"),
            "A_R": float("nan"),
            "E_sym": float("nan"),
            "J_sym": 1.0,
            "J_shape": 1.0,
            "J_overlap": 1.0,
            "J_A": 1.0,
            "J_tau": 1.0,
            "tau_L": float("nan"),
            "tau_R": float("nan"),
            "tau": None,
            "qL": None,
            "qR": None,
            "phase": None,
            "thetaL": None,
            "thetaR": None,
        }

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
    c = np.array([-margins[k] for k in margins], dtype=float)
    V_proc = float(np.sum(np.maximum(0.0, c / CONSTRAINT_SCALE)))
    feas_proc = bool(np.all(c <= 0.0) and np.isfinite(sym["A_L"]))

    A_L = float(sym["A_L"]) if np.isfinite(sym["A_L"]) else 1e6
    J_sym = float(sym["J_sym"])
    V = V_proc
    feas = feas_proc
    if eps_A is not None:
        c11 = A_L - float(eps_A)
        V = V_proc + max(0.0, c11 / max(float(eps_A), EPS_AREA))
        feas = feas_proc and (A_L <= float(eps_A) + 1e-9)

    metrics = {
        "T_peak": T_peak,
        "t_peak": tp,
        "t217_up": float(tu) if np.isfinite(tu) else None,
        "t217_dn": float(td) if np.isfinite(td) else None,
        "r_up_max": r_up_max,
        "r_up_min": r_up_min,
        "r_dn_min": r_dn_min,
        "r_dn_max": r_dn_max,
        "tau_150_190": tau_150_190,
        "tau_217": tau_217,
        "tau_L": sym["tau_L"],
        "tau_R": sym["tau_R"],
        "A_L": A_L if feas_proc else None,
        "A_R": sym["A_R"] if np.isfinite(sym["A_R"]) else None,
        "E_sym": sym["E_sym"] if np.isfinite(sym["E_sym"]) else None,
        "J_sym": J_sym,
        "J_shape": sym["J_shape"],
        "J_overlap": sym["J_overlap"],
        "J_A": sym["J_A"],
        "J_tau": sym["J_tau"],
    }
    return Eval4(
        y=y.copy(),
        z=z,
        A_L=A_L,
        A_R=float(sym["A_R"]) if np.isfinite(sym["A_R"]) else 1e6,
        J_sym=J_sym,
        J_shape=float(sym["J_shape"]),
        J_overlap=float(sym["J_overlap"]),
        E_sym=float(sym["E_sym"]) if np.isfinite(sym["E_sym"]) else 1e6,
        J_A=float(sym["J_A"]),
        J_tau=float(sym["J_tau"]),
        V=V,
        feasible_process=feas_proc,
        feasible=feas,
        metrics=metrics,
        constraints=c,
        margins=margins,
        t=t if keep_curve else None,
        T=T if keep_curve else None,
        tau=sym["tau"] if keep_curve else None,
        qL=sym["qL"] if keep_curve else None,
        qR=sym["qR"] if keep_curve else None,
        phase=sym["phase"] if keep_curve else None,
        thetaL=sym["thetaL"] if keep_curve else None,
        thetaR=sym["thetaR"] if keep_curve else None,
    )


def evaluate_z4(
    z: np.ndarray,
    plate: q1.PlateParams,
    counter: EvalCounter | None = None,
    dt: float = 0.5,
    eps_A: float | None = None,
    keep_curve: bool = False,
) -> Eval4:
    return evaluate_y4(q3.decode(z), plate, counter=counter, dt=dt, eps_A=eps_A, keep_curve=keep_curve)


def deb_better_jsym(a: Eval4, b: Eval4) -> bool:
    """最小化 J_sym（含面积上限时的 Deb 规则）。"""
    if a.feasible and not b.feasible:
        return True
    if (not a.feasible) and b.feasible:
        return False
    if a.feasible and b.feasible:
        if abs(a.J_sym - b.J_sym) > 1e-12:
            return a.J_sym < b.J_sym
        return a.A_L < b.A_L
    return a.V < b.V


# ---------------------------------------------------------------------------
# 初始种群 / CDE / COBYLA
# ---------------------------------------------------------------------------
def init_population(
    npop: int,
    rng: np.random.Generator,
    seeds: list[np.ndarray] | None = None,
) -> np.ndarray:
    sampler = LatinHypercube(d=DIM, seed=int(rng.integers(0, 2**31 - 1)))
    pop = sampler.random(n=npop)
    if seeds:
        for i, y in enumerate(seeds[:npop]):
            pop[i] = q3.encode(np.asarray(y, dtype=float))
    return pop


def run_cde_jsym(
    plate: q1.PlateParams,
    npop: int,
    max_eval: int,
    dt: float,
    rng: np.random.Generator,
    eps_A: float | None,
    seeds: list[np.ndarray] | None = None,
    F: float = 0.7,
    CR: float = 0.9,
) -> tuple[Eval4, list[Eval4]]:
    counter = EvalCounter()
    pop_z = init_population(npop, rng, seeds=seeds)
    pop = [evaluate_z4(z, plate, counter=counter, dt=dt, eps_A=eps_A) for z in pop_z]
    best = pop[0]
    for ind in pop[1:]:
        if deb_better_jsym(ind, best):
            best = ind

    while counter.n < max_eval:
        for i in range(npop):
            if counter.n >= max_eval:
                break
            idxs = list(range(npop))
            idxs.remove(i)
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
            mutant = q3.reflect_bounds(pop_z[r1] + F * (pop_z[r2] - pop_z[r3]))
            jrand = int(rng.integers(0, DIM))
            trial = pop_z[i].copy()
            for j in range(DIM):
                if rng.random() <= CR or j == jrand:
                    trial[j] = mutant[j]
            trial = q3.reflect_bounds(trial)
            child = evaluate_z4(trial, plate, counter=counter, dt=dt, eps_A=eps_A)
            if deb_better_jsym(child, pop[i]):
                pop[i] = child
                pop_z[i] = trial
                if deb_better_jsym(child, best):
                    best = child
    return best, pop


def select_elites(pop: list[Eval4], n: int = 3, min_dist: float = 0.05) -> list[Eval4]:
    feas = [e for e in pop if e.feasible]
    feas.sort(key=lambda e: (e.J_sym, e.A_L))
    elites: list[Eval4] = []
    for e in feas:
        if all(np.linalg.norm(e.z - o.z) >= min_dist for o in elites):
            elites.append(e)
        if len(elites) >= n:
            break
    if len(elites) < n:
        for e in feas:
            if all(e is not o for o in elites):
                elites.append(e)
            if len(elites) >= n:
                break
    if not elites:
        elites = sorted(pop, key=lambda e: e.V)[:n]
    return elites


def refine_cobyla_jsym(
    elite: Eval4,
    plate: q1.PlateParams,
    dt: float,
    eps_A: float | None,
    maxfun: int = 60,
) -> Eval4:
    cache: dict[bytes, Eval4] = {}

    def eval_cached(z: np.ndarray) -> Eval4:
        key = np.asarray(z, dtype=float).tobytes()
        if key not in cache:
            cache[key] = evaluate_z4(z, plate, dt=dt, eps_A=eps_A)
        return cache[key]

    def objective(z):
        ev = eval_cached(z)
        return ev.J_sym if ev.feasible else ev.J_sym + 10.0 * (1.0 + ev.V)

    cons = []
    for j in range(10):

        def make_c(jj):
            def fun(z, j=jj):
                return -eval_cached(z).constraints[j]

            return fun

        cons.append({"type": "ineq", "fun": make_c(j)})

    if eps_A is not None:

        def area_con(z, ea=eps_A):
            return float(ea) - eval_cached(z).A_L

        cons.append({"type": "ineq", "fun": area_con})

    for j in range(DIM):

        def lo(z, j=j):
            return float(z[j])

        def hi(z, j=j):
            return float(1.0 - z[j])

        cons.append({"type": "ineq", "fun": lo})
        cons.append({"type": "ineq", "fun": hi})

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
    if deb_better_jsym(cand, elite):
        return cand
    return elite


def multi_start_cobyla(
    pop: list[Eval4],
    plate: q1.PlateParams,
    dt: float,
    eps_A: float | None,
    n_elites: int = 3,
    maxfun: int = 60,
) -> Eval4:
    elites = select_elites(pop, n=n_elites)
    best = elites[0]
    for e in elites:
        refined = refine_cobyla_jsym(e, plate, dt=dt, eps_A=eps_A, maxfun=maxfun)
        if deb_better_jsym(refined, best):
            best = refined
    return best


# ---------------------------------------------------------------------------
# Pareto / 膝点
# ---------------------------------------------------------------------------
def nondominated(points: list[Eval4]) -> list[Eval4]:
    feas = [p for p in points if p.feasible_process and np.isfinite(p.A_L)]
    keep = []
    for i, a in enumerate(feas):
        dominated = False
        for j, b in enumerate(feas):
            if i == j:
                continue
            if (b.A_L <= a.A_L and b.J_sym <= a.J_sym) and (
                b.A_L < a.A_L - 1e-9 or b.J_sym < a.J_sym - 1e-12
            ):
                dominated = True
                break
        if not dominated:
            keep.append(a)
    keep.sort(key=lambda e: e.A_L)
    # 不重复保留同一目标点；重复点会制造虚假的竖直/水平前沿线段。
    unique: list[Eval4] = []
    for e in keep:
        if not any(abs(e.A_L - u.A_L) <= 1e-7 and abs(e.J_sym - u.J_sym) <= 1e-10 for u in unique):
            unique.append(e)
    return unique


def practical_front(points: list[Eval4], j_tol: float = 5e-4) -> list[Eval4]:
    """对精确 Pareto 点作 ε-支配筛选，去掉数值上无意义的平台点。

    dt=0.025 与 dt=0.0125 的复核表明 J_sym 的离散误差约为 3e-4，
    因此面积明显增加而 J_sym 改善不足 5e-4 的点不应左右膝点判断。
    两个极端端点始终保留。
    """
    exact = nondominated(points)
    if len(exact) <= 2:
        return exact
    kept = [exact[0]]
    for e in exact[1:-1]:
        if e.J_sym < kept[-1].J_sym - j_tol:
            kept.append(e)
    if exact[-1] is not kept[-1]:
        kept.append(exact[-1])
    return kept


def densify_front(
    points: list[Eval4],
    plate: q1.PlateParams,
    dt: float,
    max_area_step: float = 2.5,
    j_tol: float = 5e-4,
) -> list[Eval4]:
    """用相邻工艺参数的延续插值填补前沿大空档，并在细网格复核。"""
    anchors = practical_front(points, j_tol=j_tol)
    candidates = list(anchors)
    for left, right in zip(anchors[:-1], anchors[1:]):
        gap = right.A_L - left.A_L
        n_seg = max(1, int(np.ceil(gap / max_area_step)))
        for k in range(1, n_seg):
            f = k / n_seg
            y = (1.0 - f) * left.y + f * right.y
            ev = evaluate_y4(y, plate, dt=dt)
            if ev.feasible_process:
                candidates.append(ev)
    return practical_front(candidates, j_tol=j_tol)


def select_knee_and_ideal(
    front: list[Eval4], A3: float, As: float, J3: float, Jmin: float
) -> tuple[Eval4 | None, Eval4 | None, list[dict]]:
    """返回 (膝点, 最近理想点, 归一化表)。"""
    if len(front) == 0:
        return None, None, []

    dA = As - A3
    dJ = J3 - Jmin
    rows = []
    for e in front:
        if abs(dA) < 1e-9:
            A_hat = 0.0
        else:
            A_hat = (e.A_L - A3) / dA
        if abs(dJ) < 1e-12:
            J_hat = 0.0
        else:
            J_hat = (e.J_sym - Jmin) / dJ
        A_hat = float(np.clip(A_hat, 0.0, 1.0))
        J_hat = float(np.clip(J_hat, 0.0, 1.0))
        # 有符号距离：正值才表示点位于端点连线靠近理想点的一侧。
        # 取绝对值会把“远离理想点的内凹平台”误判为膝点。
        d_line = (1.0 - A_hat - J_hat) / np.sqrt(2.0)
        D_ideal = float(np.hypot(A_hat, J_hat))
        rows.append(
            {
                "A_L": e.A_L,
                "J_sym": e.J_sym,
                "A_hat": A_hat,
                "J_hat": J_hat,
                "d_line": d_line,
                "D_ideal": D_ideal,
                "y": e.y.copy(),
                "ev": e,
            }
        )

    # 膝点：内部点到端点连线距离最大
    interior = [r for r in rows if 0.02 < r["A_hat"] < 0.98]
    if not interior:
        interior = rows
    knee_row = max(interior, key=lambda r: r["d_line"])
    ideal_row = min(rows, key=lambda r: r["D_ideal"])
    return knee_row["ev"], ideal_row["ev"], rows


# ---------------------------------------------------------------------------
# 轻量 NSGA-II（交叉验证）
# ---------------------------------------------------------------------------
def run_nsga2(
    plate: q1.PlateParams,
    npop: int,
    max_eval: int,
    dt: float,
    rng: np.random.Generator,
    seeds: list[np.ndarray] | None = None,
) -> list[Eval4]:
    """简化 NSGA-II：非支配排序 + 拥挤距离，Deb 约束支配。"""
    counter = EvalCounter()
    pop_z = init_population(npop, rng, seeds=seeds)
    pop = [evaluate_z4(z, plate, counter=counter, dt=dt, eps_A=None) for z in pop_z]

    def dominates(a: Eval4, b: Eval4) -> bool:
        # 约束支配
        if a.feasible_process and not b.feasible_process:
            return True
        if (not a.feasible_process) and b.feasible_process:
            return False
        if (not a.feasible_process) and (not b.feasible_process):
            return a.V < b.V
        return (a.A_L <= b.A_L and a.J_sym <= b.J_sym) and (
            a.A_L < b.A_L - 1e-9 or a.J_sym < b.J_sym - 1e-12
        )

    def nondom_sort(inds: list[Eval4]) -> list[list[int]]:
        n = len(inds)
        S = [[] for _ in range(n)]
        n_dom = [0] * n
        fronts: list[list[int]] = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if dominates(inds[p], inds[q]):
                    S[p].append(q)
                elif dominates(inds[q], inds[p]):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return fronts[:-1]

    def crowding(inds: list[Eval4], front: list[int]) -> dict[int, float]:
        dist = {i: 0.0 for i in front}
        if len(front) <= 2:
            for i in front:
                dist[i] = float("inf")
            return dist
        for obj_get in (lambda e: e.A_L, lambda e: e.J_sym):
            ordered = sorted(front, key=lambda i: obj_get(inds[i]))
            dist[ordered[0]] = float("inf")
            dist[ordered[-1]] = float("inf")
            vals = [obj_get(inds[i]) for i in ordered]
            span = vals[-1] - vals[0]
            if span < 1e-15:
                continue
            for k in range(1, len(ordered) - 1):
                dist[ordered[k]] += (vals[k + 1] - vals[k - 1]) / span
        return dist

    def tournament(a: Eval4, b: Eval4, rank_a: int, rank_b: int, cd_a: float, cd_b: float) -> Eval4:
        if rank_a < rank_b:
            return a
        if rank_b < rank_a:
            return b
        return a if cd_a >= cd_b else b

    while counter.n < max_eval:
        fronts = nondom_sort(pop)
        rank = {}
        crowd = {}
        for r, fr in enumerate(fronts):
            cd = crowding(pop, fr)
            for i in fr:
                rank[i] = r
                crowd[i] = cd[i]

        offspring_z = []
        offspring = []
        while len(offspring) < npop and counter.n < max_eval:
            i1, i2 = rng.choice(npop, size=2, replace=False)
            p1 = tournament(pop[i1], pop[i2], rank[i1], rank[i2], crowd[i1], crowd[i2])
            j1, j2 = rng.choice(npop, size=2, replace=False)
            p2 = tournament(pop[j1], pop[j2], rank[j1], rank[j2], crowd[j1], crowd[j2])
            # SBX + 多项式变异（复用 q3）
            c1z, c2z = q3._sbx(p1.z, p2.z, rng)
            c1z = q3._poly_mutate(c1z, rng)
            child = evaluate_z4(c1z, plate, counter=counter, dt=dt, eps_A=None)
            offspring_z.append(c1z)
            offspring.append(child)
            if len(offspring) >= npop or counter.n >= max_eval:
                break
            c2z = q3._poly_mutate(c2z, rng)
            child2 = evaluate_z4(c2z, plate, counter=counter, dt=dt, eps_A=None)
            offspring_z.append(c2z)
            offspring.append(child2)

        combined = pop + offspring
        fronts = nondom_sort(combined)
        new_pop: list[Eval4] = []
        for fr in fronts:
            if len(new_pop) + len(fr) <= npop:
                new_pop.extend(combined[i] for i in fr)
            else:
                cd = crowding(combined, fr)
                ordered = sorted(fr, key=lambda i: cd[i], reverse=True)
                need = npop - len(new_pop)
                new_pop.extend(combined[i] for i in ordered[:need])
                break
        pop = new_pop

    return nondominated(pop)


# ---------------------------------------------------------------------------
# I/O / 绘图
# ---------------------------------------------------------------------------
def load_plate() -> q1.PlateParams:
    return q3.load_plate()


def load_q3_solution() -> tuple[np.ndarray, float]:
    path = ROOT / "results" / "q3" / "summary.json"
    if not path.exists():
        raise FileNotFoundError("未找到 results/q3/summary.json，请先运行 python code/q3.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    y = data["best"]["y"]
    y3 = np.array([y["S1_5"], y["S6"], y["S7"], y["S8_9"], y["v"]], dtype=float)
    A3 = float(data["best"]["A"])
    return y3, A3


def calibrate_area_endpoint(
    candidate_y: list[np.ndarray],
    plate: q1.PlateParams,
    dt_refine: float,
    dt_verify: float,
    n_elites: int,
    maxfun_refine: int,
    maxfun_verify: int,
) -> Eval4:
    """在与问题四一致的细网格上重新校准最小面积端点。

    问题三历史结果可能只在粗网格上完成优化、随后仅做细网格复算。
    直接把该点当作 Pareto 左端，会导致第四问搜索到比所谓面积最优点
    面积更小的可行解。这里从问题三解和纯对称搜索种群中挑选低面积
    种子，先在 dt_refine 上多起点 COBYLA，再在 dt_verify 上精修一次。
    """
    unique_y: list[np.ndarray] = []
    for y in candidate_y:
        yy = np.asarray(y, dtype=float)
        zy = q3.encode(yy)
        if not any(np.linalg.norm(zy - q3.encode(old)) < 1e-8 for old in unique_y):
            unique_y.append(yy.copy())

    mid_pop = [q3.evaluate_y(y, plate, dt=dt_refine) for y in unique_y]
    mid_best = q3.multi_start_cobyla(
        mid_pop,
        plate,
        dt=dt_refine,
        n_elites=n_elites,
        maxfun=maxfun_refine,
    )

    verify_seed = q3.evaluate_y(mid_best.y, plate, dt=dt_verify)
    verify_best = q3.refine_cobyla(
        verify_seed,
        plate,
        dt=dt_verify,
        maxfun=maxfun_verify,
    )
    if not verify_best.feasible:
        verified = [q3.evaluate_y(e.y, plate, dt=dt_verify) for e in mid_pop if e.feasible]
        verified = [e for e in verified if e.feasible]
        if not verified:
            raise RuntimeError("无法在高精度网格上恢复可行的面积端点")
        verify_best = min(verified, key=lambda e: e.A)

    return evaluate_y4(verify_best.y, plate, dt=dt_verify, keep_curve=True)


def ev_to_dict(ev: Eval4) -> dict:
    return {
        "y": {n: float(v) for n, v in zip(VAR_NAMES, ev.y)},
        "setpoints": q3.expand_setpoints(ev.y).tolist(),
        "A_L": float(ev.A_L),
        "A_R": float(ev.A_R),
        "J_sym": float(ev.J_sym),
        "J_shape": float(ev.J_shape),
        "J_overlap": float(ev.J_overlap),
        "E_sym": float(ev.E_sym),
        "J_A": float(ev.J_A),
        "J_tau": float(ev.J_tau),
        "feasible": bool(ev.feasible_process),
        "metrics": {
            k: (float(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else v)
            for k, v in ev.metrics.items()
        },
        "margins": {k: float(v) for k, v in ev.margins.items()},
    }


def plot_curve(ev: Eval4, path: Path, title: str) -> None:
    assert ev.t is not None and ev.T is not None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ev.t, ev.T, color="#1f4e79", lw=1.8, label="炉温曲线")
    ax.axhline(217, color="#c0392b", ls="--", lw=1, label="217 °C")
    tu, tp, td = ev.metrics.get("t217_up"), ev.metrics["t_peak"], ev.metrics.get("t217_dn")
    if tu is not None and td is not None:
        mL = (ev.t >= tu) & (ev.t <= tp)
        mR = (ev.t >= tp) & (ev.t <= td)
        ax.fill_between(ev.t[mL], 217, ev.T[mL], where=ev.T[mL] >= 217, color="#f4a261", alpha=0.5, label=f"A_L={ev.A_L:.1f}")
        ax.fill_between(ev.t[mR], 217, ev.T[mR], where=ev.T[mR] >= 217, color="#2a9d8f", alpha=0.35, label=f"A_R={ev.A_R:.1f}")
        ax.axvline(tp, color="gray", ls=":", lw=1, label="峰值")
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("温度 T / °C")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mirror(ev: Eval4, path: Path) -> None:
    if (
        ev.tau is None
        or ev.qL is None
        or ev.qR is None
        or ev.phase is None
        or ev.thetaL is None
        or ev.thetaR is None
    ):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(ev.tau, ev.qL, label="q_L（左侧）", lw=1.8)
    ax.plot(ev.tau, ev.qR, label="q_R（右侧）", lw=1.8)
    ax.fill_between(
        ev.tau,
        ev.qL,
        ev.qR,
        color="#e9c46a",
        alpha=0.45,
        label=f"绝对镜像误差 E={ev.E_sym:.2f}",
    )
    ax.set_xlabel("相对峰值时间偏移 τ / s")
    ax.set_ylabel("超温 T-217 / °C")
    ax.set_title(f"实际时间镜像  τL={ev.metrics['tau_L']:.2f}s, τR={ev.metrics['tau_R']:.2f}s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ev.phase, ev.thetaL, label="θ_L（左侧归一化）", lw=1.8)
    ax.plot(ev.phase, ev.thetaR, label="θ_R（右侧归一化）", lw=1.8)
    ax.fill_between(ev.phase, ev.thetaL, ev.thetaR, color="#90be6d", alpha=0.35)
    ax.set_xlabel("各侧归一化相位 s")
    ax.set_ylabel("归一化超温")
    ax.set_title(
        f"形状 J_shape={ev.J_shape:.4f}, 时间 J_tau={ev.J_tau:.4f}\n"
        f"主指标 max={ev.J_sym:.4f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pareto(
    front_rows: list[dict],
    knee: Eval4 | None,
    ideal: Eval4 | None,
    y3: Eval4,
    ys: Eval4,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    AL = [r["A_L"] for r in front_rows]
    JS = [r["J_sym"] for r in front_rows]
    ax.plot(AL, JS, "o-", color="#1f4e79", label="Pareto 前沿")
    ax.scatter(
        [y3.A_L],
        [y3.J_sym],
        s=80,
        marker="s",
        color="#e76f51",
        label="面积端点（细网格校准）",
        zorder=5,
    )
    ax.scatter([ys.A_L], [ys.J_sym], s=80, marker="D", color="#2a9d8f", label="对称端点", zorder=5)
    if knee is not None:
        ax.scatter([knee.A_L], [knee.J_sym], s=120, marker="*", color="#e9c46a", edgecolors="k", label="膝点", zorder=6)
    if ideal is not None:
        ax.scatter([ideal.A_L], [ideal.J_sym], s=90, marker="^", color="#264653", label="最近理想点", zorder=5)
    ax.set_xlabel(r"$A_L$ / (°C·s)")
    ax.set_ylabel(r"$J_{\mathrm{sym}}$")
    ax.set_title("问题四 Pareto 前沿")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def neighborhood_check(ev: Eval4, plate: q1.PlateParams, dt: float) -> list[dict]:
    deltas = [1.0, 1.0, 1.0, 1.0, 0.5]
    rows = []
    for j, name in enumerate(VAR_NAMES):
        for sign in (-1.0, 1.0):
            y2 = ev.y.copy()
            y2[j] = float(np.clip(y2[j] + sign * deltas[j], L_BOUNDS[j], U_BOUNDS[j]))
            if abs(y2[j] - ev.y[j]) < 1e-12:
                continue
            e2 = evaluate_y4(y2, plate, dt=dt)
            rows.append(
                {
                    "var": name,
                    "delta": sign * deltas[j],
                    "A_L": e2.A_L,
                    "J_sym": e2.J_sym,
                    "J_shape": e2.J_shape,
                    "J_tau": e2.J_tau,
                    "feasible": e2.feasible_process,
                    "dA": e2.A_L - ev.A_L if e2.feasible_process else None,
                    "dJ": e2.J_sym - ev.J_sym if e2.feasible_process else None,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="问题四：面积—对称性双目标优化")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--nsga", action="store_true", help="额外跑 NSGA-II 交叉验证")
    p.add_argument("--seed", type=int, default=2020)
    return p.parse_args()


def budget_from_args(args: argparse.Namespace) -> dict:
    if args.fast:
        return dict(
            npop=20,
            max_eval_end=500,
            max_eval_eps=350,
            dt_search=0.25,
            dt_refine=0.1,
            dt_verify=0.05,
            elites=2,
            cobyla_maxfun=40,
            cobyla_high_maxfun=30,
            area_elites=3,
            area_cobyla_maxfun=50,
            area_high_maxfun=35,
            lambdas=np.linspace(0.0, 1.0, 5),
        )
    if args.full:
        return dict(
            npop=44,
            max_eval_end=3000,
            max_eval_eps=1500,
            dt_search=0.1,
            dt_refine=0.05,
            dt_verify=0.025,
            elites=5,
            cobyla_maxfun=100,
            cobyla_high_maxfun=80,
            area_elites=7,
            area_cobyla_maxfun=160,
            area_high_maxfun=100,
            lambdas=np.linspace(0.0, 1.0, 11),
        )
    return dict(
        npop=32,
        max_eval_end=1200,
        max_eval_eps=650,
        dt_search=0.2,
        dt_refine=0.05,
        dt_verify=0.025,
        elites=3,
        cobyla_maxfun=60,
        cobyla_high_maxfun=45,
        area_elites=5,
        area_cobyla_maxfun=100,
        area_high_maxfun=70,
        lambdas=np.linspace(0.0, 1.0, 7),
    )


def main() -> None:
    args = parse_args()
    cfg = budget_from_args(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("问题四：阴影面积与对称性双目标优化")
    print(
        f"种群={cfg['npop']}, 端点预算={cfg['max_eval_end']}, "
        f"ε子问题预算={cfg['max_eval_eps']}, λ点数={len(cfg['lambdas'])}, "
        f"dt={cfg['dt_search']}/{cfg['dt_refine']}/{cfg['dt_verify']}"
    )
    print("=" * 60)

    plate = load_plate()
    y3_raw, A3_file = load_q3_solution()
    print(f"读取问题三解: y={y3_raw}, A_file={A3_file:.4f}")

    # ---- 阶段一：读取旧端点，搜索纯对称端点 ----
    print("\n[1] 复算问题三输入解 ...")
    y3_input_ev = evaluate_y4(y3_raw, plate, dt=cfg["dt_verify"], keep_curve=True)
    print(
        f"  y3_input: A_L={y3_input_ev.A_L:.4f}, J_sym={y3_input_ev.J_sym:.6f}, "
        f"feas={y3_input_ev.feasible_process}"
    )

    print("[2] 求纯对称最优端点 (min J_sym) ...")
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    best_s, pop_s = run_cde_jsym(
        plate,
        cfg["npop"],
        cfg["max_eval_end"],
        cfg["dt_search"],
        rng,
        eps_A=None,
        seeds=[y3_raw],
    )
    best_s = multi_start_cobyla(
        pop_s, plate, cfg["dt_search"], None, n_elites=cfg["elites"], maxfun=cfg["cobyla_maxfun"]
    )
    # 在高精度模型上重新排序并局部优化，不能只把粗网格结果拿来复算。
    hi_seed_s = select_elites(pop_s + [best_s], n=max(2, cfg["elites"] * 2), min_dist=0.02)
    hi_pop_s = [evaluate_y4(e.y, plate, dt=cfg["dt_refine"]) for e in hi_seed_s]
    best_s_hi = multi_start_cobyla(
        hi_pop_s,
        plate,
        cfg["dt_refine"],
        None,
        n_elites=cfg["elites"],
        maxfun=cfg["cobyla_high_maxfun"],
    )
    ys_ev = evaluate_y4(best_s_hi.y, plate, dt=cfg["dt_verify"], keep_curve=True)
    As = ys_ev.A_L
    Jmin = ys_ev.J_sym
    print(
        f"  ys: A_L={As:.4f}, J_sym={Jmin:.6f}, feas={ys_ev.feasible_process}, "
        f"用时 {time.perf_counter()-t0:.1f}s"
    )

    # 问题三旧结果可能只在粗网格上优化。利用已产生的种群重新校准面积端点，
    # 防止 Pareto 前沿出现“比问题三最优点面积还小”的逻辑矛盾。
    print("[3] 在细网格上校准最小面积端点 ...")
    t_area = time.perf_counter()
    y_area_ev = calibrate_area_endpoint(
        [y3_raw] + [e.y for e in pop_s] + [best_s.y, ys_ev.y],
        plate,
        dt_refine=cfg["dt_refine"],
        dt_verify=cfg["dt_verify"],
        n_elites=cfg["area_elites"],
        maxfun_refine=cfg["area_cobyla_maxfun"],
        maxfun_verify=cfg["area_high_maxfun"],
    )
    A3 = y_area_ev.A_L
    J3 = y_area_ev.J_sym
    print(
        f"  area: A_L={A3:.4f}, J_sym={J3:.6f}, feas={y_area_ev.feasible_process}, "
        f"旧端点面积差={y3_input_ev.A_L-A3:.4f}, 用时 {time.perf_counter()-t_area:.1f}s"
    )

    if abs(As - A3) < 1e-3 and abs(Jmin - J3) < 1e-6:
        print("  两端点几乎重合；仍扫描放宽面积上限，确认是否存在可改善对称性的折中。")
        As_scan = A3 * 1.25  # 允许至多牺牲约 25% 面积去换对称性
    else:
        As_scan = As

    # 对称搜索种群中的可行个体也进入候选
    print("\n[4] ε-约束扫描 Pareto ...")
    lambdas = list(cfg["lambdas"])
    pareto_cands: list[Eval4] = [y3_input_ev, y_area_ev, ys_ev]
    for e in pop_s:
        if e.feasible_process:
            pareto_cands.append(evaluate_y4(e.y, plate, dt=cfg["dt_verify"]))

    prev_y = y_area_ev.y.copy()
    for k, lam in enumerate(lambdas):
        eps_A = A3 + float(lam) * (As_scan - A3)
        # 粗网格只负责产生候选；面积边界在高精度局部阶段重新执行。
        eps_search = eps_A + 2.0
        print(f"  λ={lam:.2f}, ε_A={eps_A:.3f} ...", end="", flush=True)
        rng_k = np.random.default_rng(args.seed + 10 + k)
        seeds = [y_area_ev.y, ys_ev.y, prev_y, y3_raw]
        best_k, pop_k = run_cde_jsym(
            plate,
            cfg["npop"],
            cfg["max_eval_eps"],
            cfg["dt_search"],
            rng_k,
            eps_A=eps_search,
            seeds=seeds,
        )
        best_k = multi_start_cobyla(
            pop_k,
            plate,
            cfg["dt_search"],
            eps_search,
            n_elites=cfg["elites"],
            maxfun=cfg["cobyla_maxfun"],
        )
        # 用较细网格重新排序若干粗网格精英，并再次运行 COBYLA。
        hi_seed_k = select_elites(pop_k + [best_k], n=max(2, cfg["elites"] * 2), min_dist=0.02)
        eps_refine = eps_A + 0.35
        hi_pop_k = [
            evaluate_y4(e.y, plate, dt=cfg["dt_refine"], eps_A=eps_refine)
            for e in hi_seed_k
        ]
        best_k_hi = multi_start_cobyla(
            hi_pop_k,
            plate,
            cfg["dt_refine"],
            eps_refine,
            n_elites=cfg["elites"],
            maxfun=cfg["cobyla_high_maxfun"],
        )
        for e in pop_k:
            if e.feasible:  # 满足面积上限+工艺
                pareto_cands.append(evaluate_y4(e.y, plate, dt=cfg["dt_verify"]))
        hi = evaluate_y4(best_k_hi.y, plate, dt=cfg["dt_verify"], keep_curve=True)
        if hi.feasible_process and hi.A_L <= eps_A + 0.75:
            pareto_cands.append(hi)
            prev_y = hi.y.copy()
            print(f" A_L={hi.A_L:.3f}, J={hi.J_sym:.5f}")
        else:
            print(f" 丢弃(高精度不可用/超面积) A_L={hi.A_L:.3f}, feas={hi.feasible_process}")

    exact_front = nondominated(pareto_cands)

    # 若 ε 扫描又发现了更小面积点，同步更新面积端点，避免端点被前沿支配。
    if exact_front:
        area_front = min(exact_front, key=lambda e: (e.A_L, e.J_sym))
        if area_front.A_L < y_area_ev.A_L - 1e-7:
            y_area_ev = evaluate_y4(
                area_front.y, plate, dt=cfg["dt_verify"], keep_curve=True
            )

    # 对数值平台作 ε-支配筛选，再沿相邻工艺参数做连续插值并细网格复核。
    # 这一步只补充实际可行点，不对目标值进行图形插值。
    front = densify_front(
        exact_front,
        plate,
        dt=cfg["dt_verify"],
        max_area_step=2.5,
        j_tol=5e-4,
    )

    # 加密点先用于定位最近理想区域，再对该区域做一次真正的约束局部优化；
    # 最终推荐解因此不是简单的参数线性插值。
    if front:
        A_tmp = [e.A_L for e in front]
        J_tmp = [e.J_sym for e in front]
        _, ideal_seed, _ = select_knee_and_ideal(
            front,
            min(A_tmp),
            max(A_tmp),
            max(J_tmp),
            min(J_tmp),
        )
        if ideal_seed is not None:
            local_eps = ideal_seed.A_L + 0.75
            local_seed = evaluate_y4(
                ideal_seed.y,
                plate,
                dt=cfg["dt_refine"],
                eps_A=local_eps,
            )
            local_mid = refine_cobyla_jsym(
                local_seed,
                plate,
                dt=cfg["dt_refine"],
                eps_A=local_eps,
                maxfun=max(cfg["cobyla_high_maxfun"], 60),
            )
            local_hi = evaluate_y4(local_mid.y, plate, dt=cfg["dt_verify"])
            if (
                local_hi.feasible_process
                and local_hi.A_L <= ideal_seed.A_L + 1.5
                and local_hi.J_sym < ideal_seed.J_sym - 1e-6
            ):
                exact_front = nondominated(exact_front + [local_hi])
                front = densify_front(
                    exact_front,
                    plate,
                    dt=cfg["dt_verify"],
                    max_area_step=2.5,
                    j_tol=5e-4,
                )
                print(
                    f"  理想区域局部精修: A_L={local_hi.A_L:.3f}, "
                    f"J={local_hi.J_sym:.5f}"
                )
    print(f"  精确非支配点数: {len(exact_front)}; ε-支配并加密后: {len(front)}")

    # ε 扫描可能找到比独立端点搜索更好的纯对称解；报告端点必须取
    # 全部已发现非支配解中的最小 J_sym，避免“对称端点”被前沿点支配。
    if front:
        ys_front = min(front, key=lambda e: (e.J_sym, e.A_L))
        if ys_front.J_sym < ys_ev.J_sym - 1e-10 or (
            abs(ys_front.J_sym - ys_ev.J_sym) <= 1e-10
            and ys_front.A_L < ys_ev.A_L - 1e-7
        ):
            ys_ev = evaluate_y4(ys_front.y, plate, dt=cfg["dt_verify"], keep_curve=True)

    # 膝点选择用实际观察到的端点范围
    A_front = [e.A_L for e in front]
    J_front = [e.J_sym for e in front]
    A3_use, As_use = min(A_front), max(A_front)
    Jmin_use, J3_use = min(J_front), max(J_front)
    knee, ideal, norm_rows = select_knee_and_ideal(front, A3_use, As_use, J3_use, Jmin_use)
    # 非凸或近似线性的前沿上，几何膝点对采样密度较敏感；最近理想点
    # 同时惩罚两个归一化目标，作为主推荐点更稳定。膝点保留作诊断。
    final = ideal if ideal is not None else (knee if knee is not None else y_area_ev)
    if ideal is not None:
        print(
            f"\n[5] 膝点: A_L={knee.A_L:.4f}, J_sym={knee.J_sym:.6f}; "
            f"最近理想点/推荐解: A_L={ideal.A_L:.4f}, J_sym={ideal.J_sym:.6f}"
        )
    # 若前沿几乎单点，最终仍优先报告校准后的面积端点。
    if abs(As_use - A3_use) < 1.0 and abs(J3_use - Jmin_use) < 1e-4:
        print("  Pareto 几乎退化为单点，最终采用校准后的面积端点。")
        final = y_area_ev
        knee = y_area_ev
        ideal = y_area_ev

    # 最终高精度曲线（保证带曲线）
    final = evaluate_y4(final.y, plate, dt=cfg["dt_verify"], keep_curve=True)

    # ---- NSGA-II 可选 ----
    nsga_front: list[Eval4] = []
    if args.nsga:
        print("\n[6] NSGA-II 交叉验证 ...")
        rng_n = np.random.default_rng(args.seed + 99)
        nsga_raw = run_nsga2(
            plate,
            cfg["npop"],
            cfg["max_eval_end"] + cfg["max_eval_eps"],
            cfg["dt_search"],
            rng_n,
            seeds=[y3_raw, ys_ev.y],
        )
        nsga_front = [
            evaluate_y4(e.y, plate, dt=cfg["dt_verify"]) for e in nsga_raw if e.feasible_process
        ]
        nsga_front = nondominated(nsga_front)
        print(f"  NSGA-II 非支配点: {len(nsga_front)}")

    print("\n[7] 邻域检查与导出 ...")
    neigh = neighborhood_check(final, plate, dt=cfg["dt_verify"])
    pd.DataFrame(neigh).to_csv(OUT_DIR / "neighborhood.csv", index=False, encoding="utf-8-sig")

    assert final.t is not None and final.T is not None
    pd.DataFrame({"t": final.t, "T": final.T}).to_csv(
        OUT_DIR / "best_curve.csv", index=False, encoding="utf-8-sig"
    )

    def pareto_row(e: Eval4) -> dict:
        return {
            "A_L": e.A_L,
            "A_R": e.A_R,
            "J_sym": e.J_sym,
            "J_shape": e.J_shape,
            "J_overlap": e.J_overlap,
            "E_sym": e.E_sym,
            "J_A": e.J_A,
            "J_tau": e.J_tau,
            **{n: float(v) for n, v in zip(VAR_NAMES, e.y)},
        }

    pareto_table = [pareto_row(e) for e in front]
    exact_pareto_table = [pareto_row(e) for e in exact_front]
    pd.DataFrame(pareto_table).to_csv(OUT_DIR / "pareto_front.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(exact_pareto_table).to_csv(
        OUT_DIR / "pareto_exact.csv", index=False, encoding="utf-8-sig"
    )

    if norm_rows:
        pd.DataFrame(
            [
                {
                    "A_L": r["A_L"],
                    "J_sym": r["J_sym"],
                    "A_hat": r["A_hat"],
                    "J_hat": r["J_hat"],
                    "d_line": r["d_line"],
                    "D_ideal": r["D_ideal"],
                }
                for r in norm_rows
            ]
        ).to_csv(OUT_DIR / "pareto_normalized.csv", index=False, encoding="utf-8-sig")

    summary = {
        "budget": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()
        },
        "seed": args.seed,
        "endpoint_q3_input": ev_to_dict(y3_input_ev),
        "endpoint_q3": ev_to_dict(y_area_ev),
        "endpoint_area": ev_to_dict(y_area_ev),
        "endpoint_sym": ev_to_dict(ys_ev),
        "final": ev_to_dict(final),
        "knee": ev_to_dict(knee) if knee is not None else None,
        "ideal": ev_to_dict(ideal) if ideal is not None else None,
        "pareto": pareto_table,
        "pareto_exact": exact_pareto_table,
        "nsga_pareto": [
            {
                "A_L": e.A_L,
                "J_sym": e.J_sym,
                "J_shape": e.J_shape,
                "J_tau": e.J_tau,
                **{n: float(v) for n, v in zip(VAR_NAMES, e.y)},
            }
            for e in nsga_front
        ],
        "plate_params": {
            "eta_pre": plate.eta_pre,
            "eta_soak": plate.eta_soak,
            "eta_ref": plate.eta_ref,
            "eta_cool": plate.eta_cool,
            "eta_r": plate.eta_r,
            "alpha": plate.alpha,
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_curve(
        final,
        FIG_DIR / "final_curve.png",
        f"问题四最终解  A_L={final.A_L:.2f}, J_sym={final.J_sym:.4f}, v={final.y[4]:.2f}",
    )
    plot_mirror(final, FIG_DIR / "mirror_compare.png")
    plot_mirror(y_area_ev, FIG_DIR / "area_endpoint_mirror.png")
    plot_mirror(ys_ev, FIG_DIR / "sym_endpoint_mirror.png")
    if norm_rows:
        plot_pareto(norm_rows, knee, ideal, y_area_ev, ys_ev, FIG_DIR / "pareto_front.png")
    plot_curve(
        y_area_ev,
        FIG_DIR / "q3_endpoint.png",
        f"面积端点（细网格校准）  A_L={y_area_ev.A_L:.2f}, J={y_area_ev.J_sym:.4f}",
    )
    plot_curve(ys_ev, FIG_DIR / "sym_endpoint.png", f"对称端点  A_L={ys_ev.A_L:.2f}, J={ys_ev.J_sym:.4f}")

    # 与问题三对比
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(y_area_ev.t, y_area_ev.T, label="面积端点（细网格校准）", lw=1.6)
    ax.plot(final.t, final.T, label="问题四最终", lw=1.6)
    ax.axhline(217, color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("温度 T / °C")
    ax.set_title("问题三 vs 问题四炉温曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q3_vs_q4.png", dpi=150)
    plt.close(fig)

    print("\n========== 问题四结果 ==========")
    print(
        f"问题三输入: A_L={y3_input_ev.A_L:.4f}, J_sym={y3_input_ev.J_sym:.6f}"
    )
    print(f"面积端点:   A_L={y_area_ev.A_L:.4f}, J_sym={y_area_ev.J_sym:.6f}")
    print(f"对称端点:   A_L={ys_ev.A_L:.4f}, J_sym={ys_ev.J_sym:.6f}")
    print(f"最终折中:   A_L={final.A_L:.4f}, J_sym={final.J_sym:.6f}")
    print(f"  S1-5={final.y[0]:.4f}, S6={final.y[1]:.4f}, S7={final.y[2]:.4f}, "
          f"S8-9={final.y[3]:.4f}, v={final.y[4]:.4f}")
    print(
        f"  J_shape={final.J_shape:.4f}, J_tau={final.J_tau:.4f}, "
        f"J_overlap(旧)={final.J_overlap:.4f}"
    )
    print(f"  E_sym={final.E_sym:.4f}, J_A={final.J_A:.4f}")
    print(f"输出: {OUT_DIR}")
    print(f"图像: {FIG_DIR}")


if __name__ == "__main__":
    main()
