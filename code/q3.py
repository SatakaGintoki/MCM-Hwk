"""问题三：在工艺约束下最小化升温侧超温面积。"""
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

ROOT = q1.ROOT
OUT_DIR = ROOT / "results" / "q3"
FIG_DIR = ROOT / "figures" / "q3"

L_BOUNDS = np.array([165.0, 185.0, 225.0, 245.0, 65.0])
U_BOUNDS = np.array([185.0, 205.0, 245.0, 265.0, 100.0])
DIM = 5
VAR_NAMES = ["S1_5", "S6", "S7", "S8_9", "v"]

CONSTRAINT_SCALE = np.array([3.0, 3.0, 3.0, 3.0, 60.0, 60.0, 50.0, 50.0, 10.0, 10.0])
SLOPE_UP_TOL = 1e-3

SEED_Q2 = np.array([182.0, 203.0, 237.0, 254.0, 79.60])


def encode(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y, dtype=float) - L_BOUNDS) / (U_BOUNDS - L_BOUNDS)


def decode(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    return L_BOUNDS + z * (U_BOUNDS - L_BOUNDS)


def expand_setpoints(y: np.ndarray) -> np.ndarray:
    s15, s6, s7, s89, _ = y
    return np.array([s15, s15, s15, s15, s15, s6, s7, s89, s89, 25.0, 25.0], dtype=float)


def reflect_bounds(z: np.ndarray) -> np.ndarray:
    """Reflect trial coordinates into the unit cube."""
    z = np.asarray(z, dtype=float).copy()
    for j in range(z.size):
        while z[j] < 0.0 or z[j] > 1.0:
            if z[j] < 0.0:
                z[j] = -z[j]
            if z[j] > 1.0:
                z[j] = 2.0 - z[j]
        z[j] = float(np.clip(z[j], 0.0, 1.0))
    return z


def simulate_plate_fast(
    setpoints: np.ndarray,
    u_cm_per_min: float,
    params: q1.PlateParams,
    dt: float = 0.5,
    n_nodes: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the calibrated model with a precomputed ambient field."""
    S = q1._as_setpoints(setpoints)
    v = u_cm_per_min / 60.0
    t_end = q1.FURNACE_LEN / v
    M = n_nodes - 1
    dz = q1.HALF_THICKNESS / M
    r = params.alpha * dt / dz**2
    n_steps = int(np.ceil(t_end / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x_all = v * t[1:]
    C_all = np.asarray(q1.ambient_temperature(x_all, S), dtype=float)

    T_center = np.empty(n_steps + 1)
    u = np.full(n_nodes, 25.0)
    T_center[0] = 25.0
    lower = np.zeros(n_nodes - 1)
    diag = np.zeros(n_nodes)
    upper = np.zeros(n_nodes - 1)

    for n in range(n_steps):
        C = float(C_all[n])
        Ck = C + 273.15
        Ts = u[-1]
        Tsk = Ts + 273.15
        Rcoef = (Ck + Tsk) * (Ck**2 + Tsk**2)
        eta = params.eta_at(float(x_all[n]))
        eta_eff = eta + params.eta_r * Rcoef

        diag[0] = 1.0 + 2.0 * r
        upper[0] = -2.0 * r
        for j in range(1, M):
            lower[j - 1] = -r
            diag[j] = 1.0 + 2.0 * r
            upper[j] = -r
        lower[M - 1] = -1.0 / dz
        diag[M] = 1.0 / dz + eta_eff

        rhs = u.copy()
        rhs[M] = eta_eff * C

        for _ in range(2):
            u_new = q1.thomas_solve(lower, diag, upper, rhs)
            Ts = u_new[-1]
            Tsk = Ts + 273.15
            Rcoef = (Ck + Tsk) * (Ck**2 + Tsk**2)
            eta_eff = eta + params.eta_r * Rcoef
            lower[M - 1] = -1.0 / dz
            diag[M] = 1.0 / dz + eta_eff
            rhs[:-1] = u[:-1]
            rhs[M] = eta_eff * C
            diag[0] = 1.0 + 2.0 * r
            upper[0] = -2.0 * r
            for j in range(1, M):
                lower[j - 1] = -r
                diag[j] = 1.0 + 2.0 * r
                upper[j] = -r

        u = q1.thomas_solve(lower, diag, upper, rhs)
        T_center[n + 1] = u[0]
    return t, T_center


def _cross_time(t: np.ndarray, T: np.ndarray, level: float, rising: bool) -> float:
    if rising:
        idx = np.where((T[:-1] < level) & (T[1:] >= level))[0]
    else:
        idx = np.where((T[:-1] >= level) & (T[1:] < level))[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0 if rising else -1])
    return float(t[i] + (level - T[i]) / (T[i + 1] - T[i] + 1e-30) * (t[i + 1] - t[i]))


def shadow_area(t: np.ndarray, T: np.ndarray, t217_up: float, t_peak: float) -> float:
    """Integrate superheat from the liquidus crossing to the peak."""
    if not (np.isfinite(t217_up) and np.isfinite(t_peak)) or t_peak <= t217_up:
        return float("nan")
    mask = (t >= t217_up) & (t <= t_peak)
    tt = t[mask]
    TT = T[mask]
    T217 = 217.0
    T_left = float(np.interp(t217_up, t, T))
    T_right = float(np.interp(t_peak, t, T))
    if tt.size == 0 or tt[0] > t217_up + 1e-12:
        tt = np.insert(tt, 0, t217_up)
        TT = np.insert(TT, 0, T_left)
    else:
        tt[0], TT[0] = t217_up, T_left
    if tt[-1] < t_peak - 1e-12:
        tt = np.append(tt, t_peak)
        TT = np.append(TT, T_right)
    else:
        tt[-1], TT[-1] = t_peak, T_right
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(TT - T217, tt))


@dataclass
class EvalResult:
    y: np.ndarray
    z: np.ndarray
    A: float
    V: float
    feasible: bool
    metrics: dict
    constraints: np.ndarray
    margins: dict
    t: np.ndarray | None = None
    T: np.ndarray | None = None


@dataclass
class EvalCounter:
    n: int = 0


def evaluate_y(
    y: np.ndarray,
    plate: q1.PlateParams,
    counter: EvalCounter | None = None,
    dt: float = 0.5,
    keep_curve: bool = False,
) -> EvalResult:
    y = np.asarray(y, dtype=float)
    z = encode(y)
    S = expand_setpoints(y)
    v = float(y[4])
    t, T = simulate_plate_fast(S, v, plate, dt=dt)
    if counter is not None:
        counter.n += 1

    i_peak = int(np.argmax(T))
    t_peak = float(t[i_peak])
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

    t150 = _cross_time(t, T, 150.0, True)
    t190 = _cross_time(t, T, 190.0, True)
    if np.isfinite(t150) and np.isfinite(t190) and t190 > t150 and t190 <= t_peak + 1e-9:
        tau_150_190 = float(t190 - t150)
    else:
        tau_150_190 = float("nan")

    t217_up = _cross_time(t, T, 217.0, True)
    t217_dn = _cross_time(t, T, 217.0, False)
    if np.isfinite(t217_up) and np.isfinite(t217_dn) and t217_dn > t217_up:
        tau_217 = float(t217_dn - t217_up)
    else:
        tau_217 = float("nan")

    A = shadow_area(t, T, t217_up, t_peak)

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
    V = float(np.sum(np.maximum(0.0, c / CONSTRAINT_SCALE)))
    feasible = bool(np.all(c <= 0.0) and np.isfinite(A))
    if not np.isfinite(A):
        A = 1e6
        feasible = False
        V = max(V, 1e3)

    metrics = {
        "T_peak": T_peak,
        "t_peak": t_peak,
        "t217_up": float(t217_up) if np.isfinite(t217_up) else None,
        "r_up_max": r_up_max,
        "r_up_min": r_up_min,
        "r_dn_min": r_dn_min,
        "r_dn_max": r_dn_max,
        "tau_150_190": tau_150_190,
        "tau_217": tau_217,
        "A": float(A) if feasible or np.isfinite(A) else None,
    }
    return EvalResult(
        y=y.copy(),
        z=z,
        A=float(A),
        V=V,
        feasible=feasible,
        metrics=metrics,
        constraints=c,
        margins=margins,
        t=t if keep_curve else None,
        T=T if keep_curve else None,
    )


def evaluate_z(
    z: np.ndarray,
    plate: q1.PlateParams,
    counter: EvalCounter | None = None,
    dt: float = 0.5,
    keep_curve: bool = False,
) -> EvalResult:
    return evaluate_y(decode(z), plate, counter=counter, dt=dt, keep_curve=keep_curve)


def deb_better(a: EvalResult, b: EvalResult) -> bool:
    """True 表示 a 优于 b（Deb 可行性规则）。"""
    if a.feasible and not b.feasible:
        return True
    if (not a.feasible) and b.feasible:
        return False
    if a.feasible and b.feasible:
        return a.A < b.A
    return a.V < b.V


def init_population(npop: int, rng: np.random.Generator, include_q2_seed: bool = True) -> np.ndarray:
    sampler = LatinHypercube(d=DIM, seed=int(rng.integers(0, 2**31 - 1)))
    pop = sampler.random(n=npop)
    if include_q2_seed and npop > 0:
        pop[0] = encode(SEED_Q2)
    return pop


@dataclass
class SearchHistory:
    evals: list[int] = field(default_factory=list)
    best_A: list[float] = field(default_factory=list)
    best_V: list[float] = field(default_factory=list)
    best_feasible: list[bool] = field(default_factory=list)


def _record(hist: SearchHistory, counter: EvalCounter, best: EvalResult) -> None:
    hist.evals.append(counter.n)
    hist.best_A.append(best.A if best.feasible else float("nan"))
    hist.best_V.append(best.V)
    hist.best_feasible.append(best.feasible)


def run_cde(
    plate: q1.PlateParams,
    npop: int,
    max_eval: int,
    dt: float,
    rng: np.random.Generator,
    F: float = 0.7,
    CR: float = 0.9,
) -> tuple[EvalResult, list[EvalResult], SearchHistory]:
    counter = EvalCounter()
    hist = SearchHistory()
    pop_z = init_population(npop, rng)
    pop = [evaluate_z(z, plate, counter=counter, dt=dt) for z in pop_z]
    best = pop[0]
    for ind in pop[1:]:
        if deb_better(ind, best):
            best = ind
    _record(hist, counter, best)

    while counter.n < max_eval:
        for i in range(npop):
            if counter.n >= max_eval:
                break
            idxs = list(range(npop))
            idxs.remove(i)
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
            mutant = pop_z[r1] + F * (pop_z[r2] - pop_z[r3])
            mutant = reflect_bounds(mutant)
            jrand = int(rng.integers(0, DIM))
            trial = pop_z[i].copy()
            for j in range(DIM):
                if rng.random() <= CR or j == jrand:
                    trial[j] = mutant[j]
            trial = reflect_bounds(trial)
            child = evaluate_z(trial, plate, counter=counter, dt=dt)
            if deb_better(child, pop[i]):
                pop[i] = child
                pop_z[i] = trial
                if deb_better(child, best):
                    best = child
            _record(hist, counter, best)
    return best, pop, hist


def _tournament(pop: list[EvalResult], rng: np.random.Generator, k: int = 2) -> EvalResult:
    cand = [pop[i] for i in rng.choice(len(pop), size=k, replace=False)]
    win = cand[0]
    for c in cand[1:]:
        if deb_better(c, win):
            win = c
    return win


def _sbx(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, eta: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    c1, c2 = p1.copy(), p2.copy()
    for j in range(DIM):
        if rng.random() > 0.5:
            continue
        u = rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        c1[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
        c2[j] = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
    return reflect_bounds(c1), reflect_bounds(c2)


def _poly_mutate(z: np.ndarray, rng: np.random.Generator, pm: float = 0.2, eta: float = 20.0) -> np.ndarray:
    z = z.copy()
    for j in range(DIM):
        if rng.random() > pm:
            continue
        u = rng.random()
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
        z[j] = z[j] + delta
    return reflect_bounds(z)


def run_ga(
    plate: q1.PlateParams,
    npop: int,
    max_eval: int,
    dt: float,
    rng: np.random.Generator,
    pc: float = 0.9,
    pm: float = 0.2,
) -> tuple[EvalResult, list[EvalResult], SearchHistory]:
    counter = EvalCounter()
    hist = SearchHistory()
    pop_z = init_population(npop, rng)
    pop = [evaluate_z(z, plate, counter=counter, dt=dt) for z in pop_z]
    best = min(pop, key=lambda e: (0 if e.feasible else 1, e.A if e.feasible else e.V))
    best = pop[0]
    for ind in pop[1:]:
        if deb_better(ind, best):
            best = ind
    _record(hist, counter, best)

    while counter.n < max_eval:
        new_z: list[np.ndarray] = []
        new_pop: list[EvalResult] = []
        elite_idx = 0
        for i in range(1, npop):
            if deb_better(pop[i], pop[elite_idx]):
                elite_idx = i
        new_z.append(pop[elite_idx].z.copy())
        new_pop.append(pop[elite_idx])

        while len(new_pop) < npop and counter.n < max_eval:
            p1 = _tournament(pop, rng)
            p2 = _tournament(pop, rng)
            if rng.random() < pc:
                c1z, c2z = _sbx(p1.z, p2.z, rng)
            else:
                c1z, c2z = p1.z.copy(), p2.z.copy()
            c1z = _poly_mutate(c1z, rng, pm=pm)
            child = evaluate_z(c1z, plate, counter=counter, dt=dt)
            new_z.append(c1z)
            new_pop.append(child)
            if deb_better(child, best):
                best = child
            _record(hist, counter, best)
            if len(new_pop) >= npop or counter.n >= max_eval:
                break
            c2z = _poly_mutate(c2z, rng, pm=pm)
            child2 = evaluate_z(c2z, plate, counter=counter, dt=dt)
            new_z.append(c2z)
            new_pop.append(child2)
            if deb_better(child2, best):
                best = child2
            _record(hist, counter, best)

        while len(new_pop) < npop:
            new_pop.append(pop[len(new_pop) % len(pop)])
            new_z.append(new_pop[-1].z.copy())
        pop = new_pop[:npop]
    return best, pop, hist


def run_pso(
    plate: q1.PlateParams,
    npop: int,
    max_eval: int,
    dt: float,
    rng: np.random.Generator,
    c1: float = 2.05,
    c2: float = 2.05,
    chi: float = 0.729,
) -> tuple[EvalResult, list[EvalResult], SearchHistory]:
    counter = EvalCounter()
    hist = SearchHistory()
    pos = init_population(npop, rng)
    vel = rng.uniform(-0.2, 0.2, size=(npop, DIM))
    pop = [evaluate_z(z, plate, counter=counter, dt=dt) for z in pos]
    pbest_z = pos.copy()
    pbest = list(pop)
    gbest = pop[0]
    gbest_z = pos[0].copy()
    for i in range(1, npop):
        if deb_better(pop[i], gbest):
            gbest = pop[i]
            gbest_z = pos[i].copy()
    _record(hist, counter, gbest)

    while counter.n < max_eval:
        for i in range(npop):
            if counter.n >= max_eval:
                break
            r1 = rng.random(DIM)
            r2 = rng.random(DIM)
            vel[i] = chi * (
                vel[i]
                + c1 * r1 * (pbest_z[i] - pos[i])
                + c2 * r2 * (gbest_z - pos[i])
            )
            pos[i] = reflect_bounds(pos[i] + vel[i])
            cur = evaluate_z(pos[i], plate, counter=counter, dt=dt)
            pop[i] = cur
            if deb_better(cur, pbest[i]):
                pbest[i] = cur
                pbest_z[i] = pos[i].copy()
            if deb_better(cur, gbest):
                gbest = cur
                gbest_z = pos[i].copy()
            _record(hist, counter, gbest)
    return gbest, pbest, hist


def select_elites(pop: list[EvalResult], n: int = 3, min_dist: float = 0.05) -> list[EvalResult]:
    feas = [e for e in pop if e.feasible]
    feas.sort(key=lambda e: e.A)
    elites: list[EvalResult] = []
    for e in feas:
        if all(np.linalg.norm(e.z - o.z) >= min_dist for o in elites):
            elites.append(e)
        if len(elites) >= n:
            break
    if len(elites) < n:
        for e in feas:
            if all(e is not chosen for chosen in elites):
                elites.append(e)
            if len(elites) >= n:
                break
    return elites


def refine_cobyla(
    elite: EvalResult,
    plate: q1.PlateParams,
    dt: float,
    maxfun: int = 80,
) -> EvalResult:
    cache: dict[bytes, EvalResult] = {}
    counter = EvalCounter()

    def eval_cached(z: np.ndarray) -> EvalResult:
        key = np.asarray(z, dtype=float).tobytes()
        if key not in cache:
            cache[key] = evaluate_z(z, plate, counter=counter, dt=dt)
        return cache[key]

    def objective(z):
        ev = eval_cached(z)
        return ev.A if ev.feasible else ev.A + 1e3 * (1.0 + ev.V)

    cons = []
    for j in range(10):

        def make_fun(jj):
            def fun(z, j=jj):
                ev = eval_cached(z)
                return -ev.constraints[j]  # >= 0

            return fun

        cons.append({"type": "ineq", "fun": make_fun(j)})

    for j in range(DIM):

        def lo(z, j=j):
            return float(z[j])

        def hi(z, j=j):
            return float(1.0 - z[j])

        cons.append({"type": "ineq", "fun": lo})
        cons.append({"type": "ineq", "fun": hi})

    z0 = elite.z.copy()
    try:
        res = minimize(
            objective,
            z0,
            method="COBYLA",
            constraints=cons,
            options={"maxiter": maxfun, "rhobeg": 0.05, "tol": 1e-6},
        )
        cand = eval_cached(res.x)
    except Exception:
        return elite

    if cand.feasible and (not elite.feasible or cand.A < elite.A):
        return cand
    return elite


def multi_start_cobyla(
    pop: list[EvalResult],
    plate: q1.PlateParams,
    dt: float,
    n_elites: int = 3,
    maxfun: int = 80,
) -> EvalResult:
    elites = select_elites(pop, n=n_elites)
    if not elites:
        worst = sorted(pop, key=lambda e: e.V)[:n_elites]
        elites = worst
    best = elites[0]
    for e in elites:
        refined = refine_cobyla(e, plate, dt=dt, maxfun=maxfun)
        if deb_better(refined, best):
            best = refined
    return best


def load_plate() -> q1.PlateParams:
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
            eta_cool_late=p.get("eta_cool_late", p["eta_cool"]),
            eta_r=p.get("eta_r", 0.0),
            alpha=p.get("alpha", q1.ALPHA_FIXED),
        )
    print("未找到 q1 summary，现场标定...")
    t_obs, y_obs = q1.load_calibration_data()
    plate, metrics = q1.fit_plate(t_obs, y_obs, fit_radiation=False)
    print(f"标定完成 RMSE={metrics['rmse']:.4f} °C")
    return plate


def result_to_dict(ev: EvalResult) -> dict:
    return {
        "y": {n: float(v) for n, v in zip(VAR_NAMES, ev.y)},
        "setpoints": expand_setpoints(ev.y).tolist(),
        "A": float(ev.A),
        "V": float(ev.V),
        "feasible": bool(ev.feasible),
        "metrics": {
            k: (float(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else v)
            for k, v in ev.metrics.items()
        },
        "margins": {k: float(v) for k, v in ev.margins.items()},
        "at_bound": {
            n: bool(
                abs(ev.y[i] - L_BOUNDS[i]) < (0.05 if i < 4 else 0.05)
                or abs(ev.y[i] - U_BOUNDS[i]) < (0.05 if i < 4 else 0.05)
            )
            for i, n in enumerate(VAR_NAMES)
        },
    }


def plot_optimal_curve(ev: EvalResult, path: Path) -> None:
    assert ev.t is not None and ev.T is not None
    t, T = ev.t, ev.T
    t217 = ev.metrics.get("t217_up")
    tp = ev.metrics["t_peak"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, T, color="#1f4e79", lw=1.8, label="最优炉温曲线")
    ax.axhline(217, color="#c0392b", ls="--", lw=1, label="217 °C")
    if t217 is not None and np.isfinite(t217):
        mask = (t >= t217) & (t <= tp)
        ax.fill_between(t[mask], 217, T[mask], where=T[mask] >= 217, color="#f4a261", alpha=0.55, label=f"阴影面积 A={ev.A:.2f}")
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("温度 T / °C")
    ax.set_title(
        f"问题三最优曲线  S={ev.y[0]:.1f}/{ev.y[1]:.1f}/{ev.y[2]:.1f}/{ev.y[3]:.1f}, "
        f"v={ev.y[4]:.2f} cm/min"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_margins(ev: EvalResult, path: Path) -> None:
    names = list(ev.margins.keys())
    vals = np.array([ev.margins[k] / s for k, s in zip(names, CONSTRAINT_SCALE)])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("归一化余量 margin / scale")
    ax.set_title("最优解约束余量")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(histories: dict[str, SearchHistory], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, h in histories.items():
        A = np.array(h.best_A, dtype=float)
        ev = np.array(h.evals)
        mask = np.isfinite(A)
        if mask.any():
            ax.plot(ev[mask], A[mask], label=name, lw=1.6)
    ax.set_xlabel("累计仿真次数")
    ax.set_ylabel("当前最优可行面积 A / (°C·s)")
    ax.set_title("全局搜索收敛曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def neighborhood_check(ev: EvalResult, plate: q1.PlateParams, dt: float) -> list[dict]:
    """最优点邻域单变量扰动。"""
    deltas = [1.0, 1.0, 1.0, 1.0, 0.5]  # °C / cm/min
    rows = []
    for j, name in enumerate(VAR_NAMES):
        for sign in (-1.0, 1.0):
            y2 = ev.y.copy()
            y2[j] = float(np.clip(y2[j] + sign * deltas[j], L_BOUNDS[j], U_BOUNDS[j]))
            if abs(y2[j] - ev.y[j]) < 1e-12:
                continue
            e2 = evaluate_y(y2, plate, dt=dt)
            rows.append(
                {
                    "var": name,
                    "delta": sign * deltas[j],
                    "y_j": float(y2[j]),
                    "A": float(e2.A),
                    "feasible": bool(e2.feasible),
                    "V": float(e2.V),
                    "dA": float(e2.A - ev.A) if e2.feasible and ev.feasible else None,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="问题三：最小阴影面积优化")
    p.add_argument("--fast", action="store_true", help="快速冒烟（小种群/少评价）")
    p.add_argument("--full", action="store_true", help="接近笔记预算（较慢）")
    p.add_argument("--no-compare", action="store_true", help="跳过 GA/PSO 对照")
    p.add_argument("--seed", type=int, default=2020)
    return p.parse_args()


def budget_from_args(args: argparse.Namespace) -> dict:
    if args.fast:
        return dict(
            npop=20, max_eval=400, dt_search=0.5, dt_refine=0.1,
            dt_verify=0.05, elites=2, cobyla_maxfun=40,
        )
    if args.full:
        return dict(
            npop=75, max_eval=8000, dt_search=0.1, dt_refine=0.05,
            dt_verify=0.025, elites=5, cobyla_maxfun=100,
        )
    return dict(
        npop=36, max_eval=1500, dt_search=0.1, dt_refine=0.05,
        dt_verify=0.025, elites=5, cobyla_maxfun=80,
    )


def main() -> None:
    args = parse_args()
    cfg = budget_from_args(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("问题三：最小阴影面积炉温曲线")
    print(f"种群={cfg['npop']}, 全局预算={cfg['max_eval']}, dt_search={cfg['dt_search']}")
    print("=" * 60)

    plate = load_plate()
    print(
        f"参数: eta_pre={plate.eta_pre:.4f}, soak={plate.eta_soak:.4f}, "
        f"ref={plate.eta_ref:.4f}, cool10={plate.eta_cool:.4f}, "
        f"cool11={plate.eta_cool_late:.4f}"
    )

    rng = np.random.default_rng(args.seed)
    histories: dict[str, SearchHistory] = {}
    algo_rows = []

    print("\n[1] 约束差分进化 CDE ...")
    t0 = time.perf_counter()
    best_cde, pop_cde, hist_cde = run_cde(
        plate, cfg["npop"], cfg["max_eval"], cfg["dt_search"], rng
    )
    print(
        f"  CDE 完成: eval≈{hist_cde.evals[-1]}, "
        f"feas={best_cde.feasible}, A={best_cde.A:.4f}, V={best_cde.V:.4f}, "
        f"用时 {time.perf_counter()-t0:.1f}s"
    )
    histories["CDE"] = hist_cde

    print("[2] 多起点 COBYLA 精修 (CDE 精英) ...")
    best_cde_loc = multi_start_cobyla(
        pop_cde, plate, dt=cfg["dt_search"], n_elites=cfg["elites"], maxfun=cfg["cobyla_maxfun"]
    )
    if deb_better(best_cde_loc, best_cde):
        best_cde = best_cde_loc
    print(f"  精修后: feas={best_cde.feasible}, A={best_cde.A:.4f}, y={best_cde.y}")

    algo_rows.append(
        {
            "algo": "CDE+COBYLA",
            "feasible": best_cde.feasible,
            "A": best_cde.A,
            "V": best_cde.V,
            **{n: float(v) for n, v in zip(VAR_NAMES, best_cde.y)},
        }
    )

    if not args.no_compare:
        print("\n[3] GA 对照（同预算）...")
        rng_ga = np.random.default_rng(args.seed + 1)
        t0 = time.perf_counter()
        best_ga, pop_ga, hist_ga = run_ga(
            plate, cfg["npop"], cfg["max_eval"], cfg["dt_search"], rng_ga
        )
        refined_ga = multi_start_cobyla(
            pop_ga, plate, dt=cfg["dt_search"], n_elites=cfg["elites"], maxfun=cfg["cobyla_maxfun"]
        )
        if deb_better(refined_ga, best_ga):
            best_ga = refined_ga
        histories["GA"] = hist_ga
        print(
            f"  GA: feas={best_ga.feasible}, A={best_ga.A:.4f}, "
            f"用时 {time.perf_counter()-t0:.1f}s"
        )
        algo_rows.append(
            {
                "algo": "GA+COBYLA",
                "feasible": best_ga.feasible,
                "A": best_ga.A,
                "V": best_ga.V,
                **{n: float(v) for n, v in zip(VAR_NAMES, best_ga.y)},
            }
        )

        print("[4] PSO 对照（同预算）...")
        rng_pso = np.random.default_rng(args.seed + 2)
        t0 = time.perf_counter()
        best_pso, pop_pso, hist_pso = run_pso(
            plate, cfg["npop"], cfg["max_eval"], cfg["dt_search"], rng_pso
        )
        refined_pso = multi_start_cobyla(
            pop_pso, plate, dt=cfg["dt_search"], n_elites=cfg["elites"], maxfun=cfg["cobyla_maxfun"]
        )
        if deb_better(refined_pso, best_pso):
            best_pso = refined_pso
        histories["PSO"] = hist_pso
        print(
            f"  PSO: feas={best_pso.feasible}, A={best_pso.A:.4f}, "
            f"用时 {time.perf_counter()-t0:.1f}s"
        )
        algo_rows.append(
            {
                "algo": "PSO+COBYLA",
                "feasible": best_pso.feasible,
                "A": best_pso.A,
                "V": best_pso.V,
                **{n: float(v) for n, v in zip(VAR_NAMES, best_pso.y)},
            }
        )

    candidates = [best_cde]
    if not args.no_compare:
        candidates.extend([best_ga, best_pso])
    best = candidates[0]
    for c in candidates[1:]:
        if deb_better(c, best):
            best = c

    coarse_pool = list(pop_cde) + candidates
    if not args.no_compare:
        coarse_pool.extend(pop_ga)
        coarse_pool.extend(pop_pso)

    print(f"\n[5] 中网格精修 (dt={cfg['dt_refine']}) ...")
    coarse_elites = select_elites(
        coarse_pool, n=max(cfg["elites"] * 2, 4), min_dist=0.02
    )
    mid_pool = [evaluate_y(e.y, plate, dt=cfg["dt_refine"]) for e in coarse_elites]
    mid_pool.extend(evaluate_y(c.y, plate, dt=cfg["dt_refine"]) for c in candidates)
    best_mid = multi_start_cobyla(
        mid_pool,
        plate,
        dt=cfg["dt_refine"],
        n_elites=cfg["elites"],
        maxfun=cfg["cobyla_maxfun"],
    )
    print(f"  中网格: feas={best_mid.feasible}, A={best_mid.A:.6f}")

    print(f"[6] 官方网格精修 (dt={cfg['dt_verify']}) ...")
    mid_elites = select_elites(
        mid_pool + [best_mid], n=max(cfg["elites"], 3), min_dist=0.01
    )
    hi_pool = [evaluate_y(e.y, plate, dt=cfg["dt_verify"]) for e in mid_elites]
    hi_pool.append(evaluate_y(best_mid.y, plate, dt=cfg["dt_verify"]))
    best_hi = multi_start_cobyla(
        hi_pool,
        plate,
        dt=cfg["dt_verify"],
        n_elites=cfg["elites"],
        maxfun=cfg["cobyla_maxfun"],
    )
    best_hi = evaluate_y(best_hi.y, plate, dt=cfg["dt_verify"], keep_curve=True)
    print(
        f"  官方网格: feas={best_hi.feasible}, A={best_hi.A:.6f}, "
        f"Tpeak={best_hi.metrics['T_peak']:.4f}, "
        f"tau150={best_hi.metrics['tau_150_190']}, tau217={best_hi.metrics['tau_217']}"
    )
    if not best_hi.feasible:
        raise RuntimeError("三级网格精修后仍未得到官方报告网格上的可行解")

    print("\n[7] 邻域检查 ...")
    neigh = neighborhood_check(best_hi, plate, dt=cfg["dt_verify"])
    pd.DataFrame(neigh).to_csv(OUT_DIR / "neighborhood.csv", index=False, encoding="utf-8-sig")

    assert best_hi.t is not None and best_hi.T is not None
    pd.DataFrame({"t": best_hi.t, "T": best_hi.T}).to_csv(
        OUT_DIR / "best_curve.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(algo_rows).to_csv(OUT_DIR / "algo_compare.csv", index=False, encoding="utf-8-sig")

    summary = {
        "budget": cfg,
        "seed": args.seed,
        "best": result_to_dict(best_hi),
        "algo_compare": algo_rows,
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
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_optimal_curve(best_hi, FIG_DIR / "optimal_curve.png")
    plot_margins(best_hi, FIG_DIR / "constraint_margins.png")
    if histories:
        plot_convergence(histories, FIG_DIR / "convergence.png")

    print("\n========== 问题三结果 ==========")
    print(f"S1-5 = {best_hi.y[0]:.4f} °C")
    print(f"S6   = {best_hi.y[1]:.4f} °C")
    print(f"S7   = {best_hi.y[2]:.4f} °C")
    print(f"S8-9 = {best_hi.y[3]:.4f} °C")
    print(f"v    = {best_hi.y[4]:.4f} cm/min")
    print(f"A*   = {best_hi.A:.6f} °C·s")
    print(f"可行 = {best_hi.feasible}")
    print(f"输出: {OUT_DIR}")
    print(f"图像: {FIG_DIR}")


if __name__ == "__main__":
    main()
