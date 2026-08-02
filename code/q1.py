"""问题一：标定传热参数并预测炉温曲线。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "q1"
FIG_DIR = ROOT / "figures" / "q1"

FRONT_LEN = 25.0
ZONE_LEN = 30.5
GAP_LEN = 5.0
N_ZONES = 11
FURNACE_LEN = FRONT_LEN + N_ZONES * ZONE_LEN + (N_ZONES - 1) * GAP_LEN + FRONT_LEN

THICKNESS_M = 1.5e-4
HALF_THICKNESS = THICKNESS_M / 2.0
ALPHA_FIXED = 1.5e-7
DT_REPORT = 0.025
DT_CALIB = 0.1

SETPOINTS_CAL = np.array([175, 175, 175, 175, 175, 195, 235, 255, 255, 25, 25], dtype=float)
SETPOINTS_Q1 = np.array([173, 173, 173, 173, 173, 198, 230, 257, 257, 25, 25], dtype=float)


def zone_interval(i: int) -> tuple[float, float]:
    """第 i 个小温区 [a_i, b_i]，i = 1..11，单位 cm。"""
    a = FRONT_LEN + (i - 1) * (ZONE_LEN + GAP_LEN)
    return a, a + ZONE_LEN


def zone_midpoint(i: int) -> float:
    a, b = zone_interval(i)
    return 0.5 * (a + b)


X_SOAK = zone_interval(6)[0]
X_REFLOW = zone_interval(7)[0]
X_COOL = zone_interval(9)[1]
X_COOL_LATE = zone_interval(10)[1]
FRONT_TRANSITION_END = zone_interval(1)[1]
COOL_GAP_TRANSITION_WIDTH = 0.5

Q1_PROBE_X = {
    "zone3_mid": zone_midpoint(3),
    "zone6_mid": zone_midpoint(6),
    "zone7_mid": zone_midpoint(7),
    "zone8_end": zone_interval(8)[1],
}


def _smoothstep01(s: float | np.ndarray) -> float | np.ndarray:
    s_arr = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    out = 3.0 * s_arr**2 - 2.0 * s_arr**3
    return float(out) if out.ndim == 0 else out


def ambient_temperature_linear(x: float | np.ndarray, setpoints: Sequence[float]) -> float | np.ndarray:
    """Piecewise-linear reference ambient field."""
    S = np.asarray(setpoints, dtype=float)
    if S.shape != (11,):
        raise ValueError("setpoints must have length 11")

    x_arr = np.asarray(x, dtype=float)
    scalar = x_arr.ndim == 0
    x_arr = np.atleast_1d(x_arr)
    C = np.empty_like(x_arr, dtype=float)

    for k, xk in enumerate(x_arr):
        if xk < 0:
            C[k] = 25.0
        elif xk < FRONT_LEN:
            C[k] = 25.0 + (S[0] - 25.0) / FRONT_LEN * xk
        elif xk > zone_interval(11)[1]:
            C[k] = 25.0
        else:
            placed = False
            for i in range(1, 12):
                a, b = zone_interval(i)
                if a <= xk <= b:
                    C[k] = S[i - 1]
                    placed = True
                    break
                if i < 11:
                    a_next, _ = zone_interval(i + 1)
                    if b < xk < a_next:
                        C[k] = S[i - 1] + (S[i] - S[i - 1]) / GAP_LEN * (xk - b)
                        placed = True
                        break
            if not placed:
                C[k] = 25.0

    return float(C[0]) if scalar else C


def ambient_temperature(x: float | np.ndarray, setpoints: Sequence[float]) -> float | np.ndarray:
    """Equivalent ambient field with smooth inlet and cooling transitions."""
    S = np.asarray(setpoints, dtype=float)
    if S.shape != (11,):
        raise ValueError("setpoints must have length 11")
    x_arr = np.asarray(x, dtype=float)
    scalar = x_arr.ndim == 0
    xx = np.atleast_1d(x_arr)
    C = np.asarray(ambient_temperature_linear(xx, S), dtype=float)

    m_front = (xx >= 0.0) & (xx < FRONT_TRANSITION_END)
    if np.any(m_front):
        s = xx[m_front] / FRONT_TRANSITION_END
        C[m_front] = 25.0 + (S[0] - 25.0) * np.asarray(_smoothstep01(s))

    x_hot = zone_interval(9)[1]
    x_cold = zone_interval(10)[0]
    x_start = x_cold - COOL_GAP_TRANSITION_WIDTH
    m_hot = (xx > x_hot) & (xx <= x_start)
    C[m_hot] = S[8]
    m_drop = (xx > x_start) & (xx < x_cold)
    if np.any(m_drop):
        s = (xx[m_drop] - x_start) / COOL_GAP_TRANSITION_WIDTH
        H = np.asarray(_smoothstep01(s))
        C[m_drop] = S[8] + (S[9] - S[8]) * H

    return float(C[0]) if scalar else C


def segment_of(x: float) -> str:
    """返回四个基础区段；第二冷却阶段由参数类在 X_COOL_LATE 处分流。"""
    if x >= X_COOL:
        return "cool"
    if x >= X_REFLOW:
        return "ref"
    if x >= X_SOAK:
        return "soak"
    return "pre"


def thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = diag.size
    a, b, c, d = lower.copy(), diag.copy(), upper.copy(), rhs.copy()
    for i in range(1, n):
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.empty(n, dtype=float)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


@dataclass
class LumpedParams:
    k_pre: float
    k_soak: float
    k_ref: float
    k_cool: float
    k_cool_late: float | None = None

    def k_at(self, x: float) -> float:
        seg = segment_of(x)
        if x >= X_COOL_LATE:
            return self.k_cool if self.k_cool_late is None else self.k_cool_late
        return {"cool": self.k_cool, "ref": self.k_ref, "soak": self.k_soak}.get(seg, self.k_pre)


@dataclass
class PlateParams:
    eta_pre: float
    eta_soak: float
    eta_ref: float
    eta_cool: float
    eta_cool_late: float | None = None
    eta_r: float = 0.0
    alpha: float = ALPHA_FIXED

    def eta_at(self, x: float) -> float:
        if x >= X_COOL_LATE:
            return self.eta_cool if self.eta_cool_late is None else self.eta_cool_late
        seg = segment_of(x)
        return {
            "cool": self.eta_cool,
            "ref": self.eta_ref,
            "soak": self.eta_soak,
        }.get(seg, self.eta_pre)


def _as_setpoints(setpoints: ArrayLike) -> np.ndarray:
    S = np.asarray(setpoints, dtype=float)
    if S.shape != (11,):
        raise ValueError("setpoints must have length 11")
    return S


def simulate_lumped(
    setpoints: ArrayLike,
    u_cm_per_min: float,
    params: LumpedParams,
    t_end: float | None = None,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the lumped model by fourth-order Runge--Kutta."""
    S = _as_setpoints(setpoints)
    v = u_cm_per_min / 60.0
    if t_end is None:
        t_end = FURNACE_LEN / v
    n_steps = int(np.ceil(t_end / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    T = np.empty_like(t)
    T[0] = 25.0

    def rhs(tk: float, Tk: float) -> float:
        xk = v * tk
        return params.k_at(xk) * (float(ambient_temperature(xk, S)) - Tk)

    for n in range(n_steps):
        tn, Tn = t[n], T[n]
        k1 = rhs(tn, Tn)
        k2 = rhs(tn + 0.5 * dt, Tn + 0.5 * dt * k1)
        k3 = rhs(tn + 0.5 * dt, Tn + 0.5 * dt * k2)
        k4 = rhs(tn + dt, Tn + dt * k3)
        T[n + 1] = Tn + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, T


def simulate_plate(
    setpoints: ArrayLike,
    u_cm_per_min: float,
    params: PlateParams,
    t_end: float | None = None,
    dt: float = 0.1,
    n_nodes: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the half-thickness implicit heat-conduction model."""
    S = _as_setpoints(setpoints)
    v = u_cm_per_min / 60.0
    if t_end is None:
        t_end = FURNACE_LEN / v

    M = n_nodes - 1
    dz = HALF_THICKNESS / M
    r = params.alpha * dt / dz**2
    n_steps = int(np.ceil(t_end / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    T_center = np.empty(n_steps + 1)
    u = np.full(n_nodes, 25.0)
    T_center[0] = 25.0

    lower = np.zeros(n_nodes - 1)
    diag = np.zeros(n_nodes)
    upper = np.zeros(n_nodes - 1)

    for n in range(n_steps):
        x_next = v * t[n + 1]
        C = float(ambient_temperature(x_next, S))
        Ck = C + 273.15
        Ts = u[-1]
        Tsk = Ts + 273.15
        Rcoef = (Ck + Tsk) * (Ck**2 + Tsk**2)
        eta = params.eta_at(x_next)
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
            u_new = thomas_solve(lower, diag, upper, rhs)
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

        u = thomas_solve(lower, diag, upper, rhs)
        T_center[n + 1] = u[0]
    return t, T_center


def interpolate_temperature(t: np.ndarray, T: np.ndarray, t_query: ArrayLike) -> np.ndarray:
    return np.interp(np.asarray(t_query, dtype=float), t, T)


def load_calibration_data() -> tuple[np.ndarray, np.ndarray]:
    path = next(ROOT.glob("*.xlsx"))
    df = pd.read_excel(path)
    cols = list(df.columns)
    t = df[cols[0]].to_numpy(dtype=float)
    y = df[cols[1]].to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    return t[mask], y[mask]


def fit_lumped(t_obs: np.ndarray, y_obs: np.ndarray, u: float = 70.0) -> tuple[LumpedParams, dict]:
    t_end = max(float(t_obs[-1]) + 5.0, 400.0)

    def residuals(theta: np.ndarray) -> np.ndarray:
        p = LumpedParams(*map(float, theta))
        t, T = simulate_lumped(SETPOINTS_CAL, u, p, t_end=t_end, dt=0.1)
        return interpolate_temperature(t, T, t_obs) - y_obs

    starts = [
        np.array([0.020, 0.016, 0.023, 0.0034, 0.0100]),
        np.array([0.015, 0.020, 0.030, 0.0020, 0.0150]),
        np.array([0.025, 0.015, 0.020, 0.0060, 0.0080]),
    ]
    bounds = ([1e-4] * 5, [0.5] * 5)
    best = None
    for x0 in starts:
        res = least_squares(residuals, x0, bounds=bounds, method="trf", verbose=0)
        cost = float(np.sum(res.fun**2))
        if best is None or cost < best[0]:
            best = (cost, res)
    res = best[1]
    params = LumpedParams(*map(float, res.x))
    err = residuals(res.x)
    return params, {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "success": bool(res.success),
    }


def fit_plate(
    t_obs: np.ndarray, y_obs: np.ndarray, u: float = 70.0, fit_radiation: bool = True
) -> tuple[PlateParams, dict]:
    t_end = max(float(t_obs[-1]) + 5.0, 400.0)

    def pack(theta: np.ndarray) -> PlateParams:
        if fit_radiation:
            return PlateParams(*map(float, theta[:5]), eta_r=float(theta[5]))
        return PlateParams(*map(float, theta[:5]), eta_r=0.0)

    def residuals(theta: np.ndarray) -> np.ndarray:
        t, T = simulate_plate(SETPOINTS_CAL, u, pack(theta), t_end=t_end, dt=0.1)
        return interpolate_temperature(t, T, t_obs) - y_obs

    if fit_radiation:
        starts = [
            np.array([10.0, 8.0, 12.0, 1.7, 5.0, 1e-12]),
            np.array([7.0, 10.0, 18.0, 1.0, 8.0, 1e-12]),
            np.array([13.0, 7.0, 10.0, 3.0, 4.0, 1e-11]),
        ]
        bounds = (
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
            [200.0, 200.0, 400.0, 100.0, 100.0, 5e-7],
        )
    else:
        starts = [
            np.array([10.0, 8.0, 12.0, 1.7, 5.0]),
            np.array([7.0, 10.0, 18.0, 1.0, 8.0]),
            np.array([13.0, 7.0, 10.0, 3.0, 4.0]),
        ]
        bounds = ([0.1] * 5, [200.0, 200.0, 400.0, 100.0, 100.0])

    best = None
    for x0 in starts:
        res = least_squares(residuals, x0, bounds=bounds, method="trf", verbose=0)
        cost = float(np.sum(res.fun**2))
        if best is None or cost < best[0]:
            best = (cost, res)
    res = best[1]
    params = pack(res.x)
    err = residuals(res.x)
    return params, {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "success": bool(res.success),
    }


def process_metrics(t: np.ndarray, T: np.ndarray) -> dict:
    dTdt = np.gradient(T, t)
    i_peak = int(np.argmax(T))
    t_peak, T_peak = float(t[i_peak]), float(T[i_peak])
    rising = t <= t_peak
    in_band = rising & (T >= 150.0) & (T <= 190.0)
    dur_150_190 = float(t[in_band][-1] - t[in_band][0]) if np.any(in_band) else float("nan")

    def cross_time(level: float, rising_side: bool) -> float:
        if rising_side:
            idx = np.where((T[:-1] < level) & (T[1:] >= level))[0]
        else:
            idx = np.where((T[:-1] >= level) & (T[1:] < level))[0]
        if idx.size == 0:
            return float("nan")
        i = int(idx[0 if rising_side else -1])
        return float(t[i] + (level - T[i]) / (T[i + 1] - T[i] + 1e-30) * (t[i + 1] - t[i]))

    if np.any(T >= 217.0):
        t_up, t_down = cross_time(217.0, True), cross_time(217.0, False)
        dur_217 = float(t_down - t_up) if np.isfinite(t_up) and np.isfinite(t_down) else float("nan")
    else:
        dur_217 = float("nan")

    return {
        "T_peak": T_peak,
        "t_peak": t_peak,
        "max_heat_slope": float(np.max(dTdt)),
        "min_cool_slope": float(np.min(dTdt)),
        "dur_150_190": dur_150_190,
        "dur_above_217": dur_217,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("第一步：用附件实验工况标定参数")
    print("  温区: 175/195/235/255/25 °C, 速度: 70 cm/min")
    print("=" * 60)
    t_obs, y_obs = load_calibration_data()
    print(f"观测点数: {len(t_obs)}, t ∈ [{t_obs[0]:.1f}, {t_obs[-1]:.1f}] s")

    print("\n拟合 M1（边界感知温度场 + 五段换热）...")
    lumped, m1 = fit_lumped(t_obs, y_obs)
    print(
        f"  k_pre={lumped.k_pre:.6f}, k_soak={lumped.k_soak:.6f}, "
        f"k_ref={lumped.k_ref:.6f}, k_cool10={lumped.k_cool:.6f}, "
        f"k_cool11={lumped.k_cool_late:.6f}"
    )
    print(f"  RMSE={m1['rmse']:.4f} °C, MAE={m1['mae']:.4f} °C")

    print("\n拟合 M2（主模型，边界感知温度场 + 五段等效换热）...")
    plate, m2 = fit_plate(t_obs, y_obs, fit_radiation=False)
    print(
        f"  eta_pre={plate.eta_pre:.4f}, eta_soak={plate.eta_soak:.4f}, "
        f"eta_ref={plate.eta_ref:.4f}, eta_cool10={plate.eta_cool:.4f}, "
        f"eta_cool11={plate.eta_cool_late:.4f}"
    )
    print(f"  eta_r={plate.eta_r:.6e}")
    print(f"  RMSE={m2['rmse']:.4f} °C, MAE={m2['mae']:.4f} °C")

    t_cal_m1, T_cal_m1 = simulate_lumped(SETPOINTS_CAL, 70.0, lumped, t_end=380.0)
    t_cal_m2, T_cal_m2 = simulate_plate(SETPOINTS_CAL, 70.0, plate, t_end=380.0)
    pred_m1 = interpolate_temperature(t_cal_m1, T_cal_m1, t_obs)
    pred_m2 = interpolate_temperature(t_cal_m2, T_cal_m2, t_obs)
    max_diff = float(np.max(np.abs(pred_m1 - pred_m2)))
    print(f"\n标定工况 M1 vs M2 最大温差: {max_diff:.4f} °C")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_obs, y_obs, "k.", ms=2, label="Measured")
    axes[0].plot(t_cal_m1, T_cal_m1, label="M1")
    axes[0].plot(t_cal_m2, T_cal_m2, "--", label="M2")
    axes[0].set_ylabel("T / degC")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Calibration (u=70 cm/min)")
    axes[1].plot(t_obs, pred_m1 - y_obs, label="M1 residual")
    axes[1].plot(t_obs, pred_m2 - y_obs, label="M2 residual")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("t / s")
    axes[1].set_ylabel("residual / degC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_fit.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("第二步：冻结标定参数，代入问题一工况预测")
    print("  温区: 173/198/230/257/25 °C, 速度: 78 cm/min")
    print("=" * 60)
    u_q1 = 78.0
    t_end_q1 = FURNACE_LEN / (u_q1 / 60.0)
    t_q1, T_q1 = simulate_plate(SETPOINTS_Q1, u_q1, plate, t_end=t_end_q1, dt=DT_REPORT)
    t_q1_m1, T_q1_m1 = simulate_lumped(SETPOINTS_Q1, u_q1, lumped, t_end=t_end_q1, dt=DT_REPORT)

    probe_rows = []
    print("\n指定位置中心温度（M2）:")
    for name, x in Q1_PROBE_X.items():
        t_arr = x / (u_q1 / 60.0)
        temp = float(interpolate_temperature(t_q1, T_q1, t_arr))
        temp_m1 = float(interpolate_temperature(t_q1_m1, T_q1_m1, t_arr))
        print(f"  {name}: x={x:.2f} cm, t={t_arr:.4f} s, T={temp:.2f} °C (M1={temp_m1:.2f})")
        probe_rows.append({"name": name, "x_cm": x, "t_s": t_arr, "T_M2": temp, "T_M1": temp_m1})

    t_out = np.arange(0.0, t_end_q1 + 1e-9, 0.5)
    T_out = interpolate_temperature(t_q1, T_q1, t_out)
    result_df = pd.DataFrame({"时间(s)": t_out, "温度(摄氏度)": np.round(T_out, 4)})
    result_df.to_csv(OUT_DIR / "result.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(ROOT / "result_q1.csv", index=False, encoding="utf-8-sig")
    try:
        result_df.to_csv(ROOT / "result.csv", index=False, encoding="utf-8-sig")
        print(f"\n已写入 result.csv / result_q1.csv / results/q1/result.csv ，共 {len(t_out)} 行")
    except PermissionError:
        print(f"\nresult.csv 被占用，已写入 result_q1.csv 与 results/q1/result.csv（共 {len(t_out)} 行）")
        print("请关闭 Excel 后手动复制 results/q1/result.csv → 根目录 result.csv")

    metrics_q1 = process_metrics(t_q1, T_q1)
    print("\n问题一工艺特征（M2）:")
    for k, v in metrics_q1.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_q1, T_q1, label="M2")
    ax.plot(t_q1_m1, T_q1_m1, "--", label="M1")
    for row in probe_rows:
        ax.axvline(row["t_s"], color="gray", lw=0.8, alpha=0.5)
        ax.plot(row["t_s"], row["T_M2"], "ro", ms=5)
    ax.set_xlabel("t / s")
    ax.set_ylabel("center T / degC")
    ax.set_title("Problem 1 (u=78 cm/min)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q1_profile.png", dpi=150)
    plt.close(fig)

    summary = {
        "official_grid": {
            "dt_report": DT_REPORT,
            "dt_calib": DT_CALIB,
            "note": "预测与 result.csv 使用 dt_report；标定拟合可用 dt_calib",
        },
        "lumped_params": {
            "k_pre": lumped.k_pre,
            "k_soak": lumped.k_soak,
            "k_ref": lumped.k_ref,
            "k_cool": lumped.k_cool,
            "k_cool_late": lumped.k_cool_late,
        },
        "plate_params": {
            "eta_pre": plate.eta_pre,
            "eta_soak": plate.eta_soak,
            "eta_ref": plate.eta_ref,
            "eta_cool": plate.eta_cool,
            "eta_cool_late": plate.eta_cool_late,
            "eta_r": plate.eta_r,
            "alpha": plate.alpha,
        },
        "calibration": {"M1": m1, "M2": m2, "max_diff_M1_M2": max_diff},
        "probes": probe_rows,
        "q1_metrics": metrics_q1,
        "result_rows": int(len(t_out)),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(probe_rows).to_csv(OUT_DIR / "probe_temperatures.csv", index=False)
    print(f"\n摘要: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
