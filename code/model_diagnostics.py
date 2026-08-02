"""问题一环境温度场的诊断性模型比较。

该脚本不覆盖正式结果，只比较三个集中参数候选：

M0  现有炉前线性温度场 + 单一冷却换热系数；
M1  炉前带环境散热的双曲温度场 + 单一冷却换热系数；
M2  炉前双曲温度场 + 冷却前/后两阶段换热系数。

M2 保持小温区 10--11 的设定环境温度为 25 °C，不用长距离高温
指数尾巴替代冷却区设定值。脚本用于判断结构误差来自哪里，并检查新增
参数是否可辨识；在模型定稿前不会自动重跑问题二至四。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q1


DT_FIT = 0.1
COOL_SPLIT_X = q1.zone_interval(11)[0]


@dataclass(frozen=True)
class Candidate:
    name: str
    split_cooling: bool
    lossy_front: bool
    delayed_cooling_gap: bool = False
    split_front_exchange: bool = False


CANDIDATES = (
    Candidate("M0_linear_front_one_cool", split_cooling=False, lossy_front=False),
    Candidate("M1_lossy_front_one_cool", split_cooling=False, lossy_front=True),
    Candidate("M2_lossy_front_two_cool", split_cooling=True, lossy_front=True),
    Candidate(
        "M3_lossy_front_two_cool_delayed_gap",
        split_cooling=True,
        lossy_front=True,
        delayed_cooling_gap=True,
    ),
    Candidate(
        "M4_split_front_two_cool_delayed_gap",
        split_cooling=True,
        lossy_front=True,
        delayed_cooling_gap=True,
        split_front_exchange=True,
    ),
)


def _front_fraction(x: np.ndarray, gamma: float) -> np.ndarray:
    """解 C''-beta^2(C-25)=0 后的无量纲炉前温升比例。"""
    s = np.clip(x / q1.FRONT_LEN, 0.0, 1.0)
    if abs(gamma) < 1e-6:
        return s
    return np.sinh(gamma * s) / np.sinh(gamma)


def ambient_candidate(
    x: np.ndarray,
    setpoints: np.ndarray,
    lossy_front: bool,
    gamma: float,
    cool_start_fraction: float = 0.0,
) -> np.ndarray:
    """保留正式模型的受控温区/间隙，仅替换炉前无控区。"""
    x = np.asarray(x, dtype=float)
    C = np.asarray(q1.ambient_temperature_linear(x, setpoints), dtype=float)
    if lossy_front:
        m = (x >= 0.0) & (x < q1.FRONT_LEN)
        C[m] = 25.0 + (setpoints[0] - 25.0) * _front_fraction(x[m], gamma)
    # 第9区后的5 cm无控间隙：允许热区影响延续到间隙内部某一点，
    # 但强制在第10区入口回到其25 °C设定值，避免不物理的长高温尾巴。
    if cool_start_fraction > 0.0:
        x_hot = q1.zone_interval(9)[1]
        x_cold = q1.zone_interval(10)[0]
        x_start = x_hot + cool_start_fraction * (x_cold - x_hot)
        m_hot = (x > x_hot) & (x <= x_start)
        m_drop = (x > x_start) & (x < x_cold)
        C[m_hot] = setpoints[8]
        if np.any(m_drop):
            s = (x[m_drop] - x_start) / max(x_cold - x_start, 1e-12)
            C[m_drop] = setpoints[8] + (setpoints[9] - setpoints[8]) * s
    return C


def unpack(theta: np.ndarray, candidate: Candidate) -> dict:
    if candidate.split_front_exchange:
        k_front, k_pre, k_soak, k_ref, k_cool_1, k_cool_2, gamma, cool_start_fraction = theta
    elif candidate.split_cooling:
        k_pre, k_soak, k_ref, k_cool_1, k_cool_2, gamma = theta[:6]
        cool_start_fraction = theta[6] if candidate.delayed_cooling_gap else 0.0
        k_front = k_pre
    else:
        k_pre, k_soak, k_ref, k_cool_1 = theta[:4]
        k_cool_2 = k_cool_1
        gamma = theta[4] if candidate.lossy_front else 0.0
        cool_start_fraction = 0.0
        k_front = k_pre
    return {
        "k_front": float(k_front),
        "k_pre": float(k_pre),
        "k_soak": float(k_soak),
        "k_ref": float(k_ref),
        "k_cool_1": float(k_cool_1),
        "k_cool_2": float(k_cool_2),
        "gamma_front": float(gamma),
        "cool_start_fraction": float(cool_start_fraction),
    }


def simulate(
    theta: np.ndarray,
    candidate: Candidate,
    setpoints: np.ndarray = q1.SETPOINTS_CAL,
    speed: float = 70.0,
    dt: float = DT_FIT,
    t_end: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """分段常系数下用指数更新集中参数能量方程。"""
    p = unpack(theta, candidate)
    n = int(np.ceil(t_end / dt))
    t = np.linspace(0.0, n * dt, n + 1)
    x = (speed / 60.0) * t[1:]
    C = ambient_candidate(
        x,
        np.asarray(setpoints, dtype=float),
        candidate.lossy_front,
        p["gamma_front"],
        p["cool_start_fraction"],
    )

    k = np.full(n, p["k_pre"], dtype=float)
    k[x < q1.FRONT_LEN] = p["k_front"]
    k[x >= q1.X_SOAK] = p["k_soak"]
    k[x >= q1.X_REFLOW] = p["k_ref"]
    k[x >= q1.X_COOL] = p["k_cool_1"]
    if candidate.split_cooling:
        k[x >= COOL_SPLIT_X] = p["k_cool_2"]

    T = np.empty(n + 1, dtype=float)
    T[0] = 25.0
    decay = np.exp(-k * dt)
    for i in range(n):
        T[i + 1] = C[i] + (T[i] - C[i]) * decay[i]
    return t, T


def _starts_and_bounds(candidate: Candidate) -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    base = np.array([0.012, 0.018, 0.030, 0.004], dtype=float)
    if not candidate.lossy_front:
        starts = [base, np.array([0.008, 0.015, 0.04, 0.003]), np.array([0.02, 0.02, 0.025, 0.007])]
        return starts, (np.full(4, 1e-4), np.full(4, 0.5))
    if not candidate.split_cooling:
        starts = [
            np.r_[base, 2.0],
            np.array([0.012, 0.018, 0.030, 0.003, 5.0]),
            np.array([0.020, 0.020, 0.025, 0.006, 8.0]),
        ]
        return starts, (np.r_[np.full(4, 1e-4), 1e-3], np.r_[np.full(4, 0.5), 30.0])
    if candidate.split_front_exchange:
        starts = [
            np.array([0.010, 0.018, 0.018, 0.025, 0.004, 0.010, 8.0, 0.80]),
            np.array([0.030, 0.015, 0.020, 0.030, 0.003, 0.015, 15.0, 0.90]),
            np.array([0.080, 0.020, 0.020, 0.025, 0.005, 0.020, 25.0, 0.95]),
            np.array([0.005, 0.012, 0.015, 0.040, 0.002, 0.010, 4.0, 0.50]),
        ]
        return (
            starts,
            (np.r_[np.full(6, 1e-4), 1e-3, 0.0], np.r_[np.full(6, 0.5), 30.0, 0.98]),
        )
    if candidate.delayed_cooling_gap:
        starts = [
            np.array([0.012, 0.018, 0.030, 0.002, 0.008, 8.0, 0.50]),
            np.array([0.010, 0.020, 0.035, 0.001, 0.015, 15.0, 0.75]),
            np.array([0.020, 0.020, 0.025, 0.004, 0.020, 20.0, 0.90]),
            np.array([0.008, 0.015, 0.040, 0.003, 0.010, 5.0, 0.25]),
        ]
        return (
            starts,
            (np.r_[np.full(5, 1e-4), 1e-3, 0.0], np.r_[np.full(5, 0.5), 30.0, 0.98]),
        )
    starts = [
        np.array([0.012, 0.018, 0.030, 0.002, 0.008, 4.0]),
        np.array([0.010, 0.020, 0.035, 0.001, 0.015, 7.0]),
        np.array([0.020, 0.020, 0.025, 0.004, 0.020, 9.0]),
        np.array([0.008, 0.015, 0.040, 0.003, 0.010, 2.0]),
    ]
    return starts, (np.r_[np.full(5, 1e-4), 1e-3], np.r_[np.full(5, 0.5), 30.0])


def fit_candidate(candidate: Candidate, t_obs: np.ndarray, y_obs: np.ndarray) -> tuple[np.ndarray, dict]:
    starts, bounds = _starts_and_bounds(candidate)

    def residual(theta: np.ndarray) -> np.ndarray:
        t, T = simulate(theta, candidate, t_end=max(400.0, float(t_obs[-1]) + 1.0))
        return np.interp(t_obs, t, T) - y_obs

    best = None
    for start in starts:
        res = least_squares(
            residual,
            start,
            bounds=bounds,
            method="trf",
            x_scale="jac",
            max_nfev=500,
        )
        sse = float(np.dot(res.fun, res.fun))
        if best is None or sse < best[0]:
            best = (sse, res)
    assert best is not None
    sse, res = best
    err = residual(res.x)
    n, k = len(err), len(res.x)
    # 仅用于同一数据上的候选复杂度比较；不能替代外部验证。
    aic = n * np.log(max(sse / n, 1e-30)) + 2 * k
    bic = n * np.log(max(sse / n, 1e-30)) + k * np.log(n)
    singular = np.linalg.svd(res.jac, compute_uv=False)
    cond = float(singular[0] / max(singular[-1], 1e-30))
    return res.x, {
        "success": bool(res.success),
        "nfev": int(res.nfev),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "sse": sse,
        "aic": float(aic),
        "bic": float(bic),
        "jacobian_condition": cond,
        "residual": err,
    }


def summarize(candidate: Candidate, theta: np.ndarray, stats: dict, t_obs: np.ndarray, y_obs: np.ndarray) -> dict:
    t, T = simulate(theta, candidate, t_end=400.0)
    pred = np.interp(t_obs, t, T)
    err = pred - y_obs
    blocks = {}
    for lo, hi in ((19.0, 40.0), (80.0, 200.0), (300.0, 340.0)):
        mask = (t_obs >= lo) & (t_obs <= hi)
        blocks[f"{lo:g}-{hi:g}"] = {
            "mean": float(np.mean(err[mask])),
            "rmse": float(np.sqrt(np.mean(err[mask] ** 2))),
            "max_abs": float(np.max(np.abs(err[mask]))),
        }
    peak_i = int(np.argmax(T))
    out = {
        "model": candidate.name,
        "params": unpack(theta, candidate),
        "rmse": stats["rmse"],
        "mae": stats["mae"],
        "max_abs": stats["max_abs"],
        "aic": stats["aic"],
        "bic": stats["bic"],
        "jacobian_condition": stats["jacobian_condition"],
        "peak_T": float(T[peak_i]),
        "peak_t": float(t[peak_i]),
        "blocks": blocks,
    }
    return out


def main() -> None:
    t_obs, y_obs = q1.load_calibration_data()
    rows = []
    details = []
    for candidate in CANDIDATES:
        print(f"fit {candidate.name} ...", flush=True)
        theta, stats = fit_candidate(candidate, t_obs, y_obs)
        detail = summarize(candidate, theta, stats, t_obs, y_obs)
        details.append(detail)
        rows.append({k: detail[k] for k in ("model", "rmse", "mae", "max_abs", "aic", "bic", "jacobian_condition", "peak_T", "peak_t")})
        print(json.dumps(detail, ensure_ascii=False, indent=2))

    out = q1.ROOT / "results" / "model_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    (out / "candidate_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
