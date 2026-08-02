"""Redraw the figures used by the competition paper.

The script reads the finalized numerical outputs instead of rerunning any
optimization.  Every plot is therefore a presentation layer for an existing
result, not a second source of numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

from q1 import (  # noqa: E402
    FURNACE_LEN,
    FRONT_LEN,
    GAP_LEN,
    N_ZONES,
    SETPOINTS_CAL,
    PlateParams,
    ambient_temperature,
    interpolate_temperature,
    load_calibration_data,
    simulate_plate,
    zone_interval,
)


COLORS = {
    "blue": "#2F5597",
    "orange": "#D97935",
    "red": "#B03A2E",
    "green": "#4C8C6B",
    "gray": "#666666",
    "light": "#E9EEF5",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "#D8D8D8",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plate_params() -> PlateParams:
    data = json.loads((ROOT / "results" / "q1" / "summary.json").read_text(encoding="utf-8"))
    p = data["plate_params"]
    return PlateParams(
        p["eta_pre"], p["eta_soak"], p["eta_ref"],
        p["eta_cool"], p["eta_cool_late"], p.get("eta_r", 0.0), p["alpha"]
    )


def furnace_and_ambient() -> None:
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.55], "hspace": 0.12},
    )

    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.spines[["left", "bottom"]].set_visible(False)
    ax0.add_patch(plt.Rectangle((0, 0.18), FRONT_LEN, 0.5, color="#ECECEC", ec="#777777", lw=0.7))
    ax0.text(FRONT_LEN / 2, 0.43, "炉前 25 cm", ha="center", va="center", fontsize=8)
    group_colors = ["#DCE6F1"] * 5 + ["#FCE4D6", "#FFF2CC"] + ["#F4B183"] * 2 + ["#DDEBF7"] * 2
    for i in range(1, N_ZONES + 1):
        a, b = zone_interval(i)
        ax0.add_patch(plt.Rectangle((a, 0.18), b - a, 0.5, color=group_colors[i - 1], ec="#555555", lw=0.7))
        ax0.text((a + b) / 2, 0.43, str(i), ha="center", va="center", fontsize=8)
        if i < N_ZONES:
            ax0.add_patch(plt.Rectangle((b, 0.18), GAP_LEN, 0.5, color="white", ec="#AAAAAA", lw=0.45))
    rear_start = zone_interval(11)[1]
    ax0.add_patch(plt.Rectangle((rear_start, 0.18), FRONT_LEN, 0.5, color="#ECECEC", ec="#777777", lw=0.7))
    ax0.text(rear_start + FRONT_LEN / 2, 0.43, "炉后", ha="center", va="center", fontsize=8)
    ax0.annotate("传送方向", xy=(FURNACE_LEN * 0.95, 0.88), xytext=(FURNACE_LEN * 0.76, 0.88),
                 arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLORS["gray"]},
                 ha="center", va="center", color=COLORS["gray"])

    x = np.linspace(0, FURNACE_LEN, 3000)
    c = ambient_temperature(x, SETPOINTS_CAL)
    ax1.plot(x, c, color=COLORS["orange"], lw=1.8, label=r"等效环境温度 $C(x)$")
    ax1.axhline(25, color="#888888", lw=0.8, ls="--")
    ax1.fill_between(x, 25, c, color=COLORS["orange"], alpha=0.08)
    ax1.set_xlim(0, FURNACE_LEN)
    ax1.set_ylim(15, 275)
    ax1.set_ylabel("温度 / ℃")
    ax1.set_xlabel("沿炉膛方向的位置 x / cm")
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", frameon=False)
    ax1.annotate("炉口边界影响", xy=(38, float(ambient_temperature(38, SETPOINTS_CAL))), xytext=(70, 80),
                 arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 0.8}, fontsize=8)
    gap_x = zone_interval(6)[1] + 2.5
    ax1.annotate("5 cm 间隙线性过渡", xy=(gap_x, float(ambient_temperature(gap_x, SETPOINTS_CAL))), xytext=(255, 175),
                 arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 0.8}, fontsize=8)
    ax1.annotate("进入冷却区前急降", xy=(zone_interval(10)[0], 35), xytext=(315, 90),
                 arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 0.8}, fontsize=8)
    save(fig, "furnace_ambient")


def calibration_fit() -> None:
    p = plate_params()
    t_obs, y_obs = load_calibration_data()
    t, y = simulate_plate(SETPOINTS_CAL, 70.0, p, t_end=380.0, dt=0.025)
    pred = interpolate_temperature(t, y, t_obs)
    residual = pred - y_obs

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 5.3), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08})
    ax0.scatter(t_obs, y_obs, s=8, facecolor="white", edgecolor="#333333", lw=0.45,
                alpha=0.75, label="实测数据")
    ax0.plot(t, y, color=COLORS["blue"], lw=1.8, label="模型计算")
    ax0.set_ylabel("中心温度 / ℃")
    ax0.grid()
    ax0.legend(frameon=False, ncol=2, loc="upper left")
    ax0.text(0.985, 0.08, "RMSE = 1.12 ℃\nMAE = 0.72 ℃", transform=ax0.transAxes,
             ha="right", va="bottom", fontsize=8.5,
             bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB", "lw": 0.6})
    ax1.axhline(0, color="#777777", lw=0.8)
    ax1.plot(t_obs, residual, color=COLORS["orange"], lw=0.9)
    ax1.fill_between(t_obs, 0, residual, color=COLORS["orange"], alpha=0.16)
    ax1.set_ylabel("残差 / ℃")
    ax1.set_xlabel("时间 t / s")
    ax1.grid(axis="y")
    save(fig, "calibration_fit")


def q1_profile() -> None:
    curve = pd.read_csv(ROOT / "results" / "q1" / "result.csv")
    probes = pd.read_csv(ROOT / "results" / "q1" / "probe_temperatures.csv")
    t = curve.iloc[:, 0].to_numpy(float)
    T = curve.iloc[:, 1].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(t, T, color=COLORS["blue"], lw=1.9)
    ax.axhspan(150, 190, color=COLORS["green"], alpha=0.09, label="150–190 ℃")
    ax.axhline(217, color=COLORS["orange"], lw=0.9, ls="--", label="217 ℃")
    labels = ["3 区中点", "6 区中点", "7 区中点", "8 区出口"]
    offsets = [(4, 9), (4, 9), (4, 9), (4, 18)]
    for (_, row), label, offset in zip(probes.iterrows(), labels, offsets):
        ax.scatter(row["t_s"], row["T_M2"], s=28, color=COLORS["red"], zorder=4)
        ax.vlines(row["t_s"], 25, row["T_M2"], color="#999999", lw=0.55, ls=":")
        ax.annotate(f"{label}\n{row['T_M2']:.2f} ℃", (row["t_s"], row["T_M2"]),
                    xytext=offset, textcoords="offset points", fontsize=7.8)
    peak_i = int(np.argmax(T))
    ax.annotate(f"峰值 {T[peak_i]:.2f} ℃", (t[peak_i], T[peak_i]), xytext=(-68, -33),
                textcoords="offset points", arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=8.2)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(20, 260)
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("焊接区域中心温度 / ℃")
    ax.grid()
    ax.legend(frameon=False, loc="upper left")
    save(fig, "q1_profile")


def q2_constraints() -> None:
    data = pd.read_csv(ROOT / "results" / "q2" / "speed_scan.csv").sort_values("v")
    data = data.drop_duplicates("v")
    v = data["v"].to_numpy(float)
    peak = data["T_peak"].to_numpy(float)
    tau = data["tau_217"].to_numpy(float)
    v_lo, v_hi = 68.0016, 79.5936

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8.5, 5.5), sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    for ax in (ax0, ax1):
        ax.axvspan(v_lo, v_hi, color=COLORS["green"], alpha=0.12, label="可行速度区间")
        ax.axvline(v_hi, color=COLORS["red"], lw=1.0, ls="--")
        ax.grid()
    ax0.plot(v, peak, color=COLORS["blue"], lw=1.7)
    ax0.axhspan(240, 250, color=COLORS["orange"], alpha=0.09)
    ax0.axhline(240, color=COLORS["orange"], lw=1.0, ls="--")
    ax0.set_ylabel("峰值温度 / ℃")
    ax0.annotate("上边界由峰值下限确定", xy=(v_hi, 240), xytext=(82.2, 242.0),
                 arrowprops={"arrowstyle": "->", "lw": 0.75}, fontsize=8)
    ax0.legend(frameon=False, loc="upper right")
    ax1.plot(v, tau, color=COLORS["blue"], lw=1.7)
    ax1.axhspan(40, 90, color=COLORS["orange"], alpha=0.09)
    ax1.axhline(90, color=COLORS["orange"], lw=1.0, ls="--")
    ax1.annotate("下边界由液相时间上限确定", xy=(v_lo, 90), xytext=(72.5, 84),
                 arrowprops={"arrowstyle": "->", "lw": 0.75}, fontsize=8)
    ax1.set_ylabel(r"$T>217$ ℃ 的时间 / s")
    ax1.set_xlabel("过炉速度 v / (cm/min)")
    ax1.set_xlim(65, 100)
    save(fig, "q2_constraint_boundary")


def q3_area() -> None:
    curve = pd.read_csv(ROOT / "results" / "q3" / "best_curve.csv")
    summary = json.loads((ROOT / "results" / "q3" / "summary.json").read_text(encoding="utf-8"))["best"]
    t, T = curve["t"].to_numpy(float), curve["T"].to_numpy(float)
    tu, tp = summary["metrics"]["t217_up"], summary["metrics"]["t_peak"]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(t, T, color=COLORS["blue"], lw=1.9)
    mask = (t >= tu) & (t <= tp)
    ax.fill_between(t[mask], 217, T[mask], color=COLORS["orange"], alpha=0.36,
                    label=r"目标面积 $A=410.68$ ℃·s")
    ax.axhline(217, color="#777777", lw=0.9, ls="--")
    ax.axvline(tu, color="#999999", lw=0.7, ls=":")
    ax.axvline(tp, color="#999999", lw=0.7, ls=":")
    ax.scatter([tu, tp], [217, summary["metrics"]["T_peak"]], color=COLORS["red"], s=24, zorder=4)
    ax.text(tu, 205, r"$t_u$", ha="center", va="top")
    ax.text(tp, 205, r"$t_p$", ha="center", va="top")
    ax.set_xlim(110, 285)
    ax.set_ylim(120, 255)
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("中心温度 / ℃")
    ax.grid()
    ax.legend(frameon=False, loc="upper left")
    save(fig, "q3_area")


def q4_tradeoff_and_mirror() -> None:
    exact = pd.read_csv(ROOT / "results" / "q4" / "pareto_exact.csv").sort_values("A_L")
    practical = pd.read_csv(ROOT / "results" / "q4" / "pareto_front.csv").sort_values("A_L")
    curve = pd.read_csv(ROOT / "results" / "q4" / "best_curve.csv")
    summary = json.loads((ROOT / "results" / "q4" / "summary.json").read_text(encoding="utf-8"))["final"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.2, 4.15), gridspec_kw={"wspace": 0.28})
    ax0.scatter(exact["A_L"], exact["J_sym"], s=28, facecolor="white", edgecolor="#888888",
                lw=0.8, label="精确非支配解")
    ax0.plot(practical["A_L"], practical["J_sym"], color=COLORS["blue"], lw=1.4, marker="o",
             ms=4.5, label="实用前沿")
    names = ["面积最小", "均衡备选", "对称优先"]
    offsets = [(5, 7), (-34, 9), (-46, -17)]
    for (_, row), name, offset in zip(practical.iterrows(), names, offsets):
        ax0.annotate(name, (row["A_L"], row["J_sym"]), xytext=offset,
                     textcoords="offset points", fontsize=7.8)
    ax0.set_xlabel(r"升温侧面积 $A_L$ / (℃·s)")
    ax0.set_ylabel(r"对称指标 $J_{sym}$")
    ax0.grid()
    ax0.legend(frameon=False, loc="upper right")

    t, T = curve["t"].to_numpy(float), curve["T"].to_numpy(float)
    m = summary["metrics"]
    tau_l, tau_r = m["tau_L"], m["tau_R"]
    tau = np.linspace(0, max(tau_l, tau_r), 500)
    ql = np.maximum(np.where(tau <= tau_l, np.interp(m["t_peak"] - tau, t, T) - 217, 0), 0)
    qr = np.maximum(np.where(tau <= tau_r, np.interp(m["t_peak"] + tau, t, T) - 217, 0), 0)
    ax1.plot(tau, ql, color=COLORS["orange"], lw=1.8, label="峰值左侧")
    ax1.plot(tau, qr, color=COLORS["blue"], lw=1.8, label="峰值右侧镜像")
    ax1.fill_between(tau, ql, qr, color=COLORS["red"], alpha=0.16,
                     label=r"镜像误差 $E_{sym}$")
    ax1.set_xlabel("距峰值时刻的镜像时间 / s")
    ax1.set_ylabel("超过 217 ℃ 的温差 / ℃")
    ax1.grid()
    ax1.legend(frameon=False, loc="upper right")
    save(fig, "q4_tradeoff_mirror")


def sensitivity() -> None:
    data = pd.read_csv(ROOT / "results" / "verification" / "param_sensitivity.csv")
    data = data[data["param"] != "baseline"].copy()
    order = ["eta_pre", "eta_soak", "eta_ref", "eta_cool", "eta_cool_late"]
    labels = ["预热段", "恒温段", "回流段", "急冷段", "后冷却段"]
    fig, ax = plt.subplots(figsize=(8.3, 4.1))
    y = np.arange(len(order))
    width = 0.34
    for k, (delta, color, legend) in enumerate([(-0.05, COLORS["blue"], "−5%"), (0.05, COLORS["orange"], "+5%")]):
        vals = []
        for name in order:
            row = data[(data["param"] == name) & np.isclose(data["delta"], delta)]
            vals.append(float(row["dA"].iloc[0]) if not row.empty else 0.0)
        ax.barh(y + (k - 0.5) * width, vals, height=width, color=color, alpha=0.9, label=legend)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("问题三面积变化 ΔA / (℃·s)")
    ax.grid(axis="x")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    save(fig, "parameter_sensitivity")


def main() -> None:
    furnace_and_ambient()
    calibration_fit()
    q1_profile()
    q2_constraints()
    q3_area()
    q4_tradeoff_and_mirror()
    sensitivity()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
