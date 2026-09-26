#!/usr/bin/env python3
"""Channel-wise operational breakdown figures released with the paper.

These plots are supplementary to the manuscript. Uses the standardized 3-seed cache
(results_cache_standard_dcrnn_fixed_3seed), vMAE only, operational grid 0--35%.
Per-channel tau_c = 0.10 * mu_c / sigma_c from the manuscript dataset table
(same formula as the paper's aggregate tau).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(
    os.environ.get(
        "SPARSITY_RESULTS_CACHE",
        str(REPO / "artifacts" / "paper-release" / "evaluation" / "cache" / "channel_breakdown"),
    )
).resolve()
PAPER = Path(
    os.environ.get("SPARSITY_PAPER_DIR", str(REPO / "reproduced" / "channel_breakdown"))
).resolve()
OUT = PAPER

MODELS = ["DCRNN", "D2STGNN", "Mamba4Traffic", "Trafformer"]
DATASETS = [("PEMSD4", "PEMS04"), ("PEMSD8", "PEMS08")]
CHANNELS = ["flow", "occupancy", "speed"]
MODEL_COLORS = {
    "DCRNN": "#1f77b4",
    "D2STGNN": "#2ca02c",
    "Mamba4Traffic": "#9467bd",
    "Trafformer": "#ff7f0e",
}
MODEL_MARKERS = {
    "DCRNN": "o",
    "D2STGNN": "s",
    "Mamba4Traffic": "D",
    "Trafformer": "^",
}

# Manuscript Table tab:dataset_stats_detailed (train-split scaler used for z_{0,c}).
TAU = {
    "PEMSD4": {
        "flow": 0.10 * 208.85 / 157.33,
        "occupancy": 0.10 * 0.0520 / 0.0488,
        "speed": 0.10 * 63.44 / 8.25,
    },
    "PEMSD8": {
        "flow": 0.10 * 230.24 / 145.89,
        "occupancy": 0.10 * 0.0646 / 0.0450,
        "speed": 0.10 * 63.78 / 6.56,
    },
}


def _interpolate_crossover(xs: np.ndarray, ys: np.ndarray, threshold: float) -> float | None:
    if ys[0] >= threshold:
        return 0.0
    for i in range(1, len(xs)):
        if ys[i - 1] < threshold <= ys[i]:
            slope = (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1])
            if slope == 0:
                continue
            return float(xs[i - 1] + (threshold - ys[i - 1]) / slope)
    return None


def _spread_label_x(unique_pcts: list[int], min_gap: float = 4.0) -> dict[int, float]:
    """Keep label boxes at least min_gap apart on the sparsity axis."""
    xs: dict[int, float] = {}
    for pct in unique_pcts:
        x = 12.0 if pct == 0 else float(pct)
        if xs:
            prev = max(xs.values())
            if x < prev + min_gap:
                x = prev + min_gap
        xs[pct] = min(x, 33.0)
    return xs


def load_breakdown() -> pd.DataFrame:
    path = CACHE / "figures" / "multiseed_new_breakdown_all_metrics_mean_std.csv"
    df = pd.read_csv(path)
    df = df[df["grid"] == "breakdown"].copy()
    return df


def plot_channel(df: pd.DataFrame, ds_key: str, ds_disp: str, channel: str) -> None:
    thresh = TAU[ds_key][channel]
    fig, ax = plt.subplots(figsize=(11, 7))
    sub = df[df["dataset"] == ds_key]
    col = f"vMAE_{channel}_mean"
    std_col = f"vMAE_{channel}_std"
    crossings: list[tuple[str, float]] = []

    for display in MODELS:
        mdf = sub[sub["model_figure"] == display].sort_values("sparsity")
        if mdf.empty:
            mdf = sub[sub["model"] == display].sort_values("sparsity")
        if mdf.empty:
            continue
        xs = mdf["sparsity"].to_numpy(dtype=float)
        ys = mdf[col].to_numpy(dtype=float)
        x_pct = xs * 100
        if std_col in mdf.columns and float(mdf[std_col].max()) > 0:
            ax.fill_between(
                x_pct,
                ys - mdf[std_col].to_numpy(dtype=float),
                ys + mdf[std_col].to_numpy(dtype=float),
                color=MODEL_COLORS[display],
                alpha=0.15,
                linewidth=0,
            )
        ax.plot(
            x_pct,
            ys,
            marker=MODEL_MARKERS[display],
            linewidth=2.8,
            markersize=9,
            label=display,
            color=MODEL_COLORS[display],
        )
        cross = _interpolate_crossover(xs * 100, ys, thresh)
        if cross is not None:
            crossings.append((display, cross))
            ax.plot(cross, thresh, "+", markersize=14, markeredgewidth=2.5, color=MODEL_COLORS[display])

    ax.axhline(thresh, color="#c0392b", linestyle="--", linewidth=2, alpha=0.85)
    ax.text(
        35.5,
        thresh,
        rf"10% threshold ($\tau_{{{channel[0]}}}={thresh:.3f}$)",
        fontsize=11,
        color="#c0392b",
        va="bottom",
        ha="right",
    )

    # One label per rounded percent. Close crossings get spread x and a short leader.
    unique_pcts = sorted({int(round(cross)) for _, cross in crossings})
    y0, y1 = ax.get_ylim()
    y_span = max(y1 - y0, 1e-6)
    # When the threshold sits near the axis, hug the line so labels stay
    # off the curves. Mid-plot thresholds keep a larger split above/below.
    if (thresh - y0) / y_span < 0.28:
        pad = 0.022
    else:
        pad = max(0.035 * y_span, 0.03)
    y_up = thresh + pad
    y_dn = thresh - pad
    if y_dn < y0 + 0.02 * y_span:
        y_dn = thresh + 2 * pad
    label_x = _spread_label_x(unique_pcts)
    for i, pct in enumerate(unique_pcts):
        y_lab = y_up if i % 2 == 0 else y_dn
        x_lab = label_x[pct]
        ax.annotate(
            f"{pct}%",
            xy=(float(pct), thresh),
            xytext=(x_lab, y_lab),
            textcoords="data",
            fontsize=11,
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8, shrinkA=2, shrinkB=2),
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="#333333",
                linewidth=1.0,
            ),
            zorder=5,
        )

    ax.set_xlim(-1, 37)
    ax.set_xlabel("Sensor sparsity (%)", fontsize=14)
    ax.set_ylabel(rf"$v$MAE ({channel})", fontsize=14)
    ax.set_title(
        rf"{ds_disp}: operational breakdown — {channel} channel",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    stem = f"fig_breakdown_{ds_disp.lower()}_{channel}"
    _save(fig, stem)


def plot_aggregated(df: pd.DataFrame, ds_key: str, ds_disp: str) -> None:
    tau_avg = float(np.mean(list(TAU[ds_key].values())))
    fig, ax = plt.subplots(figsize=(12, 7.5))
    sub = df[df["dataset"] == ds_key]
    for display in MODELS:
        mdf = sub[sub["model_figure"] == display].sort_values("sparsity")
        if mdf.empty:
            mdf = sub[sub["model"] == display].sort_values("sparsity")
        if mdf.empty:
            continue
        x_pct = mdf["sparsity"].to_numpy(dtype=float) * 100
        ys = mdf["vMAE_a_mean"].to_numpy(dtype=float)
        ax.plot(
            x_pct,
            ys,
            marker=MODEL_MARKERS[display],
            linewidth=2.8,
            markersize=9,
            label=display,
            color=MODEL_COLORS[display],
        )
    ax.axhline(tau_avg, color="#c0392b", linestyle="--", linewidth=2, alpha=0.85)
    ax.text(
        35.5,
        tau_avg,
        rf"Avg 10% threshold ($\tau={tau_avg:.2f}$)",
        fontsize=11,
        color="#c0392b",
        va="bottom",
        ha="right",
    )
    ax.set_xlim(-1, 37)
    ax.set_xlabel("Sensor sparsity (%)", fontsize=14)
    ax.set_ylabel(r"$v$MAE$_a$", fontsize=14)
    ax.set_title(
        rf"{ds_disp}: operational breakdown — channel-averaged $v$MAE$_a$",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, f"fig_breakdown_{ds_disp.lower()}_aggregated_vmae")


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
        }
    )
    df = load_breakdown()
    for ds_key, ds_disp in DATASETS:
        for channel in CHANNELS:
            plot_channel(df, ds_key, ds_disp, channel)
    print(f"\nDone — figures in {OUT}")


if __name__ == "__main__":
    main()
