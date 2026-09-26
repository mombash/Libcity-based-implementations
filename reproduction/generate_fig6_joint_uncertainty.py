#!/usr/bin/env python3
"""CPU-only Figure 6 alternatives combining mask-seed SD and alpha sensitivity.

The calculation uses the Figure 6 ordinate: the mean of the
three channel-wise weighted MAE degradation factors. No model is loaded and
no evaluation or GPU computation is performed.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("SPARSITY_RESULTS_CACHE", str(REPO / "artifacts" / "paper-release" / "evaluation" / "cache" / "manuscript"))).resolve()
PAPER = Path(os.environ.get("SPARSITY_PAPER_DIR", str(REPO / "reproduced" / "manuscript"))).resolve()
SOURCE = Path(os.environ.get("SPARSITY_FIG6_SOURCE", str(CACHE / "long_form" / "multiseed_new_all_metrics.csv"))).resolve()
OUT = PAPER / "derived" / "fig6_joint_uncertainty"
FIGURES = PAPER / "figures"
RESULTS = OUT / "results"

DATASETS = ["PEMS04", "PEMS08"]
MODELS = ["DCRNN", "D2STGNN", "Mamba4Traffic", "Trafformer"]
SEEDS = [43, 44, 45]
CHANNELS = ["flow", "occupancy", "speed"]
ALPHAS = np.arange(10.0, 40.0001, 0.5)
SELECTED = [10.0, 20.0, 40.0]

COLORS = {
    "DCRNN": "#1f77b4",
    "D2STGNN": "#2ca02c",
    "Mamba4Traffic": "#9467bd",
    "Trafformer": "#ff7f0e",
}
MARKERS = {"DCRNN": "o", "D2STGNN": "s", "Mamba4Traffic": "D", "Trafformer": "^"}

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
    }
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    df = df[
        (df["lineage"] == "multiseed_new")
        & (df["grid"] == "breakdown")
        & (df["sparsity"] <= 0.3500001)
        & (df["dataset_display"].isin(DATASETS))
        & (df["model_table"].isin(MODELS))
        & (df["mask_seed"].isin(SEEDS))
    ].copy()
    expected = len(DATASETS) * len(MODELS) * len(SEEDS) * 8
    if len(df) != expected:
        raise RuntimeError(f"Expected {expected} rows; found {len(df)}")
    return df


def calculate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                g = df[
                    (df["dataset_display"] == dataset)
                    & (df["model_table"] == model)
                    & (df["mask_seed"] == seed)
                ].sort_values("sparsity")
                rho = g["sparsity"].to_numpy(float)
                baseline = float(g.loc[np.isclose(g["sparsity"], 0.0), "vMAE_a"].iloc[0])
                degradation = []
                for channel in CHANNELS:
                    values = g[f"vMAE_{channel}"].to_numpy(float)
                    degradation.append(values / values[0])
                degradation = np.vstack(degradation)
                for alpha in ALPHAS:
                    weights = np.exp(-alpha * rho**2)
                    weights /= weights.sum()
                    channel_scores = degradation @ weights
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Model": model,
                            "Seed": seed,
                            "alpha": float(alpha),
                            "Baseline_vMAE_a": baseline,
                            "Dbar_MAE": float(channel_scores.mean()),
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["Dataset", "Model", "alpha"], sort=False)
        .agg(
            Baseline_vMAE_a=("Baseline_vMAE_a", "mean"),
            Dbar_MAE=("Dbar_MAE", "mean"),
            Mask_seed_SD=("Dbar_MAE", "std"),
        )
        .reset_index()
    )
    summary["Robustness_rank"] = summary.groupby(["Dataset", "alpha"])["Dbar_MAE"].rank().astype(int)
    return detail, summary


def model_stats(summary: pd.DataFrame, dataset: str, model: str) -> dict[str, float]:
    g = summary[(summary["Dataset"] == dataset) & (summary["Model"] == model)].sort_values("alpha")
    cur = g[np.isclose(g["alpha"], 20.0)].iloc[0]
    lo = g[np.isclose(g["alpha"], 40.0)].iloc[0]
    hi = g[np.isclose(g["alpha"], 10.0)].iloc[0]
    return {
        "x": float(cur["Baseline_vMAE_a"]),
        "current": float(cur["Dbar_MAE"]),
        "sd": float(cur["Mask_seed_SD"]),
        "alpha10": float(hi["Dbar_MAE"]),
        "alpha40": float(lo["Dbar_MAE"]),
        "low": float(g["Dbar_MAE"].min()),
        "high": float(g["Dbar_MAE"].max()),
    }


def label_point(ax: plt.Axes, dataset: str, model: str, x: float, y: float) -> None:
    offsets = {
        ("PEMS04", "D2STGNN"): (6, 0),
        ("PEMS04", "Mamba4Traffic"): (6, -10),
        ("PEMS04", "DCRNN"): (-6, 10),
        ("PEMS04", "Trafformer"): (6, -11),
        ("PEMS08", "D2STGNN"): (6, 8),
        ("PEMS08", "Mamba4Traffic"): (6, -12),
        ("PEMS08", "DCRNN"): (-6, 9),
        ("PEMS08", "Trafformer"): (-6, -12),
    }
    dx, dy = offsets[(dataset, model)]
    ax.annotate(
        model,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx > 0 else "right",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=COLORS[model],
        zorder=10,
    )


def style_axis(ax: plt.Axes, dataset: str) -> None:
    ax.set_title(dataset, fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Baseline $vMAE_a$ at $\rho=0$ (lower is better)")
    ax.grid(True, alpha=0.28, linestyle="--", zorder=0)
    ax.tick_params(labelsize=8)
    if dataset == "PEMS04":
        ax.set_xlim(0.147, 0.245)
        ax.set_ylim(0.88, 6.32)
    else:
        ax.set_xlim(0.145, 0.273)
        ax.set_ylim(1.30, 2.78)


def draw_nested(ax: plt.Axes, summary: pd.DataFrame, dataset: str) -> None:
    for model in MODELS:
        s = model_stats(summary, dataset, model)
        color = COLORS[model]
        ax.plot([s["x"], s["x"]], [s["low"], s["high"]], color=color, lw=9, alpha=0.22, solid_capstyle="round", zorder=2)
        ax.plot([s["x"], s["x"]], [s["low"], s["high"]], color=color, lw=1.5, alpha=0.95, zorder=3)
        ax.errorbar(s["x"], s["current"], yerr=s["sd"], fmt="none", ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=6)
        ax.scatter(s["x"], s["current"], s=70, marker=MARKERS[model], c=color, edgecolors="black", linewidths=0.9, zorder=7)
        label_point(ax, dataset, model, s["x"], s["current"])


def draw_endpoints(ax: plt.Axes, summary: pd.DataFrame, dataset: str) -> None:
    for model in MODELS:
        s = model_stats(summary, dataset, model)
        color = COLORS[model]
        ax.plot([s["x"], s["x"]], [s["alpha40"], s["alpha10"]], color=color, lw=2.2, alpha=0.8, zorder=2)
        ax.scatter(s["x"], s["alpha10"], s=38, marker="v", facecolors="white", edgecolors=color, linewidths=1.4, zorder=4)
        ax.scatter(s["x"], s["alpha40"], s=38, marker="^", facecolors="white", edgecolors=color, linewidths=1.4, zorder=4)
        ax.errorbar(s["x"], s["current"], yerr=s["sd"], fmt="none", ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=6)
        ax.scatter(s["x"], s["current"], s=70, marker=MARKERS[model], c=color, edgecolors="black", linewidths=0.9, zorder=7)
        label_point(ax, dataset, model, s["x"], s["current"])


def draw_jitter(ax: plt.Axes, summary: pd.DataFrame, dataset: str) -> None:
    xspan = 0.152 if dataset == "PEMS04" else 0.128
    offsets = {10.0: -0.018 * xspan, 20.0: 0.0, 40.0: 0.018 * xspan}
    for model in MODELS:
        g = summary[(summary["Dataset"] == dataset) & (summary["Model"] == model) & (summary["alpha"].isin(SELECTED))].sort_values("alpha")
        x0 = float(g["Baseline_vMAE_a"].iloc[0])
        xs = np.array([x0 + offsets[float(a)] for a in g["alpha"]])
        ys = g["Dbar_MAE"].to_numpy(float)
        es = g["Mask_seed_SD"].to_numpy(float)
        color = COLORS[model]
        ax.plot(xs, ys, color=color, lw=1.4, alpha=0.8, zorder=2)
        for x, y, e, a in zip(xs, ys, es, g["alpha"]):
            filled = np.isclose(a, 20.0)
            ax.errorbar(x, y, yerr=e, fmt="none", ecolor="black", elinewidth=1.1, capsize=3, capthick=1.1, zorder=5)
            ax.scatter(x, y, s=62 if filled else 42, marker=MARKERS[model], facecolors=color if filled else "white", edgecolors=color if not filled else "black", linewidths=1.0, zorder=6)
        current = g[np.isclose(g["alpha"], 20.0)].iloc[0]
        label_point(ax, dataset, model, x0, float(current["Dbar_MAE"]))


def legend_for(kind: str) -> list[Line2D]:
    if kind == "nested":
        return [
            Line2D([0], [0], color="#666666", lw=8, alpha=0.25, label=r"mean range, $\alpha=10$--40"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor="#999999", lw=1.2, label=r"$\alpha=20$ mean $\pm1\sigma$ masks"),
        ]
    if kind == "endpoints":
        return [
            Line2D([0], [0], marker="v", color="#666666", markerfacecolor="white", lw=1.5, label=r"$\alpha=10$"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor="#999999", lw=1.2, label=r"$\alpha=20$ mean $\pm1\sigma$ masks"),
            Line2D([0], [0], marker="^", color="#666666", markerfacecolor="white", lw=1.5, label=r"$\alpha=40$"),
        ]
    return [
        Line2D([0], [0], marker="o", color="#666666", markerfacecolor="white", lw=1.2, label=r"$\alpha=10,20,40$ (x-offset visual only)"),
        Line2D([0], [0], color="black", marker="_", lw=1.2, label=r"each mean $\pm1\sigma$ masks"),
    ]


def make_combined(summary: pd.DataFrame, kind: str, stem: str) -> None:
    drawers = {"nested": draw_nested, "endpoints": draw_endpoints, "jitter": draw_jitter}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.45))
    for ax, dataset in zip(axes, DATASETS):
        drawers[kind](ax, summary, dataset)
        style_axis(ax, dataset)
    axes[0].set_ylabel(r"Weighted degradation $\bar{D}_{MAE}$ (lower is better)")
    fig.legend(handles=legend_for(kind), loc="upper center", ncol=3 if kind != "nested" else 2, frameon=False, bbox_to_anchor=(0.5, 1.015), fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def make_dropins(summary: pd.DataFrame) -> None:
    for dataset, letter, tag in [("PEMS04", "a", "pems04"), ("PEMS08", "b", "pems08")]:
        fig, ax = plt.subplots(figsize=(12, 9))
        draw_nested(ax, summary, dataset)
        style_axis(ax, dataset)
        ax.set_ylabel(r"Weighted Degradation $\bar{D}_{MAE}$" + "\n" + r"$\leftarrow$ More Robust (lower)", fontsize=18)
        ax.set_xlabel(r"Baseline $vMAE_a$ (at 0% sparsity)" + "\n" + r"$\leftarrow$ Better Accuracy (lower)", fontsize=18)
        ax.set_title(f"{dataset}: Accuracy vs Robustness Tradeoff", fontsize=21, fontweight="bold")
        ax.tick_params(labelsize=15)
        ax.legend(handles=legend_for("nested"), loc="upper right", frameon=True, fontsize=13)
        fig.tight_layout()
        stem = f"fig6{letter}_accuracy_robustness_{tag}_alpha_envelope"
        fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_tables(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[summary["alpha"].isin(SELECTED)].copy()
    selected.to_csv(RESULTS / "fig6_joint_selected_alpha.csv", index=False)
    envelope = (
        summary.groupby(["Dataset", "Model"], sort=False)
        .agg(
            Baseline_vMAE_a=("Baseline_vMAE_a", "first"),
            Alpha_envelope_min=("Dbar_MAE", "min"),
            Alpha_envelope_max=("Dbar_MAE", "max"),
        )
        .reset_index()
    )
    current = summary[np.isclose(summary["alpha"], 20.0)][["Dataset", "Model", "Dbar_MAE", "Mask_seed_SD", "Robustness_rank"]]
    envelope = envelope.merge(current, on=["Dataset", "Model"])
    envelope.to_csv(RESULTS / "fig6_joint_envelope_summary.csv", index=False)
    summary.to_csv(RESULTS / "fig6_joint_alpha_sweep.csv", index=False)
    return envelope


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _, summary = calculate(load_data())
    envelope = write_tables(summary)
    # All ranks must remain fixed over the 61-point range.
    rank_counts = summary.groupby(["Dataset", "Model"])["Robustness_rank"].nunique()
    if int(rank_counts.max()) != 1:
        raise RuntimeError("A robustness rank changes over alpha=10--40")
    make_combined(summary, "nested", "fig6_joint_nested_recommended")
    make_combined(summary, "endpoints", "fig6_joint_endpoint_coded")
    make_combined(summary, "jitter", "fig6_joint_three_alpha_jitter")
    make_dropins(summary)
    print(envelope.to_string(index=False))
    print(f"Wrote joint-uncertainty Figure 6 alternatives to {OUT}")


if __name__ == "__main__":
    main()
