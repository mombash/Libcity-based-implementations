"""Shared plotting helpers for sparsity paper figures (4-model eval set).

Note on vMAPE / MAPE: The updated Sparsity manuscript removes raw MAPE and vMAPE
from Table 1 (occupancy instability; vMAPE de-emphasized in final tables).
Generators for those metrics remain in scripts/generate_table1_mape_variant.py for
reproducibility and archival comparison only.
"""

from __future__ import annotations

import os
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_DIR = os.path.join(ROOT_DIR, "sparsity_analysis", "results_cache")

MODELS = ["DCRNN", "D2STGNN", "MCSTMambaLST", "Trafformer"]
MODEL_INFO = {
    "DCRNN": {
        "display": "DCRNN",
        "color": "#984ea3",
        "marker": "8",
        "linewidth": 2.0,
        "markersize": 4,
        "zorder": 5,
    },
    "D2STGNN": {
        "display": "D2STGNN",
        "color": "#ff7f00",
        "marker": "^",
        "linewidth": 2.0,
        "markersize": 4,
        "zorder": 4,
    },
    "MCSTMambaLST": {
        "display": "Mamba4Traffic",
        "color": "#377eb8",
        "marker": "o",
        "linewidth": 2.5,
        "markersize": 5,
        "zorder": 8,
    },
    "Trafformer": {
        "display": "Trafformer",
        "color": "#f781bf",
        "marker": "*",
        "linewidth": 2.0,
        "markersize": 5,
        "zorder": 3,
    },
}

DATASETS = ["PEMSD4", "PEMSD8"]
THRESHOLD = {"PEMSD4": 0.41, "PEMSD8": 0.42}

# LaTeX includegraphics names (journal = 3 figures; conference = 2 figures).
JOURNAL_FIGURES = [
    "fig1_{dataset}_weighted_robustness",
    "fig2_{dataset}_ranking_heatmap",
    "fig3_{dataset}_degradation_curves",
]
CONFERENCE_FIGURES = [
    "fig1_{dataset}_weighted_robustness",
    "fig3_{dataset}_degradation_curves",
]


def ds_display(dataset: str) -> str:
    return "PEMS04" if dataset == "PEMSD4" else "PEMS08"


def load_from_cache(
    cache_dir: str,
    lineage: Literal["paper_original", "multiseed_new"],
    grid: Literal["paper", "breakdown"] = "paper",
) -> pd.DataFrame:
    """Load figure-ready vMAE_a series from results_cache CSV exports."""
    if lineage == "paper_original":
        path = os.path.join(cache_dir, "figures", f"paper_original_{grid}_vmae_a.csv")
    else:
        path = os.path.join(cache_dir, "figures", f"multiseed_new_{grid}_vmae_a_mean.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cache export: {path}")

    df = pd.read_csv(path)
    df = df[df["lineage"] == lineage].copy()
    df = df[df["model"].isin(MODELS)].copy()
    if "vMAE_a_std" not in df.columns:
        df["vMAE_a_std"] = 0.0
    return df


def _interpolate_crossover(xs, ys, threshold):
    for i in range(len(xs) - 1):
        if ys[i] <= threshold < ys[i + 1]:
            slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
            if slope == 0:
                continue
            return xs[i] + (threshold - ys[i]) / slope
    return None


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def plot_weighted_robustness(df: pd.DataFrame, dataset: str, out_dir: str, *, title_suffix: str = "") -> str:
    """Paper fig1: baseline vMAE vs OME scatter (67% low + 33% high sparsity)."""
    scatter = []
    for model in MODELS:
        mdf = df[(df["model"] == model) & (df["dataset"] == dataset)].sort_values("sparsity")
        if mdf.empty:
            continue
        v0_rows = mdf[mdf["sparsity"] == 0.0]
        if v0_rows.empty:
            continue
        v0 = v0_rows["vMAE_a"].values[0]

        low = mdf[(mdf["sparsity"] > 0) & (mdf["sparsity"] <= 0.35)]
        high = mdf[mdf["sparsity"] > 0.35]
        s_low = (low["vMAE_a"] / v0).mean() if not low.empty else 1.0
        s_high = (high["vMAE_a"] / v0).mean() if not high.empty else 1.0
        ome = 0.67 * s_low + 0.33 * s_high

        scatter.append(
            {
                "model": model,
                "display": MODEL_INFO[model]["display"],
                "baseline": v0,
                "ome": ome,
                "color": MODEL_INFO[model]["color"],
            }
        )

    fig, ax = plt.subplots(figsize=(8, 7))
    for s in scatter:
        ax.scatter(
            s["baseline"],
            s["ome"],
            c=s["color"],
            s=150,
            edgecolors="black",
            linewidth=0.8,
            zorder=5,
        )
        label_y_offset = 5
        label_x_offset = 5
        label_ha = "left"
        if s["display"] == "DCRNN" and dataset == "PEMSD4":
            label_y_offset = -12
        if s["display"] == "Trafformer":
            label_x_offset = -5
            label_ha = "right"
        ax.annotate(
            s["display"],
            (s["baseline"], s["ome"]),
            fontsize=9,
            fontweight="bold",
            xytext=(label_x_offset, label_y_offset),
            textcoords="offset points",
            ha=label_ha,
        )

    ax.set_xlabel("Overall Error (MAE) ← Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylabel("Weighted Degradation (OME) ← Robustness", fontsize=14, fontweight="bold")
    title = f"Operational Metric of Error (OME) ({ds_display(dataset)})"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f×"))
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, f"fig1_{dataset}_weighted_robustness")


def plot_ranking_heatmap(df: pd.DataFrame, dataset: str, out_dir: str, *, title_suffix: str = "") -> str:
    """Paper fig2: models × sparsity heatmap with rank and vMAE annotations."""
    sub = df[df["dataset"] == dataset].copy()
    levels = sorted(sub["sparsity"].unique())
    common = [lvl for lvl in levels if round(lvl * 100) % 10 == 0]
    sub = sub[sub["sparsity"].isin(common)]

    pivot = sub.pivot_table(index="model", columns="sparsity", values="vMAE_a")
    pivot.index = [MODEL_INFO[m]["display"] for m in pivot.index]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    rank = pivot.rank(axis=0).astype(int)
    annot = rank.copy().astype(str)
    for row in annot.index:
        for col in annot.columns:
            annot.at[row, col] = f"#{rank.at[row, col]}\n{pivot.at[row, col]:.2f}"

    fig, ax = plt.subplots(figsize=(max(10, len(common) * 0.9), max(6, len(pivot) * 0.6)))
    sns.heatmap(
        pivot,
        annot=np.array(annot),
        fmt="",
        cmap="Reds",
        linewidths=0.5,
        linecolor="#eeeeee",
        annot_kws={"fontsize": 9, "fontweight": "bold", "color": "black"},
        cbar_kws={"label": "vMAE (Lower=Better)"},
        ax=ax,
    )
    ax.set_xticklabels(
        [f"{int(float(x.get_text()) * 100)}%" for x in ax.get_xticklabels()],
        fontsize=11,
        fontweight="bold",
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, fontweight="bold", rotation=0)
    title = f"Model Performance Ranking ({ds_display(dataset)})"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Sparsity Level", fontsize=14, fontweight="bold")
    ax.set_ylabel("Model", fontsize=14, fontweight="bold")
    return _save(fig, out_dir, f"fig2_{dataset}_ranking_heatmap")


def plot_degradation_curves(
    df: pd.DataFrame,
    dataset: str,
    out_dir: str,
    *,
    title_suffix: str = "",
    show_std: bool = False,
) -> str:
    """Paper fig3: vMAE_a vs sparsity with 10% usability threshold."""
    thresh = THRESHOLD[dataset]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, color="#cccccc")
    ax.set_facecolor("white")

    for model in MODELS:
        mdf = df[(df["model"] == model) & (df["dataset"] == dataset)].sort_values("sparsity")
        if mdf.empty:
            continue
        info = MODEL_INFO[model]

        xs = mdf["sparsity"].values
        ys = mdf["vMAE_a"].values
        cross = _interpolate_crossover(xs, ys, thresh)

        base_name = info["display"]
        if cross is not None:
            label = f"{base_name} (ρ*={cross * 100:.0f}%)"
        elif ys.min() > thresh:
            label = f"{base_name} (ρ*=0%)"
        else:
            label = f"{base_name} (ρ*>100%)"

        x_pct = mdf["sparsity"] * 100
        ax.plot(
            x_pct,
            mdf["vMAE_a"],
            marker=info["marker"],
            color=info["color"],
            linewidth=info["linewidth"],
            markersize=info["markersize"],
            label=label,
            zorder=info["zorder"],
            alpha=0.9,
        )
        if show_std and "vMAE_a_std" in mdf.columns and mdf["vMAE_a_std"].max() > 0:
            ax.fill_between(
                x_pct,
                mdf["vMAE_a"] - mdf["vMAE_a_std"],
                mdf["vMAE_a"] + mdf["vMAE_a_std"],
                color=info["color"],
                alpha=0.15,
                linewidth=0,
            )

    thresh_color = "#a0522d"
    ax.axhline(
        thresh,
        color=thresh_color,
        ls="--",
        lw=1.5,
        zorder=1,
        alpha=0.7,
        label=f"10% Threshold ({thresh:.2f})",
    )

    ax.set_xlabel("Sensor Sparsity (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("vMAE (Normalized)", fontsize=14, fontweight="bold")
    title = f"Degradation Curve: {ds_display(dataset)}"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=True, shadow=True)
    ax.set_xlim(-2, 102)
    ax.set_ylim(bottom=0)
    return _save(fig, out_dir, f"fig3_{dataset}_degradation_curves")


PLOT_FNS = {
    "fig1": plot_weighted_robustness,
    "fig2": plot_ranking_heatmap,
    "fig3": plot_degradation_curves,
}


def generate_figure_set(
    df: pd.DataFrame,
    out_dir: str,
    *,
    paper_set: Literal["journal", "conference"] = "journal",
    title_suffix: str = "",
    show_std: bool = False,
) -> list[str]:
    """Generate all figures for the requested paper set."""
    fig_keys = ["fig1", "fig2", "fig3"] if paper_set == "journal" else ["fig1", "fig3"]
    saved = []
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            continue
        for key in fig_keys:
            kwargs = {"title_suffix": title_suffix}
            if key == "fig3":
                kwargs["show_std"] = show_std
            path = PLOT_FNS[key](df, dataset, out_dir, **kwargs)
            saved.append(path)
    return saved
