#!/usr/bin/env python3
"""Generate Table II, Table III, and key sparsity figures from multiseed results cache.

Outputs:
  table2/tabular_multiseed_vmae_a.tex          — multilevel sparsity (paper grid)
  table3/tabular_multiseed_horizon_10pct.tex   — vMAE_a at 10% by horizon
  figures/fig1{a,b}_operational_breakdown_*.pdf
  figures/fig2{a,b}_ranking_matrix_*.pdf
  figures/fig5{a,b}_percentage_increase_*.pdf
  figures/fig6{a,b}_accuracy_robustness_*.pdf
      (Table I Avg bar_D_MAE ±1σ over masks; x-axis = vMAE_a at rho=0)
  figures/fig6{a,b}_accuracy_robustness_*_nobars.pdf
      (same points without error bars)

Data: Bigscity-LibCity/sparsity_analysis/results_cache (lineage multiseed_new, seeds 43–45)
"""

from __future__ import annotations

import math
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(
    os.environ.get(
        "SPARSITY_RESULTS_CACHE",
        str(REPO / "artifacts" / "paper-release" / "evaluation" / "cache" / "manuscript"),
    )
).resolve()
PAPER = Path(
    os.environ.get("SPARSITY_PAPER_DIR", str(REPO / "reproduced" / "manuscript"))
).resolve()
OUT_TABLE2 = PAPER / "table2" / "tabular_multiseed_vmae_a.tex"
OUT_TABLE3 = PAPER / "table3" / "tabular_multiseed_horizon_10pct.tex"
OUT_FIGURES = PAPER / "figures"

SEEDS = [43, 44, 45]
SPARSITY_PCT_COLS = [f"{p}%" for p in range(0, 101, 10)]
BREAKDOWN_LEVELS = [round(x * 0.05, 2) for x in range(8)]  # 0..0.35

MODELS = [
    ("DCRNN", "DCRNN"),
    ("D2STGNN", "D2STGNN"),
    ("Mamba4Traffic", "Mamba4Traffic"),
    ("Trafformer", "Trafformer"),
]
RAW_MODEL = {
    "DCRNN": "DCRNN",
    "D2STGNN": "D2STGNN",
    "Mamba4Traffic": "Mamba4Traffic",
    "Trafformer": "Trafformer",
}
DATASETS = [("PEMSD4", "PEMS04"), ("PEMSD8", "PEMS08")]
THRESHOLD = {"PEMSD4": 0.33, "PEMSD8": 0.42}

DISPLAY = {d: d for d, _ in MODELS}
MODEL_KEY_TO_DISPLAY = {key: disp for disp, key in MODELS}
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
MECHANISM_COLORS = {
    "RNN-GNN": "#1f77b4",
    "Dynamic-GNN": "#2ca02c",
    "Transformer": "#ff7f0e",
    "State-Space": "#9467bd",
}
SPATIAL_MECHANISM = {
    "DCRNN": "RNN-GNN",
    "D2STGNN": "Dynamic-GNN",
    "Trafformer": "Transformer",
    "Mamba4Traffic": "State-Space",
}

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


def gaussian_weights(rhos: list[float]) -> dict[float, float]:
    w = [math.exp(-20.0 * rho * rho) for rho in rhos]
    total = sum(w)
    return {rho: x / total for rho, x in zip(rhos, w)}


def load_paper_vmae() -> pd.DataFrame:
    path = CACHE / "tables" / "multiseed_new_table2_paper_vmae_a_wide.csv"
    return pd.read_csv(path)


def load_breakdown_vmae() -> pd.DataFrame:
    path = CACHE / "figures" / "multiseed_new_breakdown_vmae_a_mean.csv"
    df = pd.read_csv(path)
    if "vMAE_a_std" not in df.columns:
        df["vMAE_a_std"] = 0.0
    return df


def load_horizon_at_10pct() -> dict[tuple[str, str, str], list[float]]:
    """Return {(dataset_key, display, Hk): [seed values]} for k=1..12."""
    out: dict[tuple[str, str, str], list[float]] = {}
    for ds_key, ds_disp in DATASETS:
        for display, _ in MODELS:
            raw_model = RAW_MODEL[display]
            for seed in SEEDS:
                csv_path = (
                    CACHE
                    / "raw"
                    / "multiseed_new"
                    / raw_model
                    / ds_key
                    / "paper"
                    / f"seed{seed}"
                    / "sparsity_results.csv"
                )
                if not csv_path.is_file():
                    raise FileNotFoundError(csv_path)
                df = pd.read_csv(csv_path)
                row = df[df["sparsity"].between(0.1 - 1e-9, 0.1 + 1e-9)]
                if row.empty:
                    raise RuntimeError(f"No 10% row in {csv_path}")
                r = row.iloc[0]
                for h in range(1, 13):
                    col = f"vMAE_H{h}"
                    key = (ds_disp, display, f"H{h}")
                    out.setdefault(key, []).append(float(r[col]))
    return out


def fmt_value(val: float) -> str:
    return f"{val:.3f}"


def style_matrix_by_column(matrix: list[list[float]]) -> list[list[str]]:
    """Bold best and underline second-best in each column across models (lower is better)."""
    n_models = len(matrix)
    n_cols = len(matrix[0])
    out = [[fmt_value(matrix[m][c]) for c in range(n_cols)] for m in range(n_models)]

    for c in range(n_cols):
        col_vals = [matrix[m][c] for m in range(n_models)]
        order = sorted(range(n_models), key=lambda m: col_vals[m])
        best, second = order[0], order[1]
        out[best][c] = f"\\textbf{{{out[best][c]}}}"
        if abs(col_vals[best] - col_vals[second]) > 1e-9:
            out[second][c] = f"\\underline{{{out[second][c]}}}"

    return out


GROUP_BLOCK_PAD = "0.4em"  # symmetric vertical pad before DCRNN / after Trafformer per dataset block


def write_table2() -> None:
    wide = load_paper_vmae()
    pad = GROUP_BLOCK_PAD
    lines = [
        "% Auto-generated by scripts/generate_multiseed_tables_figures.py — do not edit by hand.",
        "% Channel-averaged vMAE_a mean over mask seeds 43--45; paper grid 0--100% at 10% steps.",
        "",
        "  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ccccccccccccc@{}}",
        "  \\hline",
        "  Dataset   & Model          & 0\\%    & 10\\%   & 20\\%   & 30\\%   & 40\\%   & 50\\%   & 60\\%   & 70\\%   & 80\\%   & 90\\%   & 100\\% \\\\",
        "  \\hline",
        f"  \\noalign{{\\vskip {pad}}}",
    ]

    for ds_idx, ds_disp in enumerate(("PEMS04", "PEMS08")):
        matrix: list[list[float]] = []
        for display, _ in MODELS:
            row_df = wide[(wide["Dataset"] == ds_disp) & (wide["Model"] == display)]
            if row_df.empty:
                raise RuntimeError(f"Missing Table II row for {ds_disp} {display}")
            r = row_df.iloc[0]
            matrix.append([float(r[col]) for col in SPARSITY_PCT_COLS])
        styled_rows = style_matrix_by_column(matrix)

        for idx, (display, _) in enumerate(MODELS):
            styled = styled_rows[idx]
            if idx == 0:
                lines.append(
                    f"  \\multirow{{4}}{{*}}{{{ds_disp}}} & {display:<14} & {' & '.join(styled)} \\\\"
                )
            else:
                lines.append(f"  & {display:<14} & {' & '.join(styled)} \\\\")
        lines.append(f"  \\noalign{{\\vskip {pad}}}")
        lines.append("  \\hline")
        if ds_idx == 0:
            lines.append(f"  \\noalign{{\\vskip {pad}}}")

    lines.extend(["  \\end{tabular*}", ""])
    OUT_TABLE2.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE2.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_TABLE2}")


def write_table3() -> None:
    horizon = load_horizon_at_10pct()
    pad = GROUP_BLOCK_PAD
    lines = [
        "% Auto-generated by scripts/generate_multiseed_tables_figures.py — do not edit by hand.",
        "% Channel-averaged vMAE at 10% sparsity by horizon; mean over mask seeds 43--45.",
        "",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l l cccccccccccc@{}}",
        "\\hline",
        "Dataset & Model & H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10 & H11 & H12 \\\\",
        "\\hline",
        f"\\noalign{{\\vskip {pad}}}",
    ]

    for ds_idx, ds_disp in enumerate(("PEMS04", "PEMS08")):
        matrix: list[list[float]] = []
        for display, _ in MODELS:
            matrix.append(
                [
                    statistics.mean(horizon[(ds_disp, display, f"H{h}")])
                    for h in range(1, 13)
                ]
            )
        styled_rows = style_matrix_by_column(matrix)

        for idx, (display, _) in enumerate(MODELS):
            styled = styled_rows[idx]
            if idx == 0:
                lines.append(
                    f"\\multirow{{4}}{{*}}{{{ds_disp}}} & {display:<14} & {' & '.join(styled)} \\\\"
                )
            else:
                lines.append(f"& {display:<14} & {' & '.join(styled)} \\\\")
        lines.append(f"\\noalign{{\\vskip {pad}}}")
        lines.append("\\hline")
        if ds_idx == 0:
            lines.append(f"\\noalign{{\\vskip {pad}}}")

    lines.extend(["\\end{tabular*}", ""])
    OUT_TABLE3.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE3.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_TABLE3}")


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


def plot_operational_breakdown(df: pd.DataFrame, ds_key: str, ds_disp: str) -> None:
    thresh = THRESHOLD[ds_key]
    fig, ax = plt.subplots(figsize=(12, 8))
    sub = df[df["dataset"] == ds_key].copy()
    crossings: list[tuple[str, float]] = []

    for display, model_key in MODELS:
        mdf = sub[sub["model"] == model_key].sort_values("sparsity")
        if mdf.empty:
            continue
        xs = mdf["sparsity"].values
        ys = mdf["vMAE_a"].values
        cross = _interpolate_crossover(xs, ys, thresh)
        if cross is not None:
            crossings.append((display, cross * 100))

        x_pct = mdf["sparsity"] * 100
        if "vMAE_a_std" in mdf.columns and mdf["vMAE_a_std"].max() > 0:
            ax.fill_between(
                x_pct,
                mdf["vMAE_a"] - mdf["vMAE_a_std"],
                mdf["vMAE_a"] + mdf["vMAE_a_std"],
                color=MODEL_COLORS[display],
                alpha=0.15,
                linewidth=0,
                zorder=2,
            )

        ax.plot(
            x_pct,
            mdf["vMAE_a"],
            marker=MODEL_MARKERS[display],
            linewidth=3,
            markersize=10,
            label=DISPLAY[display],
            color=MODEL_COLORS[display],
            zorder=3,
        )

    ax.axhline(thresh, color="red", linestyle="--", linewidth=2, alpha=0.8)
    ax.text(
        36,
        thresh - 0.03,
        rf"10% Error Threshold ($vMAE_a$={thresh:.2f})",
        fontsize=12,
        color="red",
        ha="right",
        va="top",
    )

    if ds_key == "PEMSD8":
        # D2STGNN and Trafformer both cross near 11--12%. D2STGNN is lifted
        # above the threshold so the two % labels do not overlap.
        offsets = {
            "Trafformer": (0, -30),
            "D2STGNN": (20, 24),
            "Mamba4Traffic": (8, 20),
            "DCRNN": (-22, 22),
        }
    else:
        # Trafformer, D2STGNN, and DCRNN all cross near 4--6%.
        offsets = {
            "DCRNN": (32, 20),
            "D2STGNN": (30, -32),
            "Mamba4Traffic": (0, 18),
            "Trafformer": (-28, -32),
        }

    for display, cross_s in crossings:
        off = offsets.get(display, (0, 18))
        ax.plot(cross_s, thresh, "+", markersize=15, markeredgewidth=3, color=MODEL_COLORS[display])
        ax.annotate(
            f"{cross_s:.0f}%",
            (cross_s, thresh),
            textcoords="offset points",
            xytext=off,
            fontsize=16,
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=MODEL_COLORS[display],
                linewidth=2,
            ),
        )

    ax.set_xlabel("Sensor Sparsity (%)", fontsize=20)
    ax.set_ylabel(r"$vMAE_a$", fontsize=20)
    ax.set_title(f"{ds_disp}: $vMAE_a$ Degradation (Finer Granularity)", fontsize=24, fontweight="bold")
    ax.legend(loc="upper left", fontsize=16)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(-1, 37)
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    tag = "pems04" if ds_key == "PEMSD4" else "pems08"
    out = OUT_FIGURES / f"fig1{'a' if ds_key == 'PEMSD4' else 'b'}_operational_breakdown_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_ranking_matrix(df: pd.DataFrame, ds_key: str, ds_disp: str) -> None:
    sub = df[df["dataset"] == ds_key].copy()
    levels = sorted(sub["sparsity"].unique())
    common = [lvl for lvl in levels if round(lvl * 100) % 10 == 0]
    sub = sub[sub["sparsity"].isin(common)]
    pivot = sub.pivot_table(index="model", columns="sparsity", values="vMAE_a")
    pivot.index = [MODEL_KEY_TO_DISPLAY[m] for m in pivot.index]
    ranks = pivot.rank(axis=0, method="min")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        ranks,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        cbar=False,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 18, "weight": "bold"},
    )
    ax.set_xlabel("Sparsity Level", fontsize=20)
    ax.set_ylabel("Model", fontsize=20)
    ax.set_title(
        f"{ds_disp}: Model Ranking by $vMAE_a$ at Each Sparsity Level",
        fontsize=24,
        fontweight="bold",
    )
    ax.set_xticklabels([f"{int(float(x.get_text()) * 100)}%" for x in ax.get_xticklabels()], rotation=0, fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, rotation=0)
    fig.tight_layout()
    tag = "pems04" if ds_key == "PEMSD4" else "pems08"
    out = OUT_FIGURES / f"fig2{'a' if ds_key == 'PEMSD4' else 'b'}_ranking_matrix_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_percentage_increase(df: pd.DataFrame, ds_key: str, ds_disp: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    sub = df[df["dataset"] == ds_key].copy()

    for display, model_key in MODELS:
        mdf = sub[sub["model"] == model_key].sort_values("sparsity")
        if mdf.empty or not (mdf["sparsity"] == 0.0).any():
            continue
        baseline = float(mdf.loc[mdf["sparsity"] == 0.0, "vMAE_a"].iloc[0])
        relative = mdf["vMAE_a"] / baseline
        x_pct = mdf["sparsity"] * 100

        if "vMAE_a_std" in mdf.columns and mdf["vMAE_a_std"].max() > 0:
            rel_std = mdf["vMAE_a_std"] / baseline
            ax.fill_between(
                x_pct,
                relative - rel_std,
                relative + rel_std,
                color=MODEL_COLORS[display],
                alpha=0.15,
                linewidth=0,
                zorder=2,
            )

        ax.plot(
            x_pct,
            relative,
            marker=MODEL_MARKERS[display],
            linewidth=3,
            markersize=10,
            label=DISPLAY[display],
            color=MODEL_COLORS[display],
            zorder=3,
        )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=2)
    ax.set_xlabel("Sensor Sparsity (%)", fontsize=20)
    ax.set_ylabel(
        r"Degradation $D(\rho) = \frac{vMAE_a(\rho)}{vMAE_a(0)}$",
        fontsize=20,
    )
    ax.set_title(f"{ds_disp}: Relative Degradation Curves", fontsize=24, fontweight="bold")
    ax.legend(loc="upper left", fontsize=16)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(-1, 37)
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    tag = "pems04" if ds_key == "PEMSD4" else "pems08"
    out = OUT_FIGURES / f"fig5{'a' if ds_key == 'PEMSD4' else 'b'}_percentage_increase_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def load_table1_avg_bar_d_mae() -> dict[tuple[str, str], tuple[float, float]]:
    """Avg bar_D_MAE taken from the Table I sidecar CSV (same numbers as the paper table).

    Returns {(dataset_key, display_model): (mean_over_seeds, sample_std)}.
    """
    path = CACHE / "tables" / "multiseed_new_table1_breakdown035_weighted_degradation_mean_std.csv"
    df = pd.read_csv(path)
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in df[df["metric_row"] == "bar_D_MAE"].iterrows():
        display = str(row["Model"])
        out[("PEMSD4", display)] = (float(row["PEMS04_Avg"]), float(row["PEMS04_Avg_std"]))
        out[("PEMSD8", display)] = (float(row["PEMS08_Avg"]), float(row["PEMS08_Avg_std"]))
    missing = [
        (ds, display)
        for ds, _ in DATASETS
        for display, _ in MODELS
        if (ds, display) not in out
    ]
    if missing:
        raise RuntimeError(f"Missing Table I bar_D_MAE rows: {missing}")
    return out


def plot_accuracy_robustness(
    df: pd.DataFrame,
    ds_key: str,
    ds_disp: str,
    *,
    with_mask_std: bool = False,
    with_trendline: bool = False,
) -> None:
    # Y-axis: Table I Avg bar_D_MAE (channel-mean of per-channel weighted D).
    # X-axis: channel-averaged vMAE_a at rho=0 (same baseline as Table II / Fig. 6 caption).
    # Optional vertical bars: sample std of Avg bar_D_MAE over mask seeds 43--45.
    table1_d = load_table1_avg_bar_d_mae()
    sub = df[(df["dataset"] == ds_key) & (df["sparsity"] <= 0.35 + 1e-9)].copy()
    results = []

    for display, model_key in MODELS:
        mdf = sub[sub["model"] == model_key].sort_values("sparsity")
        if mdf.empty or not (mdf["sparsity"] == 0.0).any():
            continue
        baseline = float(mdf.loc[mdf["sparsity"] == 0.0, "vMAE_a"].iloc[0])
        wdeg, wdeg_std = table1_d[(ds_key, display)]
        results.append(
            {
                "display": display,
                "baseline": baseline,
                "avg_deg": wdeg,
                "avg_deg_std": wdeg_std,
                "color": MODEL_COLORS[display],
            }
        )

    fig, ax = plt.subplots(figsize=(12, 9))
    for r in results:
        if with_mask_std and r["avg_deg_std"] > 0:
            ax.errorbar(
                r["baseline"],
                r["avg_deg"],
                yerr=r["avg_deg_std"],
                fmt="none",
                ecolor=r["color"],
                elinewidth=2.5,
                capsize=8,
                capthick=2.5,
                zorder=4,
                alpha=0.9,
            )
        ax.scatter(
            r["baseline"],
            r["avg_deg"],
            s=500,
            c=r["color"],
            edgecolors="black",
            linewidths=2,
            label=r["display"],
            alpha=0.85,
            zorder=5,
        )

    label_offsets = {
        "DCRNN": (60, 0),
        "D2STGNN": (60, 0),
        "Mamba4Traffic": (60, 0),
        "Trafformer": (-70, 0),
    }
    for r in results:
        off = label_offsets.get(r["display"], (60, 0))
        ax.annotate(
            r["display"],
            xy=(r["baseline"], r["avg_deg"]),
            xytext=off,
            textcoords="offset points",
            fontsize=16,
            fontweight="bold",
            ha="left" if off[0] > 0 else "right",
            va="center",
            arrowprops=dict(arrowstyle="->", color="gray", lw=2),
            alpha=0.95,
        )

    if with_trendline and len(results) >= 2:
        xs = np.array([r["baseline"] for r in results], dtype=float)
        ys = np.array([r["avg_deg"] for r in results], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        ss_res = float(np.sum((ys - (slope * xs + intercept)) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        x_line = np.linspace(float(np.min(xs)), float(np.max(xs)), 50)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#444444",
            linestyle="--",
            linewidth=2,
            zorder=1,
            label=rf"OLS fit ($R^2={r2:.2f}$)",
        )

    ax.set_xlabel(
        r"Baseline $vMAE_a$ (at 0% sparsity)" + "\n" + r"$\leftarrow$ Better Accuracy (lower)",
        fontsize=20,
    )
    ax.set_ylabel(
        r"Weighted Degradation $\bar{D}_{MAE}$" + "\n" + r"$\leftarrow$ More Robust (lower)",
        fontsize=20,
    )
    title = f"{ds_disp}: Accuracy vs Robustness Tradeoff"
    if with_mask_std:
        title += r" ($\pm 1\sigma$ over masks)"
    ax.set_title(title, fontsize=22 if with_mask_std else 24, fontweight="bold")
    ax.legend(loc="upper right", fontsize=16)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    tag = "pems04" if ds_key == "PEMSD4" else "pems08"
    letter = "a" if ds_key == "PEMSD4" else "b"
    # Default manuscript files include mask std bars; _nobars is the alternate.
    # Trendline variants are extra files and do not replace the paper Fig. 6.
    suffix = "" if with_mask_std else "_nobars"
    if with_trendline:
        suffix = f"{suffix}_trendline" if suffix else "_trendline"
    out = OUT_FIGURES / f"fig6{letter}_accuracy_robustness_{tag}{suffix}.pdf"
    png = OUT_FIGURES / f"fig6{letter}_accuracy_robustness_{tag}{suffix}.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def write_figures(
    *,
    only_fig6: bool = False,
    only_fig1: bool = False,
    fig6_trendline: bool = False,
) -> None:
    breakdown = load_breakdown_vmae()
    paper = pd.read_csv(CACHE / "figures" / "multiseed_new_paper_vmae_a_mean.csv")
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    for ds_key, ds_disp in DATASETS:
        if only_fig1:
            plot_operational_breakdown(breakdown, ds_key, ds_disp)
            continue
        if not only_fig6:
            plot_operational_breakdown(breakdown, ds_key, ds_disp)
            plot_ranking_matrix(paper, ds_key, ds_disp)
            plot_percentage_increase(breakdown, ds_key, ds_disp)
        plot_accuracy_robustness(breakdown, ds_key, ds_disp, with_mask_std=True)
        plot_accuracy_robustness(breakdown, ds_key, ds_disp, with_mask_std=False)
        if fig6_trendline:
            plot_accuracy_robustness(
                breakdown, ds_key, ds_disp, with_mask_std=True, with_trendline=True
            )
            plot_accuracy_robustness(
                breakdown, ds_key, ds_disp, with_mask_std=False, with_trendline=True
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-fig6",
        action="store_true",
        help="Regenerate only accuracy-robustness scatter plots (Fig. 6).",
    )
    parser.add_argument(
        "--only-fig1",
        action="store_true",
        help="Regenerate only operational-breakdown plots (Fig. 1).",
    )
    parser.add_argument(
        "--fig6-trendline",
        action="store_true",
        help="Also write Fig. 6 variants with an OLS trendline and R^2 (extra files).",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip Table II / Table III regeneration.",
    )
    args = parser.parse_args()
    if not args.skip_tables and not args.only_fig6 and not args.only_fig1:
        write_table2()
        write_table3()
    write_figures(
        only_fig6=args.only_fig6,
        only_fig1=args.only_fig1,
        fig6_trendline=args.fig6_trendline,
    )
    print("Done.")


if __name__ == "__main__":
    main()
