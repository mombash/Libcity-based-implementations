#!/usr/bin/env python3
"""
Side-by-side comparison of paper sparsity figures: paper_original vs multiseed_new.

For each figure type and dataset, renders a two-panel PDF:
  left  = paper_original (seed 42, old checkpoints)
  right = multiseed_new (mean over mask seeds 43–45, new checkpoints)

Also writes a summary grid PDF with all comparisons on one page per figure type.

Usage:
  python scripts/compare_sparsity_paper_figures.py
  python scripts/compare_sparsity_paper_figures.py --paper-set conference
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt

from sparsity_paper_figures_lib import (
    DATASETS,
    DEFAULT_CACHE_DIR,
    ROOT_DIR,
    load_from_cache,
    plot_degradation_curves,
    plot_ranking_heatmap,
    plot_weighted_robustness,
)


def _render_panel(ax, plot_fn, df, dataset, *, title: str, show_std: bool = False) -> None:
    """Render a figure into an existing axes by temporarily redirecting pyplot state."""
    # Use a throwaway figure to get the artist layout, then copy to target axes.
    # Simpler: clear ax and call plot logic inline via dedicated axis-aware helpers.
    # For maintainability we re-open saved approach: plot to temp path is heavy;
    # instead draw directly on ax by duplicating minimal layout here.

    # Delegate: each plot function creates its own figure. We use a wrapper that
    # plots on a provided axes by importing lower-level code paths.
    ax.clear()
    ax.set_title(title, fontsize=12, fontweight="bold")

    if plot_fn is plot_weighted_robustness:
        _weighted_on_ax(ax, df, dataset)
    elif plot_fn is plot_ranking_heatmap:
        _heatmap_on_ax(ax, df, dataset)
    elif plot_fn is plot_degradation_curves:
        _curves_on_ax(ax, df, dataset, show_std=show_std)


def _weighted_on_ax(ax, df, dataset):
    from sparsity_paper_figures_lib import MODELS, MODEL_INFO
    import matplotlib.ticker as mtick

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
                "display": MODEL_INFO[model]["display"],
                "baseline": v0,
                "ome": ome,
                "color": MODEL_INFO[model]["color"],
            }
        )

    for s in scatter:
        ax.scatter(s["baseline"], s["ome"], c=s["color"], s=80, edgecolors="black", linewidth=0.6, zorder=5)
        ax.annotate(s["display"], (s["baseline"], s["ome"]), fontsize=7, fontweight="bold", xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Baseline vMAE", fontsize=10)
    ax.set_ylabel("OME (×)", fontsize=10)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f×"))
    ax.grid(True, alpha=0.3)


def _heatmap_on_ax(ax, df, dataset):
    from sparsity_paper_figures_lib import MODEL_INFO
    import numpy as np
    import seaborn as sns

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

    sns.heatmap(
        pivot,
        annot=np.array(annot),
        fmt="",
        cmap="Reds",
        linewidths=0.5,
        linecolor="#eeeeee",
        annot_kws={"fontsize": 7, "fontweight": "bold", "color": "black"},
        cbar_kws={"label": "vMAE"},
        ax=ax,
    )
    ax.set_xticklabels([f"{int(float(x.get_text()) * 100)}%" for x in ax.get_xticklabels()], fontsize=8, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
    ax.set_xlabel("Sparsity", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)


def _curves_on_ax(ax, df, dataset, *, show_std: bool):
    from sparsity_paper_figures_lib import MODELS, MODEL_INFO, THRESHOLD, _interpolate_crossover

    thresh = THRESHOLD[dataset]
    for model in MODELS:
        mdf = df[(df["model"] == model) & (df["dataset"] == dataset)].sort_values("sparsity")
        if mdf.empty:
            continue
        info = MODEL_INFO[model]
        xs = mdf["sparsity"].values
        ys = mdf["vMAE_a"].values
        cross = _interpolate_crossover(xs, ys, thresh)
        if cross is not None:
            label = f"{info['display']} ({cross * 100:.0f}%)"
        elif ys.min() > thresh:
            label = f"{info['display']} (0%)"
        else:
            label = info["display"]
        x_pct = mdf["sparsity"] * 100
        ax.plot(x_pct, mdf["vMAE_a"], marker=info["marker"], color=info["color"], linewidth=1.5, markersize=4, label=label)
        if show_std and "vMAE_a_std" in mdf.columns and mdf["vMAE_a_std"].max() > 0:
            ax.fill_between(
                x_pct,
                mdf["vMAE_a"] - mdf["vMAE_a_std"],
                mdf["vMAE_a"] + mdf["vMAE_a_std"],
                color=info["color"],
                alpha=0.15,
                linewidth=0,
            )

    ax.axhline(thresh, color="#a0522d", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Sparsity (%)", fontsize=10)
    ax.set_ylabel("vMAE", fontsize=10)
    ax.legend(loc="upper left", fontsize=6, frameon=True)
    ax.set_xlim(-2, 102)
    ax.grid(True, alpha=0.3)


COMPARISONS = [
    ("fig1_weighted_robustness", plot_weighted_robustness, "Fig 1: Weighted robustness (OME)"),
    ("fig2_ranking_heatmap", plot_ranking_heatmap, "Fig 2: Ranking heatmap"),
    ("fig3_degradation_curves", plot_degradation_curves, "Fig 3: Degradation curves"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side sparsity figure comparison.")
    parser.add_argument("--paper-set", choices=["journal", "conference"], default="journal")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(ROOT_DIR, "sparsity_analysis", "figures", "comparison"),
    )
    args = parser.parse_args()

    df_orig = load_from_cache(args.cache_dir, "paper_original")
    df_new = load_from_cache(args.cache_dir, "multiseed_new")

    comparisons = COMPARISONS if args.paper_set == "journal" else COMPARISONS[:1] + COMPARISONS[2:]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Sparsity figure comparison (paper_original vs multiseed_new)")
    print("=" * 60)

    saved = []
    for dataset in DATASETS:
        ds = "PEMS04" if dataset == "PEMSD4" else "PEMS08"
        for slug, plot_fn, desc in comparisons:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"{desc} — {ds}", fontsize=14, fontweight="bold")

            _render_panel(
                axes[0],
                plot_fn,
                df_orig,
                dataset,
                title="Paper original (seed 42)",
            )
            _render_panel(
                axes[1],
                plot_fn,
                df_new,
                dataset,
                title="Multiseed new (mean 43–45)",
                show_std=(plot_fn is plot_degradation_curves),
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = os.path.join(args.out_dir, f"compare_{slug}_{dataset}.pdf")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}")
            saved.append(out_path)

    # Summary grid: one row per figure type, columns = PEMS04 left pair + PEMS08 right pair
    if args.paper_set == "journal":
        n_rows = len(comparisons)
        fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row, (slug, plot_fn, desc) in enumerate(comparisons):
            for col_ds, dataset in enumerate(DATASETS):
                ds = "PEMS04" if dataset == "PEMSD4" else "PEMS08"
                _render_panel(axes[row, col_ds * 2], plot_fn, df_orig, dataset, title=f"{ds} orig")
                _render_panel(
                    axes[row, col_ds * 2 + 1],
                    plot_fn,
                    df_new,
                    dataset,
                    title=f"{ds} new",
                    show_std=(plot_fn is plot_degradation_curves),
                )
            axes[row, 0].set_ylabel(desc, fontsize=10, fontweight="bold")

        fig.suptitle("Sparsity figures: paper original (left) vs multiseed new (right)", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        grid_path = os.path.join(args.out_dir, "compare_all_figures_grid.pdf")
        fig.savefig(grid_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {grid_path}")
        saved.append(grid_path)

    print(f"\nGenerated {len(saved)} comparison PDF(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
