#!/usr/bin/env python3
"""Render Figure 6 with discrete alpha markers and mask-seed whiskers.

Triangle, haloed dot, and square encode alpha 10, 20, and 40.  There is no
line between the sensitivity markers, which avoids suggesting that each model
can move independently anywhere inside an interval.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.lines import Line2D
import numpy as np

from reproduction import generate_fig6_joint_uncertainty as base


def draw_discrete_alpha(ax: plt.Axes, summary, dataset: str) -> None:
    for model in base.MODELS:
        stats = base.model_stats(summary, dataset, model)
        color = base.COLORS[model]

        # Secondary sensitivity states.  They are deliberately disconnected.
        ax.scatter(
            stats["x"],
            stats["alpha10"],
            s=32,
            marker="^",
            color=color,
            edgecolors="none",
            alpha=0.34,
            zorder=3,
        )
        ax.scatter(
            stats["x"],
            stats["alpha40"],
            s=27,
            marker="s",
            color=color,
            edgecolors="none",
            alpha=0.34,
            zorder=3,
        )

        # Primary alpha=20 estimate.  A translucent model-color halo sits
        # behind the seed whisker and an opaque model-color center dot.
        ax.scatter(
            stats["x"],
            stats["current"],
            s=94,
            marker="o",
            color=color,
            edgecolors="none",
            alpha=0.38,
            zorder=5,
        )
        ax.errorbar(
            stats["x"],
            stats["current"],
            yerr=stats["sd"],
            fmt="none",
            ecolor="black",
            elinewidth=1.35,
            capsize=5,
            capthick=1.45,
            zorder=7,
        )
        ax.scatter(
            stats["x"],
            stats["current"],
            s=17,
            marker="o",
            color=color,
            edgecolors="none",
            alpha=1.0,
            zorder=8,
        )
        base.label_point(ax, dataset, model, stats["x"], stats["current"])


def alpha20_points(summary, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([base.model_stats(summary, dataset, model)["x"] for model in base.MODELS])
    ys = np.array([base.model_stats(summary, dataset, model)["current"] for model in base.MODELS])
    return xs, ys


def draw_trendline(ax: plt.Axes, summary, dataset: str) -> dict[str, float]:
    """Linear fit through the four alpha=20 means."""
    xs, ys = alpha20_points(summary, dataset)
    slope, intercept = np.polyfit(xs, ys, 1)
    ss_res = float(np.sum((ys - (slope * xs + intercept)) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(xs, ys)[0, 1])
    x_line = np.linspace(float(np.min(xs)), float(np.max(xs)), 50)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="#4a4a4a",
        linestyle="--",
        linewidth=1.35,
        alpha=0.32,
        zorder=1,
        solid_capstyle="round",
    )
    return {"r": r, "r2": r2}


def legend_handles(ax: plt.Axes) -> list:
    h10 = Line2D(
        [0],
        [0],
        marker="^",
        markersize=6,
        color="none",
        markerfacecolor="#777777",
        markeredgewidth=0,
        alpha=0.42,
        label=r"$\alpha=10$ mean",
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    eb = ax.errorbar(
        [xlim[0] - 10.0],
        [ylim[0] - 10.0],
        yerr=[[2.0], [2.0]],
        fmt="o",
        color="black",
        markerfacecolor="#888888",
        markeredgecolor="black",
        markersize=7,
        elinewidth=1.4,
        capsize=5,
        capthick=1.45,
        zorder=0,
    )
    eb.set_label(r"$\alpha=20$ mean $\pm1\sigma$ masks")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    h40 = Line2D(
        [0],
        [0],
        marker="s",
        markersize=5.5,
        color="none",
        markerfacecolor="#777777",
        markeredgewidth=0,
        alpha=0.42,
        label=r"$\alpha=40$ mean",
    )
    hfit = Line2D(
        [0],
        [0],
        color="#4a4a4a",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="linear fit",
    )
    return [h10, eb, h40, hfit]


def legend_kwargs() -> dict:
    return {
        "handler_map": {ErrorbarContainer: HandlerErrorbar(yerr_size=0.95)},
        "handleheight": 2.1,
        "borderpad": 0.55,
        "frameon": False,
    }


def save_figure(fig: plt.Figure, stems: list[str]) -> None:
    for stem in stems:
        fig.savefig(base.FIGURES / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(base.FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")


def make_combined(summary) -> dict[str, dict[str, float]]:
    stats_out: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.45))
    for ax, dataset in zip(axes, base.DATASETS):
        stats_out[dataset] = draw_trendline(ax, summary, dataset)
        draw_discrete_alpha(ax, summary, dataset)
        base.style_axis(ax, dataset)
    axes[0].set_ylabel(r"Weighted degradation $\bar{D}_{MAE}$ (lower is better)")
    fig.tight_layout(rect=(0, 0.10, 1, 1.0))
    fig.legend(
        handles=legend_handles(axes[0]),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=8.0,
        **legend_kwargs(),
    )
    save_figure(
        fig,
        [
            "fig6_joint_discrete_alpha_recommended",
            "fig6_joint_discrete_alpha_recommended_trendline",
            "fig6_joint_nested_recommended",
        ],
    )
    plt.close(fig)
    return stats_out


def make_dropins(summary) -> None:
    specs = [("PEMS04", "a", "pems04"), ("PEMS08", "b", "pems08")]
    for dataset, letter, tag in specs:
        fig, ax = plt.subplots(figsize=(12, 9))
        draw_trendline(ax, summary, dataset)
        draw_discrete_alpha(ax, summary, dataset)
        base.style_axis(ax, dataset)
        ax.set_ylabel(
            r"Weighted Degradation $\bar{D}_{MAE}$" + "\n" + r"$\leftarrow$ More Robust (lower)",
            fontsize=18,
        )
        ax.set_xlabel(
            r"Baseline $vMAE_a$ (at 0% sparsity)" + "\n" + r"$\leftarrow$ Better Accuracy (lower)",
            fontsize=18,
        )
        ax.set_title(f"{dataset} Accuracy vs Robustness Tradeoff", fontsize=21, fontweight="bold")
        ax.tick_params(labelsize=15)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        ax.legend(
            handles=legend_handles(ax),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            fontsize=13,
            frameon=False,
            **{k: v for k, v in legend_kwargs().items() if k != "frameon"},
        )
        stem = f"fig6{letter}_accuracy_robustness_{tag}_alpha_envelope"
        save_figure(fig, [stem, f"{stem}_discrete"])
        plt.close(fig)


def main() -> None:
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    _, summary = base.calculate(base.load_data())
    fit_stats = make_combined(summary)
    make_dropins(summary)
    for dataset, stats in fit_stats.items():
        print(f"{dataset}: Pearson r={stats['r']:.3f}, linear-fit R^2={stats['r2']:.2f}")
    print("Wrote discrete-alpha Figure 6 with trendline as default")


if __name__ == "__main__":
    main()
