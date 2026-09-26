#!/usr/bin/env python3
"""Redraw the paper weighting-function figure with alpha 10, 20, and 40."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PAPER = Path(os.environ.get("SPARSITY_PAPER_DIR", str(REPO / "reproduced" / "manuscript"))).resolve()
OUT = PAPER / "figures" / "weighting_function.pdf"

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
    }
)


def main() -> None:
    rho = np.linspace(0.0, 1.0, 400)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    specs = [
        (10, "#c47a00", "--", 1.6, r"$\alpha=10$"),
        (20, "#1f4e9b", "-", 2.15, r"$\alpha=20$ (default)"),
        (40, "#6b3a2a", ":", 1.7, r"$\alpha=40$"),
    ]
    for alpha, color, ls, lw, label in specs:
        ax.plot(rho, np.exp(-alpha * rho**2), color=color, linestyle=ls, linewidth=lw, label=label)
    ax.axvline(0.10, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7, zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Sparsity level ($\rho$)")
    ax.set_ylabel(r"Weight $w(\rho)$")
    ax.set_title(r"Gaussian weights $w(\rho)=e^{-\alpha\rho^2}$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
