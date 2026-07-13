#!/usr/bin/env python3
"""
Generate exactly the sparsity figures referenced in the paper LaTeX.

Paper figure set (journal, results_section.tex):
  - fig1_{PEMSD4,PEMSD8}_weighted_robustness.pdf   (OME scatter)
  - fig2_{PEMSD4,PEMSD8}_ranking_heatmap.pdf
  - fig3_{PEMSD4,PEMSD8}_degradation_curves.pdf

Conference reduced version omits fig2 (ranking heatmap).

Data source: sparsity_analysis/results_cache/ (paper_original or multiseed_new).

This is intentionally separate from scripts/generate_paper_figures.py, which
produces extra figures (normalized degradation, scatter/bars variants) under
different filenames.

Usage:
  python scripts/generate_sparsity_paper_figures.py --lineage paper_original
  python scripts/generate_sparsity_paper_figures.py --lineage multiseed_new --show-std
  python scripts/generate_sparsity_paper_figures.py --lineage both --paper-set conference
"""

from __future__ import annotations

import argparse
import os

from sparsity_paper_figures_lib import (
    DEFAULT_CACHE_DIR,
    ROOT_DIR,
    generate_figure_set,
    load_from_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper sparsity figures from results_cache.")
    parser.add_argument(
        "--lineage",
        choices=["paper_original", "multiseed_new", "both"],
        default="both",
        help="Which result lineage to plot (default: both).",
    )
    parser.add_argument(
        "--paper-set",
        choices=["journal", "conference"],
        default="journal",
        help="journal = 3 figure types; conference = fig1 + fig3 only.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Path to sparsity_analysis/results_cache.",
    )
    parser.add_argument(
        "--out-root",
        default=os.path.join(ROOT_DIR, "sparsity_analysis", "figures"),
        help="Root output directory; figures go to {out-root}/{lineage}/.",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Draw ±1 std bands on fig3 (multiseed_new only; ignored for paper_original).",
    )
    args = parser.parse_args()

    lineages = ["paper_original", "multiseed_new"] if args.lineage == "both" else [args.lineage]

    print("=" * 60)
    print("Sparsity paper figures (LaTeX filenames)")
    print(f"  paper set: {args.paper_set}")
    print(f"  cache:     {args.cache_dir}")
    print("=" * 60)

    all_saved = []
    for lineage in lineages:
        df = load_from_cache(args.cache_dir, lineage)
        out_dir = os.path.join(args.out_root, lineage)
        suffix = "paper original (seed 42)" if lineage == "paper_original" else "multiseed mean (43–45)"
        show_std = args.show_std and lineage == "multiseed_new"
        print(f"\n{lineage}: {len(df)} rows → {out_dir}")
        saved = generate_figure_set(
            df,
            out_dir,
            paper_set=args.paper_set,
            title_suffix=suffix,
            show_std=show_std,
        )
        for path in saved:
            print(f"  Saved: {path}")
        all_saved.extend(saved)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(all_saved)} PDF(s) under {args.out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
