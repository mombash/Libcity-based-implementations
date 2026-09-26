#!/usr/bin/env python3
"""Rebuild the submission tables and figures from the released caches."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = REPO if (REPO / "evaluation" / "cache").is_dir() else REPO / "artifacts" / "paper-release"
ARTIFACT_ROOT = Path(
    os.environ.get("SPARSITY_ARTIFACT_DIR", DEFAULT_ARTIFACT_ROOT)
).resolve()


def reproduce_manuscript() -> None:
    cache = ARTIFACT_ROOT / "evaluation" / "cache" / "manuscript"
    output = Path(
        os.environ.get("SPARSITY_OUTPUT_DIR", REPO / "reproduced" / "manuscript")
    ).resolve()
    os.environ["SPARSITY_RESULTS_CACHE"] = str(cache)
    os.environ["SPARSITY_PAPER_DIR"] = str(output)

    from reproduction import generate_fig6_joint_uncertainty as fig6_base
    from reproduction import generate_fig6_nested as fig6
    from reproduction import generate_manuscript as manuscript
    from reproduction import generate_table1 as table1
    from reproduction import generate_weighting_function as weighting

    table1.main()
    manuscript.write_table2()
    manuscript.write_table3()

    breakdown = manuscript.load_breakdown_vmae()
    paper = manuscript.pd.read_csv(
        cache / "figures" / "multiseed_new_paper_vmae_a_mean.csv"
    )
    manuscript.OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    for dataset, display in manuscript.DATASETS:
        manuscript.plot_operational_breakdown(breakdown, dataset, display)
        manuscript.plot_ranking_matrix(paper, dataset, display)
        manuscript.plot_percentage_increase(breakdown, dataset, display)

    weighting.main()
    _, summary = fig6_base.calculate(fig6_base.load_data())
    fig6.make_combined(summary)
    recommended = fig6_base.FIGURES / "fig6_joint_discrete_alpha_recommended.pdf"
    canonical = fig6_base.FIGURES / "fig6_accuracy_robustness_joint.pdf"
    shutil.copy2(recommended, canonical)

    expected = [
        output / "table1" / "tabular_multiseed_vmae_rmse.tex",
        output / "table2" / "tabular_multiseed_vmae_a.tex",
        output / "table3" / "tabular_multiseed_horizon_10pct.tex",
        output / "figures" / "weighting_function.pdf",
        output / "figures" / "fig1a_operational_breakdown_pems04.pdf",
        output / "figures" / "fig1b_operational_breakdown_pems08.pdf",
        output / "figures" / "fig2a_ranking_matrix_pems04.pdf",
        output / "figures" / "fig2b_ranking_matrix_pems08.pdf",
        output / "figures" / "fig5a_percentage_increase_pems04.pdf",
        output / "figures" / "fig5b_percentage_increase_pems08.pdf",
        canonical,
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError("Missing outputs: " + ", ".join(map(str, missing)))
    print(f"Rebuilt all submission tables and figures in {output}")


def reproduce_channel_breakdown() -> None:
    cache = ARTIFACT_ROOT / "evaluation" / "cache" / "channel_breakdown"
    output = Path(
        os.environ.get(
            "SPARSITY_OUTPUT_DIR", REPO / "reproduced" / "channel_breakdown"
        )
    ).resolve()
    os.environ["SPARSITY_RESULTS_CACHE"] = str(cache)
    os.environ["SPARSITY_PAPER_DIR"] = str(output)

    from reproduction import generate_channel_breakdown as channel_breakdown

    channel_breakdown.main()
    expected = [
        output / f"fig_breakdown_{dataset}_{channel}.pdf"
        for dataset in ("pems04", "pems08")
        for channel in ("flow", "occupancy", "speed")
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError("Missing outputs: " + ", ".join(map(str, missing)))
    print(f"Rebuilt all six channel-breakdown plots in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("manuscript", "channel-breakdown"))
    args = parser.parse_args()
    if args.target == "manuscript":
        reproduce_manuscript()
    else:
        reproduce_channel_breakdown()


if __name__ == "__main__":
    main()
