#!/usr/bin/env python3
"""Build CSV cache of sparsity eval results for paper figures and tables.

Creates ``sparsity_analysis/results_cache/`` with:
  - run_manifest.csv          — source path for every eval run
  - raw/                      — per-run sparsity_results.csv (symlinks)
  - long_form/                — one row per (run × sparsity level)
  - figures/                  — figure-ready vMAE_a series (paper grid + breakdown)
  - tables/                   — wide table2-style exports + baseline table1-style rows

Lineages:
  paper_original  — mask_seed=42 on original paper checkpoints
  multiseed_new   — mask_seeds 43–45 on retrained / current checkpoints (job_plan MODELS)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sparsity.job_plan import (  # noqa: E402
    BREAKDOWN_LABEL,
    MODELS,
    PAPER_LABEL,
    SPARSITY_GRIDS,
)

SPARCITY_CACHE = REPO_ROOT / "sparcity_cache"
OUT_ROOT = REPO_ROOT / "sparsity_analysis" / "results_cache"

PAPER_MASK_SEED = 42
MULTI_MASK_SEEDS = [43, 44, 45]

# Original paper checkpoint timestamps (mask_seed=42 runs in sparcity_cache).
PAPER_ORIGINAL_CKPT: dict[tuple[str, str], str] = {
    ("DCRNN", "PEMSD4"): "20250911_143658",
    ("DCRNN", "PEMSD8"): "20250904_044845",
    ("D2STGNN", "PEMSD4"): "20250911_144426",
    ("D2STGNN", "PEMSD8"): "20251202_210455",
    ("MCSTMambaLST_Ablation", "PEMSD4"): "20250925_214013",
    ("MCSTMambaLST_Ablation", "PEMSD8"): "20250925_203759",
    ("Trafformer", "PEMSD4"): "20251230_051435",
    ("Trafformer", "PEMSD8"): "20250907_211301",
}

FIGURE_MODEL = {
    "DCRNN": "DCRNN",
    "D2STGNN": "D2STGNN",
    "MCSTMambaLST_Ablation": "MCSTMambaLST",
    "Trafformer": "Trafformer",
}

TABLE_DISPLAY = {
    "DCRNN": "DCRNN",
    "D2STGNN": "D2STGNN",
    "MCSTMambaLST_Ablation": "Mamba4Traffic",
    "Trafformer": "Trafformer",
}

DATASET_DISPLAY = {"PEMSD4": "PEMS04", "PEMSD8": "PEMS08"}

KEY_METRICS = [
    "vMAE_a",
    "vRMSE_a",
    "vMAPE_a",
    "vMAE_flow",
    "vMAE_occupancy",
    "vMAE_speed",
    "vRMSE_flow",
    "vRMSE_occupancy",
    "vRMSE_speed",
]

PAPER_GRID_PCT = [f"{int(round(r * 100))}%" for r in SPARSITY_GRIDS[PAPER_LABEL]]
BREAKDOWN_GRID_PCT = [f"{int(round(r * 100))}%" for r in SPARSITY_GRIDS[BREAKDOWN_LABEL]]

# Table I (tab:sparsity_degradation): exp(-20 ρ²) weights over the paper grid.
TABLE1_METRIC_BASES = ("vMAE", "vRMSE")
TABLE1_CHANNELS = ("flow", "occupancy", "speed", "a")
TABLE1_ROW_LABELS = (
    "bar_vMAE",
    "bar_D_MAE",
    "bar_vRMSE",
    "bar_D_RMSE",
)
TABLE1_DATASETS = ("PEMSD4", "PEMSD8")


def new_checkpoint_ts(model: str, dataset: str) -> str:
    name = Path(MODELS[model][dataset]).name
    prefix = f"{model}_{dataset}_"
    if name.startswith(prefix):
        return name[len(prefix) :]
    raise ValueError(f"Unexpected cache dir name {name!r} for {model}/{dataset}")


def grid_label(levels: list[float]) -> str | None:
    rounded = [round(x, 4) for x in levels]
    if rounded == [round(x, 4) for x in SPARSITY_GRIDS[PAPER_LABEL]]:
        return PAPER_LABEL
    if rounded == [round(x, 4) for x in SPARSITY_GRIDS[BREAKDOWN_LABEL]]:
        return BREAKDOWN_LABEL
    return None


def pct_label(sparsity: float, grid: str) -> str:
    if grid == PAPER_LABEL:
        return f"{int(round(sparsity * 100))}%"
    return f"{sparsity * 100:g}%"


def iter_indexed_runs() -> Iterator[dict[str, Any]]:
    for meta_path in SPARCITY_CACHE.glob("*/eval_metadata.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        grid = grid_label(meta["sparsity_levels"])
        if grid is None:
            continue
        csv_path = meta_path.parent / "sparsity_results.csv"
        if not csv_path.is_file():
            continue
        yield {
            "run_dir": meta_path.parent.name,
            "run_path": meta_path.parent,
            "csv_path": csv_path,
            "meta_path": meta_path,
            "model": meta["model"],
            "dataset": meta["dataset"],
            "mask_seed": int(meta["mask_seed"]),
            "grid": grid,
            "checkpoint_ts": meta["original_checkpoint_timestamp"],
            "eval_ts": meta["eval_timestamp"],
            "batch_size": meta.get("batch_size"),
            "model_dir": meta.get("model_dir"),
        }


def pick_run(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["eval_ts"])


def select_lineage_runs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_runs = list(iter_indexed_runs())
    paper: list[dict[str, Any]] = []
    multi: list[dict[str, Any]] = []

    for model in MODELS:
        for dataset in MODELS[model]:
            old_ckpt = PAPER_ORIGINAL_CKPT[(model, dataset)]
            new_ckpt = new_checkpoint_ts(model, dataset)
            for grid in (PAPER_LABEL, BREAKDOWN_LABEL):
                cands = [
                    r
                    for r in all_runs
                    if r["model"] == model
                    and r["dataset"] == dataset
                    and r["grid"] == grid
                    and r["mask_seed"] == PAPER_MASK_SEED
                    and r["checkpoint_ts"] == old_ckpt
                ]
                run = pick_run(cands)
                if run:
                    entry = {**run, "lineage": "paper_original"}
                    paper.append(entry)

                for seed in MULTI_MASK_SEEDS:
                    cands = [
                        r
                        for r in all_runs
                        if r["model"] == model
                        and r["dataset"] == dataset
                        and r["grid"] == grid
                        and r["mask_seed"] == seed
                        and r["checkpoint_ts"] == new_ckpt
                    ]
                    run = pick_run(cands)
                    if run:
                        multi.append({**run, "lineage": "multiseed_new"})

    return paper, multi


def read_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def symlink_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        dst.write_bytes(src.read_bytes())
    else:
        os.symlink(src.resolve(), dst)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_long_form(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for rec in read_results_csv(run["csv_path"]):
            sparsity = float(rec["sparsity"])
            row: dict[str, Any] = {
                "lineage": run["lineage"],
                "model": run["model"],
                "model_figure": FIGURE_MODEL[run["model"]],
                "model_table": TABLE_DISPLAY[run["model"]],
                "dataset": run["dataset"],
                "dataset_display": DATASET_DISPLAY[run["dataset"]],
                "grid": run["grid"],
                "mask_seed": run["mask_seed"],
                "sparsity": sparsity,
                "sparsity_pct": pct_label(sparsity, run["grid"]),
                "checkpoint_ts": run["checkpoint_ts"],
                "eval_ts": run["eval_ts"],
                "run_dir": run["run_dir"],
                "source_csv": str(run["csv_path"].relative_to(REPO_ROOT)),
            }
            for m in KEY_METRICS:
                if m in rec:
                    row[m] = float(rec[m])
            rows.append(row)
    return rows


def aggregate_multiseed(long_form: list[dict[str, Any]], grid: str) -> list[dict[str, Any]]:
    subset = [r for r in long_form if r["lineage"] == "multiseed_new" and r["grid"] == grid]
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for r in subset:
        groups.setdefault((r["model"], r["dataset"], r["sparsity"]), []).append(r)

    out: list[dict[str, Any]] = []
    for (model, dataset, sparsity), items in sorted(groups.items()):
        if len(items) != len(MULTI_MASK_SEEDS):
            continue
        row: dict[str, Any] = {
            "model": model,
            "model_figure": FIGURE_MODEL[model],
            "model_table": TABLE_DISPLAY[model],
            "dataset": dataset,
            "dataset_display": DATASET_DISPLAY[dataset],
            "grid": grid,
            "sparsity": sparsity,
            "sparsity_pct": pct_label(sparsity, grid),
            "n_seeds": len(items),
        }
        for metric in KEY_METRICS:
            vals = [it[metric] for it in items if metric in it]
            if not vals:
                continue
            row[f"{metric}_mean"] = statistics.mean(vals)
            row[f"{metric}_std"] = (
                statistics.stdev(vals) if len(vals) > 1 else 0.0
            )
            for it in items:
                row[f"{metric}_seed{it['mask_seed']}"] = it[metric]
        out.append(row)
    return out


def figure_series(long_form: list[dict[str, Any]], *, lineage: str, grid: str) -> list[dict[str, Any]]:
    rows = []
    for r in long_form:
        if r["lineage"] != lineage or r["grid"] != grid:
            continue
        rows.append(
            {
                "lineage": lineage,
                "model": r["model_figure"],
                "dataset": r["dataset"],
                "dataset_display": r["dataset_display"],
                "sparsity": r["sparsity"],
                "vMAE_a": r["vMAE_a"],
                "mask_seed": r["mask_seed"],
                "checkpoint_ts": r["checkpoint_ts"],
                "source_csv": r["source_csv"],
            }
        )
    return rows


def figure_series_multiseed_mean(agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "lineage": "multiseed_new",
            "model": r["model_figure"],
            "dataset": r["dataset"],
            "dataset_display": r["dataset_display"],
            "sparsity": r["sparsity"],
            "vMAE_a": r["vMAE_a_mean"],
            "vMAE_a_std": r["vMAE_a_std"],
        }
        for r in agg
    ]


def table2_wide(long_form: list[dict[str, Any]], *, lineage: str, grid: str) -> list[dict[str, Any]]:
    pct_cols = PAPER_GRID_PCT if grid == PAPER_LABEL else BREAKDOWN_GRID_PCT
    rows_out: list[dict[str, Any]] = []
    for model in MODELS:
        for dataset in MODELS[model]:
            recs = [
                r
                for r in long_form
                if r["lineage"] == lineage
                and r["grid"] == grid
                and r["model"] == model
                and r["dataset"] == dataset
            ]
            if not recs:
                continue
            by_sp = {r["sparsity_pct"]: r["vMAE_a"] for r in recs}
            row: dict[str, Any] = {
                "lineage": lineage,
                "Dataset": DATASET_DISPLAY[dataset],
                "Model": TABLE_DISPLAY[model],
                "model_key": model,
                "dataset_key": dataset,
                "grid": grid,
            }
            for col in pct_cols:
                row[col] = by_sp.get(col)
            rows_out.append(row)
    return rows_out


def table2_wide_multiseed(agg: list[dict[str, Any]], grid: str) -> list[dict[str, Any]]:
    pct_cols = PAPER_GRID_PCT if grid == PAPER_LABEL else BREAKDOWN_GRID_PCT
    rows_out = []
    for model in MODELS:
        for dataset in MODELS[model]:
            recs = [r for r in agg if r["model"] == model and r["dataset"] == dataset]
            if not recs:
                continue
            row: dict[str, Any] = {
                "lineage": "multiseed_new",
                "Dataset": DATASET_DISPLAY[dataset],
                "Model": TABLE_DISPLAY[model],
                "model_key": model,
                "dataset_key": dataset,
                "grid": grid,
            }
            by_sp = {r["sparsity_pct"]: r for r in recs}
            for col in pct_cols:
                rec = by_sp.get(col)
                if rec is None:
                    row[col] = None
                    row[f"{col}_std"] = None
                else:
                    row[col] = rec["vMAE_a_mean"]
                    row[f"{col}_std"] = rec["vMAE_a_std"]
            rows_out.append(row)
    return rows_out


def sparsity_weights(sparsities: list[float]) -> list[float]:
    w = [math.exp(-20.0 * rho * rho) for rho in sparsities]
    total = sum(w)
    return [x / total for x in w]


def table1_weighted_metrics_for_series(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Weighted bar metrics and degradations for one (model, dataset, seed) paper grid."""
    ordered = sorted(rows, key=lambda r: float(r["sparsity"]))
    weights = sparsity_weights([float(r["sparsity"]) for r in ordered])
    baseline = ordered[0]
    out: dict[str, list[float]] = {}
    for mb in TABLE1_METRIC_BASES:
        channels = [f"{mb}_{ch}" for ch in TABLE1_CHANNELS]
        w_vals = [
            sum(weights[i] * float(ordered[i][col]) for i in range(len(ordered)))
            for col in channels
        ]
        d_vals = [
            sum(weights[i] * float(ordered[i][col]) / float(baseline[col]) for i in range(len(ordered)))
            for col in channels
        ]
        out[f"bar_{mb}"] = w_vals
        out[f"bar_D_{mb[1:]}"] = d_vals
    return out


def _table1_wide_row(
    model: str,
    *,
    lineage: str,
    metric_row: str,
    values: dict[str, list[float]],
    stds: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "lineage": lineage,
        "Model": TABLE_DISPLAY[model],
        "model_key": model,
        "metric_row": metric_row,
    }
    ch_labels = ("Flow", "Occ", "Speed", "Avg")
    for ds in TABLE1_DATASETS:
        ds_tag = DATASET_DISPLAY[ds]
        for ci, ch in enumerate(ch_labels):
            row[f"{ds_tag}_{ch}"] = values[ds][ci]
            if stds is not None:
                row[f"{ds_tag}_{ch}_std"] = stds[ds][ci]
    return row


def table1_weighted_degradation(long_form: list[dict[str, Any]], *, lineage: str) -> list[dict[str, Any]]:
    """LaTeX Table I: bar vMAE/vRMSE and bar D for all channels + Avg, both datasets."""
    rows_out: list[dict[str, Any]] = []
    for model in MODELS:
        per_ds: dict[str, dict[str, list[float]]] = {}
        for dataset in MODELS[model]:
            recs = [
                r
                for r in long_form
                if r["lineage"] == lineage
                and r["grid"] == PAPER_LABEL
                and r["model"] == model
                and r["dataset"] == dataset
            ]
            if not recs:
                continue
            per_ds[dataset] = table1_weighted_metrics_for_series(recs)

        if len(per_ds) != len(TABLE1_DATASETS):
            continue

        for metric_row in TABLE1_ROW_LABELS:
            values = {ds: per_ds[ds][metric_row] for ds in TABLE1_DATASETS}
            rows_out.append(_table1_wide_row(model, lineage=lineage, metric_row=metric_row, values=values))
    return rows_out


def table1_weighted_degradation_multiseed(
    long_form: list[dict[str, Any]], *, grid: str = PAPER_LABEL
) -> list[dict[str, Any]]:
    """LaTeX Table I with mean ± std over mask seeds 43–45."""
    rows_out: list[dict[str, Any]] = []
    for model in MODELS:
        seed_metrics: dict[str, list[dict[str, list[float]]]] = {ds: [] for ds in TABLE1_DATASETS}
        for dataset in MODELS[model]:
            for seed in MULTI_MASK_SEEDS:
                recs = [
                    r
                    for r in long_form
                    if r["lineage"] == "multiseed_new"
                    and r["grid"] == grid
                    and r["model"] == model
                    and r["dataset"] == dataset
                    and r["mask_seed"] == seed
                ]
                if not recs:
                    continue
                seed_metrics[dataset].append(table1_weighted_metrics_for_series(recs))

        if any(len(seed_metrics[ds]) != len(MULTI_MASK_SEEDS) for ds in TABLE1_DATASETS):
            continue

        for metric_row in TABLE1_ROW_LABELS:
            values: dict[str, list[float]] = {}
            stds: dict[str, list[float]] = {}
            for ds in TABLE1_DATASETS:
                channel_vals = list(zip(*[m[metric_row] for m in seed_metrics[ds]]))
                values[ds] = [statistics.mean(v) for v in channel_vals]
                stds[ds] = [
                    statistics.stdev(v) if len(v) > 1 else 0.0 for v in channel_vals
                ]
            rows_out.append(
                _table1_wide_row(
                    model,
                    lineage="multiseed_new",
                    metric_row=metric_row,
                    values=values,
                    stds=stds,
                )
            )
    return rows_out


def table1_baseline(long_form: list[dict[str, Any]], *, lineage: str) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in MODELS[model]:
            recs = [
                r
                for r in long_form
                if r["lineage"] == lineage
                and r["grid"] == PAPER_LABEL
                and r["model"] == model
                and r["dataset"] == dataset
                and float(r["sparsity"]) == 0.0
            ]
            if not recs:
                continue
            if lineage == "paper_original":
                r = recs[0]
                rows.append(
                    {
                        "lineage": lineage,
                        "Dataset": DATASET_DISPLAY[dataset],
                        "Model": TABLE_DISPLAY[model],
                        "metric": "vMAE",
                        "Flow": r["vMAE_flow"],
                        "Occ": r["vMAE_occupancy"],
                        "Speed": r["vMAE_speed"],
                        "Avg": r["vMAE_a"],
                        "mask_seed": r["mask_seed"],
                        "checkpoint_ts": r["checkpoint_ts"],
                    }
                )
    return rows


def table1_baseline_multiseed(agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in MODELS[model]:
            recs = [
                r
                for r in agg
                if r["model"] == model and r["dataset"] == dataset and float(r["sparsity"]) == 0.0
            ]
            if not recs:
                continue
            r = recs[0]
            rows.append(
                {
                    "lineage": "multiseed_new",
                    "Dataset": DATASET_DISPLAY[dataset],
                    "Model": TABLE_DISPLAY[model],
                    "metric": "vMAE",
                    "Flow_mean": r["vMAE_flow_mean"],
                    "Flow_std": r["vMAE_flow_std"],
                    "Occ_mean": r["vMAE_occupancy_mean"],
                    "Occ_std": r["vMAE_occupancy_std"],
                    "Speed_mean": r["vMAE_speed_mean"],
                    "Speed_std": r["vMAE_speed_std"],
                    "Avg_mean": r["vMAE_a_mean"],
                    "Avg_std": r["vMAE_a_std"],
                }
            )
    return rows


def comparison_table(
    paper_lf: list[dict[str, Any]], multi_agg_paper: list[dict[str, Any]], grid: str
) -> list[dict[str, Any]]:
    rows = []
    pct_cols = PAPER_GRID_PCT if grid == PAPER_LABEL else BREAKDOWN_GRID_PCT
    for model in MODELS:
        for dataset in MODELS[model]:
            for col in pct_cols:
                sparsity = float(col.replace("%", "")) / 100.0
                if grid == BREAKDOWN_LABEL and col not in [pct_label(s, grid) for s in SPARSITY_GRIDS[BREAKDOWN_LABEL]]:
                    # pct_label for 5% is "5%" etc - match via sparsity
                    pass
                paper = next(
                    (
                        r
                        for r in paper_lf
                        if r["lineage"] == "paper_original"
                        and r["model"] == model
                        and r["dataset"] == dataset
                        and r["grid"] == grid
                        and abs(float(r["sparsity"]) - sparsity) < 1e-6
                    ),
                    None,
                )
                multi = next(
                    (
                        r
                        for r in multi_agg_paper
                        if r["model"] == model
                        and r["dataset"] == dataset
                        and abs(float(r["sparsity"]) - sparsity) < 1e-6
                    ),
                    None,
                )
                if not paper or not multi:
                    continue
                rows.append(
                    {
                        "grid": grid,
                        "dataset": dataset,
                        "dataset_display": DATASET_DISPLAY[dataset],
                        "model": model,
                        "model_table": TABLE_DISPLAY[model],
                        "sparsity_pct": col,
                        "paper_original_vMAE_a": paper["vMAE_a"],
                        "multiseed_mean_vMAE_a": multi["vMAE_a_mean"],
                        "multiseed_std_vMAE_a": multi["vMAE_a_std"],
                        "delta_mean_minus_paper": multi["vMAE_a_mean"] - paper["vMAE_a"],
                    }
                )
    return rows


def build_manifest(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        rows.append(
            {
                "lineage": run["lineage"],
                "model": run["model"],
                "dataset": run["dataset"],
                "grid": run["grid"],
                "mask_seed": run["mask_seed"],
                "checkpoint_ts": run["checkpoint_ts"],
                "eval_ts": run["eval_ts"],
                "batch_size": run.get("batch_size"),
                "run_dir": run["run_dir"],
                "source_csv": str(run["csv_path"].relative_to(REPO_ROOT)),
                "source_metadata": str(run["meta_path"].relative_to(REPO_ROOT)),
                "model_dir": run.get("model_dir"),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build sparsity results CSV cache for figures/tables")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--copy", action="store_true", help="Copy CSVs instead of symlinking")
    args = ap.parse_args()

    out = args.out_dir.resolve()
    paper_runs, multi_runs = select_lineage_runs()
    all_runs = paper_runs + multi_runs

    missing_paper = []
    missing_multi = []
    for model in MODELS:
        for dataset in MODELS[model]:
            for grid in (PAPER_LABEL, BREAKDOWN_LABEL):
                if not any(
                    r["model"] == model and r["dataset"] == dataset and r["grid"] == grid
                    for r in paper_runs
                ):
                    missing_paper.append(f"{model}/{dataset}/{grid}")
                for seed in MULTI_MASK_SEEDS:
                    if not any(
                        r["model"] == model
                        and r["dataset"] == dataset
                        and r["grid"] == grid
                        and r["mask_seed"] == seed
                        for r in multi_runs
                    ):
                        missing_multi.append(f"{model}/{dataset}/{grid}/seed{seed}")

    # raw cache
    for run in all_runs:
        rel = (
            Path("raw")
            / run["lineage"]
            / run["model"]
            / run["dataset"]
            / run["grid"]
            / f"seed{run['mask_seed']}"
            / "sparsity_results.csv"
        )
        symlink_or_copy(run["csv_path"], out / rel, copy=args.copy)

    long_paper = build_long_form(paper_runs)
    long_multi = build_long_form(multi_runs)
    long_all = long_paper + long_multi

    agg_paper = aggregate_multiseed(long_multi, PAPER_LABEL)
    agg_breakdown = aggregate_multiseed(long_multi, BREAKDOWN_LABEL)

    manifest = build_manifest(all_runs)
    if manifest:
        write_csv(out / "run_manifest.csv", list(manifest[0].keys()), manifest)

    if long_paper:
        write_csv(out / "long_form/paper_original_all_metrics.csv", list(long_paper[0].keys()), long_paper)
    if long_multi:
        write_csv(out / "long_form/multiseed_new_all_metrics.csv", list(long_multi[0].keys()), long_multi)
    if long_all:
        write_csv(out / "long_form/all_lineages_all_metrics.csv", list(long_all[0].keys()), long_all)

    # figure-ready
    for grid in (PAPER_LABEL, BREAKDOWN_LABEL):
        fig_p = figure_series(long_paper, lineage="paper_original", grid=grid)
        if fig_p:
            write_csv(out / f"figures/paper_original_{grid}_vmae_a.csv", list(fig_p[0].keys()), fig_p)
        agg = agg_paper if grid == PAPER_LABEL else agg_breakdown
        fig_m = figure_series_multiseed_mean(agg)
        if fig_m:
            write_csv(out / f"figures/multiseed_new_{grid}_vmae_a_mean.csv", list(fig_m[0].keys()), fig_m)
        if agg:
            write_csv(
                out / f"figures/multiseed_new_{grid}_all_metrics_mean_std.csv",
                list(agg[0].keys()),
                agg,
            )

    # tables
    for grid in (PAPER_LABEL, BREAKDOWN_LABEL):
        t2p = table2_wide(long_paper, lineage="paper_original", grid=grid)
        if t2p:
            pct_cols = PAPER_GRID_PCT if grid == PAPER_LABEL else BREAKDOWN_GRID_PCT
            cols = ["lineage", "Dataset", "Model", "model_key", "dataset_key", "grid", *pct_cols]
            write_csv(out / f"tables/paper_original_table2_{grid}_vmae_a_wide.csv", cols, t2p)

        agg = agg_paper if grid == PAPER_LABEL else agg_breakdown
        t2m = table2_wide_multiseed(agg, grid)
        if t2m:
            pct_cols = PAPER_GRID_PCT if grid == PAPER_LABEL else BREAKDOWN_GRID_PCT
            cols = ["lineage", "Dataset", "Model", "model_key", "dataset_key", "grid"]
            for col in pct_cols:
                cols.extend([col, f"{col}_std"])
            write_csv(out / f"tables/multiseed_new_table2_{grid}_vmae_a_wide.csv", cols, t2m)

        cmp_rows = comparison_table(long_paper, agg, grid)
        if cmp_rows:
            write_csv(
                out / f"tables/comparison_paper_original_vs_multiseed_{grid}_vmae_a.csv",
                list(cmp_rows[0].keys()),
                cmp_rows,
            )

    t1p = table1_weighted_degradation(long_paper, lineage="paper_original")
    if t1p:
        write_csv(out / "tables/paper_original_table1_weighted_degradation.csv", list(t1p[0].keys()), t1p)
    t1m = table1_weighted_degradation_multiseed(long_multi, grid=PAPER_LABEL)
    if t1m:
        write_csv(out / "tables/multiseed_new_table1_weighted_degradation_mean_std.csv", list(t1m[0].keys()), t1m)
    t1m_bd = table1_weighted_degradation_multiseed(long_multi, grid=BREAKDOWN_LABEL)
    if t1m_bd:
        write_csv(
            out / "tables/multiseed_new_table1_breakdown035_weighted_degradation_mean_std.csv",
            list(t1m_bd[0].keys()),
            t1m_bd,
        )

    # Legacy baseline-only exports (0% sparsity point estimates).
    t1b_p = table1_baseline(long_paper, lineage="paper_original")
    if t1b_p:
        write_csv(out / "tables/paper_original_table1_baseline_vmae.csv", list(t1b_p[0].keys()), t1b_p)
    t1b_m = table1_baseline_multiseed(agg_paper)
    if t1b_m:
        write_csv(out / "tables/multiseed_new_table1_baseline_vmae_mean_std.csv", list(t1b_m[0].keys()), t1b_m)

    summary = {
        "out_dir": str(out),
        "paper_original_runs": len(paper_runs),
        "multiseed_new_runs": len(multi_runs),
        "missing_paper_original": missing_paper,
        "missing_multiseed_new": missing_multi,
    }
    (out / "cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
