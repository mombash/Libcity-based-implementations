#!/usr/bin/env python3
"""Build the portable paper cache from an explicit 48-run selection manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any


SEEDS = (43, 44, 45)
MODELS = ("DCRNN", "D2STGNN", "Mamba4Traffic", "Trafformer")
DATASETS = ("PEMSD4", "PEMSD8")
GRIDS = ("paper", "breakdown")
DATASET_DISPLAY = {"PEMSD4": "PEMS04", "PEMSD8": "PEMS08"}
KEY_METRICS = (
    "vMAE_a",
    "vRMSE_a",
    "vMAPE_a",
    "vMAE_flow",
    "vMAE_occupancy",
    "vMAE_speed",
    "vRMSE_flow",
    "vRMSE_occupancy",
    "vRMSE_speed",
)
CHANNELS = ("flow", "occupancy", "speed", "a")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct_label(rho: float) -> str:
    return f"{rho * 100:g}%"


def validate_selection(rows: list[dict[str, str]]) -> None:
    expected = {
        (model, dataset, grid, seed)
        for model in MODELS
        for dataset in DATASETS
        for grid in GRIDS
        for seed in SEEDS
    }
    actual = {
        (row["model"], row["dataset"], row["grid"], int(row["mask_seed"]))
        for row in rows
    }
    if len(rows) != 48 or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"Selection must contain the exact 48 paper runs; missing={missing}, extra={extra}"
        )


def build_long_form(
    selection: list[dict[str, str]],
    runs_root: Path,
    output: Path,
    *,
    allow_dedicated_trafformer_breakdown: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for selected in selection:
        run_dir = selected["run_dir"]
        source_dir = runs_root / run_dir
        source_csv = source_dir / "sparsity_results.csv"
        metadata_path = source_dir / "eval_metadata.json"
        config_path = source_dir / "training_config.json"
        masks_path = source_dir / "sparsity_results_masks.json"
        for required in (source_csv, metadata_path, config_path, masks_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(selected["mask_seed"])
        if int(metadata["mask_seed"]) != seed or metadata["mask_zero_mode"] != "inverse":
            raise RuntimeError(f"Incorrect mask protocol in {metadata_path}")
        if config.get("scaler") != "standard":
            raise RuntimeError(f"Non-standard scaler in {config_path}")
        if config.get("data_col") != [
            "traffic_flow",
            "traffic_occupancy",
            "traffic_speed",
        ]:
            raise RuntimeError(f"Incorrect channel order in {config_path}")
        if selected["model"] == "Trafformer" and selected["dataset"] == "PEMSD4":
            dedicated_breakdown = (
                allow_dedicated_trafformer_breakdown
                and selected["grid"] == "breakdown"
                and selected["checkpoint_ts"] == "20260817_065509"
            )
            if int(config.get("sample_stride", 1)) != 1 and not dedicated_breakdown:
                raise RuntimeError(f"Trafformer/PEMS04 stride is not one in {config_path}")
        if selected["model"] == "D2STGNN" and int(config.get("gap", 3)) != 3:
            raise RuntimeError(f"D2STGNN gap is not three in {config_path}")

        raw_rel = (
            Path("raw")
            / "multiseed_new"
            / selected["model"]
            / selected["dataset"]
            / selected["grid"]
            / f"seed{seed}"
            / "sparsity_results.csv"
        )
        raw_path = output / raw_rel
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_csv, raw_path)

        records = read_csv(source_csv)
        expected_levels = 11 if selected["grid"] == "paper" else 8
        if len(records) != expected_levels:
            raise RuntimeError(f"Wrong grid length in {source_csv}: {len(records)}")
        for record in records:
            rho = float(record["sparsity"])
            channel_mean = statistics.mean(
                float(record[f"vMAE_{channel}"])
                for channel in ("flow", "occupancy", "speed")
            )
            if not math.isclose(channel_mean, float(record["vMAE_a"]), abs_tol=1e-12):
                raise RuntimeError(f"vMAE_a is not the three-channel mean in {source_csv}")
            row: dict[str, Any] = {
                "lineage": "multiseed_new",
                "model": selected["model"],
                "model_figure": selected["model"],
                "model_table": selected["model"],
                "dataset": selected["dataset"],
                "dataset_display": DATASET_DISPLAY[selected["dataset"]],
                "grid": selected["grid"],
                "mask_seed": seed,
                "sparsity": rho,
                "sparsity_pct": pct_label(rho),
                "checkpoint_ts": selected["checkpoint_ts"],
                "eval_ts": selected["eval_ts"],
                "run_dir": run_dir,
                "source_csv": raw_rel.as_posix(),
            }
            row.update({metric: float(record[metric]) for metric in KEY_METRICS})
            long_rows.append(row)

        manifest_rows.append(
            {
                "lineage": "multiseed_new",
                "model": selected["model"],
                "dataset": selected["dataset"],
                "grid": selected["grid"],
                "mask_seed": seed,
                "checkpoint_ts": selected["checkpoint_ts"],
                "eval_ts": selected["eval_ts"],
                "run_dir": run_dir,
                "source_csv": raw_rel.as_posix(),
                "source_metadata": f"provenance/{run_dir}/eval_metadata.json",
                "checkpoint_dir": f"checkpoints/{selected['model']}_{selected['dataset']}_{selected['checkpoint_ts']}",
            }
        )
    return long_rows, manifest_rows


def aggregate(long_rows: list[dict[str, Any]], grid: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODELS:
        for dataset in DATASETS:
            levels = sorted(
                {
                    row["sparsity"]
                    for row in long_rows
                    if row["model"] == model
                    and row["dataset"] == dataset
                    and row["grid"] == grid
                }
            )
            for rho in levels:
                items = [
                    row
                    for row in long_rows
                    if row["model"] == model
                    and row["dataset"] == dataset
                    and row["grid"] == grid
                    and row["sparsity"] == rho
                ]
                if [item["mask_seed"] for item in items] != list(SEEDS):
                    items.sort(key=lambda item: item["mask_seed"])
                if len(items) != 3:
                    raise RuntimeError(f"Incomplete seed group: {model}/{dataset}/{grid}/{rho}")
                row: dict[str, Any] = {
                    "model": model,
                    "model_figure": model,
                    "model_table": model,
                    "dataset": dataset,
                    "dataset_display": DATASET_DISPLAY[dataset],
                    "grid": grid,
                    "sparsity": rho,
                    "sparsity_pct": pct_label(rho),
                    "n_seeds": 3,
                }
                for metric in KEY_METRICS:
                    values = [item[metric] for item in items]
                    row[f"{metric}_mean"] = statistics.mean(values)
                    row[f"{metric}_std"] = statistics.stdev(values)
                    for item in items:
                        row[f"{metric}_seed{item['mask_seed']}"] = item[metric]
                output.append(row)
    return output


def table2(aggregated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = [f"{pct}%" for pct in range(0, 101, 10)]
    output: list[dict[str, Any]] = []
    for model in MODELS:
        for dataset in DATASETS:
            items = [
                row
                for row in aggregated
                if row["model"] == model and row["dataset"] == dataset
            ]
            by_pct = {row["sparsity_pct"]: row for row in items}
            row: dict[str, Any] = {
                "lineage": "multiseed_new",
                "Dataset": DATASET_DISPLAY[dataset],
                "Model": model,
                "model_key": model,
                "dataset_key": dataset,
                "grid": "paper",
            }
            for column in columns:
                row[column] = by_pct[column]["vMAE_a_mean"]
                row[f"{column}_std"] = by_pct[column]["vMAE_a_std"]
            output.append(row)
    return output


def weights(rows: list[dict[str, Any]]) -> list[float]:
    raw = [math.exp(-20.0 * float(row["sparsity"]) ** 2) for row in rows]
    total = sum(raw)
    return [value / total for value in raw]


def seed_weighted(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    rows = sorted(rows, key=lambda row: row["sparsity"])
    weight = weights(rows)
    baseline = rows[0]
    output: dict[str, list[float]] = {}
    for metric in ("vMAE", "vRMSE"):
        columns = [f"{metric}_{channel}" for channel in CHANNELS]
        output[f"bar_{metric}"] = [
            sum(weight[index] * rows[index][column] for index in range(len(rows)))
            for column in columns
        ]
        output[f"bar_D_{metric[1:]}"] = [
            sum(
                weight[index] * rows[index][column] / baseline[column]
                for index in range(len(rows))
            )
            for column in columns
        ]
    return output


def table1(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    labels = ("Flow", "Occ", "Speed", "Avg")
    for model in MODELS:
        per_dataset: dict[str, list[dict[str, list[float]]]] = {}
        for dataset in DATASETS:
            per_seed = []
            for seed in SEEDS:
                rows = [
                    row
                    for row in long_rows
                    if row["model"] == model
                    and row["dataset"] == dataset
                    and row["grid"] == "breakdown"
                    and row["mask_seed"] == seed
                ]
                per_seed.append(seed_weighted(rows))
            per_dataset[dataset] = per_seed
        for metric in ("bar_vMAE", "bar_D_MAE", "bar_vRMSE", "bar_D_RMSE"):
            row: dict[str, Any] = {
                "lineage": "multiseed_new",
                "Model": model,
                "model_key": model,
                "metric_row": metric,
            }
            for dataset in DATASETS:
                values_by_channel = list(
                    zip(*(result[metric] for result in per_dataset[dataset]))
                )
                for label, values in zip(labels, values_by_channel):
                    column = f"{DATASET_DISPLAY[dataset]}_{label}"
                    row[column] = statistics.mean(values)
                    row[f"{column}_std"] = statistics.stdev(values)
            output.append(row)
    return output


def build(
    selection_path: Path,
    runs_root: Path,
    output: Path,
    *,
    allow_dedicated_trafformer_breakdown: bool = False,
) -> None:
    selection = read_csv(selection_path)
    validate_selection(selection)
    long_rows, manifest_rows = build_long_form(
        selection,
        runs_root,
        output,
        allow_dedicated_trafformer_breakdown=allow_dedicated_trafformer_breakdown,
    )
    paper = aggregate(long_rows, "paper")
    breakdown = aggregate(long_rows, "breakdown")

    write_csv(output / "run_manifest.csv", manifest_rows)
    write_csv(output / "long_form" / "multiseed_new_all_metrics.csv", long_rows)
    for grid, rows in (("paper", paper), ("breakdown", breakdown)):
        write_csv(
            output / "figures" / f"multiseed_new_{grid}_all_metrics_mean_std.csv",
            rows,
        )
        write_csv(
            output / "figures" / f"multiseed_new_{grid}_vmae_a_mean.csv",
            [
                {
                    "lineage": "multiseed_new",
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "dataset_display": row["dataset_display"],
                    "sparsity": row["sparsity"],
                    "vMAE_a": row["vMAE_a_mean"],
                    "vMAE_a_std": row["vMAE_a_std"],
                }
                for row in rows
            ],
        )
    write_csv(
        output / "tables" / "multiseed_new_table2_paper_vmae_a_wide.csv",
        table2(paper),
    )
    write_csv(
        output
        / "tables"
        / "multiseed_new_table1_breakdown035_weighted_degradation_mean_std.csv",
        table1(long_rows),
    )
    summary = {
        "checkpoint_profile": "standard",
        "lineage": "multiseed_new",
        "mask_seeds": list(SEEDS),
        "run_count": len(selection),
        "paper_grid": [value / 10 for value in range(11)],
        "breakdown_grid": [value / 20 for value in range(8)],
    }
    (output / "cache_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-dedicated-trafformer-breakdown",
        action="store_true",
        help="Permit the dedicated Trafformer/PEMS04 evaluation set used only by the six channel plots.",
    )
    args = parser.parse_args()
    build(
        args.selection.resolve(),
        args.runs_root.resolve(),
        args.output.resolve(),
        allow_dedicated_trafformer_breakdown=args.allow_dedicated_trafformer_breakdown,
    )


if __name__ == "__main__":
    main()
