"""Job definitions for multi-seed sparsity mask re-evaluation."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

# Final standardized-input checkpoints used by the manuscript.
STANDARD_MODELS: dict[str, dict[str, str]] = {
    "DCRNN": {
        "PEMSD4": "libcity/cache/DCRNN_PEMSD4_20260831_114637",
        "PEMSD8": "libcity/cache/DCRNN_PEMSD8_20260831_071603",
    },
    "D2STGNN": {
        "PEMSD4": "libcity/cache/D2STGNN_PEMSD4_20260816_151002",
        "PEMSD8": "libcity/cache/D2STGNN_PEMSD8_20260816_150942",
    },
    "Mamba4Traffic": {
        "PEMSD4": "libcity/cache/Mamba4Traffic_PEMSD4_20260816_151054",
        "PEMSD8": "libcity/cache/Mamba4Traffic_PEMSD8_20260816_151048",
    },
    "Trafformer": {
        "PEMSD4": "libcity/cache/Trafformer_PEMSD4_20260921_064833",
        "PEMSD8": "libcity/cache/Trafformer_PEMSD8_20260817_104712",
    },
}

STANDARD_TRAINING_LOG_FILES: dict[tuple[str, str], str] = {
    ("DCRNN", "PEMSD4"): "libcity/cache/DCRNN_PEMSD4_20260831_114637/logs/DCRNN_PEMSD4_20260831_114637-DCRNN-PEMSD4-Aug-31-2026_11-46-37.log",
    ("DCRNN", "PEMSD8"): "libcity/cache/DCRNN_PEMSD8_20260831_071603/logs/DCRNN_PEMSD8_20260831_071603-DCRNN-PEMSD8-Aug-31-2026_07-16-03.log",
    ("D2STGNN", "PEMSD4"): "libcity/cache/D2STGNN_PEMSD4_20260816_151002/logs/D2STGNN_PEMSD4_20260816_151002-D2STGNN-PEMSD4-Aug-16-2026_15-10-02.log",
    ("D2STGNN", "PEMSD8"): "libcity/cache/D2STGNN_PEMSD8_20260816_150942/logs/D2STGNN_PEMSD8_20260816_150942-D2STGNN-PEMSD8-Aug-16-2026_15-09-42.log",
    ("Mamba4Traffic", "PEMSD4"): "libcity/cache/Mamba4Traffic_PEMSD4_20260816_151054/logs/Mamba4Traffic_PEMSD4_20260816_151054-Mamba4Traffic-PEMSD4-Aug-16-2026_15-10-54.log",
    ("Mamba4Traffic", "PEMSD8"): "libcity/cache/Mamba4Traffic_PEMSD8_20260816_151048/logs/Mamba4Traffic_PEMSD8_20260816_151048-Mamba4Traffic-PEMSD8-Aug-16-2026_15-10-48.log",
    ("Trafformer", "PEMSD4"): "libcity/cache/Trafformer_PEMSD4_20260921_064833/logs/Trafformer_PEMSD4_20260921_064833-Trafformer-PEMSD4-Sep-21-2026_06-48-33.log",
    ("Trafformer", "PEMSD8"): "libcity/cache/Trafformer_PEMSD8_20260817_104712/logs/Trafformer_PEMSD8_20260817_104712-Trafformer-PEMSD8-Aug-17-2026_10-47-12.log",
}

CHECKPOINT_PROFILE = os.environ.get("SPARSITY_CHECKPOINT_PROFILE", "standard").strip().lower()
if CHECKPOINT_PROFILE == "standard":
    MODELS = STANDARD_MODELS
    TRAINING_LOG_FILES = STANDARD_TRAINING_LOG_FILES
elif CHECKPOINT_PROFILE == "raw":
    raw_map = os.environ.get("SPARSITY_RAW_CHECKPOINT_MAP")
    if not raw_map:
        raise ValueError("raw profile requires SPARSITY_RAW_CHECKPOINT_MAP; no raw IDs are published")
    payload = json.loads(Path(raw_map).read_text())
    MODELS = payload["models"]
    TRAINING_LOG_FILES = {
        tuple(key.split("|", 1)): value for key, value in payload["training_logs"].items()
    }
else:
    raise ValueError("SPARSITY_CHECKPOINT_PROFILE must be 'standard' or 'raw'")

PAPER_LABEL = "paper"
BREAKDOWN_LABEL = "breakdown"

SPARSITY_GRIDS: dict[str, list[float]] = {
    PAPER_LABEL: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    BREAKDOWN_LABEL: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
}

DEFAULT_MASK_SEEDS = [43, 44, 45]
SMOKE_MASK_SEED = 46
SMOKE_SPARSITY_LEVEL = 0.1
SMOKE_LABEL = "smoke"

# Eval batch overrides (default in eval_sparsity.py is 256).
# PEMSD4 (307 nodes): Mamba batch 256 and Trafformer batch 8 both OOM on A10-24Q
# even with ~20 GB nvidia-smi "free" (deepsense ~5 GB + large contiguous allocs).
EVAL_BATCH_SIZE: dict[tuple[str, str], int] = {
    ("Trafformer", "PEMSD4"): 1,
    ("Trafformer", "PEMSD8"): 4,
    ("Mamba4Traffic", "PEMSD4"): 64,
}


def eval_batch_size(model: str, dataset: str) -> int | None:
    return EVAL_BATCH_SIZE.get((model, dataset))


@dataclass(frozen=True)
class SparsityJob:
    name: str
    model: str
    dataset: str
    mask_seed: int
    grid: str
    model_dir: str
    log_file: str
    sparsity_levels: tuple[float, ...]
    batch_size: int | None = None

    def to_run_spec(self, *, python: str) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": build_eval_command(job=self, python=python),
            "phase": "eval",
            "model_name": self.model,
            "launcher_group": f"{self.grid}_seed{self.mask_seed}",
            "env": {},
        }


def model_dir_path(model: str, dataset: str) -> Path:
    if model not in MODELS or dataset not in MODELS[model]:
        raise KeyError(f"Unknown model/dataset: {model} / {dataset}")
    return REPO_ROOT / MODELS[model][dataset]


def training_complete_reason(model: str, dataset: str) -> str | None:
    """Return None if ready for sparsity eval; otherwise a short skip reason."""
    try:
        run_dir = model_dir_path(model, dataset)
    except KeyError as exc:
        return str(exc)

    if not run_dir.is_dir():
        return f"missing run dir {MODELS[model][dataset]}"

    model_cache = run_dir / "model_cache"
    if not model_cache.is_dir() or not (any(model_cache.glob("*.tar")) or any(model_cache.glob("*.m"))):
        return "no checkpoint in model_cache"

    summary = run_dir / "logs" / "training_summary.txt"
    if not summary.is_file():
        return "training not finished (no logs/training_summary.txt)"

    log_path = resolve_training_log(model, dataset)
    if log_path is None:
        return "training log not found"

    return None


def is_training_complete(model: str, dataset: str) -> bool:
    return training_complete_reason(model, dataset) is None


def list_incomplete_model_datasets(
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    model_list = models or list(MODELS.keys())
    dataset_list = datasets or ["PEMSD4", "PEMSD8"]
    skipped: list[tuple[str, str, str]] = []
    for model in model_list:
        if model not in MODELS:
            continue
        for dataset in dataset_list:
            if dataset not in MODELS[model]:
                continue
            reason = training_complete_reason(model, dataset)
            if reason is not None:
                skipped.append((model, dataset, reason))
    return skipped


def resolve_training_log(model: str, dataset: str) -> Path | None:
    key = (model, dataset)
    if key in TRAINING_LOG_FILES:
        path = REPO_ROOT / TRAINING_LOG_FILES[key]
        if path.is_file():
            return path

    run_dir = model_dir_path(model, dataset)
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        candidates = sorted(logs_dir.glob("*.log"))
        if candidates:
            return candidates[-1]
    return None


def iter_jobs(
    *,
    models: list[str],
    datasets: list[str],
    mask_seeds: list[int],
    grids: list[str],
    skip_incomplete: bool = True,
) -> Iterator[SparsityJob]:
    """Yield jobs with grid as the outer loop so paper (0–100%) runs before breakdown (0–35%)."""
    for grid in grids:
        if grid not in SPARSITY_GRIDS:
            raise ValueError(f"Unknown grid {grid!r}; choose from {sorted(SPARSITY_GRIDS)}")
        for model in models:
            if model not in MODELS:
                raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODELS)}")
            for dataset in datasets:
                if dataset not in MODELS[model]:
                    raise ValueError(f"Unknown dataset {dataset!r} for model {model}")
                if skip_incomplete and not is_training_complete(model, dataset):
                    continue
                log_file = training_log_file(model, dataset)
                for seed in mask_seeds:
                    yield SparsityJob(
                        name=f"{model}_{dataset}_seed{seed}_{grid}",
                        model=model,
                        dataset=dataset,
                        mask_seed=seed,
                        grid=grid,
                        model_dir=MODELS[model][dataset],
                        log_file=log_file,
                        sparsity_levels=tuple(SPARSITY_GRIDS[grid]),
                        batch_size=eval_batch_size(model, dataset),
                    )


def iter_smoke_jobs(
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    mask_seed: int = SMOKE_MASK_SEED,
    sparsity_level: float = SMOKE_SPARSITY_LEVEL,
    skip_incomplete: bool = True,
) -> Iterator[SparsityJob]:
    """One masked level per (model, dataset) to validate eval + launcher setup."""
    model_list = models or list(MODELS.keys())
    dataset_list = datasets or ["PEMSD4", "PEMSD8"]
    rho_tag = f"rho{sparsity_level:.2f}".replace(".", "p")
    for model in model_list:
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODELS)}")
        for dataset in dataset_list:
            if dataset not in MODELS[model]:
                raise ValueError(f"Unknown dataset {dataset!r} for model {model}")
            if skip_incomplete and not is_training_complete(model, dataset):
                continue
            yield SparsityJob(
                name=f"{model}_{dataset}_seed{mask_seed}_{SMOKE_LABEL}_{rho_tag}",
                model=model,
                dataset=dataset,
                mask_seed=mask_seed,
                grid=SMOKE_LABEL,
                model_dir=MODELS[model][dataset],
                log_file=training_log_file(model, dataset),
                sparsity_levels=(float(sparsity_level),),
                batch_size=eval_batch_size(model, dataset),
            )


def training_log_file(model: str, dataset: str) -> str:
    path = resolve_training_log(model, dataset)
    if path is None:
        raise ValueError(f"No training log found for {model} / {dataset}")
    return path.relative_to(REPO_ROOT).as_posix()


def build_eval_command(*, job: SparsityJob, python: str) -> str:
    cmd = [
        python,
        "eval_sparsity.py",
        "--model_dir",
        job.model_dir,
        "--log_file",
        job.log_file,
        "--sparsity_mode",
        "sweep",
        "--sparsity",
        *map(str, job.sparsity_levels),
        "--mask_seed",
        str(job.mask_seed),
        "--no_visualize_topology",
        "--no_save_series_csvs",
        "--no_plot_metrics",
    ]
    if job.batch_size is not None:
        cmd.extend(["--batch_size", str(job.batch_size)])
    return " ".join(shlex.quote(x) for x in cmd)
