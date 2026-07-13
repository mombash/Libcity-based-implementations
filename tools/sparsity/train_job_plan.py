"""Training job definitions for sparsity paper checkpoint recovery."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default for retrain recovery when paper batch_size is unavailable or OOM-prone.
RETRAIN_BATCH_SIZE = 16

# Retrain with exact paper training_config batch sizes (not forced to 16).
ORIGINAL_SETUP_TARGETS: set[tuple[str, str]] = {
    ("DCRNN", "PEMSD4"),   # paper batch_size=32
    ("DCRNN", "PEMSD8"),   # paper batch_size=64
    ("D2STGNN", "PEMSD4"),  # paper batch_size=32
    ("Trafformer", "PEMSD8"),  # paper batch_size=3 (bs=16 OOMs with sliding)
}

SKIP_RETRAIN: set[tuple[str, str]] = {
    ("D2STGNN", "PEMSD8"),  # kept Dec 2025 checkpoint
    ("D2STGNN", "PEMSD4"),  # corrected retrain done: 20260623_102537
    ("Trafformer", "PEMSD4"),  # sliding retrain done: 20260624_071738 (noslide is tail job)
}

RETRAIN_TARGETS: list[tuple[str, str]] = [
    ("DCRNN", "PEMSD4"),
    ("DCRNN", "PEMSD8"),
    ("D2STGNN", "PEMSD4"),
    ("MCSTMambaLST_Ablation", "PEMSD4"),
    ("MCSTMambaLST_Ablation", "PEMSD8"),
    ("Trafformer", "PEMSD4"),
    ("Trafformer", "PEMSD8"),
]

# Estimated wall hours to paper best epoch (batch_size=16), fastest first.
TRAIN_ESTIMATED_HOURS: dict[tuple[str, str], float] = {
    ("D2STGNN", "PEMSD4"): 5.0,
    ("Trafformer", "PEMSD4"): 7.0,
    ("Trafformer", "PEMSD8"): 8.0,
    ("MCSTMambaLST_Ablation", "PEMSD4"): 11.0,
    ("MCSTMambaLST_Ablation", "PEMSD8"): 17.0,
    ("DCRNN", "PEMSD8"): 32.0,
    ("DCRNN", "PEMSD4"): 35.0,
}

CONFIG_DIR = REPO_ROOT / "configs" / "sparsity_retrain"
PAPER_CONFIG_GLOB = "sparcity_cache/*/training_config.json"

TRAINING_STARTED_MARKERS = ("Start training ...", "Start training")

# Run after all standard retrains (Trafformer P4 non-sliding ablation; paper bs=3).
TAIL_RETRAIN_JOBS: list[dict[str, Any]] = [
    {
        "name": "train_Trafformer_PEMSD4_noslide",
        "model": "Trafformer",
        "dataset": "PEMSD4",
        "config_file": "configs/sparsity_retrain/Trafformer_PEMSD4_noslide",
        "batch_size": 3,
        "estimated_hours": 3.0,
    },
]

# Keys stored in paper training_config.json but not passed as model overrides.
_EXCLUDE_CONFIG_KEYS = frozenset(
    {
        "task",
        "model",
        "dataset",
        "saved_model",
        "train",
        "seed",
        "metrics",
        "evaluator_mode",
        "save_mode",
        "normalize_metrics",
        "normalization_method",
        "geo",
        "rel",
        "dyna",
        "gpu",
        "gpu_id",
        "device",
        "epoch",
        "train_loss",
        "exp_id",
        "robustness_test",
        "noise_type",
        "disturb_rate",
        "noise_mean",
        "noise_SD",
        "log_level",
        "log_every",
        "load_best_epoch",
        "hyper_tune",
    }
)


@dataclass(frozen=True)
class RetrainJob:
    name: str
    model: str
    dataset: str
    config_file: str
    batch_size: int = RETRAIN_BATCH_SIZE
    estimated_hours: float = 999.0

    def to_run_spec(self, *, python: str, smoke: bool = False) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": build_train_command(job=self, python=python, smoke=smoke),
            "phase": "train",
            "model_name": self.model,
            "launcher_group": "sparsity_retrain_smoke" if smoke else "sparsity_retrain",
            "estimated_hours": self.estimated_hours,
            "env": {},
        }


def iter_retrain_jobs(
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> Iterator[RetrainJob]:
    """Yield training jobs for the minimum retrain set, fastest estimated models first."""
    model_filter = set(models) if models else None
    dataset_filter = set(datasets) if datasets else None

    targets: list[tuple[str, str]] = []
    for model, dataset in RETRAIN_TARGETS:
        if (model, dataset) in SKIP_RETRAIN:
            continue
        if model_filter is not None and model not in model_filter:
            continue
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        targets.append((model, dataset))

    targets.sort(key=lambda md: (TRAIN_ESTIMATED_HOURS.get(md, 999.0), md[0], md[1]))

    for model, dataset in targets:
        config_file = write_retrain_config(model, dataset)
        batch_size = retrain_batch_size(model, dataset)
        yield RetrainJob(
            name=f"train_{model}_{dataset}",
            model=model,
            dataset=dataset,
            config_file=config_file,
            batch_size=batch_size,
            estimated_hours=TRAIN_ESTIMATED_HOURS.get((model, dataset), 999.0),
        )

    for tail in TAIL_RETRAIN_JOBS:
        model = str(tail["model"])
        dataset = str(tail["dataset"])
        if model_filter is not None and model not in model_filter:
            continue
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        yield RetrainJob(
            name=str(tail["name"]),
            model=model,
            dataset=dataset,
            config_file=str(tail["config_file"]),
            batch_size=int(tail["batch_size"]),
            estimated_hours=float(tail.get("estimated_hours", 999.0)),
        )


def load_paper_training_config(model: str, dataset: str) -> dict[str, Any]:
    """Load the oldest paper checkpoint config (seed-42 eval lineage), not retrain caches."""
    matches: list[tuple[str, Path]] = []
    for path in REPO_ROOT.glob(PAPER_CONFIG_GLOB):
        meta_path = path.parent / "eval_metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model") != model or meta.get("dataset") != dataset:
            continue
        orig_ts = str(meta.get("original_checkpoint_timestamp", ""))
        matches.append((orig_ts, path))

    if not matches:
        raise FileNotFoundError(f"No paper training_config.json for {model} / {dataset}")

    # Prefer the Sep–Dec 2025 paper checkpoints over Jun 2026 retrain eval caches.
    matches.sort(key=lambda item: (item[0], item[1].as_posix()))
    return json.loads(matches[0][1].read_text(encoding="utf-8"))


def retrain_batch_size(model: str, dataset: str) -> int:
    """Batch size for a retrain job: paper original where listed, else RETRAIN_BATCH_SIZE."""
    if (model, dataset) in ORIGINAL_SETUP_TARGETS:
        return int(load_paper_training_config(model, dataset)["batch_size"])
    return RETRAIN_BATCH_SIZE


def resolve_paper_data_sampling(paper_cfg: dict[str, Any]) -> dict[str, Any]:
    """Retrain recovery uses sliding windows (stride 1) for all models.

    Paper checkpoints often omitted ``use_sliding_window`` (LibCity default True) or,
    for Trafformer PEMSD4, used non-sliding. Retrains standardize on sliding unless
    an explicit ablation config overrides (see ``Trafformer_PEMSD4_noslide.json``).
    """
    _ = paper_cfg
    return {"use_sliding_window": True, "sample_stride": 1}


def build_retrain_config(model: str, dataset: str) -> dict[str, Any]:
    cfg = load_paper_training_config(model, dataset)
    out = {k: v for k, v in cfg.items() if k not in _EXCLUDE_CONFIG_KEYS}
    out["batch_size"] = retrain_batch_size(model, dataset)
    out.update(resolve_paper_data_sampling(cfg))
    return out


def write_retrain_config(model: str, dataset: str) -> str:
    """Write JSON under configs/sparsity_retrain/; return --config_file stem."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rel_stem = f"configs/sparsity_retrain/{model}_{dataset}"
    path = REPO_ROOT / f"{rel_stem}.json"
    payload = build_retrain_config(model, dataset)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rel_stem


def build_train_command(*, job: RetrainJob, python: str, smoke: bool = False) -> str:
    train_cmd = [
        python,
        "run_model.py",
        "--task",
        "traffic_state_pred",
        "--model",
        job.model,
        "--dataset",
        job.dataset,
        "--config_file",
        job.config_file,
        "--batch_size",
        str(job.batch_size),
        "--seed",
        "0",
        "--saved_model",
        "true",
        "--train",
        "true",
    ]
    if smoke:
        wrapper = [
            python,
            "-u",
            "tools/sparsity/run_until_train_start.py",
            "--",
            *train_cmd,
        ]
        return " ".join(shlex.quote(x) for x in wrapper)
    return " ".join(shlex.quote(x) for x in train_cmd)


def training_started_in_text(text: str) -> bool:
    return any(marker in text for marker in TRAINING_STARTED_MARKERS)
