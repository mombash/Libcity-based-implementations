#!/usr/bin/env python3
"""Build the paper artifact directory and its Zenodo tarball."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproduction.build_cache import build as build_cache  # noqa: E402

DOI = "10.5281/zenodo.21944002"
CHECKPOINTS = (
    "DCRNN_PEMSD4_20260831_114637",
    "DCRNN_PEMSD8_20260831_071603",
    "D2STGNN_PEMSD4_20260816_151002",
    "D2STGNN_PEMSD8_20260816_150942",
    "Mamba4Traffic_PEMSD4_20260816_151054",
    "Mamba4Traffic_PEMSD8_20260816_151048",
    "Trafformer_PEMSD4_20260921_064833",
    "Trafformer_PEMSD8_20260817_104712",
)
RUN_FILES = (
    "sparsity_results.csv",
    "eval_metadata.json",
    "training_config.json",
    "sparsity_results_masks.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scrub_private_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: scrub_private_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_private_paths(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        path = Path(value)
        parts = path.parts
        if "cache" in parts and "libcity" in parts:
            index = parts.index("cache")
            return (Path("checkpoints") / Path(*parts[index + 1 :])).as_posix()
        if "sparcity_cache" in parts:
            index = parts.index("sparcity_cache")
            return (Path("evaluation/runs") / Path(*parts[index + 1 :])).as_posix()
        return path.name
    return value


def copy_json_scrubbed(source: Path, destination: Path) -> None:
    payload = scrub_private_paths(json.loads(source.read_text(encoding="utf-8")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def selected_runs(manifests: list[Path]) -> list[str]:
    runs: set[str] = set()
    for manifest in manifests:
        with manifest.open(newline="", encoding="utf-8") as handle:
            runs.update(row["run_dir"] for row in csv.DictReader(handle))
    return sorted(runs)


def copy_evaluations(source_root: Path, bundle: Path, manifests: list[Path]) -> None:
    for run in selected_runs(manifests):
        source = source_root / "sparcity_cache" / run
        destination = bundle / "evaluation" / "runs" / run
        for name in RUN_FILES:
            source_file = source / name
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            destination_file = destination / name
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            if source_file.suffix == ".json":
                copy_json_scrubbed(source_file, destination_file)
            else:
                shutil.copy2(source_file, destination_file)


def copy_checkpoints(source_root: Path, bundle: Path) -> None:
    for run in CHECKPOINTS:
        source = source_root / "libcity" / "cache" / run
        destination = bundle / "checkpoints" / run
        weights = sorted((source / "model_cache").glob("*.m"))
        logs = sorted((source / "logs").glob("*.log"))
        if len(weights) != 1 or len(logs) != 1:
            raise RuntimeError(f"Expected one released weight and log for {run}")
        (destination / "model_cache").mkdir(parents=True, exist_ok=True)
        (destination / "logs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights[0], destination / "model_cache" / weights[0].name)
        shutil.copy2(logs[0], destination / "logs" / logs[0].name)
        summary = source / "logs" / "training_summary.txt"
        if summary.is_file():
            shutil.copy2(summary, destination / "logs" / summary.name)


def copy_release_inputs(bundle: Path, figures: Path) -> None:
    configs = bundle / "configurations"
    configs.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO / "configs" / "sparsity_retrain").glob("*.json")):
        shutil.copy2(source, configs / source.name)
    shutil.copytree(REPO / "masks" / "multi_seed", bundle / "masks" / "multi_seed")
    shutil.copy2(
        REPO / "masks" / "multi_seed" / "manifest.json",
        bundle / "masks" / "manifest.json",
    )
    shutil.copytree(figures, bundle / "figures" / "channel_breakdown")
    shutil.copy2(REPO / "reproduce.py", bundle / "reproduce.py")
    shutil.copytree(
        REPO / "reproduction",
        bundle / "reproduction",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def write_bundle_readme(bundle: Path) -> None:
    (bundle / "README.md").write_text(
        "# Sparsity traffic forecasting artifacts\n\n"
        "This archive accompanies DOI `10.5281/zenodo.21944002`. It contains "
        "eight standardized checkpoints, the 48 manuscript evaluation result sets, "
        "three dedicated Trafformer/PEMS04 breakdown evaluations used only for "
        "the six public companion plots, fixed masks, exact training "
        "configurations, derived caches, and the table/figure rebuild scripts. "
        "`MANIFEST.json` gives the SHA-256 digest of every archived file.\n\n"
        "## Reproduce\n\n"
        "Install `numpy`, `pandas`, `matplotlib`, and `seaborn`, change into this "
        "unpacked `paper-release` directory, and run:\n\n"
        "```bash\npython reproduce.py manuscript\n"
        "python reproduce.py channel-breakdown\n```\n\n"
        "The first command rebuilds all three manuscript tables and all eight "
        "manuscript figures in `reproduced/manuscript/`. The second rebuilds the "
        "six channel-wise PDFs in `reproduced/channel_breakdown/`. The same "
        "commands work from the GitHub repository after `artifacts/download.sh`.\n\n"
        "## Configuration and masks\n\n"
        "The exact eight model/dataset JSON files are under `configurations/`. "
        "The reported channel order is flow, occupancy, speed; scaling uses fixed "
        "per-channel training-split statistics; all 12 horizons are evaluated; "
        "and raw ground-truth zeros are retained. Masks use seeds 43, 44, and 45, "
        "with `k = round(rho N)`, and are under `masks/multi_seed/` for P10 "
        "(0--100% by 10%) and P5 (0--35% by 5%). Zero-fill is the reported input "
        "representation.\n\n"
        "The manuscript cache uses Trafformer/PEMS04 checkpoint "
        "`20260921_064833` with sample stride 1. The separately labeled "
        "`channel_breakdown` cache contains the three dedicated evaluations "
        "used to create the six public companion plots.\n",
        encoding="utf-8",
    )


def write_manifest(bundle: Path) -> None:
    files = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(bundle).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "version": "1.0.0",
        "doi": DOI,
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    (bundle / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def build_bundle(source_root: Path, bundle: Path, archive: Path, figures: Path) -> None:
    if bundle.exists() or archive.exists():
        raise FileExistsError("Choose new --output-dir and --archive paths")
    bundle.mkdir(parents=True)
    manuscript = REPO / "artifacts" / "run_manifests" / "manuscript_runs.csv"
    channels = REPO / "artifacts" / "run_manifests" / "channel_breakdown_runs.csv"
    manifest_dir = bundle / "evaluation" / "manifests"
    manifest_dir.mkdir(parents=True)
    shutil.copy2(manuscript, manifest_dir / manuscript.name)
    shutil.copy2(channels, manifest_dir / channels.name)
    build_cache(
        manuscript,
        source_root / "sparcity_cache",
        bundle / "evaluation" / "cache" / "manuscript",
    )
    build_cache(
        channels,
        source_root / "sparcity_cache",
        bundle / "evaluation" / "cache" / "channel_breakdown",
        allow_dedicated_trafformer_breakdown=True,
    )
    copy_evaluations(source_root, bundle, [manuscript, channels])
    copy_checkpoints(source_root, bundle)
    copy_release_inputs(bundle, figures)
    write_bundle_readme(bundle)
    write_manifest(bundle)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle, arcname="paper-release")
    print(f"Bundle: {bundle}")
    print(f"Archive: {archive}")
    print(f"SHA-256: {sha256(archive)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=os.environ.get("LIBCITY_DEVELOPMENT_ROOT"),
        required="LIBCITY_DEVELOPMENT_ROOT" not in os.environ,
        help="Development tree containing the selected runs and checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--channel-figures",
        type=Path,
        default=REPO / "figures" / "channel_breakdown",
    )
    args = parser.parse_args()
    build_bundle(
        args.source_root.resolve(),
        args.output_dir.resolve(),
        args.archive.resolve(),
        args.channel_figures.resolve(),
    )


if __name__ == "__main__":
    main()
