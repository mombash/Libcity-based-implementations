#!/usr/bin/env python3
"""Package Zenodo artifact bundle from working fork artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

SRC = Path("/data/rschadmin/Bigscity-LibCity")
BUNDLE = Path("/data/rschadmin/papers_github/zenodo-bundle")

# Canonical sparsity checkpoints from tools/sparsity/job_plan.py MODELS
# (Jun 23–30 2026 retrains; multiseed evals Jul 5–6 2026).
SPARSITY_CKPTS = [
    "DCRNN_PEMSD4_20260630_094809",
    "DCRNN_PEMSD8_20260630_094749",
    "D2STGNN_PEMSD4_20260623_102537",
    "D2STGNN_PEMSD8_20251202_210455",
    "MCSTMambaLST_Ablation_PEMSD4_20260629_174146",
    "MCSTMambaLST_Ablation_PEMSD8_20260630_094729",
    "Trafformer_PEMSD4_20260624_071738",
    "Trafformer_PEMSD8_20260629_060448",
]

M4T_CKPTS = [
    "MCSTMamba_PEMSD8_20250918_133423",
    "D2STGNN_PEMSD8_20251202_210455",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_checkpoint_minimal(run_name: str, dest_root: Path) -> None:
    src_run = SRC / "libcity/cache" / run_name
    if not src_run.exists():
        return
    dest = dest_root / run_name / "model_cache"
    dest.mkdir(parents=True, exist_ok=True)
    mc = src_run / "model_cache"
    if mc.exists():
        for m in mc.glob("*.m"):
            shutil.copy2(m, dest / m.name)
    cfg = src_run / "config.json"
    if cfg.exists():
        shutil.copy2(cfg, dest.parent / "config.json")


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    # raw_data
    for ds in ("PEMSD4", "PEMSD8"):
        shutil.copytree(SRC / "raw_data" / ds, BUNDLE / "raw_data" / ds, dirs_exist_ok=True)

    ckpt_sparsity = BUNDLE / "checkpoints" / "sparsity_multiseed"
    ckpt_m4t = BUNDLE / "checkpoints" / "m4t_sept2025"
    for run in SPARSITY_CKPTS:
        copy_checkpoint_minimal(run, ckpt_sparsity)
    for run in M4T_CKPTS:
        copy_checkpoint_minimal(run, ckpt_m4t)

    # sparcity_cache canonical multiseed_new (48 runs)
    manifest_csv = SRC / "sparsity_analysis/results_cache/run_manifest.csv"
    dest_cache = BUNDLE / "eval/sparsity/sparcity_cache"
    dest_cache.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open() as f:
        for row in csv.DictReader(f):
            if row["lineage"] != "multiseed_new":
                continue
            run_dir = row["run_dir"]
            src = SRC / "sparcity_cache" / run_dir
            if src.exists():
                shutil.copytree(src, dest_cache / run_dir, dirs_exist_ok=True)

    # results_cache
    shutil.copytree(
        SRC / "sparsity_analysis/results_cache",
        BUNDLE / "eval/sparsity/sparsity_analysis/results_cache",
        dirs_exist_ok=True,
    )

    # M4T baselines
    shutil.copytree(SRC / "_baselines", BUNDLE / "eval/m4t/_baselines", dirs_exist_ok=True)

    # Prediction npz files live inside each sparcity_cache run dir
    # (sparsity_results_predictions.npz). No separate npz/ tree — avoids
    # duplicating ~50GB+ of predictions.

    # README
    (BUNDLE / "README.md").write_text(
        "# Zenodo artifact bundle\n\n"
        "Lineages:\n"
        "- sparsity_multiseed: Jun 23–30 2026 retrained checkpoints + Jul 5–6 multiseed evals (seeds 43–45)\n"
        "- m4t_sept2025: Mamba4Traffic full-data `_baselines/` CSVs (+ any available .m weights)\n\n"
        "Unpack and symlink per artifacts/expected_layout/README.md in the code repo.\n"
    )

    # MANIFEST with checksums (top-level files only for speed; large dirs summarized)
    files_meta = []
    for p in sorted(BUNDLE.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(BUNDLE).as_posix()
        size = p.stat().st_size
        # Skip hashing very large files in full tree for speed - hash npz and checkpoints
        if size > 50_000_000:
            digest = "SKIPPED_LARGE"
        else:
            digest = sha256_file(p)
        files_meta.append({"path": rel, "size_bytes": size, "sha256": digest})

    manifest = {
        "version": "1.0.0",
        "zenodo_doi": "TBD",
        "bundle_root": str(BUNDLE),
        "file_count": len(files_meta),
        "total_bytes": sum(x["size_bytes"] for x in files_meta),
        "files": files_meta,
    }
    (BUNDLE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Bundle at {BUNDLE}, files={len(files_meta)}, size={manifest['total_bytes']/1e9:.2f} GB")


if __name__ == "__main__":
    main()
