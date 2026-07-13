#!/usr/bin/env python3
"""Generate additional random sparsity masks for repeated-mask robustness analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "masks" / "multi_seed"

EXISTING_SEED = 42
NEW_SEEDS = [43, 44, 45]

DATASETS = {
    "PEMSD4": {
        "num_nodes": 307,
        "grids": {
            "paper_11level": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "breakdown_8level": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        },
    },
    "PEMSD8": {
        "num_nodes": 170,
        "grids": {
            "paper_11level": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "breakdown_8level": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        },
    },
}


def masked_nodes_for_rho(num_nodes: int, sparsity: float, seed_base: int = 42) -> np.ndarray:
    """Match eval_sparsity.masked_nodes_for_rho exactly."""
    rho = float(sparsity)
    if rho <= 0.0:
        return np.array([], dtype=int)
    n_zero = int(round(num_nodes * rho))
    n_zero = min(max(n_zero, 0), num_nodes)
    if n_zero == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed_base + int(round(rho * 100)))
    idx = rng.choice(num_nodes, n_zero, replace=False)
    return np.sort(idx.astype(int))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []

    for dataset, cfg in DATASETS.items():
        num_nodes = cfg["num_nodes"]
        for grid_name, rhos in cfg["grids"].items():
            for seed in NEW_SEEDS:
                masks_by_rho = {}
                for rho in rhos:
                    key = f"{rho:.2f}"
                    masks_by_rho[key] = masked_nodes_for_rho(num_nodes, rho, seed_base=seed).tolist()

                ref42 = set(masked_nodes_for_rho(num_nodes, 0.10, seed_base=EXISTING_SEED).tolist())
                ref_new = set(masks_by_rho["0.10"])
                if ref42 == ref_new:
                    raise RuntimeError(f"seed {seed} produced identical rho=0.10 mask as seed {EXISTING_SEED}")

                payload = {
                    "meta": {
                        "dataset": dataset,
                        "num_nodes": num_nodes,
                        "mask_seed": seed,
                        "baseline_seed": EXISTING_SEED,
                        "sparsity_grid": grid_name,
                        "sparsity_levels": rhos,
                        "sparsity_mode": "sweep",
                        "notes": (
                            "Additional random masks for repeated-mask robustness analysis "
                            "(reviewer R2 comment 5)."
                        ),
                    },
                    "masks_by_rho": masks_by_rho,
                }
                out_path = OUT / dataset / grid_name / f"masks_seed{seed}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                manifest.append(str(out_path.relative_to(OUT.parent)))

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "files": manifest,
                "seeds": NEW_SEEDS,
                "existing_seed": EXISTING_SEED,
                "usage": (
                    "Re-run eval_sparsity.py with --mask_seed <seed> for each file's meta.mask_seed. "
                    "Mask node lists match eval_sparsity.masked_nodes_for_rho."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} mask files under {OUT}")


if __name__ == "__main__":
    main()
