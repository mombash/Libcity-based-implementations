# Validation Report

Generated: 2026-06-29  
Environment: `libcity-mamba` (Python 3.9, mamba-ssm 1.2.2, PyTorch 2.0+cu118)  
Repo: `/data/rschadmin/papers_github/Bigscity-LibCity` (branch `papers-release`)  
Artifacts: `/data/rschadmin/papers_github/zenodo-bundle` (15 GB tarball)

Tolerance for metric comparison: **±0.001 vMAE_a** unless noted.

## Checklist

| Step | Status | Notes |
|------|--------|-------|
| 1. Environment | PASS | `import mamba_ssm`, `import libcity` OK |
| 2. Bundle layout | PASS | 565 files, 15.94 GB; symlinks under `artifacts/expected_layout/` |
| 3. Sparsity aggregate | PASS | `build_results_cache.py` → 48 multiseed_new runs, 0 missing; Table1 CSV matches bundled `results_cache` |
| 4. Sparsity figures | PARTIAL | Scripts ported; PDF generation requires matplotlib + bundled cache (run after Zenodo unpack) |
| 5. M4T table | PASS | `build_results_table.py` writes `vehicular_results_table.csv` from `_baselines/` |
| 6. Mask reproducibility | PASS | Regenerated seed-43 JSON bit-identical to working-fork originals |
| 7. Smoke train | NOT RUN | Full training skipped (time); configs validated via `MCSTMambaLST_Ablation.json` fix |
| 8. Smoke sparsity eval | NOT RUN | Checkpoints present; eval can be run per README |
| 9. Manuscript sync | PENDING | User updates `Sparsity-Journal-Paper/main.tex` to multiseed_new numbers |

## Known gaps (see `artifacts/RETRAIN_TODO.md`)

- Sept 2025 M4T `.m` checkpoints missing for most models; `_baselines/` CSVs are canonical fallback
- M4T ablation checkpoints (5× PEMSD8) not on disk; table numbers CSV-only
- GWNET, MTGNN, STGCN, GMAN, GTS checkpoints not on disk
- STAEformer row in M4T table: external benchmark CSVs not shipped (by design); `build_results_table.py` skips missing STAE paths

## Dual lineage verification

- **Sparsity:** `multiseed_new` only in Zenodo bundle (seeds 43–45, Jun 2026 checkpoints)
- **M4T:** `_baselines/` Sept 2025 exports + `MCSTMamba_PEMSD8` + shared `D2STGNN_PEMSD8` `.m` weights

## Reproduce validation

```bash
conda activate libcity-mamba
cd /data/rschadmin/papers_github/Bigscity-LibCity
./artifacts/download.sh artifacts/expected_layout   # after Zenodo DOI set

# Or use local bundle:
ln -sfn ../zenodo-bundle/eval/sparsity/sparcity_cache sparcity_cache
ln -sfn ../zenodo-bundle/eval/m4t/_baselines _baselines

python tools/sparsity/build_results_cache.py
python build_results_table.py
python masks/generate_multi_seed_masks.py
```
