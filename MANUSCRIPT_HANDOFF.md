# Manuscript Handoff — Sparsity Journal (multiseed_new)

## What was regenerated

| Artifact | Path |
|----------|------|
| Aggregated results cache | `zenodo-bundle/eval/sparsity/sparsity_analysis/results_cache/` |
| Figures (multiseed_new) | `output/figures/multiseed_new/*.pdf` (6 PDFs) |
| Table builders | `tools/sparsity/build_results_cache.py`, `scripts/generate_table1_mape_variant.py` |

## Your action: update `Sparsity-Journal-Paper/main.tex`

1. Replace Table 1 numbers with **multiseed_new** lineage (mask seeds 43–45, Jun 2026 checkpoints).
2. **Remove MAPE and vMAPE columns** from Table 1 per editorial decision.
3. Source data:
   - `tables/multiseed_new_table1_baseline_vmae_mean_std.csv` in results_cache
   - `tables/multiseed_new_table2_{paper,breakdown}_vmae_a_wide.csv` for extended tables
4. Copy regenerated PDFs from `output/figures/multiseed_new/` into `Sparsity-Journal-Paper/figures/` as needed.

## vMAPE / MAPE scripts (kept, not in paper)

`scripts/generate_table1_mape_variant.py` still generates `tabular_mape.tex` and `tabular_vmape.tex`.
See header comment: metrics removed from manuscript due to occupancy instability (MAPE) and
final table design (vMAPE de-emphasized). Scripts retained for reproducibility.

## Key baseline vMAE_a (multiseed mean, rho=0%)

Read from bundled cache after unpack:

```bash
python -c "
import pandas as pd
df = pd.read_csv('sparsity_analysis/results_cache/tables/multiseed_new_table1_baseline_vmae_mean_std.csv')
print(df)
"
```

## M4T paper

No LaTeX changes required in this repo. M4T numbers remain from Sept 2025 `_baselines/` exports.
`vehicular_results_table.csv` generated at repo root from symlinked `_baselines/`.

## Zenodo (placeholder)

| | Placeholder |
|--|--|
| Record page | https://zenodo.org/records/TBD |
| DOI | https://doi.org/10.5281/zenodo.TBD |

Upload `zenodo-bundle.tar.gz` (or the unpacked `zenodo-bundle/` directory) to Zenodo, then replace TBD in:
- `README.md`
- `artifacts/MANIFEST.json`
- `artifacts/download.sh` (`ZENODO_RECORD_ID`)
