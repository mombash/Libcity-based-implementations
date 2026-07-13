# Checkpoint Retrain TODO

Inventory refreshed 2026-07-13 after Jun 23–30 retrains + Jul 5–6 multiseed evals.

Legend: **OK** = best-epoch `.m` on disk and in Zenodo bundle | **MISSING** = CSV fallback + retrain needed

## Sparsity multiseed (authoritative for updated Sparsity manuscript)

| Model | Dataset | Run directory | Status |
|-------|---------|---------------|--------|
| DCRNN | PEMSD4 | `DCRNN_PEMSD4_20260630_094809` | OK |
| DCRNN | PEMSD8 | `DCRNN_PEMSD8_20260630_094749` | OK |
| D2STGNN | PEMSD4 | `D2STGNN_PEMSD4_20260623_102537` | OK |
| D2STGNN | PEMSD8 | `D2STGNN_PEMSD8_20251202_210455` | OK |
| MCSTMambaLST_Ablation | PEMSD4 | `MCSTMambaLST_Ablation_PEMSD4_20260629_174146` | OK |
| MCSTMambaLST_Ablation | PEMSD8 | `MCSTMambaLST_Ablation_PEMSD8_20260630_094729` | OK |
| Trafformer | PEMSD4 | `Trafformer_PEMSD4_20260624_071738` | OK |
| Trafformer | PEMSD8 | `Trafformer_PEMSD8_20260629_060448` | OK |

Canonical pointers: `tools/sparsity/job_plan.py` `MODELS`.  
Eval: mask seeds 43–45, Jul 5–6 2026 (`sparsity_analysis/results_cache/run_manifest.csv`).

## M4T sept2025 (authoritative for Mamba4Traffic manuscript)

| Model | Dataset | Run directory | Status |
|-------|---------|---------------|--------|
| MCSTMambaLST_Ablation | PEMSD4 | `MCSTMambaLST_Ablation_PEMSD4_20250925_214013` | MISSING — use `_baselines/` CSV |
| MCSTMambaLST_Ablation | PEMSD8 | `MCSTMambaLST_Ablation_PEMSD8_20250925_203759` | MISSING — use `_baselines/` CSV |
| DCRNN | PEMSD4 | `DCRNN_PEMSD4_20250911_143658` | MISSING — use `_baselines/` CSV |
| DCRNN | PEMSD8 | `DCRNN_PEMSD8_20250904_044845` | MISSING — use `_baselines/` CSV |
| D2STGNN | PEMSD4 | `D2STGNN_PEMSD4_20250911_144426` | MISSING — use `_baselines/` CSV |
| D2STGNN | PEMSD8 | `D2STGNN_PEMSD8_20251202_210455` | OK (shared with sparsity) |

## M4T-only baselines

| Model | Dataset | Status |
|-------|---------|--------|
| GWNET | PEMSD4, PEMSD8 | MISSING — use `_baselines/` CSV |
| MTGNN | PEMSD4, PEMSD8 | MISSING — use `_baselines/` CSV |
| STGCN | PEMSD4, PEMSD8 | MISSING — use `_baselines/` CSV |
| GMAN | PEMSD4, PEMSD8 | MISSING — use `_baselines/` CSV |
| GTS | PEMSD4, PEMSD8 | MISSING — use `_baselines/` CSV |
| MCSTMamba | PEMSD4 | MISSING — retrain from `MCSTMamba.json` |
| MCSTMamba | PEMSD8 | OK — `MCSTMamba_PEMSD8_20250918_133423` |

## M4T ablation checkpoints (PEMSD8, Tab. ablation)

| Model | Status |
|-------|--------|
| MCSTMambaLST_NoEmbeddings | MISSING — CSV numbers only |
| MCSTMambaLST_NoLSTM | MISSING — CSV numbers only |
| MCSTMambaLST_NoMTemporal | MISSING — CSV numbers only |
| MCSTMambaLST_NoMamba | MISSING — CSV numbers only |
| MCSTMambaLST_NoTemporal | MISSING — CSV numbers only |

## Retrain commands (when needed)

```bash
conda activate libcity-mamba
cd /path/to/Bigscity-LibCity

python run_model.py --task traffic_state_pred --model MCSTMambaLST_Ablation \
  --dataset PEMSD4 --config_file configs/sparsity_retrain/MCSTMambaLST_Ablation_PEMSD4 \
  --batch_size 16 --seed 0 --saved_model true --train true
```

See `configs/sparsity_retrain/` and `configs/m4t/` for frozen hyperparameters.
