# Bigscity-LibCity — Sparsity & Mamba4Traffic Papers

Reproducible [LibCity](https://github.com/LibCity/Bigscity-LibCity) fork for two journal papers:

- **Sparsity journal** — sensor-masking robustness (DCRNN, D2STGNN, Trafformer, Mamba4Traffic)
- **Mamba4Traffic journal** — multivariate traffic forecasting on PEMS04/PEMS08
- **MCST-Mamba** — conference predecessor model (`MCSTMamba`)

LaTeX sources live in separate repos. Large artifacts (checkpoints, eval caches, predictions, raw data) are distributed via **Zenodo**.

## Artifacts (Zenodo)

| | Placeholder (replace after deposit) |
|--|--|
| Record page | https://zenodo.org/records/TBD |
| DOI | https://doi.org/10.5281/zenodo.TBD |
| Bundle file | `zenodo-bundle.tar.gz` (~50+ GB) |

```bash
# After you set ZENODO_RECORD_ID in artifacts/download.sh:
./artifacts/download.sh
```

Until the Zenodo deposit exists, `./artifacts/download.sh` exits with setup instructions.
See `artifacts/expected_layout/README.md` for the unpack layout.

Checkpoint lineages: `MODEL_MAP.md`, `artifacts/RETRAIN_TODO.md`.

## Environment

```bash
conda activate libcity-mamba
# Or follow mamba_environment_setup.md and requirements.txt
```

Requires Python 3.9, PyTorch 2.0+cu118, mamba-ssm 1.2.2.

## Training

```bash
# Sparsity / Mamba4Traffic model
python run_model.py --task traffic_state_pred --model MCSTMambaLST_Ablation \
  --dataset PEMSD4 --config_file configs/sparsity_retrain/MCSTMambaLST_Ablation_PEMSD4 \
  --batch_size 16 --seed 0 --saved_model true --train true

# MCST-Mamba
python run_model.py --task traffic_state_pred --model MCSTMamba --dataset PEMSD8 \
  --config_file configs/m4t/MCSTMamba --train true
```

## Evaluation

### Mamba4Traffic (full-data, vehicular metrics)

```bash
python evaluate_trained_model.py --model_dir libcity/cache/<run_dir>
python build_results_table.py
```

### Sparsity (masked inference)

```bash
python eval_sparsity.py --model_dir libcity/cache/DCRNN_PEMSD8_20260630_094749 \
  --sparsity 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --mask_seed 43

python tools/sparsity/build_results_cache.py
python scripts/generate_sparsity_paper_figures.py --lineage multiseed_new
```

## Model naming

See `MODEL_MAP.md` for LibCity keys vs paper display names (`MCSTMambaLST_Ablation` → Mamba4Traffic).

## M4T ablation retrain

```bash
python run_model.py --task traffic_state_pred --model MCSTMambaLST_Ablation \
  --dataset PEMSD8 --config_file configs/m4t/MCSTMambaLST_NoLSTM --train true
```

## Validation

See `VALIDATION.md` for the reproduction checklist. Manuscript update notes: `MANUSCRIPT_HANDOFF.md`.
