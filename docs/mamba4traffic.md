# Mamba4Traffic under the unified sparsity protocol

`Mamba4Traffic` is a first-class LibCity model key in this repository. The commands below use the standardized protocol of the sparsity benchmark: flow--occupancy--speed channel order, training-split per-channel z-scores, 12 input and output steps, sample stride 1, and static random raw-zero masking for evaluation.

These instructions describe the protocol used in this repository. They do not claim that a separate Mamba4Traffic publication used the same experimental protocol.

## Train under the unified protocol

Place the LibCity-format datasets under `raw_data/PEMSD4/` and `raw_data/PEMSD8/`, then run:

```bash
python run_model.py --task traffic_state_pred --model Mamba4Traffic --dataset PEMSD4 --config_file configs/sparsity_retrain/Mamba4Traffic_PEMSD4 --seed 0
python run_model.py --task traffic_state_pred --model Mamba4Traffic --dataset PEMSD8 --config_file configs/sparsity_retrain/Mamba4Traffic_PEMSD8 --seed 0
```

These commands train new models under the shared configuration; exact published weights are supplied by the sparsity artifact archive.

## Evaluate the released checkpoints

Download and unpack the Zenodo artifact with `bash artifacts/download.sh`. For PEMS04:

```bash
for seed in 43 44 45; do
  python eval_sparsity.py \
    --model_dir artifacts/paper-release/checkpoints/Mamba4Traffic_PEMSD4_20260816_151054 \
    --mask_seed "$seed" \
    --sparsity 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --out_csv "output/mamba4traffic_pems04_seed${seed}.csv" \
    --no_save_predictions --no_save_series_csvs --no_visualize_topology --no_plot_metrics
done
```

Use `Mamba4Traffic_PEMSD8_20260816_151048` and an output name containing `pems08` for PEMS08. No metric, scaler, horizon, channel-permutation, or fill-mode override is needed; the evaluator defaults implement the reported protocol.

## Separate future artifact

A future Mamba4Traffic-specific release can add its paper-specific configurations and commands without changing the sparsity benchmark. Keep its DOI, checksum manifest, and downloader under `artifacts/mamba4traffic/`, and keep its paper-specific configurations under `configs/mamba4traffic/`. The existing `artifacts/MANIFEST.json`, `configs/sparsity_retrain/`, and `standard` checkpoint profile remain scoped to the sparsity paper.
