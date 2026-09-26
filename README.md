# Sparsity robustness benchmark for traffic forecasting

This repository is the public implementation and reproduction package for *A Sparsity Analysis of Traffic Forecasting Architectures: How Much Missing Data is Too Much?* It contains the LibCity-based implementations, fixed masks, final configurations, checkpoint map, evaluation code, and scripts used for every reported table and figure.

## Release gate

DOI `10.5281/zenodo.21944002` is the reserved record for this paper, but it is currently an unsubmitted draft and is not yet publicly downloadable. The artifact archive has been built and verified locally. This branch must not be released or committed as ready until the record has been published and `artifacts/download.sh` succeeds anonymously.

## Rebuild the submission

Once the public artifact has been downloaded by `artifacts/download.sh`, run from the repository root:

```bash
python reproduce.py manuscript
python reproduce.py channel-breakdown
```

The first command rebuilds all three manuscript tables and all eight manuscript figures; outputs go to `reproduced/manuscript/`. The second rebuilds the six public channel-wise breakdown PDFs in `reproduced/channel_breakdown/`.

The regenerated Table I/II/III LaTeX is byte-identical to the submitted source. Direct cache checks are D2STGNN/PEMS04 baseline `vMAE_a = 0.1592339332` (reported `0.159`) and weighted mean degradation `4.8396010232` (reported `4.84x`). Cache-based reproduction needs `numpy`, `pandas`, `matplotlib`, and `seaborn`.

## Reported evaluation protocol

The defaults in `eval_sparsity.py` and `tools/sparsity/job_plan.py` implement the paper path:

- training-split per-channel means and standard deviations only;
- all 12 forecast horizons;
- `vMAE_a` as the arithmetic mean of flow, occupancy, and speed vMAE;
- raw ground-truth zeros retained in reported vehicular metrics;
- flow--occupancy--speed channel order fixed from configuration;
- permutation search disabled unless explicitly requested;
- mask seeds 43, 44, and 45 and `k = round(rho N)`;
- static random raw-zero fill transformed into model input space;
- aggregate thresholds 0.33 (PEMS04) and 0.42 (PEMS08);
- D2STGNN gap 3;
- P10: 0--100% at 10% steps; P5: 0--35% at 5% steps;
- Gaussian alpha 20, with alpha 10 and 40 in the sensitivity figure.

The final Trafformer/PEMS04 manuscript checkpoint is `Trafformer_PEMSD4_20260921_064833` with sample stride 1. The default job profile is `standard`. A non-paper raw profile is available only when an external `SPARSITY_RAW_CHECKPOINT_MAP` is supplied.

## Re-run inference

After the artifact is public, place LibCity-format PEMS04/PEMS08 data under `raw_data/`, then run a released checkpoint directly:

```bash
python eval_sparsity.py \
  --model_dir artifacts/paper-release/checkpoints/DCRNN_PEMSD4_20260831_114637 \
  --log_file artifacts/paper-release/checkpoints/DCRNN_PEMSD4_20260831_114637/logs/DCRNN_PEMSD4_20260831_114637-DCRNN-PEMSD4-Aug-31-2026_11-46-37.log \
  --sparsity 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

No protocol flags are needed. Pass `--mask_seed 44` or `45` for the other reported masks. Exact training configurations are under `configs/sparsity_retrain/`.

## Alternative fill representations

Static random zero-fill remains the reported protocol. The code exposes only two fill representations for future study:

```bash
python eval_sparsity.py ... --fill_mode last_observation
python eval_sparsity.py ... --fill_mode training_mean
```

These switches are extensions; no results for them are reported. Spatially correlated failures, temporally correlated/intermittent failures, and non-zero or stuck faults remain future work rather than released experiment modes.

## Final checkpoints

| Model | PEMS04 | PEMS08 |
|---|---|---|
| DCRNN | `DCRNN_PEMSD4_20260831_114637` | `DCRNN_PEMSD8_20260831_071603` |
| D2STGNN | `D2STGNN_PEMSD4_20260816_151002` | `D2STGNN_PEMSD8_20260816_150942` |
| Mamba4Traffic | `Mamba4Traffic_PEMSD4_20260816_151054` | `Mamba4Traffic_PEMSD8_20260816_151048` |
| Trafformer | `Trafformer_PEMSD4_20260921_064833` | `Trafformer_PEMSD8_20260817_104712` |

`Mamba4Traffic` is the canonical public key. See [Mamba4Traffic under the unified protocol](docs/mamba4traffic.md) for training, checkpoint evaluation, and future artifact organization. This fork derives from [LibCity](https://github.com/LibCity/Bigscity-LibCity).
