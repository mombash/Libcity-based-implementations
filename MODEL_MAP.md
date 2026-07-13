# Model Name Mapping

LibCity `--model` keys vs paper display names for the Sparsity and Mamba4Traffic journal papers.

## Sparsity Journal Paper (4-model benchmark)

| LibCity key | Paper display name | Notes |
|-------------|-------------------|-------|
| `DCRNN` | DCRNN | |
| `D2STGNN` | D2STGNN | Channel order remapped in table scripts |
| `Trafformer` | Trafformer | |
| `MCSTMambaLST_Ablation` | Mamba4Traffic | Same implementation as M4T paper |

Figure code may alias `MCSTMambaLST_Ablation` as `MCSTMambaLST` internally.

## Mamba4Traffic Journal Paper

| LibCity key | Paper display name | Notes |
|-------------|-------------------|-------|
| `MCSTMambaLST_Ablation` | Mamba4Traffic | Full model (all ablation flags false) |
| `MCSTMambaLST` | Mamba4Traffic (architecture) | FiLM + LSTM variant |
| `MCSTMamba` | MCST-Mamba | Conference predecessor |
| `GWNET` | GWNet | |
| `MTGNN` | MTGNN | |
| `STGCN` | STGCN | |
| `D2STGNN` | D2STGNN | |
| `GMAN` | GMAN | |
| `DCRNN` | DCRNN | |
| `GTS` | GTS | |
| STAEformer | STAEformer | External CSV benchmarks only (not in this repo) |

## Ablation variants (M4T Tab. ablation, PEMSD8)

| LibCity key | Disabled component |
|-------------|-------------------|
| `MCSTMambaLST_NoEmbeddings` | Embeddings |
| `MCSTMambaLST_NoLSTM` | LSTM branch |
| `MCSTMambaLST_NoMTemporal` | Mamba temporal path |
| `MCSTMambaLST_NoMamba` | Mamba blocks |
| `MCSTMambaLST_NoTemporal` | Temporal pathway |

## Checkpoint lineages

Two lineages coexist; do not mix them in eval scripts.

| Lineage | Used for | Example run dir |
|---------|----------|-----------------|
| `sparsity_multiseed` | Sparsity paper (mask seeds 43–45; Jun 23–30 2026 retrains) | `DCRNN_PEMSD4_20260630_094809` |
| `m4t_sept2025` | M4T paper full-data benchmark | `_baselines/` CSVs; sparse `.m` weights |

See `artifacts/MANIFEST.json` and `artifacts/RETRAIN_TODO.md` for on-disk availability.
