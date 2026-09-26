# Paper model and checkpoint map

The sparsity benchmark uses the same public names in code, tables, figures, and caches.

| LibCity key | Paper name |
|---|---|
| `DCRNN` | DCRNN |
| `D2STGNN` | D2STGNN |
| `Mamba4Traffic` | Mamba4Traffic |
| `Trafformer` | Trafformer |


## Standardized checkpoint lineage

| Model | PEMS04 run | PEMS08 run |
|---|---|---|
| DCRNN | `DCRNN_PEMSD4_20260831_114637` | `DCRNN_PEMSD8_20260831_071603` |
| D2STGNN | `D2STGNN_PEMSD4_20260816_151002` | `D2STGNN_PEMSD8_20260816_150942` |
| Mamba4Traffic | `Mamba4Traffic_PEMSD4_20260816_151054` | `Mamba4Traffic_PEMSD8_20260816_151048` |
| Trafformer | `Trafformer_PEMSD4_20260921_064833` | `Trafformer_PEMSD8_20260817_104712` |

These are the default `standard` profile in `tools/sparsity/job_plan.py`. The manuscript uses the Trafformer/PEMS04 checkpoint with sample stride 1. The separately labeled channel-breakdown cache contains three dedicated Trafformer/PEMS04 operational-grid evaluations used solely to reproduce the six public companion plots.
