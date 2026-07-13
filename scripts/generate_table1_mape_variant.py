#!/usr/bin/env python3
"""
# NOTE: Updated manuscript removes MAPE/vMAPE from Table 1. This script is
# retained for reproducibility and historical comparison only.

Generate Table 1 tabular bodies (vMAPE and raw MAPE variants).

Methodology (consistent across all models):
  - Sparsity levels: 0--35% at 5% steps (8 levels)
  - Weights: w(rho) = exp(-20 rho^2), normalized
  - Avg column: arithmetic mean of flow / occupancy / speed
  - Source: sparcity_cache/*_PEMSD*_*-20260120_*
  - D2STGNN PEMS08: remap checkpoint channel order [occ, speed, flow]
    to canonical [flow, occupancy, speed]
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "sparcity_cache"
OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "table1"
CACHE_TAG = '20260120'
MAX_SPARSITY = 0.35

MODELS = [
    ('DCRNN', 'DCRNN'),
    ('D2STGNN', 'D2STGNN'),
    ('Mamba4Traffic', 'MCSTMambaLST_Ablation'),
    ('Trafformer', 'Trafformer'),
]
DATASETS = ['PEMSD4', 'PEMSD8']
CHAN_NAMES = ['flow', 'occupancy', 'speed']

D2STGNN_PEMSD8_CSV_SUFFIX = {
    'flow': 'speed',
    'occupancy': 'flow',
    'speed': 'occupancy',
}
D2STGNN_PEMSD8_NPZ_IDX = [2, 0, 1]


def find_run(prefix: str, ds: str) -> Path:
    hits = sorted(CACHE.glob(f'{prefix}_{ds}_*-{CACHE_TAG}_*'))
    if not hits:
        raise FileNotFoundError(f'No run for {prefix} {ds} ({CACHE_TAG})')
    return hits[-1]


def csv_col(prefix: str, display: str, ds: str, channel: str) -> str:
    if display == 'D2STGNN' and ds == 'PEMSD8':
        suffix = D2STGNN_PEMSD8_CSV_SUFFIX[channel]
    else:
        suffix = channel
    return f'{prefix}_{suffix}'


def npz_remap(display: str, ds: str) -> list[int]:
    if display == 'D2STGNN' and ds == 'PEMSD8':
        return D2STGNN_PEMSD8_NPZ_IDX
    return [0, 1, 2]


def load_sparsity_df(run: Path) -> pd.DataFrame:
    df = pd.read_csv(run / 'sparsity_results.csv').sort_values('sparsity')
    return df[df['sparsity'] <= MAX_SPARSITY]


def weights(rhos: list[float]) -> list[float]:
    w = [math.exp(-20 * r ** 2) for r in rhos]
    total = sum(w)
    return [x / total for x in w]


def weighted(vals: list[float], rhos: list[float]) -> float:
    w = weights(rhos)
    return sum(a * b for a, b in zip(w, vals))


def aggregate_csv(df: pd.DataFrame, display: str, ds: str, prefix: str) -> tuple[list[float], list[float]]:
    rhos = df['sparsity'].tolist()
    series = {c: [] for c in CHAN_NAMES}
    for rho in rhos:
        row = df[df['sparsity'] == rho].iloc[0]
        for c in CHAN_NAMES:
            series[c].append(float(row[csv_col(prefix, display, ds, c)]))

    bar = [weighted(series[c], rhos) for c in CHAN_NAMES]
    deg = [weighted([v / series[c][0] for v in series[c]], rhos) for c in CHAN_NAMES]
    bar.append(float(np.mean(bar)))
    deg.append(float(np.mean(deg)))
    return bar, deg


def masked_mape_per_channel(preds, gts, null_val=0.0):
    out = []
    for c in range(preds.shape[-1]):
        p = preds[..., c]
        g = gts[..., c]
        mask = (np.abs(g - null_val) > 1e-5).astype(np.float32)
        if mask.mean() < 1e-8:
            out.append(np.nan)
            continue
        mask = mask / mask.mean()
        denom = np.where(np.abs(g) < 1e-5, 1e-5, np.abs(g))
        mape = np.nan_to_num(np.abs(p - g) / denom * mask)
        out.append(float(mape.mean()) * 100.0)
    return out


def aggregate_mape(run: Path, display: str, ds: str, df: pd.DataFrame) -> tuple[list[float], list[float]]:
    npz = np.load(run / 'sparsity_results_predictions.npz', allow_pickle=True)
    rhos = df['sparsity'].tolist()
    remap = npz_remap(display, ds)
    series = {c: [] for c in CHAN_NAMES}
    for rho in rhos:
        key = f'{rho:.2f}'
        raw = masked_mape_per_channel(npz[f'{key}_predictions'], npz[f'{key}_ground_truth'])
        ch = [raw[i] for i in remap]
        for i, name in enumerate(CHAN_NAMES):
            series[name].append(ch[i])

    bar = [weighted(series[c], rhos) for c in CHAN_NAMES]
    deg = [weighted([v / series[c][0] for v in series[c]], rhos) for c in CHAN_NAMES]
    bar.append(float(np.mean(bar)))
    deg.append(float(np.mean(deg)))
    return bar, deg


def compute_tables() -> dict:
    tables = {}
    for ds in DATASETS:
        tables[ds] = {}
        for display, prefix in MODELS:
            run = find_run(prefix, ds)
            df = load_sparsity_df(run)
            mae = aggregate_csv(df, display, ds, 'vMAE')
            rmse = aggregate_csv(df, display, ds, 'vRMSE')
            vmape = aggregate_csv(df, display, ds, 'vMAPE')
            mape = aggregate_mape(run, display, ds, df)
            tables[ds][display] = {
                'mae': {'w': mae[0], 'd': mae[1]},
                'rmse': {'w': rmse[0], 'd': rmse[1]},
                'vmape': {'w': vmape[0], 'd': vmape[1]},
                'mape': {'w': mape[0], 'd': mape[1]},
            }
    return tables


def fmt_num(x: float) -> str:
    if x >= 100:
        return f'{x:.1f}'
    return f'{x:.3f}'


def fmt_deg(x: float) -> str:
    return f'{x:.2f}$\\times$'


def apply_rank_styles(values: list[float], deg: bool = False) -> list[str]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    best, second = order[0], order[1]
    out = [fmt_deg(v) if deg else fmt_num(v) for v in values]
    out[best] = f'\\textbf{{{out[best]}}}'
    if abs(values[best] - values[second]) > 1e-9:
        out[second] = f'\\underline{{{out[second]}}}'
    return out


def style_metric_rows(tables: dict, metric: str, field: str) -> dict[str, list[str]]:
    styled = {m: [] for m, _ in MODELS}
    for ds_idx, ds in enumerate(DATASETS):
        for col in range(4):
            col_vals = [tables[ds][m][metric][field][col] for m, _ in MODELS]
            col_styled = apply_rank_styles(col_vals, deg=(field == 'd'))
            for i, (m, _) in enumerate(MODELS):
                styled[m].append(col_styled[i])
    return styled


def write_tabular(
    out_path: Path,
    tables: dict,
    pct_metric: str,
    pct_label: str,
    pct_deg_label: str,
    comment_lines: list[str],
) -> None:
    mae_w = style_metric_rows(tables, 'mae', 'w')
    mae_d = style_metric_rows(tables, 'mae', 'd')
    rmse_w = style_metric_rows(tables, 'rmse', 'w')
    rmse_d = style_metric_rows(tables, 'rmse', 'd')
    pct_w = style_metric_rows(tables, pct_metric, 'w')
    pct_d = style_metric_rows(tables, pct_metric, 'd')

    header = [
        '    \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}cccccccccc@{}}',
        '    \\hline',
        '    \\multicolumn{2}{c}{Dataset} & \\multicolumn{4}{c}{PEMS04} & \\multicolumn{4}{c}{PEMS08} \\\\ \\hline',
        '    Model                  & Metric                & Flow      & Occupancy & Speed     & \\multicolumn{1}{c}{Avg} & Flow      & Occupancy & Speed     & Avg \\\\ \\hline\\\\[0.5em]',
    ]
    lines = comment_lines + header
    for idx, (model, _) in enumerate(MODELS):
        end_rule = '\\hline' if idx == len(MODELS) - 1 else '\\hline \\\\[0.1em]'
        lines.extend([
            f'    \\multirow{{6}}{{*}}{{{model}}} ',
            f'    & $\\bar{{v}}$MAE & {" & ".join(mae_w[model])} \\\\',
            f'    & \\multicolumn{{1}}{{c}}{{$\\bar{{D}}_{{MAE}}$}} & {" & ".join(mae_d[model])} \\\\[0.1em] \\\\[0.1em]',
            '',
            f'    & $\\bar{{v}}$RMSE & {" & ".join(rmse_w[model])} \\\\',
            f'    & \\multicolumn{{1}}{{c}}{{$\\bar{{D}}_{{RMSE}}$}} & {" & ".join(rmse_d[model])} \\\\[0.1em] \\\\[0.1em]',
            '',
            f'    & {pct_label} & {" & ".join(pct_w[model])} \\\\',
            f'    & \\multicolumn{{1}}{{c}}{{{pct_deg_label}}} & {" & ".join(pct_d[model])} \\\\ \\\\[0.1em]{end_rule}',
            '',
        ])
    lines.extend(['    \\end{tabular*}', ''])
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {out_path}')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = compute_tables()

    shared_comment = [
        '% Auto-generated by scripts/generate_table1_mape_variant.py',
        f'% Gaussian weights over 0--35% sparsity (8 levels); cache tag {CACHE_TAG}.',
        '% D2STGNN PEMS08 channels remapped from checkpoint order [occ, speed, flow].',
    ]

    write_tabular(
        OUT_DIR / 'tabular_vmape.tex',
        tables,
        'vmape',
        r'$\bar{v}$MAPE',
        r'$\bar{D}_{MAPE}$',
        shared_comment + ['% Loaded when \\def\\tableonepercentmode{vmape} in table1_config.tex.', ''],
    )
    write_tabular(
        OUT_DIR / 'tabular_mape.tex',
        tables,
        'mape',
        r'$\bar{\text{MAPE}}$',
        r'$\bar{D}_{\text{MAPE}}$',
        shared_comment + ['% Loaded when \\def\\tableonepercentmode{mape} in table1_config.tex.', ''],
    )


if __name__ == '__main__':
    main()
