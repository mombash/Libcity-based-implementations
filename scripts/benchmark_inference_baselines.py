#!/usr/bin/env python3
"""
Benchmark inference time for all baseline models in _baselines/PEMSD8.
Discovers models from the baseline folder: each subdir (e.g. DCRNN, GWNET) has
Normalized/Average/{timestamp}_{Model}_PEMSD8.csv; the timestamp maps to
libcity/cache/{Model}_PEMSD8_{YYYYMMDD}_{HHMMSS}/ and the latest checkpoint
(model_cache/{Model}_PEMSD8.m or model_cache/{Model}_PEMSD8_epoch{N}.tar).
Measures average inference time per batch (batch_size=4) over warmup + 50 runs.
Also reports parameter counts.

Usage: python benchmark_inference_baselines.py
"""
import sys
import os
import re
import time
import json
import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch
import numpy as np

BASELINES_ROOT = REPO_ROOT / 'artifacts' / 'expected_layout' / 'eval' / 'm4t' / '_baselines' / 'PEMSD8'
CACHE_ROOT = REPO_ROOT / 'libcity' / 'cache'

# Hardcoded exp_id for baselines where CSV-derived or latest cache is wrong
EXP_ID_OVERRIDES = {
    'MTGNN': 'MTGNN_PEMSD8_20250904_045138',
    'STGCN': 'STGCN_PEMSD8_20250904_045234',
}

# CSV filename pattern: 2025_09_06_22_30_03_DCRNN_PEMSD8.csv -> exp_id DCRNN_PEMSD8_20250906_223003
CSV_PATTERN = re.compile(r'^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(.+)_PEMSD8\.csv$')

NUM_WARMUP = 10
NUM_RUNS = 50
BATCH_SIZE = 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def discover_baseline_models():
    """
    List _baselines/PEMSD8 subdirs and for each find the CSV in Normalized/Average/.
    Return list of dicts: name (display), model_class, exp_id, cache_dir, checkpoint_path.
    """
    if not os.path.isdir(BASELINES_ROOT):
        return []
    models = []
    for model_name in sorted(os.listdir(BASELINES_ROOT)):
        subdir = os.path.join(BASELINES_ROOT, model_name)
        if not os.path.isdir(subdir):
            continue
        avg_dir = os.path.join(subdir, 'Normalized', 'Average')
        if not os.path.isdir(avg_dir):
            continue
        # Find CSV(s); use the one that matches timestamp -> cache
        csvs = glob.glob(os.path.join(avg_dir, '*_PEMSD8.csv'))
        exp_id = None
        for csv_path in sorted(csvs):
            base = os.path.basename(csv_path)
            m = CSV_PATTERN.match(base)
            if not m:
                continue
            y, mo, d, h, mi, s, mod = m.groups()
            if mod != model_name:
                continue
            exp_id = f"{model_name}_PEMSD8_{y}{mo}{d}_{h}{mi}{s}"
            break
        if model_name in EXP_ID_OVERRIDES:
            exp_id = EXP_ID_OVERRIDES[model_name]
        if not exp_id:
            continue
        cache_dir = os.path.join(CACHE_ROOT, exp_id)
        if not os.path.isdir(cache_dir) and model_name not in EXP_ID_OVERRIDES:
            # CSV timestamp may be evaluation date; use latest cache for this model
            prefix = f"{model_name}_PEMSD8_"
            candidates = [d for d in os.listdir(CACHE_ROOT)
                          if d.startswith(prefix) and os.path.isdir(os.path.join(CACHE_ROOT, d))]
            if not candidates:
                continue
            cache_dir = os.path.join(CACHE_ROOT, max(candidates))
            exp_id = os.path.basename(cache_dir)
        if not os.path.isdir(cache_dir):
            continue
        model_cache_dir = os.path.join(cache_dir, 'model_cache')
        if not os.path.isdir(model_cache_dir):
            continue
        # Prefer .m (final saved model), else latest epoch .tar
        model_file_m = os.path.join(model_cache_dir, f'{model_name}_PEMSD8.m')
        if os.path.isfile(model_file_m):
            checkpoint_path = model_file_m
        else:
            tars = glob.glob(os.path.join(model_cache_dir, f'{model_name}_PEMSD8_epoch*.tar'))
            if not tars:
                continue
            def epoch_num(p):
                b = os.path.basename(p)
                match = re.search(r'_epoch(\d+)\.tar$', b)
                return int(match.group(1)) if match else -1
            latest_tar = max(tars, key=epoch_num)
            checkpoint_path = latest_tar
        models.append({
            'name': model_name,
            'model': model_name,
            'exp_id': exp_id,
            'cache_dir': cache_dir,
            'checkpoint_path': checkpoint_path,
            'is_tar': checkpoint_path.endswith('.tar'),
        })
    return models


def load_checkpoint_state_dict(checkpoint_path, device):
    """Load model state dict from .m (tuple) or .tar (dict with model_state_dict)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, tuple):
        return checkpoint[0]
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint


def load_cached_config_overrides(cache_dir):
    """
    Load config_overrides from the run's eval_params.json if present.
    GTS (and similar models) depend on data_feature['train_data'].shape, which is
    determined by train_rate and dataset; the checkpoint was trained with a specific
    split, so we must use the same config when building the dataset.
    """
    eval_glob = os.path.join(cache_dir, 'evaluations', 'eval_*', 'eval_params.json')
    for path in sorted(glob.glob(eval_glob), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            overrides = data.get('config_overrides') or data.get('config')
            if overrides:
                return overrides
        except Exception:
            continue
    return None


print(f"Device: {device}")
print(f"Warmup: {NUM_WARMUP}, Runs: {NUM_RUNS}, Batch size: {BATCH_SIZE}")
print(f"Baselines root: {BASELINES_ROOT}")
print("=" * 80)

MODELS = discover_baseline_models()
if not MODELS:
    print("No baseline models found under _baselines/PEMSD8 (expected CSV in .../Normalized/Average/*_PEMSD8.csv)")
    sys.exit(1)

print(f"Discovered {len(MODELS)} models: {[m['name'] for m in MODELS]}\n")

results = {}

for minfo in MODELS:
    name = minfo['name']
    model_class = minfo['model']
    exp_id = minfo['exp_id']
    checkpoint_path = minfo['checkpoint_path']

    if not os.path.exists(checkpoint_path):
        print(f"[SKIP] {name}: checkpoint not found at {checkpoint_path}")
        continue

    print(f"\n[{name}] exp_id={exp_id}")
    print(f"  Checkpoint: {checkpoint_path}")

    try:
        from libcity.utils import get_model
        from libcity.config import ConfigParser
        from libcity.data import get_dataset

        other_args = {'exp_id': exp_id}
        # GTS: fc input size = (train_data.shape[0] - 2*kernal_size + 2)*16, so we must
        # use the same train_rate/split as the run that produced the checkpoint.
        cache_dir = minfo.get('cache_dir') or os.path.join(CACHE_ROOT, exp_id)
        if model_class == 'GTS':
            overrides = load_cached_config_overrides(cache_dir)
            if overrides:
                other_args = {**overrides, **other_args}
        config = ConfigParser('traffic_state_pred', model_class, 'PEMSD8',
                             other_args=other_args)
        dataset = get_dataset(config)
        train_data, valid_data, test_data = dataset.get_data()
        data_feature = dataset.get_data_feature()
        model = get_model(config, data_feature)

        state_dict = load_checkpoint_state_dict(checkpoint_path, device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())

        test_iter = iter(test_data)
        sample_batch = next(test_iter)
        sample_batch.to_tensor(device)
        for k in sample_batch.data:
            if isinstance(sample_batch.data[k], torch.Tensor):
                sample_batch.data[k] = sample_batch.data[k][:BATCH_SIZE]

        with torch.no_grad():
            for _ in range(NUM_WARMUP):
                _ = model.predict(sample_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(NUM_RUNS):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model.predict(sample_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        avg_ms = np.mean(times)
        std_ms = np.std(times)
        results[name] = {
            'params': int(num_params),
            'avg_ms': float(avg_ms),
            'std_ms': float(std_ms),
            'exp_id': exp_id,
        }

        print(f"  Parameters: {num_params:,}")
        print(f"  Inference:  {avg_ms:.2f} ± {std_ms:.2f} ms/batch")

        del model, sample_batch
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

# Summary table (order by discovery)
print("\n" + "=" * 80)
print(f"{'Model':<25} {'Params':>12} {'Inference (ms)':>20}")
print("-" * 60)
for minfo in MODELS:
    name = minfo['name']
    if name in results:
        r = results[name]
        print(f"{name:<25} {r['params']:>12,} {r['avg_ms']:>12.2f} ± {r['std_ms']:.2f}")

# Save JSON
out_path = str(REPO_ROOT / 'efficiency' / 'efficiency_results_baselines.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
