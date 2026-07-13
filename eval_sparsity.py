#!/usr/bin/env python
"""
MCSTMambaLST_Ablation sparsity evaluation for PEMSD8.

This script supports two modes for evaluating model robustness to sensor sparsity:

1. SWEEP MODE (default): --sparsity_mode=sweep
   Sweeps through sparsity levels (0.0 to 1.0), randomly masking sensors.
   Usage: python eval_sparsity.py --model_dir /path/to/run --sparsity 0.0 0.1 0.2 ...

2. SCHEME MODE: --sparsity_mode=scheme
   Uses predefined sensor removal schemes from a JSON file.
   Default scheme file: raw_data/PEMSD8/sensor_removal_schemes.json
   Usage: python eval_sparsity.py --model_dir /path/to/run --sparsity_mode scheme [--schemes hub_top_10pct ...]
   Or with custom file: python eval_sparsity.py --model_dir /path/to/run --sparsity_mode scheme \
          --scheme_file /path/to/custom_schemes.json
   
   The scheme JSON format:
   {
     "scheme_name": {
       "sensors": [list of sensor indices to mask],
       "count": number_of_sensors,
       "description": "...",
       "rationale": "..."
     },
     ...
   }
   
   Similar schemes are grouped by prefix (hub_, peripheral_, etc.) into subfolders.

At each sparsity level/scheme, this script computes:
- Global MAE / RMSE / MAPE over all horizons, nodes, channels
- Per-horizon MAE / RMSE (12-step array)
- Last-horizon MAE / RMSE
- Vehicular z-score-based metrics per channel (vMAE, vRMSE, vMAPE)
- Channel-averaged vehicular metrics (vMAE_a, vRMSE_a, vMAPE_a)

Vehicular metrics follow the Traffic Mamba (Mamba4Traffic) paper:
- Targets are z-scored per channel using training-set mean/std (by default)
- Metrics are computed on the normalized scale, by default at the last horizon.
"""

import argparse
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import itertools
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import topology visualization functions
try:
    from visualize_mae_topology import (
        load_rel_file,
        create_graph,
        calculate_mae_per_sensor_from_arrays,
        visualize_mae_on_topology,
        visualize_mae_comparison,
        create_mae_ranking_plot,
        create_schemes_overview_plot,
        print_top_bottom_sensors,
    )
    TOPOLOGY_VIS_AVAILABLE = True
except ImportError as e:
    print(f"[warn] Could not import visualize_mae_topology: {e}")
    TOPOLOGY_VIS_AVAILABLE = False


def setup_logging(log_file_path):
    """
    Set up logging to both console and file.
    """
    # Create logger
    logger = logging.getLogger("eval_sparsity")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


class TeeOutput:
    """
    Utility class to tee stdout/stderr to both console and a file.
    """
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        self.file.close()
        
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.close()


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_model

def plot_example_timeseries(
    horizons,
    gt,
    preds_by_rho,
    channel_name,
    example_idx,
    node_idx,
    out_prefix
):
    """
    Plot GT vs predictions across sparsity for a single example/node/channel.

    horizons: 1D array of horizon steps (len T_out)
    gt:       1D array [T_out]
    preds_by_rho: dict {rho_value (float) -> 1D array [T_out]}
    channel_name: 'flow' / 'occupancy' / 'speed'
    out_prefix:   e.g. 'sparsity_testing/sparsity_curve_PEMSD8_full_groups_ts'
    """
    plt.figure(figsize=(9, 5))
    plt.plot(horizons, gt, color="k", linewidth=2.0, label="GT")

    # Sort by rho so legend order is nice
    for rho in sorted(preds_by_rho.keys()):
        series = preds_by_rho[rho]
        plt.plot(
            horizons,
            series,
            marker="o",
            linestyle="--",
            linewidth=1.0,
            markersize=4,
            label=f"rho={rho:.1f}"
        )

    plt.xlabel("Horizon (step)")
    if channel_name == "flow":
        plt.ylabel("flow value")
    elif channel_name == "occupancy":
        plt.ylabel("occupancy value")
    else:
        plt.ylabel("speed value")

    plt.title(
        f"GT vs predictions across sparsity\n"
        f"example={example_idx}, node={node_idx}, channel={channel_name}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    out_path = (
        f"{out_prefix}_example{example_idx}_node{node_idx}_{channel_name}.png"
    )
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved example timeseries plot to: {out_path}")

def mask_sensors(x, sparsity, rng):
    """
    Zero out random nodes in the input.
    x: [B, T, N, F] (numpy)
    sparsity in [0,1]: fraction of nodes to mask.

    Semantics in this helper alone:
    for a given call, we randomly select round(N * sparsity)
    sensor indices and set ALL their features over ALL input timesteps to zero.

    NOTE: the main script now uses a fixed node set per rho, applied manually,
    and does not call this function anymore. Kept for reference.
    """
    if sparsity <= 0.0:
        return x

    B, T, N, F = x.shape
    n_zero = int(round(N * sparsity))
    n_zero = min(max(n_zero, 0), N)

    mask = np.ones(N, dtype=np.float32)
    if n_zero > 0:
        idx = rng.choice(N, n_zero, replace=False)
        mask[idx] = 0.0

    # Broadcast to [B, T, N, F]
    return x * mask[None, None, :, None]


def masked_nodes_for_rho(num_nodes, sparsity, seed_base=42):
    """
    Deterministic node set for a given sparsity rho:
    - always the same indices for the same rho and num_nodes.
    """
    rho = float(sparsity)
    if rho <= 0.0:
        return np.array([], dtype=int)
    n_zero = int(round(num_nodes * rho))
    n_zero = min(max(n_zero, 0), num_nodes)
    if n_zero == 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed_base + int(round(rho * 100)))
    idx = rng.choice(num_nodes, n_zero, replace=False)
    idx = np.sort(idx.astype(int))
    return idx


def extract_config_from_log(log_path):
    """
    Parse a LibCity training log to extract the FULL config dict.
    
    The log typically contains a line like:
    2025-12-01 20:29:55,366 - INFO - {'task': 'traffic_state_pred', 'model': 'MCSTMambaLST', ...}
    
    Returns:
        dict: Full config dict parsed from the log, or empty dict if not found.
    """
    import ast
    
    config_dict = {}
    
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                # Look for a line containing a dict starting with {'task':
                if "{'task':" in line or '{"task":' in line:
                    # Extract the dict portion - find the first { and match to the end
                    start_idx = line.find("{")
                    if start_idx == -1:
                        continue
                    dict_str = line[start_idx:].strip()
                    
                    # Handle device(...) which isn't valid Python literal
                    # Replace device(type='cuda', index=0) with a string placeholder
                    dict_str = re.sub(r"device\([^)]+\)", "'cuda:0'", dict_str)
                    
                    try:
                        config_dict = ast.literal_eval(dict_str)
                        if isinstance(config_dict, dict) and 'task' in config_dict:
                            break
                    except (ValueError, SyntaxError):
                        # Try to find end of dict more carefully
                        brace_count = 0
                        end_idx = start_idx
                        for i, ch in enumerate(line[start_idx:]):
                            if ch == '{':
                                brace_count += 1
                            elif ch == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = start_idx + i + 1
                                    break
                        if end_idx > start_idx:
                            dict_str = line[start_idx:end_idx]
                            dict_str = re.sub(r"device\([^)]+\)", "'cuda:0'", dict_str)
                            try:
                                config_dict = ast.literal_eval(dict_str)
                                if isinstance(config_dict, dict) and 'task' in config_dict:
                                    break
                            except (ValueError, SyntaxError):
                                pass
    except Exception as e:
        print(f"[warn] Failed to parse config from log: {e}")
    
    return config_dict


def extract_task_model_dataset_from_log(log_path):
    """
    Parse a LibCity training log to extract task, model and dataset.
    Falls back to inferring from filename when not explicitly present.
    
    Returns:
        tuple: (task, model, dataset, config_overrides)
            - config_overrides is a dict of model-specific parameters from the log
    """
    # First try to get the full config
    full_config = extract_config_from_log(log_path)
    
    task = full_config.get('task')
    model = full_config.get('model')
    dataset = full_config.get('dataset')
    
    # If full config parsing failed, fall back to regex parsing
    if not task or not model or not dataset:
        try:
            with open(log_path, "r", errors="ignore") as f:
                for line in f:
                    line_l = line.strip()
                    # Key: value
                    m_task = re.search(r'^\s*task\s*[:=]\s*([A-Za-z0-9_]+)', line_l)
                    m_model = re.search(r'^\s*model\s*[:=]\s*([A-Za-z0-9_]+)', line_l)
                    m_data = re.search(r'^\s*dataset\s*[:=]\s*([A-Za-z0-9_]+)', line_l)
                    # JSON-like
                    j_task = re.search(r'"task"\s*:\s*"([^"]+)"', line_l)
                    j_model = re.search(r'"model"\s*:\s*"([^"]+)"', line_l)
                    j_data = re.search(r'"dataset"\s*:\s*"([^"]+)"', line_l)
                    if task is None and m_task:
                        task = m_task.group(1)
                    if model is None and m_model:
                        model = m_model.group(1)
                    if dataset is None and m_data:
                        dataset = m_data.group(1)
                    if task is None and j_task:
                        task = j_task.group(1)
                    if model is None and j_model:
                        model = j_model.group(1)
                    if dataset is None and j_data:
                        dataset = j_data.group(1)
                    if task and model and dataset:
                        break
        except Exception:
            pass

    # Fallback: infer from filename segments "...-<model>-<dataset>-....log"
    if model is None or dataset is None:
        basename = os.path.basename(log_path)
        parts = basename.split("-")
        if len(parts) >= 3:
            if model is None:
                model = parts[1]
            if dataset is None:
                dataset = parts[2].split(".")[0]

    # Fallback task default commonly used in LibCity for these models
    if task is None:
        task = "traffic_state_pred"

    # Pass all config parameters from the log as overrides
    # This ensures the model architecture matches exactly what was used during training
    # Exclude keys that should not be overridden (runtime/path-specific)
    exclude_keys = {
        'device', 'gpu_id', 'exp_id', 'train', 'saved_model',
        'cache_dataset', 'log_level', 'log_every',
        'hyper_tune', 'load_best_epoch',
    }
    
    config_overrides = {k: v for k, v in full_config.items() 
                        if k not in exclude_keys and not isinstance(v, dict)}
    
    if config_overrides:
        print(f"[config] Extracted {len(config_overrides)} config parameters from training log")
    
    return task, model, dataset, config_overrides


def compute_metrics_global(preds, gts):
    """
    preds, gts: numpy arrays with same shape [B_tot, T_out, N, C]
    Returns MAE, RMSE, MAPE (percentage), aggregated over all dims.
    """
    diff = preds - gts
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    denom = np.abs(gts)
    denom = np.where(denom < 1e-5, 1e-5, denom)
    mape = float(np.mean(np.abs(diff) / denom) * 100.0)
    return mae, rmse, mape


def compute_metrics_global_masked(preds, gts, null_val=0.0):
    """
    LibCity-style masked MAE / RMSE / MAPE.

    - preds, gts: [B_tot, T_out, N, C]
    - Positions where gts == null_val (≈ 0) are ignored in the average.
    - MAPE here is in *fraction* (0.1 ≈ 10%), like LibCity's masked_MAPE.
    """
    diff = preds - gts

    # Build mask: 1 where valid, 0 where GT is "null"
    if np.isnan(null_val):
        mask = ~np.isnan(gts)
    else:
        mask = np.abs(gts - null_val) > 1e-5  # treat near-zero as null as well

    mask = mask.astype(np.float32)
    if mask.mean() < 1e-8:
        raise RuntimeError("All targets are masked out in compute_metrics_global_masked.")
    mask = mask / mask.mean()  # LibCity style re-normalization

    # Masked MAE
    mae = np.abs(diff)
    mae = np.nan_to_num(mae * mask)
    mae = float(mae.mean())

    # Masked RMSE
    mse = diff ** 2
    mse = np.nan_to_num(mse * mask)
    rmse = float(np.sqrt(mse.mean()))

    # Masked MAPE (fraction, not %)
    denom = np.abs(gts)
    denom = np.where(denom < 1e-5, 1e-5, denom)
    mape = np.abs(diff) / denom
    mape = np.nan_to_num(mape * mask)
    mape = float(mape.mean())  # 0.1 ≈ 10%

    return mae, rmse, mape

def compute_metrics_masked_zscore(preds, gts, mu_vec, std_vec, null_val=0.0, eps=1e-5):
    """
    Mask is built on RAW gts: |gts - null_val| > eps.
    Metrics are computed AFTER per-channel z-scoring with (mu_vec,std_vec).
    Returns masked_zMAE, masked_zRMSE, masked_zMAPE  (z-MAPE is a fraction).
    """
    # 1) mask on raw
    if np.isnan(null_val):
        mask = ~np.isnan(gts)
    else:
        mask = np.abs(gts - null_val) > eps
    mask = mask.astype(np.float32)
    if mask.mean() < 1e-8:
        raise RuntimeError("All targets masked in compute_metrics_masked_zscore.")
    mask = mask / mask.mean()

    # 2) flatten and z-score
    B,T,N,C = gts.shape
    g = gts.reshape(-1, C).astype(np.float64)
    p = preds.reshape(-1, C).astype(np.float64)
    m = mask.reshape(-1, C).astype(np.float64)

    mu = np.asarray(mu_vec, dtype=np.float64)
    sd = np.asarray(std_vec, dtype=np.float64)
    sd = np.where(sd < 1e-8, 1e-8, sd)

    g_t = (g - mu) / sd
    p_t = (p - mu) / sd
    d_t = p_t - g_t

    zmae = float(np.nan_to_num(np.abs(d_t) * m).mean())
    zrmse = float(np.sqrt(np.nan_to_num((d_t**2) * m).mean()))

    denom = np.abs(g_t)
    denom = np.where(denom < 1e-5, 1e-5, denom)
    zmape = float(np.nan_to_num((np.abs(d_t) / denom) * m).mean())
    return zmae, zrmse, zmape

def compute_metrics_per_horizon(preds, gts):
    """
    preds, gts: [B_tot, T_out, N, C]
    Returns:
      mae_h:  np.ndarray shape [T_out]
      rmse_h: np.ndarray shape [T_out]
    These are directly comparable to LibCity's horizon-wise MAE/RMSE.
    """
    diff = preds - gts  # [B, T, N, C]
    mae_h = np.mean(np.abs(diff), axis=(0, 2, 3))
    rmse_h = np.sqrt(np.mean(diff ** 2, axis=(0, 2, 3)))
    return mae_h, rmse_h


def compute_on_nodes(preds, gts, nodes_idx, null_val=0.0):
    """
    Restrict metrics to a subset of nodes.

    preds, gts: [B, T, N, C]
    nodes_idx: 1D array/list of node indices
    """
    if nodes_idx is None:
        return dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan,
                    mMAE=np.nan, mRMSE=np.nan, mMAPE=np.nan)
    nodes_idx = np.asarray(nodes_idx, dtype=int)
    if nodes_idx.size == 0:
        return dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan,
                    mMAE=np.nan, mRMSE=np.nan, mMAPE=np.nan)

    p = preds[:, :, nodes_idx, :]
    y = gts[:, :, nodes_idx, :]
    MAE, RMSE, MAPE = compute_metrics_global(p, y)
    mMAE, mRMSE, mMAPE = compute_metrics_global_masked(p, y, null_val=null_val)
    return dict(MAE=MAE, RMSE=RMSE, MAPE=MAPE,
                mMAE=mMAE, mRMSE=mRMSE, mMAPE=mMAPE)


def inverse_transform_like_evaluator(arr_np, scaler, feature_dim, output_dim):
    """
    Emulate evaluator behavior: place output_dim channels into a full feature_dim tensor,
    inverse_transform with the shared scaler, then slice back the first output_dim channels.
    """
    if scaler is None or type(scaler).__name__ == "NoneScaler":
        return arr_np.astype(np.float32)
    B, T, N, C = arr_np.shape
    assert C == output_dim
    full = np.zeros((B, T, N, feature_dim), dtype=np.float32)
    full[..., :output_dim] = arr_np.astype(np.float32)
    try:
        inv_full = scaler.inverse_transform(torch.from_numpy(full)).numpy()
        return inv_full[..., :output_dim].astype(np.float32)
    except Exception:
        return arr_np.astype(np.float32)


def compute_train_channel_stats(train_loader, output_dim, scaler=None, feature_dim=None):
    """
    Compute per-channel mean and std over the entire TRAINING target set y.

    This matches the Traffic Mamba vehicular definition, where normalization
    uses training-set statistics per channel.
    """
    sum_y = np.zeros(output_dim, dtype=np.float64)
    sum_y2 = np.zeros(output_dim, dtype=np.float64)
    n_total = 0

    for batch in train_loader:
        y_np = np.stack(batch["y"], axis=0).astype(np.float32)  # [B, T_out, N, F_all]
        y_np = y_np[..., :output_dim]                           # [B, T_out, N, C]

        # Inverse like evaluator when feature_dim is known
        if feature_dim is not None and scaler is not None and type(scaler).__name__ != "NoneScaler":
            y_np = inverse_transform_like_evaluator(y_np, scaler, feature_dim, output_dim).astype(np.float64)
        y_np = y_np.astype(np.float64)

        y_flat = y_np.reshape(-1, output_dim)                   # [B*T_out*N, C]

        sum_y += y_flat.sum(axis=0)
        sum_y2 += np.square(y_flat).sum(axis=0)
        n_total += y_flat.shape[0]

    if n_total == 0:
        raise RuntimeError("Training loader is empty when computing channel stats.")

    mean = sum_y / n_total
    var = sum_y2 / n_total - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean, std


def compute_vehicular_metrics(preds_all, gts_all, output_dim,
                              mu_vec=None, std_vec=None,
                              scope="last"):
    """
    Vehicular metrics (vMAE, vRMSE, vMAPE)

    - They are z-scored per channel: x_tilde = (x - mu) / std.
    - Metrics are computed on the normalized values:
        vMAE_c  = MAE( y_tilde^c, yhat_tilde^c )
        vRMSE_c = RMSE( y_tilde^c, yhat_tilde^c )
        vMAPE_c = MAPE( y_tilde^c, yhat_tilde^c ).

    scope:
        "last" -> use only the last horizon (horizon T_out)
        "all"  -> use all horizons
    """
    # Select horizon scope
    if scope == "last":
        preds_use = preds_all[:, -1:, :, :]  # [B, 1, N, C]
        gts_use = gts_all[:, -1:, :, :]
    else:
        preds_use = preds_all
        gts_use = gts_all

    Btot, T_out, N_nodes, C = preds_use.shape
    assert C == output_dim, f"Expected {output_dim} channels, got {C}"

    pred_flat = preds_use.reshape(-1, C)
    gts_flat = gts_use.reshape(-1, C)

    eps = 1e-8

    # If no mu/std provided, fall back to test-set stats on this slice
    if mu_vec is None or std_vec is None:
        mu_vec = gts_flat.mean(axis=0)
        std_vec = gts_flat.std(axis=0)

    mu_vec = np.asarray(mu_vec, dtype=np.float64)
    std_vec = np.asarray(std_vec, dtype=np.float64)
    std_vec = np.where(std_vec < eps, eps, std_vec)

    vMAE_c = []
    vRMSE_c = []
    vMAPE_c = []

    for c in range(C):
        y_c = gts_flat[:, c].astype(np.float64)
        yhat_c = pred_flat[:, c].astype(np.float64)

        y_tilde = (y_c - mu_vec[c]) / std_vec[c]
        yhat_tilde = (yhat_c - mu_vec[c]) / std_vec[c]

        diff_tilde = y_tilde - yhat_tilde

        vmae_c = float(np.mean(np.abs(diff_tilde)))
        vrmse_c = float(np.sqrt(np.mean(diff_tilde ** 2)))

        denom = np.abs(y_tilde)
        denom = np.where(denom < 1e-5, 1e-5, denom)

        vmape_c = float(np.mean(np.abs(diff_tilde) / denom))

        vMAE_c.append(vmae_c)
        vRMSE_c.append(vrmse_c)
        vMAPE_c.append(vmape_c)

    vMAE_c = np.array(vMAE_c)
    vRMSE_c = np.array(vRMSE_c)
    vMAPE_c = np.array(vMAPE_c)

    vMAE_a = float(vMAE_c.mean())
    vRMSE_a = float(vRMSE_c.mean())
    vMAPE_a = float(vMAPE_c.mean())

    return vMAE_c, vRMSE_c, vMAPE_c, vMAE_a, vRMSE_a, vMAPE_a


def compute_vehicular_metrics_per_horizon(preds_all, gts_all, output_dim,
                                          mu_vec=None, std_vec=None):
    """
    Compute vMAE per horizon step (H1, H2, ..., H12).
    
    Uses the SAME normalization as compute_vehicular_metrics to ensure consistency.
    
    Args:
        preds_all, gts_all: [B, T_out, N, C] arrays
        output_dim: number of channels (C)
        mu_vec, std_vec: normalization parameters (same as used in aggregate metrics)
        
    Returns:
        vMAE_per_horizon: np.ndarray of shape [T_out] containing channel-averaged vMAE per horizon
    """
    Btot, T_out, N_nodes, C = preds_all.shape
    assert C == output_dim, f"Expected {output_dim} channels, got {C}"
    
    eps = 1e-8
    
    # If no mu/std provided, compute from all data
    if mu_vec is None or std_vec is None:
        flat = gts_all.reshape(-1, C)
        mu_vec = flat.mean(axis=0)
        std_vec = flat.std(axis=0)
    
    mu_vec = np.asarray(mu_vec, dtype=np.float64)
    std_vec = np.asarray(std_vec, dtype=np.float64)
    std_vec = np.where(std_vec < eps, eps, std_vec)
    
    vMAE_per_horizon = []
    
    for h in range(T_out):
        preds_h = preds_all[:, h, :, :]  # [B, N, C]
        gts_h = gts_all[:, h, :, :]      # [B, N, C]
        
        vMAE_c_h = []
        for c in range(C):
            y_c = gts_h[:, :, c].astype(np.float64).flatten()
            yhat_c = preds_h[:, :, c].astype(np.float64).flatten()
            
            y_tilde = (y_c - mu_vec[c]) / std_vec[c]
            yhat_tilde = (yhat_c - mu_vec[c]) / std_vec[c]
            
            vmae_c = float(np.mean(np.abs(y_tilde - yhat_tilde)))
            vMAE_c_h.append(vmae_c)
        
        # Channel-averaged vMAE for this horizon
        vMAE_a_h = float(np.mean(vMAE_c_h))
        vMAE_per_horizon.append(vMAE_a_h)
    
    return np.array(vMAE_per_horizon)


def compute_evaluator_norm_stats(preds_all, gts_all, scope="last"):
    """
    Emulate LibCity TrafficStateEvaluator z-score normalization:
    - Compute mean/std from the combined [y_true, y_pred] slice for each channel
    - Stats are computed over all axes except the last (feature) axis
    """
    # Select horizon scope
    if scope == "last":
        preds_use = preds_all[:, -1:, :, :]  # [B, 1, N, C]
        gts_use = gts_all[:, -1:, :, :]
    else:
        preds_use = preds_all
        gts_use = gts_all
    # Flatten along batch/time/node, keep channel
    p = preds_use.reshape(-1, preds_use.shape[-1]).astype(np.float64)
    g = gts_use.reshape(-1, gts_use.shape[-1]).astype(np.float64)
    combined = np.concatenate([g, p], axis=0)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean, std

def compute_vehicular_on_nodes(preds, gts, output_dim, mu_vec, std_vec, scope, nodes_idx):
    """
    Vehicular metrics restricted to a subset of nodes.
    """
    if nodes_idx is None:
        nan3 = np.array([np.nan, np.nan, np.nan])
        return nan3, nan3, nan3, np.nan, np.nan, np.nan
    nodes_idx = np.asarray(nodes_idx, dtype=int)
    if nodes_idx.size == 0:
        nan3 = np.array([np.nan, np.nan, np.nan])
        return nan3, nan3, nan3, np.nan, np.nan, np.nan

    p = preds[:, :, nodes_idx, :]
    y = gts[:, :, nodes_idx, :]
    return compute_vehicular_metrics(p, y, output_dim,
                                     mu_vec=mu_vec, std_vec=std_vec,
                                     scope=scope)


def inverse_transform_first_output_channels(arr_np, scaler, output_dim, mu_train, std_train):
    """
    Robust inverse-transform for arrays shaped [B, T, N, C] where C == output_dim.
    Tries scaler.inverse_transform with per-channel stats, otherwise falls back to train-set mu/std.
    """
    # Prefer scaler-based inverse if it clearly has per-channel statistics
    try:
        if scaler is not None and type(scaler).__name__ != "NoneScaler":
            sc_mu = getattr(scaler, "mean", None) or getattr(scaler, "mean_", None)
            sc_std = getattr(scaler, "std", None) or getattr(scaler, "std_", None)
            if sc_mu is not None and sc_std is not None:
                sc_mu_np = np.asarray(sc_mu).reshape(-1)
                sc_std_np = np.asarray(sc_std).reshape(-1)
                if sc_mu_np.size >= output_dim and sc_std_np.size >= output_dim:
                    # Use library inverse_transform on full tensor
                    return scaler.inverse_transform(torch.from_numpy(arr_np)).numpy()
    except Exception:
        # Fall through to the manual path
        pass

    # Manual per-channel inverse using training-set stats (fallback)
    mu = np.asarray(mu_train, dtype=np.float64).reshape(-1)[:output_dim]
    sd = np.asarray(std_train, dtype=np.float64).reshape(-1)[:output_dim]
    sd = np.where(sd < 1e-8, 1e-8, sd)
    return (arr_np * sd[None, None, None, :]) + mu[None, None, None, :]


def group_schemes_by_prefix(scheme_names):
    """
    Group scheme names by their prefix (e.g., hub_, peripheral_, chain_).
    Returns dict: {group_name -> [scheme_names]}
    """
    groups = {}
    for name in scheme_names:
        # Extract prefix: everything before the last underscore+number or known suffixes
        parts = name.split('_')
        if len(parts) >= 2:
            # Common patterns: hub_top_10pct, peripheral_30pct, random_10pct, chain_longest
            # Group by first part or first two parts depending on pattern
            if parts[0] in ['hub', 'peripheral', 'random', 'betweenness', 'floating', 'path', 'cluster', 'chain', 'interconnected', 'critical']:
                if parts[0] == 'interconnected':
                    group = 'interconnected_chain'
                elif parts[0] == 'floating':
                    group = 'floating_hub'
                else:
                    group = parts[0]
            else:
                group = parts[0]
        else:
            group = name
        
        if group not in groups:
            groups[group] = []
        groups[group].append(name)
    
    return groups


_EPOCH_VAL_LOSS_RE = re.compile(
    r"Epoch \[(\d+)/\d+\].*?train_loss: ([\d.]+), val_loss: ([\d.]+)",
)


def _collect_epoch_checkpoints(model_cache_dir):
    """Return {epoch: path} for *_epoch{N}.tar files under model_cache."""
    epoch_tar = {}
    for root, _dirs, files in os.walk(model_cache_dir):
        for fn in files:
            if not fn.lower().endswith(".tar"):
                continue
            m = re.search(r"_epoch(\d+)\.tar$", fn, flags=re.IGNORECASE)
            if m:
                epoch_tar[int(m.group(1))] = os.path.join(root, fn)
    return epoch_tar


def _best_val_epoch_from_log(log_path):
    """Parse training log and return (best_epoch, best_val_loss) or (None, None)."""
    if not log_path or not os.path.isfile(log_path):
        return None, None

    best_epoch = None
    best_val = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _EPOCH_VAL_LOSS_RE.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            val_loss = float(match.group(3))
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
    return best_epoch, best_val


def resolve_checkpoint_path(*, model_cache_dir, log_path, explicit_checkpoint=None):
    """
    Resolve a checkpoint .tar for sparsity eval.

    Default policy: use the epoch with minimum validation loss recorded in the
    training log (matches LibCity ``load_best_epoch`` training behavior).
    Falls back to the latest saved epoch only when the log cannot be parsed.
    """
    if explicit_checkpoint is not None:
        return os.path.abspath(explicit_checkpoint), {
            "checkpoint_selection": "explicit",
            "best_val_epoch": None,
            "best_val_loss": None,
        }

    epoch_tar = _collect_epoch_checkpoints(model_cache_dir)
    if not epoch_tar:
        tar_files = []
        for root, _dirs, files in os.walk(model_cache_dir):
            for fn in files:
                if fn.lower().endswith(".tar"):
                    tar_files.append(os.path.join(root, fn))
        if not tar_files:
            raise RuntimeError(f"No .tar checkpoints found under {model_cache_dir}")
        tar_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        chosen = tar_files[0]
        print(
            "[warn] No *_epoch{N}.tar checkpoints found; using newest .tar by mtime: "
            f"{os.path.basename(chosen)}"
        )
        return os.path.abspath(chosen), {
            "checkpoint_selection": "mtime_fallback",
            "best_val_epoch": None,
            "best_val_loss": None,
        }

    best_epoch, best_val = _best_val_epoch_from_log(log_path)
    if best_epoch is not None:
        if best_epoch in epoch_tar:
            return epoch_tar[best_epoch], {
                "checkpoint_selection": "best_val",
                "best_val_epoch": best_epoch,
                "best_val_loss": best_val,
            }

        available = sorted(epoch_tar)
        lower = [ep for ep in available if ep <= best_epoch]
        if lower:
            picked = lower[-1]
        else:
            picked = min(available, key=lambda ep: abs(ep - best_epoch))
        print(
            f"[warn] Best-val epoch {best_epoch} has no saved checkpoint; "
            f"using nearest saved epoch {picked} instead."
        )
        return epoch_tar[picked], {
            "checkpoint_selection": "nearest_saved_to_best_val",
            "best_val_epoch": best_epoch,
            "best_val_loss": best_val,
            "checkpoint_epoch": picked,
        }

    latest_epoch = max(epoch_tar)
    print(
        "[warn] Could not parse validation losses from training log; "
        f"falling back to latest saved epoch {latest_epoch}."
    )
    return epoch_tar[latest_epoch], {
        "checkpoint_selection": "latest_fallback",
        "best_val_epoch": None,
        "best_val_loss": None,
        "checkpoint_epoch": latest_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=None,
        help="Optional explicit checkpoint path. If not set, best validation-loss epoch from the training log is used."
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to a LibCity cache run directory containing logs/, model_cache/, etc."
    )
    parser.add_argument(
        "--log_file", default=None,
        help="Optional explicit training log path. If not set, the latest .log under model_dir/logs is used."
    )
    # Sparsity mode selection
    parser.add_argument(
        "--sparsity_mode", choices=["sweep", "scheme"], default="sweep",
        help="Sparsity evaluation mode: 'sweep' (default) sweeps through sparsity levels; 'scheme' uses predefined sensor removal schemes from a JSON file."
    )
    parser.add_argument(
        "--scheme_file", default=os.path.join(ROOT_DIR, "raw_data/PEMSD8/sensor_removal_schemes.json"),
        help="Path to JSON file with sensor removal schemes. Default: raw_data/PEMSD8/sensor_removal_schemes.json"
    )
    parser.add_argument(
        "--schemes", type=str, nargs="*", default=None,
        help="Optional list of specific scheme names to run (for scheme mode). If not set, all schemes in the file are run."
    )
    parser.add_argument(
        "--sparsity", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out_csv", default=None,
        help="Output CSV path. If not set, saved under the run directory in ./sparcity_cache."
    )
    parser.add_argument(
        "--mask_seed", type=int, default=42,
        help="Deterministic seed base for selecting masked sensors per sparsity."
    )
    parser.add_argument(
        "--mask_zero_mode",
        choices=["normalized", "inverse"],
        default="normalized",
        help=(
            "How to set masked inputs for the first output_dim channels: "
            "'normalized' writes the normalized value that corresponds to raw zero (default); "
            "'inverse' performs inverse->zero->re-scale to achieve raw-zero semantics."
        ),
    )
    parser.add_argument(
        "--vehicular_norm", choices=["evaluator", "train", "test"], default="evaluator",
        help="Z-score baseline: 'evaluator' (dataset scaler stats), 'train' (paper-style), or 'test' (on-slice)."
    )
    parser.add_argument(
        "--vehicular_scope", choices=["last", "all"], default="last",
        help="Horizon scope for vehicular metrics: 'last' (paper) or 'all'."
    )
    parser.add_argument(
        "--permute_channels",
        choices=["auto", "none", "pred", "gt"],
        default="auto",
        help=(
            "Resolve possible channel order mismatch for DCRNN checkpoints. "
            "'pred': permute predictions to dataset/scaler order. "
            "'gt': permute ground-truth to model order before inverse-scaling. "
            "'auto': try both on first batch (post-inverse MAE) and pick the better. "
            "'none': disable permutation."
        ),
    )
    parser.add_argument(
        "--permute_inputs",
        choices=["auto", "none"],
        default="auto",
        help=(
            "For DCRNN, try permuting ONLY the first output_dim feature channels of input X to match the "
            "checkpoint's expected training order. Auto evaluates all 6 perms on the first batch."
        ),
    )
    parser.add_argument(
        "--sparsity_stats_npz", default=None,
        help="Optional npz from precompute_node_sparsity.py; if set, compute sparse/non-sparse group metrics."
    )
    parser.add_argument(
        "--sparse_frac_thresh", type=float, default=None,
        help="Override threshold inside sparsity_stats_npz meta (fraction of zeros to mark a node as sparse)."
    )
    parser.add_argument(
        "--group_mode", choices=["flow_occ_any", "flow_only", "occ_only", "all3_any"],
        default="flow_occ_any",
        help="How to define sparse nodes from zero_frac[channel]."
    )
    parser.add_argument(
        "--save_masks_json", default=None,
        help="Path to write per-rho masked node indices (default: out_csv with '_masks.json')."
    )
    parser.add_argument(
        "--plot_outputs", action="store_true",
        help="If set, plot GT vs predictions for one sample/node/channel across all sparsities."
    )
    parser.add_argument(
        "--plot_example_idx", type=int, default=0,
        help="Index in the concatenated test set example to plot (default: 0)."
    )
    parser.add_argument(
        "--plot_node_idx", type=int, default=0,
        help="Node index (0 .. num_nodes-1) to plot."
    )
    parser.add_argument(
        "--plot_channel", type=str, default="flow",
        help="Channel to plot: 'flow', 'occupancy', 'speed' or a numeric index 0/1/2."
    )
    parser.add_argument(
        "--plot_global_outputs", action="store_true",
        help="If set, plot GT vs globally averaged predictions (over all nodes, batches, channels) across sparsities."
    )
    parser.add_argument(
        "--eval_from_npz",
        default=None,
        help=(
            "Optional: path to a LibCity predictions .npz (evaluate_cache) to compute metrics directly "
            "without running the model. Expects keys like 'prediction'/'pred' and 'truth'/'gt'."
        ),
    )
    parser.add_argument(
        "--track_masked_groups", action="store_true",
        help="Track metrics separately for nodes that are masked vs unmasked at each sparsity level."
    )
    parser.add_argument(
        "--save_series_csvs", action="store_true", default=True,
        help="If set, export per-sensor CSVs of Input vs GT vs Prediction across the test set, for each sparsity."
    )
    parser.add_argument(
        "--no_save_series_csvs", action="store_true",
        help="Disable per-sensor CSV export."
    )
    parser.add_argument(
        "--save_series_dir", default=None,
        help="Directory to write per-sensor CSVs (default: out_csv with '_series')."
    )
    parser.add_argument(
        "--plot_metrics", action="store_true", default=True,
        help="If set, generate a single plot with MAE/RMSE/MAPE vs sparsity (enabled by default)."
    )
    parser.add_argument(
        "--no_plot_metrics", action="store_true",
        help="Disable the sparsity vs MAE per-variable plot."
    )
    parser.add_argument(
        "--metrics_mode", choices=["base", "masked", "vehicular"], default="base",
        help="Which set of metrics to plot: base (MAE/RMSE/MAPE), masked_*, or vehicular averages."
    )
    parser.add_argument(
        "--debug_per_channel", action="store_true",
        help="If set, print per-channel means/MAE/RMSE summaries (after inverse-scaling)."
    )
    parser.add_argument(
        "--only_zero", action="store_true",
        help="If set, run only at sparsity=0.0 (overrides --sparsity list)."
    )
    
    # Topology visualization arguments
    parser.add_argument(
        "--visualize_topology", action="store_true", default=True,
        help="Generate per-sensor MAE topology visualizations (default: enabled)."
    )
    parser.add_argument(
        "--no_visualize_topology", action="store_true",
        help="Disable per-sensor MAE topology visualizations."
    )
    parser.add_argument(
        "--topology_channel", type=int, default=2,
        help="Channel index for topology visualization (0, 1, or 2; default: 2/flow)."
    )
    parser.add_argument(
        "--topology_channel_name", type=str, default=None,
        help="Custom name for the topology visualization channel (e.g., 'Flow', 'Speed'). Auto-detected if not set."
    )
    parser.add_argument(
        "--rel_file", type=str, default=None,
        help="Path to .rel file for network topology. Auto-detected from dataset if not set."
    )

    args = parser.parse_args()
    if args.only_zero:
        args.sparsity = [0.0]
    
    # Handle --no_visualize_topology flag
    if args.no_visualize_topology:
        args.visualize_topology = False
    
    # Handle --no_save_series_csvs flag
    if args.no_save_series_csvs:
        args.save_series_csvs = False

    if args.no_plot_metrics:
        args.plot_metrics = False

    # Validate scheme mode arguments
    if args.sparsity_mode == "scheme":
        if not os.path.isfile(args.scheme_file):
            parser.error(f"Scheme file not found: {args.scheme_file}")

    channel_map = {"flow": 0, "occupancy": 1, "speed": 2}
    try:
        plot_channel_idx = int(args.plot_channel)
    except ValueError:
        plot_channel_idx = channel_map.get(args.plot_channel.lower(), 0)
    
    device = torch.device(args.device)

    # Resolve training log and extract task/model/dataset, then construct ConfigParser
    logs_dir = os.path.join(args.model_dir, "logs")
    if args.log_file is not None:
        log_path = args.log_file
    else:
        # pick latest .log under logs_dir
        cand_logs = []
        if os.path.isdir(logs_dir):
            for fn in os.listdir(logs_dir):
                if fn.endswith(".log"):
                    cand_logs.append(os.path.join(logs_dir, fn))
        if not cand_logs:
            raise RuntimeError(f"No .log files found under {logs_dir}. Provide --log_file explicitly.")
        cand_logs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        log_path = cand_logs[0]

    task, model_name, dataset_name, log_config_overrides = extract_task_model_dataset_from_log(log_path)
    print(f"From training log:")
    print(f"  log_path = {log_path}")
    print(f"  task     = {task}")
    print(f"  model    = {model_name}")
    print(f"  dataset  = {dataset_name}")
    is_dcrnn = (str(model_name).upper() == "DCRNN")

    # Create timestamped run directory under ./sparcity_cache
    # Format: {model_name}_{dataset}_{original_timestamp}-{current_timestamp}
    current_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract original timestamp from model_dir basename
    # Expected formats: "run_YYYYMMDD_HHMMSS", "MODEL_DATASET_YYYYMMDD_HHMMSS", etc.
    model_dir_basename = os.path.basename(os.path.normpath(args.model_dir))
    original_ts_match = re.search(r'(\d{8}_\d{6})', model_dir_basename)
    if original_ts_match:
        original_ts = original_ts_match.group(1)
    else:
        # Fallback: use the model_dir basename or a placeholder
        original_ts = model_dir_basename.replace(" ", "_")[:20]  # Truncate if too long
    
    run_base = os.path.abspath("./sparcity_cache")
    os.makedirs(run_base, exist_ok=True)
    
    run_dir = os.path.join(run_base, f"{model_name}_{dataset_name}_{original_ts}-{current_ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging to capture all output to a file
    log_file_path = os.path.join(run_dir, "eval_sparsity.log")
    tee_output = TeeOutput(log_file_path)
    sys.stdout = tee_output
    sys.stderr = tee_output
    
    print(f"Outputs will be saved to: {run_dir}")
    print(f"Log file: {log_file_path}")

    # Resolve checkpoint if not provided (best val epoch, not latest saved epoch).
    model_cache_dir = os.path.join(args.model_dir, "model_cache")
    if not os.path.isdir(model_cache_dir):
        raise RuntimeError(f"model_cache directory not found: {model_cache_dir}")

    ckpt_path, ckpt_meta = resolve_checkpoint_path(
        model_cache_dir=model_cache_dir,
        log_path=log_path,
        explicit_checkpoint=args.checkpoint,
    )
    if ckpt_meta.get("checkpoint_selection") == "best_val":
        print(
            f"Using best-val checkpoint: epoch {ckpt_meta['best_val_epoch']} "
            f"(val_loss={ckpt_meta['best_val_loss']:.4f})"
        )
    print(f"Using checkpoint: {ckpt_path}")

    # Save evaluation metadata and config to run_dir
    eval_metadata = {
        "eval_timestamp": current_ts,
        "original_checkpoint_timestamp": original_ts,
        "model_dir": os.path.abspath(args.model_dir),
        "checkpoint_path": os.path.abspath(ckpt_path),
        "checkpoint_selection": ckpt_meta.get("checkpoint_selection"),
        "best_val_epoch": ckpt_meta.get("best_val_epoch"),
        "best_val_loss": ckpt_meta.get("best_val_loss"),
        "training_log_path": os.path.abspath(log_path),
        "task": task,
        "model": model_name,
        "dataset": dataset_name,
        "sparsity_mode": args.sparsity_mode,
        "mask_seed": args.mask_seed,
        "mask_zero_mode": args.mask_zero_mode,
        "batch_size": args.batch_size,
        "device": args.device,
        "visualize_topology": args.visualize_topology,
        "topology_channel": args.topology_channel,
        "topology_channel_name": args.topology_channel_name,
    }
    
    # Add mode-specific metadata
    if args.sparsity_mode == "sweep":
        eval_metadata["sparsity_levels"] = args.sparsity
    else:
        eval_metadata["scheme_file"] = os.path.abspath(args.scheme_file)
        eval_metadata["schemes_requested"] = args.schemes
    
    # Save eval metadata
    metadata_path = os.path.join(run_dir, "eval_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(eval_metadata, f, indent=2, default=str)
    print(f"Saved evaluation metadata to: {metadata_path}")
    
    # Save the training config extracted from log
    if log_config_overrides:
        training_config_path = os.path.join(run_dir, "training_config.json")
        with open(training_config_path, "w") as f:
            # Convert any non-serializable values
            serializable_config = {}
            for k, v in log_config_overrides.items():
                try:
                    json.dumps(v)
                    serializable_config[k] = v
                except (TypeError, ValueError):
                    serializable_config[k] = str(v)
            json.dump(serializable_config, f, indent=2)
        print(f"Saved training config to: {training_config_path}")

    # Default out_csv under run_dir if not provided
    if args.out_csv is None:
        args.out_csv = os.path.join(run_dir, "sparsity_results.csv")
    else:
        # If user passed a directory, drop file into it; otherwise honor file path
        if os.path.isdir(args.out_csv):
            args.out_csv = os.path.join(args.out_csv, "sparsity_results.csv")
        else:
            out_dirname = os.path.dirname(os.path.abspath(args.out_csv))
            if out_dirname:
                os.makedirs(out_dirname, exist_ok=True)

    # CLI overrides - start with config from training log, then add CLI args
    overrides = {}
    
    # First, apply config overrides extracted from the training log
    # These ensure the model architecture matches the checkpoint
    if log_config_overrides:
        overrides.update(log_config_overrides)
        print(f"[config] Applied overrides from log: {list(log_config_overrides.keys())}")
        # Log data_col explicitly - critical for understanding channel ordering
        if 'data_col' in log_config_overrides:
            print(f"[config] data_col from training: {log_config_overrides['data_col']}")
            print(f"         Channel 0={log_config_overrides['data_col'][0]}, "
                  f"Channel 1={log_config_overrides['data_col'][1]}, "
                  f"Channel 2={log_config_overrides['data_col'][2]}")
    
    # CLI overrides take precedence
    overrides["batch_size"] = args.batch_size
    overrides["device"] = args.device
    
    # CRITICAL: Disable dataset caching during evaluation.
    # The cache filename does NOT include 'data_col', so a cache created with one
    # channel ordering (e.g. [occ, speed, flow]) would be reused even when evaluating
    # a model trained with a different ordering (e.g. [flow, occ, speed]).
    # This causes severe input channel misalignment that cannot be fixed by output permutation.
    overrides["cache_dataset"] = False

    config = ConfigParser(task, model_name, dataset_name,
                          other_args=overrides, train=False)

    # Dataset, data_feature, scaler

    dataset = get_dataset(config)
    train_loader, val_loader, test_loader = dataset.get_data()
    data_feature = dataset.get_data_feature()

    scaler = data_feature.get("scaler", None)
    num_nodes = data_feature.get("num_nodes")
    feature_dim = data_feature.get("feature_dim")
    output_dim = data_feature.get("output_dim")

    # Fast path: compute metrics directly from saved predictions (sanity-check mode)
    if args.eval_from_npz is not None:
        print(f"\n[Sanity] Loading saved predictions from: {args.eval_from_npz}")
        npz = np.load(args.eval_from_npz, allow_pickle=True)
        # Heuristics for keys
        cand_pred = ["prediction", "pred", "y_pred", "yhat", "y_predicted"]
        cand_gt = ["truth", "gt", "y_true", "target", "label"]
        preds_np = None
        gts_np = None
        for k in cand_pred:
            if k in npz:
                preds_np = np.array(npz[k])
                break
        for k in cand_gt:
            if k in npz:
                gts_np = np.array(npz[k])
                break
        if preds_np is None or gts_np is None:
            raise RuntimeError(f"Could not find prediction/truth arrays in npz. Keys: {list(npz.keys())}")
        print(f"  loaded pred shape: {preds_np.shape}, gt shape: {gts_np.shape}")

        # Try to coerce to [B, T, N, C]
        def coerce_shape(arr):
            a = np.array(arr)
            if a.ndim == 4:
                return a
            if a.ndim == 3:
                # assume [B,T,N] and C=1
                return a[..., None]
            if a.ndim == 5:
                # sometimes an extra singleton
                if a.shape[-1] == 1:
                    return a.reshape(a.shape[0], a.shape[1], a.shape[2], a.shape[3])
            raise RuntimeError(f"Unexpected array ndim for coerce: {a.shape}")

        preds_np = coerce_shape(preds_np)
        gts_np = coerce_shape(gts_np)

        # If extra channels exist, keep first output_dim channels
        if preds_np.shape[-1] > output_dim:
            preds_np = preds_np[..., :output_dim]
        if gts_np.shape[-1] > output_dim:
            gts_np = gts_np[..., :output_dim]

        # Metrics
        mae, rmse, mape = compute_metrics_global(preds_np, gts_np)
        m_mae, m_rmse, m_mape = compute_metrics_global_masked(preds_np, gts_np, null_val=0.0)
        mae_h, rmse_h = compute_metrics_per_horizon(preds_np, gts_np)
        mae_last = float(mae_h[-1])
        rmse_last = float(rmse_h[-1])

        # Per-channel summary
        if preds_np.shape[-1] == 3:
            pred_mean_c = preds_np.mean(axis=(0, 1, 2))
            gt_mean_c = gts_np.mean(axis=(0, 1, 2))
            mae_c = np.mean(np.abs(preds_np - gts_np), axis=(0, 1, 2))
            rmse_c = np.sqrt(np.mean((preds_np - gts_np) ** 2, axis=(0, 1, 2)))
            print(f"  [Sanity] per-channel GT mean:   {np.array2string(gt_mean_c, precision=4)}")
            print(f"  [Sanity] per-channel Pred mean: {np.array2string(pred_mean_c, precision=4)}")
            print(f"  [Sanity] per-channel MAE:       {np.array2string(mae_c, precision=4)}")
            print(f"  [Sanity] per-channel RMSE:      {np.array2string(rmse_c, precision=4)}")

        print(f"  Global  MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        print(f"  Masked  MAE={m_mae:.4f}, RMSE={m_rmse:.4f}, masked_MAPE={m_mape*100:.2f}%")
        print(f"  Horizon MAE (12 steps): {np.array2string(mae_h, precision=4)}")
        print(f"  Horizon RMSE (12 steps): {np.array2string(rmse_h, precision=4)}")
        print(f"  Last-horizon MAE={mae_last:.4f}, RMSE={rmse_last:.4f}")
        # Do not write CSV/masks/plots in sanity mode
        return
    print("\nData feature summary:")
    print(f"  num_nodes   = {num_nodes}")
    print(f"  feature_dim = {feature_dim}")
    print(f"  output_dim  = {output_dim}")
    print(f"  scaler      = {scaler} (type: {type(scaler).__name__})")

    # Try to print scaler channel stats if available
    try:
        sc_mu = getattr(scaler, "mean", None)
        if sc_mu is None:
            sc_mu = getattr(scaler, "mean_", None)
        sc_std = getattr(scaler, "std", None)
        if sc_std is None:
            sc_std = getattr(scaler, "std_", None)
        if sc_mu is not None and sc_std is not None:
            sc_mu_np = np.asarray(sc_mu).reshape(-1)
            sc_std_np = np.asarray(sc_std).reshape(-1)
            print(f"  scaler.mean: {np.array2string(sc_mu_np, precision=4)}")
            print(f"  scaler.std:  {np.array2string(sc_std_np, precision=4)}")
    except Exception as e:
        print(f"  [warn] Could not read scaler stats: {e}")

    # Load topology graph for visualization (if enabled)
    topology_graph = None
    topology_channel_name = None
    if args.visualize_topology and TOPOLOGY_VIS_AVAILABLE:
        # Determine rel file path
        rel_path = args.rel_file
        if rel_path is None:
            rel_path = os.path.join(ROOT_DIR, "raw_data", dataset_name, f"{dataset_name}.rel")
        
        if os.path.isfile(rel_path):
            print(f"\n[Topology] Loading network topology from: {rel_path}")
            try:
                rel_df = load_rel_file(rel_path)
                topology_graph = create_graph(rel_df)
                print(f"[Topology] Created graph with {topology_graph.number_of_nodes()} nodes and {topology_graph.number_of_edges()} edges")
            except Exception as e:
                print(f"[Topology] Failed to load graph: {e}")
                topology_graph = None
        else:
            print(f"[Topology] Rel file not found: {rel_path}")
            print(f"[Topology] Topology visualization will be skipped.")
        
        # Determine channel name for visualization
        if args.topology_channel_name:
            topology_channel_name = args.topology_channel_name
        else:
            # Try to infer from data_col in config
            data_col = log_config_overrides.get('data_col', ['flow', 'occupy', 'speed'])
            if args.topology_channel < len(data_col):
                topology_channel_name = data_col[args.topology_channel].capitalize()
            else:
                topology_channel_name = f"Channel{args.topology_channel}"
        print(f"[Topology] Using channel {args.topology_channel} ({topology_channel_name}) for visualization")
    elif args.visualize_topology and not TOPOLOGY_VIS_AVAILABLE:
        print("[Topology] Visualization requested but visualize_mae_topology module not available.")
    
    # Storage for baseline (rho=0) MAE data for comparison plots
    baseline_mae_per_sensor = None
    
    # Storage for scheme data (for overview plot)
    scheme_data_list = []
    
    # Storage for raw predictions (for .npz export)
    predictions_storage = {}  # key -> {'predictions': array, 'ground_truth': array, 'scheme_name': str, 'rho': float}

    # Prepare evaluator-style normalization stats (from dataset scaler) for first output_dim channels
    mu_eval = None
    std_eval = None
    try:
        if scaler is not None and type(scaler).__name__ != "NoneScaler":
            sc_mu = getattr(scaler, "mean", None) or getattr(scaler, "mean_", None)
            sc_std = getattr(scaler, "std", None) or getattr(scaler, "std_", None)
            if sc_mu is not None and sc_std is not None:
                sc_mu_np = np.asarray(sc_mu).reshape(-1)
                sc_std_np = np.asarray(sc_std).reshape(-1)
                if sc_mu_np.size == 1:
                    # Single scalar -> broadcast to output_dim
                    mu_eval = np.repeat(float(sc_mu_np[0]), output_dim)
                    std_eval = np.repeat(float(max(sc_std_np[0], 1e-8)), output_dim)
                else:
                    # Per-feature stats -> take first output_dim channels
                    mu_eval = sc_mu_np[:output_dim].astype(np.float64)
                    std_eval = np.maximum(sc_std_np[:output_dim].astype(np.float64), 1e-8)
    except Exception:
        # Leave as None to fall back to test-slice stats when used
        pass

    # Prepare base directory for series CSV export
    series_base_dir = None
    if args.save_series_csvs:
        series_base_dir = args.save_series_dir or args.out_csv.replace(".csv", "_series")
        os.makedirs(series_base_dir, exist_ok=True)
        print(f"[Series] Will save per-sensor CSVs under: {series_base_dir}")

    # Training-set statistics for vehicular metrics (if requested)

    mu_train = None
    std_train = None
    if args.vehicular_norm == "train":
        print("\nComputing training-set channel stats for vehicular metrics ...")
        mu_train, std_train = compute_train_channel_stats(
            train_loader, output_dim, scaler=scaler, feature_dim=feature_dim
        )
        print(f"  mu_train: {mu_train}")
        print(f"  std_train: {std_train}")

    # Build model and load checkpoint

    model = get_model(config, data_feature).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, tuple):
        state = ck[0]
    else:
        state = ck.get("model_state_dict", ck)
    # Be robust to minor architecture/config mismatches across runs
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[warn] Strict checkpoint load failed, attempting partial load.\n{e}")
        current_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []
        for key, value in state.items():
            if key in current_state and current_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                skipped_keys.append(key)
        missing_in_ckpt = [k for k in current_state.keys() if k not in filtered_state]
        model.load_state_dict(filtered_state, strict=False)
        print(
            f"[warn] Loaded partial checkpoint: "
            f"applied={len(filtered_state)} keys, "
            f"skipped_incompatible={len(skipped_keys)}, "
            f"missing_in_checkpoint={len(missing_in_ckpt)}"
        )
    model.eval()

    # Load node sparsity stats and define groups (optional)

    nodes_sparse = []
    nodes_non_sparse = []
    sparse_threshold = None

    if args.sparsity_stats_npz is not None:
        print(f"\nLoading node sparsity stats from {args.sparsity_stats_npz}")
        stats = np.load(args.sparsity_stats_npz, allow_pickle=True)
        zero_frac = stats["zero_frac"]  # [N, 3] in channel order [flow, occupancy, speed]
        meta = json.loads(str(stats["meta"]))
        default_thr = float(meta.get("sparse_frac_thresh", 0.2))
        sparse_threshold = args.sparse_frac_thresh if args.sparse_frac_thresh is not None else default_thr

        sparse_flow = (zero_frac[:, 0] >= sparse_threshold)
        sparse_occ = (zero_frac[:, 1] >= sparse_threshold)
        sparse_speed = (zero_frac[:, 2] >= sparse_threshold)

        if args.group_mode == "flow_occ_any":
            sparse_mask = np.logical_or(sparse_flow, sparse_occ)
        elif args.group_mode == "flow_only":
            sparse_mask = sparse_flow
        elif args.group_mode == "occ_only":
            sparse_mask = sparse_occ
        else:
            sparse_mask = np.logical_or(np.logical_or(sparse_flow, sparse_occ), sparse_speed)

        nodes_sparse = np.where(sparse_mask)[0].astype(int).tolist()
        nodes_non_sparse = np.where(~sparse_mask)[0].astype(int).tolist()

        print(f"  sparse_frac_thresh = {sparse_threshold}")
        print(f"  group_mode         = {args.group_mode}")
        print(f"  |nodes_sparse|     = {len(nodes_sparse)}")
        print(f"  |nodes_non_sparse| = {len(nodes_non_sparse)}")

    masks_json = args.save_masks_json or args.out_csv.replace(".csv", "_masks.json")
    masks_registry = {
        "meta": {
            "group_mode": args.group_mode,
            "sparse_frac_thresh": sparse_threshold,
            "mask_seed": int(args.mask_seed),
            "nodes_sparse": nodes_sparse,
            "nodes_non_sparse": nodes_non_sparse,
            "num_nodes": int(num_nodes),
            "sparsity_mode": args.sparsity_mode,
        },
        "masks_by_rho": {},
        "masks_by_scheme": {},
    }

    # Load and process schemes if in scheme mode
    schemes_data = {}
    scheme_groups = {}
    scheme_iter_list = []  # List of (scheme_name, sensors_list, group_name) tuples
    
    if args.sparsity_mode == "scheme":
        print(f"\n[Scheme Mode] Loading schemes from: {args.scheme_file}")
        with open(args.scheme_file, "r") as f:
            schemes_data = json.load(f)
        
        # Filter to requested schemes if specified
        if args.schemes:
            schemes_data = {k: v for k, v in schemes_data.items() if k in args.schemes}
            if not schemes_data:
                raise RuntimeError(f"No matching schemes found. Requested: {args.schemes}")
        
        print(f"  Loaded {len(schemes_data)} scheme(s)")
        
        # Group schemes by prefix for subfolder organization
        scheme_groups = group_schemes_by_prefix(list(schemes_data.keys()))
        print(f"  Scheme groups: {list(scheme_groups.keys())}")
        
        # Build iteration list with group info
        for group_name, scheme_names in scheme_groups.items():
            for sn in scheme_names:
                sensors = schemes_data[sn].get("sensors", [])
                scheme_iter_list.append((sn, sensors, group_name))
        
        # Create group subdirectories
        for group_name in scheme_groups.keys():
            group_dir = os.path.join(run_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)
        
        # Save schemes info to run_dir
        schemes_info_path = os.path.join(run_dir, "schemes_info.json")
        with open(schemes_info_path, "w") as f:
            json.dump({
                "groups": {g: list(s) for g, s in scheme_groups.items()},
                "schemes": {k: {"sensors": v.get("sensors", []), 
                               "count": v.get("count", len(v.get("sensors", []))),
                               "description": v.get("description", "")}
                           for k, v in schemes_data.items()}
            }, f, indent=2)
        print(f"  Saved schemes info to: {schemes_info_path}")

    # Sparsity evaluation (sweep or scheme mode)

    results = []
    results_by_group = {}  # For scheme mode: {group_name -> [results]}

    # For plotting GT vs preds across sparsity/schemes
    ts_gt = None                      # ground truth time series (12 horizons)
    ts_preds_by_rho = {}              # rho/scheme -> predicted time series
    plot_ex_idx = None                # resolved example idx within test set
    plot_node_idx = None              # resolved node idx (clamped)
    # For plotting per-channel globally averaged outputs across sparsities/schemes
    # Shapes: [T_out, C] where C=output_dim=3 (flow, occ, speed)
    global_gt_h_c = None                 # [T_out, C]
    global_pred_h_by_rho_c = {}          # rho/scheme -> [T_out, C]

    # FIX: Cache DCRNN permutation settings to prevent "re-detection" at high sparsity
    cached_dcrnn_channel_perm = None
    cached_dcrnn_perm_target = None
    cached_dcrnn_input_perm = None

    # Build iteration list based on mode
    if args.sparsity_mode == "sweep":
        # Sweep mode: iterate over sparsity levels
        iter_list = [(rho, None, None) for rho in args.sparsity]  # (rho, scheme_name, group_name)
    else:
        # Scheme mode: iterate over schemes
        
        iter_list = [(0.0, "baseline", None)]  # Start with true baseline
        iter_list.extend([(None, sn, gn) for sn, sensors, gn in scheme_iter_list])
        # Create a mapping for sensor lookup (empty for baseline)
        scheme_sensors_map = {"baseline": []}  # baseline has no masked sensors
        scheme_sensors_map.update({sn: sensors for sn, sensors, gn in scheme_iter_list})


    for iter_idx, (rho, scheme_name, group_name) in enumerate(iter_list):
        # Determine masked indices and labels based on mode
        if args.sparsity_mode == "sweep":
            rho = float(rho)
            iter_label = f"sparsity {rho * 100:.1f}%"
            iter_key = f"{rho:.2f}"
            masked_idx = masked_nodes_for_rho(num_nodes, rho, seed_base=args.mask_seed)
            masks_registry["masks_by_rho"][iter_key] = [int(i) for i in masked_idx]
            current_group_dir = None
        else:
            # Scheme mode
            sensors = scheme_sensors_map[scheme_name]
            masked_idx = np.array(sensors, dtype=int)
            rho = len(masked_idx) / num_nodes  # Compute effective sparsity
            # Handle baseline specially for labeling
            if scheme_name == "baseline":
                iter_label = f"sparsity 0.0% (baseline)"
            else:
                iter_label = f"scheme '{scheme_name}' ({len(masked_idx)} sensors, {rho * 100:.1f}%)"
            iter_key = scheme_name
            masks_registry["masks_by_scheme"][scheme_name] = [int(i) for i in masked_idx]
            current_group_dir = os.path.join(run_dir, group_name) if group_name else run_dir
        
        print(f"\n===== {iter_label} =====")
        
        # Initialize with cached values (if any)
        dcrnn_channel_perm = cached_dcrnn_channel_perm
        dcrnn_perm_target = cached_dcrnn_perm_target
        dcrnn_input_perm = cached_dcrnn_input_perm

        preds_all = []
        gts_all = []

        # Initialize per-sensor series storage (for CSV export) for this iteration
        if args.save_series_csvs:
            if args.sparsity_mode == "sweep":
                rho_dir = os.path.join(series_base_dir, f"rho_{rho:.2f}")
            else:
                # For scheme mode, organize by group/scheme
                # Handle baseline case where group_name is None
                if group_name is not None:
                    rho_dir = os.path.join(series_base_dir, group_name, scheme_name)
                else:
                    rho_dir = os.path.join(series_base_dir, scheme_name)
            os.makedirs(rho_dir, exist_ok=True)
            series_rows_by_node = [[] for _ in range(num_nodes)]

        for batch in tqdm(test_loader, desc=iter_label):
            # batch["X"], batch["y"] are lists of np arrays
            X_np = np.stack(batch["X"], axis=0).astype(np.float32)  # [B, T_in, N, F]
            y_np = np.stack(batch["y"], axis=0).astype(np.float32)  # [B, T_out, N, F_all]

            # Use only first output_dim channels as ground truth
            gts_np = y_np[..., :output_dim]  # [B, T_out, N, output_dim]

            # Mask sensors in input (same masked_idx for all batches at this rho)
            if masked_idx.size > 0:
                X_masked = X_np.copy()
                x_first = X_masked[..., :output_dim]
                if args.mask_zero_mode == "inverse":
                    # Previous implementation: inverse -> zero in raw units -> re-normalize
                    if scaler is not None and type(scaler).__name__ != "NoneScaler":
                        x_first_inv = scaler.inverse_transform(torch.from_numpy(x_first)).numpy()
                    else:
                        x_first_inv = x_first
                    # zero in original units
                    x_first_inv[:, :, masked_idx, :] = 0.0
                    # re-apply normalization
                    if scaler is not None and type(scaler).__name__ != "NoneScaler":
                        try:
                            x_first_norm = scaler.transform(torch.from_numpy(x_first_inv)).numpy()
                        except Exception:
                            sc_mu = getattr(scaler, "mean", None) or getattr(scaler, "mean_", None)
                            sc_std = getattr(scaler, "std", None) or getattr(scaler, "std_", None)
                            if sc_mu is not None and sc_std is not None:
                                mu = np.asarray(sc_mu).reshape(-1)[:output_dim]
                                sd = np.asarray(sc_std).reshape(-1)[:output_dim]
                                sd = np.where(sd < 1e-8, 1e-8, sd)
                                x_first_norm = (x_first_inv - mu) / sd
                            else:
                                x_first_norm = x_first_inv
                    else:
                        x_first_norm = x_first_inv
                    X_masked[..., :output_dim] = x_first_norm
                else:
                    # Default: write the normalized value that corresponds to raw zero
                    # Compute per-channel normalized zero for the first output_dim channels
                    normalized_zero = np.zeros((output_dim,), dtype=np.float32)
                    if scaler is not None and type(scaler).__name__ != "NoneScaler":
                        sc_mu = getattr(scaler, "mean", None) or getattr(scaler, "mean_", None)
                        sc_std = getattr(scaler, "std", None) or getattr(scaler, "std_", None)
                        if sc_mu is not None and sc_std is not None:
                            mu = np.asarray(sc_mu).reshape(-1)
                            sd = np.asarray(sc_std).reshape(-1)
                            if mu.size >= output_dim and sd.size >= output_dim:
                                sd_use = np.where(sd[:output_dim] < 1e-8, 1e-8, sd[:output_dim])
                                normalized_zero = (-mu[:output_dim] / sd_use).astype(np.float32)
                        else:
                            sc_min = getattr(scaler, "min", None) or getattr(scaler, "min_", None)
                            sc_max = getattr(scaler, "max", None) or getattr(scaler, "max_", None)
                            if sc_min is not None and sc_max is not None:
                                mn = np.asarray(sc_min).reshape(-1)
                                mx = np.asarray(sc_max).reshape(-1)
                                if mn.size >= output_dim and mx.size >= output_dim:
                                    denom = mx[:output_dim] - mn[:output_dim]
                                    denom = np.where(denom < 1e-8, 1e-8, denom)
                                    normalized_zero = (-(mn[:output_dim]) / denom).astype(np.float32)
                    # Assign normalized zero across all timesteps for masked nodes
                    x_first[:, :, masked_idx, :] = normalized_zero.reshape(1, 1, 1, -1)
                    X_masked[..., :output_dim] = x_first
            else:
                X_masked = X_np

            # Auto-detect INPUT channel permutation for DCRNN on the first batch
            if (is_dcrnn
                and output_dim == 3
                and args.permute_inputs == "auto"):
                
                # Only detect if not already cached
                if dcrnn_input_perm is None:
                    perms = list(itertools.permutations([0, 1, 2], 3))

                    def apply_input_perm(x_np, perm):
                        # permute ONLY the first output_dim channels; keep any remaining features as-is
                        x_copy = x_np.copy()
                        x_copy[..., :output_dim] = x_copy[..., list(perm)]
                        return x_copy

                    best_mae = None
                    best_in_perm = None
                    for in_perm in perms:
                        X_tmp = apply_input_perm(X_masked, in_perm)
                        X_t_tmp = torch.from_numpy(X_tmp).to(device)
                        # IMPORTANT for GMAN/DCRNN correctness: provide full 'y' (with time features)
                        y_t_tmp = torch.from_numpy(y_np).to(device)
                        with torch.no_grad():
                            pred_t_tmp = model.predict({"X": X_t_tmp, "y": y_t_tmp})
                        pred_np_tmp = pred_t_tmp.detach().cpu().numpy()
                        if pred_np_tmp.shape[-1] != output_dim:
                            pred_np_tmp = pred_np_tmp[..., :output_dim]
                        # To be robust to possible output channel order, try all pred perms and pick minimal MAE
                        maes = []
                        for out_perm in perms:
                            pp = pred_np_tmp[..., list(out_perm)]
                            if scaler is not None and type(scaler).__name__ != "NoneScaler":
                                pp_inv = scaler.inverse_transform(torch.from_numpy(pp)).numpy()
                                gg_inv = scaler.inverse_transform(torch.from_numpy(gts_np)).numpy()
                                m = float(np.mean(np.abs(pp_inv - gg_inv)))
                            else:
                                m = float(np.mean(np.abs(pp - gts_np)))
                            maes.append(m)
                        m_best = min(maes)
                        if best_mae is None or m_best < best_mae:
                            best_mae = m_best
                            best_in_perm = list(in_perm)
                    
                    dcrnn_input_perm = best_in_perm
                    cached_dcrnn_input_perm = best_in_perm # CACHE IT
                    print(f"  [DCRNN] Selected INPUT permutation {dcrnn_input_perm} (min batch MAE={best_mae:.4f})")

            # Apply chosen input permutation (if any)
            if dcrnn_input_perm is not None:
                X_masked = X_masked.copy()
                X_masked[..., :output_dim] = X_masked[..., dcrnn_input_perm]

            X_t = torch.from_numpy(X_masked).to(device)
            # Provide FULL y (all features) to the model, since some models (e.g., GMAN)
            # derive temporal embeddings from the last channels of y.
            y_t_full = torch.from_numpy(y_np).to(device)  # still normalized

            batch_for_model = {"X": X_t, "y": y_t_full}

            with torch.no_grad():
                pred_t = model.predict(batch_for_model)


            preds_np = pred_t.detach().cpu().numpy()  # [B, T_out, N, output_dim_expected]

            # Safety: ensure last dim matches output_dim
            if preds_np.shape[-1] != output_dim:
                preds_np = preds_np[..., :output_dim]

            # Auto/controlled channel permutation before inverse transform (works for any model)
            if output_dim == 3 and args.permute_channels != "none":
                
                # Only detect if not already cached
                if dcrnn_channel_perm is None:
                    perms = list(itertools.permutations([0, 1, 2], 3))
                    mae_best = None
                    best_perm = None
                    best_target = None

                    def eval_perm_mae(perm, target):
                        # permute the selected side (pred or gt) BEFORE inverse scaling
                        if target == "pred":
                            pp = preds_np[..., list(perm)]
                            gg = gts_np
                        else:  # target == "gt"
                            pp = preds_np
                            gg = gts_np[..., list(perm)]
                        if scaler is not None and type(scaler).__name__ != "NoneScaler":
                            pp_inv = scaler.inverse_transform(torch.from_numpy(pp)).numpy()
                            gg_inv = scaler.inverse_transform(torch.from_numpy(gg)).numpy()
                            return float(np.mean(np.abs(pp_inv - gg_inv)))
                        else:
                            return float(np.mean(np.abs(pp - gg)))

                    if args.permute_channels in ("pred", "gt"):
                        # Evaluate only the requested target
                        target = args.permute_channels
                        maes = []
                        for perm in perms:
                            m = eval_perm_mae(perm, target)
                            maes.append(m)
                        idx = int(np.argmin(maes))
                        best_perm = list(perms[idx])
                        best_target = target
                        mae_best = maes[idx]
                    else:
                        # auto: try both targets and pick globally best
                        maes_pred = []
                        maes_gt = []
                        for perm in perms:
                            maes_pred.append(eval_perm_mae(perm, "pred"))
                            maes_gt.append(eval_perm_mae(perm, "gt"))
                        idx_pred = int(np.argmin(maes_pred))
                        idx_gt = int(np.argmin(maes_gt))
                        if maes_pred[idx_pred] <= maes_gt[idx_gt]:
                            best_perm = list(perms[idx_pred])
                            best_target = "pred"
                            mae_best = maes_pred[idx_pred]
                        else:
                            best_perm = list(perms[idx_gt])
                            best_target = "gt"
                            mae_best = maes_gt[idx_gt]

                    dcrnn_channel_perm = best_perm
                    dcrnn_perm_target = best_target
                    
                    # CACHE IT
                    cached_dcrnn_channel_perm = best_perm
                    cached_dcrnn_perm_target = best_target
                    
                    print(
                        f"  [DCRNN] Selected channel permutation {dcrnn_channel_perm} "
                        f"target='{dcrnn_perm_target}' (min batch MAE={mae_best:.4f})"
                    )

                # Apply the detected permutation BEFORE inverse transform
                if dcrnn_perm_target == "pred":
                    preds_np = preds_np[..., dcrnn_channel_perm]
                elif dcrnn_perm_target == "gt":
                    gts_np = gts_np[..., dcrnn_channel_perm]

            # Inverse transform if scaler is non-trivial (to original units)
            if True:
                preds_np = inverse_transform_like_evaluator(
                    preds_np, scaler, feature_dim, output_dim
                )
                gts_np = inverse_transform_like_evaluator(
                    gts_np, scaler, feature_dim, output_dim
                )

            # Collect per-sensor Input/GT/Pred rows for CSV export (after inverse scaling)
            if args.save_series_csvs:
                # Use only the first output_dim channels of inputs, matching GT/Pred
                x_used_norm = X_masked[..., :output_dim]
                x_used = inverse_transform_like_evaluator(
                    x_used_norm, scaler, feature_dim, output_dim
                )

                B_now, T_in_now, N_now, C_now = x_used.shape
                _, T_out_now, _, _ = preds_np.shape
                # Align each output horizon h to input timestep index start_idx + h
                start_idx = max(0, T_in_now - T_out_now)

                for b in range(B_now):
                    for n in range(N_now):
                        for h in range(T_out_now):
                            x_ti = min(start_idx + h, T_in_now - 1)
                            x_vals = x_used[b, x_ti, n, :]  # [C]
                            gt_vals = gts_np[b, h, n, :]    # [C]
                            pr_vals = preds_np[b, h, n, :]  # [C]
                            row = []
                            for c in range(output_dim):
                                row.extend([
                                    float(x_vals[c]),
                                    float(gt_vals[c]),
                                    float(pr_vals[c]),
                                ])
                            series_rows_by_node[n].append(row)

            # Optional: first-batch DCRNN diagnostics right after inverse transform
            if True and dcrnn_channel_perm is not None:
                # Compute simple per-channel means on this first batch only
                # Shapes: [B, T_out, N, C]
                b_pred_mean = preds_np.mean(axis=(0, 1, 2))
                b_gt_mean = gts_np.mean(axis=(0, 1, 2))
                # Only print occasionally to avoid spam
                # print(f"  [DCRNN debug:first-batch] GT mean   {np.array2string(b_gt_mean, precision=4)}")
                # print(f"  [DCRNN debug:first-batch] Pred mean {np.array2string(b_pred_mean, precision=4)}")

            preds_all.append(preds_np)
            gts_all.append(gts_np)

        preds_all = np.concatenate(preds_all, axis=0)
        gts_all = np.concatenate(gts_all, axis=0)
        print(f"  preds shape: {preds_all.shape}, gts shape: {gts_all.shape}")
        
        # Store predictions for .npz export
        predictions_storage[iter_key] = {
            'predictions': preds_all.copy(),
            'ground_truth': gts_all.copy(),
            'scheme_name': scheme_name if args.sparsity_mode == "scheme" else None,
            'rho': rho,
            'masked_nodes': list(masked_idx),
            'group_name': group_name if args.sparsity_mode == "scheme" else None,
        }

        # Optional per-channel stats and errors (after inverse scaling already applied above)
        if args.debug_per_channel and preds_all.shape[-1] == 3:
            # Per-channel means (global over batch, time, node)
            pred_mean_c = preds_all.mean(axis=(0, 1, 2))
            gt_mean_c = gts_all.mean(axis=(0, 1, 2))
            # Per-channel MAE
            mae_c = np.mean(np.abs(preds_all - gts_all), axis=(0, 1, 2))
            rmse_c = np.sqrt(np.mean((preds_all - gts_all) ** 2, axis=(0, 1, 2)))
            print(f"  [debug] per-channel GT mean:   {np.array2string(gt_mean_c, precision=4)}")
            print(f"  [debug] per-channel Pred mean: {np.array2string(pred_mean_c, precision=4)}")
            print(f"  [debug] per-channel MAE:       {np.array2string(mae_c, precision=4)}")
            print(f"  [debug] per-channel RMSE:      {np.array2string(rmse_c, precision=4)}")


        # ----------------------------------------------------
        # Metrics on nodes that were actually masked at this rho
        # vs nodes that remained unmasked
        # ----------------------------------------------------
        masked_group = {}
        unmasked_group = {}
        masked_vMAE_a = masked_vRMSE_a = masked_vMAPE_a = np.nan
        unmasked_vMAE_a = unmasked_vRMSE_a = unmasked_vMAPE_a = np.nan

        if args.track_masked_groups:
            all_idx = np.arange(num_nodes)

            if masked_idx.size > 0:
                unmasked_idx = np.setdiff1d(all_idx, masked_idx)

                # basic metrics
                masked_group = compute_on_nodes(
                    preds_all, gts_all, masked_idx, null_val=0.0
                )
                unmasked_group = compute_on_nodes(
                    preds_all, gts_all, unmasked_idx, null_val=0.0
                )

                # vehicular group averages using selected baseline
                if args.vehicular_norm == "train":
                    mu_vec, std_vec = mu_train, std_train
                elif args.vehicular_norm == "evaluator":
                    mu_vec, std_vec = compute_evaluator_norm_stats(preds_all, gts_all, scope=args.vehicular_scope)
                else:
                    mu_vec, std_vec = None, None

                vMAE_c_mask, vRMSE_c_mask, vMAPE_c_mask, vMAE_a_mask, vRMSE_a_mask, vMAPE_a_mask = \
                    compute_vehicular_on_nodes(
                        preds_all, gts_all, output_dim,
                        mu_vec, std_vec, args.vehicular_scope,
                        masked_idx
                    )
                vMAE_c_unmask, vRMSE_c_unmask, vMAPE_c_unmask, vMAE_a_unmask, vRMSE_a_unmask, vMAPE_a_unmask = \
                    compute_vehicular_on_nodes(
                        preds_all, gts_all, output_dim,
                        mu_vec, std_vec, args.vehicular_scope,
                        unmasked_idx
                    )

                masked_vMAE_a, masked_vRMSE_a, masked_vMAPE_a = \
                    vMAE_a_mask, vRMSE_a_mask, vMAPE_a_mask
                unmasked_vMAE_a, unmasked_vRMSE_a, unmasked_vMAPE_a = \
                    vMAE_a_unmask, vRMSE_a_unmask, vMAPE_a_unmask
            else:
                # rho == 0 case, no masked nodes
                masked_group = dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan,
                                    mMAE=np.nan, mRMSE=np.nan, mMAPE=np.nan)
                unmasked_group = compute_on_nodes(
                    preds_all, gts_all, all_idx, null_val=0.0
                )

        # ----------------------------------------------------
        # Per-channel global series: mean over batches+nodes,
        # keep channels separate: shape [T_out, C]
        # ----------------------------------------------------
        if args.plot_global_outputs:
            # average over batch (axis 0) and nodes (axis 2), keep [T_out, C]
            gt_h_c = gts_all.mean(axis=(0, 2))      # [T_out, C]
            pred_h_c = preds_all.mean(axis=(0, 2))  # [T_out, C]

            if global_gt_h_c is None:
                global_gt_h_c = gt_h_c.copy()
            global_pred_h_by_rho_c[iter_key] = pred_h_c.copy()


        # Store one example's time series for plotting
        if args.plot_outputs:
            B_tot = preds_all.shape[0]
            # Clamp indices to valid range once
            if plot_ex_idx is None:
                plot_ex_idx = min(max(args.plot_example_idx, 0), B_tot - 1)
                plot_node_idx = min(max(args.plot_node_idx, 0), num_nodes - 1)

                print(f"  [plot] Using example idx={plot_ex_idx}, node idx={plot_node_idx}, "
                      f"channel={plot_channel_idx} for output timeseries plots.")

            # shape: [T_out]
            gt_ts = gts_all[plot_ex_idx, :, plot_node_idx, plot_channel_idx]
            pred_ts = preds_all[plot_ex_idx, :, plot_node_idx, plot_channel_idx]

            # Save GT once and preds per iteration (rho or scheme)
            if ts_gt is None:
                ts_gt = gt_ts.copy()
            ts_preds_by_rho[iter_key] = pred_ts.copy()

        # Global + per-horizon metrics (all nodes)
        mae, rmse, mape = compute_metrics_global(preds_all, gts_all)
        m_mae, m_rmse, m_mape = compute_metrics_global_masked(preds_all, gts_all, null_val=0.0)

        # Last-horizon masked metrics (to compare with evaluator per-horizon)
        preds_last = preds_all[:, -1:, :, :]
        gts_last = gts_all[:, -1:, :, :]
        m_mae_last, m_rmse_last, m_mape_last = compute_metrics_global_masked(preds_last, gts_last, null_val=0.0)


        # masked metrics in z-space
        if args.vehicular_norm == "train":
            mu_vec, std_vec = mu_train, std_train
        elif args.vehicular_norm == "evaluator":
            mu_vec, std_vec = compute_evaluator_norm_stats(preds_all, gts_all, scope=args.vehicular_scope)
        else:
            mu_vec, std_vec = None, None
        mz_mae, mz_rmse, mz_mape = compute_metrics_masked_zscore(preds_all, gts_all, mu_vec, std_vec)


        mae_h, rmse_h = compute_metrics_per_horizon(preds_all, gts_all)
        mae_last = float(mae_h[-1])
        rmse_last = float(rmse_h[-1])

        # Additional diagnostics: per-channel MAE per horizon and aggregates (original units)
        try:
            abs_diff_all = np.abs(preds_all - gts_all)  # [B, T, N, C]
            # Per-horizon per-channel MAE, averaged over batch and nodes -> [T, C]
            mae_h_c = abs_diff_all.mean(axis=(0, 2))
            # Per-channel average across horizons -> [C]
            mae_c_avg = mae_h_c.mean(axis=0)
            # Per-channel last-horizon -> [C]
            mae_c_last = mae_h_c[-1, :]
            # Log concise comparisons to evaluator per_channel CSVs
            if output_dim == 3:
                print(f"  [per-channel raw] MAE_h ch0: {np.array2string(mae_h_c[:, 0], precision=4)}")
                print(f"  [per-channel raw] MAE_h ch1: {np.array2string(mae_h_c[:, 1], precision=4)}")
                print(f"  [per-channel raw] MAE_h ch2: {np.array2string(mae_h_c[:, 2], precision=4)}")
            print(f"  [per-channel raw] MAE_avg:  {np.array2string(mae_c_avg, precision=4)}")
            print(f"  [per-channel raw] MAE_last: {np.array2string(mae_c_last, precision=4)}")
        except Exception as _e:
            pass

        # Vehicular metrics (z-score)
        if args.vehicular_norm == "train":
            mu_vec, std_vec = mu_train, std_train
        elif args.vehicular_norm == "evaluator":
            mu_vec, std_vec = compute_evaluator_norm_stats(preds_all, gts_all, scope=args.vehicular_scope)
        else:
            mu_vec, std_vec = None, None

        vMAE_c, vRMSE_c, vMAPE_c, vMAE_a, vRMSE_a, vMAPE_a = compute_vehicular_metrics(
            preds_all,
            gts_all,
            output_dim,
            mu_vec=mu_vec,
            std_vec=std_vec,
            scope=args.vehicular_scope,
        )

        # Compute per-horizon vMAE using the SAME normalization for consistency
        vMAE_per_horizon = compute_vehicular_metrics_per_horizon(
            preds_all, gts_all, output_dim,
            mu_vec=mu_vec, std_vec=std_vec
        )

        # Align vehicular averages to evaluator-style normalized metrics (reported alongside per-channel)
        vMAE_a = mz_mae
        vRMSE_a = mz_rmse
        vMAPE_a = mz_mape

        # Group metrics (if sparsity_stats_npz provided)
        extra = {}
        if args.sparsity_stats_npz is not None:
            N_nodes = preds_all.shape[2]
            all_idx = np.arange(N_nodes)

            group_all = compute_on_nodes(preds_all, gts_all, all_idx, null_val=0.0)
            group_nonsp = compute_on_nodes(preds_all, gts_all, nodes_non_sparse, null_val=0.0) if nodes_non_sparse else None
            group_sp = compute_on_nodes(preds_all, gts_all, nodes_sparse, null_val=0.0) if nodes_sparse else None

            vMAE_c_ns, vRMSE_c_ns, vMAPE_c_ns, vMAE_a_ns, vRMSE_a_ns, vMAPE_a_ns = \
                compute_vehicular_on_nodes(
                    preds_all, gts_all, output_dim,
                    mu_vec, std_vec, args.vehicular_scope,
                    nodes_non_sparse
                ) if nodes_non_sparse else (np.array([np.nan]*3),)*3 + (np.nan, np.nan, np.nan)

            vMAE_c_sp, vRMSE_c_sp, vMAPE_c_sp, vMAE_a_sp, vRMSE_a_sp, vMAPE_a_sp = \
                compute_vehicular_on_nodes(
                    preds_all, gts_all, output_dim,
                    mu_vec, std_vec, args.vehicular_scope,
                    nodes_sparse
                ) if nodes_sparse else (np.array([np.nan]*3),)*3 + (np.nan, np.nan, np.nan)

            if group_nonsp:
                print(f"  Non-sparse MAE={group_nonsp['MAE']:.4f}, Sparse MAE={(group_sp or {}).get('MAE', float('nan')):.4f}")

            extra.update({
                "MAE_all_nodes": group_all["MAE"],
                "RMSE_all_nodes": group_all["RMSE"],
                "MAPE_all_nodes": group_all["MAPE"],
                "masked_MAE_all_nodes": group_all["mMAE"],
                "masked_RMSE_all_nodes": group_all["mRMSE"],
                "masked_MAPE_all_nodes": group_all["mMAPE"],
                "MAE_non_sparse": (group_nonsp or {}).get("MAE", np.nan),
                "RMSE_non_sparse": (group_nonsp or {}).get("RMSE", np.nan),
                "MAPE_non_sparse": (group_nonsp or {}).get("MAPE", np.nan),
                "masked_MAE_non_sparse": (group_nonsp or {}).get("mMAE", np.nan),
                "masked_RMSE_non_sparse": (group_nonsp or {}).get("mRMSE", np.nan),
                "masked_MAPE_non_sparse": (group_nonsp or {}).get("mMAPE", np.nan),
                "MAE_sparse": (group_sp or {}).get("MAE", np.nan),
                "RMSE_sparse": (group_sp or {}).get("RMSE", np.nan),
                "MAPE_sparse": (group_sp or {}).get("MAPE", np.nan),
                "masked_MAE_sparse": (group_sp or {}).get("mMAE", np.nan),
                "masked_RMSE_sparse": (group_sp or {}).get("mRMSE", np.nan),
                "masked_MAPE_sparse": (group_sp or {}).get("mMAPE", np.nan),
                "vMAE_a_non_sparse": vMAE_a_ns,
                "vRMSE_a_non_sparse": vRMSE_a_ns,
                "vMAPE_a_non_sparse": vMAPE_a_ns,
                "vMAE_a_sparse": vMAE_a_sp,
                "vRMSE_a_sparse": vRMSE_a_sp,
                "vMAPE_a_sparse": vMAPE_a_sp,
            })

        # metrics on  masked vs unmasked sensors for this rho
        if args.track_masked_groups:
            N_nodes = preds_all.shape[2]
            all_idx = np.arange(N_nodes, dtype=int)
            masked_nodes = masked_idx.astype(int)  # deterministic per rho
            unmasked_nodes = np.setdiff1d(all_idx, masked_nodes)

            # masked group
            if masked_nodes.size > 0:
                masked_group = compute_on_nodes(preds_all, gts_all, masked_nodes, null_val=0.0)
                vMAE_c_m, vRMSE_c_m, vMAPE_c_m, vMAE_a_m, vRMSE_a_m, vMAPE_a_m = \
                    compute_vehicular_on_nodes(
                        preds_all, gts_all, output_dim,
                        mu_vec, std_vec, args.vehicular_scope,
                        masked_nodes
                    )
            else:
                masked_group = dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan,
                                    mMAE=np.nan, mRMSE=np.nan, mMAPE=np.nan)
                vMAE_a_m = vRMSE_a_m = vMAPE_a_m = np.nan

            # unmasked group (might be empty only at rho=1.0)
            if unmasked_nodes.size > 0:
                unmasked_group = compute_on_nodes(preds_all, gts_all, unmasked_nodes, null_val=0.0)
                vMAE_c_u, vRMSE_c_u, vMAPE_c_u, vMAE_a_u, vRMSE_a_u, vMAPE_a_u = \
                    compute_vehicular_on_nodes(
                        preds_all, gts_all, output_dim,
                        mu_vec, std_vec, args.vehicular_scope,
                        unmasked_nodes
                    )
            else:
                unmasked_group = dict(MAE=np.nan, RMSE=np.nan, MAPE=np.nan,
                                      mMAE=np.nan, mRMSE=np.nan, mMAPE=np.nan)
                vMAE_a_u = vRMSE_a_u = vMAPE_a_u = np.nan

            extra.update({
                "MAE_masked_nodes": masked_group["MAE"],
                "RMSE_masked_nodes": masked_group["RMSE"],
                "MAPE_masked_nodes": masked_group["MAPE"],
                "masked_MAE_masked_nodes": masked_group["mMAE"],
                "masked_RMSE_masked_nodes": masked_group["mRMSE"],
                "masked_MAPE_masked_nodes": masked_group["mMAPE"],

                "MAE_unmasked_nodes": unmasked_group["MAE"],
                "RMSE_unmasked_nodes": unmasked_group["RMSE"],
                "MAPE_unmasked_nodes": unmasked_group["MAPE"],
                "masked_MAE_unmasked_nodes": unmasked_group["mMAE"],
                "masked_RMSE_unmasked_nodes": unmasked_group["mRMSE"],
                "masked_MAPE_unmasked_nodes": unmasked_group["mMAPE"],

                "vMAE_a_masked_nodes": vMAE_a_m,
                "vRMSE_a_masked_nodes": vRMSE_a_m,
                "vMAPE_a_masked_nodes": vMAPE_a_m,
                "vMAE_a_unmasked_nodes": vMAE_a_u,
                "vRMSE_a_unmasked_nodes": vRMSE_a_u,
                "vMAPE_a_unmasked_nodes": vMAPE_a_u,
            })


        # Logging (global)
        print(f"  Global  MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        print(f"  Masked  MAE={m_mae:.4f}, RMSE={m_rmse:.4f}, masked_MAPE={m_mape*100:.2f}%")
        print(f"  Horizon MAE (12 steps): {np.array2string(mae_h, precision=4)}")
        print(f"  Horizon RMSE (12 steps): {np.array2string(rmse_h, precision=4)}")
        print(f"  Last-horizon MAE={mae_last:.4f}, RMSE={rmse_last:.4f}")
        print(f"  Vehicular per channel vMAE:  {np.array2string(vMAE_c,  precision=4)}")
        print(f"  Vehicular per channel vRMSE: {np.array2string(vRMSE_c, precision=4)}")
        print(f"  Vehicular per channel vMAPE: {np.array2string(vMAPE_c, precision=4)}")
        print(f"  Vehicular averaged vMAE_a={vMAE_a:.4f}, vRMSE_a={vRMSE_a:.4f}, vMAPE_a={vMAPE_a:.4f}")
        # Also report evaluator-like (z-score) masked metrics for comparability
        print(f"  [zscore] masked_zMAE={mz_mae:.4f}, masked_zRMSE={mz_rmse:.4f}, masked_zMAPE={mz_mape:.4f}")
        print(f"  [last] masked_MAE={m_mae_last:.4f}, masked_RMSE={m_rmse_last:.4f}, masked_MAPE={m_mape_last*100:.2f}%")
        print(f"  Per-horizon vMAE: {np.array2string(vMAE_per_horizon, precision=4)}")

        # Store results
        # Channel indices are assumed to be [flow, occupancy, speed].
        # Per-channel MAE in original units (averaged over batches, horizons, nodes)
        mae_per_channel = np.mean(np.abs(preds_all - gts_all), axis=(0, 1, 2))
        row = {
            "sparsity": rho,
            "num_masked": len(masked_idx),
            "MAE": mae,
            "RMSE": rmse,
            # Use masked MAPE as the primary MAPE (percent), matching LibCity evaluator outputs.
            "MAPE": m_mape * 100.0,
            # Keep unmasked MAPE (percent) for reference.
            "MAPE_unmasked": mape,
            # Also keep masked MAPE in fractional form for downstream analyses.
            "masked_MAPE_frac": m_mape,
            "MAE_last": mae_last,
            "RMSE_last": rmse_last,

            "MAE_flow":       float(mae_per_channel[0]),
            "MAE_occupancy":  float(mae_per_channel[1]),
            "MAE_speed":      float(mae_per_channel[2]),

            "vMAE_flow":       float(vMAE_c[0]),
            "vMAE_occupancy":  float(vMAE_c[1]),
            "vMAE_speed":      float(vMAE_c[2]),

            "vRMSE_flow":      float(vRMSE_c[0]),
            "vRMSE_occupancy": float(vRMSE_c[1]),
            "vRMSE_speed":     float(vRMSE_c[2]),

            "vMAPE_flow":      float(vMAPE_c[0]),
            "vMAPE_occupancy": float(vMAPE_c[1]),
            "vMAPE_speed":     float(vMAPE_c[2]),

            "vMAE_a": vMAE_a,
            "vRMSE_a": vRMSE_a,
            "vMAPE_a": vMAPE_a,

            "masked_MAE":  m_mae,
            "masked_RMSE": m_rmse,
            # Kept for backward-compatibility; prefer 'MAPE' (percent) or 'masked_MAPE_frac' (fraction).
            "masked_MAPE": m_mape,

            "masked_zMAE":  mz_mae,
            "masked_zRMSE": mz_rmse,
            "masked_zMAPE": mz_mape,
            # For direct comparison with LibCity evaluator magnitudes
            "eval_like_MAE": mz_mae,
            "eval_like_RMSE": mz_rmse,
            "eval_like_MAPE": mz_mape,
            # Last-horizon masked metrics (percent + fraction)
            "masked_MAE_last": m_mae_last,
            "masked_RMSE_last": m_rmse_last,
            "masked_MAPE_last": m_mape_last,           # fraction
            "masked_MAPE_last_percent": m_mape_last*100.0,
        }

        # Add scheme-specific columns for scheme mode
        if args.sparsity_mode == "scheme":
            row["scheme_name"] = scheme_name
            row["group_name"] = group_name
            row["scheme_description"] = schemes_data.get(scheme_name, {}).get("description", "")
        
        # Add per-horizon vMAE columns (H1, H2, ..., H12)
        for h_idx, vmae_h in enumerate(vMAE_per_horizon):
            row[f"vMAE_H{h_idx+1}"] = float(vmae_h)
        
        row.update(extra)
        results.append(row)
        
        # Track results by group for scheme mode
        if args.sparsity_mode == "scheme":
            if group_name not in results_by_group:
                results_by_group[group_name] = []
            results_by_group[group_name].append(row)

        # Write per-sensor CSVs for this iteration
        if args.save_series_csvs:
            header = []
            for c in range(output_dim):
                header.extend([f"Channel{c}_Input", f"Channel{c}_GT", f"Channel{c}_Pred"])
            for n in range(num_nodes):
                csv_path = os.path.join(rho_dir, f"sensor_{n:04d}.csv")
                if len(series_rows_by_node[n]) == 0:
                    # Create an empty file with header to indicate no data
                    pd.DataFrame(columns=header).to_csv(csv_path, index=False)
                else:
                    pd.DataFrame(series_rows_by_node[n], columns=header).to_csv(csv_path, index=False)
            print(f"[Series] Saved per-sensor CSVs for {iter_label} to {rho_dir}")

        # Generate topology visualization for this sparsity level/scheme
        if args.visualize_topology and TOPOLOGY_VIS_AVAILABLE and topology_graph is not None:
            try:
                # Compute per-sensor MAE from arrays
                mae_per_sensor = calculate_mae_per_sensor_from_arrays(
                    preds_all, gts_all, channel_idx=args.topology_channel
                )
                
                if args.sparsity_mode == "sweep":
                    vis_output_dir = run_dir
                    vis_prefix = f"topology_rho{rho:.2f}"
                else:
                    # Scheme mode: put in group subdirectory
                    vis_output_dir = current_group_dir if current_group_dir else run_dir
                    vis_prefix = f"topology_{scheme_name}"
                
                # Store baseline for comparison plots
                if (args.sparsity_mode == "sweep" and rho == 0.0) or \
                   (args.sparsity_mode == "scheme" and scheme_name == "baseline"):
                    baseline_mae_per_sensor = mae_per_sensor.copy()
                    
                    # Generate baseline topology plot
                    baseline_output = os.path.join(vis_output_dir, f"{vis_prefix}_channel{args.topology_channel}.png")
                    visualize_mae_on_topology(
                        topology_graph, mae_per_sensor, baseline_output,
                        title=f"{model_name} - {topology_channel_name} MAE by Sensor",
                        channel_name=topology_channel_name,
                        masked_nodes=masked_idx,
                        rho=rho
                    )
                    
                    # Generate ranking plot for baseline
                    ranking_output = os.path.join(vis_output_dir, f"{vis_prefix}_ranking_channel{args.topology_channel}.png")
                    create_mae_ranking_plot(
                        mae_per_sensor, topology_graph, ranking_output,
                        masked_nodes=masked_idx, rho=rho
                    )
                    print_top_bottom_sensors(mae_per_sensor, topology_graph, masked_nodes=masked_idx)
                else:
                    # Generate comparison plot (baseline vs current)
                    if baseline_mae_per_sensor is not None:
                        comparison_output = os.path.join(
                            vis_output_dir, 
                            f"{vis_prefix}_comparison_channel{args.topology_channel}.png"
                        )
                        visualize_mae_comparison(
                            topology_graph, baseline_mae_per_sensor, mae_per_sensor,
                            comparison_output,
                            rho=rho,
                            masked_nodes=masked_idx,
                            title_prefix=model_name,
                            channel_name=topology_channel_name
                        )
                        
                        # Store scheme data for overview plot
                        scheme_data_list.append({
                            'name': scheme_name if args.sparsity_mode == "scheme" else f"rho={rho:.0%}",
                            'mae_per_sensor': mae_per_sensor.copy(),
                            'masked_nodes': list(masked_idx),
                            'rho': rho,
                            'group': group_name if args.sparsity_mode == "scheme" else "sweep"
                        })
                    else:
                        # No baseline available, just do single plot
                        single_output = os.path.join(vis_output_dir, f"{vis_prefix}_channel{args.topology_channel}.png")
                        visualize_mae_on_topology(
                            topology_graph, mae_per_sensor, single_output,
                            title=f"{model_name} - {topology_channel_name} MAE by Sensor",
                            channel_name=topology_channel_name,
                            masked_nodes=masked_idx,
                            rho=rho
                        )
                
                print(f"[Topology] Generated visualization for {iter_label}")
            except Exception as e:
                print(f"[Topology] Error generating visualization: {e}")
                import traceback
                traceback.print_exc()

    # Save CSV and plots

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved results to: {args.out_csv}")

    # Plot per-variable MAE vs sparsity (sweep mode only - line plot)
    if args.plot_metrics and args.sparsity_mode == "sweep":
        try:
            dfp = df.copy()
            x = dfp["sparsity"].values
            y_flow = dfp["vMAE_flow"].values.astype(float)
            y_occ = dfp["vMAE_occupancy"].values.astype(float)
            y_speed = dfp["vMAE_speed"].values.astype(float)
            plt.figure(figsize=(8.5, 5.0))
            plt.plot(x, y_flow, marker="o", linewidth=2, label="flow")
            plt.plot(x, y_occ, marker="s", linewidth=2, label="occupancy")
            plt.plot(x, y_speed, marker="^", linewidth=2, label="speed")
            plt.xlabel("Sparsity (masked node fraction)")
            plt.ylabel("vMAE (normalized)")
            plt.title("Sparsity vs normalized MAE per variable")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out_png = args.out_csv.replace(".csv", "_mae_per_variable.png")
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"Saved metrics plot to: {out_png}")
        except Exception as e:
            print(f"[warn] Failed to generate metrics plot: {e}")
    
    # Save masks registry
    with open(masks_json, "w") as f:
        json.dump(masks_registry, f, indent=2)
    print(f"Saved masks registry to: {masks_json}")
    
    # Generate overview plot showing all schemes/sparsity levels
    if args.visualize_topology and TOPOLOGY_VIS_AVAILABLE and topology_graph is not None:
        if baseline_mae_per_sensor is not None and len(scheme_data_list) > 0:
            try:
                overview_output = os.path.join(run_dir, f"schemes_overview_channel{args.topology_channel}.png")
                create_schemes_overview_plot(
                    topology_graph, baseline_mae_per_sensor, scheme_data_list,
                    overview_output,
                    title_prefix=model_name,
                    channel_name=topology_channel_name
                )
            except Exception as e:
                print(f"[Overview] Error generating overview plot: {e}")
                import traceback
                traceback.print_exc()

    # Summary of topology visualizations
    if args.visualize_topology and TOPOLOGY_VIS_AVAILABLE and topology_graph is not None:
        print(f"\n[Topology] Visualization summary:")
        print(f"  - Channel: {args.topology_channel} ({topology_channel_name})")
        if args.sparsity_mode == "sweep":
            print(f"  - Generated {len(args.sparsity)} topology visualizations (one per sparsity level)")
        else:
            print(f"  - Generated {len(scheme_iter_list)} topology visualizations (one per scheme)")
        print(f"  - Output directory: {run_dir}")
    elif args.visualize_topology and not TOPOLOGY_VIS_AVAILABLE:
        print(f"\n[Topology] Visualization was requested but module was not available.")
    elif args.visualize_topology and topology_graph is None:
        print(f"\n[Topology] Visualization was requested but network topology could not be loaded.")
    
    # Save raw predictions to .npz file
    if len(predictions_storage) > 0:
        npz_output_path = args.out_csv.replace(".csv", "_predictions.npz")
        print(f"\n[Saving] Writing raw predictions to: {npz_output_path}")
        
        # Prepare data for saving
        npz_data = {}
        npz_data['metadata'] = json.dumps({
            'model': model_name,
            'dataset': dataset_name,
            'sparsity_mode': args.sparsity_mode,
            'num_schemes': len(predictions_storage),
            'output_dim': output_dim,
            'num_nodes': num_nodes,
        }, default=str)
        
        # Save each scheme's predictions
        for key, data in predictions_storage.items():
            # Use key as prefix (e.g., "rho_0.30" or "scheme_name")
            npz_data[f'{key}_predictions'] = data['predictions']
            npz_data[f'{key}_ground_truth'] = data['ground_truth']
            npz_data[f'{key}_masked_nodes'] = np.array(data['masked_nodes'], dtype=np.int32)
            npz_data[f'{key}_metadata'] = json.dumps({
                'scheme_name': data['scheme_name'],
                'rho': float(data['rho']),
                'group_name': data['group_name'],
            }, default=str)
        
        # Save to .npz
        np.savez_compressed(npz_output_path, **npz_data)
        print(f"[Saving] Saved {len(predictions_storage)} schemes/sparsity levels to {npz_output_path}")
        print(f"         Shape per scheme: predictions {predictions_storage[list(predictions_storage.keys())[0]]['predictions'].shape}")
    
    # Close the log file
    print(f"\nEvaluation complete. All outputs saved to: {run_dir}")
    if 'tee_output' in dir():
        tee_output.close()
        sys.stdout = tee_output.stdout
        sys.stderr = tee_output.stderr


if __name__ == "__main__":
    main()
