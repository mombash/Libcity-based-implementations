#!/usr/bin/env python3
"""
Generate figures using the FINAL OME weighting scheme from the paper:
  w(ρ) = e^(-20ρ²)
  W(ρ) = w(ρ) / Σw(ρ')
  v̄MAE_a = Σ W(ρ) × vMAE_a(ρ)

This applies weights to raw vMAE values (not degradation).
Uses 35% cutoff with 5% intervals.
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import os

matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, "sparcity_cache")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sparsity_analysis_paper", "ome_final")

MODELS = ['DCRNN', 'D2STGNN', 'MCSTMambaLST', 'Trafformer']
DISPLAY = {'DCRNN': 'DCRNN', 'D2STGNN': 'D2STGNN', 'MCSTMambaLST': 'Mamba4Traffic', 'Trafformer': 'Trafformer'}
MODEL_COLORS = {'DCRNN': '#1f77b4', 'D2STGNN': '#2ca02c', 'MCSTMambaLST': '#9467bd', 'Trafformer': '#ff7f0e'}
MODEL_MARKERS = {'DCRNN': 'o', 'D2STGNN': 's', 'MCSTMambaLST': 'D', 'Trafformer': '^'}
MECHANISM_COLORS = {'RNN-GNN': '#1f77b4', 'Dynamic-GNN': '#2ca02c', 'Transformer': '#ff7f0e', 'State-Space': '#9467bd'}
SPATIAL_MECHANISM = {'DCRNN': 'RNN-GNN', 'D2STGNN': 'Dynamic-GNN', 'Trafformer': 'Transformer', 'MCSTMambaLST': 'State-Space'}

SPARSITY_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_finer_data(dataset):
    data = []
    for model in MODELS:
        if model == 'MCSTMambaLST':
            pattern = f"{CACHE_DIR}/MCSTMambaLST_Ablation_{dataset}_*-20260120*/sparsity_results.csv"
        else:
            pattern = f"{CACHE_DIR}/{model}_{dataset}_*-20260120*/sparsity_results.csv"
        files = sorted(glob.glob(pattern))
        if files:
            df = pd.read_csv(files[-1])
            df = df[df['sparsity'] <= 0.35]
            for _, row in df.iterrows():
                data.append({'model': model, 'sparsity': row['sparsity'], 'vMAE_a': row['vMAE_a']})
    return pd.DataFrame(data)


def gaussian_weights():
    """w(ρ) = e^(-20ρ²) - Gaussian-like decay centered at 0"""
    raw = {s: np.exp(-20 * s**2) for s in SPARSITY_LEVELS}
    total = sum(raw.values())
    return {s: w/total for s, w in raw.items()}


def uniform_weights():
    return {s: 1.0/len(SPARSITY_LEVELS) for s in SPARSITY_LEVELS}


def compute_weighted_vmae(df, weights):
    """Compute weighted average vMAE (OME)"""
    results = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model].set_index('sparsity')
        weighted_sum = sum(weights.get(s, 0) * model_df.loc[s, 'vMAE_a'] 
                          for s in weights if s in model_df.index)
        results[model] = weighted_sum
    return results


def compute_weighted_degradation(df, weights):
    """Compute weighted average degradation D̄"""
    results = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model].set_index('sparsity')
        if 0.0 not in model_df.index:
            continue
        baseline = model_df.loc[0.0, 'vMAE_a']
        weighted_sum = sum(weights.get(s, 0) * (model_df.loc[s, 'vMAE_a'] / baseline) 
                          for s in weights if s in model_df.index)
        results[model] = weighted_sum
    return results


def ds_name(dataset):
    return 'PEMS04' if dataset == 'PEMSD4' else 'PEMS08'


# ============================================================================
# FIGURE 1: vMAE Degradation with Threshold
# ============================================================================
def fig1(dataset, df):
    threshold = 0.33 if dataset == 'PEMSD4' else 0.42
    fig, ax = plt.subplots(figsize=(12, 8))
    
    crossings = []
    for model in MODELS:
        model_df = df[df['model'] == model].sort_values('sparsity')
        if model_df.empty:
            continue
        sparsities = model_df['sparsity'].values * 100
        vmaes = model_df['vMAE_a'].values
        ax.plot(sparsities, vmaes, marker=MODEL_MARKERS[model], linewidth=3, markersize=10,
               label=DISPLAY[model], color=MODEL_COLORS[model])
        
        if vmaes[0] >= threshold:
            crossings.append((model, 0))
            ax.plot(0, threshold, '+', markersize=15, markeredgewidth=3, color=MODEL_COLORS[model])
        else:
            for i in range(1, len(vmaes)):
                if vmaes[i-1] < threshold <= vmaes[i]:
                    frac = (threshold - vmaes[i-1]) / (vmaes[i] - vmaes[i-1])
                    cross_s = sparsities[i-1] + frac * (sparsities[i] - sparsities[i-1])
                    crossings.append((model, cross_s))
                    ax.plot(cross_s, threshold, '+', markersize=15, markeredgewidth=3, color=MODEL_COLORS[model])
                    break
    
    if dataset == 'PEMSD8':
        label_offsets = {'Trafformer': (-15, -25), 'D2STGNN': (15, -25), 'MCSTMambaLST': (-15, 18), 'DCRNN': (15, 18)}
    else:
        # PEMS04: Trafformer starts above threshold, move its 0% label to the right and below
        label_offsets = {'DCRNN': (0, 18), 'D2STGNN': (0, -25), 'MCSTMambaLST': (0, 18), 'Trafformer': (25, -25)}
    
    for model, cross_s in crossings:
        offset = label_offsets.get(model, (0, 18))
        ax.annotate(f'{cross_s:.0f}%', (cross_s, threshold), textcoords='offset points', 
                   xytext=offset, fontsize=16, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=MODEL_COLORS[model], linewidth=2))
    
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.8)
    # Threshold text on right side, below the line
    ax.text(36, threshold - 0.03, f'10% Error Threshold (vMAE$_a$={threshold:.2f})', fontsize=12, color='red', ha='right', va='top')
    ax.set_xlabel('Sensor Sparsity (%)', fontsize=20)
    ax.set_ylabel('vMAE$_a$', fontsize=20)
    ax.set_title(f'{ds_name(dataset)}: vMAE$_a$ Degradation Under Sparsity', fontsize=24, fontweight='bold')
    ax.legend(loc='upper left', fontsize=16)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(-1, 37)
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_{dataset.lower().replace('pemsd','pems')}_vmae_threshold.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig1_{dataset.lower().replace('pemsd','pems')}_vmae_threshold.pdf")


# ============================================================================
# FIGURE 2: Ranking Heatmap (per-sparsity level)
# ============================================================================
def fig2(dataset, df):
    pivot = df.pivot(index='model', columns='sparsity', values='vMAE_a')
    pivot.index = pivot.index.map(DISPLAY)
    ranks = pivot.rank(axis=0, method='min')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(ranks, annot=True, fmt='.0f', cmap='Reds', cbar=False, ax=ax,
                linewidths=0.5, linecolor='white', annot_kws={'size': 18, 'weight': 'bold'})
    ax.set_xlabel('Sparsity Level', fontsize=20)
    ax.set_ylabel('Model', fontsize=20)
    ax.set_title(f'{ds_name(dataset)}: Model Ranking by vMAE$_a$ at Each Sparsity Level', fontsize=24, fontweight='bold')
    ax.set_xticklabels([f'{int(s*100)}%' for s in SPARSITY_LEVELS], rotation=0, fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_{dataset.lower().replace('pemsd','pems')}_ranking_heatmap.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig2_{dataset.lower().replace('pemsd','pems')}_ranking_heatmap.pdf")


# ============================================================================
# FIGURE 5: Degradation Curves (relative to baseline)
# ============================================================================
def fig5(dataset, df):
    fig, ax = plt.subplots(figsize=(12, 8))
    for model in MODELS:
        model_df = df[df['model'] == model].sort_values('sparsity')
        if model_df.empty or 0.0 not in model_df['sparsity'].values:
            continue
        baseline = model_df[model_df['sparsity'] == 0.0]['vMAE_a'].values[0]
        relative = model_df['vMAE_a'] / baseline
        ax.plot(model_df['sparsity'] * 100, relative, marker=MODEL_MARKERS[model], 
               linewidth=3, markersize=10, label=DISPLAY[model], color=MODEL_COLORS[model])
    
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=2)
    ax.set_xlabel('Sensor Sparsity (%)', fontsize=20)
    ax.set_ylabel(r'Degradation $D(\rho) = \frac{vMAE_a(\rho)}{vMAE_a(0)}$', fontsize=20)
    ax.set_title(f'{ds_name(dataset)}: Relative Degradation Curves', fontsize=24, fontweight='bold')
    ax.legend(loc='upper left', fontsize=16)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(-1, 37)
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_{dataset.lower().replace('pemsd','pems')}_degradation.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig5_{dataset.lower().replace('pemsd','pems')}_degradation.pdf")


# ============================================================================
# FIGURE 6 OPT1: Accuracy vs Robustness Scatter (using weighted degradation D̄)
# ============================================================================
def fig6_opt1(dataset, df):
    weights = gaussian_weights()
    results = []
    for model in MODELS:
        model_df = df[df['model'] == model].set_index('sparsity')
        if 0.0 not in model_df.index:
            continue
        baseline = model_df.loc[0.0, 'vMAE_a']
        weighted_deg = sum(weights.get(s, 0) * (model_df.loc[s, 'vMAE_a'] / baseline) for s in weights if s in model_df.index)
        results.append({'model': model, 'display': DISPLAY[model], 'baseline': baseline, 'avg_deg': weighted_deg,
                       'mechanism': SPATIAL_MECHANISM[model], 'color': MECHANISM_COLORS[SPATIAL_MECHANISM[model]]})
    results = sorted(results, key=lambda x: x['avg_deg'])
    for i, r in enumerate(results): 
        r['rank'] = i + 1
    
    fig, ax = plt.subplots(figsize=(12, 9))
    for mechanism in ['State-Space', 'RNN-GNN', 'Dynamic-GNN', 'Transformer']:
        mech_results = [r for r in results if r['mechanism'] == mechanism]
        for r in mech_results:
            ax.scatter(r['baseline'], r['avg_deg'], s=500, c=r['color'], edgecolors='black', linewidths=2,
                      label=mechanism if r == mech_results[0] else '_nolegend_', alpha=0.85, zorder=5)
    
    label_offsets = {'DCRNN': (60, 0), 'D2STGNN': (60, 0), 'MCSTMambaLST': (60, 0), 'Trafformer': (-70, 0)}
    for r in results:
        ax.annotate(f"#{r['rank']}", xy=(r['baseline'], r['avg_deg']), fontsize=14, fontweight='bold', 
                   ha='center', va='center', color='white', zorder=10)
        offset = label_offsets.get(r['model'], (60, 0))
        ax.annotate(r['display'], xy=(r['baseline'], r['avg_deg']), xytext=offset, textcoords='offset points',
                   fontsize=16, fontweight='bold', ha='left' if offset[0] > 0 else 'right', va='center',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2), alpha=0.95)
    
    ax.set_xlabel(r'Baseline vMAE$_a$ (at 0% sparsity)' + '\n← Better Accuracy (lower)', fontsize=20)
    ax.set_ylabel(r'Weighted Degradation $\bar{D}$' + '\n← More Robust (lower)', fontsize=20)
    ax.set_title(f'{ds_name(dataset)}: Accuracy vs Robustness Tradeoff', fontsize=24, fontweight='bold')
    ax.legend(loc='upper right', fontsize=16)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_opt1_{dataset.lower().replace('pemsd','pems')}_scatter.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig6_opt1_{dataset.lower().replace('pemsd','pems')}_scatter.pdf")


# ============================================================================
# FIGURE 6 OPT2: Accuracy vs Robustness Bars
# ============================================================================
def fig6_opt2(dataset, df):
    weights = gaussian_weights()
    results = []
    for model in MODELS:
        model_df = df[df['model'] == model].set_index('sparsity')
        if 0.0 not in model_df.index:
            continue
        baseline = model_df.loc[0.0, 'vMAE_a']
        weighted_deg = sum(weights.get(s, 0) * (model_df.loc[s, 'vMAE_a'] / baseline) for s in weights if s in model_df.index)
        results.append({'model': model, 'display': DISPLAY[model], 'baseline': baseline, 'avg_deg': weighted_deg, 'color': MODEL_COLORS[model]})
    results = sorted(results, key=lambda x: x['avg_deg'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    y_pos = np.arange(len(results))
    displays = [r['display'] for r in results]
    colors = [r['color'] for r in results]
    
    baselines = [r['baseline'] for r in results]
    bars1 = ax1.barh(y_pos, baselines, color=colors, edgecolor='none', height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(displays, fontsize=18)
    ax1.set_xlabel(r'Baseline vMAE$_a$', fontsize=16)
    ax1.set_title('Baseline Accuracy', fontsize=22, fontweight='bold')
    ax1.set_xlim(0, max(baselines) * 1.4)
    for bar, val in zip(bars1, baselines):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', ha='left', va='center', fontsize=16, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    degs = [r['avg_deg'] for r in results]
    bars2 = ax2.barh(y_pos, degs, color=colors, edgecolor='none', height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(displays, fontsize=18)
    ax2.set_xlabel(r'Weighted Degradation $\bar{D}$', fontsize=16)
    ax2.set_title('Robustness', fontsize=22, fontweight='bold')
    ax2.set_xlim(0, max(degs) * 1.3)
    for bar, val in zip(bars2, degs):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}×', ha='left', va='center', fontsize=16, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle(f'{ds_name(dataset)}: Accuracy vs Robustness', fontsize=26, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/fig6_opt2_{dataset.lower().replace('pemsd','pems')}_bars.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig6_opt2_{dataset.lower().replace('pemsd','pems')}_bars.pdf")


# ============================================================================
# FIGURE OME: Uniform vs Weighted vMAE Comparison
# ============================================================================
def fig_ome_comparison(dataset, df):
    ome_weighted = compute_weighted_vmae(df, gaussian_weights())
    ome_uniform = compute_weighted_vmae(df, uniform_weights())
    
    models_sorted = sorted(ome_weighted.keys(), key=lambda m: ome_weighted[m])
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(models_sorted))
    width = 0.35
    
    uniform_vals = [ome_uniform[m] for m in models_sorted]
    ome_vals = [ome_weighted[m] for m in models_sorted]
    displays = [DISPLAY[m] for m in models_sorted]
    
    bars1 = ax.barh(y_pos - width/2, uniform_vals, width, label='Uniform Mean', color='#7fbf7b', edgecolor='black')
    bars2 = ax.barh(y_pos + width/2, ome_vals, width, label='Weighted (w=e^{-20ρ²})', color='#af8dc3', edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(displays, fontsize=18)
    ax.set_xlabel('Average vMAE$_a$', fontsize=18)
    ax.set_title(f'{ds_name(dataset)}: Uniform vs Weighted Average vMAE$_a$', fontsize=24, fontweight='bold')
    ax.legend(loc='lower right', fontsize=14)
    ax.grid(axis='x', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, val in zip(bars1, uniform_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', ha='left', va='center', fontsize=12)
    for bar, val in zip(bars2, ome_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', ha='left', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_ome_{dataset.lower().replace('pemsd','pems')}_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig_ome_{dataset.lower().replace('pemsd','pems')}_comparison.pdf")


# ============================================================================
# WEIGHT VISUALIZATION
# ============================================================================
def fig_weight_distribution():
    """Visualize the Gaussian weighting scheme"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    weights = gaussian_weights()
    x = [s * 100 for s in SPARSITY_LEVELS]
    y = [weights[s] * 100 for s in SPARSITY_LEVELS]
    
    ax.bar(x, y, width=4, color='#3498db', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Sparsity Level (%)', fontsize=16)
    ax.set_ylabel('Weight (%)', fontsize=16)
    ax.set_title(r'Weighting Scheme: $w(\rho) = e^{-20\rho^2}$', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(s)}%' for s in x], fontsize=14)
    ax.grid(axis='y', alpha=0.4)
    ax.tick_params(axis='y', labelsize=14)
    
    # Add weight values on bars
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 1, f'{yi:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_weight_distribution.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig_weight_distribution.pdf")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("Generating FINAL OME Figures")
    print("Weighting: w(ρ) = e^(-20ρ²)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Show weights
    weights = gaussian_weights()
    print("\nWeight Distribution:")
    for s in SPARSITY_LEVELS:
        print(f"  ρ={s*100:4.0f}%: w={weights[s]*100:5.1f}%")
    
    # Generate weight visualization
    print("\nGenerating figures...")
    fig_weight_distribution()
    
    for dataset in ['PEMSD4', 'PEMSD8']:
        print(f"\n{ds_name(dataset)}:")
        df = load_finer_data(dataset)
        
        if df.empty:
            print(f"  ERROR: No data found")
            continue
        
        fig1(dataset, df)
        fig2(dataset, df)
        fig5(dataset, df)
        fig6_opt1(dataset, df)
        fig6_opt2(dataset, df)
        fig_ome_comparison(dataset, df)
        
        # Print summary
        print(f"\n  Summary for {ds_name(dataset)}:")
        ome = compute_weighted_vmae(df, weights)
        deg = compute_weighted_degradation(df, weights)
        sorted_by_ome = sorted(ome.keys(), key=lambda m: ome[m])
        sorted_by_deg = sorted(deg.keys(), key=lambda m: deg[m])
        
        print(f"  Ranking by Weighted vMAE (lower = better accuracy):")
        for rank, m in enumerate(sorted_by_ome, 1):
            print(f"    #{rank} {DISPLAY[m]:<15} v̄MAE={ome[m]:.3f}")
        
        print(f"  Ranking by Weighted Degradation (lower = more robust):")
        for rank, m in enumerate(sorted_by_deg, 1):
            print(f"    #{rank} {DISPLAY[m]:<15} D̄={deg[m]:.3f}×")
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
