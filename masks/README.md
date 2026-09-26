# Sparsity mask definitions

The reported sweep masks are generated with `numpy.random.default_rng` and fixed seeds 43, 44, and 45. For each sparsity level `rho`:

```python
k = int(round(rho * num_nodes))
rng = np.random.default_rng(seed + int(round(rho * 100)))
indices = sorted(rng.choice(num_nodes, k, replace=False))
```

The mask is static across batches and all 12 input timesteps. The same mask for a given dataset, grid, seed, and sparsity level is reused across models.

- P10 (`paper`): 0%, 10%, ..., 100%.
- P5 (`breakdown`): 0%, 5%, ..., 35%.

The 12 JSON files under `multi_seed/` are the released PEMS04/PEMS08 masks for both grids and all three seeds. Regenerate them with:

```bash
python masks/generate_multi_seed_masks.py
```

Each evaluation also records the realized indices in `sparsity_results_masks.json`.
