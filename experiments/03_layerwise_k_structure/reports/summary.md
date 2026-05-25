# Summary: Layerwise And K-Dimensional Structure

## Claim

Accessibility varies systematically across layers and subspace dimension `k`, and it accumulates as more latent directions are included.

## Main Result

- Layers covered: `0..6`.
- Dimensions covered: `k = 1, 2, 4, 8, 16, 32`.
- Subspaces covered: `delta_forward_1d`, `delta_pca`, `state_pca`.
- Maximum mean adjusted accessibility: `0.3192` for `delta_forward_1d`, layer `5`, correct examples.
- Mean `k50`: `11.80`.
- Mean `k80`: `23.85`.

## Claim Support

| Claim component | File |
|---|---|
| Layer x k structure | `outputs/layerwise_k_summary.csv`; `outputs/layerwise_k_scores.csv` |
| Heatmap | `figures/fig05_layerwise_heatmap.svg` |
| Compressibility | `outputs/compressibility_summary.csv`; `figures/fig05_compressibility_curves.svg` |
| Multi-architecture boundary check | `outputs/multiarch_summary.csv`; `outputs/multiarch_compressibility.csv` |
| Reproduction config | `config/reproduce.json` |
