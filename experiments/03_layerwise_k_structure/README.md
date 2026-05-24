# Experiment 3: Layerwise And K-Dimensional Structure

## Purpose

Study how accessibility changes across transformer layers, latent subspace dimension `k`, and architecture family. This experiment tests whether the accessibility signal is structured rather than a one-off artifact of a single layer or dimension.

## Data

- `data/prompts.csv`: shared prompt table for layerwise and k-dimensional sweeps.

## Outputs

- `outputs/layerwise_k_summary.csv`: aggregate layer and dimension summary.
- `outputs/layerwise_k_scores.csv`: detailed layer-by-dimension accessibility scores.
- `outputs/layerwise_k_trajectory.csv`: trajectory-style layer summaries.
- `outputs/compressibility_summary.csv`: k-compressibility results.
- `outputs/multiarch_summary.csv`: cross-architecture summary.
- `outputs/multiarch_compressibility.csv`: architecture-level compressibility table.

## Figures

- `figures/fig04_layerwise_heatmap.svg`
- `figures/fig04_layerwise_heatmap.png`
- `figures/fig05_compressibility_curves.svg`
- `figures/fig05_compressibility_curves.png`

## Reports

- `reports/summary.md`
- `reports/layerwise_k_report.md`
- `reports/multiarch_report.md`

## Claim Support

This experiment supports the claim that accessibility has layerwise and k-dimensional structure. The mapping from claim to CSV and figure is also listed in `../../CLAIM_MAP.md`.

## Related Scripts

- `scripts/run_layerwise_k_adjusted_rho.py`
- `scripts/run_multiarch_layerwise_accessibility.py`

## Reproduce

Use `config/reproduce.json` for exact commands and seeds.
