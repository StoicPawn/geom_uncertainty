# Control: Direct Gradient Baselines

## Purpose

Compare accessibility `rho(B)` against direct gradient and Jacobian baselines:

- `||Pi_B grad_z H||`
- `||Pi_B grad_z Var||`
- `||F^{1/2} J B||`
- `||J B||`

## Outputs

- `outputs/gradient_baseline_scores.csv`: per-row predictor values.
- `outputs/gradient_baseline_correlations.csv`: Pearson/Spearman comparisons for `|Delta H|`, `|Delta Var|`, and directional success.

## Figures

- `figures/fig08_gradient_baselines.svg`
- `figures/fig08_gradient_baselines.png`

## Reproduce

Use `config/reproduce.json`.
