# Control: Top-k Robustness

## Purpose

Test whether accessibility and steering conclusions are stable under the output-lens choice.

## Top-k Values

`16, 32, 64, 128, 256`

## Outputs

- `outputs/topk_scores.csv`: rho, scalar uncertainty, gradient baselines, Fisher/Jacobian norms by top-k.
- `outputs/topk_steering_records.csv`: per-intervention accessible/random/grad-orthogonal steering records.
- `outputs/topk_robustness_summary.csv`: aggregate rho summaries.
- `outputs/topk_rank_stability.csv`: rho ranking stability against top-k 32.
- `outputs/topk_layer_trend_stability.csv`: layer-trend stability against top-k 32.
- `outputs/topk_steering_summary.csv`: steering effect and preservation summary by top-k.

## Figures

- `figures/fig07_topk_robustness.svg`
- `figures/fig07_topk_robustness.png`

## Reproduce

Use `config/reproduce.json`.
