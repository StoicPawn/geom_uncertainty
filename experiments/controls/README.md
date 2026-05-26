# Controls And Appendix Ablations

This directory keeps the controls needed to defend the paper claims.

## Core Controls

| Control | Question addressed |
|---|---|
| `gradient_baselines/` | Do projected gradients and Fisher/Jacobian norms explain the signal? |
| `full_regression/` | Does rho remain useful after scalar, gradient, layer, model, task, and route controls? |
| `semantic_preservation/` | Does steering preserve more than top-1 output identity? |
| `topk_robustness/` | Are rho rankings and steering effects stable across output top-k? |
| `fisher_output_energy_control/` | Does accessible steering still move uncertainty efficiently at equal output movement? |
| `statistical_diagnostics/` | What are the bootstrap CIs, duplicate checks, failure modes, and matched scalar-gradient controls? |

## Appendix Controls

| Control | Question addressed |
|---|---|
| `random_subspaces/` | Is the signal just a generic random-subspace effect? |
| `euclidean_ablation/` | Does Fisher geometry add information beyond Euclidean projection? |
| `shuffled_surprisal/` | Does the true centered-surprisal direction matter? |
| `full_vocab_sanity/` | How close is top-k geometry to full-vocabulary behavior on a tiny model? |
| `out_of_sample_generalization/` | Do train-estimated routes transfer to held-out prompts? |
| `random_init_vs_pretrained/` | How much structure comes from learned weights rather than architecture? |
| `external_uncertainty_comparators/` | How do accessible varentropy and semantic/consistency uncertainty proxies differ? |
