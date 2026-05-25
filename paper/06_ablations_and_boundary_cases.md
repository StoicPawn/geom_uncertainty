# Ablations And Boundary Cases

Confirmatory controls that support the main claims:

- `statistical_diagnostics`
- `full_vocab_sanity`
- `topk_robustness`
- `fisher_output_energy_control`
- `semantic_preservation`

Exploratory and diagnostic controls that bound the claims:

- `random_subspaces`
- `euclidean_ablation`
- `shuffled_surprisal`
- `gradient_baselines`
- `full_regression`

Boundary cases include top-k sensitivity, projected-gradient baselines that outperform `rho` as raw local predictors, duplicate or near-duplicate prompts, and failure modes where high `rho` does not produce large movement or answer neighborhoods degrade.
