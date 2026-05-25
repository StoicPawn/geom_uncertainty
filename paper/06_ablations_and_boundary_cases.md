# Ablations And Boundary Cases

Confirmatory controls that support the main claims:

- `statistical_diagnostics`
- `full_vocab_sanity`
- `out_of_sample_generalization`
- `random_init_vs_pretrained`
- `topk_robustness`
- `fisher_output_energy_control`
- `semantic_preservation`

Exploratory and diagnostic controls that bound the claims:

- `random_subspaces`
- `euclidean_ablation`
- `shuffled_surprisal`
- `gradient_baselines`
- `full_regression`

Boundary cases include top-k sensitivity, projected-gradient baselines that outperform `rho` as raw local predictors, train-route generalization that is close to held-out oracle but not clearly better than random routes, duplicate or near-duplicate prompts, and failure modes where high `rho` does not produce large movement or answer neighborhoods degrade.
