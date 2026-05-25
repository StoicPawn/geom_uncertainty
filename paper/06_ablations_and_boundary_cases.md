# Ablations And Boundary Cases

Confirmatory controls that support the main claims:

- `statistical_diagnostics`
- `minimal_intervention_energy`
- `equal_output_movement`
- `full_vocab_sanity`
- `out_of_sample_generalization`
- `random_init_vs_pretrained`
- `external_uncertainty_comparators`
- `uncertainty_circuits`
- `topk_robustness`
- `fisher_output_energy_control`
- `semantic_preservation`

Exploratory and diagnostic controls that bound the claims:

- `random_subspaces`
- `euclidean_ablation`
- `shuffled_surprisal`
- `gradient_baselines`
- `full_regression`
- `brittle_confidence`

Boundary cases include top-k sensitivity, projected-gradient baselines that outperform `rho` as raw local predictors, train-route generalization that is close to held-out oracle but not clearly better than random routes, external semantic uncertainty metrics that measure different objects, duplicate or near-duplicate prompts, brittle-confidence perturbations where low-rho high-confidence predictions are not consistently more fragile, and failure modes where high `rho` does not produce large movement or answer neighborhoods degrade.
