# Claim Map

This file maps each paper claim to the experiment, primary CSV artifacts, paper-ready figure, and reproduction config.

| Paper claim | Experiment | CSV support | Figure | Config |
|---|---|---|---|---|
| Scalar entropy/varentropy does not determine accessibility. | `experiments/01_matched_scalar_uncertainty/` | `outputs/same_uncertainty_different_rho_pairs.csv`; `outputs/same_uncertainty_different_accessibility_pairs.csv` | `figures/fig02_scalar_matched_pairs.svg` | `config/reproduce.json` |
| Accessibility predicts local uncertainty movement. | `experiments/02_local_perturbation_prediction/` | `outputs/intervention_by_rho_quartile.csv`; `outputs/intervention_residual_correlations.csv`; `outputs/intervention_records.csv` | `figures/fig03_accessibility_predicts_movement.svg` | `config/reproduce.json` |
| Accessibility has layerwise and k-dimensional structure. | `experiments/03_layerwise_k_structure/` | `outputs/layerwise_k_summary.csv`; `outputs/layerwise_k_scores.csv`; `outputs/compressibility_summary.csv` | `figures/fig04_layerwise_heatmap.svg`; `figures/fig05_compressibility_curves.svg` | `config/reproduce.json` |
| Accessible steering changes uncertainty more efficiently than controls. | `experiments/04_uncertainty_steering/` | `outputs/equal_output_energy_contrasts.csv`; `outputs/directionality_summary.csv`; `outputs/specificity_summary.csv`; `outputs/rho_dependency.csv` | `figures/fig06_uncertainty_steering_main.svg` | `config/reproduce.json` |
| Random subspaces do not explain the full signal. | `experiments/controls/random_subspaces/` | `outputs/subspace_scores_with_random.csv`; `outputs/random_control_contrasts.csv` | summarized in Fig. 6 | `config/reproduce.json` |
| Fisher geometry matters beyond Euclidean projection. | `experiments/controls/euclidean_ablation/` | `outputs/metric_ablation_scores.csv`; `outputs/prompt_features_ablation.csv` | control table artifact | `config/reproduce.json` |
| The real centered-surprisal direction matters. | `experiments/controls/shuffled_surprisal/` | `outputs/metric_ablation_scores.csv`; `outputs/prompt_features_ablation.csv` | control table artifact | `config/reproduce.json` |
| Steering is not merely larger output movement. | `experiments/controls/fisher_output_energy_control/` | `outputs/equal_output_energy_contrasts.csv` | Fig. 6 | `config/reproduce.json` |
