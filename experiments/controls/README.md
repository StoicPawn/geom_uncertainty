# Controls, Ablations, And Boundary Cases

This folder separates controls from the four main experiments while keeping them close to the claims they support.

| Control | Question addressed | Primary output |
|---|---|---|
| `statistical_diagnostics/` | What are the bootstrap CIs, effect sizes, duplicate checks, and failure modes? | `outputs/bootstrap_steering_contrasts.csv` |
| `full_vocab_sanity/` | Does top-k accessibility track full-vocabulary accessibility on the tiny model? | `outputs/topk_vs_full_vocab_summary.csv` |
| `out_of_sample_generalization/` | Do train-estimated routes transfer to held-out prompts? | `outputs/oos_contrasts.csv` |
| `random_init_vs_pretrained/` | Is accessibility mostly learned structure or random architectural geometry? | `outputs/pretrained_vs_random_contrasts.csv` |
| `topk_robustness/` | Are rho rankings, layer trends, and steering effects stable across output top-k? | `outputs/topk_rank_stability.csv` |
| `gradient_baselines/` | Does rho add information relative to projected gradients and Fisher/Jacobian norms? | `outputs/gradient_baseline_correlations.csv` |
| `full_regression/` | Does rho remain predictive after scalar, geometric, model, layer, and prompt controls? | `outputs/full_regression_ols.csv` |
| `semantic_preservation/` | Does steering preserve top-5/top-10/KL/candidate-mass structure beyond top-1? | `outputs/semantic_topk_preservation_summary.csv` |
| `random_subspaces/` | Is accessibility just a generic random-subspace effect? | `outputs/random_control_contrasts.csv` |
| `euclidean_ablation/` | Does Fisher geometry add information beyond Euclidean geometry? | `outputs/metric_ablation_scores.csv` |
| `shuffled_surprisal/` | Does the true centered-surprisal direction matter? | `outputs/metric_ablation_scores.csv` |
| `fisher_output_energy_control/` | Does accessible steering still win at equal output movement? | `outputs/equal_output_energy_contrasts.csv` |

The paper-level summary is in `reports/final_reproducibility_and_results_report.md`.
