# Controls, Ablations, And Boundary Cases

This folder separates controls from the four main experiments while keeping them close to the claims they support.

| Control | Question addressed | Primary output |
|---|---|---|
| `random_subspaces/` | Is accessibility just a generic random-subspace effect? | `outputs/random_control_contrasts.csv` |
| `euclidean_ablation/` | Does Fisher geometry add information beyond Euclidean geometry? | `outputs/metric_ablation_scores.csv` |
| `shuffled_surprisal/` | Does the true centered-surprisal direction matter? | `outputs/metric_ablation_scores.csv` |
| `fisher_output_energy_control/` | Does accessible steering still win at equal output movement? | `outputs/equal_output_energy_contrasts.csv` |

The paper-level summary is in `reports/final_reproducibility_and_results_report.md`.
