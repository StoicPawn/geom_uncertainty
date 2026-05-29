# Appendix Artifact Map

This file separates paper-supporting artifacts from the main narrative spine.

## Appendix controls retained for the paper

| Artifact | Role |
|---|---|
| `experiments/controls/gradient_baselines/` | Gradient baseline comparisons and caveats. |
| `experiments/controls/full_regression/` | Controlled regression checks. |
| `experiments/controls/semantic_preservation/` | Output and top-k preservation checks. |
| `experiments/controls/topk_robustness/` | Robustness to top-k/top-m choices. |
| `experiments/controls/fisher_output_energy_control/` | Equal Fisher-output movement controls. |
| `experiments/controls/statistical_diagnostics/` | Same scalar uncertainty and gradient magnitude diagnostics. |
| `experiments/controls/scale_external_robustness/` | Model and family coverage matrix. |
| `experiments/04_uncertainty_steering/01_minimal_intervention_energy/` | Minimal-energy support with gradient caveats. |
| `experiments/04_uncertainty_steering/decoder_main_battery/` | Decoder-only replication battery. |
| `experiments/05_controllability_mapping/` | Diagnostic controllability mapping plus the paper-facing budgeted allocation policy. |

## Exploratory or boundary artifacts

These are not primary paper claims. The complete versions are kept on branch `experiments-full` for transparency rather than being used as the main paper spine.

| Artifact on `experiments-full` | Recommended framing |
|---|---|
| `experiments/controls/rho_guided_selective_reliability/` | Auxiliary reliability feature, not a primary claim. |
| `experiments/06_non_nlp_safety_policies/` | Boundary/stress test, not a primary claim. |
| `experiments/controls/llm_equal_cost_review_policy/` | Equal-cost review-policy stress test, separate from controllability allocation. |

See `experiments_exploratory/README.md` on this branch for the list of artifacts intentionally excluded from the paper spine.

## Branch policy

- `main`: paper-facing spine plus necessary appendix controls.
- `experiments-full`: complete snapshot of all experiments and side explorations.
- `paper-main-clean`: proposed cleaned paper-facing branch before merging into `main`.
