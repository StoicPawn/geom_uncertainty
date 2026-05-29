# Exploratory and Boundary Experiments

This directory intentionally keeps the paper-facing branch from advertising boundary experiments as primary claims.

The full artifacts remain available on branch `experiments-full`.

## Boundary artifacts kept on `experiments-full`

| Artifact | Status for paper |
|---|---|
| `experiments/controls/rho_guided_selective_reliability/` | Auxiliary reliability feature only. Not a main claim. |
| `experiments/06_non_nlp_safety_policies/` | Boundary/stress test. Not a main claim. |
| `experiments/controls/llm_equal_cost_review_policy/` | Equal-cost review-policy stress test. Separate from route-level controllability allocation. |

## Paper-facing rule

The main paper should focus on:

1. non-redundancy of rho;
2. local perturbation validity;
3. equal-output/choose-route intervention validity;
4. budgeted controllability-aware allocation.

Review/reliability/safety-policy experiments should be discussed only as limitations or future work unless explicitly moved back into the main claim map.
