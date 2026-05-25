# Control: Statistical Diagnostics

## Purpose

Add uncertainty intervals, effect sizes, prompt-audit checks, and failure-mode tables on top of the existing curated outputs.

## Confirmatory Analyses

- bootstrap CIs for accessible-vs-random and accessible-vs-gradient-orthogonal steering contrasts;
- bootstrap CIs for `rho` quartile movement contrasts;
- bootstrap CIs for top-k robustness;
- same scalar uncertainty plus same projected-gradient controls but different `rho`;
- answer-preserving qualitative steering examples.

## Exploratory / Diagnostic Analyses

- residual `rho` effects after scalar, gradient, and geometric controls;
- exact, normalized, and near-duplicate prompt audit;
- failure-mode table.

## Outputs

- `outputs/bootstrap_steering_contrasts.csv`
- `outputs/bootstrap_rho_quartiles.csv`
- `outputs/bootstrap_topk_robustness.csv`
- `outputs/bootstrap_residual_effects.csv`
- `outputs/same_scalar_gradient_different_rho_pairs.csv`
- `outputs/prompt_duplicate_summary.csv`
- `outputs/prompt_near_duplicates.csv`
- `outputs/failure_modes.csv`

## Reproduce

Use `config/reproduce.json`.
