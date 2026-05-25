# Control: Full Regression

## Purpose

Test whether `rho` remains predictive after controlling for scalar uncertainty, confidence, margin, Jacobian/Fisher norms, gradient baselines, layer, model, and prompt category.

## Controls

`entropy`, `varentropy`, `confidence`, `margin`, `jacobian_fro_norm`, `fisher_output_energy`, `grad_entropy_proj_norm`, `grad_varentropy_proj_norm`, `top_k_output`, `layer`, `model`, `task`, `topic`, `subspace_family`, `subspace_k`, `direction`, `sign`.

## Outputs

- `outputs/full_regression_ols.csv`: OLS coefficients, t-statistics, p-values, and R2 for rho after controls.

## Reproduce

Use `config/reproduce.json`.
