# Experiment 2: Local Perturbation Prediction

## Purpose

Test whether accessibility predicts local entropy and varentropy movement under controlled perturbations. This is the bridge between the geometric decomposition and observable local changes in uncertainty.

## Data

- `data/prompts.csv`: prompt set used for the local perturbation run.

## Outputs

- `outputs/intervention_summary.csv`: aggregate perturbation effects.
- `outputs/intervention_by_rho_quartile.csv`: effect sizes grouped by accessibility quartile.
- `outputs/intervention_residual_correlations.csv`: residual correlations after scalar controls.
- `outputs/intervention_records.csv`: per-example intervention records.

## Figures

- `figures/fig03_accessibility_predicts_movement.svg`
- `figures/fig03_accessibility_predicts_movement.png`
- `figures/fig04_rho_vs_delta_controls.svg`
- `figures/fig04_rho_vs_delta_controls.png`

## Reports

- `reports/summary.md`
- `reports/report.md`

## Claim Support

This experiment supports the claim that accessibility predicts local uncertainty movement. The mapping from claim to CSV and figure is also listed in `../../CLAIM_MAP.md`.

## Related Scripts

- `scripts/run_distilbert_geometry_interventions.py`

## Reproduce

Use `config/reproduce.json` for exact commands and seeds.
