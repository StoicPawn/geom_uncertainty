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

## Reports

- `reports/report.md`

## Related Scripts

- `scripts/run_distilbert_geometry_interventions.py`
