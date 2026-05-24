# Experiment 1: Matched Scalar Uncertainty

## Purpose

Show that scalar uncertainty does not determine accessibility. The matched-pair outputs compare examples with similar entropy or varentropy but substantially different `rho` or accessible varentropy.

## Data

- `data/source_prompts.csv`: prompt table used by the scalar matching analyses.

## Outputs

- `outputs/same_uncertainty_different_rho_pairs.csv`: examples matched on scalar uncertainty with different accessibility ratios.
- `outputs/same_uncertainty_different_accessibility_pairs.csv`: examples matched on scalar uncertainty with different accessible components.
- `outputs/matched_error_correct_summary.csv`: summary for matched correct/error cases.

## Reports

- `reports/geometric_diagnostics_report.md`
- `reports/compressibility_report.md`

## Related Scripts

- `scripts/analyze_geometric_diagnostics_from_outputs.py`
- `scripts/analyze_layerwise_k_compressibility.py`
