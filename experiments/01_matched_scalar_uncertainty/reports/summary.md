# Summary: Scalar Uncertainty Does Not Determine Accessibility

## Claim

Entropy and varentropy are not sufficient to determine accessibility. Two examples can have similar scalar uncertainty while differing sharply in `rho` or adjusted accessible varentropy.

## Main Result

- 300 scalar-matched rho pairs are retained in `outputs/same_uncertainty_different_rho_pairs.csv`.
- Maximum `rho_abs_diff`: `0.9771`.
- Median within-pair entropy difference: `0.0337`.
- Median within-pair varentropy difference: `0.0852`.
- Adjusted-accessibility pairs reach `rho_adjusted_abs_diff = 1.0507`.

## Claim Support

| Claim component | File |
|---|---|
| Similar scalar uncertainty, different rho | `outputs/same_uncertainty_different_rho_pairs.csv` |
| Similar scalar uncertainty, different adjusted accessibility | `outputs/same_uncertainty_different_accessibility_pairs.csv` |
| Scalar scatter figure | `figures/fig02_scalar_uncertainty_vs_rho.svg` |
| Matched-pair figure | `figures/fig03_scalar_matched_pairs.svg` |
| Reproduction config | `config/reproduce.json` |
