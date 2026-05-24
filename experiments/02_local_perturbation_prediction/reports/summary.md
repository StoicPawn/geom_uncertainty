# Summary: Accessibility Predicts Local Uncertainty Movement

## Claim

Higher accessibility predicts larger local movement in entropy and varentropy under controlled perturbations.

## Main Result

At each tested epsilon, the highest rho quartile moves entropy and varentropy much more than the lowest quartile.

```text
epsilon=0.25: q1 Delta H=0.0009, q4 Delta H=0.0343; q1 |Delta Var|=0.0038, q4 |Delta Var|=0.0509
epsilon=0.50: q1 Delta H=0.0007, q4 Delta H=0.0684; q1 |Delta Var|=0.0080, q4 |Delta Var|=0.1036
epsilon=1.00: q1 Delta H=-0.0031, q4 Delta H=0.1351; q1 |Delta Var|=0.0216, q4 |Delta Var|=0.2127
```

## Claim Support

| Claim component | File |
|---|---|
| Quartile trend | `outputs/intervention_by_rho_quartile.csv` |
| Aggregate movement | `outputs/intervention_summary.csv` |
| Residual checks | `outputs/intervention_residual_correlations.csv` |
| Per-example audit trail | `outputs/intervention_records.csv` |
| Paper figure | `figures/fig03_accessibility_predicts_movement.svg` |
| Reproduction config | `config/reproduce.json` |
