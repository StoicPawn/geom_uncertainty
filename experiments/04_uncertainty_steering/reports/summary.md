# Summary: Uncertainty Steering

## Claim

Accessible directions steer uncertainty more efficiently than matched controls and mostly preserve monitored answer content at small perturbation sizes.

## Main Result

```text
Directional control:
decrease directional_rate = 0.9857
increase directional_rate = 0.9581

Equal Fisher-output-energy vs gradient-orthogonal control:
|Delta H| ratio = 5.88x to 32.40x
|Delta Var| ratio = 3.40x to 5.23x

Equal Fisher-output-energy vs random control:
|Delta H| ratio = 1.74x to 1.83x
|Delta Var| ratio = 1.60x to 1.67x

Rho dependency after controls:
partial corr rho -> |Delta H|   = 0.3837
partial corr rho -> |Delta Var| = 0.3245
```

Specificity:

- At `epsilon=0.02`, mean top-1 change rate is `0.0142`; target-correctness change rate is `0.0003`.
- At `epsilon=0.05`, mean top-1 change rate is `0.0345`; target-correctness change rate is `0.0010`.
- Qualitative answer-preserving examples are in `reports/answer_preserving_qualitative_examples.md`.

## Claim Support

| Claim component | File |
|---|---|
| Directionality and monotonicity | `outputs/directionality_summary.csv` |
| Equal Fisher-output-energy comparison | `outputs/equal_output_energy_contrasts.csv` |
| Specificity / preservation | `outputs/specificity_summary.csv` |
| Rho dependency after controls | `outputs/rho_dependency.csv` |
| Replication matrix | `outputs/replication_matrix.csv` |
| Per-example audit trail | `outputs/steering_records.csv` |
| Answer-preserving qualitative cases | `outputs/answer_preserving_qualitative_examples.csv`; `reports/answer_preserving_qualitative_examples.md` |
| Main paper figure | `figures/fig06_steering_vs_controls_ci.svg` |
| Supplementary steering figure | `figures/fig06_uncertainty_steering_main.svg` |
| Reproduction config | `config/reproduce.json` |
