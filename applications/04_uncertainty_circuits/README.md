# Uncertainty Circuits

Application-level route-interpretability test kept separate from the four main experiments.

## Question

Are high-accessibility routes causal control points for uncertainty, or is `rho` only a descriptive local geometry?

## Result

High-rho routes produce more uncertainty movement than low-rho and random routes under the same Fisher-output movement budget, and substantially more than gradient-selected routes in this battery. Top-1 changes and candidate-set KL do not increase with `rho`, so the effect is not just generic answer disruption.

## Key Artifacts

- `outputs/route_scores.csv`: route-level accessibility scores.
- `outputs/circuit_intervention_records.csv`: Fisher-normalized interventions.
- `outputs/circuit_route_contrasts.csv`: high-rho route contrasts.
- `outputs/rho_causal_effect_correlations.csv`: rho-to-causal-effect correlations.
- `reports/report.md`: concise result table.

## Reproduce

```powershell
python scripts\run_brittle_confidence_and_circuit_applications.py --models distilbert-base-uncased,bert-base-uncased,google/bert_uncased_L-2_H-128_A-2 --top-k 32 --subspace-k 8 --max-prompts-per-task 6 --output-eps 0.05 --seed 20260530
```
