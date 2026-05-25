# Brittle Confidence

Application-level matched perturbation test kept separate from the four main experiments.

## Question

Among similarly high-confidence predictions, are low-accessibility cases more fragile under semantically innocuous prompt changes?

## Result

The current checked-in battery is mixed/negative. Low-rho cases show slightly more KL drift and top-10 neighborhood drop, but high-rho cases show more answer flips, probability drop, entropy increase, and overall fragility. This should be treated as a boundary-case diagnostic, not as a positive claim.

## Key Artifacts

- `outputs/matched_high_confidence_groups.csv`: matched high-rho/low-rho groups.
- `outputs/brittle_perturbation_records.csv`: all prompt perturbation records.
- `outputs/brittle_summary.csv`: perturbation-type summaries.
- `outputs/brittle_matched_contrasts.csv`: low-rho minus high-rho contrasts.
- `reports/report.md`: concise result table.

## Reproduce

```powershell
python scripts\run_brittle_confidence_and_circuit_applications.py --models distilbert-base-uncased,bert-base-uncased,google/bert_uncased_L-2_H-128_A-2 --top-k 32 --subspace-k 8 --max-prompts-per-task 6 --output-eps 0.05 --seed 20260530
```
