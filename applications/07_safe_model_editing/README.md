# Safe Model Editing Diagnostic

## Claim

Accessible uncertainty may identify facts/beliefs that can be locally edited with smaller interventions and fewer side effects.

## Scope

The runner performs local representation edits in masked-LM logit-lens space. It is a safe model-editing diagnostic, not a persistent weight-editing method.

## Result Status

The first run is mixed/negative for the strong claim. `rho` is weakly positive for safe-success, but target-specific edit-gradient and Fisher-output norms are stronger predictors of edit cost. Keep this as a diagnostic unless a persistent weight-editing version or a larger edit suite reverses the result.

## Artifacts

- `outputs/edit_records.csv`
- `outputs/side_effect_records.csv`
- `outputs/edit_summary.csv`
- `outputs/rho_quartile_summary.csv`
- `outputs/predictor_benchmark.csv`
- `outputs/qualitative_safe_edit_examples.csv`
- `reports/report.md`

## Reproduce

```powershell
python scripts\run_safe_model_editing_application.py --models distilbert-base-uncased,google/bert_uncased_L-2_H-128_A-2 --max-prompts-per-task 8 --top-k 32 --subspace-k 8 --random-subspaces 1 --side-prompts-per-edit 12 --seed 20260601
```
