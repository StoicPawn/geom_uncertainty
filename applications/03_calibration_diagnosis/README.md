# Application 3: Calibration Diagnosis

Separate application for asking whether miscalibration is internally steerable through high-accessibility routes.

## Question

Classical calibration asks whether probabilities match empirical correctness. This test asks whether the calibration error can be moved through an internal route with low accuracy loss.

## Main Metrics

- ECE, NLL, Brier score, accuracy.
- Full-vocabulary KL drift and top-k preservation.
- Calibration steerability score: ECE improvement minus accuracy-loss rate.

## Status

Completed on the three local masked-LM models.

## Key Result

High-rho routes improve target probability, NLL, Brier score, and exact-target accuracy much more than low-rho routes at the same Fisher-output movement budget. ECE is mixed: the clean model is already low-confidence and low-accuracy on exact target tokens, so increasing target probability can worsen aggregate ECE while improving NLL/Brier and accuracy.

This should be framed as an internal calibration-steerability diagnostic, not as an ECE-only improvement claim.

## Reproduce

```powershell
python scripts\run_calibration_diagnosis.py --max-prompts-per-task 10 --top-k 32 --subspace-ks 8 --output-eps 0.05 --seed 20260532
```
