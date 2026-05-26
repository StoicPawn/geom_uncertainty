# Experiment 5: Uncertainty Controllability Mapping

This diagnostic experiment asks whether rho maps where uncertainty is locally controllable across model, layer, task, route family, and subspace dimension.

The target is not correctness. The targets are:

- `max_uncertainty_movement`: maximum local entropy movement over the intervention sweep.
- `safe_movement_target`: entropy movement above threshold with preserved top-1, high top-k Jaccard, and bounded output drift.
- `minimal_energy`: minimum applied Fisher-output energy needed to reach the entropy movement threshold.

The comparison is deliberately conservative:

```text
controls
controls + rho
```

Controls include entropy, confidence, margin, varentropy, projected-gradient norm, Fisher-output energy, Jacobian norm, layer, model, task, route family, subspace dimension, and budget mode.

## Result

```text
safe_movement AUPRC
controls:      0.673757
controls+rho:  0.763993

safe_movement log-loss
controls:      0.433265
controls+rho:  0.372353

minimal_energy MAE log10
controls:      0.391496
controls+rho:  0.346423
```

Bootstrap deltas are positive for safe movement AUPRC, log-loss improvement, maximum movement MAE, and minimal-energy MAE.

## Artifacts

- `outputs/controllability_targets.csv`
- `outputs/intervention_points.csv`
- `outputs/mapping_metrics.csv`
- `outputs/mapping_bootstrap_ci.csv`
- `reports/uncertainty_controllability_mapping_test.md`
- `config/reproduce.json`

## Caveat

Masked-LM models requested for this test were not available in the local Hugging Face cache and were skipped without network download. The completed run covers local decoder models: distilgpt2, Qwen 0.5B, Qwen 1.5B Instruct, and Phi-2.
