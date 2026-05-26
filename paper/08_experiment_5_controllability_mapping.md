# Experiment 5: Uncertainty Controllability Mapping

The final diagnostic experiment asks whether `rho` helps map where uncertainty is controllable across model, layer, route family, subspace dimension, task, and intervention budget.

Unlike the exploratory confidence-repair tests, this experiment does not use correctness as the primary target. It uses uncertainty-control targets:

- maximum local entropy movement;
- safe movement: entropy movement above threshold while preserving top-1, keeping high top-k Jaccard, and staying under a drift cap;
- minimal applied Fisher-output energy to reach the entropy movement threshold.

The predictive comparison is:

```text
controls
controls + rho
```

Controls include entropy, confidence, margin, varentropy, projected-gradient norm, Fisher-output energy, Jacobian norm, layer, model, task, route family, subspace dimension, and budget mode.

## Result

Adding `rho` improves the diagnostic policy:

```text
safe_movement AUPRC:       0.674 -> 0.764
safe_movement AUROC:       0.888 -> 0.919
safe_movement log-loss:    0.433 -> 0.372
minimal_energy MAE log10:  0.391 -> 0.346
```

Bootstrap deltas are positive for safe movement AUPRC, log-loss improvement, maximum movement MAE, and minimal-energy MAE.

This supports the defensible application-level claim: `rho` is useful for mapping where uncertainty is locally controllable. It should not be framed as a generic error predictor or as the strongest raw intervention direction.
