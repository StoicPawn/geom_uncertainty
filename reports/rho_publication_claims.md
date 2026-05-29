# Rho Publication Claim Status

This note states the paper-facing claim for the clean branch.

## Headline Claim

`rho(B)` is a local Fisher-output accessibility coefficient for uncertainty controllability. It measures whether a local uncertainty direction is reachable through a given internal route/subspace, rather than measuring scalar uncertainty itself.

## Main Evidence

1. Scalar-matched cases can have different rho values.
2. Higher-rho routes show larger local uncertainty movement under controlled perturbations.
3. High-rho routes move uncertainty more than low-rho and random routes at matched Fisher-output movement.
4. Under fixed-budget route allocation, adding rho improves SafeMove at 20-30% budgets; the cleanest 20% deltas are:

```text
controls+rho - controls-only:        +0.014195, CI [0.005509, 0.026392]
gradient+rho - gradient-only:        +0.027604, CI [0.011319, 0.055714]
rho-only - within-example shuffled:  +0.034033, CI [0.020348, 0.051609]
```

## Wording To Use

> We introduce rho, a Fisher-output accessibility coefficient that measures how much of a local uncertainty direction is reachable through a given internal route. Rho is non-redundant with scalar uncertainty, predicts local uncertainty movement, supports route-level intervention selection, and improves fixed-budget controllability-aware allocation.

## Claims To Avoid

- Do not claim rho is a generic error predictor.
- Do not claim rho uniformly beats direct gradients.
- Do not claim rho uniformly improves human-review triage at equal review cost.
- Do not claim broad model-family replication unless checked-in outputs support it.
