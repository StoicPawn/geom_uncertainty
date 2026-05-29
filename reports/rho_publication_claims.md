# Rho Publication Claim Status

This note separates the publishable claims from boundary conditions.

## Supported Headline

`rho(B)` is a local Fisher-output accessibility coefficient for uncertainty controllability. It is not just scalar uncertainty, confidence, margin, or gradient magnitude rewritten.

The strongest evidence is the combination of:

- scalar-matched examples with large rho differences;
- rho-quartile movement under local perturbations;
- equal-Fisher-output steering where accessible directions move uncertainty more than random and orthogonal controls;
- controllability mapping where controls+rho improves safe-movement and minimal-energy prediction;
- route-selection tests where high-rho beats low-rho and random routes;
- budgeted allocation tests where adding rho improves fixed-budget safe route selection at SafeMove@20%.

## Strongest Results

```text
controllability safe_movement AUPRC: controls 0.673757 -> controls+rho 0.763993
controllability safe_movement log-loss: controls 0.433265 -> controls+rho 0.372353
controllability minimal-energy MAE log10: controls 0.391496 -> controls+rho 0.346423
route selection safe_movement: high-rho 0.095238 vs random 0.047619 vs low-rho 0.004762
budgeted allocation SafeMove@20% deltas:
  controls+rho - controls-only: +0.014195, CI [0.005509, 0.026392]
  gradient+rho - gradient-only: +0.027604, CI [0.011319, 0.055714]
  rho-only - within-example shuffled rho: +0.034033, CI [0.020348, 0.051609]
choose-route abs Delta entropy: high-rho 0.048148 vs random 0.037249 vs low-rho 0.027510
```

## Decision-Policy Status

Rho is useful for decision policies when the decision is phrased as controllability:

- "Which route should I intervene through?"
- "Where is uncertainty locally movable?"
- "Should this high-uncertainty case be reviewed or attempted with a route/refinement?"
- "Given a fixed budget, which candidate routes should receive the intervention?"

Rho is not yet established as a universal accept/review score. Equal-cost LLM review selectors show useful rho combinations, especially under source shift, but Fisher-output-energy is stronger in grouped-prompt and leave-one-model-out panels. Non-NLP equal-cost review is a boundary case: rho does not beat entropy/gradient overall. The budgeted allocation test is stronger than the review tests because it selects routes under a matched intervention panel; its cleanest result is at 20-30% budgets, while SafeMove@10% is positive but has bootstrap intervals that can cross zero for some key comparisons.

## Claims To Avoid

- Do not claim rho is a generic error predictor.
- Do not claim rho uniformly beats direct gradients.
- Do not claim rho uniformly improves human-review triage at equal review cost.
- Do not claim broad RoBERTa/Llama/Mistral replication unless those checked-in outputs are present.

## Publication Wording

Best short version:

> We introduce rho, a Fisher-output accessibility coefficient that measures how much of a local uncertainty direction is reachable through a given internal route. Across matched uncertainty cases, local interventions, equal-output steering, controllability mapping, route-selection tests, and fixed-budget allocation, rho identifies where uncertainty is controllable beyond scalar uncertainty and standard geometric controls. Its decision-policy role is controllability-aware routing, not generic error prediction.

Boundary sentence:

> Direct gradient and Fisher-energy baselines remain strong, and equal-cost review policies expose boundary cases; rho should be used as a route-level controllability feature rather than sold as a standalone reliability score.
