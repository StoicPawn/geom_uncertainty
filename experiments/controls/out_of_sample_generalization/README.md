# Control: Out-of-Sample Route Generalization

## Purpose

Test whether a route/subspace estimated on train prompts can be reused on held-out prompts.

## Protocol

- Build `state_pca` and `delta_pca` subspaces from train prompts.
- Evaluate accessibility and accessible steering on held-out prompts.
- Compare train-derived routes to held-out oracle routes and pooled random routes.

## Main Reading

Train-derived routes generalize approximately to held-out oracle routes for BERT and DistilBERT in this small run. Pooled random routes remain competitive, especially for delta routes, so this is evidence for route reusability but not yet a decisive global steering theorem.

## Outputs

- `outputs/oos_scores.csv`
- `outputs/oos_steering_records.csv`
- `outputs/oos_score_summary.csv`
- `outputs/oos_steering_summary.csv`
- `outputs/oos_contrasts.csv`

## Reproduce

Use `config/reproduce.json`.
