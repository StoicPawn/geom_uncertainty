# Control: Semantic And Top-k Preservation

## Purpose

Extend preservation checks beyond top-1.

## Metrics

- selected top-1 changed rate;
- selected top-5 Jaccard;
- full-vocabulary top-5 Jaccard;
- full-vocabulary top-10 Jaccard;
- KL divergence on the original candidate set;
- L1 probability change on selected candidates;
- full-vocabulary mass retained on the original candidate set;
- semantic-cluster L1 shift;
- embedding-centroid shift.

## Outputs

- `outputs/semantic_topk_preservation.csv`: per-intervention preservation metrics.
- `outputs/semantic_topk_preservation_summary.csv`: summary by top-k, direction, and sign.

## Reproduce

Use `config/reproduce.json`.
