# Control: Full-Vocabulary Tiny-Model Sanity

## Purpose

Check whether top-k Fisher accessibility is directionally consistent with full-vocabulary Fisher accessibility on the local tiny masked-LM model.

## Method

The run uses `google/bert_uncased_L-2_H-128_A-2` and computes full-vocabulary accessibility without forming a full `F^{1/2}` matrix:

```text
rho = u^T F C (C^T F C)^+ C^T F u / (u^T F u), where C = J B
```

## Main Reading

Top-k accessibility overestimates absolute full-vocabulary `rho` at small k, but the gap decreases as k grows. This supports using top-k as a local output-lens approximation while keeping the full-vocabulary limitation explicit.

## Outputs

- `outputs/full_vocab_rho_scores.csv`
- `outputs/topk_vs_full_vocab_summary.csv`
- `outputs/prompt_tables.csv`

## Reproduce

Use `config/reproduce.json`.
