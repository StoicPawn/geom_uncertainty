# Control: Full-Vocabulary Tiny-Model Sanity Check

This sanity check computes full-vocabulary Fisher-kernel accessibility on `google/bert_uncased_L-2_H-128_A-2` and compares top-k rho against full-vocabulary rho.

The implementation avoids forming a full `F^{1/2}` matrix. It uses the identity:

```text
rho = u^T F C (C^T F C)^+ C^T F u / (u^T F u), where C = J B
```

## Summary

```text
 top_k_output  n  rho_topk_mean  rho_full_vocab_mean  spearman_rho_topk_vs_full  rho_abs_diff_mean  rho_abs_diff_median  rho_abs_diff_max
           16 55       0.623132             0.243873                   0.352309           0.379258             0.385095          0.735211
           32 55       0.403209             0.243873                   0.506566           0.180706             0.179588          0.363303
           64 55       0.329744             0.243873                   0.479221           0.117410             0.124000          0.253465
          128 55       0.283869             0.243873                   0.497691           0.091462             0.080518          0.212883
          256 55       0.263895             0.243873                   0.495166           0.082927             0.085282          0.210231
```

## Files

```text
full_vocab_rho_scores.csv
topk_vs_full_vocab_summary.csv
prompt_tables.csv
report.md
```