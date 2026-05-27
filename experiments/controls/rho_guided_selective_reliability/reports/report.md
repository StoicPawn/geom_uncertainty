# Rho-Guided Selective Reliability

This disruptive test asks whether `rho` improves a non-oracle selective reliability policy: accept when predicted reliable, otherwise abstain/route or intervene/retrieve according to uncertainty and rho-actionability.

Correctness is never used as a decision feature. It is used only for held-out evaluation and cross-validated training labels. The split is grouped by prompt identity/text so the policy is evaluated out of sample on prompt groups, not memorized rows.

## Verdict
Supported on AURC, log-loss

## Dataset Coverage
```text
             source                             model  n  correct_rate
    decoder_battery                 Qwen/Qwen2.5-0.5B 12      0.166667
    decoder_battery                   microsoft/phi-2 12      0.333333
masked_full_battery                 bert-base-uncased 21      0.523810
masked_full_battery           distilbert-base-uncased 21      0.476190
masked_full_battery google/bert_uncased_L-2_H-128_A-2 21      0.095238
        masked_topk                 bert-base-uncased 15      0.533333
        masked_topk           distilbert-base-uncased 15      0.400000
        masked_topk google/bert_uncased_L-2_H-128_A-2 15      0.000000
```

## Metrics
```text
                policy   n  accuracy    brier  log_loss    auroc  average_precision  aurc_mean_risk  risk_at_50pct_coverage  risk_at_80pct_coverage  high_pred_reliable_error_rate  high_pred_reliable_coverage  top20pct_error_rate
         pred_baseline 132  0.325758 0.153992  0.511038 0.863601           0.758355        0.469461                0.439394                 0.59434                       0.200000                     0.265152             0.148148
pred_baseline_plus_rho 132  0.325758 0.118298  0.384790 0.905932           0.828244        0.448106                0.393939                 0.59434                       0.129032                     0.234848             0.148148
```

## Baseline Vs Baseline+rho Deltas
```text
                           metric    value
          delta_brier_improvement 0.035694
       delta_log_loss_improvement 0.126248
           delta_aurc_improvement 0.021355
         delta_risk50_improvement 0.045455
         delta_risk80_improvement 0.000000
delta_high_conf_error_improvement 0.070968
```

## Bootstrap CI
```text
                           metric    ci_low   median  ci_high
           delta_aurc_improvement  0.007246 0.023724 0.043499
          delta_brier_improvement  0.020456 0.035250 0.052599
delta_high_conf_error_improvement -0.000025 0.067024 0.175926
       delta_log_loss_improvement  0.073595 0.124629 0.186695
         delta_risk50_improvement -0.015157 0.044118 0.105263
         delta_risk80_improvement  0.000000 0.000000 0.000000
```

## Risk-Coverage
```text
 requested_coverage  pred_baseline  pred_baseline_plus_rho  risk_improvement
                0.2       0.148148                0.148148          0.000000
                0.3       0.275000                0.200000          0.075000
                0.4       0.396226                0.358491          0.037736
                0.5       0.439394                0.393939          0.045455
                0.6       0.500000                0.487500          0.012500
                0.7       0.559140                0.537634          0.021505
                0.8       0.594340                0.594340          0.000000
                0.9       0.638655                0.638655          0.000000
                1.0       0.674242                0.674242          0.000000
```

## Rho-Guided Action Mix
```text
               action  n  coverage  error_rate  rho_action_mean  predicted_correct_mean
     abstain_or_route 32  0.242424    1.000000         0.756102                0.015598
               accept 40  0.303030    0.200000         0.919660                0.894401
intervene_or_retrieve 26  0.196970    0.846154         0.984900                0.209333
  pass_to_other_model 34  0.257576    0.794118         0.851710                0.225204
```

## Files
```text
selective_reliability_dataset.csv
selective_reliability_predictions.csv
selective_reliability_metrics.csv
selective_reliability_deltas.csv
selective_reliability_bootstrap_ci.csv
risk_coverage_curve.csv
rho_guided_action_mix.csv
report.md
```
