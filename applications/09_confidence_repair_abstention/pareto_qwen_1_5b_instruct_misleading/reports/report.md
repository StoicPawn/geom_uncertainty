# Confidence Repair And Selective Abstention

Objective test of confidence repair on overconfident-wrong cases and selective abstention policies.

## Setup

```json
{
  "status": "completed",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "device": "cuda:0",
  "torch_dtype": "float16",
  "seed": 20260605,
  "n_items": 216,
  "n_records": 432,
  "answer_token_ids": {
    "A": 362,
    "B": 425,
    "C": 356,
    "D": 422
  },
  "layers": "auto",
  "pca_dim": 1,
  "eps": "0.025,0.05,0.1,0.2,0.4,0.8",
  "high_confidence_threshold": 0.7,
  "low_confidence_threshold": 0.45,
  "max_items": 216,
  "bootstrap": 150,
  "prompt_mode": "misleading"
}
```

## Group Counts

```text
outcome_group   n  confidence_mean  accuracy
  correct_low  62             0.25       1.0
    wrong_low 154             0.25       0.0
```

## Calibration Metrics

```text
                      condition  accuracy  mean_confidence  overconfident_wrong_rate      nll  brier      ece
                         before  0.287037             0.25                       0.0 1.386294   0.75 0.037037
accessible_rho_after_eps_median  0.287037             0.25                       0.0 1.386294   0.75 0.037037
```

## Repair Summary

```text
outcome_group                      method   eps   n  answer_preserved_rate  correct_after_rate  confidence_delta  wrong_confidence_reduced_rate  correct_confidence_increased_rate  overcorrection_rate  semantic_drift_proxy_mean  gold_probability_delta
  correct_low              accessible_rho 0.025 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low              accessible_rho 0.050 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low              accessible_rho 0.100 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low              accessible_rho 0.200 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low              accessible_rho 0.400 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low              accessible_rho 0.800 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low                     low_rho 0.025 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low                     low_rho 0.050 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low                     low_rho 0.100 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low                     low_rho 0.200 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low                     low_rho 0.400 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low                     low_rho 0.800 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          projected_gradient 0.025 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          projected_gradient 0.050 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          projected_gradient 0.100 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          projected_gradient 0.200 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          projected_gradient 0.400 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          projected_gradient 0.800 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low           random_activation 0.025 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low           random_activation 0.050 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low           random_activation 0.100 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low           random_activation 0.200 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low           random_activation 0.400 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low           random_activation 0.800 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          standard_delta_pca 0.025 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          standard_delta_pca 0.050 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          standard_delta_pca 0.100 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          standard_delta_pca 0.200 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          standard_delta_pca 0.400 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low          standard_delta_pca 0.800 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low            temperature_1.25 0.000 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low             temperature_1.5 0.000 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low               temperature_2 0.000 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low wrong_logit_gradient_normed 0.025 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low wrong_logit_gradient_normed 0.050 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low wrong_logit_gradient_normed 0.100 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low wrong_logit_gradient_normed 0.200 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low wrong_logit_gradient_normed 0.400 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
  correct_low wrong_logit_gradient_normed 0.800 124                    1.0                 1.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low              accessible_rho 0.025 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low              accessible_rho 0.050 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low              accessible_rho 0.100 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low              accessible_rho 0.200 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low              accessible_rho 0.400 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low              accessible_rho 0.800 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low                     low_rho 0.025 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low                     low_rho 0.050 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low                     low_rho 0.100 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low                     low_rho 0.200 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low                     low_rho 0.400 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low                     low_rho 0.800 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low          projected_gradient 0.025 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low          projected_gradient 0.050 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low          projected_gradient 0.100 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low          projected_gradient 0.200 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low          projected_gradient 0.400 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low          projected_gradient 0.800 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low           random_activation 0.025 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low           random_activation 0.050 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
    wrong_low           random_activation 0.100 308                    1.0                 0.0               0.0                            0.0                                0.0                  0.0                        0.0                     0.0
```

## Selective Policy Benchmark

```text
               policy        metric  value   n
           confidence auroc_correct    NaN 216
           confidence auprc_correct    NaN 216
     negative_entropy auroc_correct    NaN 216
     negative_entropy auprc_correct    NaN 216
               margin auroc_correct    NaN 216
               margin auprc_correct    NaN 216
     self_consistency auroc_correct    NaN 216
     self_consistency auprc_correct    NaN 216
verbalized_confidence auroc_correct    NaN 216
verbalized_confidence auprc_correct    NaN 216
                  rho auroc_correct    NaN 216
                  rho auprc_correct    NaN 216
    accessible_policy auroc_correct    NaN 216
    accessible_policy auprc_correct    NaN 216
```

## Selective Risk Curves

```text
               policy  target_coverage  coverage  selective_risk  answered
           confidence             0.25       1.0        0.712963       216
           confidence             0.50       1.0        0.712963       216
           confidence             0.75       1.0        0.712963       216
           confidence             1.00       1.0        0.712963       216
     negative_entropy             0.25       1.0        0.712963       216
     negative_entropy             0.50       1.0        0.712963       216
     negative_entropy             0.75       1.0        0.712963       216
     negative_entropy             1.00       1.0        0.712963       216
               margin             0.25       1.0        0.712963       216
               margin             0.50       1.0        0.712963       216
               margin             0.75       1.0        0.712963       216
               margin             1.00       1.0        0.712963       216
     self_consistency             0.25       1.0        0.712963       216
     self_consistency             0.50       1.0        0.712963       216
     self_consistency             0.75       1.0        0.712963       216
     self_consistency             1.00       1.0        0.712963       216
verbalized_confidence             0.25       1.0        0.712963       216
verbalized_confidence             0.50       1.0        0.712963       216
verbalized_confidence             0.75       1.0        0.712963       216
verbalized_confidence             1.00       1.0        0.712963       216
                  rho             0.25       1.0        0.712963       216
                  rho             0.50       1.0        0.712963       216
                  rho             0.75       1.0        0.712963       216
                  rho             1.00       1.0        0.712963       216
    accessible_policy             0.25       1.0        0.712963       216
    accessible_policy             0.50       1.0        0.712963       216
    accessible_policy             0.75       1.0        0.712963       216
    accessible_policy             1.00       1.0        0.712963       216
```

## Route Score Summary

```text
    route  layer   n  rho_mean  rho_median
delta_pca     14 216       0.0         0.0
delta_pca     27 216       0.0         0.0
   random     14 216       0.0         0.0
   random     27 216       0.0         0.0
state_pca     14 216       0.0         0.0
state_pca     27 216       0.0         0.0
```

## Bootstrap Confidence Intervals

```text
               metric       group                      method   eps  coverage   ci_low   median  ci_high
             accuracy        base                        none   NaN       NaN 0.234838 0.287037 0.342593
answer_preserved_rate correct_low              accessible_rho 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low              accessible_rho 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low              accessible_rho 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low              accessible_rho 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low              accessible_rho 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low              accessible_rho 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low                     low_rho 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low                     low_rho 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low                     low_rho 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low                     low_rho 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low                     low_rho 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low                     low_rho 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          projected_gradient 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          projected_gradient 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          projected_gradient 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          projected_gradient 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          projected_gradient 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          projected_gradient 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low           random_activation 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low           random_activation 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low           random_activation 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low           random_activation 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low           random_activation 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low           random_activation 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          standard_delta_pca 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          standard_delta_pca 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          standard_delta_pca 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          standard_delta_pca 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          standard_delta_pca 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low          standard_delta_pca 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low            temperature_1.25 0.000       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low             temperature_1.5 0.000       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low               temperature_2 0.000       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low wrong_logit_gradient_normed 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low wrong_logit_gradient_normed 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low wrong_logit_gradient_normed 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low wrong_logit_gradient_normed 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low wrong_logit_gradient_normed 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_low wrong_logit_gradient_normed 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low              accessible_rho 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low              accessible_rho 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low              accessible_rho 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low              accessible_rho 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low              accessible_rho 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low              accessible_rho 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low                     low_rho 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low                     low_rho 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low                     low_rho 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low                     low_rho 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low                     low_rho 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low                     low_rho 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low          projected_gradient 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low          projected_gradient 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low          projected_gradient 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low          projected_gradient 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low          projected_gradient 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low          projected_gradient 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low           random_activation 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_low           random_activation 0.050       NaN 1.000000 1.000000 1.000000
```

## Wrong-High Pareto

```text
(empty)
```

## Wrong-High Pareto Bootstrap CI

```text
(empty)
```

## Interpretation Guardrails

- The intervention is local logit-lens hidden-state repair, not a full rollout benchmark.
- Groups use absolute global confidence thresholds, not within-error medians.
- The dataset is synthetic QA/reasoning; treat results as a scaling pilot, not deployment evidence.
- A positive result requires improved selective risk/calibration without destroying correct answers; otherwise it is a failure mode.
