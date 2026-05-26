# Confidence Repair And Selective Abstention

Objective test of confidence repair on overconfident-wrong cases and selective abstention policies.

## Setup

```json
{
  "status": "completed",
  "model": "Qwen/Qwen2.5-0.5B",
  "device": "cuda:0",
  "torch_dtype": "float16",
  "seed": 20260603,
  "n_items": 36,
  "n_records": 72,
  "answer_token_ids": {
    "A": 362,
    "B": 425,
    "C": 356,
    "D": 422
  },
  "layers": "auto",
  "pca_dim": 1,
  "eps": "0.05,0.1,0.2,0.4",
  "confidence_threshold": 0.5863053104674223
}
```

## Group Counts

```text
outcome_group  n  confidence_mean  accuracy
 correct_high 13         0.954474       1.0
  correct_low 12         0.538172       1.0
   wrong_high  6         0.382613       0.0
    wrong_low  5         0.328930       0.0
```

## Calibration Metrics

```text
                      condition  accuracy  mean_confidence  overconfident_wrong_rate      nll    brier      ece
                         before  0.694444         0.633515                       0.0 0.637060 0.336229 0.167401
accessible_rho_after_eps_median  0.722222         0.744922                       0.0 0.524014 0.293685 0.087696
```

## Repair Summary

```text
outcome_group             method  eps  n  answer_preserved_rate  correct_after_rate  confidence_delta  wrong_confidence_reduced_rate  correct_confidence_increased_rate  overcorrection_rate  gold_probability_delta
 correct_high     accessible_rho 0.05 26               1.000000            0.500000         -0.000859                       0.461538                           0.423077                  0.0                0.000322
 correct_high     accessible_rho 0.10 26               1.000000            0.500000         -0.001464                       0.461538                           0.500000                  0.0                0.000575
 correct_high     accessible_rho 0.20 26               1.000000            0.500000         -0.003071                       0.461538                           0.500000                  0.0                0.001242
 correct_high     accessible_rho 0.40 26               1.000000            0.500000         -0.006226                       0.500000                           0.500000                  0.0                0.002460
 correct_high            low_rho 0.05 26               1.000000            0.500000         -0.000052                       0.307692                           0.423077                  0.0                0.000035
 correct_high            low_rho 0.10 26               1.000000            0.500000         -0.000018                       0.192308                           0.384615                  0.0                0.000097
 correct_high            low_rho 0.20 26               1.000000            0.500000         -0.000154                       0.307692                           0.500000                  0.0                0.000240
 correct_high            low_rho 0.40 26               1.000000            0.500000         -0.000167                       0.384615                           0.500000                  0.0                0.000444
 correct_high projected_gradient 0.05 26               1.000000            0.500000         -0.030754                       0.500000                           0.500000                  0.0                0.007989
 correct_high projected_gradient 0.10 26               0.923077            0.538462         -0.059551                       0.500000                           0.500000                  0.0                0.015854
 correct_high projected_gradient 0.20 26               0.538462            0.615385         -0.060326                       0.500000                           0.500000                  0.0                0.030274
 correct_high projected_gradient 0.40 26               0.500000            0.615385         -0.006644                       0.500000                           0.500000                  0.0                0.050655
 correct_high  random_activation 0.05 26               1.000000            0.500000          0.000372                       0.192308                           0.038462                  0.0               -0.000122
 correct_high  random_activation 0.10 26               1.000000            0.500000          0.000685                       0.192308                           0.076923                  0.0               -0.000240
 correct_high  random_activation 0.20 26               1.000000            0.500000          0.001308                       0.192308                           0.269231                  0.0               -0.000342
 correct_high  random_activation 0.40 26               1.000000            0.500000          0.002687                       0.192308                           0.269231                  0.0               -0.000783
 correct_high standard_delta_pca 0.05 26               1.000000            0.500000         -0.000996                       0.500000                           0.423077                  0.0                0.000300
 correct_high standard_delta_pca 0.10 26               1.000000            0.500000         -0.001658                       0.500000                           0.269231                  0.0                0.000525
 correct_high standard_delta_pca 0.20 26               1.000000            0.500000         -0.003442                       0.500000                           0.423077                  0.0                0.001151
 correct_high standard_delta_pca 0.40 26               1.000000            0.500000         -0.006936                       0.500000                           0.423077                  0.0                0.002280
 correct_high   temperature_1.25 0.00 26               1.000000            0.500000         -0.030970                       0.500000                           0.000000                  0.0                0.001042
 correct_high    temperature_1.5 0.00 26               1.000000            0.500000         -0.059504                       0.500000                           0.000000                  0.0               -0.004134
 correct_high      temperature_2 0.00 26               1.000000            0.500000         -0.115160                       0.500000                           0.000000                  0.0               -0.027915
  correct_low     accessible_rho 0.05 24               1.000000            0.500000         -0.000320                       0.416667                           0.250000                  0.0                0.000287
  correct_low     accessible_rho 0.10 24               1.000000            0.500000         -0.000503                       0.500000                           0.291667                  0.0                0.000772
  correct_low     accessible_rho 0.20 24               1.000000            0.500000         -0.000930                       0.458333                           0.458333                  0.0                0.001868
  correct_low     accessible_rho 0.40 24               1.000000            0.500000         -0.001961                       0.458333                           0.500000                  0.0                0.003789
  correct_low            low_rho 0.05 24               1.000000            0.500000          0.000087                       0.208333                           0.250000                  0.0                0.000169
  correct_low            low_rho 0.10 24               0.958333            0.500000          0.000836                       0.291667                           0.416667                  0.0                0.001252
  correct_low            low_rho 0.20 24               0.958333            0.500000          0.001284                       0.416667                           0.416667                  0.0                0.001844
  correct_low            low_rho 0.40 24               0.958333            0.500000          0.003014                       0.458333                           0.458333                  0.0                0.003877
  correct_low projected_gradient 0.05 24               0.916667            0.500000         -0.016169                       0.500000                           0.500000                  0.0                0.016653
  correct_low projected_gradient 0.10 24               0.791667            0.541667         -0.024502                       0.500000                           0.500000                  0.0                0.033085
  correct_low projected_gradient 0.20 24               0.500000            0.708333          0.009573                       0.500000                           0.500000                  0.0                0.063487
  correct_low projected_gradient 0.40 24               0.500000            0.708333          0.079705                       0.500000                           0.500000                  0.0                0.110492
  correct_low  random_activation 0.05 24               1.000000            0.500000         -0.000523                       0.416667                           0.083333                  0.0               -0.000091
  correct_low  random_activation 0.10 24               1.000000            0.500000         -0.000742                       0.333333                           0.125000                  0.0               -0.000206
  correct_low  random_activation 0.20 24               1.000000            0.500000         -0.001202                       0.333333                           0.166667                  0.0               -0.000353
  correct_low  random_activation 0.40 24               1.000000            0.500000         -0.002036                       0.333333                           0.291667                  0.0               -0.000678
  correct_low standard_delta_pca 0.05 24               1.000000            0.500000         -0.001106                       0.458333                           0.166667                  0.0               -0.000138
  correct_low standard_delta_pca 0.10 24               0.958333            0.500000         -0.001799                       0.500000                           0.125000                  0.0                0.000354
  correct_low standard_delta_pca 0.20 24               0.958333            0.500000         -0.003197                       0.500000                           0.208333                  0.0                0.000825
  correct_low standard_delta_pca 0.40 24               0.958333            0.500000         -0.006305                       0.500000                           0.208333                  0.0                0.001730
  correct_low   temperature_1.25 0.00 24               1.000000            0.500000         -0.047086                       0.500000                           0.000000                  0.0               -0.022266
  correct_low    temperature_1.5 0.00 24               1.000000            0.500000         -0.085092                       0.500000                           0.000000                  0.0               -0.042704
  correct_low      temperature_2 0.00 24               1.000000            0.500000         -0.141859                       0.500000                           0.000000                  0.0               -0.076304
   wrong_high     accessible_rho 0.05 12               1.000000            0.083333         -0.000334                       0.583333                           0.083333                  0.0                0.000650
   wrong_high     accessible_rho 0.10 12               1.000000            0.083333         -0.001706                       0.666667                           0.083333                  0.0                0.001759
   wrong_high     accessible_rho 0.20 12               1.000000            0.083333         -0.003438                       0.916667                           0.083333                  0.0                0.003893
   wrong_high     accessible_rho 0.40 12               1.000000            0.083333         -0.007040                       0.916667                           0.083333                  0.0                0.007997
   wrong_high            low_rho 0.05 12               1.000000            0.083333         -0.000614                       0.666667                           0.083333                  0.0                0.000911
   wrong_high            low_rho 0.10 12               1.000000            0.083333         -0.001573                       0.750000                           0.083333                  0.0                0.001952
   wrong_high            low_rho 0.20 12               1.000000            0.083333         -0.002257                       0.750000                           0.083333                  0.0                0.004056
   wrong_high            low_rho 0.40 12               1.000000            0.083333         -0.005391                       0.916667                           0.083333                  0.0                0.008872
   wrong_high projected_gradient 0.05 12               0.916667            0.083333         -0.035575                       0.916667                           0.083333                  0.0                0.003968
   wrong_high projected_gradient 0.10 12               0.750000            0.083333         -0.060159                       0.916667                           0.083333                  0.0                0.009303
   wrong_high projected_gradient 0.20 12               0.416667            0.166667         -0.044546                       0.916667                           0.083333                  0.0                0.017484
   wrong_high projected_gradient 0.40 12               0.416667            0.166667          0.007005                       0.916667                           0.083333                  0.0                0.032713
   wrong_high  random_activation 0.05 12               1.000000            0.083333         -0.000115                       0.416667                           0.000000                  0.0                0.000084
   wrong_high  random_activation 0.10 12               1.000000            0.083333         -0.000204                       0.416667                           0.083333                  0.0                0.000222
```

## Selective Policy Benchmark

```text
               policy        metric    value  n
           confidence auroc_correct 0.901818 36
           confidence auprc_correct 0.967737 36
     negative_entropy auroc_correct 0.887273 36
     negative_entropy auprc_correct 0.960917 36
               margin auroc_correct 0.909091 36
               margin auprc_correct 0.968247 36
     self_consistency auroc_correct 0.600000 36
     self_consistency auprc_correct 0.739724 36
verbalized_confidence auroc_correct 0.723636 36
verbalized_confidence auprc_correct 0.868264 36
                  rho auroc_correct 0.898182 36
                  rho auprc_correct 0.959425 36
    accessible_policy auroc_correct 0.916364 36
    accessible_policy auprc_correct 0.969551 36
```

## Selective Risk Curves

```text
               policy  target_coverage  coverage  selective_risk  answered
           confidence             0.25  0.250000        0.000000         9
           confidence             0.50  0.500000        0.000000        18
           confidence             0.75  0.750000        0.185185        27
           confidence             1.00  1.000000        0.305556        36
     negative_entropy             0.25  0.250000        0.000000         9
     negative_entropy             0.50  0.500000        0.000000        18
     negative_entropy             0.75  0.750000        0.185185        27
     negative_entropy             1.00  1.000000        0.305556        36
               margin             0.25  0.250000        0.000000         9
               margin             0.50  0.500000        0.000000        18
               margin             0.75  0.750000        0.148148        27
               margin             1.00  1.000000        0.305556        36
     self_consistency             0.25  0.861111        0.258065        31
     self_consistency             0.50  0.861111        0.258065        31
     self_consistency             0.75  0.861111        0.258065        31
     self_consistency             1.00  1.000000        0.305556        36
verbalized_confidence             0.25  0.250000        0.111111         9
verbalized_confidence             0.50  0.500000        0.222222        18
verbalized_confidence             0.75  0.750000        0.222222        27
verbalized_confidence             1.00  1.000000        0.305556        36
                  rho             0.25  0.250000        0.000000         9
                  rho             0.50  0.500000        0.055556        18
                  rho             0.75  0.750000        0.148148        27
                  rho             1.00  1.000000        0.305556        36
    accessible_policy             0.25  0.250000        0.000000         9
    accessible_policy             0.50  0.500000        0.000000        18
    accessible_policy             0.75  0.750000        0.185185        27
    accessible_policy             1.00  1.000000        0.305556        36
```

## Route Score Summary

```text
    route  layer  n  rho_mean  rho_median
delta_pca     12 36  0.569431    0.570262
delta_pca     23 36  0.538262    0.557992
   random     12 36  0.478709    0.482789
   random     23 36  0.556464    0.652617
state_pca     12 36  0.018453    0.007706
state_pca     23 36  0.510100    0.554470
```

## Interpretation Guardrails

- The intervention is local logit-lens hidden-state repair, not a full rollout benchmark.
- The dataset is small and synthetic QA/reasoning; treat results as a pilot, not deployment evidence.
- A positive result requires improved selective risk/calibration without destroying correct answers; otherwise it is a failure mode.
