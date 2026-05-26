# Confidence Repair And Selective Abstention

Objective test of confidence repair on overconfident-wrong cases and selective abstention policies.

## Setup

```json
{
  "status": "completed",
  "model": "distilgpt2",
  "device": "cuda:0",
  "torch_dtype": "float16",
  "seed": 20260604,
  "n_items": 216,
  "n_records": 432,
  "answer_token_ids": {
    "A": 317,
    "B": 347,
    "C": 327,
    "D": 360
  },
  "layers": "auto",
  "pca_dim": 1,
  "eps": "0.05,0.1,0.2,0.4",
  "high_confidence_threshold": 0.7,
  "low_confidence_threshold": 0.45,
  "max_items": 216,
  "bootstrap": 300
}
```

## Group Counts

```text
outcome_group   n  confidence_mean  accuracy
 correct_high  61         0.870319       1.0
  correct_mid   1         0.689113       1.0
   wrong_high 149         0.869671       0.0
    wrong_mid   5         0.646642       0.0
```

## Calibration Metrics

```text
                      condition  accuracy  mean_confidence  overconfident_wrong_rate      nll    brier      ece
                         before  0.287037         0.863855                  0.689815 2.364256 1.195672 0.576818
accessible_rho_after_eps_median  0.287037         0.993037                  0.712963 4.956855 1.412166 0.706000
```

## Repair Summary

```text
outcome_group             method  eps   n  answer_preserved_rate  correct_after_rate  confidence_delta  wrong_confidence_reduced_rate  correct_confidence_increased_rate  overcorrection_rate  semantic_drift_proxy_mean  gold_probability_delta
 correct_high     accessible_rho 0.05 122                    1.0                 1.0          0.000263                       0.000000                           0.319672                  0.0               8.620549e-07                0.000263
 correct_high     accessible_rho 0.10 122                    1.0                 1.0          0.000645                       0.000000                           0.557377                  0.0               2.372279e-06                0.000645
 correct_high     accessible_rho 0.20 122                    1.0                 1.0          0.001159                       0.000000                           0.581967                  0.0               6.968415e-06                0.001159
 correct_high     accessible_rho 0.40 122                    1.0                 1.0          0.002193                       0.000000                           0.770492                  0.0               2.504368e-05                0.002193
 correct_high            low_rho 0.05 122                    1.0                 1.0          0.000048                       0.000000                           0.204918                  0.0               5.211503e-07                0.000048
 correct_high            low_rho 0.10 122                    1.0                 1.0          0.000221                       0.000000                           0.311475                  0.0               8.753047e-07                0.000221
 correct_high            low_rho 0.20 122                    1.0                 1.0          0.000376                       0.000000                           0.581967                  0.0               1.586884e-06                0.000376
 correct_high            low_rho 0.40 122                    1.0                 1.0          0.000683                       0.000000                           0.680328                  0.0               4.733797e-06                0.000683
 correct_high projected_gradient 0.05 122                    1.0                 1.0          0.002495                       0.000000                           0.852459                  0.0               2.256426e-05                0.002495
 correct_high projected_gradient 0.10 122                    1.0                 1.0          0.004845                       0.000000                           0.967213                  0.0               8.532363e-05                0.004845
 correct_high projected_gradient 0.20 122                    1.0                 1.0          0.009084                       0.000000                           1.000000                  0.0               3.146269e-04                0.009084
 correct_high projected_gradient 0.40 122                    1.0                 1.0          0.016238                       0.000000                           1.000000                  0.0               1.115313e-03                0.016238
 correct_high  random_activation 0.05 122                    1.0                 1.0          0.000024                       0.000000                           0.204918                  0.0               4.947795e-07                0.000024
 correct_high  random_activation 0.10 122                    1.0                 1.0         -0.000039                       0.000000                           0.254098                  0.0               6.035054e-07               -0.000039
 correct_high  random_activation 0.20 122                    1.0                 1.0         -0.000022                       0.000000                           0.377049                  0.0               1.432725e-06               -0.000022
 correct_high  random_activation 0.40 122                    1.0                 1.0         -0.000023                       0.000000                           0.385246                  0.0               3.302447e-06               -0.000023
 correct_high standard_delta_pca 0.05 122                    1.0                 1.0         -0.000429                       0.000000                           0.229508                  0.0               1.329466e-06               -0.000429
 correct_high standard_delta_pca 0.10 122                    1.0                 1.0         -0.000849                       0.000000                           0.213115                  0.0               3.460175e-06               -0.000849
 correct_high standard_delta_pca 0.20 122                    1.0                 1.0         -0.001793                       0.000000                           0.237705                  0.0               1.269388e-05               -0.001793
 correct_high standard_delta_pca 0.40 122                    1.0                 1.0         -0.003683                       0.000000                           0.229508                  0.0               4.925976e-05               -0.003683
 correct_high   temperature_1.25 0.00 122                    1.0                 1.0         -0.035276                       0.000000                           0.000000                  0.0               3.305511e-03               -0.035276
 correct_high    temperature_1.5 0.00 122                    1.0                 1.0         -0.075157                       0.000000                           0.000000                  0.0               1.173447e-02               -0.075157
 correct_high      temperature_2 0.00 122                    1.0                 1.0         -0.156946                       0.000000                           0.000000                  0.0               3.619209e-02               -0.156946
  correct_mid     accessible_rho 0.05   2                    1.0                 1.0          0.000092                       0.000000                           0.500000                  0.0               1.904039e-07                0.000092
  correct_mid     accessible_rho 0.10   2                    1.0                 1.0          0.000571                       0.000000                           0.500000                  0.0               1.437878e-06                0.000571
  correct_mid     accessible_rho 0.20   2                    1.0                 1.0          0.001054                       0.000000                           0.500000                  0.0               4.592972e-06                0.001054
  correct_mid     accessible_rho 0.40   2                    1.0                 1.0          0.002539                       0.000000                           0.500000                  0.0               2.694374e-05                0.002539
  correct_mid            low_rho 0.05   2                    1.0                 1.0          0.000000                       0.000000                           0.000000                  0.0               0.000000e+00                0.000000
  correct_mid            low_rho 0.10   2                    1.0                 1.0          0.000081                       0.000000                           0.500000                  0.0               1.676499e-07                0.000081
  correct_mid            low_rho 0.20   2                    1.0                 1.0         -0.000323                       0.000000                           0.000000                  0.0               9.865998e-07               -0.000323
  correct_mid            low_rho 0.40   2                    1.0                 1.0          0.000401                       0.000000                           1.000000                  0.0               2.760190e-06                0.000401
  correct_mid projected_gradient 0.05   2                    1.0                 1.0          0.002389                       0.000000                           0.500000                  0.0               2.332169e-05                0.002389
  correct_mid projected_gradient 0.10   2                    1.0                 1.0          0.004647                       0.000000                           1.000000                  0.0               7.959423e-05                0.004647
  correct_mid projected_gradient 0.20   2                    1.0                 1.0          0.008474                       0.000000                           1.000000                  0.0               2.851213e-04                0.008474
  correct_mid projected_gradient 0.40   2                    1.0                 1.0          0.015129                       0.000000                           1.000000                  0.0               1.007523e-03                0.015129
  correct_mid  random_activation 0.05   2                    1.0                 1.0         -0.000404                       0.000000                           0.000000                  0.0               8.001378e-07               -0.000404
  correct_mid  random_activation 0.10   2                    1.0                 1.0         -0.000404                       0.000000                           0.000000                  0.0               8.001378e-07               -0.000404
  correct_mid  random_activation 0.20   2                    1.0                 1.0          0.000092                       0.000000                           0.500000                  0.0               1.904039e-07                0.000092
  correct_mid  random_activation 0.40   2                    1.0                 1.0         -0.000643                       0.000000                           0.000000                  0.0               3.990976e-06               -0.000643
  correct_mid standard_delta_pca 0.05   2                    1.0                 1.0         -0.000497                       0.000000                           0.000000                  0.0               9.715705e-07               -0.000497
  correct_mid standard_delta_pca 0.10   2                    1.0                 1.0         -0.000990                       0.000000                           0.000000                  0.0               3.847487e-06               -0.000990
  correct_mid standard_delta_pca 0.20   2                    1.0                 1.0         -0.001513                       0.000000                           0.500000                  0.0               1.049812e-05               -0.001513
  correct_mid standard_delta_pca 0.40   2                    1.0                 1.0         -0.002721                       0.000000                           0.000000                  0.0               2.885906e-05               -0.002721
  correct_mid   temperature_1.25 0.00   2                    1.0                 1.0         -0.037526                       0.000000                           0.000000                  0.0               3.716793e-03               -0.037526
  correct_mid    temperature_1.5 0.00   2                    1.0                 1.0         -0.082334                       0.000000                           0.000000                  0.0               1.345489e-02               -0.082334
  correct_mid      temperature_2 0.00   2                    1.0                 1.0         -0.176444                       0.000000                           0.000000                  0.0               4.191623e-02               -0.176444
   wrong_high     accessible_rho 0.05 298                    1.0                 0.0         -0.000296                       0.348993                           0.000000                  0.0               1.008765e-06                0.000088
   wrong_high     accessible_rho 0.10 298                    1.0                 0.0         -0.000628                       0.520134                           0.000000                  0.0               2.181734e-06                0.000174
   wrong_high     accessible_rho 0.20 298                    1.0                 0.0         -0.001186                       0.573826                           0.000000                  0.0               6.633881e-06                0.000301
   wrong_high     accessible_rho 0.40 298                    1.0                 0.0         -0.002314                       0.721477                           0.000000                  0.0               2.397212e-05                0.000567
   wrong_high            low_rho 0.05 298                    1.0                 0.0         -0.000150                       0.265101                           0.000000                  0.0               6.718227e-07                0.000043
   wrong_high            low_rho 0.10 298                    1.0                 0.0         -0.000292                       0.345638                           0.000000                  0.0               9.690142e-07                0.000062
   wrong_high            low_rho 0.20 298                    1.0                 0.0         -0.000552                       0.550336                           0.000000                  0.0               2.271860e-06                0.000094
   wrong_high            low_rho 0.40 298                    1.0                 0.0         -0.000961                       0.677852                           0.000000                  0.0               6.440825e-06                0.000172
   wrong_high projected_gradient 0.05 298                    1.0                 0.0         -0.003375                       0.798658                           0.000000                  0.0               3.225438e-05                0.000811
   wrong_high projected_gradient 0.10 298                    1.0                 0.0         -0.006971                       0.969799                           0.000000                  0.0               1.308874e-04                0.001649
   wrong_high projected_gradient 0.20 298                    1.0                 0.0         -0.014738                       1.000000                           0.000000                  0.0               5.494522e-04                0.003478
   wrong_high projected_gradient 0.40 298                    1.0                 0.0         -0.033006                       1.000000                           0.000000                  0.0               2.445765e-03                0.007782
   wrong_high  random_activation 0.05 298                    1.0                 0.0          0.000002                       0.204698                           0.000000                  0.0               6.198132e-07                0.000031
   wrong_high  random_activation 0.10 298                    1.0                 0.0          0.000017                       0.271812                           0.000000                  0.0               9.570964e-07               -0.000009
```

## Selective Policy Benchmark

```text
               policy        metric    value   n
           confidence auroc_correct 0.499005 216
           confidence auprc_correct 0.286478 216
     negative_entropy auroc_correct 0.509164 216
     negative_entropy auprc_correct 0.290448 216
               margin auroc_correct 0.482876 216
               margin auprc_correct 0.281842 216
     self_consistency auroc_correct      NaN 216
     self_consistency auprc_correct      NaN 216
verbalized_confidence auroc_correct 0.449885 216
verbalized_confidence auprc_correct 0.263171 216
                  rho auroc_correct 0.558442 216
                  rho auprc_correct 0.412782 216
    accessible_policy auroc_correct 0.508274 216
    accessible_policy auprc_correct 0.349081 216
```

## Selective Risk Curves

```text
               policy  target_coverage  coverage  selective_risk  answered
           confidence             0.25      0.25        0.703704        54
           confidence             0.50      0.50        0.703704       108
           confidence             0.75      0.75        0.709877       162
           confidence             1.00      1.00        0.712963       216
     negative_entropy             0.25      0.25        0.703704        54
     negative_entropy             0.50      0.50        0.703704       108
     negative_entropy             0.75      0.75        0.697531       162
     negative_entropy             1.00      1.00        0.712963       216
               margin             0.25      0.25        0.740741        54
               margin             0.50      0.50        0.722222       108
               margin             0.75      0.75        0.716049       162
               margin             1.00      1.00        0.712963       216
     self_consistency             0.25      1.00        0.712963       216
     self_consistency             0.50      1.00        0.712963       216
     self_consistency             0.75      1.00        0.712963       216
     self_consistency             1.00      1.00        0.712963       216
verbalized_confidence             0.25      0.25        0.759259        54
verbalized_confidence             0.50      0.50        0.759259       108
verbalized_confidence             0.75      0.75        0.722222       162
verbalized_confidence             1.00      1.00        0.712963       216
                  rho             0.25      0.25        0.592593        54
                  rho             0.50      0.50        0.703704       108
                  rho             0.75      0.75        0.716049       162
                  rho             1.00      1.00        0.712963       216
    accessible_policy             0.25      0.25        0.703704        54
    accessible_policy             0.50      0.50        0.740741       108
    accessible_policy             0.75      0.75        0.709877       162
    accessible_policy             1.00      1.00        0.712963       216
```

## Route Score Summary

```text
    route  layer   n  rho_mean  rho_median
delta_pca      3 216  0.891556    0.889326
delta_pca      5 216  0.917649    0.920894
   random      3 216  0.348095    0.347905
   random      5 216  0.106532    0.080230
state_pca      3 216  0.896210    0.902326
state_pca      5 216  0.991699    0.993888
```

## Bootstrap Confidence Intervals

```text
               metric        group             method  eps  coverage    ci_low   median  ci_high
             accuracy         base               none  NaN       NaN  0.236111 0.287037 0.337963
answer_preserved_rate correct_high     accessible_rho 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high     accessible_rho 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high     accessible_rho 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high     accessible_rho 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high            low_rho 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high            low_rho 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high            low_rho 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high            low_rho 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high projected_gradient 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high projected_gradient 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high projected_gradient 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high projected_gradient 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high  random_activation 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high  random_activation 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high  random_activation 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high  random_activation 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high standard_delta_pca 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high standard_delta_pca 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high standard_delta_pca 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high standard_delta_pca 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high   temperature_1.25 0.00       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high    temperature_1.5 0.00       NaN  1.000000 1.000000 1.000000
answer_preserved_rate correct_high      temperature_2 0.00       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high     accessible_rho 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high     accessible_rho 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high     accessible_rho 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high     accessible_rho 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high            low_rho 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high            low_rho 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high            low_rho 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high            low_rho 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high projected_gradient 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high projected_gradient 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high projected_gradient 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high projected_gradient 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high  random_activation 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high  random_activation 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high  random_activation 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high  random_activation 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high standard_delta_pca 0.05       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high standard_delta_pca 0.10       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high standard_delta_pca 0.20       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high standard_delta_pca 0.40       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high   temperature_1.25 0.00       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high    temperature_1.5 0.00       NaN  1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high      temperature_2 0.00       NaN  1.000000 1.000000 1.000000
     confidence_delta correct_high     accessible_rho 0.05       NaN  0.000156 0.000250 0.000368
     confidence_delta correct_high     accessible_rho 0.10       NaN  0.000505 0.000638 0.000783
     confidence_delta correct_high     accessible_rho 0.20       NaN  0.000930 0.001155 0.001409
     confidence_delta correct_high     accessible_rho 0.40       NaN  0.001779 0.002186 0.002622
     confidence_delta correct_high            low_rho 0.05       NaN -0.000034 0.000043 0.000133
     confidence_delta correct_high            low_rho 0.10       NaN  0.000132 0.000220 0.000306
     confidence_delta correct_high            low_rho 0.20       NaN  0.000309 0.000374 0.000458
     confidence_delta correct_high            low_rho 0.40       NaN  0.000566 0.000682 0.000795
     confidence_delta correct_high projected_gradient 0.05       NaN  0.002271 0.002483 0.002748
     confidence_delta correct_high projected_gradient 0.10       NaN  0.004444 0.004833 0.005321
     confidence_delta correct_high projected_gradient 0.20       NaN  0.008326 0.009078 0.009966
     confidence_delta correct_high projected_gradient 0.40       NaN  0.014873 0.016240 0.017851
     confidence_delta correct_high  random_activation 0.05       NaN -0.000054 0.000024 0.000084
```

## Interpretation Guardrails

- The intervention is local logit-lens hidden-state repair, not a full rollout benchmark.
- Groups use absolute global confidence thresholds, not within-error medians.
- The dataset is synthetic QA/reasoning; treat results as a scaling pilot, not deployment evidence.
- A positive result requires improved selective risk/calibration without destroying correct answers; otherwise it is a failure mode.
