# Confidence Repair And Selective Abstention

Objective test of confidence repair on overconfident-wrong cases and selective abstention policies.

## Setup

```json
{
  "status": "completed",
  "model": "distilgpt2",
  "device": "cuda:0",
  "torch_dtype": "float16",
  "seed": 20260605,
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
  "eps": "0.025,0.05,0.1,0.2,0.4,0.8",
  "high_confidence_threshold": 0.7,
  "low_confidence_threshold": 0.45,
  "max_items": 216,
  "bootstrap": 200,
  "prompt_mode": "normal"
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
outcome_group                      method   eps   n  answer_preserved_rate  correct_after_rate  confidence_delta  wrong_confidence_reduced_rate  correct_confidence_increased_rate  overcorrection_rate  semantic_drift_proxy_mean  gold_probability_delta
 correct_high              accessible_rho 0.025 122                    1.0                 1.0          0.000152                            0.0                           0.286885                  0.0               6.773980e-07                0.000152
 correct_high              accessible_rho 0.050 122                    1.0                 1.0          0.000263                            0.0                           0.319672                  0.0               8.620549e-07                0.000263
 correct_high              accessible_rho 0.100 122                    1.0                 1.0          0.000645                            0.0                           0.557377                  0.0               2.372279e-06                0.000645
 correct_high              accessible_rho 0.200 122                    1.0                 1.0          0.001159                            0.0                           0.581967                  0.0               6.968415e-06                0.001159
 correct_high              accessible_rho 0.400 122                    1.0                 1.0          0.002193                            0.0                           0.770492                  0.0               2.504368e-05                0.002193
 correct_high              accessible_rho 0.800 122                    1.0                 1.0          0.004311                            0.0                           0.844262                  0.0               9.362786e-05                0.004311
 correct_high                     low_rho 0.025 122                    1.0                 1.0          0.000074                            0.0                           0.163934                  0.0               3.569367e-07                0.000074
 correct_high                     low_rho 0.050 122                    1.0                 1.0          0.000114                            0.0                           0.155738                  0.0               3.974488e-07                0.000114
 correct_high                     low_rho 0.100 122                    1.0                 1.0          0.000148                            0.0                           0.295082                  0.0               8.113795e-07                0.000148
 correct_high                     low_rho 0.200 122                    1.0                 1.0          0.000473                            0.0                           0.598361                  0.0               2.112928e-06                0.000473
 correct_high                     low_rho 0.400 122                    1.0                 1.0          0.000849                            0.0                           0.606557                  0.0               6.150718e-06                0.000849
 correct_high                     low_rho 0.800 122                    1.0                 1.0          0.001698                            0.0                           0.803279                  0.0               2.230705e-05                0.001698
 correct_high          projected_gradient 0.025 122                    1.0                 1.0          0.001298                            0.0                           0.704918                  0.0               6.379382e-06                0.001298
 correct_high          projected_gradient 0.050 122                    1.0                 1.0          0.002495                            0.0                           0.852459                  0.0               2.256426e-05                0.002495
 correct_high          projected_gradient 0.100 122                    1.0                 1.0          0.004845                            0.0                           0.967213                  0.0               8.532363e-05                0.004845
 correct_high          projected_gradient 0.200 122                    1.0                 1.0          0.009084                            0.0                           1.000000                  0.0               3.146269e-04                0.009084
 correct_high          projected_gradient 0.400 122                    1.0                 1.0          0.016238                            0.0                           1.000000                  0.0               1.115313e-03                0.016238
 correct_high          projected_gradient 0.800 122                    1.0                 1.0          0.026102                            0.0                           1.000000                  0.0               3.452820e-03                0.026102
 correct_high           random_activation 0.025 122                    1.0                 1.0          0.000011                            0.0                           0.131148                  0.0               3.243808e-07                0.000011
 correct_high           random_activation 0.050 122                    1.0                 1.0          0.000019                            0.0                           0.172131                  0.0               4.295980e-07                0.000019
 correct_high           random_activation 0.100 122                    1.0                 1.0          0.000031                            0.0                           0.368852                  0.0               8.361412e-07                0.000031
 correct_high           random_activation 0.200 122                    1.0                 1.0          0.000043                            0.0                           0.418033                  0.0               1.269840e-06                0.000043
 correct_high           random_activation 0.400 122                    1.0                 1.0          0.000028                            0.0                           0.426230                  0.0               2.616528e-06                0.000028
 correct_high           random_activation 0.800 122                    1.0                 1.0          0.000044                            0.0                           0.434426                  0.0               9.257398e-06                0.000044
 correct_high          standard_delta_pca 0.025 122                    1.0                 1.0         -0.000250                            0.0                           0.188525                  0.0               9.635819e-07               -0.000250
 correct_high          standard_delta_pca 0.050 122                    1.0                 1.0         -0.000429                            0.0                           0.229508                  0.0               1.329466e-06               -0.000429
 correct_high          standard_delta_pca 0.100 122                    1.0                 1.0         -0.000849                            0.0                           0.213115                  0.0               3.460175e-06               -0.000849
 correct_high          standard_delta_pca 0.200 122                    1.0                 1.0         -0.001793                            0.0                           0.237705                  0.0               1.269388e-05               -0.001793
 correct_high          standard_delta_pca 0.400 122                    1.0                 1.0         -0.003683                            0.0                           0.229508                  0.0               4.925976e-05               -0.003683
 correct_high          standard_delta_pca 0.800 122                    1.0                 1.0         -0.007695                            0.0                           0.377049                  0.0               2.044613e-04               -0.007695
 correct_high            temperature_1.25 0.000 122                    1.0                 1.0         -0.035276                            0.0                           0.000000                  0.0               3.305511e-03               -0.035276
 correct_high             temperature_1.5 0.000 122                    1.0                 1.0         -0.075157                            0.0                           0.000000                  0.0               1.173447e-02               -0.075157
 correct_high               temperature_2 0.000 122                    1.0                 1.0         -0.156946                            0.0                           0.000000                  0.0               3.619209e-02               -0.156946
 correct_high wrong_logit_gradient_normed 0.025 122                    1.0                 1.0         -0.000108                            0.0                           0.286885                  0.0               8.831504e-07               -0.000108
 correct_high wrong_logit_gradient_normed 0.050 122                    1.0                 1.0         -0.000350                            0.0                           0.196721                  0.0               1.084905e-06               -0.000350
 correct_high wrong_logit_gradient_normed 0.100 122                    1.0                 1.0         -0.000780                            0.0                           0.073770                  0.0               2.676714e-06               -0.000780
 correct_high wrong_logit_gradient_normed 0.200 122                    1.0                 1.0         -0.001635                            0.0                           0.065574                  0.0               9.854776e-06               -0.001635
 correct_high wrong_logit_gradient_normed 0.400 122                    1.0                 1.0         -0.003240                            0.0                           0.008197                  0.0               3.557027e-05               -0.003240
 correct_high wrong_logit_gradient_normed 0.800 122                    1.0                 1.0         -0.006819                            0.0                           0.000000                  0.0               1.485285e-04               -0.006819
  correct_mid              accessible_rho 0.025   2                    1.0                 1.0          0.000000                            0.0                           0.000000                  0.0               0.000000e+00                0.000000
  correct_mid              accessible_rho 0.050   2                    1.0                 1.0          0.000092                            0.0                           0.500000                  0.0               1.904039e-07                0.000092
  correct_mid              accessible_rho 0.100   2                    1.0                 1.0          0.000571                            0.0                           0.500000                  0.0               1.437878e-06                0.000571
  correct_mid              accessible_rho 0.200   2                    1.0                 1.0          0.001054                            0.0                           0.500000                  0.0               4.592972e-06                0.001054
  correct_mid              accessible_rho 0.400   2                    1.0                 1.0          0.002539                            0.0                           0.500000                  0.0               2.694374e-05                0.002539
  correct_mid              accessible_rho 0.800   2                    1.0                 1.0          0.005297                            0.0                           1.000000                  0.0               1.183359e-04                0.005297
  correct_mid                     low_rho 0.025   2                    1.0                 1.0          0.000000                            0.0                           0.000000                  0.0               0.000000e+00                0.000000
  correct_mid                     low_rho 0.050   2                    1.0                 1.0          0.000491                            0.0                           0.500000                  0.0               9.585379e-07                0.000491
  correct_mid                     low_rho 0.100   2                    1.0                 1.0          0.000491                            0.0                           0.500000                  0.0               9.585379e-07                0.000491
  correct_mid                     low_rho 0.200   2                    1.0                 1.0          0.000564                            0.0                           1.000000                  0.0               1.537709e-06                0.000564
  correct_mid                     low_rho 0.400   2                    1.0                 1.0          0.001283                            0.0                           0.500000                  0.0               7.193653e-06                0.001283
  correct_mid                     low_rho 0.800   2                    1.0                 1.0          0.002207                            0.0                           1.000000                  0.0               2.042180e-05                0.002207
  correct_mid          projected_gradient 0.025   2                    1.0                 1.0          0.000975                            0.0                           0.500000                  0.0               3.808284e-06                0.000975
  correct_mid          projected_gradient 0.050   2                    1.0                 1.0          0.002389                            0.0                           0.500000                  0.0               2.332169e-05                0.002389
  correct_mid          projected_gradient 0.100   2                    1.0                 1.0          0.004647                            0.0                           1.000000                  0.0               7.959423e-05                0.004647
  correct_mid          projected_gradient 0.200   2                    1.0                 1.0          0.008474                            0.0                           1.000000                  0.0               2.851213e-04                0.008474
  correct_mid          projected_gradient 0.400   2                    1.0                 1.0          0.015129                            0.0                           1.000000                  0.0               1.007523e-03                0.015129
  correct_mid          projected_gradient 0.800   2                    1.0                 1.0          0.024406                            0.0                           1.000000                  0.0               3.138426e-03                0.024406
  correct_mid           random_activation 0.025   2                    1.0                 1.0          0.000000                            0.0                           0.000000                  0.0               0.000000e+00                0.000000
  correct_mid           random_activation 0.050   2                    1.0                 1.0          0.000092                            0.0                           0.500000                  0.0               1.904039e-07                0.000092
  correct_mid           random_activation 0.100   2                    1.0                 1.0          0.000000                            0.0                           0.000000                  0.0               0.000000e+00                0.000000
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
   random      3 216  0.874141    0.874427
   random      5 216  0.146583    0.109025
state_pca      3 216  0.896210    0.902326
state_pca      5 216  0.991699    0.993888
```

## Bootstrap Confidence Intervals

```text
               metric        group                      method   eps  coverage   ci_low   median  ci_high
             accuracy         base                        none   NaN       NaN 0.222222 0.287037 0.347338
answer_preserved_rate correct_high              accessible_rho 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high              accessible_rho 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high              accessible_rho 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high              accessible_rho 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high              accessible_rho 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high              accessible_rho 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high                     low_rho 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high                     low_rho 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high                     low_rho 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high                     low_rho 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high                     low_rho 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high                     low_rho 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          projected_gradient 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          projected_gradient 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          projected_gradient 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          projected_gradient 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          projected_gradient 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          projected_gradient 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high           random_activation 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high           random_activation 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high           random_activation 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high           random_activation 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high           random_activation 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high           random_activation 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          standard_delta_pca 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          standard_delta_pca 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          standard_delta_pca 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          standard_delta_pca 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          standard_delta_pca 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high          standard_delta_pca 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high            temperature_1.25 0.000       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high             temperature_1.5 0.000       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high               temperature_2 0.000       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high wrong_logit_gradient_normed 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high wrong_logit_gradient_normed 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high wrong_logit_gradient_normed 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high wrong_logit_gradient_normed 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high wrong_logit_gradient_normed 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high wrong_logit_gradient_normed 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high              accessible_rho 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high              accessible_rho 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high              accessible_rho 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high              accessible_rho 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high              accessible_rho 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high              accessible_rho 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high                     low_rho 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high                     low_rho 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high                     low_rho 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high                     low_rho 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high                     low_rho 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high                     low_rho 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high          projected_gradient 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high          projected_gradient 0.050       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high          projected_gradient 0.100       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high          projected_gradient 0.200       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high          projected_gradient 0.400       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high          projected_gradient 0.800       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high           random_activation 0.025       NaN 1.000000 1.000000 1.000000
answer_preserved_rate   wrong_high           random_activation 0.050       NaN 1.000000 1.000000 1.000000
```

## Wrong-High Pareto

```text
                     method   eps   n   drift_mean  wrong_confidence_reduction_mean  wrong_confidence_reduced_rate  answer_preserved_rate
                    low_rho 0.025 298 4.658650e-07                         0.000140                       0.191275                    1.0
          random_activation 0.025 298 5.425823e-07                         0.000028                       0.157718                    1.0
             accessible_rho 0.025 298 7.277605e-07                         0.000180                       0.305369                    1.0
          random_activation 0.050 298 7.372394e-07                         0.000040                       0.234899                    1.0
                    low_rho 0.050 298 7.507495e-07                         0.000144                       0.197987                    1.0
          random_activation 0.100 298 9.397498e-07                         0.000057                       0.315436                    1.0
wrong_logit_gradient_normed 0.025 298 1.004137e-06                         0.000201                       0.469799                    1.0
             accessible_rho 0.050 298 1.008765e-06                         0.000296                       0.348993                    1.0
         standard_delta_pca 0.025 298 1.284313e-06                         0.000360                       0.577181                    1.0
                    low_rho 0.100 298 1.346118e-06                         0.000313                       0.379195                    1.0
wrong_logit_gradient_normed 0.050 298 1.583031e-06                         0.000483                       0.546980                    1.0
          random_activation 0.200 298 1.733571e-06                         0.000081                       0.359060                    1.0
         standard_delta_pca 0.050 298 1.899594e-06                         0.000567                       0.550336                    1.0
             accessible_rho 0.100 298 2.181734e-06                         0.000628                       0.520134                    1.0
                    low_rho 0.200 298 3.676466e-06                         0.000640                       0.476510                    1.0
wrong_logit_gradient_normed 0.100 298 3.711560e-06                         0.001020                       0.667785                    1.0
          random_activation 0.400 298 4.122731e-06                         0.000145                       0.426174                    1.0
         standard_delta_pca 0.100 298 4.714426e-06                         0.001140                       0.644295                    1.0
             accessible_rho 0.200 298 6.633881e-06                         0.001186                       0.573826                    1.0
         projected_gradient 0.025 298 8.434145e-06                         0.001669                       0.674497                    1.0
wrong_logit_gradient_normed 0.200 298 1.205054e-05                         0.002025                       0.822148                    1.0
                    low_rho 0.400 298 1.248351e-05                         0.001246                       0.560403                    1.0
          random_activation 0.800 298 1.338307e-05                         0.000244                       0.426174                    1.0
         standard_delta_pca 0.200 298 1.638323e-05                         0.002292                       0.604027                    1.0
             accessible_rho 0.400 298 2.397212e-05                         0.002314                       0.721477                    1.0
         projected_gradient 0.050 298 3.225438e-05                         0.003375                       0.798658                    1.0
wrong_logit_gradient_normed 0.400 298 4.509247e-05                         0.004052                       0.919463                    1.0
                    low_rho 0.800 298 4.840180e-05                         0.002641                       0.751678                    1.0
         standard_delta_pca 0.400 298 6.448208e-05                         0.004660                       0.567114                    1.0
             accessible_rho 0.800 298 9.796873e-05                         0.004851                       0.838926                    1.0
         projected_gradient 0.100 298 1.308874e-04                         0.006971                       0.969799                    1.0
wrong_logit_gradient_normed 0.800 298 1.821385e-04                         0.008335                       1.000000                    1.0
         standard_delta_pca 0.800 298 2.625798e-04                         0.009651                       0.580537                    1.0
         projected_gradient 0.200 298 5.494522e-04                         0.014738                       1.000000                    1.0
         projected_gradient 0.400 298 2.445765e-03                         0.033006                       1.000000                    1.0
           temperature_1.25 0.000 298 3.472986e-03                         0.038933                       1.000000                    1.0
         projected_gradient 0.800 298 1.179738e-02                         0.081333                       1.000000                    1.0
            temperature_1.5 0.000 298 1.215790e-02                         0.081209                       1.000000                    1.0
              temperature_2 0.000 298 3.684209e-02                         0.164972                       1.000000                    1.0
```

## Wrong-High Pareto Bootstrap CI

```text
                     method   eps  drift_ci_low  drift_median  drift_ci_high  reduction_ci_low  reduction_median  reduction_ci_high  reduced_rate_ci_low  reduced_rate_median  reduced_rate_ci_high
             accessible_rho 0.025  6.112996e-07  7.246864e-07   8.563391e-07          0.000120          0.000183           0.000250             0.265101             0.305369              0.345638
             accessible_rho 0.050  8.169463e-07  1.001066e-06   1.224116e-06          0.000241          0.000294           0.000366             0.298574             0.352349              0.406040
             accessible_rho 0.100  1.829328e-06  2.197473e-06   2.647440e-06          0.000544          0.000632           0.000729             0.479866             0.520134              0.560403
             accessible_rho 0.200  5.492324e-06  6.600955e-06   7.952989e-06          0.001054          0.001189           0.001340             0.543624             0.573826              0.607466
             accessible_rho 0.400  2.011437e-05  2.408519e-05   2.886793e-05          0.002053          0.002328           0.002662             0.684564             0.721477              0.761745
             accessible_rho 0.800  8.169250e-05  9.865102e-05   1.182751e-04          0.004333          0.004869           0.005518             0.805369             0.838926              0.875839
                    low_rho 0.025  3.728275e-07  4.599122e-07   5.686668e-07          0.000092          0.000143           0.000194             0.157718             0.191275              0.228188
                    low_rho 0.050  5.837569e-07  7.568818e-07   9.574204e-07          0.000090          0.000149           0.000211             0.164430             0.201342              0.234983
                    low_rho 0.100  1.031015e-06  1.353604e-06   1.662654e-06          0.000229          0.000313           0.000386             0.338926             0.379195              0.426174
                    low_rho 0.200  2.512014e-06  3.678756e-06   4.707340e-06          0.000489          0.000647           0.000759             0.432802             0.476510              0.520218
                    low_rho 0.400  7.760024e-06  1.235484e-05   1.673221e-05          0.000961          0.001254           0.001513             0.510067             0.563758              0.614094
                    low_rho 0.800  3.030916e-05  4.825620e-05   6.304710e-05          0.002075          0.002639           0.003112             0.711409             0.751678              0.802013
         projected_gradient 0.025  7.922865e-06  8.480325e-06   9.257261e-06          0.001580          0.001678           0.001800             0.637584             0.674497              0.711409
         projected_gradient 0.050  3.003779e-05  3.237708e-05   3.509014e-05          0.003179          0.003386           0.003640             0.761661             0.798658              0.835654
         projected_gradient 0.100  1.236454e-04  1.314474e-04   1.405232e-04          0.006595          0.006991           0.007439             0.942953             0.969799              0.986577
         projected_gradient 0.200  5.220847e-04  5.501404e-04   5.847259e-04          0.013993          0.014748           0.015705             1.000000             1.000000              1.000000
         projected_gradient 0.400  2.318534e-03  2.445175e-03   2.595090e-03          0.031338          0.033006           0.035062             1.000000             1.000000              1.000000
         projected_gradient 0.800  1.127283e-02  1.180124e-02   1.246565e-02          0.077786          0.081323           0.085613             1.000000             1.000000              1.000000
          random_activation 0.025  4.223804e-07  5.388414e-07   6.844469e-07         -0.000027          0.000030           0.000076             0.124161             0.157718              0.194631
          random_activation 0.050  5.797611e-07  7.273744e-07   9.193986e-07         -0.000016          0.000042           0.000094             0.187836             0.234899              0.285235
          random_activation 0.100  7.702781e-07  9.442537e-07   1.144908e-06         -0.000006          0.000058           0.000131             0.271728             0.315436              0.362416
          random_activation 0.200  1.457303e-06  1.712940e-06   2.034172e-06          0.000007          0.000083           0.000182             0.305369             0.362416              0.409983
          random_activation 0.400  3.348802e-06  4.180941e-06   5.013120e-06          0.000011          0.000143           0.000316             0.369044             0.426174              0.480117
          random_activation 0.800  1.083577e-05  1.337895e-05   1.641294e-05         -0.000010          0.000229           0.000573             0.375839             0.426174              0.473154
         standard_delta_pca 0.025  1.131312e-06  1.282114e-06   1.514596e-06          0.000299          0.000364           0.000425             0.523406             0.580537              0.630956
         standard_delta_pca 0.050  1.639411e-06  1.903360e-06   2.180753e-06          0.000495          0.000568           0.000642             0.500000             0.550336              0.587332
         standard_delta_pca 0.100  4.264194e-06  4.722393e-06   5.277672e-06          0.001049          0.001144           0.001238             0.614094             0.644295              0.681292
         standard_delta_pca 0.200  1.520491e-05  1.638733e-05   1.789876e-05          0.002146          0.002296           0.002454             0.573826             0.604027              0.637584
         standard_delta_pca 0.400  6.055559e-05  6.455798e-05   7.016650e-05          0.004401          0.004665           0.005010             0.543624             0.570470              0.600671
         standard_delta_pca 0.800  2.480526e-04  2.622747e-04   2.826754e-04          0.009125          0.009657           0.010322             0.550336             0.580537              0.614178
           temperature_1.25 0.000  3.383367e-03  3.469114e-03   3.582029e-03          0.037573          0.038909           0.040438             1.000000             1.000000              1.000000
            temperature_1.5 0.000  1.185863e-02  1.214104e-02   1.250919e-02          0.078719          0.081125           0.084110             1.000000             1.000000              1.000000
              temperature_2 0.000  3.603995e-02  3.681278e-02   3.768945e-02          0.161050          0.164764           0.169979             1.000000             1.000000              1.000000
wrong_logit_gradient_normed 0.025  8.579803e-07  9.961215e-07   1.174400e-06          0.000153          0.000201           0.000263             0.406040             0.471477              0.530453
wrong_logit_gradient_normed 0.050  1.351878e-06  1.586693e-06   1.858808e-06          0.000426          0.000487           0.000558             0.493289             0.550336              0.593960
wrong_logit_gradient_normed 0.100  3.412440e-06  3.719671e-06   4.120653e-06          0.000955          0.001024           0.001119             0.630872             0.667785              0.708138
wrong_logit_gradient_normed 0.200  1.115998e-05  1.207355e-05   1.315583e-05          0.001901          0.002033           0.002181             0.781879             0.822148              0.855789
wrong_logit_gradient_normed 0.400  4.245817e-05  4.511992e-05   4.844965e-05          0.003815          0.004056           0.004338             0.885906             0.922819              0.949748
wrong_logit_gradient_normed 0.800  1.720683e-04  1.821002e-04   1.955813e-04          0.007876          0.008343           0.008956             1.000000             1.000000              1.000000
```

## Interpretation Guardrails

- The intervention is local logit-lens hidden-state repair, not a full rollout benchmark.
- Groups use absolute global confidence thresholds, not within-error medians.
- The dataset is synthetic QA/reasoning; treat results as a scaling pilot, not deployment evidence.
- A positive result requires improved selective risk/calibration without destroying correct answers; otherwise it is a failure mode.
