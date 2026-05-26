# Confidence Repair And Selective Abstention

Objective test of confidence repair on overconfident-wrong cases and selective abstention policies.

## Setup

```json
{
  "status": "completed",
  "model": "Qwen/Qwen2.5-0.5B",
  "device": "cuda:0",
  "torch_dtype": "float16",
  "seed": 20260604,
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
  "eps": "0.05,0.1,0.2,0.4",
  "high_confidence_threshold": 0.7,
  "low_confidence_threshold": 0.45,
  "max_items": 216,
  "bootstrap": 300
}
```

## Group Counts

```text
outcome_group  n  confidence_mean  accuracy
 correct_high 85         0.897703       1.0
  correct_low 36         0.350955       1.0
  correct_mid 31         0.594638       1.0
    wrong_low 63         0.342502       0.0
    wrong_mid  1         0.451736       0.0
```

## Calibration Metrics

```text
                      condition  accuracy  mean_confidence  overconfident_wrong_rate      nll    brier      ece
                         before  0.703704         0.599085                       0.0 0.690242 0.359072 0.112621
accessible_rho_after_eps_median  0.703704         0.712706                       0.0 0.573760 0.314581 0.080437
```

## Repair Summary

```text
outcome_group             method  eps   n  answer_preserved_rate  correct_after_rate  confidence_delta  wrong_confidence_reduced_rate  correct_confidence_increased_rate  overcorrection_rate  semantic_drift_proxy_mean  gold_probability_delta
 correct_high     accessible_rho 0.05 170               1.000000            0.582353         -0.000713                       0.417647                           0.376471                  0.0               1.849080e-06            4.174791e-04
 correct_high     accessible_rho 0.10 170               1.000000            0.582353         -0.001450                       0.417647                           0.429412                  0.0               7.122967e-06            8.177106e-04
 correct_high     accessible_rho 0.20 170               0.988235            0.588235         -0.002858                       0.417647                           0.441176                  0.0               2.813050e-05            1.761933e-03
 correct_high     accessible_rho 0.40 170               0.964706            0.588235         -0.005369                       0.417647                           0.535294                  0.0               1.117802e-04            3.545330e-03
 correct_high            low_rho 0.05 170               1.000000            0.582353         -0.000093                       0.305882                           0.252941                  0.0               7.032564e-07            7.082545e-07
 correct_high            low_rho 0.10 170               1.000000            0.582353         -0.000124                       0.252941                           0.311765                  0.0               2.472243e-06            2.129283e-05
 correct_high            low_rho 0.20 170               1.000000            0.582353         -0.000246                       0.317647                           0.400000                  0.0               8.932231e-06            1.189615e-04
 correct_high            low_rho 0.40 170               0.988235            0.588235         -0.000565                       0.352941                           0.429412                  0.0               3.455073e-05            2.867280e-04
 correct_high projected_gradient 0.05 170               0.888235            0.600000         -0.014065                       0.417647                           0.582353                  0.0               9.496466e-04            1.176817e-02
 correct_high projected_gradient 0.10 170               0.770588            0.629412         -0.018470                       0.417647                           0.582353                  0.0               3.724939e-03            2.314683e-02
 correct_high projected_gradient 0.20 170               0.588235            0.670588          0.002316                       0.417647                           0.582353                  0.0               1.386490e-02            4.376850e-02
 correct_high projected_gradient 0.40 170               0.582353            0.670588          0.054796                       0.417647                           0.582353                  0.0               4.333881e-02            7.257703e-02
 correct_high  random_activation 0.05 170               1.000000            0.582353          0.000054                       0.200000                           0.141176                  0.0               1.806371e-06            3.908166e-05
 correct_high  random_activation 0.10 170               1.000000            0.582353          0.000015                       0.200000                           0.229412                  0.0               7.020372e-06            1.528904e-04
 correct_high  random_activation 0.20 170               0.988235            0.588235          0.000040                       0.223529                           0.317647                  0.0               2.705921e-05            3.272974e-04
 correct_high  random_activation 0.40 170               0.970588            0.582353          0.000379                       0.211765                           0.294118                  0.0               1.083187e-04            7.273274e-04
 correct_high standard_delta_pca 0.05 170               1.000000            0.582353         -0.000525                       0.370588                           0.311765                  0.0               1.125413e-06            6.038339e-05
 correct_high standard_delta_pca 0.10 170               1.000000            0.582353         -0.001050                       0.376471                           0.258824                  0.0               3.393766e-06            1.046908e-04
 correct_high standard_delta_pca 0.20 170               0.994118            0.588235         -0.002175                       0.376471                           0.247059                  0.0               1.347124e-05            2.946692e-04
 correct_high standard_delta_pca 0.40 170               0.976471            0.588235         -0.004101                       0.376471                           0.405882                  0.0               5.175041e-05            6.925907e-04
 correct_high   temperature_1.25 0.00 170               1.000000            0.582353         -0.029813                       0.417647                           0.000000                  0.0               2.209304e-03           -9.028182e-03
 correct_high    temperature_1.5 0.00 170               1.000000            0.582353         -0.059163                       0.417647                           0.000000                  0.0               7.864639e-03           -2.308588e-02
 correct_high      temperature_2 0.00 170               1.000000            0.582353         -0.116543                       0.417647                           0.000000                  0.0               2.451609e-02           -5.937098e-02
  correct_low     accessible_rho 0.05  72               1.000000            0.486111         -0.000601                       0.444444                           0.236111                  0.0               3.381677e-06            8.623432e-04
  correct_low     accessible_rho 0.10  72               1.000000            0.486111         -0.001258                       0.486111                           0.305556                  0.0               1.445308e-05            1.952023e-03
  correct_low     accessible_rho 0.20  72               0.972222            0.500000         -0.001784                       0.500000                           0.416667                  0.0               5.594045e-05            4.226371e-03
  correct_low     accessible_rho 0.40  72               0.944444            0.513889         -0.003185                       0.513889                           0.458333                  0.0               2.226656e-04            8.327415e-03
  correct_low            low_rho 0.05  72               1.000000            0.486111         -0.000360                       0.319444                           0.208333                  0.0               2.435395e-06            8.825419e-05
  correct_low            low_rho 0.10  72               0.986111            0.486111         -0.000409                       0.361111                           0.291667                  0.0               9.547228e-06            5.531592e-04
  correct_low            low_rho 0.20  72               0.972222            0.500000         -0.000353                       0.416667                           0.347222                  0.0               3.125393e-05            1.316062e-03
  correct_low            low_rho 0.40  72               0.972222            0.500000         -0.000223                       0.500000                           0.375000                  0.0               1.201612e-04            3.121039e-03
  correct_low projected_gradient 0.05  72               0.805556            0.569444         -0.006239                       0.513889                           0.486111                  0.0               1.041393e-03            2.018876e-02
  correct_low projected_gradient 0.10  72               0.611111            0.666667          0.007833                       0.513889                           0.486111                  0.0               4.110161e-03            4.040542e-02
  correct_low projected_gradient 0.20  72               0.513889            0.722222          0.058557                       0.513889                           0.486111                  0.0               1.531789e-02            7.720725e-02
  correct_low projected_gradient 0.40  72               0.486111            0.750000          0.143340                       0.513889                           0.486111                  0.0               4.807304e-02            1.341888e-01
  correct_low  random_activation 0.05  72               1.000000            0.486111          0.000246                       0.152778                           0.138889                  0.0               2.737921e-06            4.310592e-05
  correct_low  random_activation 0.10  72               1.000000            0.486111          0.000374                       0.194444                           0.180556                  0.0               8.514995e-06            2.825597e-04
  correct_low  random_activation 0.20  72               0.986111            0.486111          0.001076                       0.222222                           0.222222                  0.0               3.061167e-05            9.579687e-04
  correct_low  random_activation 0.40  72               0.972222            0.500000          0.002306                       0.208333                           0.263889                  0.0               1.207990e-04            1.895867e-03
  correct_low standard_delta_pca 0.05  72               1.000000            0.486111         -0.000689                       0.291667                           0.152778                  0.0               2.193626e-06           -3.613503e-04
  correct_low standard_delta_pca 0.10  72               0.986111            0.486111         -0.001214                       0.375000                           0.111111                  0.0               4.705790e-06           -1.384631e-04
  correct_low standard_delta_pca 0.20  72               0.986111            0.486111         -0.001735                       0.361111                           0.111111                  0.0               1.531002e-05            2.578465e-05
  correct_low standard_delta_pca 0.40  72               0.972222            0.500000         -0.003454                       0.402778                           0.166667                  0.0               6.238715e-05           -3.789124e-05
  correct_low   temperature_1.25 0.00  72               1.000000            0.486111         -0.037931                       0.513889                           0.000000                  0.0               1.732918e-03           -1.773229e-02
  correct_low    temperature_1.5 0.00  72               1.000000            0.486111         -0.065081                       0.513889                           0.000000                  0.0               5.188699e-03           -3.037476e-02
  correct_low      temperature_2 0.00  72               1.000000            0.486111         -0.101029                       0.513889                           0.000000                  0.0               1.278505e-02           -4.699881e-02
  correct_mid     accessible_rho 0.05  62               1.000000            0.580645         -0.000490                       0.419355                           0.370968                  0.0               3.148567e-06            7.704191e-04
  correct_mid     accessible_rho 0.10  62               1.000000            0.580645         -0.001006                       0.419355                           0.419355                  0.0               1.099659e-05            1.511504e-03
  correct_mid     accessible_rho 0.20  62               1.000000            0.580645         -0.001937                       0.419355                           0.451613                  0.0               4.010385e-05            3.182634e-03
  correct_mid     accessible_rho 0.40  62               1.000000            0.580645         -0.003823                       0.419355                           0.516129                  0.0               1.592447e-04            6.322113e-03
  correct_mid            low_rho 0.05  62               1.000000            0.580645          0.000118                       0.338710                           0.290323                  0.0               1.235259e-06            2.525892e-04
  correct_mid            low_rho 0.10  62               1.000000            0.580645          0.000291                       0.322581                           0.403226                  0.0               4.714451e-06            5.826218e-04
  correct_mid            low_rho 0.20  62               1.000000            0.580645          0.000501                       0.354839                           0.370968                  0.0               1.544991e-05            9.554417e-04
  correct_mid            low_rho 0.40  62               1.000000            0.580645          0.000671                       0.419355                           0.403226                  0.0               6.046234e-05            1.764455e-03
  correct_mid projected_gradient 0.05  62               0.887097            0.596774         -0.012412                       0.419355                           0.580645                  0.0               9.747130e-04            1.667814e-02
  correct_mid projected_gradient 0.10  62               0.677419            0.661290         -0.009070                       0.419355                           0.580645                  0.0               3.825427e-03            3.303358e-02
  correct_mid projected_gradient 0.20  62               0.580645            0.677419          0.027519                       0.419355                           0.580645                  0.0               1.418051e-02            6.275230e-02
  correct_mid projected_gradient 0.40  62               0.580645            0.693548          0.091318                       0.419355                           0.580645                  0.0               4.423498e-02            1.069380e-01
  correct_mid  random_activation 0.05  62               1.000000            0.580645          0.000261                       0.177419                           0.209677                  0.0               2.709938e-06            1.177642e-04
  correct_mid  random_activation 0.10  62               1.000000            0.580645          0.000511                       0.145161                           0.225806                  0.0               9.878270e-06            1.155482e-04
```

## Selective Policy Benchmark

```text
               policy        metric    value   n
           confidence auroc_correct 0.888158 216
           confidence auprc_correct 0.961099 216
     negative_entropy auroc_correct 0.879729 216
     negative_entropy auprc_correct 0.957312 216
               margin auroc_correct 0.891345 216
               margin auprc_correct 0.961554 216
     self_consistency auroc_correct 0.587891 216
     self_consistency auprc_correct 0.742769 216
verbalized_confidence auroc_correct 0.594829 216
verbalized_confidence auprc_correct 0.801230 216
                  rho auroc_correct 0.768298 216
                  rho auprc_correct 0.870850 216
    accessible_policy auroc_correct 0.872122 216
    accessible_policy auprc_correct 0.952152 216
```

## Selective Risk Curves

```text
               policy  target_coverage  coverage  selective_risk  answered
           confidence             0.25  0.250000        0.000000        54
           confidence             0.50  0.500000        0.000000       108
           confidence             0.75  0.750000        0.166667       162
           confidence             1.00  1.000000        0.296296       216
     negative_entropy             0.25  0.250000        0.000000        54
     negative_entropy             0.50  0.500000        0.000000       108
     negative_entropy             0.75  0.750000        0.179012       162
     negative_entropy             1.00  1.000000        0.296296       216
               margin             0.25  0.250000        0.000000        54
               margin             0.50  0.500000        0.000000       108
               margin             0.75  0.750000        0.160494       162
               margin             1.00  1.000000        0.296296       216
     self_consistency             0.25  0.824074        0.252809       178
     self_consistency             0.50  0.824074        0.252809       178
     self_consistency             0.75  0.824074        0.252809       178
     self_consistency             1.00  1.000000        0.296296       216
verbalized_confidence             0.25  0.250000        0.166667        54
verbalized_confidence             0.50  0.500000        0.268519       108
verbalized_confidence             0.75  0.750000        0.277778       162
verbalized_confidence             1.00  1.000000        0.296296       216
                  rho             0.25  0.250000        0.111111        54
                  rho             0.50  0.500000        0.138889       108
                  rho             0.75  0.750000        0.179012       162
                  rho             1.00  1.000000        0.296296       216
    accessible_policy             0.25  0.250000        0.000000        54
    accessible_policy             0.50  0.500000        0.018519       108
    accessible_policy             0.75  0.750000        0.185185       162
    accessible_policy             1.00  1.000000        0.296296       216
```

## Route Score Summary

```text
    route  layer   n  rho_mean  rho_median
delta_pca     12 216  0.377943    0.379404
delta_pca     23 216  0.515431    0.547903
   random     12 216  0.393909    0.416063
   random     23 216  0.404687    0.325817
state_pca     12 216  0.031125    0.012007
state_pca     23 216  0.487832    0.479437
```

## Bootstrap Confidence Intervals

```text
               metric        group             method  eps  coverage   ci_low   median  ci_high
             accuracy         base               none  NaN       NaN 0.636458 0.699074 0.757060
answer_preserved_rate correct_high     accessible_rho 0.05       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high     accessible_rho 0.10       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high     accessible_rho 0.20       NaN 0.965629 0.988235 1.000000
answer_preserved_rate correct_high     accessible_rho 0.40       NaN 0.930691 0.963886 0.987882
answer_preserved_rate correct_high            low_rho 0.05       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high            low_rho 0.10       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high            low_rho 0.20       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high            low_rho 0.40       NaN 0.965827 0.988700 1.000000
answer_preserved_rate correct_high projected_gradient 0.05       NaN 0.841343 0.886786 0.921963
answer_preserved_rate correct_high projected_gradient 0.10       NaN 0.708333 0.766667 0.810811
answer_preserved_rate correct_high projected_gradient 0.20       NaN 0.546491 0.584270 0.621951
answer_preserved_rate correct_high projected_gradient 0.40       NaN 0.542933 0.578947 0.616015
answer_preserved_rate correct_high  random_activation 0.05       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high  random_activation 0.10       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high  random_activation 0.20       NaN 0.965629 0.988235 1.000000
answer_preserved_rate correct_high  random_activation 0.40       NaN 0.941858 0.970588 0.993979
answer_preserved_rate correct_high standard_delta_pca 0.05       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high standard_delta_pca 0.10       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high standard_delta_pca 0.20       NaN 0.977011 0.994186 1.000000
answer_preserved_rate correct_high standard_delta_pca 0.40       NaN 0.945387 0.976190 0.994189
answer_preserved_rate correct_high   temperature_1.25 0.00       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high    temperature_1.5 0.00       NaN 1.000000 1.000000 1.000000
answer_preserved_rate correct_high      temperature_2 0.00       NaN 1.000000 1.000000 1.000000
answer_preserved_rate  correct_low     accessible_rho 0.05       NaN 1.000000 1.000000 1.000000
answer_preserved_rate  correct_low     accessible_rho 0.10       NaN 1.000000 1.000000 1.000000
answer_preserved_rate  correct_low     accessible_rho 0.20       NaN 0.930833 0.973329 1.000000
answer_preserved_rate  correct_low     accessible_rho 0.40       NaN 0.878788 0.947368 0.986308
answer_preserved_rate  correct_low            low_rho 0.05       NaN 1.000000 1.000000 1.000000
answer_preserved_rate  correct_low            low_rho 0.10       NaN 0.952331 0.986486 1.000000
answer_preserved_rate  correct_low            low_rho 0.20       NaN 0.935484 0.972973 1.000000
answer_preserved_rate  correct_low            low_rho 0.40       NaN 0.935484 0.972973 1.000000
answer_preserved_rate  correct_low projected_gradient 0.05       NaN 0.725806 0.809762 0.895980
answer_preserved_rate  correct_low projected_gradient 0.10       NaN 0.526656 0.617157 0.721446
answer_preserved_rate  correct_low projected_gradient 0.20       NaN 0.430556 0.516129 0.604160
answer_preserved_rate  correct_low projected_gradient 0.40       NaN 0.397238 0.486645 0.562500
answer_preserved_rate  correct_low  random_activation 0.05       NaN 1.000000 1.000000 1.000000
answer_preserved_rate  correct_low  random_activation 0.10       NaN 1.000000 1.000000 1.000000
answer_preserved_rate  correct_low  random_activation 0.20       NaN 0.952331 0.986486 1.000000
answer_preserved_rate  correct_low  random_activation 0.40       NaN 0.930833 0.973329 1.000000
answer_preserved_rate  correct_low standard_delta_pca 0.05       NaN 1.000000 1.000000 1.000000
answer_preserved_rate  correct_low standard_delta_pca 0.10       NaN 0.952331 0.986486 1.000000
answer_preserved_rate  correct_low standard_delta_pca 0.20       NaN 0.952331 0.986486 1.000000
answer_preserved_rate  correct_low standard_delta_pca 0.40       NaN 0.930833 0.973329 1.000000
answer_preserved_rate  correct_low   temperature_1.25 0.00       NaN 1.000000 1.000000 1.000000
answer_preserved_rate  correct_low    temperature_1.5 0.00       NaN 1.000000 1.000000 1.000000
answer_preserved_rate  correct_low      temperature_2 0.00       NaN 1.000000 1.000000 1.000000
answer_preserved_rate    wrong_low     accessible_rho 0.05       NaN 0.972133 0.992424 1.000000
answer_preserved_rate    wrong_low     accessible_rho 0.10       NaN 0.942422 0.976190 1.000000
answer_preserved_rate    wrong_low     accessible_rho 0.20       NaN 0.909571 0.942465 0.976190
answer_preserved_rate    wrong_low     accessible_rho 0.40       NaN 0.831275 0.888889 0.930321
answer_preserved_rate    wrong_low            low_rho 0.05       NaN 0.972133 0.992424 1.000000
answer_preserved_rate    wrong_low            low_rho 0.10       NaN 0.960317 0.983607 1.000000
answer_preserved_rate    wrong_low            low_rho 0.20       NaN 0.948610 0.976376 1.000000
answer_preserved_rate    wrong_low            low_rho 0.40       NaN 0.908368 0.953125 0.982912
answer_preserved_rate    wrong_low projected_gradient 0.05       NaN 0.696021 0.763336 0.824087
answer_preserved_rate    wrong_low projected_gradient 0.10       NaN 0.511646 0.585826 0.658730
answer_preserved_rate    wrong_low projected_gradient 0.20       NaN 0.367326 0.433893 0.509259
answer_preserved_rate    wrong_low projected_gradient 0.40       NaN 0.261024 0.330508 0.402621
answer_preserved_rate    wrong_low  random_activation 0.05       NaN 0.959330 0.984495 1.000000
```

## Interpretation Guardrails

- The intervention is local logit-lens hidden-state repair, not a full rollout benchmark.
- Groups use absolute global confidence thresholds, not within-error medians.
- The dataset is synthetic QA/reasoning; treat results as a scaling pilot, not deployment evidence.
- A positive result requires improved selective risk/calibration without destroying correct answers; otherwise it is a failure mode.
