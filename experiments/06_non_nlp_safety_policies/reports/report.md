# Non-NLP Rho-Guided Safety Policy Tests

These are complete non-NLP decision-policy tests on real local sklearn datasets, not synthetic mock rows. The medical test uses the Wisconsin breast-cancer diagnostic classification dataset. The perception test uses the handwritten-digits image dataset as a compact vision-perception surrogate; it is not claimed to be an autonomous-driving dataset.

For each fold, a small MLP is trained from scratch, hidden-state PCA routes are fitted on the training split only, and test decisions use scalar uncertainty, entropy-gradient norm, and rho thresholds calibrated on the training split only. Correctness is used only for evaluation. The rho policies defer high-uncertainty/low-rho cases and apply a light hidden-route refinement to high-uncertainty/high-rho cases under an output-drift cap; the guarded variant sends the case to review if refinement fails to move entropy below the fold's training-calibrated uncertainty threshold.

## Route Diagnostics
```text
                  domain    n  base_accuracy  mean_entropy  mean_rho_high  mean_rho_low  high_uncertainty_rate  low_accessibility_given_high_uncertainty  mean_entropy_drop_after_intervention  mean_output_drift_after_intervention  top1_changed_after_intervention
   medical_breast_cancer 1707       0.971295      0.019133       0.728532      0.003823               0.314587                                  0.510242                              0.008239                              0.002249                         0.000000
vision_digits_perception 5391       0.971063      0.059984       0.568473      0.008969               0.322018                                  0.494240                              0.028868                              0.007835                         0.001298
```

## Policy Summary
```text
                  domain                      policy    n  coverage  human_review_cost  intervention_rate  automatic_accuracy  error_on_non_deferred  safe_decision_rate  false_intervention_rate  near_miss_surrogate_rate  compute_sensor_budget  preservation_when_no_change_needed  mean_output_drift  mean_entropy_drop
   medical_breast_cancer        entropy_human_review 1707  0.685413           0.314587           0.000000            0.997436               0.002564            0.998243                 0.287639                  0.001757               0.314587                            1.000000           0.000000           0.000000
   medical_breast_cancer       gradient_human_review 1707  0.685413           0.314587           0.000000            0.997436               0.002564            0.998243                 0.287639                  0.001757               0.314587                            1.000000           0.000000           0.000000
   medical_breast_cancer   rho_guided_guarded_refine 1707  0.829525           0.170475           0.144112            0.988701               0.011299            0.990627                 0.287639                  0.009373               0.206503                            1.000000           0.000849           0.003653
   medical_breast_cancer rho_guided_review_or_refine 1707  0.839484           0.160516           0.154071            0.986741               0.013259            0.988869                 0.287639                  0.011131               0.199033                            1.000000           0.001080           0.004150
vision_digits_perception        entropy_human_review 5391  0.677982           0.322018           0.000000            0.997264               0.002736            0.998145                 0.294936                  0.001855               0.322018                            1.000000           0.000000           0.000000
vision_digits_perception       gradient_human_review 5391  0.678724           0.321276           0.000000            0.997267               0.002733            0.998145                 0.294194                  0.001855               0.321276                            1.000000           0.000000           0.000000
vision_digits_perception   rho_guided_guarded_refine 5391  0.812651           0.187349           0.134669            0.993381               0.006619            0.994621                 0.294936                  0.005379               0.221017                            1.000000           0.002216           0.010631
vision_digits_perception rho_guided_review_or_refine 5391  0.840846           0.159154           0.162864            0.981028               0.018972            0.984047                 0.294936                  0.013727               0.199870                            0.999775           0.003710           0.014131
```

## Rho-Guided Contrasts
```text
                  domain                                             contrast                             metric  rho_guided  control  delta_rho_minus_control
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review                           coverage    0.839484 0.685413                 0.154071
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review                  human_review_cost    0.160516 0.314587                -0.154071
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review                 automatic_accuracy    0.986741 0.997436                -0.010695
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review              error_on_non_deferred    0.013259 0.002564                 0.010695
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review                 safe_decision_rate    0.988869 0.998243                -0.009373
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review            false_intervention_rate    0.287639 0.287639                 0.000000
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review           near_miss_surrogate_rate    0.011131 0.001757                 0.009373
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review              compute_sensor_budget    0.199033 0.314587                -0.115554
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review preservation_when_no_change_needed    1.000000 1.000000                 0.000000
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review                           coverage    0.839484 0.685413                 0.154071
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review                  human_review_cost    0.160516 0.314587                -0.154071
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review                 automatic_accuracy    0.986741 0.997436                -0.010695
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review              error_on_non_deferred    0.013259 0.002564                 0.010695
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review                 safe_decision_rate    0.988869 0.998243                -0.009373
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review            false_intervention_rate    0.287639 0.287639                 0.000000
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review           near_miss_surrogate_rate    0.011131 0.001757                 0.009373
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review              compute_sensor_budget    0.199033 0.314587                -0.115554
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review preservation_when_no_change_needed    1.000000 1.000000                 0.000000
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review                           coverage    0.829525 0.685413                 0.144112
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review                  human_review_cost    0.170475 0.314587                -0.144112
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review                 automatic_accuracy    0.988701 0.997436                -0.008735
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review              error_on_non_deferred    0.011299 0.002564                 0.008735
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review                 safe_decision_rate    0.990627 0.998243                -0.007616
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review            false_intervention_rate    0.287639 0.287639                 0.000000
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review           near_miss_surrogate_rate    0.009373 0.001757                 0.007616
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review              compute_sensor_budget    0.206503 0.314587                -0.108084
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review preservation_when_no_change_needed    1.000000 1.000000                 0.000000
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review                           coverage    0.829525 0.685413                 0.144112
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review                  human_review_cost    0.170475 0.314587                -0.144112
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review                 automatic_accuracy    0.988701 0.997436                -0.008735
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review              error_on_non_deferred    0.011299 0.002564                 0.008735
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review                 safe_decision_rate    0.990627 0.998243                -0.007616
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review            false_intervention_rate    0.287639 0.287639                 0.000000
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review           near_miss_surrogate_rate    0.009373 0.001757                 0.007616
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review              compute_sensor_budget    0.206503 0.314587                -0.108084
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review preservation_when_no_change_needed    1.000000 1.000000                 0.000000
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review                           coverage    0.840846 0.677982                 0.162864
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review                  human_review_cost    0.159154 0.322018                -0.162864
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review                 automatic_accuracy    0.981028 0.997264                -0.016236
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review              error_on_non_deferred    0.018972 0.002736                 0.016236
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review                 safe_decision_rate    0.984047 0.998145                -0.014098
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review            false_intervention_rate    0.294936 0.294936                 0.000000
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review           near_miss_surrogate_rate    0.013727 0.001855                 0.011872
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review              compute_sensor_budget    0.199870 0.322018                -0.122148
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review preservation_when_no_change_needed    0.999775 1.000000                -0.000225
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review                           coverage    0.840846 0.678724                 0.162122
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review                  human_review_cost    0.159154 0.321276                -0.162122
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review                 automatic_accuracy    0.981028 0.997267                -0.016239
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review              error_on_non_deferred    0.018972 0.002733                 0.016239
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review                 safe_decision_rate    0.984047 0.998145                -0.014098
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review            false_intervention_rate    0.294936 0.294194                 0.000742
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review           near_miss_surrogate_rate    0.013727 0.001855                 0.011872
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review              compute_sensor_budget    0.199870 0.321276                -0.121406
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review preservation_when_no_change_needed    0.999775 1.000000                -0.000225
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review                           coverage    0.812651 0.677982                 0.134669
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review                  human_review_cost    0.187349 0.322018                -0.134669
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review                 automatic_accuracy    0.993381 0.997264                -0.003884
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review              error_on_non_deferred    0.006619 0.002736                 0.003884
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review                 safe_decision_rate    0.994621 0.998145                -0.003524
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review            false_intervention_rate    0.294936 0.294936                 0.000000
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review           near_miss_surrogate_rate    0.005379 0.001855                 0.003524
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review              compute_sensor_budget    0.221017 0.322018                -0.101002
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review preservation_when_no_change_needed    1.000000 1.000000                 0.000000
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review                           coverage    0.812651 0.678724                 0.133927
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review                  human_review_cost    0.187349 0.321276                -0.133927
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review                 automatic_accuracy    0.993381 0.997267                -0.003887
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review              error_on_non_deferred    0.006619 0.002733                 0.003887
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review                 safe_decision_rate    0.994621 0.998145                -0.003524
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review            false_intervention_rate    0.294936 0.294194                 0.000742
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review           near_miss_surrogate_rate    0.005379 0.001855                 0.003524
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review              compute_sensor_budget    0.221017 0.321276                -0.100260
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review preservation_when_no_change_needed    1.000000 1.000000                 0.000000
```

## Cluster Bootstrap CI
```text
                  domain                                             contrast                             metric    ci_low    median   ci_high
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review                 automatic_accuracy -0.012304 -0.008615 -0.005161
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review              compute_sensor_budget -0.120388 -0.107958 -0.095837
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review                           coverage  0.127782  0.143944  0.160518
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review              error_on_non_deferred  0.005161  0.008615  0.012304
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review            false_intervention_rate  0.000000  0.000000  0.000000
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review                  human_review_cost -0.160518 -0.143944 -0.127782
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review           near_miss_surrogate_rate  0.004681  0.007611  0.010557
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review preservation_when_no_change_needed  0.000000  0.000000  0.000000
   medical_breast_cancer    rho_guided_guarded_refine_vs_entropy_human_review                 safe_decision_rate -0.010557 -0.007611 -0.004681
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review                 automatic_accuracy -0.012304 -0.008615 -0.005161
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review              compute_sensor_budget -0.120388 -0.107958 -0.095837
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review                           coverage  0.127782  0.143944  0.160518
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review              error_on_non_deferred  0.005161  0.008615  0.012304
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review            false_intervention_rate  0.000000  0.000000  0.000000
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review                  human_review_cost -0.160518 -0.143944 -0.127782
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review           near_miss_surrogate_rate  0.004681  0.007611  0.010557
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review preservation_when_no_change_needed  0.000000  0.000000  0.000000
   medical_breast_cancer   rho_guided_guarded_refine_vs_gradient_human_review                 safe_decision_rate -0.010557 -0.007611 -0.004681
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review                 automatic_accuracy -0.015219 -0.010567 -0.006166
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review              compute_sensor_budget -0.128370 -0.115182 -0.102810
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review                           coverage  0.137081  0.153576  0.171161
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review              error_on_non_deferred  0.006166  0.010567  0.015219
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review            false_intervention_rate  0.000000  0.000000  0.000000
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review                  human_review_cost -0.171161 -0.153576 -0.137081
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review           near_miss_surrogate_rate  0.005834  0.009368  0.012925
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review preservation_when_no_change_needed  0.000000  0.000000  0.000000
   medical_breast_cancer  rho_guided_review_or_refine_vs_entropy_human_review                 safe_decision_rate -0.012925 -0.009368 -0.005834
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review                 automatic_accuracy -0.015219 -0.010567 -0.006166
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review              compute_sensor_budget -0.128370 -0.115182 -0.102810
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review                           coverage  0.137081  0.153576  0.171161
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review              error_on_non_deferred  0.006166  0.010567  0.015219
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review            false_intervention_rate  0.000000  0.000000  0.000000
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review                  human_review_cost -0.171161 -0.153576 -0.137081
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review           near_miss_surrogate_rate  0.005834  0.009368  0.012925
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review preservation_when_no_change_needed  0.000000  0.000000  0.000000
   medical_breast_cancer rho_guided_review_or_refine_vs_gradient_human_review                 safe_decision_rate -0.012925 -0.009368 -0.005834
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review                 automatic_accuracy -0.005807 -0.003888 -0.002252
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review              compute_sensor_budget -0.107581 -0.101039 -0.094340
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review                           coverage  0.125787  0.134719  0.143442
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review              error_on_non_deferred  0.002252  0.003888  0.005807
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review            false_intervention_rate  0.000000  0.000000  0.000000
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review                  human_review_cost -0.143442 -0.134719 -0.125787
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review           near_miss_surrogate_rate  0.002226  0.003524  0.005009
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review preservation_when_no_change_needed  0.000000  0.000000  0.000000
vision_digits_perception    rho_guided_guarded_refine_vs_entropy_human_review                 safe_decision_rate -0.005009 -0.003524 -0.002226
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review                 automatic_accuracy -0.005811 -0.003890 -0.002256
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review              compute_sensor_budget -0.107597 -0.100320 -0.092681
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review                           coverage  0.124117  0.133902  0.143337
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review              error_on_non_deferred  0.002256  0.003890  0.005811
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review            false_intervention_rate -0.001484  0.000742  0.002789
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review                  human_review_cost -0.143337 -0.133902 -0.124117
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review           near_miss_surrogate_rate  0.002226  0.003524  0.005009
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review preservation_when_no_change_needed  0.000000  0.000000  0.000000
vision_digits_perception   rho_guided_guarded_refine_vs_gradient_human_review                 safe_decision_rate -0.005009 -0.003524 -0.002226
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review                 automatic_accuracy -0.021136 -0.016251 -0.012122
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review              compute_sensor_budget -0.129016 -0.122218 -0.115212
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review                           coverage  0.153616  0.162957  0.172021
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review              error_on_non_deferred  0.012122  0.016251  0.021136
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review            false_intervention_rate  0.000000  0.000000  0.000000
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review                  human_review_cost -0.172021 -0.162957 -0.153616
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review           near_miss_surrogate_rate  0.008900  0.011867  0.015584
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review preservation_when_no_change_needed -0.000676 -0.000224  0.000000
vision_digits_perception  rho_guided_review_or_refine_vs_entropy_human_review                 safe_decision_rate -0.018193 -0.014098 -0.010390
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review                 automatic_accuracy -0.021140 -0.016253 -0.012125
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review              compute_sensor_budget -0.129225 -0.121625 -0.113240
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review                           coverage  0.151733  0.162478  0.172292
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review              error_on_non_deferred  0.012125  0.016253  0.021140
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review            false_intervention_rate -0.001484  0.000742  0.002789
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review                  human_review_cost -0.172292 -0.162478 -0.151733
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review           near_miss_surrogate_rate  0.008900  0.011867  0.015584
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review preservation_when_no_change_needed -0.000676 -0.000224  0.000000
vision_digits_perception rho_guided_review_or_refine_vs_gradient_human_review                 safe_decision_rate -0.018193 -0.014098 -0.010390
```

## Files
```text
non_nlp_route_records.csv
non_nlp_policy_records.csv
non_nlp_policy_summary.csv
non_nlp_policy_contrasts.csv
non_nlp_policy_bootstrap_ci.csv
report.md
```
