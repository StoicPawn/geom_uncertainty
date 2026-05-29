# Non-NLP Rho-Guided Safety Policy Tests

These are complete non-NLP decision-policy tests on real local sklearn datasets, not synthetic mock rows. The medical test uses the Wisconsin breast-cancer diagnostic classification dataset. The perception test uses the handwritten-digits image dataset as a compact vision-perception surrogate; it is not claimed to be an autonomous-driving dataset.

For each fold, a small MLP is trained from scratch, hidden-state PCA routes are fitted on the training split only, and test decisions use scalar uncertainty, entropy-gradient norm, Jacobian norm, and rho thresholds calibrated on the training split only. Correctness is used only for evaluation. The rho policies defer high-uncertainty/low-rho cases and apply a light hidden-route refinement to high-uncertainty/high-rho cases under an output-drift cap; the guarded variant sends the case to review if refinement fails to move entropy below the fold's training-calibrated uncertainty threshold.

The equal-cost panel is the stricter inverse test: entropy, entropy-gradient norm, input-Jacobian Frobenius norm, and rho all defer the same fraction of held-out cases within each fold. The rho score ranks cases by high uncertainty and low local accessibility, so a gain there means better triage at the same review cost rather than a different budget.

## Equal-Cost Verdict
Equal-cost selector test is a boundary case: rho does not beat entropy/gradient at the same review cost overall. It does beat the Jacobian selector on the digits perception surrogate.

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

## Equal Review-Cost Selector Summary
```text
                  domain  target_review_cost                policy    n  review_cost  coverage  automatic_accuracy  error_on_non_deferred  safe_decision_rate  reviewed_error_rate  deferred_error_capture
   medical_breast_cancer                0.17  entropy_fixed_review 1707     0.166960  0.833040            0.995077               0.004923            0.995899             0.147368                0.857143
   medical_breast_cancer                0.17 gradient_fixed_review 1707     0.166960  0.833040            0.995077               0.004923            0.995899             0.147368                0.857143
   medical_breast_cancer                0.17 jacobian_fixed_review 1707     0.166960  0.833040            0.996484               0.003516            0.997071             0.154386                0.897959
   medical_breast_cancer                0.17      rho_fixed_review 1707     0.166960  0.833040            0.984529               0.015471            0.987112             0.094737                0.551020
   medical_breast_cancer                0.19  entropy_fixed_review 1707     0.191564  0.808436            0.995652               0.004348            0.996485             0.131498                0.877551
   medical_breast_cancer                0.19 gradient_fixed_review 1707     0.191564  0.808436            0.995652               0.004348            0.996485             0.131498                0.877551
   medical_breast_cancer                0.19 jacobian_fixed_review 1707     0.191564  0.808436            0.997101               0.002899            0.997657             0.137615                0.918367
   medical_breast_cancer                0.19      rho_fixed_review 1707     0.191564  0.808436            0.984058               0.015942            0.987112             0.082569                0.551020
vision_digits_perception                0.17  entropy_fixed_review 5391     0.169727  0.830273            0.996872               0.003128            0.997403             0.155191                0.910256
vision_digits_perception                0.17 gradient_fixed_review 5391     0.169727  0.830273            0.996872               0.003128            0.997403             0.155191                0.910256
vision_digits_perception                0.17 jacobian_fixed_review 5391     0.169727  0.830273            0.978776               0.021224            0.982378             0.066667                0.391026
vision_digits_perception                0.17      rho_fixed_review 5391     0.169727  0.830273            0.996425               0.003575            0.997032             0.153005                0.897436
vision_digits_perception                0.19  entropy_fixed_review 5391     0.189204  0.810796            0.996797               0.003203            0.997403             0.139216                0.910256
vision_digits_perception                0.19 gradient_fixed_review 5391     0.189204  0.810796            0.996797               0.003203            0.997403             0.139216                0.910256
vision_digits_perception                0.19 jacobian_fixed_review 5391     0.189204  0.810796            0.980325               0.019675            0.984047             0.068627                0.448718
vision_digits_perception                0.19      rho_fixed_review 5391     0.189204  0.810796            0.996340               0.003660            0.997032             0.137255                0.897436
```

## Equal Review-Cost Rho Contrasts
```text
                  domain  target_review_cost                                  contrast                 metric      rho  control  delta_rho_minus_control
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review            review_cost 0.166960 0.166960                 0.000000
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review               coverage 0.833040 0.833040                 0.000000
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review     automatic_accuracy 0.984529 0.995077                -0.010549
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review  error_on_non_deferred 0.015471 0.004923                 0.010549
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review     safe_decision_rate 0.987112 0.995899                -0.008787
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review    reviewed_error_rate 0.094737 0.147368                -0.052632
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review deferred_error_capture 0.551020 0.857143                -0.306122
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review            review_cost 0.166960 0.166960                 0.000000
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review               coverage 0.833040 0.833040                 0.000000
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review     automatic_accuracy 0.984529 0.995077                -0.010549
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review  error_on_non_deferred 0.015471 0.004923                 0.010549
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review     safe_decision_rate 0.987112 0.995899                -0.008787
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review    reviewed_error_rate 0.094737 0.147368                -0.052632
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review deferred_error_capture 0.551020 0.857143                -0.306122
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review            review_cost 0.166960 0.166960                 0.000000
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review               coverage 0.833040 0.833040                 0.000000
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review     automatic_accuracy 0.984529 0.996484                -0.011955
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review  error_on_non_deferred 0.015471 0.003516                 0.011955
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review     safe_decision_rate 0.987112 0.997071                -0.009959
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review    reviewed_error_rate 0.094737 0.154386                -0.059649
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review deferred_error_capture 0.551020 0.897959                -0.346939
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review            review_cost 0.191564 0.191564                 0.000000
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review               coverage 0.808436 0.808436                 0.000000
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review     automatic_accuracy 0.984058 0.995652                -0.011594
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review  error_on_non_deferred 0.015942 0.004348                 0.011594
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review     safe_decision_rate 0.987112 0.996485                -0.009373
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review    reviewed_error_rate 0.082569 0.131498                -0.048930
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review deferred_error_capture 0.551020 0.877551                -0.326531
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review            review_cost 0.191564 0.191564                 0.000000
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review               coverage 0.808436 0.808436                 0.000000
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review     automatic_accuracy 0.984058 0.995652                -0.011594
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review  error_on_non_deferred 0.015942 0.004348                 0.011594
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review     safe_decision_rate 0.987112 0.996485                -0.009373
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review    reviewed_error_rate 0.082569 0.131498                -0.048930
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review deferred_error_capture 0.551020 0.877551                -0.326531
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review            review_cost 0.191564 0.191564                 0.000000
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review               coverage 0.808436 0.808436                 0.000000
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review     automatic_accuracy 0.984058 0.997101                -0.013043
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review  error_on_non_deferred 0.015942 0.002899                 0.013043
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review     safe_decision_rate 0.987112 0.997657                -0.010545
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review    reviewed_error_rate 0.082569 0.137615                -0.055046
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review deferred_error_capture 0.551020 0.918367                -0.367347
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review            review_cost 0.169727 0.169727                 0.000000
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review               coverage 0.830273 0.830273                 0.000000
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review     automatic_accuracy 0.996425 0.996872                -0.000447
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review  error_on_non_deferred 0.003575 0.003128                 0.000447
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review     safe_decision_rate 0.997032 0.997403                -0.000371
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review    reviewed_error_rate 0.153005 0.155191                -0.002186
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review deferred_error_capture 0.897436 0.910256                -0.012821
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review            review_cost 0.169727 0.169727                 0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review               coverage 0.830273 0.830273                 0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review     automatic_accuracy 0.996425 0.996872                -0.000447
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review  error_on_non_deferred 0.003575 0.003128                 0.000447
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review     safe_decision_rate 0.997032 0.997403                -0.000371
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review    reviewed_error_rate 0.153005 0.155191                -0.002186
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review deferred_error_capture 0.897436 0.910256                -0.012821
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review            review_cost 0.169727 0.169727                 0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review               coverage 0.830273 0.830273                 0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review     automatic_accuracy 0.996425 0.978776                 0.017650
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review  error_on_non_deferred 0.003575 0.021224                -0.017650
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review     safe_decision_rate 0.997032 0.982378                 0.014654
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review    reviewed_error_rate 0.153005 0.066667                 0.086339
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review deferred_error_capture 0.897436 0.391026                 0.506410
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review            review_cost 0.189204 0.189204                 0.000000
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review               coverage 0.810796 0.810796                 0.000000
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review     automatic_accuracy 0.996340 0.996797                -0.000458
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review  error_on_non_deferred 0.003660 0.003203                 0.000458
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review     safe_decision_rate 0.997032 0.997403                -0.000371
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review    reviewed_error_rate 0.137255 0.139216                -0.001961
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review deferred_error_capture 0.897436 0.910256                -0.012821
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review            review_cost 0.189204 0.189204                 0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review               coverage 0.810796 0.810796                 0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review     automatic_accuracy 0.996340 0.996797                -0.000458
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review  error_on_non_deferred 0.003660 0.003203                 0.000458
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review     safe_decision_rate 0.997032 0.997403                -0.000371
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review    reviewed_error_rate 0.137255 0.139216                -0.001961
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review deferred_error_capture 0.897436 0.910256                -0.012821
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review            review_cost 0.189204 0.189204                 0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review               coverage 0.810796 0.810796                 0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review     automatic_accuracy 0.996340 0.980325                 0.016015
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review  error_on_non_deferred 0.003660 0.019675                -0.016015
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review     safe_decision_rate 0.997032 0.984047                 0.012985
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review    reviewed_error_rate 0.137255 0.068627                 0.068627
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review deferred_error_capture 0.897436 0.448718                 0.448718
```

## Equal Review-Cost Bootstrap CI
```text
                  domain  target_review_cost                                  contrast                 metric    ci_low    median   ci_high
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review     automatic_accuracy -0.014789 -0.010549 -0.006320
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review               coverage  0.000000  0.000000  0.000000
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review deferred_error_capture -0.425532 -0.302678 -0.181818
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review  error_on_non_deferred  0.006320  0.010549  0.014789
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review            review_cost  0.000000  0.000000  0.000000
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review    reviewed_error_rate -0.073684 -0.052632 -0.031579
   medical_breast_cancer                0.17  rho_fixed_review_vs_entropy_fixed_review     safe_decision_rate -0.012317 -0.008787 -0.005266
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review     automatic_accuracy -0.014789 -0.010549 -0.006320
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review               coverage  0.000000  0.000000  0.000000
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review deferred_error_capture -0.425532 -0.302678 -0.181818
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review  error_on_non_deferred  0.006320  0.010549  0.014789
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review            review_cost  0.000000  0.000000  0.000000
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review    reviewed_error_rate -0.073684 -0.052632 -0.031579
   medical_breast_cancer                0.17 rho_fixed_review_vs_gradient_fixed_review     safe_decision_rate -0.012317 -0.008787 -0.005266
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review     automatic_accuracy -0.016197 -0.011963 -0.007032
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review               coverage  0.000000  0.000000  0.000000
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review deferred_error_capture -0.468085 -0.350000 -0.207547
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review  error_on_non_deferred  0.007032  0.011963  0.016197
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review            review_cost  0.000000  0.000000  0.000000
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review    reviewed_error_rate -0.080702 -0.059649 -0.035088
   medical_breast_cancer                0.17 rho_fixed_review_vs_jacobian_fixed_review     safe_decision_rate -0.013490 -0.009965 -0.005858
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review     automatic_accuracy -0.015942 -0.011594 -0.007246
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review               coverage  0.000000  0.000000  0.000000
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review deferred_error_capture -0.428571 -0.325581 -0.226380
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review  error_on_non_deferred  0.007246  0.011594  0.015942
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review            review_cost  0.000000  0.000000  0.000000
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review    reviewed_error_rate -0.067278 -0.048930 -0.030488
   medical_breast_cancer                0.19  rho_fixed_review_vs_entropy_fixed_review     safe_decision_rate -0.012888 -0.009373 -0.005855
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review     automatic_accuracy -0.015942 -0.011594 -0.007246
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review               coverage  0.000000  0.000000  0.000000
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review deferred_error_capture -0.428571 -0.325581 -0.226380
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review  error_on_non_deferred  0.007246  0.011594  0.015942
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review            review_cost  0.000000  0.000000  0.000000
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review    reviewed_error_rate -0.067278 -0.048930 -0.030488
   medical_breast_cancer                0.19 rho_fixed_review_vs_gradient_fixed_review     safe_decision_rate -0.012888 -0.009373 -0.005855
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review     automatic_accuracy -0.016685 -0.013043 -0.008696
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review               coverage  0.000000  0.000000  0.000000
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review deferred_error_capture -0.470588 -0.366667 -0.269133
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review  error_on_non_deferred  0.008696  0.013043  0.016685
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review            review_cost  0.000000  0.000000  0.000000
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review    reviewed_error_rate -0.071031 -0.055046 -0.036807
   medical_breast_cancer                0.19 rho_fixed_review_vs_jacobian_fixed_review     safe_decision_rate -0.013511 -0.010545 -0.007034
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review     automatic_accuracy -0.001117 -0.000447  0.000000
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review               coverage  0.000000  0.000000  0.000000
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review deferred_error_capture -0.033790 -0.012346  0.000000
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review  error_on_non_deferred  0.000000  0.000447  0.001117
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review            review_cost  0.000000  0.000000  0.000000
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review    reviewed_error_rate -0.005464 -0.002186  0.000000
vision_digits_perception                0.17  rho_fixed_review_vs_entropy_fixed_review     safe_decision_rate -0.000927 -0.000371  0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review     automatic_accuracy -0.001117 -0.000447  0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review               coverage  0.000000  0.000000  0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review deferred_error_capture -0.033790 -0.012346  0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review  error_on_non_deferred  0.000000  0.000447  0.001117
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review            review_cost  0.000000  0.000000  0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review    reviewed_error_rate -0.005464 -0.002186  0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_gradient_fixed_review     safe_decision_rate -0.000927 -0.000371  0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review     automatic_accuracy  0.013402  0.017650  0.021890
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review               coverage  0.000000  0.000000  0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review deferred_error_capture  0.396734  0.505848  0.618437
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review  error_on_non_deferred -0.021890 -0.017650 -0.013402
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review            review_cost  0.000000  0.000000  0.000000
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review    reviewed_error_rate  0.065574  0.086339  0.107104
vision_digits_perception                0.17 rho_fixed_review_vs_jacobian_fixed_review     safe_decision_rate  0.011128  0.014654  0.018175
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review     automatic_accuracy -0.001143 -0.000457  0.000000
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review               coverage  0.000000  0.000000  0.000000
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review deferred_error_capture -0.033790 -0.012346  0.000000
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review  error_on_non_deferred  0.000000  0.000457  0.001143
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review            review_cost  0.000000  0.000000  0.000000
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review    reviewed_error_rate -0.004902 -0.001961  0.000000
vision_digits_perception                0.19  rho_fixed_review_vs_entropy_fixed_review     safe_decision_rate -0.000927 -0.000371  0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review     automatic_accuracy -0.001143 -0.000457  0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review               coverage  0.000000  0.000000  0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review deferred_error_capture -0.033790 -0.012346  0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review  error_on_non_deferred  0.000000  0.000457  0.001143
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review            review_cost  0.000000  0.000000  0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review    reviewed_error_rate -0.004902 -0.001961  0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_gradient_fixed_review     safe_decision_rate -0.000927 -0.000371  0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review     automatic_accuracy  0.012109  0.016013  0.020128
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review               coverage  0.000000  0.000000  0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review deferred_error_capture  0.342666  0.447205  0.560832
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review  error_on_non_deferred -0.020128 -0.016013 -0.012109
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review            review_cost  0.000000  0.000000  0.000000
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review    reviewed_error_rate  0.051936  0.068627  0.086275
vision_digits_perception                0.19 rho_fixed_review_vs_jacobian_fixed_review     safe_decision_rate  0.009820  0.012983  0.016321
```

## Files
```text
non_nlp_route_records.csv
non_nlp_policy_records.csv
non_nlp_policy_summary.csv
non_nlp_policy_contrasts.csv
non_nlp_policy_bootstrap_ci.csv
non_nlp_fixed_cost_policy_records.csv
non_nlp_fixed_cost_summary.csv
non_nlp_fixed_cost_contrasts.csv
non_nlp_fixed_cost_bootstrap_ci.csv
report.md
```
