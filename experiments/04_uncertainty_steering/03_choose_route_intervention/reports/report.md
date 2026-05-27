# Choose-The-Route Intervention Test

This derived intervention test asks whether choosing the high-rho route for each example moves uncertainty more efficiently than low-rho, gradient-selected, random, and equal-Fisher-output controls. It reuses checked-in intervention records; no correctness label is used to choose a route.

## Route Summary
```text
           candidate_policy   n  rho_mean  abs_delta_entropy  abs_delta_varentropy  output_drift  target_success_rate  top10_preservation  top1_changed_rate  semantic_drift
equal_output_energy_control 210  0.777808           0.024568              0.045539          0.05             0.992754            0.929726           0.028571        0.001250
  gradient_orthogonal_route 210  0.777808           0.002641              0.016405          0.05             0.500000            0.927128           0.028571        0.001252
             gradient_route 210  0.729977           0.047841              0.077716          0.05             1.000000            0.957576           0.004762        0.001254
             high_rho_route 210  0.777808           0.048148              0.078786          0.05             1.000000            0.961039           0.009524        0.001254
              low_rho_route 210  0.429667           0.027510              0.050399          0.05             1.000000            0.939683           0.042857        0.001249
               random_route 210  0.624149           0.037249              0.063799          0.05             1.000000            0.952525           0.004762        0.001249
```

## High-Rho Contrasts
```text
                                     contrast               metric  high_rho  control  delta_high_minus_control
high_rho_route_vs_equal_output_energy_control    abs_delta_entropy  0.048148 0.024568                  0.023580
high_rho_route_vs_equal_output_energy_control abs_delta_varentropy  0.078786 0.045539                  0.033247
high_rho_route_vs_equal_output_energy_control         output_drift  0.050000 0.050000                  0.000000
high_rho_route_vs_equal_output_energy_control  target_success_rate  1.000000 0.992754                  0.007246
high_rho_route_vs_equal_output_energy_control   top10_preservation  0.961039 0.929726                  0.031313
high_rho_route_vs_equal_output_energy_control    top1_changed_rate  0.009524 0.028571                  0.019048
  high_rho_route_vs_gradient_orthogonal_route    abs_delta_entropy  0.048148 0.002641                  0.045507
  high_rho_route_vs_gradient_orthogonal_route abs_delta_varentropy  0.078786 0.016405                  0.062381
  high_rho_route_vs_gradient_orthogonal_route         output_drift  0.050000 0.050000                  0.000000
  high_rho_route_vs_gradient_orthogonal_route  target_success_rate  1.000000 0.500000                  0.500000
  high_rho_route_vs_gradient_orthogonal_route   top10_preservation  0.961039 0.927128                  0.033911
  high_rho_route_vs_gradient_orthogonal_route    top1_changed_rate  0.009524 0.028571                  0.019048
             high_rho_route_vs_gradient_route    abs_delta_entropy  0.048148 0.047841                  0.000307
             high_rho_route_vs_gradient_route abs_delta_varentropy  0.078786 0.077716                  0.001070
             high_rho_route_vs_gradient_route         output_drift  0.050000 0.050000                  0.000000
             high_rho_route_vs_gradient_route  target_success_rate  1.000000 1.000000                  0.000000
             high_rho_route_vs_gradient_route   top10_preservation  0.961039 0.957576                  0.003463
             high_rho_route_vs_gradient_route    top1_changed_rate  0.009524 0.004762                 -0.004762
              high_rho_route_vs_low_rho_route    abs_delta_entropy  0.048148 0.027510                  0.020638
              high_rho_route_vs_low_rho_route abs_delta_varentropy  0.078786 0.050399                  0.028387
              high_rho_route_vs_low_rho_route         output_drift  0.050000 0.050000                  0.000000
              high_rho_route_vs_low_rho_route  target_success_rate  1.000000 1.000000                  0.000000
              high_rho_route_vs_low_rho_route   top10_preservation  0.961039 0.939683                  0.021356
              high_rho_route_vs_low_rho_route    top1_changed_rate  0.009524 0.042857                  0.033333
               high_rho_route_vs_random_route    abs_delta_entropy  0.048148 0.037249                  0.010899
               high_rho_route_vs_random_route abs_delta_varentropy  0.078786 0.063799                  0.014987
               high_rho_route_vs_random_route         output_drift  0.050000 0.050000                  0.000000
               high_rho_route_vs_random_route  target_success_rate  1.000000 1.000000                  0.000000
               high_rho_route_vs_random_route   top10_preservation  0.961039 0.952525                  0.008514
               high_rho_route_vs_random_route    top1_changed_rate  0.009524 0.004762                 -0.004762
```

## Cluster Bootstrap CI
```text
                                     contrast               metric    ci_low    median  ci_high
high_rho_route_vs_equal_output_energy_control    abs_delta_entropy  0.021875  0.023572 0.025292
high_rho_route_vs_equal_output_energy_control abs_delta_varentropy  0.029014  0.033120 0.037412
high_rho_route_vs_equal_output_energy_control         output_drift  0.000000  0.000000 0.000000
high_rho_route_vs_equal_output_energy_control  target_success_rate  0.000000  0.007143 0.022901
high_rho_route_vs_equal_output_energy_control   top10_preservation  0.017601  0.031313 0.045455
high_rho_route_vs_equal_output_energy_control    top1_changed_rate  0.000000  0.019048 0.042857
  high_rho_route_vs_gradient_orthogonal_route    abs_delta_entropy  0.043123  0.045575 0.048062
  high_rho_route_vs_gradient_orthogonal_route abs_delta_varentropy  0.056518  0.062321 0.068934
  high_rho_route_vs_gradient_orthogonal_route         output_drift  0.000000  0.000000 0.000000
  high_rho_route_vs_gradient_orthogonal_route  target_success_rate  0.408707  0.503247 0.586521
  high_rho_route_vs_gradient_orthogonal_route   top10_preservation  0.019333  0.033911 0.048633
  high_rho_route_vs_gradient_orthogonal_route    top1_changed_rate  0.000000  0.019048 0.042857
             high_rho_route_vs_gradient_route    abs_delta_entropy -0.000364  0.000314 0.000929
             high_rho_route_vs_gradient_route abs_delta_varentropy -0.000718  0.001008 0.002930
             high_rho_route_vs_gradient_route         output_drift  0.000000  0.000000 0.000000
             high_rho_route_vs_gradient_route  target_success_rate  0.000000  0.000000 0.000000
             high_rho_route_vs_gradient_route   top10_preservation -0.005195  0.003463 0.012121
             high_rho_route_vs_gradient_route    top1_changed_rate -0.014286 -0.004762 0.000000
              high_rho_route_vs_low_rho_route    abs_delta_entropy  0.018052  0.020633 0.023307
              high_rho_route_vs_low_rho_route abs_delta_varentropy  0.023871  0.028397 0.033368
              high_rho_route_vs_low_rho_route         output_drift  0.000000  0.000000 0.000000
              high_rho_route_vs_low_rho_route  target_success_rate  0.000000  0.000000 0.000000
              high_rho_route_vs_low_rho_route   top10_preservation  0.006926  0.021645 0.036371
              high_rho_route_vs_low_rho_route    top1_changed_rate  0.009524  0.033333 0.057143
               high_rho_route_vs_random_route    abs_delta_entropy  0.008729  0.010942 0.013124
               high_rho_route_vs_random_route abs_delta_varentropy  0.010787  0.014951 0.019779
               high_rho_route_vs_random_route         output_drift  0.000000  0.000000 0.000000
               high_rho_route_vs_random_route  target_success_rate  0.000000  0.000000 0.000000
               high_rho_route_vs_random_route   top10_preservation -0.002900  0.008514 0.020635
               high_rho_route_vs_random_route    top1_changed_rate -0.014286 -0.004762 0.000000
```

## Minimal Energy To Target
```text
candidate_policy  n  minimal_energy_to_entropy_target  minimal_energy_to_varentropy_target  entropy_target_success_rate  varentropy_target_success_rate  semantic_top10_preservation  top1_changed_rate
  gradient_route 72                          0.983804                             1.491467                     0.972222                        0.833333                     0.957071           0.013889
  high_rho_route 72                          1.033399                             1.427363                     0.972222                        0.861111                     0.957071           0.013889
   low_rho_route 72                          1.040343                             1.669061                     0.972222                        0.777778                     0.944444           0.013889
    random_route 72                          1.105793                             2.001282                     0.972222                        0.777778                     0.946970           0.013889
```

## Files
```text
choose_route_records.csv
choose_route_summary.csv
choose_route_contrasts.csv
choose_route_bootstrap_ci.csv
choose_route_minimal_energy.csv
report.md
```
