# Application 4: Uncertainty Circuits

This application asks whether high-accessibility layer/subspace routes are causally better control points for uncertainty than low-accessibility, random, or gradient-selected alternatives.

All interventions are Fisher-output-normalized to the same `||F^{1/2}J delta z||` budget.

## Skipped Models
```text
(empty)
```

## Route Summary
```text
               route_role      direction_kind     sign   n  rho_mean  abs_delta_entropy_mean  abs_delta_varentropy_mean  selected_top1_changed_rate  target_correct_changed_rate  full_vocab_top10_jaccard_mean  candidate_kl_mean
   entropy_gradient_route    entropy_gradient decrease 144  0.587924                0.033779                   0.069905                    0.041667                     0.000000                       0.946970           0.001240
   entropy_gradient_route    entropy_gradient increase 144  0.587924                0.031651                   0.066976                    0.041667                     0.006944                       0.958754           0.001260
           high_rho_route          accessible decrease 144  0.598556                0.038484                   0.076382                    0.027778                     0.000000                       0.947391           0.001213
           high_rho_route          accessible increase 144  0.598556                0.038154                   0.078584                    0.013889                     0.000000                       0.953493           0.001306
            low_rho_route          accessible decrease 144  0.515234                0.035773                   0.070721                    0.020833                     0.000000                       0.953704           0.001210
            low_rho_route          accessible increase 144  0.515234                0.035667                   0.073396                    0.013889                     0.000000                       0.952231           0.001309
             random_route          accessible decrease 144  0.548149                0.037121                   0.073188                    0.027778                     0.000000                       0.948443           0.001224
             random_route          accessible increase 144  0.548149                0.036207                   0.073913                    0.027778                     0.000000                       0.960227           0.001278
varentropy_gradient_route varentropy_gradient decrease 144  0.588266                0.032469                   0.072975                    0.034722                     0.000000                       0.942340           0.001239
varentropy_gradient_route varentropy_gradient increase 144  0.588266                0.030394                   0.070199                    0.055556                     0.006944                       0.960017           0.001264
```

## High-Rho Route Contrasts
```text
                                 comparison               metric   n  high_minus_control_mean  high_over_control_mean  high_better_rate
            high_rho_route_vs_low_rho_route    abs_delta_entropy 288                 0.002598                1.111919          0.590278
            high_rho_route_vs_low_rho_route abs_delta_varentropy 288                 0.005424                1.115032          0.548611
             high_rho_route_vs_random_route    abs_delta_entropy 288                 0.001655                1.075693          0.618056
             high_rho_route_vs_random_route abs_delta_varentropy 288                 0.003933                1.093379          0.604167
   high_rho_route_vs_entropy_gradient_route    abs_delta_entropy 288                 0.005603                1.214681          1.000000
   high_rho_route_vs_entropy_gradient_route abs_delta_varentropy 288                 0.009042                2.052477          0.878472
high_rho_route_vs_varentropy_gradient_route    abs_delta_entropy 288                 0.006887                1.579855          1.000000
high_rho_route_vs_varentropy_gradient_route abs_delta_varentropy 288                 0.005896                1.104887          0.850694
```

## Rho-Causal Effect Correlations
```text
                subset               outcome    n  spearman
                   all     abs_delta_entropy 1440  0.804097
                   all  abs_delta_varentropy 1440  0.724136
                   all selected_top1_changed 1440 -0.142744
                   all      candidate_set_kl 1440  0.001320
accessible_routes_only     abs_delta_entropy  864  0.838909
accessible_routes_only  abs_delta_varentropy  864  0.737336
accessible_routes_only selected_top1_changed  864 -0.149195
accessible_routes_only      candidate_set_kl  864  0.000953
```

## Files
```text
route_scores.csv
circuit_intervention_records.csv
circuit_route_summary.csv
circuit_route_contrasts.csv
rho_causal_effect_correlations.csv
skipped_models.csv
report.md
```
