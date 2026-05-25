# Control: Direct Gradient Baselines

This control compares `rho(B)` against direct local gradient and Jacobian baselines.

## Predictors

- `||Pi_B grad_z H||`
- `||Pi_B grad_z Var||`
- `||F^{1/2} J B||`
- `||J B||`

## Correlations

```text
         subset              outcome                  predictor     n  spearman   pearson
accessible_only    abs_delta_entropy                        rho  4830  0.627714  0.616779
accessible_only    abs_delta_entropy    grad_entropy_projection  4830  0.882700  0.815946
accessible_only    abs_delta_entropy grad_varentropy_projection  4830  0.787023  0.656793
accessible_only    abs_delta_entropy         fisher_output_norm  4830 -0.046586 -0.060333
accessible_only    abs_delta_entropy          jacobian_fro_norm  4830  0.159971  0.214205
accessible_only    abs_delta_entropy                  trace_fim  4830 -0.046586 -0.028348
accessible_only abs_delta_varentropy                        rho  4830  0.354401  0.346335
accessible_only abs_delta_varentropy    grad_entropy_projection  4830  0.681537  0.558505
accessible_only abs_delta_varentropy grad_varentropy_projection  4830  0.812691  0.768864
accessible_only abs_delta_varentropy         fisher_output_norm  4830 -0.046211 -0.049754
accessible_only abs_delta_varentropy          jacobian_fro_norm  4830  0.185262  0.188532
accessible_only abs_delta_varentropy                  trace_fim  4830 -0.046211 -0.018736
accessible_only  directional_success                        rho  4830       NaN       NaN
accessible_only  directional_success    grad_entropy_projection  4830       NaN       NaN
accessible_only  directional_success grad_varentropy_projection  4830       NaN       NaN
accessible_only  directional_success         fisher_output_norm  4830       NaN       NaN
accessible_only  directional_success          jacobian_fro_norm  4830       NaN       NaN
accessible_only  directional_success                  trace_fim  4830       NaN       NaN
 all_directions    abs_delta_entropy                        rho 14490  0.208806  0.321026
 all_directions    abs_delta_entropy    grad_entropy_projection 14490  0.264269  0.425854
 all_directions    abs_delta_entropy grad_varentropy_projection 14490  0.220357  0.337029
 all_directions    abs_delta_entropy         fisher_output_norm 14490 -0.018640 -0.033635
 all_directions    abs_delta_entropy          jacobian_fro_norm 14490  0.041655  0.096840
 all_directions    abs_delta_entropy                  trace_fim 14490 -0.018640 -0.015876
 all_directions abs_delta_varentropy                        rho 14490  0.211962  0.238826
 all_directions abs_delta_varentropy    grad_entropy_projection 14490  0.362111  0.371699
 all_directions abs_delta_varentropy grad_varentropy_projection 14490  0.380423  0.453369
 all_directions abs_delta_varentropy         fisher_output_norm 14490 -0.015424 -0.026712
 all_directions abs_delta_varentropy          jacobian_fro_norm 14490  0.103170  0.124654
 all_directions abs_delta_varentropy                  trace_fim 14490 -0.015424 -0.007628
 all_directions  directional_success                        rho 14490  0.012798  0.012892
 all_directions  directional_success    grad_entropy_projection 14490  0.019658  0.014537
 all_directions  directional_success grad_varentropy_projection 14490  0.020287  0.015710
 all_directions  directional_success         fisher_output_norm 14490 -0.003372 -0.002874
 all_directions  directional_success          jacobian_fro_norm 14490  0.000190  0.001012
 all_directions  directional_success                  trace_fim 14490 -0.003372 -0.001802
```

## Files

```text
gradient_baseline_scores.csv
gradient_baseline_correlations.csv
report.md
```