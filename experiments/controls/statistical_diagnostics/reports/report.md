# Statistical And Diagnostic Package

This package adds uncertainty estimates and paper-facing diagnostics on top of the curated experiment outputs.

## Confirmatory Analyses

- Bootstrap CIs for accessible-vs-control steering contrasts.
- Bootstrap CIs for rho-quartile movement contrasts.
- Bootstrap CIs for top-k robustness of rho rankings.
- Same scalar uncertainty plus same projected-gradient controls but different rho.
- Answer-preserving qualitative steering examples in Experiment 4.

## Exploratory / Diagnostic Analyses

- Residual rho effects after scalar, gradient, and geometric controls.
- Prompt duplicate and near-duplicate audit.
- Failure-mode table for high-rho/low-movement, low-rho/high-movement, top-k instability, and answer degradation.

## Key Tables

### Steering Contrasts

```text
                    analysis  top_k_output     sign                 control            metric   n  accessible_mean  control_mean  mean_difference  mean_difference_ci_low  mean_difference_ci_high     ratio  ratio_ci_low  ratio_ci_high  cohen_d_paired  cohen_d_ci_low  cohen_d_ci_high  standardized_mean_difference  standardized_mean_difference_ci_low  standardized_mean_difference_ci_high
steering_equal_fisher_output            16 decrease          random_control abs_delta_entropy 483         0.031961      0.013521         0.018439                0.017486                 0.019422  2.363744      2.224365       2.529811        1.687397        1.597595         1.801447                      1.687397                             1.597595                              1.801447
steering_equal_fisher_output            16 decrease grad_orthogonal_control abs_delta_entropy 483         0.031961      0.001056         0.030904                0.029640                 0.032166 30.259339     27.513786      32.993689        2.185765        2.091157         2.297922                      2.185765                             2.091157                              2.297922
steering_equal_fisher_output            16 increase          random_control abs_delta_entropy 483         0.031087      0.012359         0.018728                0.017748                 0.019783  2.515245      2.349279       2.695153        1.677980        1.563185         1.811641                      1.677980                             1.563185                              1.811641
steering_equal_fisher_output            16 increase grad_orthogonal_control abs_delta_entropy 483         0.031087      0.001057         0.030030                0.028695                 0.031490 29.415667     26.814298      32.028106        1.860046        1.758501         1.974313                      1.860046                             1.758501                              1.974313
steering_equal_fisher_output            32 decrease          random_control abs_delta_entropy 483         0.034137      0.013666         0.020471                0.019398                 0.021593  2.498001      2.352230       2.658297        1.622122        1.518809         1.742263                      1.622122                             1.518809                              1.742263
steering_equal_fisher_output            32 decrease grad_orthogonal_control abs_delta_entropy 483         0.034137      0.001122         0.033015                0.031628                 0.034550 30.429240     27.316724      34.063264        2.017944        1.920420         2.118357                      2.017944                             1.920420                              2.118357
steering_equal_fisher_output            32 increase          random_control abs_delta_entropy 483         0.032931      0.012605         0.020326                0.019227                 0.021481  2.612513      2.445649       2.818439        1.623483        1.527250         1.731382                      1.623483                             1.527250                              1.731382
steering_equal_fisher_output            32 increase grad_orthogonal_control abs_delta_entropy 483         0.032931      0.001130         0.031800                0.030164                 0.033319 29.132231     26.374883      32.386289        1.780983        1.697581         1.877269                      1.780983                             1.697581                              1.877269
steering_equal_fisher_output            64 decrease          random_control abs_delta_entropy 483         0.036298      0.013630         0.022668                0.021476                 0.023981  2.663050      2.489990       2.857760        1.539982        1.449993         1.655590                      1.539982                             1.449993                              1.655590
steering_equal_fisher_output            64 decrease grad_orthogonal_control abs_delta_entropy 483         0.036298      0.001064         0.035234                0.033607                 0.036894 34.119908     31.051802      37.423942        1.894854        1.805807         2.005416                      1.894854                             1.805807                              2.005416
steering_equal_fisher_output            64 increase          random_control abs_delta_entropy 483         0.035052      0.012637         0.022414                0.021201                 0.023740  2.773694      2.573218       2.987181        1.521625        1.428915         1.632281                      1.521625                             1.428915                              1.632281
steering_equal_fisher_output            64 increase grad_orthogonal_control abs_delta_entropy 483         0.035052      0.001071         0.033981                0.032386                 0.035829 32.726519     29.816946      36.251388        1.696600        1.613577         1.793105                      1.696600                             1.613577                              1.793105
steering_equal_fisher_output           128 decrease          random_control abs_delta_entropy 483         0.038357      0.014636         0.023721                0.022334                 0.025028  2.620768      2.449429       2.831353        1.523976        1.446734         1.618851                      1.523976                             1.446734                              1.618851
steering_equal_fisher_output           128 decrease grad_orthogonal_control abs_delta_entropy 483         0.038357      0.001116         0.037240                0.035398                 0.039146 34.364891     31.248998      37.818084        1.810903        1.719600         1.919085                      1.810903                             1.719600                              1.919085
steering_equal_fisher_output           128 increase          random_control abs_delta_entropy 483         0.037125      0.013799         0.023326                0.021979                 0.024767  2.690321      2.512599       2.898970        1.500448        1.420905         1.609443                      1.500448                             1.420905                              1.609443
steering_equal_fisher_output           128 increase grad_orthogonal_control abs_delta_entropy 483         0.037125      0.001111         0.036014                0.034111                 0.038002 33.423483     30.300750      36.833188        1.635255        1.549342         1.734377                      1.635255                             1.549342                              1.734377
```

### Rho Quartiles

```text
                analysis  epsilon               metric  n_high  n_low  high_mean  low_mean  mean_difference  mean_difference_ci_low  mean_difference_ci_high     ratio  ratio_ci_low  ratio_ci_high  cohen_d  cohen_d_ci_low  cohen_d_ci_high  standardized_mean_difference  standardized_mean_difference_ci_low  standardized_mean_difference_ci_high
rho_quartile_q4_minus_q1     0.25    abs_delta_entropy    1439   1440   0.039740  0.001262         0.038477                0.037095                 0.039728 31.481269     29.734898      33.472137 2.062002        1.982187         2.147255                      2.062002                             1.982187                              2.147255
rho_quartile_q4_minus_q1     0.25 abs_delta_varentropy    1439   1440   0.072535  0.004763         0.067773                0.064985                 0.070824 15.230545     14.089123      16.657581 1.631307        1.562902         1.711990                      1.631307                             1.562902                              1.711990
rho_quartile_q4_minus_q1     0.50    abs_delta_entropy    1439   1440   0.077620  0.003045         0.074575                0.072062                 0.077317 25.493782     23.930288      27.131849 2.034833        1.959845         2.113238                      2.034833                             1.959845                              2.113238
rho_quartile_q4_minus_q1     0.50 abs_delta_varentropy    1439   1440   0.143036  0.010432         0.132604                0.126880                 0.138358 13.711678     12.699555      14.894308 1.665491        1.596509         1.741734                      1.665491                             1.596509                              1.741734
rho_quartile_q4_minus_q1     1.00    abs_delta_entropy    1439   1440   0.146871  0.010812         0.136059                0.130696                 0.141255 13.584012     12.716649      14.596231 1.902123        1.829015         1.983121                      1.902123                             1.829015                              1.983121
rho_quartile_q4_minus_q1     1.00 abs_delta_varentropy    1439   1440   0.274644  0.029146         0.245498                0.234896                 0.256823  9.422957      8.793115      10.172588 1.674835        1.612933         1.744324                      1.674835                             1.612933                              1.744324
```

### Top-k Robustness

```text
                 analysis  top_k_output   n  rho_spearman  rho_spearman_ci_low  rho_spearman_ci_high  rho_abs_diff_mean  rho_abs_diff_ci_low  rho_abs_diff_ci_high
topk_rank_stability_vs_32            16 483      0.766992             0.723113              0.809435           0.183136             0.172248              0.195095
topk_rank_stability_vs_32            64 483      0.913329             0.894702              0.929967           0.099870             0.093394              0.106535
topk_rank_stability_vs_32           128 483      0.843911             0.810675              0.872087           0.159003             0.150059              0.167964
topk_rank_stability_vs_32           256 483      0.788191             0.745186              0.826858           0.194283             0.183953              0.203726
```

### Residual Effects

```text
                                                    analysis              outcome     n  spearman_residual_effect  spearman_ci_low  spearman_ci_high                                                                                                                                                                                        controls
residual_rho_effect_after_scalar_gradient_geometric_controls    abs_delta_entropy 14490                  0.136562         0.120657          0.152241 entropy,varentropy,confidence,margin,jacobian_fro_norm,fisher_output_energy,grad_entropy_proj_norm,grad_varentropy_proj_norm,top_k_output,layer,model,task,topic,subspace_family,direction,sign
residual_rho_effect_after_scalar_gradient_geometric_controls abs_delta_varentropy 14490                  0.144354         0.128572          0.160865 entropy,varentropy,confidence,margin,jacobian_fro_norm,fisher_output_energy,grad_entropy_proj_norm,grad_varentropy_proj_norm,top_k_output,layer,model,task,topic,subspace_family,direction,sign
```

### Same Scalar + Same Projected Gradients + Different Rho

```text
 n_pairs  rho_abs_diff_mean  rho_abs_diff_max  match_distance_z_median  entropy_abs_diff_median  varentropy_abs_diff_median  grad_entropy_proj_norm_abs_diff_median  grad_varentropy_proj_norm_abs_diff_median
      86           0.248692          0.493507                 0.315163                      0.0                         0.0                                0.027088                                   0.052638
```

### Prompt Duplicate Audit

```text
 n_prompt_rows  n_unique_exact_prompts  n_unique_normalized_prompts  n_exact_duplicate_rows  n_normalized_duplicate_rows  n_exact_duplicate_rows_within_source  n_normalized_duplicate_rows_within_source  n_exact_duplicate_groups  n_normalized_duplicate_groups  n_near_duplicate_pairs  near_duplicate_threshold
          1779                     342                          342                    1777                         1777                                  1128                                       1128                       340                            340                     103                      0.92
```

### Failure Modes

```text
              failure_type  n
  rho_unstable_across_topk 80
         top10_degradation 40
              top1_changed 22
high_rho_low_delta_entropy  1
```

## Claim Boundary

`rho` should not be described as beating direct projected gradients as a raw infinitesimal predictor. The supported claim is narrower and cleaner: `rho` is a geometric accessibility coefficient that decomposes varentropy with respect to an internal route, remains informative after controls, and identifies directions where uncertainty can be moved while preserving the answer neighborhood.