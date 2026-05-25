# Test 1: Minimal Intervention Energy

This test asks the inverse controllability question: given a fixed target uncertainty movement, how much latent intervention energy is needed?

- Entropy target: `0.02`
- Varentropy target: `0.04`
- Main MLM lens: top-k `32`

Costs are estimated two ways: the main top-k 32 and decoder records use local-linear control costs from equal Fisher-output-energy interventions, while the original MLM full battery contributes an interpolated multi-epsilon grid estimate.

## Target Summary
```text
                    source subspace_family     sign    n  rho_mean  cost_entropy_tau_median  cost_varentropy_tau_median  entropy_target_success_rate  varentropy_target_success_rate  top1_changed_rate  top10_jaccard_mean
 decoder_main_local_linear       delta_pca decrease   36  0.778149                 1.137287                    2.001282                     0.972222                        0.750000           0.000000            0.939394
 decoder_main_local_linear       delta_pca increase   36  0.778149                 1.068478                    1.878311                     0.972222                        0.805556           0.027778            0.954545
 decoder_main_local_linear       state_pca decrease   36  0.810693                 0.979156                    1.532087                     0.972222                        0.833333           0.000000            0.954545
 decoder_main_local_linear       state_pca increase   36  0.810693                 0.960917                    1.424959                     0.972222                        0.888889           0.027778            0.954545
mlm_full_grid_interpolated       delta_pca decrease  792  0.521419                 0.391770                    0.441651                     0.797980                        0.753788           0.049242                 NaN
mlm_full_grid_interpolated       delta_pca increase  792  0.521419                 0.389057                    0.439547                     0.743687                        0.722222           0.083333                 NaN
mlm_full_grid_interpolated          random decrease 1584  0.511858                 0.405763                    0.466065                     0.821970                        0.773359           0.059343                 NaN
mlm_full_grid_interpolated          random increase 1584  0.511858                 0.402682                    0.463769                     0.745581                        0.740530           0.083965                 NaN
mlm_full_grid_interpolated       state_pca decrease  792  0.527729                 0.363338                    0.422065                     0.835859                        0.790404           0.041667                 NaN
mlm_full_grid_interpolated       state_pca increase  792  0.527729                 0.356638                    0.413634                     0.758838                        0.757576           0.087121                 NaN
   mlm_topk32_local_linear       delta_pca decrease  115  0.514965                 0.230321                    0.212062                     0.747826                        0.800000           0.017391            0.946772
   mlm_topk32_local_linear       delta_pca increase  115  0.514965                 0.251214                    0.224794                     0.669565                        0.747826           0.026087            0.951252
   mlm_topk32_local_linear          random decrease  184  0.504465                 0.244829                    0.259171                     0.788043                        0.777174           0.027174            0.953722
   mlm_topk32_local_linear          random increase  184  0.504465                 0.260037                    0.262109                     0.733696                        0.739130           0.016304            0.946146
   mlm_topk32_local_linear       state_pca decrease  184  0.530461                 0.233084                    0.256090                     0.815217                        0.755435           0.027174            0.955863
   mlm_topk32_local_linear       state_pca increase  184  0.530461                 0.242514                    0.259996                     0.750000                        0.777174           0.021739            0.948946
```

## Predictor Benchmark
```text
                    source                     outcome                 predictor    n  spearman_with_easy_score  auroc_easy_raw  auroc_easy_direction_adjusted
 decoder_main_local_linear    neg_log_cost_entropy_tau    grad_entropy_proj_norm  144                  0.928465        0.939429                       0.939429
 decoder_main_local_linear    neg_log_cost_entropy_tau    fisher_output_fro_norm  144                  0.821947        0.878086                       0.878086
 decoder_main_local_linear    neg_log_cost_entropy_tau                   entropy  144                  0.385912        0.648920                       0.648920
 decoder_main_local_linear    neg_log_cost_entropy_tau                varentropy  144                  0.193463        0.596451                       0.596451
 decoder_main_local_linear    neg_log_cost_entropy_tau                       rho  144                 -0.235915        0.437114                       0.562886
 decoder_main_local_linear    neg_log_cost_entropy_tau                confidence  144                 -0.353444        0.373457                       0.626543
 decoder_main_local_linear    neg_log_cost_entropy_tau       rho_random_adjusted    0                       NaN             NaN                            NaN
 decoder_main_local_linear    neg_log_cost_entropy_tau grad_varentropy_proj_norm    0                       NaN             NaN                            NaN
 decoder_main_local_linear    neg_log_cost_entropy_tau         jacobian_fro_norm    0                       NaN             NaN                            NaN
 decoder_main_local_linear neg_log_cost_varentropy_tau    grad_entropy_proj_norm  144                  0.837196        0.902006                       0.902006
 decoder_main_local_linear neg_log_cost_varentropy_tau    fisher_output_fro_norm  144                  0.826545        0.908565                       0.908565
 decoder_main_local_linear neg_log_cost_varentropy_tau                   entropy  144                  0.498500        0.732253                       0.732253
 decoder_main_local_linear neg_log_cost_varentropy_tau                varentropy  144                 -0.016082        0.512346                       0.512346
 decoder_main_local_linear neg_log_cost_varentropy_tau                       rho  144                 -0.299250        0.355710                       0.644290
 decoder_main_local_linear neg_log_cost_varentropy_tau                confidence  144                 -0.425409        0.304012                       0.695988
 decoder_main_local_linear neg_log_cost_varentropy_tau       rho_random_adjusted    0                       NaN             NaN                            NaN
 decoder_main_local_linear neg_log_cost_varentropy_tau grad_varentropy_proj_norm    0                       NaN             NaN                            NaN
 decoder_main_local_linear neg_log_cost_varentropy_tau         jacobian_fro_norm    0                       NaN             NaN                            NaN
mlm_full_grid_interpolated    neg_log_cost_entropy_tau    grad_entropy_proj_norm 4967                  0.792490        0.894583                       0.894583
mlm_full_grid_interpolated    neg_log_cost_entropy_tau                varentropy 4967                  0.572104        0.781060                       0.781060
mlm_full_grid_interpolated    neg_log_cost_entropy_tau    fisher_output_fro_norm 4967                  0.218863        0.600297                       0.600297
mlm_full_grid_interpolated    neg_log_cost_entropy_tau                       rho 4967                  0.213203        0.601240                       0.601240
mlm_full_grid_interpolated    neg_log_cost_entropy_tau                   entropy 4967                 -0.473813        0.262637                       0.737363
mlm_full_grid_interpolated    neg_log_cost_entropy_tau       rho_random_adjusted    0                       NaN             NaN                            NaN
mlm_full_grid_interpolated    neg_log_cost_entropy_tau                confidence    0                       NaN             NaN                            NaN
mlm_full_grid_interpolated    neg_log_cost_entropy_tau grad_varentropy_proj_norm    0                       NaN             NaN                            NaN
mlm_full_grid_interpolated    neg_log_cost_entropy_tau         jacobian_fro_norm    0                       NaN             NaN                            NaN
mlm_full_grid_interpolated neg_log_cost_varentropy_tau    grad_entropy_proj_norm 4793                  0.612026        0.803320                       0.803320
mlm_full_grid_interpolated neg_log_cost_varentropy_tau                varentropy 4793                  0.311411        0.658656                       0.658656
mlm_full_grid_interpolated neg_log_cost_varentropy_tau    fisher_output_fro_norm 4793                  0.243830        0.614274                       0.614274
mlm_full_grid_interpolated neg_log_cost_varentropy_tau                       rho 4793                  0.120591        0.569730                       0.569730
mlm_full_grid_interpolated neg_log_cost_varentropy_tau                   entropy 4793                 -0.225634        0.381620                       0.618380
mlm_full_grid_interpolated neg_log_cost_varentropy_tau       rho_random_adjusted    0                       NaN             NaN                            NaN
mlm_full_grid_interpolated neg_log_cost_varentropy_tau                confidence    0                       NaN             NaN                            NaN
mlm_full_grid_interpolated neg_log_cost_varentropy_tau grad_varentropy_proj_norm    0                       NaN             NaN                            NaN
mlm_full_grid_interpolated neg_log_cost_varentropy_tau         jacobian_fro_norm    0                       NaN             NaN                            NaN
   mlm_topk32_local_linear    neg_log_cost_entropy_tau    grad_entropy_proj_norm  966                  0.952105        0.980768                       0.980768
   mlm_topk32_local_linear    neg_log_cost_entropy_tau grad_varentropy_proj_norm  966                  0.903000        0.963948                       0.963948
   mlm_topk32_local_linear    neg_log_cost_entropy_tau                       rho  966                  0.757127        0.872345                       0.872345
   mlm_topk32_local_linear    neg_log_cost_entropy_tau                confidence  966                  0.740733        0.866674                       0.866674
   mlm_topk32_local_linear    neg_log_cost_entropy_tau                varentropy  966                  0.724665        0.860047                       0.860047
   mlm_topk32_local_linear    neg_log_cost_entropy_tau         jacobian_fro_norm  966                  0.362446        0.673373                       0.673373
   mlm_topk32_local_linear    neg_log_cost_entropy_tau    fisher_output_fro_norm  966                  0.305349        0.647662                       0.647662
   mlm_topk32_local_linear    neg_log_cost_entropy_tau       rho_random_adjusted  966                  0.186534        0.596717                       0.596717
   mlm_topk32_local_linear    neg_log_cost_entropy_tau                   entropy  966                 -0.672768        0.167361                       0.832639
   mlm_topk32_local_linear neg_log_cost_varentropy_tau grad_varentropy_proj_norm  966                  0.874921        0.940235                       0.940235
   mlm_topk32_local_linear neg_log_cost_varentropy_tau    grad_entropy_proj_norm  966                  0.774702        0.886816                       0.886816
   mlm_topk32_local_linear neg_log_cost_varentropy_tau                       rho  966                  0.603605        0.785997                       0.785997
   mlm_topk32_local_linear neg_log_cost_varentropy_tau                confidence  966                  0.533756        0.764899                       0.764899
   mlm_topk32_local_linear neg_log_cost_varentropy_tau                varentropy  966                  0.475140        0.731670                       0.731670
   mlm_topk32_local_linear neg_log_cost_varentropy_tau    fisher_output_fro_norm  966                  0.302907        0.650474                       0.650474
   mlm_topk32_local_linear neg_log_cost_varentropy_tau         jacobian_fro_norm  966                  0.235165        0.623538                       0.623538
   mlm_topk32_local_linear neg_log_cost_varentropy_tau       rho_random_adjusted  966                  0.161706        0.584299                       0.584299
   mlm_topk32_local_linear neg_log_cost_varentropy_tau                   entropy  966                 -0.392851        0.308126                       0.691874
```

## Residual Rho Effects
```text
                    source                     outcome    n  rho_residual_spearman  rho_beta_standardized                                                                                                                                                                                  controls
 decoder_main_local_linear    neg_log_cost_entropy_tau  144               0.591409               0.199281                                                                         entropy,varentropy,confidence,grad_entropy_proj_norm,fisher_output_fro_norm,model,task,layer,subspace_family,sign
 decoder_main_local_linear neg_log_cost_varentropy_tau  144               0.353974               0.187757                                                                         entropy,varentropy,confidence,grad_entropy_proj_norm,fisher_output_fro_norm,model,task,layer,subspace_family,sign
mlm_full_grid_interpolated    neg_log_cost_entropy_tau 4967              -0.008649              -0.017827                                                               entropy,varentropy,grad_entropy_proj_norm,fisher_output_fro_norm,trace_fim,logit_norm,model,task,layer,subspace_family,sign
mlm_full_grid_interpolated neg_log_cost_varentropy_tau 4793              -0.084242              -0.118466                                                               entropy,varentropy,grad_entropy_proj_norm,fisher_output_fro_norm,trace_fim,logit_norm,model,task,layer,subspace_family,sign
   mlm_topk32_local_linear    neg_log_cost_entropy_tau  966               0.586278               0.433678 entropy,varentropy,confidence,margin,grad_entropy_proj_norm,grad_varentropy_proj_norm,fisher_output_fro_norm,jacobian_fro_norm,trace_fim,logit_norm,model,task,layer,subspace_family,sign
   mlm_topk32_local_linear neg_log_cost_varentropy_tau  966               0.406130               0.301277 entropy,varentropy,confidence,margin,grad_entropy_proj_norm,grad_varentropy_proj_norm,fisher_output_fro_norm,jacobian_fro_norm,trace_fim,logit_norm,model,task,layer,subspace_family,sign
```

## Files
```text
minimal_energy_cost_records.csv
minimal_energy_predictor_benchmark.csv
minimal_energy_residual_effects.csv
minimal_energy_target_summary.csv
baseline_availability.csv
report.md
```
