# Control: Random-Init Versus Pretrained

This control compares the same architecture before and after learned weights. If pretrained and random-init rho structure diverges, accessibility is less likely to be a purely architectural/Jacobian degree-of-freedom artifact.

## Summary

```text
                            model  init_state split route_source subspace_family  subspace_k  layer  n  rho_mean  rho_std  v_access_mean  v_inaccess_mean  grad_entropy_proj_norm_mean  grad_varentropy_proj_norm_mean
                bert-base-uncased  pretrained   all random_route          random           8      4 21  0.451672 0.171720       0.343365         0.333010                     0.091953                        0.238247
                bert-base-uncased  pretrained   all random_route          random           8      8 21  0.551356 0.201390       0.800755         0.410460                     0.100076                        0.210047
                bert-base-uncased  pretrained   all random_route          random           8     12 21  0.720683 0.160560       1.321388         0.408809                     0.207436                        0.360525
                bert-base-uncased  pretrained   all within_model       delta_pca           8      4 21  0.492993 0.198884       0.384148         0.292227                     0.103464                        0.261654
                bert-base-uncased  pretrained   all within_model       delta_pca           8      8 21  0.487075 0.204939       0.733725         0.477490                     0.105726                        0.217738
                bert-base-uncased  pretrained   all within_model       state_pca           8      4 21  0.546467 0.137980       0.402357         0.274019                     0.120210                        0.301370
                bert-base-uncased  pretrained   all within_model       state_pca           8      8 21  0.619048 0.190297       0.871531         0.339684                     0.139131                        0.285537
                bert-base-uncased  pretrained   all within_model       state_pca           8     12 21  0.744718 0.146873       1.356299         0.373899                     0.258717                        0.462542
                bert-base-uncased random_init   all random_route          random           8      4 21  0.232474 0.072035       0.005915         0.020625                     0.001781                        0.004115
                bert-base-uncased random_init   all random_route          random           8      8 21  0.363138 0.138962       0.006599         0.011241                     0.001828                        0.004014
                bert-base-uncased random_init   all random_route          random           8     12 21  0.302254 0.096056       0.007278         0.017123                     0.001785                        0.003894
                bert-base-uncased random_init   all within_model       delta_pca           8      4 21  0.277835 0.082362       0.007654         0.018885                     0.001919                        0.004406
                bert-base-uncased random_init   all within_model       delta_pca           8      8 21  0.309550 0.169825       0.005647         0.012193                     0.001621                        0.003543
                bert-base-uncased random_init   all within_model       state_pca           8      4 21  0.322807 0.168627       0.009725         0.016815                     0.001959                        0.004491
                bert-base-uncased random_init   all within_model       state_pca           8      8 21  0.279628 0.105102       0.005121         0.012719                     0.001612                        0.003557
                bert-base-uncased random_init   all within_model       state_pca           8     12 21  0.238246 0.078693       0.005566         0.018835                     0.001476                        0.003232
google/bert_uncased_L-2_H-128_A-2  pretrained   all random_route          random           8      1 21  0.268713 0.081975       0.209942         0.553386                     0.088915                        0.198369
google/bert_uncased_L-2_H-128_A-2  pretrained   all random_route          random           8      2 21  0.474149 0.149319       0.517897         0.467696                     0.152723                        0.348499
google/bert_uncased_L-2_H-128_A-2  pretrained   all within_model       delta_pca           8      1 21  0.385786 0.102447       0.317805         0.445522                     0.109351                        0.257577
google/bert_uncased_L-2_H-128_A-2  pretrained   all within_model       state_pca           8      1 21  0.314628 0.130155       0.261439         0.501889                     0.070630                        0.149124
google/bert_uncased_L-2_H-128_A-2  pretrained   all within_model       state_pca           8      2 21  0.519171 0.179005       0.590323         0.395269                     0.218631                        0.476452
google/bert_uncased_L-2_H-128_A-2 random_init   all random_route          random           8      1 21  0.219535 0.071170       0.000968         0.003294                     0.000557                        0.001186
google/bert_uncased_L-2_H-128_A-2 random_init   all random_route          random           8      2 21  0.305349 0.055401       0.001342         0.002929                     0.000714                        0.001492
google/bert_uncased_L-2_H-128_A-2 random_init   all within_model       delta_pca           8      1 21  0.294146 0.123914       0.001257         0.003005                     0.000687                        0.001444
google/bert_uncased_L-2_H-128_A-2 random_init   all within_model       state_pca           8      1 21  0.246102 0.093723       0.001057         0.003205                     0.000627                        0.001329
google/bert_uncased_L-2_H-128_A-2 random_init   all within_model       state_pca           8      2 21  0.350504 0.151756       0.001507         0.002765                     0.000698                        0.001464
```

## Pretrained Vs Random Contrasts

```text
                            model route_source subspace_family  n_pretrained  n_random_init  rho_pretrained_mean  rho_random_init_mean  rho_pretrained_minus_random  rho_pretrained_over_random  rho_rank_spearman_pretrained_vs_random
                bert-base-uncased random_route          random            63             63             0.574570              0.299289                     0.275282                    1.919787                                0.161770
                bert-base-uncased within_model       delta_pca            42             42             0.490034              0.293692                     0.196342                    1.668529                               -0.024876
                bert-base-uncased within_model       state_pca            63             63             0.636744              0.280227                     0.356517                    2.272243                               -0.105271
google/bert_uncased_L-2_H-128_A-2 random_route          random            42             42             0.371431              0.262442                     0.108989                    1.415289                                0.319990
google/bert_uncased_L-2_H-128_A-2 within_model       delta_pca            21             21             0.385786              0.294146                     0.091639                    1.311543                               -0.654545
google/bert_uncased_L-2_H-128_A-2 within_model       state_pca            42             42             0.416900              0.298303                     0.118597                    1.397570                                0.219350
```

## Files

```text
prompt_tables.csv
random_init_scores.csv
random_init_summary.csv
pretrained_vs_random_contrasts.csv
report.md
```