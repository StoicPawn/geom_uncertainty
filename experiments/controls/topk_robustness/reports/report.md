# Control: Top-k Robustness

This control repeats the geometry and steering summaries for top-k output lenses `16, 32, 64, 128, 256`.

## Coverage

- Score rows: `2415`
- Steering rows: `14490`
- Models: `bert-base-uncased, distilbert-base-uncased, google/bert_uncased_L-2_H-128_A-2`
- Top-k values: `16, 32, 64, 128, 256`

## Rank Stability

```text
 top_k_output  reference_top_k   n  rho_spearman_vs_ref  rho_abs_diff_mean_vs_ref
           16               32 483             0.766992                  0.183136
           32               32 483             1.000000                  0.000000
           64               32 483             0.913329                  0.099870
          128               32 483             0.843911                  0.159003
          256               32 483             0.788191                  0.194283
```

## Layer-Trend Stability

```text
 top_k_output                             model subspace_family  subspace_k  n_layers  layer_trend_spearman_vs_ref  layer_mean_abs_diff_vs_ref
           16                 bert-base-uncased       delta_pca           8         2                          NaN                    0.160182
           16                 bert-base-uncased          random           8         3                          1.0                    0.163220
           16                 bert-base-uncased       state_pca           8         3                          1.0                    0.145563
           16           distilbert-base-uncased       delta_pca           8         2                          NaN                    0.206168
           16           distilbert-base-uncased          random           8         3                          0.5                    0.193403
           16           distilbert-base-uncased       state_pca           8         3                          1.0                    0.200813
           16 google/bert_uncased_L-2_H-128_A-2       delta_pca           8         1                          NaN                    0.184116
           16 google/bert_uncased_L-2_H-128_A-2          random           8         2                          NaN                    0.178511
           16 google/bert_uncased_L-2_H-128_A-2       state_pca           8         2                          NaN                    0.195436
           32                 bert-base-uncased       delta_pca           8         2                          NaN                    0.000000
           32                 bert-base-uncased          random           8         3                          1.0                    0.000000
           32                 bert-base-uncased       state_pca           8         3                          1.0                    0.000000
           32           distilbert-base-uncased       delta_pca           8         2                          NaN                    0.000000
           32           distilbert-base-uncased          random           8         3                          1.0                    0.000000
           32           distilbert-base-uncased       state_pca           8         3                          1.0                    0.000000
           32 google/bert_uncased_L-2_H-128_A-2       delta_pca           8         1                          NaN                    0.000000
           32 google/bert_uncased_L-2_H-128_A-2          random           8         2                          NaN                    0.000000
           32 google/bert_uncased_L-2_H-128_A-2       state_pca           8         2                          NaN                    0.000000
           64                 bert-base-uncased       delta_pca           8         2                          NaN                    0.091689
           64                 bert-base-uncased          random           8         3                          1.0                    0.092263
           64                 bert-base-uncased       state_pca           8         3                          1.0                    0.074062
           64           distilbert-base-uncased       delta_pca           8         2                          NaN                    0.141955
           64           distilbert-base-uncased          random           8         3                          1.0                    0.104893
           64           distilbert-base-uncased       state_pca           8         3                          1.0                    0.099824
           64 google/bert_uncased_L-2_H-128_A-2       delta_pca           8         1                          NaN                    0.045958
           64 google/bert_uncased_L-2_H-128_A-2          random           8         2                          NaN                    0.087554
           64 google/bert_uncased_L-2_H-128_A-2       state_pca           8         2                          NaN                    0.090251
          128                 bert-base-uncased       delta_pca           8         2                          NaN                    0.154824
          128                 bert-base-uncased          random           8         3                          1.0                    0.160542
          128                 bert-base-uncased       state_pca           8         3                          1.0                    0.124807
          128           distilbert-base-uncased       delta_pca           8         2                          NaN                    0.220849
          128           distilbert-base-uncased          random           8         3                          1.0                    0.181884
          128           distilbert-base-uncased       state_pca           8         3                          1.0                    0.154816
          128 google/bert_uncased_L-2_H-128_A-2       delta_pca           8         1                          NaN                    0.068164
          128 google/bert_uncased_L-2_H-128_A-2          random           8         2                          NaN                    0.139340
          128 google/bert_uncased_L-2_H-128_A-2       state_pca           8         2                          NaN                    0.142647
          256                 bert-base-uncased       delta_pca           8         2                          NaN                    0.197628
          256                 bert-base-uncased          random           8         3                          1.0                    0.204939
          256                 bert-base-uncased       state_pca           8         3                          1.0                    0.153081
          256           distilbert-base-uncased       delta_pca           8         2                          NaN                    0.271003
          256           distilbert-base-uncased          random           8         3                          1.0                    0.225520
          256           distilbert-base-uncased       state_pca           8         3                          1.0                    0.186489
          256 google/bert_uncased_L-2_H-128_A-2       delta_pca           8         1                          NaN                    0.076018
          256 google/bert_uncased_L-2_H-128_A-2          random           8         2                          NaN                    0.159670
          256 google/bert_uncased_L-2_H-128_A-2       state_pca           8         2                          NaN                    0.168483
```

## Rho Summary

```text
 top_k_output                             model subspace_family  subspace_k  n  rho_mean  rho_std  entropy_mean  varentropy_mean  grad_entropy_proj_norm_mean  grad_varentropy_proj_norm_mean  fisher_output_fro_norm_mean
           16                 bert-base-uncased       delta_pca           8 46  0.684525 0.189608      2.391611         0.735132                     0.100298                        0.198135                     0.329688
           16                 bert-base-uncased          random           8 69  0.722026 0.163526      2.276289         0.864498                     0.122473                        0.223022                     0.326038
           16                 bert-base-uncased       state_pca           8 69  0.750161 0.163706      2.276289         0.864498                     0.166803                        0.307941                     0.428229
           16           distilbert-base-uncased       delta_pca           8 46  0.705948 0.147420      2.572672         0.431548                     0.107592                        0.242585                     0.459116
           16           distilbert-base-uncased          random           8 69  0.681105 0.166016      2.467542         0.549484                     0.114528                        0.241725                     0.447656
           16           distilbert-base-uncased       state_pca           8 69  0.727858 0.134108      2.467542         0.549484                     0.149280                        0.314388                     0.525839
           16 google/bert_uncased_L-2_H-128_A-2       delta_pca           8 23  0.710696 0.131939      2.558281         0.420482                     0.149569                        0.316671                     0.618138
           16 google/bert_uncased_L-2_H-128_A-2          random           8 46  0.626607 0.173203      2.472137         0.575092                     0.107950                        0.220248                     0.467489
           16 google/bert_uncased_L-2_H-128_A-2       state_pca           8 46  0.619814 0.187591      2.472137         0.575092                     0.117041                        0.218344                     0.461707
           32                 bert-base-uncased       delta_pca           8 46  0.524343 0.215052      2.964650         1.053327                     0.112548                        0.240983                     0.338921
           32                 bert-base-uncased          random           8 69  0.558806 0.199720      2.787751         1.230536                     0.135408                        0.264843                     0.339510
           32                 bert-base-uncased       state_pca           8 69  0.604598 0.204822      2.787751         1.230536                     0.186288                        0.364971                     0.440262
           32           distilbert-base-uncased       delta_pca           8 46  0.499780 0.186864      3.208789         0.597679                     0.110707                        0.275946                     0.468365
           32           distilbert-base-uncased          random           8 69  0.487703 0.178199      3.043909         0.778958                     0.122098                        0.278966                     0.459931
           32           distilbert-base-uncased       state_pca           8 69  0.527045 0.188586      3.043909         0.778958                     0.161799                        0.374719                     0.542545
           32 google/bert_uncased_L-2_H-128_A-2       delta_pca           8 23  0.526580 0.117376      3.101061         0.737350                     0.170252                        0.383507                     0.628896
           32 google/bert_uncased_L-2_H-128_A-2          random           8 46  0.448096 0.159066      3.025464         0.902894                     0.120684                        0.265006                     0.487520
           32 google/bert_uncased_L-2_H-128_A-2       state_pca           8 46  0.424379 0.163338      3.025464         0.902894                     0.135336                        0.266222                     0.496428
           64                 bert-base-uncased       delta_pca           8 46  0.432654 0.217995      3.543672         1.404650                     0.122935                        0.287844                     0.346206
           64                 bert-base-uncased          random           8 69  0.466543 0.212838      3.293360         1.635033                     0.147350                        0.311251                     0.347439
           64                 bert-base-uncased       state_pca           8 69  0.530536 0.198566      3.293360         1.635033                     0.204048                        0.433043                     0.447604
           64           distilbert-base-uncased       delta_pca           8 46  0.357825 0.197259      3.844129         0.777562                     0.108469                        0.302445                     0.472120
           64           distilbert-base-uncased          random           8 69  0.382810 0.183302      3.605102         1.037189                     0.130247                        0.315512                     0.467790
           64           distilbert-base-uncased       state_pca           8 69  0.427221 0.207300      3.605102         1.037189                     0.171311                        0.432139                     0.551929
           64 google/bert_uncased_L-2_H-128_A-2       delta_pca           8 23  0.480622 0.115369      3.620369         1.134944                     0.196890                        0.460801                     0.634625
           64 google/bert_uncased_L-2_H-128_A-2          random           8 46  0.360541 0.128983      3.555668         1.303791                     0.139595                        0.314416                     0.505211
           64 google/bert_uncased_L-2_H-128_A-2       state_pca           8 46  0.334127 0.153912      3.555668         1.303791                     0.158064                        0.319407                     0.521744
          128                 bert-base-uncased       delta_pca           8 46  0.369519 0.218551      4.122463         1.788223                     0.130184                        0.333876                     0.350621
          128                 bert-base-uncased          random           8 69  0.398264 0.214713      3.788156         2.076752                     0.155677                        0.355477                     0.353069
          128                 bert-base-uncased       state_pca           8 69  0.479790 0.206658      3.788156         2.076752                     0.219337                        0.500722                     0.453309
```

## Steering Summary

```text
 top_k_output               direction     sign   n  abs_delta_entropy_mean  abs_delta_varentropy_mean  directional_success_rate  selected_top1_changed_rate  selected_top5_jaccard_mean  full_vocab_top5_jaccard_mean  full_vocab_top10_jaccard_mean  selected_kl_mean  candidate_set_kl_mean  selected_prob_l1_mean  cluster_distribution_l1_mean  embedding_centroid_delta_norm_mean  candidate_mass_retention_ratio_mean
           16           accessible_ls decrease 483                0.031961                   0.056365                  1.000000                    0.014493                    0.971705                      0.971705                       0.950562          0.001219               0.001219               0.040820                      0.025785                            0.015190                             1.022006
           16           accessible_ls increase 483                0.031087                   0.055404                  1.000000                    0.012422                    0.964310                      0.964310                       0.948868          0.001286               0.001286               0.042069                      0.026382                            0.015534                             0.971759
           16 grad_orthogonal_control decrease 483                0.001056                   0.006627                  0.799172                    0.043478                    0.936015                      0.936015                       0.924245          0.001250               0.001250               0.038278                      0.020253                            0.012389                             0.999539
           16 grad_orthogonal_control increase 483                0.001057                   0.006575                  0.213251                    0.051760                    0.930494                      0.930494                       0.917937          0.001250               0.001250               0.038217                      0.020204                            0.012366                             0.996692
           16          random_control decrease 483                0.013521                   0.025047                  0.993789                    0.047619                    0.946860                      0.946860                       0.931489          0.001237               0.001237               0.039620                      0.023194                            0.013697                             1.006558
           16          random_control increase 483                0.012359                   0.023682                  0.923395                    0.047619                    0.936508                      0.936508                       0.932681          0.001263               0.001263               0.040341                      0.023679                            0.014016                             0.991160
           32           accessible_ls decrease 483                0.034137                   0.064890                  1.000000                    0.024845                    0.959282                      0.959282                       0.952883          0.001227               0.001227               0.040531                      0.022267                            0.013259                             1.020285
           32           accessible_ls increase 483                0.032931                   0.063841                  1.000000                    0.020704                    0.961353                      0.961353                       0.948428          0.001274               0.001274               0.041435                      0.022623                            0.013432                             0.977781
           32 grad_orthogonal_control decrease 483                0.001122                   0.007741                  0.784679                    0.037267                    0.936705                      0.936705                       0.929027          0.001249               0.001249               0.038015                      0.016873                            0.010162                             0.999239
           32 grad_orthogonal_control increase 483                0.001130                   0.007949                  0.217391                    0.047619                    0.934832                      0.934832                       0.933873          0.001251               0.001251               0.038057                      0.016900                            0.010175                             0.998366
           32          random_control decrease 483                0.013666                   0.028567                  0.995859                    0.047619                    0.942719                      0.942719                       0.929858          0.001238               0.001238               0.039374                      0.018851                            0.011515                             1.006042
           32          random_control increase 483                0.012605                   0.027720                  0.933747                    0.037267                    0.940550                      0.940550                       0.932353          0.001261               0.001261               0.039988                      0.019199                            0.011724                             0.992232
           64           accessible_ls decrease 483                0.036298                   0.074781                  1.000000                    0.024845                    0.961353                      0.961353                       0.952318          0.001230               0.001230               0.040422                      0.019659                            0.011834                             1.017521
           64           accessible_ls increase 483                0.035052                   0.073649                  1.000000                    0.020704                    0.964113                      0.964113                       0.949369          0.001271               0.001271               0.041180                      0.019898                            0.011931                             0.981440
           64 grad_orthogonal_control decrease 483                0.001064                   0.008080                  0.797101                    0.031056                    0.937198                      0.937198                       0.928916          0.001249               0.001249               0.038313                      0.013893                            0.008626                             0.999138
           64 grad_orthogonal_control increase 483                0.001071                   0.008790                  0.202899                    0.043478                    0.930494                      0.930494                       0.932555          0.001251               0.001251               0.038333                      0.013928                            0.008637                             0.999207
           64          random_control decrease 483                0.013630                   0.030944                  0.991718                    0.045549                    0.943606                      0.943606                       0.935065          0.001242               0.001242               0.039366                      0.016109                            0.009819                             1.005039
           64          random_control increase 483                0.012637                   0.029818                  0.937888                    0.045549                    0.947550                      0.947550                       0.936382          0.001258               0.001258               0.039780                      0.016358                            0.009964                             0.993655
          128           accessible_ls decrease 483                0.038357                   0.084954                  1.000000                    0.026915                    0.963423                      0.963423                       0.947174          0.001232               0.001232               0.040286                      0.017834                            0.010683                             1.015036
          128           accessible_ls increase 483                0.037125                   0.083706                  1.000000                    0.026915                    0.954649                      0.954649                       0.948554          0.001269               0.001269               0.040961                      0.018022                            0.010745                             0.984234
          128 grad_orthogonal_control decrease 483                0.001116                   0.011075                  0.772257                    0.055901                    0.943113                      0.943113                       0.936571          0.001251               0.001251               0.038591                      0.011954                            0.007507                             0.999453
          128 grad_orthogonal_control increase 483                0.001111                   0.011618                  0.227743                    0.033126                    0.935128                      0.935128                       0.939018          0.001249               0.001249               0.038592                      0.011938                            0.007513                             0.999254
          128          random_control decrease 483                0.014636                   0.035210                  0.987578                    0.045549                    0.946170                      0.946170                       0.931426          0.001242               0.001242               0.039278                      0.013882                            0.008512                             1.004561
          128          random_control increase 483                0.013799                   0.034560                  0.935818                    0.051760                    0.940156                      0.940156                       0.941966          0.001258               0.001258               0.039704                      0.014101                            0.008655                             0.994467
          256           accessible_ls decrease 483                0.040791                   0.095647                  1.000000                    0.028986                    0.959479                      0.959479                       0.950875          0.001232               0.001232               0.040129                      0.016535                            0.009786                             1.012333
          256           accessible_ls increase 483                0.039624                   0.094466                  1.000000                    0.028986                    0.958986                      0.958986                       0.947613          0.001269               0.001269               0.040773                      0.016683                            0.009831                             0.987062
          256 grad_orthogonal_control decrease 483                0.001146                   0.011856                  0.726708                    0.035197                    0.950508                      0.950508                       0.939833          0.001250               0.001250               0.038419                      0.010235                            0.006621                             0.999958
          256 grad_orthogonal_control increase 483                0.001139                   0.012115                  0.273292                    0.053830                    0.940649                      0.940649                       0.932806          0.001250               0.001250               0.038410                      0.010171                            0.006612                             0.999103
          256          random_control decrease 483                0.015486                   0.039234                  0.993789                    0.033126                    0.947550                      0.947550                       0.943409          0.001243               0.001243               0.039252                      0.011574                            0.007594                             1.003629
          256          random_control increase 483                0.014853                   0.039161                  0.954451                    0.051760                    0.937888                      0.937888                       0.931802          0.001257               0.001257               0.039628                      0.011803                            0.007707                             0.995574
```

## Files

```text
prompt_tables.csv
topk_scores.csv
topk_steering_records.csv
topk_robustness_summary.csv
topk_rank_stability.csv
topk_layer_trend_stability.csv
topk_steering_summary.csv
skipped_models.csv
report.md
```