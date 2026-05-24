# Uncertainty Steering Full Battery

This suite targets five checks: directional monotonicity, equal Fisher-output-energy controls, specificity, rho dependency after scalar controls, and replication across available local MLM models/tasks/layers/subspace dimensions/random seeds.

RoBERTa was not present in the local Hugging Face cache; no network download was attempted.

## Skipped Models
```text
(empty)
```

## Prompt Counts
```text
                            model           task observed_condition  size
                bert-base-uncased ambiguous_open          ambiguous    12
                bert-base-uncased   factual_deep            correct     6
                bert-base-uncased   factual_deep              error     6
                bert-base-uncased factual_simple            correct     5
                bert-base-uncased factual_simple              error     4
          distilbert-base-uncased ambiguous_open          ambiguous    12
          distilbert-base-uncased   factual_deep            correct     5
          distilbert-base-uncased   factual_deep              error     7
          distilbert-base-uncased factual_simple            correct     5
          distilbert-base-uncased factual_simple              error     4
google/bert_uncased_L-2_H-128_A-2 ambiguous_open          ambiguous    12
google/bert_uncased_L-2_H-128_A-2   factual_deep            correct     1
google/bert_uncased_L-2_H-128_A-2   factual_deep              error    11
google/bert_uncased_L-2_H-128_A-2 factual_simple            correct     1
google/bert_uncased_L-2_H-128_A-2 factual_simple              error     8
```

## 1. Directional And Monotonic Control
```text
            model           task  layer subspace_family  k     sign  n  directional_rate  monotonic_abs_rate  entropy_slope_mean
bert-base-uncased ambiguous_open      4       delta_pca  1 decrease 12          1.000000            1.000000           -0.208573
bert-base-uncased ambiguous_open      4       delta_pca  1 increase 12          0.916667            0.916667            0.131764
bert-base-uncased ambiguous_open      4       delta_pca  4 decrease 12          1.000000            1.000000           -0.472190
bert-base-uncased ambiguous_open      4       delta_pca  4 increase 12          1.000000            1.000000            0.378530
bert-base-uncased ambiguous_open      4       delta_pca  8 decrease 12          1.000000            1.000000           -0.546050
bert-base-uncased ambiguous_open      4       delta_pca  8 increase 12          1.000000            1.000000            0.461274
bert-base-uncased ambiguous_open      4          random  1 decrease 12          1.000000            0.250000           -0.224877
bert-base-uncased ambiguous_open      4          random  1 increase 12          0.583333            0.250000            0.137739
bert-base-uncased ambiguous_open      4          random  4 decrease 12          1.000000            0.166667           -0.410678
bert-base-uncased ambiguous_open      4          random  4 increase 12          1.000000            0.083333            0.317457
bert-base-uncased ambiguous_open      4          random  8 decrease 12          1.000000            0.333333           -0.517708
bert-base-uncased ambiguous_open      4          random  8 increase 12          1.000000            0.500000            0.441289
bert-base-uncased ambiguous_open      4       state_pca  1 decrease 12          1.000000            1.000000           -0.228714
bert-base-uncased ambiguous_open      4       state_pca  1 increase 12          0.750000            0.916667            0.139537
bert-base-uncased ambiguous_open      4       state_pca  4 decrease 12          1.000000            1.000000           -0.450652
bert-base-uncased ambiguous_open      4       state_pca  4 increase 12          1.000000            1.000000            0.356388
bert-base-uncased ambiguous_open      4       state_pca  8 decrease 12          1.000000            1.000000           -0.499374
bert-base-uncased ambiguous_open      4       state_pca  8 increase 12          1.000000            1.000000            0.481371
bert-base-uncased ambiguous_open      8       delta_pca  1 decrease 12          0.916667            0.916667           -0.258369
bert-base-uncased ambiguous_open      8       delta_pca  1 increase 12          0.916667            0.750000            0.289728
bert-base-uncased ambiguous_open      8       delta_pca  4 decrease 12          1.000000            1.000000           -0.661166
bert-base-uncased ambiguous_open      8       delta_pca  4 increase 12          1.000000            1.000000            0.684884
bert-base-uncased ambiguous_open      8       delta_pca  8 decrease 12          1.000000            1.000000           -0.758438
bert-base-uncased ambiguous_open      8       delta_pca  8 increase 12          1.000000            1.000000            0.829295
bert-base-uncased ambiguous_open      8          random  1 decrease 12          1.000000            0.166667           -0.295603
bert-base-uncased ambiguous_open      8          random  1 increase 12          0.916667            0.250000            0.368138
bert-base-uncased ambiguous_open      8          random  4 decrease 12          1.000000            0.583333           -0.661067
bert-base-uncased ambiguous_open      8          random  4 increase 12          1.000000            0.500000            0.691333
bert-base-uncased ambiguous_open      8          random  8 decrease 12          1.000000            0.583333           -0.721081
bert-base-uncased ambiguous_open      8          random  8 increase 12          1.000000            0.583333            0.907258
bert-base-uncased ambiguous_open      8       state_pca  1 decrease 12          0.916667            0.916667           -0.415305
bert-base-uncased ambiguous_open      8       state_pca  1 increase 12          0.916667            1.000000            0.443204
bert-base-uncased ambiguous_open      8       state_pca  4 decrease 12          1.000000            1.000000           -0.636204
bert-base-uncased ambiguous_open      8       state_pca  4 increase 12          1.000000            1.000000            0.710283
bert-base-uncased ambiguous_open      8       state_pca  8 decrease 12          1.000000            1.000000           -0.729846
bert-base-uncased ambiguous_open      8       state_pca  8 increase 12          1.000000            1.000000            0.881330
bert-base-uncased ambiguous_open     12       delta_pca  1 decrease 12          1.000000            1.000000           -0.435604
bert-base-uncased ambiguous_open     12       delta_pca  1 increase 12          0.916667            0.833333            0.423584
bert-base-uncased ambiguous_open     12       delta_pca  4 decrease 12          1.000000            1.000000           -0.698783
bert-base-uncased ambiguous_open     12       delta_pca  4 increase 12          1.000000            1.000000            0.714237
bert-base-uncased ambiguous_open     12       delta_pca  8 decrease 12          1.000000            1.000000           -0.823171
bert-base-uncased ambiguous_open     12       delta_pca  8 increase 12          1.000000            1.000000            0.889460
bert-base-uncased ambiguous_open     12          random  1 decrease 12          1.000000            0.250000           -0.332915
bert-base-uncased ambiguous_open     12          random  1 increase 12          0.916667            0.083333            0.369258
bert-base-uncased ambiguous_open     12          random  4 decrease 12          1.000000            0.500000           -0.643557
bert-base-uncased ambiguous_open     12          random  4 increase 12          1.000000            0.500000            0.640515
bert-base-uncased ambiguous_open     12          random  8 decrease 12          1.000000            0.083333           -0.718428
bert-base-uncased ambiguous_open     12          random  8 increase 12          1.000000            0.500000            0.920739
bert-base-uncased ambiguous_open     12       state_pca  1 decrease 12          1.000000            1.000000           -0.418340
bert-base-uncased ambiguous_open     12       state_pca  1 increase 12          1.000000            1.000000            0.417660
bert-base-uncased ambiguous_open     12       state_pca  4 decrease 12          1.000000            1.000000           -0.650119
bert-base-uncased ambiguous_open     12       state_pca  4 increase 12          1.000000            1.000000            0.710595
bert-base-uncased ambiguous_open     12       state_pca  8 decrease 12          1.000000            1.000000           -0.831296
bert-base-uncased ambiguous_open     12       state_pca  8 increase 12          1.000000            1.000000            0.880762
bert-base-uncased   factual_deep      4       delta_pca  1 decrease 12          1.000000            0.916667           -0.294982
bert-base-uncased   factual_deep      4       delta_pca  1 increase 12          1.000000            1.000000            0.251236
bert-base-uncased   factual_deep      4       delta_pca  4 decrease 12          1.000000            1.000000           -0.573045
bert-base-uncased   factual_deep      4       delta_pca  4 increase 12          1.000000            1.000000            0.528125
bert-base-uncased   factual_deep      4       delta_pca  8 decrease 12          1.000000            1.000000           -0.698976
bert-base-uncased   factual_deep      4       delta_pca  8 increase 12          1.000000            1.000000            0.682503
bert-base-uncased   factual_deep      4          random  1 decrease 12          1.000000            0.083333           -0.330822
bert-base-uncased   factual_deep      4          random  1 increase 12          0.916667            0.000000            0.303739
bert-base-uncased   factual_deep      4          random  4 decrease 12          1.000000            0.500000           -0.534748
bert-base-uncased   factual_deep      4          random  4 increase 12          1.000000            0.500000            0.530534
bert-base-uncased   factual_deep      4          random  8 decrease 12          1.000000            0.500000           -0.686914
bert-base-uncased   factual_deep      4          random  8 increase 12          1.000000            0.500000            0.673942
bert-base-uncased   factual_deep      4       state_pca  1 decrease 12          1.000000            1.000000           -0.307878
bert-base-uncased   factual_deep      4       state_pca  1 increase 12          1.000000            1.000000            0.265725
bert-base-uncased   factual_deep      4       state_pca  4 decrease 12          1.000000            1.000000           -0.564123
bert-base-uncased   factual_deep      4       state_pca  4 increase 12          1.000000            1.000000            0.531520
bert-base-uncased   factual_deep      4       state_pca  8 decrease 12          1.000000            1.000000           -0.659439
bert-base-uncased   factual_deep      4       state_pca  8 increase 12          1.000000            1.000000            0.676461
bert-base-uncased   factual_deep      8       delta_pca  1 decrease 12          0.916667            1.000000            0.110385
bert-base-uncased   factual_deep      8       delta_pca  1 increase 12          1.000000            1.000000            0.379907
bert-base-uncased   factual_deep      8       delta_pca  4 decrease 12          1.000000            1.000000           -0.488186
bert-base-uncased   factual_deep      8       delta_pca  4 increase 12          1.000000            1.000000            0.467757
bert-base-uncased   factual_deep      8       delta_pca  8 decrease 12          1.000000            1.000000           -0.575570
bert-base-uncased   factual_deep      8       delta_pca  8 increase 12          1.000000            1.000000            0.657063
bert-base-uncased   factual_deep      8          random  1 decrease 12          0.916667            0.083333           -0.268135
bert-base-uncased   factual_deep      8          random  1 increase 12          0.916667            0.083333            0.271273
```

## 2. Equal Fisher-Output-Energy Contrasts
```text
    sign  epsilon                 control                    metric  accessible  control_value  accessible_minus_control  accessible_over_control  accessible_output_energy  control_output_energy
decrease     0.02 grad_orthogonal_control    abs_delta_entropy_mean    0.010085       0.000312                  0.009773                32.319587                      0.02                   0.02
decrease     0.02 grad_orthogonal_control abs_delta_varentropy_mean    0.016503       0.003155                  0.013348                 5.230768                      0.02                   0.02
decrease     0.02          random_control    abs_delta_entropy_mean    0.010085       0.005567                  0.004517                 1.811451                      0.02                   0.02
decrease     0.02          random_control abs_delta_varentropy_mean    0.016503       0.009916                  0.006587                 1.664248                      0.02                   0.02
increase     0.02 grad_orthogonal_control    abs_delta_entropy_mean    0.010169       0.000314                  0.009856                32.396358                      0.02                   0.02
increase     0.02 grad_orthogonal_control abs_delta_varentropy_mean    0.016766       0.003247                  0.013519                 5.162995                      0.02                   0.02
increase     0.02          random_control    abs_delta_entropy_mean    0.010169       0.005563                  0.004606                 1.828012                      0.02                   0.02
increase     0.02          random_control abs_delta_varentropy_mean    0.016766       0.010013                  0.006754                 1.674525                      0.02                   0.02
decrease     0.05 grad_orthogonal_control    abs_delta_entropy_mean    0.025126       0.001989                  0.023137                12.631322                      0.05                   0.05
decrease     0.05 grad_orthogonal_control abs_delta_varentropy_mean    0.040864       0.009258                  0.031606                 4.413959                      0.05                   0.05
decrease     0.05          random_control    abs_delta_entropy_mean    0.025126       0.014048                  0.011078                 1.788542                      0.05                   0.05
decrease     0.05          random_control abs_delta_varentropy_mean    0.040864       0.024833                  0.016031                 1.645551                      0.05                   0.05
increase     0.05 grad_orthogonal_control    abs_delta_entropy_mean    0.025628       0.002017                  0.023611                12.706596                      0.05                   0.05
increase     0.05 grad_orthogonal_control abs_delta_varentropy_mean    0.042465       0.009707                  0.032758                 4.374830                      0.05                   0.05
increase     0.05          random_control    abs_delta_entropy_mean    0.025628       0.013984                  0.011644                 1.832642                      0.05                   0.05
increase     0.05          random_control abs_delta_varentropy_mean    0.042465       0.025402                  0.017062                 1.671676                      0.05                   0.05
decrease     0.10 grad_orthogonal_control    abs_delta_entropy_mean    0.050214       0.008534                  0.041680                 5.883889                      0.10                   0.10
decrease     0.10 grad_orthogonal_control abs_delta_varentropy_mean    0.081293       0.023808                  0.057484                 3.414454                      0.10                   0.10
decrease     0.10          random_control    abs_delta_entropy_mean    0.050214       0.028933                  0.021281                 1.735533                      0.10                   0.10
decrease     0.10          random_control abs_delta_varentropy_mean    0.081293       0.050762                  0.030530                 1.601438                      0.10                   0.10
increase     0.10 grad_orthogonal_control    abs_delta_entropy_mean    0.052049       0.008722                  0.043327                 5.967771                      0.10                   0.10
increase     0.10 grad_orthogonal_control abs_delta_varentropy_mean    0.086674       0.025487                  0.061187                 3.400697                      0.10                   0.10
increase     0.10          random_control    abs_delta_entropy_mean    0.052049       0.028618                  0.023430                 1.818706                      0.10                   0.10
increase     0.10          random_control abs_delta_varentropy_mean    0.086674       0.052372                  0.034302                 1.654970                      0.10                   0.10
```

## 3. Specificity
```text
                            model           task     sign  epsilon   n  abs_delta_entropy_mean  abs_delta_varentropy_mean  selected_top1_changed_rate  target_in_topk_rate  target_correct_changed_rate  embedding_centroid_delta_norm_mean  cluster_distribution_l1_mean  rank_displacement_mean  top5_jaccard_mean  selected_prob_l1_mean  selected_kl_mean  abs_non_target_mass_delta_mean
                bert-base-uncased ambiguous_open decrease     0.02 432                0.010624                   0.019188                    0.000000             0.000000                     0.000000                            0.006518                      0.011001                0.013082           0.977623               0.016072          0.000197                             NaN
                bert-base-uncased ambiguous_open increase     0.02 432                0.010684                   0.019584                    0.000000             0.000000                     0.000000                            0.006634                      0.011198                0.011749           0.978395               0.016359          0.000203                             NaN
                bert-base-uncased ambiguous_open decrease     0.05 432                0.026434                   0.047167                    0.016204             0.000000                     0.000000                            0.016088                      0.027139                0.030022           0.948302               0.039650          0.001203                             NaN
                bert-base-uncased ambiguous_open increase     0.05 432                0.026818                   0.049625                    0.009259             0.000000                     0.000000                            0.016810                      0.028362                0.029602           0.950617               0.041451          0.001303                             NaN
                bert-base-uncased ambiguous_open decrease     0.10 432                0.052390                   0.091541                    0.050926             0.000000                     0.000000                            0.031524                      0.053150                0.051031           0.909612               0.077606          0.004651                             NaN
                bert-base-uncased ambiguous_open increase     0.10 432                0.053939                   0.100903                    0.062500             0.000000                     0.000000                            0.034385                      0.057973                0.055941           0.902888               0.084744          0.005439                             NaN
                bert-base-uncased   factual_deep decrease     0.02 432                0.012230                   0.015703                    0.002315             0.250000                     0.000000                            0.006447                      0.011323                0.012486           0.983796               0.015698          0.000196                        0.004890
                bert-base-uncased   factual_deep increase     0.02 432                0.012617                   0.015961                    0.002315             0.250000                     0.000000                            0.006602                      0.011626                0.013468           0.971451               0.016108          0.000204                        0.005189
                bert-base-uncased   factual_deep decrease     0.05 432                0.030026                   0.038855                    0.011574             0.250000                     0.000000                            0.015869                      0.027836                0.027673           0.968364               0.038591          0.001198                        0.011684
                bert-base-uncased   factual_deep increase     0.05 432                0.032320                   0.040378                    0.016204             0.250000                     0.000000                            0.016827                      0.029684                0.029847           0.937720               0.041094          0.001326                        0.013560
                bert-base-uncased   factual_deep decrease     0.10 432                0.058540                   0.077313                    0.023148             0.250000                     0.000000                            0.031019                      0.054223                0.048857           0.935957               0.075238          0.004695                        0.021599
                bert-base-uncased   factual_deep increase     0.10 432                0.067082                   0.081753                    0.053241             0.250000                     0.000000                            0.034796                      0.061600                0.054398           0.889330               0.085029          0.005667                        0.029116
                bert-base-uncased factual_simple decrease     0.02 324                0.012265                   0.020707                    0.000000             0.481481                     0.000000                            0.006975                      0.011535                0.011036           0.968107               0.015457          0.000194                        0.002383
                bert-base-uncased factual_simple increase     0.02 324                0.012725                   0.022398                    0.000000             0.481481                     0.000000                            0.007142                      0.011857                0.014076           0.974280               0.015871          0.000207                        0.002547
                bert-base-uncased factual_simple decrease     0.05 324                0.030070                   0.049384                    0.003086             0.481481                     0.000000                            0.017215                      0.028393                0.026515           0.929012               0.038064          0.001177                        0.005748
                bert-base-uncased factual_simple increase     0.05 324                0.032890                   0.059728                    0.003086             0.481481                     0.000000                            0.018243                      0.030360                0.029087           0.945473               0.040585          0.001385                        0.006801
                bert-base-uncased factual_simple decrease     0.10 324                0.059785                   0.096857                    0.018519             0.481481                     0.000000                            0.033973                      0.055887                0.044893           0.898001               0.074934          0.004597                        0.011230
                bert-base-uncased factual_simple increase     0.10 324                0.070716                   0.132862                    0.018519             0.481481                     0.000000                            0.038179                      0.063879                0.050926           0.904468               0.085151          0.006388                        0.015987
          distilbert-base-uncased ambiguous_open decrease     0.02 432                0.007731                   0.015678                    0.000000             0.000000                     0.000000                            0.007491                      0.010810                0.012416           0.980710               0.016576          0.000199                             NaN
          distilbert-base-uncased ambiguous_open increase     0.02 432                0.007461                   0.015331                    0.006944             0.000000                     0.000000                            0.007534                      0.010869                0.012977           0.981481               0.016708          0.000201                             NaN
          distilbert-base-uncased ambiguous_open decrease     0.05 432                0.019800                   0.039773                    0.046296             0.000000                     0.000000                            0.018639                      0.026905                0.031671           0.939043               0.041189          0.001230                             NaN
          distilbert-base-uncased ambiguous_open increase     0.05 432                0.018143                   0.037613                    0.048611             0.000000                     0.000000                            0.018911                      0.027281                0.033845           0.920525               0.042006          0.001270                             NaN
          distilbert-base-uncased ambiguous_open decrease     0.10 432                0.041038                   0.081164                    0.090278             0.000000                     0.000000                            0.036963                      0.053414                0.055380           0.892857               0.081481          0.004834                             NaN
          distilbert-base-uncased ambiguous_open increase     0.10 432                0.034520                   0.072612                    0.120370             0.000000                     0.000000                            0.038044                      0.054906                0.062816           0.851631               0.084734          0.005158                             NaN
          distilbert-base-uncased   factual_deep decrease     0.02 432                0.009979                   0.013579                    0.013889             0.277778                     0.000000                            0.007326                      0.011048                0.014941           0.972222               0.015821          0.000197                        0.003936
          distilbert-base-uncased   factual_deep increase     0.02 432                0.010083                   0.013640                    0.025463             0.277778                     0.000000                            0.007431                      0.011211                0.016134           0.972994               0.016091          0.000203                        0.004169
          distilbert-base-uncased   factual_deep decrease     0.05 432                0.024752                   0.033819                    0.025463             0.277778                     0.002315                            0.018128                      0.027326                0.034196           0.936728               0.039072          0.001207                        0.009455
          distilbert-base-uncased   factual_deep increase     0.05 432                0.025395                   0.034185                    0.060185             0.277778                     0.000000                            0.018786                      0.028345                0.037774           0.931768               0.040774          0.001299                        0.010880
          distilbert-base-uncased   factual_deep decrease     0.10 432                0.048916                   0.067287                    0.041667             0.277778                     0.002315                            0.035698                      0.053772                0.060325           0.887676               0.076651          0.004688                        0.017766
          distilbert-base-uncased   factual_deep increase     0.10 432                0.051419                   0.068492                    0.104167             0.277778                     0.000000                            0.038314                      0.057880                0.068533           0.862213               0.083414          0.005414                        0.023362
          distilbert-base-uncased factual_simple decrease     0.02 324                0.009207                   0.016550                    0.006173             0.370370                     0.006173                            0.007622                      0.010358                0.011177           0.988683               0.015778          0.000196                        0.002923
          distilbert-base-uncased factual_simple increase     0.02 324                0.009404                   0.016930                    0.000000             0.370370                     0.000000                            0.007726                      0.010514                0.013047           0.986626               0.016042          0.000206                        0.003073
          distilbert-base-uncased factual_simple decrease     0.05 324                0.023023                   0.041103                    0.012346             0.370370                     0.006173                            0.018997                      0.025766                0.027731           0.948560               0.039269          0.001216                        0.007312
          distilbert-base-uncased factual_simple increase     0.05 324                0.024173                   0.043436                    0.021605             0.370370                     0.009259                            0.019707                      0.026841                0.030490           0.940329               0.040982          0.001380                        0.008306
          distilbert-base-uncased factual_simple decrease     0.10 324                0.046981                   0.083593                    0.046296             0.370370                     0.012346                            0.038066                      0.051887                0.050739           0.908289               0.078275          0.004943                        0.015420
          distilbert-base-uncased factual_simple increase     0.10 324                0.051197                   0.091735                    0.086420             0.370370                     0.018519                            0.041117                      0.056402                0.060045           0.883157               0.085527          0.006472                        0.019533
google/bert_uncased_L-2_H-128_A-2 ambiguous_open decrease     0.02 288                0.008329                   0.014947                    0.010417             0.000000                     0.000000                            0.003214                      0.010829                0.012574           0.972222               0.016182          0.000199                             NaN
google/bert_uncased_L-2_H-128_A-2 ambiguous_open increase     0.02 288                0.008118                   0.014849                    0.045139             0.000000                     0.000000                            0.003251                      0.010926                0.013784           0.975694               0.016343          0.000201                             NaN
google/bert_uncased_L-2_H-128_A-2 ambiguous_open decrease     0.05 288                0.021171                   0.037451                    0.031250             0.000000                     0.000000                            0.007965                      0.026888                0.028935           0.944444               0.040128          0.001231                             NaN
google/bert_uncased_L-2_H-128_A-2 ambiguous_open increase     0.05 288                0.019879                   0.036837                    0.097222             0.000000                     0.000000                            0.008194                      0.027489                0.030671           0.954861               0.041137          0.001269                             NaN
google/bert_uncased_L-2_H-128_A-2 ambiguous_open decrease     0.10 288                0.043329                   0.074909                    0.065972             0.000000                     0.000000                            0.015700                      0.053167                0.051347           0.901124               0.079162          0.004844                             NaN
google/bert_uncased_L-2_H-128_A-2 ambiguous_open increase     0.10 288                0.038243                   0.072359                    0.131944             0.000000                     0.000000                            0.016601                      0.055492                0.057397           0.919643               0.083136          0.005147                             NaN
google/bert_uncased_L-2_H-128_A-2   factual_deep decrease     0.02 288                0.010593                   0.017090                    0.000000             0.125000                     0.000000                            0.003435                      0.011415                0.010680           0.982639               0.016135          0.000197                        0.002705
google/bert_uncased_L-2_H-128_A-2   factual_deep increase     0.02 288                0.010598                   0.017300                    0.017361             0.125000                     0.000000                            0.003515                      0.011667                0.011311           0.987269               0.016473          0.000203                        0.002805
google/bert_uncased_L-2_H-128_A-2   factual_deep decrease     0.05 288                0.026415                   0.042302                    0.013889             0.125000                     0.000000                            0.008445                      0.028078                0.026620           0.957176               0.039701          0.001205                        0.006572
google/bert_uncased_L-2_H-128_A-2   factual_deep increase     0.05 288                0.026443                   0.043630                    0.038194             0.125000                     0.000000                            0.008935                      0.029645                0.028514           0.960648               0.041796          0.001300                        0.007192
google/bert_uncased_L-2_H-128_A-2   factual_deep decrease     0.10 288                0.052479                   0.083089                    0.027778             0.125000                     0.000000                            0.016460                      0.054782                0.046770           0.936673               0.077444          0.004660                        0.012625
google/bert_uncased_L-2_H-128_A-2   factual_deep increase     0.10 288                0.052425                   0.088304                    0.076389             0.125000                     0.000000                            0.018344                      0.060794                0.050926           0.913360               0.085523          0.005415                        0.014933
google/bert_uncased_L-2_H-128_A-2 factual_simple decrease     0.02 216                0.009342                   0.015150                    0.083333             0.333333                     0.000000                            0.003424                      0.012249                0.015081           0.975309               0.016125          0.000197                        0.002194
google/bert_uncased_L-2_H-128_A-2 factual_simple increase     0.02 216                0.009313                   0.015015                    0.041667             0.333333                     0.000000                            0.003495                      0.012489                0.015152           0.939815               0.016435          0.000203                        0.002273
google/bert_uncased_L-2_H-128_A-2 factual_simple decrease     0.05 216                0.023402                   0.038043                    0.101852             0.333333                     0.000000                            0.008429                      0.030156                0.028970           0.952160               0.039757          0.001208                        0.005347
google/bert_uncased_L-2_H-128_A-2 factual_simple increase     0.05 216                0.023173                   0.037176                    0.064815             0.333333                     0.000000                            0.008871                      0.031665                0.032197           0.923721               0.041673          0.001301                        0.005846
google/bert_uncased_L-2_H-128_A-2 factual_simple decrease     0.10 216                0.046819                   0.076342                    0.134259             0.333333                     0.000000                            0.016459                      0.058801                0.054293           0.920194               0.077690          0.004688                        0.010295
google/bert_uncased_L-2_H-128_A-2 factual_simple increase     0.10 216                0.045697                   0.072592                    0.125000             0.333333                     0.000000                            0.018170                      0.064713                0.058151           0.891314               0.085231          0.005443                        0.012241
```

## 4. Rho Dependency After Controls
```text
                      outcome     n  partial_corr  rho_beta                                  subset
            abs_delta_entropy 19008      0.383748  0.037773                                     all
         abs_delta_varentropy 19008      0.324510  0.070453                                     all
embedding_centroid_delta_norm 19008      0.138937  0.004651                                     all
        selected_top1_changed 19008           NaN       NaN                                     all
            abs_delta_entropy  7128      0.377630  0.039805                 model:bert-base-uncased
         abs_delta_varentropy  7128      0.240456  0.059950                 model:bert-base-uncased
embedding_centroid_delta_norm  7128      0.169110  0.004945                 model:bert-base-uncased
        selected_top1_changed  7128           NaN       NaN                 model:bert-base-uncased
            abs_delta_entropy  7128      0.356961  0.033695           model:distilbert-base-uncased
         abs_delta_varentropy  7128      0.357058  0.070991           model:distilbert-base-uncased
embedding_centroid_delta_norm  7128      0.158600  0.005490           model:distilbert-base-uncased
        selected_top1_changed  7128           NaN       NaN           model:distilbert-base-uncased
            abs_delta_entropy  4752      0.431491  0.039041 model:google/bert_uncased_L-2_H-128_A-2
         abs_delta_varentropy  4752      0.439816  0.069664 model:google/bert_uncased_L-2_H-128_A-2
embedding_centroid_delta_norm  4752      0.172102  0.002646 model:google/bert_uncased_L-2_H-128_A-2
        selected_top1_changed  4752           NaN       NaN model:google/bert_uncased_L-2_H-128_A-2
```

## 5. Replication Matrix
```text
                            model  layer subspace_family  k  n_subspace_prompt_rows  rho_mean  rho_std                mode  n_interventions
                bert-base-uncased      4       delta_pca  1                      33  0.182361 0.212365 fisher_output_equal            19008
                bert-base-uncased      4       delta_pca  1                      33  0.182361 0.212365        latent_equal             6336
                bert-base-uncased      4       delta_pca  4                      33  0.585736 0.179468 fisher_output_equal            19008
                bert-base-uncased      4       delta_pca  4                      33  0.585736 0.179468        latent_equal             6336
                bert-base-uncased      4       delta_pca  8                      33  0.874291 0.128945 fisher_output_equal            19008
                bert-base-uncased      4       delta_pca  8                      33  0.874291 0.128945        latent_equal             6336
                bert-base-uncased      4          random  1                      66  0.168221 0.192444 fisher_output_equal            19008
                bert-base-uncased      4          random  1                      66  0.168221 0.192444        latent_equal             6336
                bert-base-uncased      4          random  4                      66  0.521280 0.192597 fisher_output_equal            19008
                bert-base-uncased      4          random  4                      66  0.521280 0.192597        latent_equal             6336
                bert-base-uncased      4          random  8                      66  0.818207 0.135379 fisher_output_equal            19008
                bert-base-uncased      4          random  8                      66  0.818207 0.135379        latent_equal             6336
                bert-base-uncased      4       state_pca  1                      33  0.180853 0.181762 fisher_output_equal            19008
                bert-base-uncased      4       state_pca  1                      33  0.180853 0.181762        latent_equal             6336
                bert-base-uncased      4       state_pca  4                      33  0.582685 0.210362 fisher_output_equal            19008
                bert-base-uncased      4       state_pca  4                      33  0.582685 0.210362        latent_equal             6336
                bert-base-uncased      4       state_pca  8                      33  0.839357 0.149515 fisher_output_equal            19008
                bert-base-uncased      4       state_pca  8                      33  0.839357 0.149515        latent_equal             6336
                bert-base-uncased      8       delta_pca  1                      33  0.136212 0.193250 fisher_output_equal            19008
                bert-base-uncased      8       delta_pca  1                      33  0.136212 0.193250        latent_equal             6336
                bert-base-uncased      8       delta_pca  4                      33  0.583199 0.202875 fisher_output_equal            19008
                bert-base-uncased      8       delta_pca  4                      33  0.583199 0.202875        latent_equal             6336
                bert-base-uncased      8       delta_pca  8                      33  0.889756 0.100304 fisher_output_equal            19008
                bert-base-uncased      8       delta_pca  8                      33  0.889756 0.100304        latent_equal             6336
                bert-base-uncased      8          random  1                      66  0.158819 0.169336 fisher_output_equal            19008
                bert-base-uncased      8          random  1                      66  0.158819 0.169336        latent_equal             6336
                bert-base-uncased      8          random  4                      66  0.577708 0.236863 fisher_output_equal            19008
                bert-base-uncased      8          random  4                      66  0.577708 0.236863        latent_equal             6336
                bert-base-uncased      8          random  8                      66  0.843923 0.134083 fisher_output_equal            19008
                bert-base-uncased      8          random  8                      66  0.843923 0.134083        latent_equal             6336
                bert-base-uncased      8       state_pca  1                      33  0.276180 0.250145 fisher_output_equal            19008
                bert-base-uncased      8       state_pca  1                      33  0.276180 0.250145        latent_equal             6336
                bert-base-uncased      8       state_pca  4                      33  0.597626 0.253476 fisher_output_equal            19008
                bert-base-uncased      8       state_pca  4                      33  0.597626 0.253476        latent_equal             6336
                bert-base-uncased      8       state_pca  8                      33  0.868228 0.122206 fisher_output_equal            19008
                bert-base-uncased      8       state_pca  8                      33  0.868228 0.122206        latent_equal             6336
                bert-base-uncased     12       delta_pca  1                      33  0.284990 0.243075 fisher_output_equal            19008
                bert-base-uncased     12       delta_pca  1                      33  0.284990 0.243075        latent_equal             6336
                bert-base-uncased     12       delta_pca  4                      33  0.655683 0.245558 fisher_output_equal            19008
                bert-base-uncased     12       delta_pca  4                      33  0.655683 0.245558        latent_equal             6336
                bert-base-uncased     12       delta_pca  8                      33  0.889558 0.103713 fisher_output_equal            19008
                bert-base-uncased     12       delta_pca  8                      33  0.889558 0.103713        latent_equal             6336
                bert-base-uncased     12          random  1                      66  0.265483 0.226723 fisher_output_equal            19008
                bert-base-uncased     12          random  1                      66  0.265483 0.226723        latent_equal             6336
                bert-base-uncased     12          random  4                      66  0.616068 0.225046 fisher_output_equal            19008
                bert-base-uncased     12          random  4                      66  0.616068 0.225046        latent_equal             6336
                bert-base-uncased     12          random  8                      66  0.879768 0.138419 fisher_output_equal            19008
                bert-base-uncased     12          random  8                      66  0.879768 0.138419        latent_equal             6336
                bert-base-uncased     12       state_pca  1                      33  0.248748 0.190259 fisher_output_equal            19008
                bert-base-uncased     12       state_pca  1                      33  0.248748 0.190259        latent_equal             6336
                bert-base-uncased     12       state_pca  4                      33  0.632084 0.231167 fisher_output_equal            19008
                bert-base-uncased     12       state_pca  4                      33  0.632084 0.231167        latent_equal             6336
                bert-base-uncased     12       state_pca  8                      33  0.915843 0.104364 fisher_output_equal            19008
                bert-base-uncased     12       state_pca  8                      33  0.915843 0.104364        latent_equal             6336
          distilbert-base-uncased      2       delta_pca  1                      33  0.179729 0.196897 fisher_output_equal            19008
          distilbert-base-uncased      2       delta_pca  1                      33  0.179729 0.196897        latent_equal             6336
          distilbert-base-uncased      2       delta_pca  4                      33  0.569790 0.241266 fisher_output_equal            19008
          distilbert-base-uncased      2       delta_pca  4                      33  0.569790 0.241266        latent_equal             6336
          distilbert-base-uncased      2       delta_pca  8                      33  0.856120 0.143623 fisher_output_equal            19008
          distilbert-base-uncased      2       delta_pca  8                      33  0.856120 0.143623        latent_equal             6336
          distilbert-base-uncased      2          random  1                      66  0.194598 0.180930 fisher_output_equal            19008
          distilbert-base-uncased      2          random  1                      66  0.194598 0.180930        latent_equal             6336
          distilbert-base-uncased      2          random  4                      66  0.507723 0.220284 fisher_output_equal            19008
          distilbert-base-uncased      2          random  4                      66  0.507723 0.220284        latent_equal             6336
          distilbert-base-uncased      2          random  8                      66  0.801577 0.144741 fisher_output_equal            19008
          distilbert-base-uncased      2          random  8                      66  0.801577 0.144741        latent_equal             6336
          distilbert-base-uncased      2       state_pca  1                      33  0.205520 0.230993 fisher_output_equal            19008
          distilbert-base-uncased      2       state_pca  1                      33  0.205520 0.230993        latent_equal             6336
          distilbert-base-uncased      2       state_pca  4                      33  0.570966 0.228032 fisher_output_equal            19008
          distilbert-base-uncased      2       state_pca  4                      33  0.570966 0.228032        latent_equal             6336
          distilbert-base-uncased      2       state_pca  8                      33  0.893179 0.071908 fisher_output_equal            19008
          distilbert-base-uncased      2       state_pca  8                      33  0.893179 0.071908        latent_equal             6336
          distilbert-base-uncased      4       delta_pca  1                      33  0.136652 0.173671 fisher_output_equal            19008
          distilbert-base-uncased      4       delta_pca  1                      33  0.136652 0.173671        latent_equal             6336
          distilbert-base-uncased      4       delta_pca  4                      33  0.486751 0.217378 fisher_output_equal            19008
          distilbert-base-uncased      4       delta_pca  4                      33  0.486751 0.217378        latent_equal             6336
          distilbert-base-uncased      4       delta_pca  8                      33  0.812179 0.125681 fisher_output_equal            19008
          distilbert-base-uncased      4       delta_pca  8                      33  0.812179 0.125681        latent_equal             6336
          distilbert-base-uncased      4          random  1                      66  0.174976 0.187840 fisher_output_equal            19008
          distilbert-base-uncased      4          random  1                      66  0.174976 0.187840        latent_equal             6336
          distilbert-base-uncased      4          random  4                      66  0.423923 0.202983 fisher_output_equal            19008
          distilbert-base-uncased      4          random  4                      66  0.423923 0.202983        latent_equal             6336
          distilbert-base-uncased      4          random  8                      66  0.763743 0.174184 fisher_output_equal            19008
          distilbert-base-uncased      4          random  8                      66  0.763743 0.174184        latent_equal             6336
          distilbert-base-uncased      4       state_pca  1                      33  0.152187 0.167433 fisher_output_equal            19008
          distilbert-base-uncased      4       state_pca  1                      33  0.152187 0.167433        latent_equal             6336
          distilbert-base-uncased      4       state_pca  4                      33  0.495061 0.195423 fisher_output_equal            19008
          distilbert-base-uncased      4       state_pca  4                      33  0.495061 0.195423        latent_equal             6336
          distilbert-base-uncased      4       state_pca  8                      33  0.801178 0.162125 fisher_output_equal            19008
          distilbert-base-uncased      4       state_pca  8                      33  0.801178 0.162125        latent_equal             6336
          distilbert-base-uncased      6       delta_pca  1                      33  0.158264 0.200375 fisher_output_equal            19008
          distilbert-base-uncased      6       delta_pca  1                      33  0.158264 0.200375        latent_equal             6336
          distilbert-base-uncased      6       delta_pca  4                      33  0.554839 0.214532 fisher_output_equal            19008
          distilbert-base-uncased      6       delta_pca  4                      33  0.554839 0.214532        latent_equal             6336
          distilbert-base-uncased      6       delta_pca  8                      33  0.859499 0.100527 fisher_output_equal            19008
          distilbert-base-uncased      6       delta_pca  8                      33  0.859499 0.100527        latent_equal             6336
          distilbert-base-uncased      6          random  1                      66  0.151579 0.180469 fisher_output_equal            19008
          distilbert-base-uncased      6          random  1                      66  0.151579 0.180469        latent_equal             6336
          distilbert-base-uncased      6          random  4                      66  0.586624 0.208065 fisher_output_equal            19008
          distilbert-base-uncased      6          random  4                      66  0.586624 0.208065        latent_equal             6336
          distilbert-base-uncased      6          random  8                      66  0.859172 0.145745 fisher_output_equal            19008
          distilbert-base-uncased      6          random  8                      66  0.859172 0.145745        latent_equal             6336
          distilbert-base-uncased      6       state_pca  1                      33  0.137801 0.124722 fisher_output_equal            19008
          distilbert-base-uncased      6       state_pca  1                      33  0.137801 0.124722        latent_equal             6336
          distilbert-base-uncased      6       state_pca  4                      33  0.569786 0.242347 fisher_output_equal            19008
          distilbert-base-uncased      6       state_pca  4                      33  0.569786 0.242347        latent_equal             6336
          distilbert-base-uncased      6       state_pca  8                      33  0.873800 0.123574 fisher_output_equal            19008
          distilbert-base-uncased      6       state_pca  8                      33  0.873800 0.123574        latent_equal             6336
google/bert_uncased_L-2_H-128_A-2      1       delta_pca  1                      33  0.133018 0.161129 fisher_output_equal            12672
google/bert_uncased_L-2_H-128_A-2      1       delta_pca  1                      33  0.133018 0.161129        latent_equal             4224
google/bert_uncased_L-2_H-128_A-2      1       delta_pca  4                      33  0.412848 0.182787 fisher_output_equal            12672
google/bert_uncased_L-2_H-128_A-2      1       delta_pca  4                      33  0.412848 0.182787        latent_equal             4224
google/bert_uncased_L-2_H-128_A-2      1       delta_pca  8                      33  0.800635 0.135160 fisher_output_equal            12672
google/bert_uncased_L-2_H-128_A-2      1       delta_pca  8                      33  0.800635 0.135160        latent_equal             4224
google/bert_uncased_L-2_H-128_A-2      1          random  1                      66  0.140323 0.141251 fisher_output_equal            12672
google/bert_uncased_L-2_H-128_A-2      1          random  1                      66  0.140323 0.141251        latent_equal             4224
google/bert_uncased_L-2_H-128_A-2      1          random  4                      66  0.477547 0.182191 fisher_output_equal            12672
google/bert_uncased_L-2_H-128_A-2      1          random  4                      66  0.477547 0.182191        latent_equal             4224
google/bert_uncased_L-2_H-128_A-2      1          random  8                      66  0.796623 0.134984 fisher_output_equal            12672
google/bert_uncased_L-2_H-128_A-2      1          random  8                      66  0.796623 0.134984        latent_equal             4224
```

## Files
```text
prompt_tables.csv
subspace_scores.csv
steering_records.csv
directionality_summary.csv
equal_output_energy_contrasts.csv
specificity_summary.csv
rho_dependency.csv
replication_matrix.csv
report.md
```
