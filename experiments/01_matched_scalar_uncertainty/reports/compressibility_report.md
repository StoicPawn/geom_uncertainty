# Layerwise k Compressibility and Accessibility Diagnostics

This report is computed from `results/layerwise_k_adjusted_rho` without rerunning model inference.

## Compressibility Summary
```text
 subspace  layer observed_condition   n  k50_mean  k80_mean  k80_missing_rate  rho_k1_mean  rho_kmax_mean  adjusted_auc_logk_mean  adjusted_initial_slope_mean
delta_pca      0            correct 183 16.524590 31.650273               0.0     0.063279            1.0               -0.030041                     0.026381
delta_pca      0              error 137 17.751825 31.299270               0.0     0.056382            1.0               -0.024288                     0.028408
delta_pca      1            correct 183 11.431694 25.748634               0.0     0.088684            1.0                0.013327                    -0.010835
delta_pca      1              error 137 14.014599 27.503650               0.0     0.074678            1.0                0.003436                    -0.008370
delta_pca      2            correct 183 13.814208 25.486339               0.0     0.096549            1.0               -0.053241                    -0.093497
delta_pca      2              error 137 15.240876 27.211679               0.0     0.057030            1.0               -0.036593                    -0.012392
delta_pca      3            correct 183  8.639344 19.868852               0.0     0.082474            1.0                0.091472                     0.166333
delta_pca      3              error 137 10.408759 24.525547               0.0     0.101690            1.0                0.022072                     0.033814
delta_pca      4            correct 183 13.502732 28.327869               0.0     0.068466            1.0                0.015342                     0.063145
delta_pca      4              error 137 12.189781 26.043796               0.0     0.089160            1.0                0.000281                     0.017223
delta_pca      5            correct 183  5.792350 15.841530               0.0     0.248084            1.0                0.063147                     0.003225
delta_pca      5              error 137 10.496350 24.116788               0.0     0.108827            1.0                0.013426                     0.010092
state_pca      0            correct 183 17.726776 26.054645               0.0     0.036711            1.0               -0.011016                     0.069091
state_pca      0              error 137 16.934307 25.576642               0.0     0.068930            1.0                0.021310                     0.073590
state_pca      1            correct 183 11.213115 21.989071               0.0     0.113144            1.0                0.046282                    -0.010056
state_pca      1              error 137 13.839416 28.437956               0.0     0.074357            1.0                0.000867                    -0.022214
state_pca      2            correct 183  9.442623 21.027322               0.0     0.136890            1.0                0.029639                    -0.025132
state_pca      2              error 137 11.605839 24.992701               0.0     0.113341            1.0                0.037669                     0.004962
state_pca      3            correct 183 10.978142 22.732240               0.0     0.135248            1.0                0.030989                     0.005012
state_pca      3              error 137 11.328467 22.554745               0.0     0.131836            1.0                0.032939                     0.014535
state_pca      4            correct 183 12.120219 26.770492               0.0     0.112928            1.0                0.038660                    -0.027745
state_pca      4              error 137 10.970803 23.299270               0.0     0.105284            1.0                0.033641                     0.005689
state_pca      5            correct 183  5.622951 13.890710               0.0     0.260394            1.0                0.070875                     0.001259
state_pca      5              error 137 10.357664 23.562044               0.0     0.083133            1.0                0.007671                     0.010823
state_pca      6            correct 183  4.934426 11.081967               0.0     0.174375            1.0               -0.025057                     0.064255
state_pca      6              error 137  9.810219 20.525547               0.0     0.112984            1.0               -0.001179                    -0.007235
```

## Same-Uncertainty / Different-Accessibility Pairs
```text
        subspace  k  layer  scalar_distance_z  rho_adjusted_abs_diff left_condition right_condition  left_rho_adjusted  right_rho_adjusted                             left_prompt                                                                right_prompt
delta_forward_1d  1      5           0.312585               1.050726        correct           error           0.734851           -0.315875  In Serbia, the capital city is [MASK].                                               Five minus two equals [MASK].
delta_forward_1d  1      5           0.187028               1.038839        correct           error           0.722964           -0.315875       The capital of Finland is [MASK].                                               Five minus two equals [MASK].
delta_forward_1d  1      5           0.159297               1.006537        correct           error           0.690662           -0.315875       Vietnam's capital city is [MASK].                                               Five minus two equals [MASK].
delta_forward_1d  1      5           0.332372               0.993497        correct           error           0.677622           -0.315875 In Hungary, the capital city is [MASK].                                               Five minus two equals [MASK].
delta_forward_1d  1      5           0.332171               0.984952        correct           error           0.669076           -0.315875    In Iraq, the capital city is [MASK].                                               Five minus two equals [MASK].
delta_forward_1d  1      5           0.322478               0.980452        correct           error           0.664577           -0.315875       The capital of Vietnam is [MASK].                                               Five minus two equals [MASK].
delta_forward_1d  1      5           0.319941               0.972342        correct         correct           0.757708           -0.214634 In Romania, the capital city is [MASK]. Gravity is famously linked to Newton and the apple; his surname was [MASK].
delta_forward_1d  1      5           0.188988               0.963339        correct           error           0.782311           -0.181028       Romania's capital city is [MASK].                                              The plural of mouse is [MASK].
       state_pca  2      6           0.089485               0.947304        correct         correct          -0.668157            0.279147      Bulgaria's capital city is [MASK].                                           Tunisia's capital city is [MASK].
delta_forward_1d  1      5           0.323110               0.947129        correct         correct           0.782311           -0.164818       Romania's capital city is [MASK].                                                     [MASK] is made by bees.
delta_forward_1d  1      5           0.240375               0.939751        correct           error           0.749933           -0.189818  More than one knife are called [MASK].                                                        A whale is a [MASK].
       state_pca  2      6           0.330257               0.927270        correct         correct          -0.668157            0.259113      Bulgaria's capital city is [MASK].                                     In Tunisia, the capital city is [MASK].
delta_forward_1d  1      5           0.230211               0.913260        correct           error           0.726491           -0.186770         Nepal's capital city is [MASK].                                  The detective Sherlock has surname [MASK].
delta_forward_1d  1      5           0.316683               0.911562        correct           error           0.724793           -0.186770         The capital of Nepal is [MASK].                                  The detective Sherlock has surname [MASK].
       state_pca  2      6           0.217262               0.908432        correct         correct          -0.668157            0.240275      Bulgaria's capital city is [MASK].                                          The capital of Thailand is [MASK].
       state_pca  2      6           0.391588               0.907404        correct         correct          -0.668157            0.239247      Bulgaria's capital city is [MASK].                                           The capital of Tunisia is [MASK].
       delta_pca  2      5           0.282391               0.907354        correct         correct          -0.359671            0.547683 In Belgium, the capital city is [MASK].                                            Norway's capital city is [MASK].
delta_forward_1d  1      5           0.319459               0.904617        correct           error           0.717847           -0.186770       Croatia's capital city is [MASK].                                  The detective Sherlock has surname [MASK].
delta_forward_1d  1      5           0.261439               0.902178        correct           error           0.715408           -0.186770       Finland's capital city is [MASK].                                  The detective Sherlock has surname [MASK].
delta_forward_1d  1      5           0.211087               0.898717        correct           error           0.582842           -0.315875          The capital of Iran is [MASK].                                               Five minus two equals [MASK].
       state_pca  2      5           0.255971               0.893256        correct           error           0.454889           -0.438367          Iran's capital city is [MASK].                                  The detective Sherlock has surname [MASK].
delta_forward_1d  1      5           0.176066               0.892694        correct           error           0.705924           -0.186770       The capital of Croatia is [MASK].                                  The detective Sherlock has surname [MASK].
       state_pca  2      6           0.394464               0.892514        correct         correct          -0.613367            0.279147       The capital of Finland is [MASK].                                           Tunisia's capital city is [MASK].
delta_forward_1d  1      5           0.332263               0.890686        correct           error           0.782311           -0.108375       Romania's capital city is [MASK].                                  The continent containing France is [MASK].
delta_forward_1d  1      5           0.328917               0.889998        correct           error           0.782311           -0.107687       Romania's capital city is [MASK].                                                        A shark is a [MASK].
delta_forward_1d  1      5           0.198828               0.889384        correct           error           0.573509           -0.315875        Russia's capital city is [MASK].                                               Five minus two equals [MASK].
delta_forward_1d  1      5           0.193099               0.887259        correct           error           0.782311           -0.104948       Romania's capital city is [MASK].                                  The continent containing Poland is [MASK].
       state_pca  2      6           0.240915               0.886861        correct         correct          -0.646586            0.240275 In Finland, the capital city is [MASK].                                          The capital of Thailand is [MASK].
       state_pca  2      6           0.007270               0.872205        correct         correct          -0.646586            0.225619 In Finland, the capital city is [MASK].                                          Thailand's capital city is [MASK].
       state_pca  2      6           0.112453               0.862706        correct         correct          -0.668157            0.194549      Bulgaria's capital city is [MASK].                                             Isaac [MASK] described gravity.
       state_pca  1      5           0.261439               0.862142        correct           error           0.612028           -0.250115       Finland's capital city is [MASK].                                  The detective Sherlock has surname [MASK].
       state_pca  2      5           0.332171               0.861702        correct           error           0.422965           -0.438736    In Iraq, the capital city is [MASK].                                               Five minus two equals [MASK].
       state_pca  2      6           0.120620               0.855244        correct         correct          -0.668157            0.187086      Bulgaria's capital city is [MASK].                                            Serbia's capital city is [MASK].
       state_pca  1      5           0.255971               0.853878        correct           error           0.603764           -0.250115          Iran's capital city is [MASK].                                  The detective Sherlock has surname [MASK].
       state_pca  2      6           0.138839               0.853642        correct         correct          -0.613367            0.240275       The capital of Finland is [MASK].                                          The capital of Thailand is [MASK].
       state_pca  2      6           0.094439               0.847798        correct         correct          -0.646586            0.201212 In Finland, the capital city is [MASK].                                             Nepal's capital city is [MASK].
       state_pca  2      6           0.355929               0.842016        correct         correct          -0.668157            0.173858      Bulgaria's capital city is [MASK].                                              Peru's capital city is [MASK].
       state_pca  2      6           0.095853               0.838986        correct         correct          -0.613367            0.225619       The capital of Finland is [MASK].                                          Thailand's capital city is [MASK].
       state_pca  2      5           0.261439               0.836001        correct           error           0.397633           -0.438367       Finland's capital city is [MASK].                                  The detective Sherlock has surname [MASK].
       state_pca  2      6           0.103116               0.835967        correct         correct          -0.668157            0.167809      Bulgaria's capital city is [MASK].                                    In Thailand, the capital city is [MASK].
```

## Outcome x Accessibility Counts
```text
        subspace  k  layer  median_threshold      outcome_accessibility  count
delta_forward_1d  1      0          0.002708 correct_high_accessibility     96
delta_forward_1d  1      0          0.002708  correct_low_accessibility     87
delta_forward_1d  1      0          0.002708   error_high_accessibility     64
delta_forward_1d  1      0          0.002708    error_low_accessibility     73
delta_forward_1d  1      1         -0.007448 correct_high_accessibility     90
delta_forward_1d  1      1         -0.007448  correct_low_accessibility     93
delta_forward_1d  1      1         -0.007448   error_high_accessibility     70
delta_forward_1d  1      1         -0.007448    error_low_accessibility     67
delta_forward_1d  1      2         -0.030417 correct_high_accessibility     93
delta_forward_1d  1      2         -0.030417  correct_low_accessibility     90
delta_forward_1d  1      2         -0.030417   error_high_accessibility     67
delta_forward_1d  1      2         -0.030417    error_low_accessibility     70
delta_forward_1d  1      3         -0.008869 correct_high_accessibility     84
delta_forward_1d  1      3         -0.008869  correct_low_accessibility     99
delta_forward_1d  1      3         -0.008869   error_high_accessibility     76
delta_forward_1d  1      3         -0.008869    error_low_accessibility     61
delta_forward_1d  1      4         -0.013875 correct_high_accessibility     76
delta_forward_1d  1      4         -0.013875  correct_low_accessibility    107
delta_forward_1d  1      4         -0.013875   error_high_accessibility     84
delta_forward_1d  1      4         -0.013875    error_low_accessibility     53
delta_forward_1d  1      5          0.067541 correct_high_accessibility    118
delta_forward_1d  1      5          0.067541  correct_low_accessibility     65
delta_forward_1d  1      5          0.067541   error_high_accessibility     42
delta_forward_1d  1      5          0.067541    error_low_accessibility     95
       delta_pca  2      0          0.005551 correct_high_accessibility    105
       delta_pca  2      0          0.005551  correct_low_accessibility     78
       delta_pca  2      0          0.005551   error_high_accessibility     78
       delta_pca  2      0          0.005551    error_low_accessibility     59
       delta_pca  2      1         -0.018288 correct_high_accessibility     88
       delta_pca  2      1         -0.018288  correct_low_accessibility     95
       delta_pca  2      1         -0.018288   error_high_accessibility     72
       delta_pca  2      1         -0.018288    error_low_accessibility     65
       delta_pca  2      2         -0.077839 correct_high_accessibility     88
       delta_pca  2      2         -0.077839  correct_low_accessibility     95
       delta_pca  2      2         -0.077839   error_high_accessibility     72
       delta_pca  2      2         -0.077839    error_low_accessibility     65
       delta_pca  2      3          0.097217 correct_high_accessibility    121
       delta_pca  2      3          0.097217  correct_low_accessibility     62
       delta_pca  2      3          0.097217   error_high_accessibility     39
       delta_pca  2      3          0.097217    error_low_accessibility     98
       delta_pca  2      4          0.009202 correct_high_accessibility    101
       delta_pca  2      4          0.009202  correct_low_accessibility     82
       delta_pca  2      4          0.009202   error_high_accessibility     59
       delta_pca  2      4          0.009202    error_low_accessibility     78
       delta_pca  2      5          0.039759 correct_high_accessibility    107
       delta_pca  2      5          0.039759  correct_low_accessibility     76
       delta_pca  2      5          0.039759   error_high_accessibility     53
       delta_pca  2      5          0.039759    error_low_accessibility     84
       delta_pca  4      0         -0.013726 correct_high_accessibility    108
       delta_pca  4      0         -0.013726  correct_low_accessibility     75
       delta_pca  4      0         -0.013726   error_high_accessibility     83
       delta_pca  4      0         -0.013726    error_low_accessibility     54
       delta_pca  4      1         -0.023731 correct_high_accessibility    100
       delta_pca  4      1         -0.023731  correct_low_accessibility     83
       delta_pca  4      1         -0.023731   error_high_accessibility     60
       delta_pca  4      1         -0.023731    error_low_accessibility     77
       delta_pca  4      2         -0.090397 correct_high_accessibility     79
       delta_pca  4      2         -0.090397  correct_low_accessibility    104
       delta_pca  4      2         -0.090397   error_high_accessibility     81
       delta_pca  4      2         -0.090397    error_low_accessibility     56
       delta_pca  4      3          0.118078 correct_high_accessibility    119
       delta_pca  4      3          0.118078  correct_low_accessibility     64
       delta_pca  4      3          0.118078   error_high_accessibility     41
       delta_pca  4      3          0.118078    error_low_accessibility     96
       delta_pca  4      4          0.009903 correct_high_accessibility     92
       delta_pca  4      4          0.009903  correct_low_accessibility     91
       delta_pca  4      4          0.009903   error_high_accessibility     68
       delta_pca  4      4          0.009903    error_low_accessibility     69
       delta_pca  4      5          0.084086 correct_high_accessibility    110
       delta_pca  4      5          0.084086  correct_low_accessibility     73
       delta_pca  4      5          0.084086   error_high_accessibility     50
       delta_pca  4      5          0.084086    error_low_accessibility     87
       state_pca  2      0         -0.041877 correct_high_accessibility    134
       state_pca  2      0         -0.041877  correct_low_accessibility     49
       state_pca  2      0         -0.041877   error_high_accessibility    110
       state_pca  2      0         -0.041877    error_low_accessibility     27
       state_pca  2      1         -0.014765 correct_high_accessibility    101
       state_pca  2      1         -0.014765  correct_low_accessibility     82
       state_pca  2      1         -0.014765   error_high_accessibility     59
       state_pca  2      1         -0.014765    error_low_accessibility     78
       state_pca  2      2          0.017843 correct_high_accessibility     95
       state_pca  2      2          0.017843  correct_low_accessibility     88
       state_pca  2      2          0.017843   error_high_accessibility     65
       state_pca  2      2          0.017843    error_low_accessibility     72
       state_pca  2      3         -0.000903 correct_high_accessibility     89
       state_pca  2      3         -0.000903  correct_low_accessibility     94
       state_pca  2      3         -0.000903   error_high_accessibility     71
       state_pca  2      3         -0.000903    error_low_accessibility     66
       state_pca  2      4         -0.016467 correct_high_accessibility     78
       state_pca  2      4         -0.016467  correct_low_accessibility    105
       state_pca  2      4         -0.016467   error_high_accessibility     82
       state_pca  2      4         -0.016467    error_low_accessibility     55
       state_pca  2      5          0.037508 correct_high_accessibility    114
       state_pca  2      5          0.037508  correct_low_accessibility     69
       state_pca  2      5          0.037508   error_high_accessibility     46
       state_pca  2      5          0.037508    error_low_accessibility     91
       state_pca  2      6         -0.020335 correct_high_accessibility     94
       state_pca  2      6         -0.020335  correct_low_accessibility     89
       state_pca  2      6         -0.020335   error_high_accessibility     66
       state_pca  2      6         -0.020335    error_low_accessibility     71
       state_pca  4      0         -0.027208 correct_high_accessibility     81
       state_pca  4      0         -0.027208  correct_low_accessibility    102
       state_pca  4      0         -0.027208   error_high_accessibility     82
       state_pca  4      0         -0.027208    error_low_accessibility     55
       state_pca  4      1          0.028754 correct_high_accessibility    110
       state_pca  4      1          0.028754  correct_low_accessibility     73
       state_pca  4      1          0.028754   error_high_accessibility     50
       state_pca  4      1          0.028754    error_low_accessibility     87
       state_pca  4      2          0.049927 correct_high_accessibility     84
       state_pca  4      2          0.049927  correct_low_accessibility     99
       state_pca  4      2          0.049927   error_high_accessibility     76
       state_pca  4      2          0.049927    error_low_accessibility     61
       state_pca  4      3          0.021353 correct_high_accessibility     83
       state_pca  4      3          0.021353  correct_low_accessibility    100
       state_pca  4      3          0.021353   error_high_accessibility     77
       state_pca  4      3          0.021353    error_low_accessibility     60
       state_pca  4      4          0.081712 correct_high_accessibility     97
       state_pca  4      4          0.081712  correct_low_accessibility     86
       state_pca  4      4          0.081712   error_high_accessibility     63
       state_pca  4      4          0.081712    error_low_accessibility     74
```

## Category Examples
```text
        subspace  k  layer                   category  rho_adjusted  rho_structured  rho_random_mean                                             prompt      target final_top1_token
delta_forward_1d  1      0    error_low_accessibility     -0.112045        0.065039         0.177083                                  Bees make [MASK].       honey            nests
delta_forward_1d  1      0    error_low_accessibility     -0.104006        0.000131         0.104136                                  Gold is a [MASK].       metal            color
delta_forward_1d  1      0    error_low_accessibility     -0.078005        0.000032         0.078037                        Japan is located in [MASK].        asia           taiwan
delta_forward_1d  1      0    error_low_accessibility     -0.077923        0.003005         0.080928                   [MASK] is famous for relativity.    einstein               it
delta_forward_1d  1      0    error_low_accessibility     -0.077921        0.000116         0.078037                      Germany is located in [MASK].      europe          austria
delta_forward_1d  1      0   error_high_accessibility      0.147670        0.244561         0.096891         The number of sides in a square is [MASK].        four             zero
delta_forward_1d  1      0   error_high_accessibility      0.137577        0.234468         0.096891       The number of sides in a triangle is [MASK].       three             zero
delta_forward_1d  1      0   error_high_accessibility      0.124491        0.175019         0.050527                 People drink coffee from a [MASK].         cup        starbucks
delta_forward_1d  1      0   error_high_accessibility      0.108699        0.159227         0.050527                   The language of China is [MASK].     chinese         mandarin
delta_forward_1d  1      0   error_high_accessibility      0.108201        0.158728         0.050527                  People usually sleep in a [MASK].         bed             cave
delta_forward_1d  1      0  correct_low_accessibility     -0.121691        0.055392         0.177083       Romeo and [MASK] are Shakespeare characters.      juliet           juliet
delta_forward_1d  1      0  correct_low_accessibility     -0.101260        0.002877         0.104136                           Earth orbits the [MASK].         sun              sun
delta_forward_1d  1      0  correct_low_accessibility     -0.077775        0.000262         0.078037                       Fish usually swim in [MASK].       water            water
delta_forward_1d  1      0  correct_low_accessibility     -0.077680        0.000357         0.078037                      Hamlet was written by [MASK]. shakespeare      shakespeare
delta_forward_1d  1      0  correct_low_accessibility     -0.075332        0.002705         0.078037                         Bread is made from [MASK].       flour            flour
delta_forward_1d  1      0 correct_high_accessibility      0.178894        0.275785         0.096891          The usual color of a clear sky is [MASK].        blue             blue
delta_forward_1d  1      0 correct_high_accessibility      0.122384        0.172912         0.050527               The Odyssey is attributed to [MASK].       homer            homer
delta_forward_1d  1      0 correct_high_accessibility      0.111714        0.162242         0.050527                  The language of Greece is [MASK].       greek            greek
delta_forward_1d  1      0 correct_high_accessibility      0.109075        0.159602         0.050527                   The capital of Greece is [MASK].      athens           athens
delta_forward_1d  1      0 correct_high_accessibility      0.106677        0.157204         0.050527                  The language of Turkey is [MASK].     turkish          turkish
delta_forward_1d  1      1    error_low_accessibility     -0.096650        0.011131         0.107781            Day has the opposite meaning of [MASK].       night              day
delta_forward_1d  1      1    error_low_accessibility     -0.089052        0.133438         0.222490                            The heart pumps [MASK].       blood           faster
delta_forward_1d  1      1    error_low_accessibility     -0.088831        0.005838         0.094669              [MASK] is a main ingredient in bread.       flour           garlic
delta_forward_1d  1      1    error_low_accessibility     -0.081016        0.001013         0.082029                         [MASK] is where fish swim.       water             this
delta_forward_1d  1      1    error_low_accessibility     -0.080954        0.004403         0.085357          Happy has the opposite meaning of [MASK].         sad        happiness
delta_forward_1d  1      1   error_high_accessibility      0.380230        0.441018         0.060787              More than one foot are called [MASK].        feet             toes
delta_forward_1d  1      1   error_high_accessibility      0.348524        0.396351         0.047828                     One third of twelve is [MASK].        four        preserved
delta_forward_1d  1      1   error_high_accessibility      0.317291        0.371251         0.053960                     The plural of goose is [MASK].       geese            goose
delta_forward_1d  1      1   error_high_accessibility      0.289953        0.351611         0.061658                      The plural of foot is [MASK].        feet           plural
delta_forward_1d  1      1   error_high_accessibility      0.284845        0.323464         0.038619                      The plural of leaf is [MASK].      leaves           plural
delta_forward_1d  1      1  correct_low_accessibility     -0.105513        0.000271         0.105784                    Isaac [MASK] described gravity.      newton           newton
delta_forward_1d  1      1  correct_low_accessibility     -0.088724        0.191367         0.280091              People in Hungary often speak [MASK].   hungarian        hungarian
delta_forward_1d  1      1  correct_low_accessibility     -0.086701        0.185336         0.272037              People in Romania often speak [MASK].    romanian         romanian
delta_forward_1d  1      1  correct_low_accessibility     -0.086412        0.017772         0.104184 [MASK] is traditionally credited with the Odyssey.       homer            homer
delta_forward_1d  1      1  correct_low_accessibility     -0.086244        0.044189         0.130433                    Milk often comes from a [MASK].         cow              cow
delta_forward_1d  1      1 correct_high_accessibility      0.423035        0.529713         0.106678                     Peru's capital city is [MASK].        lima             lima
delta_forward_1d  1      1 correct_high_accessibility      0.416782        0.543608         0.126825                    Nepal's capital city is [MASK].   kathmandu        kathmandu
delta_forward_1d  1      1 correct_high_accessibility      0.413134        0.522293         0.109159                  Vietnam's capital city is [MASK].       hanoi            hanoi
delta_forward_1d  1      1 correct_high_accessibility      0.394615        0.479121         0.084506                  Hungary's capital city is [MASK].    budapest         budapest
delta_forward_1d  1      1 correct_high_accessibility      0.386707        0.506886         0.120178                 Pakistan's capital city is [MASK].   islamabad        islamabad
delta_forward_1d  1      2    error_low_accessibility     -0.224264        0.002251         0.226515              [MASK] is a main ingredient in bread.       flour           garlic
delta_forward_1d  1      2    error_low_accessibility     -0.206888        0.019854         0.226742  [MASK] is the largest planet in the solar system.     jupiter            pluto
delta_forward_1d  1      2    error_low_accessibility     -0.200895        0.028349         0.229244             [MASK] is often called the red planet.        mars            pluto
delta_forward_1d  1      2    error_low_accessibility     -0.194444        0.000497         0.194940                         [MASK] is where fish swim.       water             this
delta_forward_1d  1      2    error_low_accessibility     -0.176748        0.038664         0.215413                   [MASK] is famous for relativity.    einstein               it
delta_forward_1d  1      2   error_high_accessibility      0.468061        0.676301         0.208240                      [MASK] painted the Mona Lisa.    leonardo               he
delta_forward_1d  1      2   error_high_accessibility      0.432339        0.662943         0.230604                  Ten divided by two equals [MASK].        five             zero
delta_forward_1d  1      2   error_high_accessibility      0.392388        0.625514         0.233126             Twelve divided by three equals [MASK].        four           equals
delta_forward_1d  1      2   error_high_accessibility      0.277082        0.546019         0.268937                      Five minus two equals [MASK].       three             zero
delta_forward_1d  1      2   error_high_accessibility      0.263422        0.297389         0.033967  Someone who acts in films can be called a [MASK].       actor          villain
delta_forward_1d  1      2  correct_low_accessibility     -0.235263        0.001082         0.236344                            [MASK] is made by bees.       honey            honey
delta_forward_1d  1      2  correct_low_accessibility     -0.218393        0.040669         0.259062              More than one leaf are called [MASK].      leaves           leaves
delta_forward_1d  1      2  correct_low_accessibility     -0.215229        0.001118         0.216346              People in Germany often speak [MASK].      german           german
delta_forward_1d  1      2  correct_low_accessibility     -0.207714        0.000451         0.208165               People in France often speak [MASK].      french           french
delta_forward_1d  1      2  correct_low_accessibility     -0.201061        0.003174         0.204235                The title pair is Romeo and [MASK].      juliet           juliet
delta_forward_1d  1      2 correct_high_accessibility      0.359048        0.403700         0.044652          Sherlock [MASK] is a fictional detective.      holmes           holmes
delta_forward_1d  1      2 correct_high_accessibility      0.311034        0.560172         0.249138       Romeo and [MASK] are Shakespeare characters.      juliet           juliet
delta_forward_1d  1      2 correct_high_accessibility      0.213430        0.309652         0.096222               In Cuba, the capital city is [MASK].      havana           havana
delta_forward_1d  1      2 correct_high_accessibility      0.208079        0.272493         0.064414                         Bread is made from [MASK].       flour            flour
delta_forward_1d  1      2 correct_high_accessibility      0.205745        0.313884         0.108139            In Jamaica, the capital city is [MASK].    kingston         kingston
delta_forward_1d  1      3    error_low_accessibility     -0.241158        0.087507         0.328666             Twelve divided by three equals [MASK].        four           equals
delta_forward_1d  1      3    error_low_accessibility     -0.207419        0.001412         0.208831                   [MASK] is famous for relativity.    einstein               it
delta_forward_1d  1      3    error_low_accessibility     -0.200574        0.126821         0.327395                  Ten divided by two equals [MASK].        five             zero
delta_forward_1d  1      3    error_low_accessibility     -0.200046        0.070406         0.270452            Subtracting two from five gives [MASK].       three             zero
delta_forward_1d  1      3    error_low_accessibility     -0.177885        0.004060         0.181946             People in Britain pay with the [MASK].       pound             euro
delta_forward_1d  1      3   error_high_accessibility      0.492152        0.568660         0.076508                       A triangle has [MASK] sides.       three              two
delta_forward_1d  1      3   error_high_accessibility      0.443920        0.585712         0.141792             Up has the opposite meaning of [MASK].        down               up
delta_forward_1d  1      3   error_high_accessibility      0.405042        0.535687         0.130645           Fast has the opposite meaning of [MASK].        slow             fast
delta_forward_1d  1      3   error_high_accessibility      0.392660        0.522296         0.129636            Old has the opposite meaning of [MASK].         new              old
delta_forward_1d  1      3   error_high_accessibility      0.388262        0.511764         0.123502               The currency of China is the [MASK].        yuan           dollar
delta_forward_1d  1      3  correct_low_accessibility     -0.120873        0.000948         0.121822               People in Spain pay with the [MASK].        euro             euro
delta_forward_1d  1      3  correct_low_accessibility     -0.119407        0.000486         0.119893              People in France pay with the [MASK].        euro             euro
delta_forward_1d  1      3  correct_low_accessibility     -0.119103        0.002203         0.121306             People in Germany pay with the [MASK].        euro             euro
delta_forward_1d  1      3  correct_low_accessibility     -0.111621        0.007787         0.119407               People in Italy pay with the [MASK].        euro             euro
delta_forward_1d  1      3  correct_low_accessibility     -0.111215        0.001467         0.112682                  The language of Poland is [MASK].      polish           polish
delta_forward_1d  1      3 correct_high_accessibility      0.390123        0.528249         0.138126               The currency of Italy is the [MASK].        euro             euro
delta_forward_1d  1      3 correct_high_accessibility      0.383497        0.553500         0.170003              More than one leaf are called [MASK].      leaves           leaves
delta_forward_1d  1      3 correct_high_accessibility      0.380640        0.510783         0.130143                         A square has [MASK] sides.        four             four
delta_forward_1d  1      3 correct_high_accessibility      0.365833        0.524531         0.158698                            [MASK] is made by bees.       honey            honey
delta_forward_1d  1      3 correct_high_accessibility      0.363686        0.565366         0.201681             The currency of Britain is the [MASK].       pound            pound
```
