# Test 2: Equal-Output-Movement Steering

This test asks whether accessible directions change uncertainty more per unit Fisher-output movement than matched alternatives.

All rows use matched `||F^{1/2}J delta z||`; the main readout is uncertainty movement divided by that output movement.

## Direction Summary
```text
      source               direction subspace_family     sign   n  fisher_output_energy_mean  entropy_efficiency_mean  varentropy_efficiency_mean  abs_delta_entropy_mean  abs_delta_varentropy_mean  selected_top1_changed_rate  top10_jaccard_mean
decoder_main           accessible_ls       delta_pca decrease  36                       0.05                 0.833187                    1.239772                0.041659                   0.061989                    0.000000            0.939394
decoder_main           accessible_ls       delta_pca increase  36                       0.05                 0.905636                    1.417021                0.045282                   0.070851                    0.027778            0.954545
decoder_main           accessible_ls          random decrease  36                       0.05                 0.806149                    1.143702                0.040307                   0.057185                    0.000000            0.940236
decoder_main           accessible_ls          random increase  36                       0.05                 0.933409                    1.466247                0.046670                   0.073312                    0.027778            0.944444
decoder_main           accessible_ls       state_pca decrease  36                       0.05                 0.849421                    1.255278                0.042471                   0.062764                    0.000000            0.954545
decoder_main           accessible_ls       state_pca increase  36                       0.05                 0.917771                    1.416444                0.045889                   0.070822                    0.027778            0.954545
decoder_main grad_orthogonal_control       delta_pca decrease  36                       0.05                 0.123144                    0.452231                0.006157                   0.022612                    0.055556            0.939394
decoder_main grad_orthogonal_control       delta_pca increase  36                       0.05                 0.083176                    0.460222                0.004159                   0.023011                    0.027778            0.925084
decoder_main grad_orthogonal_control          random decrease  36                       0.05                 0.093512                    0.514381                0.004676                   0.025719                    0.000000            0.954545
decoder_main grad_orthogonal_control          random increase  36                       0.05                 0.087331                    0.377751                0.004367                   0.018888                    0.027778            0.921717
decoder_main grad_orthogonal_control       state_pca decrease  36                       0.05                 0.063649                    0.324411                0.003182                   0.016221                    0.055556            0.914983
decoder_main grad_orthogonal_control       state_pca increase  36                       0.05                 0.063423                    0.376763                0.003171                   0.018838                    0.000000            0.920875
decoder_main          random_control       delta_pca decrease  36                       0.05                 0.403903                    0.778556                0.020195                   0.038928                    0.027778            0.891414
decoder_main          random_control       delta_pca increase  36                       0.05                 0.427313                    0.709616                0.021366                   0.035481                    0.000000            0.969697
decoder_main          random_control          random decrease  36                       0.05                 0.365232                    0.590614                0.018262                   0.029531                    0.055556            0.939394
decoder_main          random_control          random increase  36                       0.05                 0.402225                    0.706744                0.020111                   0.035337                    0.000000            0.920034
decoder_main          random_control       state_pca decrease  36                       0.05                 0.420560                    0.810597                0.021028                   0.040530                    0.083333            0.929293
decoder_main          random_control       state_pca increase  36                       0.05                 0.416212                    0.795092                0.020811                   0.039755                    0.000000            0.935185
  mlm_topk32           accessible_ls       delta_pca decrease 115                       0.05                 0.642361                    1.345154                0.032118                   0.067258                    0.017391            0.946772
  mlm_topk32           accessible_ls       delta_pca increase 115                       0.05                 0.599816                    1.289804                0.029991                   0.064490                    0.026087            0.951252
  mlm_topk32           accessible_ls          random decrease 184                       0.05                 0.689887                    1.287637                0.034494                   0.064382                    0.027174            0.953722
  mlm_topk32           accessible_ls          random increase 184                       0.05                 0.664679                    1.269437                0.033234                   0.063472                    0.016304            0.946146
  mlm_topk32           accessible_ls       state_pca decrease 184                       0.05                 0.700827                    1.278350                0.035041                   0.063918                    0.027174            0.955863
  mlm_topk32           accessible_ls       state_pca increase 184                       0.05                 0.689296                    1.276085                0.034465                   0.063804                    0.021739            0.948946
  mlm_topk32 grad_orthogonal_control       delta_pca decrease 115                       0.05                 0.020018                    0.117745                0.001001                   0.005887                    0.017391            0.931489
  mlm_topk32 grad_orthogonal_control       delta_pca increase 115                       0.05                 0.020068                    0.108851                0.001003                   0.005443                    0.078261            0.921212
  mlm_topk32 grad_orthogonal_control          random decrease 184                       0.05                 0.024196                    0.147854                0.001210                   0.007393                    0.048913            0.924204
  mlm_topk32 grad_orthogonal_control          random increase 184                       0.05                 0.024259                    0.148667                0.001213                   0.007433                    0.032609            0.941370
  mlm_topk32 grad_orthogonal_control       state_pca decrease 184                       0.05                 0.022190                    0.184956                0.001110                   0.009248                    0.038043            0.932312
  mlm_topk32 grad_orthogonal_control       state_pca increase 184                       0.05                 0.022544                    0.200629                0.001127                   0.010031                    0.043478            0.934289
  mlm_topk32          random_control       delta_pca decrease 115                       0.05                 0.257355                    0.546587                0.012868                   0.027329                    0.043478            0.922003
  mlm_topk32          random_control       delta_pca increase 115                       0.05                 0.228665                    0.519081                0.011433                   0.025954                    0.043478            0.937022
  mlm_topk32          random_control          random decrease 184                       0.05                 0.269637                    0.586087                0.013482                   0.029304                    0.038043            0.931489
  mlm_topk32          random_control          random increase 184                       0.05                 0.251917                    0.574962                0.012596                   0.028748                    0.038043            0.933630
  mlm_topk32          random_control       state_pca decrease 184                       0.05                 0.286966                    0.572071                0.014348                   0.028604                    0.059783            0.933136
  mlm_topk32          random_control       state_pca increase 184                       0.05                 0.266929                    0.555893                0.013346                   0.027795                    0.032609            0.928157
```

## Accessible Vs Controls
```text
      source subspace_family     sign                 control                     metric  accessible  control_value  accessible_minus_control  accessible_over_control
decoder_main       delta_pca decrease grad_orthogonal_control    entropy_efficiency_mean    0.833187       0.123144                  0.710043                 6.765938
decoder_main       delta_pca decrease grad_orthogonal_control varentropy_efficiency_mean    1.239772       0.452231                  0.787541                 2.741459
decoder_main       delta_pca decrease          random_control    entropy_efficiency_mean    0.833187       0.403903                  0.429284                 2.062841
decoder_main       delta_pca decrease          random_control varentropy_efficiency_mean    1.239772       0.778556                  0.461215                 1.592398
decoder_main       delta_pca increase grad_orthogonal_control    entropy_efficiency_mean    0.905636       0.083176                  0.822461                10.888247
decoder_main       delta_pca increase grad_orthogonal_control varentropy_efficiency_mean    1.417021       0.460222                  0.956798                 3.078993
decoder_main       delta_pca increase          random_control    entropy_efficiency_mean    0.905636       0.427313                  0.478324                 2.119375
decoder_main       delta_pca increase          random_control varentropy_efficiency_mean    1.417021       0.709616                  0.707405                 1.996884
decoder_main          random decrease grad_orthogonal_control    entropy_efficiency_mean    0.806149       0.093512                  0.712637                 8.620836
decoder_main          random decrease grad_orthogonal_control varentropy_efficiency_mean    1.143702       0.514381                  0.629321                 2.223454
decoder_main          random decrease          random_control    entropy_efficiency_mean    0.806149       0.365232                  0.440916                 2.207221
decoder_main          random decrease          random_control varentropy_efficiency_mean    1.143702       0.590614                  0.553088                 1.936463
decoder_main          random increase grad_orthogonal_control    entropy_efficiency_mean    0.933409       0.087331                  0.846078                10.688125
decoder_main          random increase grad_orthogonal_control varentropy_efficiency_mean    1.466247       0.377751                  1.088496                 3.881520
decoder_main          random increase          random_control    entropy_efficiency_mean    0.933409       0.402225                  0.531184                 2.320614
decoder_main          random increase          random_control varentropy_efficiency_mean    1.466247       0.706744                  0.759503                 2.074651
decoder_main       state_pca decrease grad_orthogonal_control    entropy_efficiency_mean    0.849421       0.063649                  0.785772                13.345379
decoder_main       state_pca decrease grad_orthogonal_control varentropy_efficiency_mean    1.255278       0.324411                  0.930867                 3.869403
decoder_main       state_pca decrease          random_control    entropy_efficiency_mean    0.849421       0.420560                  0.428861                 2.019740
decoder_main       state_pca decrease          random_control varentropy_efficiency_mean    1.255278       0.810597                  0.444681                 1.548584
decoder_main       state_pca increase grad_orthogonal_control    entropy_efficiency_mean    0.917771       0.063423                  0.854349                14.470746
decoder_main       state_pca increase grad_orthogonal_control varentropy_efficiency_mean    1.416444       0.376763                  1.039681                 3.759509
decoder_main       state_pca increase          random_control    entropy_efficiency_mean    0.917771       0.416212                  0.501560                 2.205058
decoder_main       state_pca increase          random_control varentropy_efficiency_mean    1.416444       0.795092                  0.621352                 1.781485
  mlm_topk32       delta_pca decrease grad_orthogonal_control    entropy_efficiency_mean    0.642361       0.020018                  0.622343                32.089739
  mlm_topk32       delta_pca decrease grad_orthogonal_control varentropy_efficiency_mean    1.345154       0.117745                  1.227408                11.424284
  mlm_topk32       delta_pca decrease          random_control    entropy_efficiency_mean    0.642361       0.257355                  0.385006                 2.496015
  mlm_topk32       delta_pca decrease          random_control varentropy_efficiency_mean    1.345154       0.546587                  0.798566                 2.461005
  mlm_topk32       delta_pca increase grad_orthogonal_control    entropy_efficiency_mean    0.599816       0.020068                  0.579748                29.889108
  mlm_topk32       delta_pca increase grad_orthogonal_control varentropy_efficiency_mean    1.289804       0.108851                  1.180953                11.849294
  mlm_topk32       delta_pca increase          random_control    entropy_efficiency_mean    0.599816       0.228665                  0.371151                 2.623120
  mlm_topk32       delta_pca increase          random_control varentropy_efficiency_mean    1.289804       0.519081                  0.770723                 2.484784
  mlm_topk32          random decrease grad_orthogonal_control    entropy_efficiency_mean    0.689887       0.024196                  0.665691                28.512624
  mlm_topk32          random decrease grad_orthogonal_control varentropy_efficiency_mean    1.287637       0.147854                  1.139783                 8.708844
  mlm_topk32          random decrease          random_control    entropy_efficiency_mean    0.689887       0.269637                  0.420250                 2.558574
  mlm_topk32          random decrease          random_control varentropy_efficiency_mean    1.287637       0.586087                  0.701549                 2.197004
  mlm_topk32          random increase grad_orthogonal_control    entropy_efficiency_mean    0.664679       0.024259                  0.640420                27.399196
  mlm_topk32          random increase grad_orthogonal_control varentropy_efficiency_mean    1.269437       0.148667                  1.120771                 8.538821
  mlm_topk32          random increase          random_control    entropy_efficiency_mean    0.664679       0.251917                  0.412762                 2.638484
  mlm_topk32          random increase          random_control varentropy_efficiency_mean    1.269437       0.574962                  0.694475                 2.207864
  mlm_topk32       state_pca decrease grad_orthogonal_control    entropy_efficiency_mean    0.700827       0.022190                  0.678637                31.582890
  mlm_topk32       state_pca decrease grad_orthogonal_control varentropy_efficiency_mean    1.278350       0.184956                  1.093394                 6.911631
  mlm_topk32       state_pca decrease          random_control    entropy_efficiency_mean    0.700827       0.286966                  0.413861                 2.442199
  mlm_topk32       state_pca decrease          random_control varentropy_efficiency_mean    1.278350       0.572071                  0.706279                 2.234599
  mlm_topk32       state_pca increase grad_orthogonal_control    entropy_efficiency_mean    0.689296       0.022544                  0.666752                30.576038
  mlm_topk32       state_pca increase grad_orthogonal_control varentropy_efficiency_mean    1.276085       0.200629                  1.075456                 6.360418
  mlm_topk32       state_pca increase          random_control    entropy_efficiency_mean    0.689296       0.266929                  0.422367                 2.582323
  mlm_topk32       state_pca increase          random_control varentropy_efficiency_mean    1.276085       0.555893                  0.720192                 2.295557
```

## Preservation
```text
      source               direction     sign   n  selected_top1_changed_rate  target_correct_changed_rate  top5_jaccard_mean  top10_jaccard_mean  selected_kl_mean
decoder_main           accessible_ls decrease 108                    0.000000                          0.0           0.972222            0.944725               NaN
decoder_main           accessible_ls increase 108                    0.027778                          0.0           0.972222            0.951178               NaN
decoder_main grad_orthogonal_control decrease 108                    0.037037                          0.0           0.917549            0.936308               NaN
decoder_main grad_orthogonal_control increase 108                    0.018519                          0.0           0.942240            0.922559               NaN
decoder_main          random_control decrease 108                    0.055556                          0.0           0.953704            0.920034               NaN
decoder_main          random_control increase 108                    0.000000                          0.0           0.938272            0.941639               NaN
  mlm_topk32           accessible_ls decrease 483                    0.024845                          0.0           0.959282            0.952883          0.001227
  mlm_topk32           accessible_ls increase 483                    0.020704                          0.0           0.961353            0.948428          0.001274
  mlm_topk32 grad_orthogonal_control decrease 483                    0.037267                          0.0           0.936705            0.929027          0.001249
  mlm_topk32 grad_orthogonal_control increase 483                    0.047619                          0.0           0.934832            0.933873          0.001251
  mlm_topk32          random_control decrease 483                    0.047619                          0.0           0.942719            0.929858          0.001238
  mlm_topk32          random_control increase 483                    0.037267                          0.0           0.940550            0.932353          0.001261
```

## Files
```text
equal_output_efficiency_records.csv
equal_output_direction_summary.csv
equal_output_control_contrasts.csv
equal_output_preservation_summary.csv
baseline_availability.csv
report.md
```
