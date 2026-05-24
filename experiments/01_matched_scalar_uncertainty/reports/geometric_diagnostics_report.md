# Geometric Diagnostics from Existing Outputs

Source: existing factual-error deep CSV outputs only. No model inference is run by this script.

## Same-Uncertainty / Different-Rho Candidate Pairs

Pairs are matched within model, layer, and subspace by low z-scored distance over entropy, varentropy, trace-FIM, gradient norm, and top-1 probability, then sorted by rho difference.
```text
            model  layer subspace  scalar_distance_z  rho_abs_diff left_condition right_condition  left_rho  right_rho                              left_prompt                                       right_prompt
bert-base-uncased     11 state_1d           0.404676      0.977057        correct           error  0.000070   0.977127        Jamaica's capital city is [MASK].             [MASK] is often called the red planet.
bert-base-uncased     11 state_1d           0.330625      0.977039        correct           error  0.000088   0.977127 In Thailand, the capital city is [MASK].             [MASK] is often called the red planet.
bert-base-uncased     11 state_1d           0.289790      0.976993        correct           error  0.000134   0.977127           Cuba's capital city is [MASK].             [MASK] is often called the red planet.
bert-base-uncased     11 delta_1d           0.446241      0.974819        correct           error  0.974976   0.000157 Left has the opposite meaning of [MASK]. [MASK] is traditionally credited with the Odyssey.
bert-base-uncased     12 state_1d           0.620109      0.973702        correct           error  0.974053   0.000351         The capital of Serbia is [MASK].                            [MASK] is made by bees.
bert-base-uncased     11 delta_1d           0.573978      0.971775        correct         correct  0.003201   0.974976          The capital of Chile is [MASK].           Left has the opposite meaning of [MASK].
bert-base-uncased     11 state_1d           0.373585      0.970555        correct           error  0.006572   0.977127           The capital of Cuba is [MASK].             [MASK] is often called the red planet.
bert-base-uncased     11 state_1d           0.334339      0.969427        correct           error  0.007700   0.977127       The capital of Thailand is [MASK].             [MASK] is often called the red planet.
bert-base-uncased     11 state_1d           0.354886      0.968434        correct           error  0.008693   0.977127           Iraq's capital city is [MASK].             [MASK] is often called the red planet.
bert-base-uncased     11 delta_1d           0.501987      0.967001        correct           error  0.967158   0.000157          The opposite of left is [MASK]. [MASK] is traditionally credited with the Odyssey.
bert-base-uncased     12 delta_1d           0.322912      0.966311        correct         correct  0.966363   0.000052        Vietnam's capital city is [MASK].                    The capital of Chile is [MASK].
bert-base-uncased     11 state_1d           0.313297      0.962849        correct           error  0.014278   0.977127     In Cuba, the capital city is [MASK].             [MASK] is often called the red planet.
bert-base-uncased     10 state_1d           0.049961      0.962319          error         correct  0.972027   0.009708                  [MASK] is made by bees.       Romeo and [MASK] are Shakespeare characters.
bert-base-uncased     11 state_1d           0.321768      0.962315        correct           error  0.014812   0.977127        Hungary's capital city is [MASK].             [MASK] is often called the red planet.
bert-base-uncased     11 state_1d           0.687265      0.960347        correct           error  0.016780   0.977127          Nepal's capital city is [MASK].             [MASK] is often called the red planet.
bert-base-uncased     12 delta_1d           0.658845      0.958095        correct         correct  0.008267   0.966363       Portugal's capital city is [MASK].                  Vietnam's capital city is [MASK].
bert-base-uncased     12 state_1d           0.507803      0.954277          error         correct  0.000351   0.954628                  [MASK] is made by bees.                    The opposite of left is [MASK].
bert-base-uncased     12 delta_1d           0.385817      0.951107        correct         correct  0.951158   0.000052        The capital of Vietnam is [MASK].                    The capital of Chile is [MASK].
bert-base-uncased     12 state_1d           0.301521      0.948481        correct           error  0.948832   0.000351   In Serbia, the capital city is [MASK].                            [MASK] is made by bees.
bert-base-uncased     11 state_1d           0.275629      0.947578        correct         correct  0.947648   0.000070           The capital of Iran is [MASK].                  Jamaica's capital city is [MASK].
bert-base-uncased     11 state_1d           0.278419      0.947560        correct         correct  0.947648   0.000088           The capital of Iran is [MASK].           In Thailand, the capital city is [MASK].
bert-base-uncased     11 state_1d           0.235208      0.947514        correct         correct  0.947648   0.000134           The capital of Iran is [MASK].                     Cuba's capital city is [MASK].
bert-base-uncased     12 state_1d           0.469647      0.946482        correct           error  0.946833   0.000351        The capital of Lebanon is [MASK].                            [MASK] is made by bees.
bert-base-uncased     12 delta_1d           0.694476      0.942891        correct         correct  0.008267   0.951158       Portugal's capital city is [MASK].                  The capital of Vietnam is [MASK].
bert-base-uncased     12 delta_1d           0.166829      0.941150        correct         correct  0.000010   0.941160        The capital of Finland is [MASK].                    The capital of Nepal is [MASK].
```

## Error/Correct Matching Controlled by Scalars

Each error prompt is nearest-neighbor matched to a correct prompt using either final-layer scalar controls or all scalar layer-profile controls.
```text
                            model       match_controls  n_error_prompts  n_correct_pool  scalar_distance_z_mean  scalar_distance_z_median  scalar_distance_z_p90  rho_delta_delta_mean  rho_delta_delta_median  rho_delta_win_rate_error_gt_correct  rho_state_delta_mean  rho_state_delta_median  rho_state_win_rate_error_gt_correct  rho_state_delta_delta_mean  rho_state_delta_delta_median  rho_state_delta_win_rate_error_gt_correct  rho_random_delta_mean  rho_random_delta_median  rho_random_win_rate_error_gt_correct
                bert-base-uncased         scalar_final              132             188                0.439799                  0.415253               0.667701              0.019123               -0.001672                             0.492424              0.041908                0.018518                             0.553030                    0.030971                      0.001873                                   0.507576               0.026850                 0.009659                              0.575758
                bert-base-uncased scalar_layer_profile              132             188                5.715038                  5.205708               9.824513              0.028788                0.003335                             0.545455             -0.002786               -0.018915                             0.371212                    0.012369                     -0.009762                                   0.416667               0.016626                 0.008557                              0.568182
          distilbert-base-uncased scalar_layer_profile              137             183                3.239510                  2.530291               5.784841             -0.034914               -0.038365                             0.291971              0.010935                0.010201                             0.562044                   -0.010226                     -0.015445                                   0.430657              -0.007473                -0.006916                              0.401460
google/bert_uncased_L-2_H-128_A-2 scalar_layer_profile              300              20                2.562720                  2.299036               4.141893             -0.007813               -0.006092                             0.453333             -0.007979               -0.010810                             0.426667                   -0.007913                     -0.008665                                   0.426667              -0.000344                -0.002661                              0.476667
```

## Layer Trajectories

Layer-wise means by observed condition and structured subspace.
```text
            model observed_condition  layer subspace_family   n  rho_mean  rho_std  entropy_mean  varentropy_mean  trace_fim_mean  grad_norm_sq_mean  top1_prob_mean
bert-base-uncased            correct      0        state_1d 188  0.085447 0.048323      3.302093         0.335259        0.020911           0.000626        0.091038
bert-base-uncased            correct      1        delta_1d 188  0.061594 0.075347      3.365465         0.235280        0.022690           0.000291        0.092652
bert-base-uncased            correct      1        state_1d 188  0.075960 0.087217      3.365465         0.235280        0.003857           0.000056        0.092652
bert-base-uncased            correct      2        delta_1d 188  0.098694 0.104056      3.171138         0.738423        0.017858           0.002172        0.182620
bert-base-uncased            correct      2        state_1d 188  0.041440 0.096864      3.171138         0.738423        0.000638           0.000028        0.182620
bert-base-uncased            correct      3        delta_1d 188  0.059984 0.107427      3.039107         0.971849        0.042074           0.002337        0.194081
bert-base-uncased            correct      3        state_1d 188  0.102041 0.138597      3.039107         0.971849        0.000492           0.000061        0.194081
bert-base-uncased            correct      4        delta_1d 188  0.048055 0.072321      3.084119         0.792530        0.010291           0.000493        0.161980
bert-base-uncased            correct      4        state_1d 188  0.254981 0.162206      3.084119         0.792530        0.000187           0.000048        0.161980
bert-base-uncased            correct      5        delta_1d 188  0.054388 0.088707      3.049505         0.860721        0.016755           0.000981        0.167106
bert-base-uncased            correct      5        state_1d 188  0.332714 0.205828      3.049505         0.860721        0.000170           0.000067        0.167106
bert-base-uncased            correct      6        delta_1d 188  0.139436 0.145107      2.851449         1.448529        0.012559           0.003770        0.280948
bert-base-uncased            correct      6        state_1d 188  0.236672 0.174996      2.851449         1.448529        0.000083           0.000034        0.280948
bert-base-uncased            correct      7        delta_1d 188  0.230633 0.216355      2.854812         1.395691        0.011399           0.004365        0.259037
bert-base-uncased            correct      7        state_1d 188  0.105122 0.115941      2.854812         1.395691        0.000106           0.000022        0.259037
bert-base-uncased            correct      8        delta_1d 188  0.180799 0.175698      2.977973         1.115350        0.009423           0.002525        0.228146
bert-base-uncased            correct      8        state_1d 188  0.092272 0.110662      2.977973         1.115350        0.000072           0.000009        0.228146
bert-base-uncased            correct      9        delta_1d 188  0.140750 0.152181      3.106568         0.779921        0.012234           0.002136        0.177736
bert-base-uncased            correct      9        state_1d 188  0.153154 0.170622      3.106568         0.779921        0.000089           0.000015        0.177736
bert-base-uncased            correct     10        delta_1d 188  0.395298 0.268754      2.445078         1.701983        0.015840           0.014048        0.381201
bert-base-uncased            correct     10        state_1d 188  0.295164 0.308178      2.445078         1.701983        0.000064           0.000052        0.381201
bert-base-uncased            correct     11        delta_1d 188  0.451761 0.295009      1.105285         1.533901        0.014868           0.011443        0.697845
bert-base-uncased            correct     11        state_1d 188  0.392021 0.292246      1.105285         1.533901        0.000026           0.000024        0.697845
bert-base-uncased            correct     12        delta_1d 188  0.389754 0.291062      1.003997         1.572256        0.005020           0.003557        0.735740
bert-base-uncased            correct     12        state_1d 188  0.675144 0.279501      1.003997         1.572256        0.000644           0.001068        0.735740
bert-base-uncased              error      0        state_1d 132  0.071258 0.048247      3.271652         0.404252        0.020778           0.000514        0.101080
bert-base-uncased              error      1        delta_1d 132  0.065267 0.084888      3.323756         0.322278        0.020374           0.000471        0.104603
bert-base-uncased              error      1        state_1d 132  0.105359 0.112209      3.323756         0.322278        0.003221           0.000097        0.104603
bert-base-uncased              error      2        delta_1d 132  0.150317 0.158887      3.173414         0.688094        0.022507           0.003600        0.163872
bert-base-uncased              error      2        state_1d 132  0.108989 0.179495      3.173414         0.688094        0.000698           0.000095        0.163872
bert-base-uncased              error      3        delta_1d 132  0.104433 0.163063      3.060214         0.770849        0.039425           0.003848        0.175997
bert-base-uncased              error      3        state_1d 132  0.152904 0.213270      3.060214         0.770849        0.000485           0.000061        0.175997
bert-base-uncased              error      4        delta_1d 132  0.133040 0.189270      2.952001         0.874275        0.008336           0.001169        0.211506
bert-base-uncased              error      4        state_1d 132  0.294367 0.249944      2.952001         0.874275        0.000182           0.000059        0.211506
bert-base-uncased              error      5        delta_1d 132  0.091504 0.137741      2.931916         0.874952        0.015978           0.001404        0.210245
bert-base-uncased              error      5        state_1d 132  0.260562 0.218034      2.931916         0.874952        0.000126           0.000035        0.210245
bert-base-uncased              error      6        delta_1d 132  0.172880 0.191664      2.788801         1.139416        0.011286           0.003289        0.263901
bert-base-uncased              error      6        state_1d 132  0.227142 0.216190      2.788801         1.139416        0.000094           0.000030        0.263901
bert-base-uncased              error      7        delta_1d 132  0.158680 0.207815      2.738102         1.151609        0.010546           0.002125        0.273910
bert-base-uncased              error      7        state_1d 132  0.245384 0.278648      2.738102         1.151609        0.000094           0.000049        0.273910
```

## Existing Saturation / Full-Space Check

This uses only already-computed subspaces. A true k-sweep is done in the intervention/ablation script.
```text
                            model subspace_family  subspace_dim  rho_mean  rho_std_mean  trace_fim_mean
                bert-base-uncased        delta_1d             1  0.176662  1.774464e-01        0.015669
                bert-base-uncased            full           768  1.000000  2.984655e-15       15.507734
                bert-base-uncased        random_2             2  0.207553  1.688144e-01        0.039717
                bert-base-uncased        state_1d             1  0.221303  1.945928e-01        0.002119
          distilbert-base-uncased        delta_1d             1  0.178820  1.739172e-01        0.019081
          distilbert-base-uncased            full           768  1.000000  2.013527e-15       28.925329
          distilbert-base-uncased        random_2             2  0.203146  1.693060e-01        0.079142
          distilbert-base-uncased        state_1d             1  0.232725  1.906985e-01        0.000996
google/bert_uncased_L-2_H-128_A-2        delta_1d             1  0.146903  1.588623e-01        0.035780
google/bert_uncased_L-2_H-128_A-2            full           128  1.000000  8.077704e-16        3.673126
google/bert_uncased_L-2_H-128_A-2        random_2             2  0.165477  1.345937e-01        0.059527
google/bert_uncased_L-2_H-128_A-2        state_1d             1  0.239474  1.182476e-01        0.007496
```

## Failure-Mode Flags

Feature AUCs here are simple mean-feature AUCs, not cross-validated classifiers. They are only quick flags.
```text
                            model  full_rho_mean  full_rho_std  full_rho_share_gt_0_99  rho_delta_mean_feature_auc  rho_state_delta_mean_feature_auc  rho_random_mean_feature_auc  scalar_layer_mean_feature_auc  hidden_all_mean_feature_auc
                bert-base-uncased            1.0  4.276458e-15                     1.0                    0.356746                          0.360775                     0.373670                       0.558833                     0.516884
          distilbert-base-uncased            1.0  3.507625e-15                     1.0                    0.176898                          0.231941                     0.227434                       0.672450                     0.629732
google/bert_uncased_L-2_H-128_A-2            1.0  8.188082e-16                     1.0                    0.469333                          0.449667                     0.459500                       0.596167                     0.288500
```
