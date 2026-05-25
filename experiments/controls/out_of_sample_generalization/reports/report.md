# Control: Out-of-Sample Route Generalization

Train-route subspaces are estimated on train prompts and evaluated on held-out prompts. Test-oracle subspaces are estimated on held-out prompts and serve as an upper-bound diagnostic, not a deployable route.

## Score Summary

```text
                  model init_state split route_source subspace_family  subspace_k  layer  n  rho_mean  rho_std  v_access_mean  v_inaccess_mean  grad_entropy_proj_norm_mean  grad_varentropy_proj_norm_mean
      bert-base-uncased pretrained  test random_route          random           8      4 10  0.410078 0.215451       0.280732         0.326841                     0.082023                        0.215247
      bert-base-uncased pretrained  test random_route          random           8      8 10  0.656748 0.180471       1.147210         0.448965                     0.145442                        0.272130
      bert-base-uncased pretrained  test random_route          random           8     12 10  0.704433 0.143431       1.220543         0.425377                     0.203956                        0.430878
      bert-base-uncased pretrained  test  test_oracle       delta_pca           8      4 10  0.506396 0.157602       0.334998         0.272575                     0.110088                        0.280921
      bert-base-uncased pretrained  test  test_oracle       delta_pca           8      8 10  0.569604 0.254449       1.090536         0.505640                     0.130292                        0.252114
      bert-base-uncased pretrained  test  test_oracle       state_pca           8      4 10  0.470543 0.148674       0.316265         0.291309                     0.095391                        0.254510
      bert-base-uncased pretrained  test  test_oracle       state_pca           8      8 10  0.665833 0.169275       1.189467         0.406708                     0.171734                        0.336947
      bert-base-uncased pretrained  test  test_oracle       state_pca           8     12 10  0.735869 0.196848       1.314744         0.331176                     0.261018                        0.574788
      bert-base-uncased pretrained  test  train_route       delta_pca           8      4 10  0.419883 0.135791       0.279425         0.328148                     0.089371                        0.232029
      bert-base-uncased pretrained  test  train_route       delta_pca           8      8 10  0.658914 0.114077       1.132812         0.463363                     0.154150                        0.308931
      bert-base-uncased pretrained  test  train_route       state_pca           8      4 10  0.503919 0.151370       0.330766         0.276807                     0.114180                        0.296175
      bert-base-uncased pretrained  test  train_route       state_pca           8      8 10  0.644319 0.231276       1.154290         0.441885                     0.160793                        0.321946
      bert-base-uncased pretrained  test  train_route       state_pca           8     12 10  0.725521 0.160714       1.293554         0.352367                     0.258660                        0.532746
      bert-base-uncased pretrained train random_route          random           8      4 11  0.431361 0.198444       0.343875         0.395047                     0.087053                        0.218754
      bert-base-uncased pretrained train random_route          random           8      8 11  0.496341 0.169926       0.515461         0.345790                     0.078752                        0.177807
      bert-base-uncased pretrained train random_route          random           8     12 11  0.721730 0.174159       1.389951         0.416863                     0.221998                        0.282977
      bert-base-uncased pretrained train  train_route       delta_pca           8      4 11  0.479843 0.187906       0.408367         0.330555                     0.104732                        0.257095
      bert-base-uncased pretrained train  train_route       delta_pca           8      8 11  0.469094 0.156102       0.463975         0.397276                     0.067437                        0.152348
      bert-base-uncased pretrained train  train_route       state_pca           8      4 11  0.545400 0.220492       0.452089         0.286833                     0.119819                        0.277585
      bert-base-uncased pretrained train  train_route       state_pca           8      8 11  0.499957 0.141865       0.468890         0.392361                     0.075834                        0.165395
      bert-base-uncased pretrained train  train_route       state_pca           8     12 11  0.806102 0.141030       1.505597         0.301217                     0.324856                        0.486762
distilbert-base-uncased pretrained  test random_route          random           8      2 10  0.483521 0.167319       0.448932         0.211393                     0.130022                        0.213183
distilbert-base-uncased pretrained  test random_route          random           8      4 10  0.446787 0.198196       0.425122         0.339879                     0.075073                        0.181758
distilbert-base-uncased pretrained  test random_route          random           8      6 10  0.659018 0.209118       1.008442         0.309989                     0.250964                        0.503996
distilbert-base-uncased pretrained  test  test_oracle       delta_pca           8      2 10  0.446673 0.175861       0.415147         0.245178                     0.123573                        0.214287
distilbert-base-uncased pretrained  test  test_oracle       delta_pca           8      4 10  0.550549 0.138239       0.482459         0.282542                     0.104609                        0.257810
distilbert-base-uncased pretrained  test  test_oracle       state_pca           8      2 10  0.519907 0.187570       0.480455         0.179870                     0.150351                        0.229424
distilbert-base-uncased pretrained  test  test_oracle       state_pca           8      4 10  0.518362 0.257200       0.515344         0.249657                     0.116834                        0.286973
distilbert-base-uncased pretrained  test  test_oracle       state_pca           8      6 10  0.612978 0.300213       0.997104         0.321327                     0.344561                        0.632046
distilbert-base-uncased pretrained  test  train_route       delta_pca           8      2 10  0.432104 0.213658       0.424684         0.235641                     0.115646                        0.212214
distilbert-base-uncased pretrained  test  train_route       delta_pca           8      4 10  0.529389 0.176241       0.496332         0.268669                     0.091120                        0.226610
distilbert-base-uncased pretrained  test  train_route       state_pca           8      2 10  0.496537 0.169625       0.443926         0.216399                     0.121644                        0.223912
distilbert-base-uncased pretrained  test  train_route       state_pca           8      4 10  0.527222 0.155098       0.475195         0.289806                     0.119723                        0.290349
distilbert-base-uncased pretrained  test  train_route       state_pca           8      6 10  0.602200 0.204719       0.924718         0.393713                     0.281759                        0.541440
distilbert-base-uncased pretrained train random_route          random           8      2 11  0.448484 0.168338       0.198430         0.200046                     0.097622                        0.260468
distilbert-base-uncased pretrained train random_route          random           8      4 11  0.398542 0.120152       0.161367         0.229322                     0.042206                        0.107668
distilbert-base-uncased pretrained train random_route          random           8      6 11  0.690235 0.169788       1.382692         0.494681                     0.289393                        0.434555
distilbert-base-uncased pretrained train  train_route       delta_pca           8      2 11  0.511207 0.121167       0.220600         0.177876                     0.091537                        0.235719
distilbert-base-uncased pretrained train  train_route       delta_pca           8      4 11  0.450184 0.120729       0.190220         0.200469                     0.060622                        0.154877
distilbert-base-uncased pretrained train  train_route       state_pca           8      2 11  0.451682 0.190592       0.209065         0.189411                     0.101959                        0.273769
```

## Steering Summary

```text
                            model init_state split route_source subspace_family  subspace_k     sign  n  abs_delta_entropy_mean  abs_delta_varentropy_mean  directional_success_rate  selected_top1_changed_rate  full_vocab_top10_jaccard_mean
                bert-base-uncased pretrained  test random_route          random           8 decrease 30                0.042362                   0.082559                       1.0                    0.000000                       0.963636
                bert-base-uncased pretrained  test random_route          random           8 increase 30                0.041795                   0.082388                       1.0                    0.000000                       0.939394
                bert-base-uncased pretrained  test  test_oracle       delta_pca           8 decrease 20                0.037807                   0.073444                       1.0                    0.000000                       0.946970
                bert-base-uncased pretrained  test  test_oracle       delta_pca           8 increase 20                0.036166                   0.072967                       1.0                    0.000000                       0.954545
                bert-base-uncased pretrained  test  test_oracle       state_pca           8 decrease 30                0.043419                   0.083940                       1.0                    0.000000                       0.963636
                bert-base-uncased pretrained  test  test_oracle       state_pca           8 increase 30                0.043982                   0.086130                       1.0                    0.066667                       0.957576
                bert-base-uncased pretrained  test  train_route       delta_pca           8 decrease 20                0.038021                   0.073594                       1.0                    0.000000                       0.945455
                bert-base-uncased pretrained  test  train_route       delta_pca           8 increase 20                0.036262                   0.072961                       1.0                    0.000000                       0.963636
                bert-base-uncased pretrained  test  train_route       state_pca           8 decrease 30                0.043580                   0.084279                       1.0                    0.000000                       0.957576
                bert-base-uncased pretrained  test  train_route       state_pca           8 increase 30                0.043359                   0.085086                       1.0                    0.033333                       0.946465
                bert-base-uncased pretrained train random_route          random           8 decrease 33                0.038431                   0.063403                       1.0                    0.030303                       0.923783
                bert-base-uncased pretrained train random_route          random           8 increase 33                0.038364                   0.061021                       1.0                    0.000000                       0.966942
                bert-base-uncased pretrained train  train_route       delta_pca           8 decrease 22                0.030861                   0.066559                       1.0                    0.000000                       0.950413
                bert-base-uncased pretrained train  train_route       delta_pca           8 increase 22                0.028990                   0.064375                       1.0                    0.045455                       0.958678
                bert-base-uncased pretrained train  train_route       state_pca           8 decrease 33                0.039926                   0.065128                       1.0                    0.000000                       0.945822
                bert-base-uncased pretrained train  train_route       state_pca           8 increase 33                0.040939                   0.064331                       1.0                    0.030303                       0.972452
          distilbert-base-uncased pretrained  test random_route          random           8 decrease 30                0.033754                   0.064236                       1.0                    0.066667                       0.975758
          distilbert-base-uncased pretrained  test random_route          random           8 increase 30                0.032456                   0.061928                       1.0                    0.000000                       0.957576
          distilbert-base-uncased pretrained  test  test_oracle       delta_pca           8 decrease 20                0.029009                   0.059697                       1.0                    0.100000                       0.936364
          distilbert-base-uncased pretrained  test  test_oracle       delta_pca           8 increase 20                0.026706                   0.056306                       1.0                    0.050000                       0.963636
          distilbert-base-uncased pretrained  test  test_oracle       state_pca           8 decrease 30                0.034134                   0.062955                       1.0                    0.000000                       0.957576
          distilbert-base-uncased pretrained  test  test_oracle       state_pca           8 increase 30                0.033320                   0.061971                       1.0                    0.000000                       0.981818
          distilbert-base-uncased pretrained  test  train_route       delta_pca           8 decrease 20                0.029003                   0.060071                       1.0                    0.000000                       0.945455
          distilbert-base-uncased pretrained  test  train_route       delta_pca           8 increase 20                0.026614                   0.056603                       1.0                    0.000000                       0.981818
          distilbert-base-uncased pretrained  test  train_route       state_pca           8 decrease 30                0.034038                   0.064828                       1.0                    0.000000                       0.969697
          distilbert-base-uncased pretrained  test  train_route       state_pca           8 increase 30                0.032644                   0.062526                       1.0                    0.000000                       0.963636
          distilbert-base-uncased pretrained train random_route          random           8 decrease 33                0.032133                   0.049681                       1.0                    0.000000                       0.933884
          distilbert-base-uncased pretrained train random_route          random           8 increase 33                0.030998                   0.048446                       1.0                    0.030303                       0.917355
          distilbert-base-uncased pretrained train  train_route       delta_pca           8 decrease 22                0.022606                   0.054604                       1.0                    0.000000                       0.926997
          distilbert-base-uncased pretrained train  train_route       delta_pca           8 increase 22                0.019910                   0.048975                       1.0                    0.000000                       0.902204
          distilbert-base-uncased pretrained train  train_route       state_pca           8 decrease 33                0.033330                   0.051081                       1.0                    0.030303                       0.933884
          distilbert-base-uncased pretrained train  train_route       state_pca           8 increase 33                0.033514                   0.050111                       1.0                    0.000000                       0.939394
google/bert_uncased_L-2_H-128_A-2 pretrained  test random_route          random           8 decrease 20                0.028714                   0.056599                       1.0                    0.000000                       0.963636
google/bert_uncased_L-2_H-128_A-2 pretrained  test random_route          random           8 increase 20                0.026719                   0.054617                       1.0                    0.100000                       0.972727
google/bert_uncased_L-2_H-128_A-2 pretrained  test  test_oracle       delta_pca           8 decrease 10                0.029106                   0.059302                       1.0                    0.100000                       0.963636
google/bert_uncased_L-2_H-128_A-2 pretrained  test  test_oracle       delta_pca           8 increase 10                0.026621                   0.055845                       1.0                    0.000000                       0.948485
google/bert_uncased_L-2_H-128_A-2 pretrained  test  test_oracle       state_pca           8 decrease 20                0.025649                   0.051905                       1.0                    0.100000                       0.972727
google/bert_uncased_L-2_H-128_A-2 pretrained  test  test_oracle       state_pca           8 increase 20                0.024109                   0.050936                       1.0                    0.000000                       0.974242
google/bert_uncased_L-2_H-128_A-2 pretrained  test  train_route       delta_pca           8 decrease 10                0.026300                   0.054865                       1.0                    0.100000                       0.981818
google/bert_uncased_L-2_H-128_A-2 pretrained  test  train_route       delta_pca           8 increase 10                0.024493                   0.052538                       1.0                    0.000000                       0.966667
```

## Contrasts

```text
                            model subspace_family            metric       route             control  route_mean  control_mean  route_minus_control  route_over_control  n_route  n_control
                bert-base-uncased       delta_pca               rho train_route random_route_pooled    0.539399      0.590420            -0.051021            0.913585       20         30
                bert-base-uncased       delta_pca               rho train_route         test_oracle    0.539399      0.538000             0.001399            1.002600       20         20
                bert-base-uncased       state_pca               rho train_route random_route_pooled    0.624586      0.590420             0.034167            1.057869       30         30
                bert-base-uncased       state_pca               rho train_route         test_oracle    0.624586      0.624081             0.000505            1.000809       30         30
          distilbert-base-uncased       delta_pca               rho train_route random_route_pooled    0.480747      0.529775            -0.049029            0.907454       20         30
          distilbert-base-uncased       delta_pca               rho train_route         test_oracle    0.480747      0.498611            -0.017864            0.964172       20         20
          distilbert-base-uncased       state_pca               rho train_route random_route_pooled    0.541986      0.529775             0.012211            1.023049       30         30
          distilbert-base-uncased       state_pca               rho train_route         test_oracle    0.541986      0.550416            -0.008430            0.984685       30         30
google/bert_uncased_L-2_H-128_A-2       delta_pca               rho train_route random_route_pooled    0.374210      0.429111            -0.054901            0.872060       10         20
google/bert_uncased_L-2_H-128_A-2       delta_pca               rho train_route         test_oracle    0.374210      0.452420            -0.078209            0.827131       10         10
google/bert_uncased_L-2_H-128_A-2       state_pca               rho train_route random_route_pooled    0.420336      0.429111            -0.008775            0.979551       20         20
google/bert_uncased_L-2_H-128_A-2       state_pca               rho train_route         test_oracle    0.420336      0.360531             0.059805            1.165881       20         20
                bert-base-uncased       delta_pca abs_delta_entropy train_route random_route_pooled    0.037142      0.042078            -0.004936            0.882686       40         60
                bert-base-uncased       delta_pca abs_delta_entropy train_route         test_oracle    0.037142      0.036986             0.000155            1.004202       40         40
                bert-base-uncased       state_pca abs_delta_entropy train_route random_route_pooled    0.043469      0.042078             0.001391            1.033059       60         60
                bert-base-uncased       state_pca abs_delta_entropy train_route         test_oracle    0.043469      0.043700            -0.000231            0.994713       60         60
          distilbert-base-uncased       delta_pca abs_delta_entropy train_route random_route_pooled    0.027809      0.033105            -0.005297            0.839999       40         60
          distilbert-base-uncased       delta_pca abs_delta_entropy train_route         test_oracle    0.027809      0.027857            -0.000049            0.998245       40         40
          distilbert-base-uncased       state_pca abs_delta_entropy train_route random_route_pooled    0.033341      0.033105             0.000235            1.007111       60         60
          distilbert-base-uncased       state_pca abs_delta_entropy train_route         test_oracle    0.033341      0.033727            -0.000386            0.988552       60         60
google/bert_uncased_L-2_H-128_A-2       delta_pca abs_delta_entropy train_route random_route_pooled    0.025397      0.027716            -0.002320            0.916302       20         40
google/bert_uncased_L-2_H-128_A-2       delta_pca abs_delta_entropy train_route         test_oracle    0.025397      0.027863            -0.002467            0.911466       20         20
google/bert_uncased_L-2_H-128_A-2       state_pca abs_delta_entropy train_route random_route_pooled    0.027133      0.027716            -0.000583            0.978948       40         40
google/bert_uncased_L-2_H-128_A-2       state_pca abs_delta_entropy train_route         test_oracle    0.027133      0.024879             0.002254            1.090594       40         40
```

## Files

```text
prompt_tables.csv
oos_scores.csv
oos_steering_records.csv
oos_score_summary.csv
oos_steering_summary.csv
oos_contrasts.csv
report.md
```