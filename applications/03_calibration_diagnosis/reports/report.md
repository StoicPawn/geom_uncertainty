# Application 3: Calibration Diagnosis


This application tests whether high-accessibility routes allow local calibration steering with lower accuracy loss and answer-neighborhood drift.

## Interpretation

High-rho routes are much more effective than low-rho routes for NLL, Brier score, target probability, and exact-target accuracy at the same Fisher-output budget. ECE is not improved in this run: the clean exact-token baseline has very low confidence and very low accuracy, so it is close to calibrated in a pessimistic sense. This application should therefore be read as evidence for internal target-probability/NLL/Brier steerability, with an ECE caveat.


## Calibration Summary


```text
    route_role  n  accuracy      ece      nll    brier  confidence  mean_rho  mean_full_vocab_kl  mean_top10_jaccard  accuracy_loss_rate  mean_delta_nll  mean_delta_brier  mean_delta_target_prob
clean_baseline 57  0.017544 0.030966 8.914663 1.008972    0.048510       NaN                 NaN                 NaN                 NaN             NaN               NaN                     NaN
high_rho_route 57  0.385965 0.092297 4.360887 0.680859    0.349921  0.742822            0.002426            0.942584                 0.0       -0.014644         -0.006771                0.005601
 low_rho_route 57  0.052632 0.032030 7.046303 0.981504    0.055301  0.361388            0.002580            0.942584                 0.0       -0.005145         -0.001468                0.000450
  random_route 57  0.385965 0.121204 4.260719 0.697972    0.328693  0.726609            0.001989            0.955875                 0.0       -0.020873         -0.006473                0.005025
```

## Calibration Effects


```text
    route_role  ece_improvement  nll_improvement  brier_improvement  accuracy_delta  accuracy_loss_rate  calibration_steerability_score  mean_full_vocab_kl  mean_top10_jaccard
high_rho_route        -0.061330         4.553776           0.328113        0.368421                 0.0                       -0.061330            0.002426            0.942584
 low_rho_route        -0.001064         1.868360           0.027468        0.035088                 0.0                       -0.001064            0.002580            0.942584
  random_route        -0.090238         4.653944           0.311000        0.368421                 0.0                       -0.090238            0.001989            0.955875
```

## High-Rho Route Contrasts


```text
                     comparison            metric  n_pairs  high_minus_low_mean  high_better_rate
high_rho_route_vs_low_rho_route         after_nll       57            -2.685417          0.701754
high_rho_route_vs_low_rho_route       after_brier       57            -0.300646          0.578947
high_rho_route_vs_low_rho_route     full_vocab_kl       57            -0.000154          0.491228
high_rho_route_vs_low_rho_route delta_target_prob       57             0.005151          0.701754
```

## Route Scores


```text
                  model  prompt_id                                    prompt     target           task    topic  layer subspace_family  subspace_k  rep      rho  v_access  v_inaccess  rank_whitened_jb  output_energy_unit  first_order_entropy  clean_correct  clean_confidence  clean_nll  clean_brier
distilbert-base-uncased          1       The language of Portugal is [MASK]. portuguese   factual_deep language      2       state_pca           8    0 0.563735  0.270060    0.208995                 8            0.182233             0.094702              0          0.019823   9.392208     1.000941
distilbert-base-uncased          1       The language of Portugal is [MASK]. portuguese   factual_deep language      2       delta_pca           8    0 0.510358  0.244489    0.234565                 8            0.242782             0.120046              0          0.019823   9.392208     1.000941
distilbert-base-uncased          1       The language of Portugal is [MASK]. portuguese   factual_deep language      2          random           8    0 0.465688  0.223090    0.255965                 8            0.168129             0.079412              0          0.019823   9.392208     1.000941
distilbert-base-uncased          2                 The red planet is [MASK].       mars factual_simple  science      2       state_pca           8    0 0.507031  0.305307    0.296840                 8            0.155106             0.085703              0          0.017412  11.902941     1.001271
distilbert-base-uncased          2                 The red planet is [MASK].       mars factual_simple  science      2       delta_pca           8    0 0.609891  0.367244    0.234903                 8            0.191310             0.115935              0          0.017412  11.902941     1.001271
distilbert-base-uncased          2                 The red planet is [MASK].       mars factual_simple  science      2          random           8    0 0.514565  0.309844    0.292303                 8            0.141738             0.078897              0          0.017412  11.902941     1.001271
distilbert-base-uncased          4           Isaac [MASK] described gravity.     newton   factual_deep  culture      2       state_pca           8    0 0.506949  0.216202    0.210274                 8            0.171468             0.079728              0          0.031361   7.511795     1.003212
distilbert-base-uncased          4           Isaac [MASK] described gravity.     newton   factual_deep  culture      2       delta_pca           8    0 0.658908  0.281008    0.145468                 8            0.149619             0.079313              0          0.031361   7.511795     1.003212
distilbert-base-uncased          4           Isaac [MASK] described gravity.     newton   factual_deep  culture      2          random           8    0 0.438072  0.186827    0.239649                 8            0.146349             0.063257              0          0.031361   7.511795     1.003212
distilbert-base-uncased          5           The capital of Syria is [MASK].   damascus factual_simple  capital      2       state_pca           8    0 0.425054  0.097729    0.132192                 8            0.137622             0.043023              0          0.008515   9.769184     1.000501
distilbert-base-uncased          5           The capital of Syria is [MASK].   damascus factual_simple  capital      2       delta_pca           8    0 0.416567  0.095777    0.134143                 8            0.168315             0.052090              0          0.008515   9.769184     1.000501
distilbert-base-uncased          5           The capital of Syria is [MASK].   damascus factual_simple  capital      2          random           8    0 0.244225  0.056152    0.173768                 8            0.144915             0.034340              0          0.008515   9.769184     1.000501
distilbert-base-uncased          7    A person who flies planes is a [MASK].      pilot   factual_deep     role      2       state_pca           8    0 0.578903  0.201201    0.146354                 8            0.143616             0.064419              0          0.010428   7.489767     0.999659
distilbert-base-uncased          7    A person who flies planes is a [MASK].      pilot   factual_deep     role      2       delta_pca           8    0 0.468045  0.162671    0.184883                 8            0.220494             0.088931              0          0.010428   7.489767     0.999659
distilbert-base-uncased          7    A person who flies planes is a [MASK].      pilot   factual_deep     role      2          random           8    0 0.342412  0.119007    0.228548                 8            0.151551             0.052281              0          0.010428   7.489767     0.999659
distilbert-base-uncased          8               Two plus two equals [MASK].       four factual_simple     math      2       state_pca           8    0 0.832290  2.034891    0.410039                 8            0.278472             0.397239              0          0.071246  10.058464     1.005643
distilbert-base-uncased          8               Two plus two equals [MASK].       four factual_simple     math      2       delta_pca           8    0 0.915022  2.237165    0.207765                 8            0.274001             0.409828              0          0.071246  10.058464     1.005643
distilbert-base-uncased          8               Two plus two equals [MASK].       four factual_simple     math      2          random           8    0 0.868281  2.122885    0.322045                 8            0.166829             0.243072              0          0.071246  10.058464     1.005643
distilbert-base-uncased         10        The capital of Pakistan is [MASK].  islamabad   factual_deep  capital      2       state_pca           8    0 0.574407  0.206971    0.153350                 8            0.166354             0.075681              0          0.013319   9.271247     1.000528
distilbert-base-uncased         10        The capital of Pakistan is [MASK].  islamabad   factual_deep  capital      2       delta_pca           8    0 0.395038  0.142341    0.217981                 8            0.204318             0.077085              0          0.013319   9.271247     1.000528
distilbert-base-uncased         10        The capital of Pakistan is [MASK].  islamabad   factual_deep  capital      2          random           8    0 0.260640  0.093914    0.266407                 8            0.164181             0.050314              0          0.013319   9.271247     1.000528
distilbert-base-uncased         11    A person who drives a bus is a [MASK].     driver factual_simple     role      2       state_pca           8    0 0.462322  0.109387    0.127217                 8            0.125357             0.041460              0          0.009199   9.129897     1.000541
distilbert-base-uncased         11    A person who drives a bus is a [MASK].     driver factual_simple     role      2       delta_pca           8    0 0.508916  0.120412    0.116193                 8            0.169900             0.058956              0          0.009199   9.129897     1.000541
distilbert-base-uncased         11    A person who drives a bus is a [MASK].     driver factual_simple     role      2          random           8    0 0.298131  0.070539    0.166065                 8            0.151850             0.040330              0          0.009199   9.129897     1.000541
distilbert-base-uncased         14            The opposite of day is [MASK].      night factual_simple  lexical      2       state_pca           8    0 0.513414  0.104830    0.099352                 8            0.128848             0.041718              0          0.009739  13.176961     1.000729
distilbert-base-uncased         14            The opposite of day is [MASK].      night factual_simple  lexical      2       delta_pca           8    0 0.177770  0.036297    0.167884                 8            0.147370             0.028077              0          0.009739  13.176961     1.000729
distilbert-base-uncased         14            The opposite of day is [MASK].      night factual_simple  lexical      2          random           8    0 0.176392  0.036016    0.168165                 8            0.163204             0.030973              0          0.009739  13.176961     1.000729
distilbert-base-uncased         16 Black has the opposite meaning of [MASK].      white   factual_deep  antonym      2       state_pca           8    0 0.507041  0.181551    0.176508                 8            0.127598             0.054368              0          0.007416   8.900846     1.000114
distilbert-base-uncased         16 Black has the opposite meaning of [MASK].      white   factual_deep  antonym      2       delta_pca           8    0 0.468729  0.167833    0.190226                 8            0.187609             0.076858              0          0.007416   8.900846     1.000114
distilbert-base-uncased         16 Black has the opposite meaning of [MASK].      white   factual_deep  antonym      2          random           8    0 0.489355  0.175218    0.182841                 8            0.189687             0.079401              0          0.007416   8.900846     1.000114
distilbert-base-uncased         17           The capital of Chile is [MASK].   santiago factual_simple  capital      2       state_pca           8    0 0.612654  0.339959    0.214936                 8            0.206944             0.120661              0          0.017820  10.311307     1.000780
distilbert-base-uncased         17           The capital of Chile is [MASK].   santiago factual_simple  capital      2       delta_pca           8    0 0.516708  0.286719    0.268176                 8            0.218009             0.116736              0          0.017820  10.311307     1.000780
distilbert-base-uncased         17           The capital of Chile is [MASK].   santiago factual_simple  capital      2          random           8    0 0.515166  0.285863    0.269032                 8            0.221778             0.118576              0          0.017820  10.311307     1.000780
distilbert-base-uncased         19             Five minus two equals [MASK].      three   factual_deep     math      2       state_pca           8    0 0.894844  2.237320    0.262915                 8            0.278611             0.416737              0          0.075673  10.470501     1.006385
distilbert-base-uncased         19             Five minus two equals [MASK].      three   factual_deep     math      2       delta_pca           8    0 0.948518  2.371518    0.128718                 8            0.296908             0.457230              0          0.075673  10.470501     1.006385
distilbert-base-uncased         19             Five minus two equals [MASK].      three   factual_deep     math      2          random           8    0 0.715138  1.788013    0.712222                 8            0.179432             0.239930              0          0.075673  10.470501     1.006385
distilbert-base-uncased         20         The language of France is [MASK].     french factual_simple language      2       state_pca           8    0 0.525669  0.188033    0.169669                 8            0.157453             0.068276              0          0.016162  10.990002     1.000906
distilbert-base-uncased         20         The language of France is [MASK].     french factual_simple language      2       delta_pca           8    0 0.471007  0.168480    0.189222                 8            0.242713             0.099625              0          0.016162  10.990002     1.000906
distilbert-base-uncased         20         The language of France is [MASK].     french factual_simple language      2          random           8    0 0.453699  0.162289    0.195413                 8            0.169708             0.068367              0          0.016162  10.990002     1.000906
distilbert-base-uncased         22           The plural of person is [MASK].     people   factual_deep   plural      2       state_pca           8    0 0.309923  0.079980    0.178085                 8            0.119173             0.033703              0          0.011692   9.483383     1.000675
```

## Skipped Models


```text
(empty)
```
