# Rho-Guided Selective Reliability

This disruptive test asks whether `rho` improves a non-oracle selective reliability policy over a strong B2 baseline: scalar uncertainty, gradient/Jacobian/Fisher features, and model/source/task/topic metadata. B3 adds rho. A shuffled-rho negative control keeps the same feature dimensionality but breaks rho-row alignment.

Correctness is never used as a decision feature. It is used only for held-out evaluation and cross-validated training labels. Evaluation uses grouped-prompt, leave-one-model-out, and leave-one-source-out splits; bootstrap intervals resample the matching cluster unit rather than individual rows.

## Verdict
Supported on AURC, Brier, log-loss

## Dataset Coverage
```text
             source                             model  n  correct_rate
    decoder_battery                 Qwen/Qwen2.5-0.5B 12      0.166667
    decoder_battery                   microsoft/phi-2 12      0.333333
masked_full_battery                 bert-base-uncased 21      0.523810
masked_full_battery           distilbert-base-uncased 21      0.476190
masked_full_battery google/bert_uncased_L-2_H-128_A-2 21      0.095238
        masked_topk                 bert-base-uncased 15      0.533333
        masked_topk           distilbert-base-uncased 15      0.400000
        masked_topk google/bert_uncased_L-2_H-128_A-2 15      0.000000
```

## Final Split Table
```text
               split             metric  baseline     +rho  +shuffled_rho  delta_rho  delta_shuffled_rho
      grouped_prompt           log_loss  0.511038 0.384790       0.517878   0.126248           -0.006840
      grouped_prompt              brier  0.153992 0.118298       0.142698   0.035694            0.011294
      grouped_prompt               AURC  0.469461 0.448106       0.471774   0.021355           -0.002313
      grouped_prompt coverage@risk=0.30  0.310606 0.356061       0.356061   0.045455            0.045455
      grouped_prompt risk@coverage=0.50  0.439394 0.393939       0.454545   0.045455           -0.015152
      grouped_prompt                ECE  0.118130 0.103999       0.101027   0.014130            0.017103
 leave_one_model_out           log_loss  0.788570 0.667130       0.759482   0.121440            0.029088
 leave_one_model_out              brier  0.191133 0.164091       0.195833   0.027042           -0.004701
 leave_one_model_out               AURC  0.498974 0.481350       0.491493   0.017624            0.007481
 leave_one_model_out coverage@risk=0.30  0.280303 0.356061       0.287879   0.075758            0.007576
 leave_one_model_out risk@coverage=0.50  0.500000 0.454545       0.484848   0.045455            0.015152
 leave_one_model_out                ECE  0.195151 0.151329       0.185414   0.043822            0.009737
leave_one_source_out           log_loss  1.851963 3.231722       2.549283  -1.379759           -0.697320
leave_one_source_out              brier  0.355386 0.375257       0.387113  -0.019871           -0.031727
leave_one_source_out               AURC  0.636755 0.645534       0.639795  -0.008779           -0.003040
leave_one_source_out coverage@risk=0.30  0.000000 0.000000       0.000000   0.000000            0.000000
leave_one_source_out risk@coverage=0.50  0.560606 0.575758       0.575758  -0.015152           -0.015152
leave_one_source_out                ECE  0.336919 0.350107       0.378826  -0.013188           -0.041907
```

## Policy Metrics
```text
       policy   n  accuracy    brier  log_loss    auroc  average_precision  aurc_mean_risk  coverage_at_risk_030  risk_at_50pct_coverage  risk_at_80pct_coverage      ece  high_pred_reliable_error_rate  high_pred_reliable_coverage  top20pct_error_rate                split
     baseline 132  0.325758 0.153992  0.511038 0.863601           0.758355        0.469461              0.310606                0.439394                0.594340 0.118130                       0.200000                     0.265152             0.148148       grouped_prompt
         +rho 132  0.325758 0.118298  0.384790 0.905932           0.828244        0.448106              0.356061                0.393939                0.594340 0.103999                       0.129032                     0.234848             0.148148       grouped_prompt
+shuffled_rho 132  0.325758 0.142698  0.517878 0.862817           0.749488        0.471774              0.356061                0.454545                0.594340 0.101027                       0.200000                     0.227273             0.185185       grouped_prompt
     baseline 132  0.325758 0.191133  0.788570 0.813953           0.667542        0.498974              0.280303                0.500000                0.603774 0.195151                       0.257143                     0.265152             0.222222  leave_one_model_out
         +rho 132  0.325758 0.164091  0.667130 0.847139           0.709736        0.481350              0.356061                0.454545                0.603774 0.151329                       0.233333                     0.227273             0.185185  leave_one_model_out
+shuffled_rho 132  0.325758 0.195833  0.759482 0.823622           0.692568        0.491493              0.287879                0.484848                0.603774 0.185414                       0.289474                     0.287879             0.185185  leave_one_model_out
     baseline 132  0.325758 0.355386  1.851963 0.581918           0.357679        0.636755              0.000000                0.560606                0.650943 0.336919                       0.658537                     0.310606             0.703704 leave_one_source_out
         +rho 132  0.325758 0.375257  3.231722 0.578521           0.342869        0.645534              0.000000                0.575758                0.632075 0.350107                       0.688889                     0.340909             0.740741 leave_one_source_out
+shuffled_rho 132  0.325758 0.387113  2.549283 0.579044           0.345107        0.639795              0.000000                0.575758                0.632075 0.378826                       0.688889                     0.340909             0.703704 leave_one_source_out
```

## Baseline Vs Baseline+rho Deltas
```text
              split                                          metric     value
     grouped_prompt                         delta_brier_improvement  0.035694
     grouped_prompt                      delta_log_loss_improvement  0.126248
     grouped_prompt                          delta_aurc_improvement  0.021355
     grouped_prompt          delta_coverage_at_risk_030_improvement  0.045455
     grouped_prompt                        delta_risk50_improvement  0.045455
     grouped_prompt                           delta_ece_improvement  0.014130
     grouped_prompt                shuffled_delta_brier_improvement  0.011294
     grouped_prompt             shuffled_delta_log_loss_improvement -0.006840
     grouped_prompt                 shuffled_delta_aurc_improvement -0.002313
     grouped_prompt shuffled_delta_coverage_at_risk_030_improvement  0.045455
     grouped_prompt               shuffled_delta_risk50_improvement -0.015152
     grouped_prompt                  shuffled_delta_ece_improvement  0.017103
leave_one_model_out                         delta_brier_improvement  0.027042
leave_one_model_out                      delta_log_loss_improvement  0.121440
leave_one_model_out                          delta_aurc_improvement  0.017624
leave_one_model_out          delta_coverage_at_risk_030_improvement  0.075758
leave_one_model_out                        delta_risk50_improvement  0.045455
leave_one_model_out                           delta_ece_improvement  0.043822
leave_one_model_out                shuffled_delta_brier_improvement -0.004701
leave_one_model_out             shuffled_delta_log_loss_improvement  0.029088
```

## Bootstrap CI
```text
               split                                          metric    ci_low    median   ci_high
      grouped_prompt                          delta_aurc_improvement  0.007246  0.023724  0.043499
      grouped_prompt                         delta_brier_improvement  0.020456  0.035250  0.052599
      grouped_prompt          delta_coverage_at_risk_030_improvement  0.000000  0.045308  0.148173
      grouped_prompt                           delta_ece_improvement -0.035864  0.015176  0.064727
      grouped_prompt                      delta_log_loss_improvement  0.073595  0.124629  0.186695
      grouped_prompt                        delta_risk50_improvement -0.015157  0.044118  0.105263
      grouped_prompt                 shuffled_delta_aurc_improvement -0.021233  0.002512  0.027164
      grouped_prompt                shuffled_delta_brier_improvement -0.011740  0.011193  0.037115
      grouped_prompt shuffled_delta_coverage_at_risk_030_improvement -0.068756  0.025947  0.079726
      grouped_prompt                  shuffled_delta_ece_improvement -0.037521  0.012009  0.058923
      grouped_prompt             shuffled_delta_log_loss_improvement -0.094321 -0.008861  0.090067
      grouped_prompt               shuffled_delta_risk50_improvement -0.073557  0.000000  0.055556
 leave_one_model_out                          delta_aurc_improvement  0.003171  0.017624  0.032872
 leave_one_model_out                         delta_brier_improvement  0.013519  0.027042  0.036902
 leave_one_model_out          delta_coverage_at_risk_030_improvement -0.006410  0.022727  0.122361
 leave_one_model_out                           delta_ece_improvement  0.001457  0.026081  0.046866
 leave_one_model_out                      delta_log_loss_improvement  0.086749  0.122190  0.145688
 leave_one_model_out                        delta_risk50_improvement  0.000000  0.030303  0.060606
 leave_one_model_out                 shuffled_delta_aurc_improvement -0.022517  0.002285  0.031574
 leave_one_model_out                shuffled_delta_brier_improvement -0.047816 -0.000631  0.028839
 leave_one_model_out shuffled_delta_coverage_at_risk_030_improvement -0.111111  0.000000  0.129630
 leave_one_model_out                  shuffled_delta_ece_improvement -0.047332  0.009737  0.045669
 leave_one_model_out             shuffled_delta_log_loss_improvement -0.107159  0.040437  0.131662
 leave_one_model_out               shuffled_delta_risk50_improvement -0.089744  0.000000  0.051282
leave_one_source_out                          delta_aurc_improvement -0.056831 -0.008779  0.086824
leave_one_source_out                         delta_brier_improvement -0.079132 -0.019871  0.011063
leave_one_source_out          delta_coverage_at_risk_030_improvement -0.045045  0.000000  0.000000
leave_one_source_out                           delta_ece_improvement -0.056763 -0.013188  0.022897
leave_one_source_out                      delta_log_loss_improvement -4.281703 -1.379759  0.318232
leave_one_source_out                        delta_risk50_improvement -0.132353 -0.015152  0.083333
leave_one_source_out                 shuffled_delta_aurc_improvement -0.037310 -0.006228  0.076386
leave_one_source_out                shuffled_delta_brier_improvement -0.054719 -0.031727 -0.012988
leave_one_source_out shuffled_delta_coverage_at_risk_030_improvement -0.198198  0.000000  0.000000
leave_one_source_out                  shuffled_delta_ece_improvement -0.050860 -0.041907 -0.028113
leave_one_source_out             shuffled_delta_log_loss_improvement -2.013091 -0.697320  0.368272
leave_one_source_out               shuffled_delta_risk50_improvement -0.102941 -0.015152  0.166667
```

## Risk-Coverage
```text
               split  requested_coverage     +rho  +shuffled_rho  baseline  risk_improvement
      grouped_prompt                 0.2 0.148148       0.185185  0.148148          0.000000
      grouped_prompt                 0.3 0.200000       0.250000  0.275000          0.075000
      grouped_prompt                 0.4 0.358491       0.377358  0.396226          0.037736
      grouped_prompt                 0.5 0.393939       0.454545  0.439394          0.045455
      grouped_prompt                 0.6 0.487500       0.512500  0.500000          0.012500
      grouped_prompt                 0.7 0.537634       0.559140  0.559140          0.021505
      grouped_prompt                 0.8 0.594340       0.594340  0.594340          0.000000
      grouped_prompt                 0.9 0.638655       0.638655  0.638655          0.000000
      grouped_prompt                 1.0 0.674242       0.674242  0.674242          0.000000
 leave_one_model_out                 0.2 0.185185       0.185185  0.222222          0.037037
 leave_one_model_out                 0.3 0.325000       0.325000  0.350000          0.025000
 leave_one_model_out                 0.4 0.377358       0.415094  0.396226          0.018868
 leave_one_model_out                 0.5 0.454545       0.484848  0.500000          0.045455
 leave_one_model_out                 0.6 0.525000       0.537500  0.525000          0.000000
 leave_one_model_out                 0.7 0.548387       0.559140  0.580645          0.032258
 leave_one_model_out                 0.8 0.603774       0.603774  0.603774          0.000000
 leave_one_model_out                 0.9 0.638655       0.638655  0.638655          0.000000
 leave_one_model_out                 1.0 0.674242       0.674242  0.674242          0.000000
leave_one_source_out                 0.2 0.740741       0.703704  0.703704         -0.037037
leave_one_source_out                 0.3 0.700000       0.675000  0.650000         -0.050000
leave_one_source_out                 0.4 0.603774       0.641509  0.603774          0.000000
leave_one_source_out                 0.5 0.575758       0.575758  0.560606         -0.015152
leave_one_source_out                 0.6 0.612500       0.587500  0.600000         -0.012500
leave_one_source_out                 0.7 0.623656       0.612903  0.623656          0.000000
leave_one_source_out                 0.8 0.632075       0.632075  0.650943          0.018868
leave_one_source_out                 0.9 0.647059       0.655462  0.663866          0.016807
leave_one_source_out                 1.0 0.674242       0.674242  0.674242          0.000000
```

## Rho-Guided Action Mix
```text
               action  n  coverage  error_rate  rho_action_mean  predicted_correct_mean
     abstain_or_route 32  0.242424    1.000000         0.756102                0.015598
               accept 40  0.303030    0.200000         0.919660                0.894401
intervene_or_retrieve 26  0.196970    0.846154         0.984900                0.209333
  pass_to_other_model 34  0.257576    0.794118         0.851710                0.225204
```

## Files
```text
selective_reliability_dataset.csv
selective_reliability_predictions.csv
selective_reliability_metrics.csv
selective_reliability_final_table.csv
selective_reliability_deltas.csv
selective_reliability_bootstrap_ci.csv
risk_coverage_curve.csv
rho_guided_action_mix.csv
report.md
```
