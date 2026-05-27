# Rho-Guided Selective Reliability

This disruptive test asks whether `rho` improves a non-oracle selective reliability policy over a strong B2 baseline: scalar uncertainty, gradient/Jacobian/Fisher features, and model/source/task/topic metadata. B3 adds one rho feature family at a time: absolute rho, intra-example adjusted rho, intra-example rank, intra-example percentile, or model/layer/route-family z-scored rho. A shuffled-rho negative control keeps the same feature dimensionality but breaks rho-row alignment.

Correctness is never used as a decision feature. It is used only for held-out evaluation and cross-validated training labels. Evaluation uses grouped-prompt, leave-one-model-out, and leave-one-source-out splits; bootstrap intervals resample the matching cluster unit rather than individual rows.

The leave-one-source-out panel is diagnostic rather than a general-reliability claim: the current dataset has three source families, and all rho variants fail on log-loss/Brier/ECE when the held-out protocol is far from the training sources.

The shuffled-rho control remains competitive in grouped-prompt and leave-one-model-out splits, so this test should be read as reliability evidence for rho-family features rather than a clean mechanistic isolation of row-aligned rho.

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
               split          rho_feature             metric  baseline     +rho  +shuffled_rho  delta_rho  delta_shuffled_rho
      grouped_prompt             absolute           log_loss  0.511038 0.382033       0.393605   0.129006            0.117434
      grouped_prompt             absolute              brier  0.153992 0.115138       0.117320   0.038853            0.036671
      grouped_prompt             absolute               AURC  0.469461 0.446009       0.446009   0.023452            0.023452
      grouped_prompt             absolute coverage@risk=0.30  0.310606 0.356061       0.356061   0.045455            0.045455
      grouped_prompt             absolute risk@coverage=0.50  0.439394 0.393939       0.393939   0.045455            0.045455
      grouped_prompt             absolute                ECE  0.118130 0.105338       0.095382   0.012792            0.022748
      grouped_prompt                  adj           log_loss  0.511038 0.400652       0.405523   0.110387            0.105515
      grouped_prompt                  adj              brier  0.153992 0.118837       0.119996   0.035155            0.033996
      grouped_prompt                  adj               AURC  0.469461 0.446009       0.448787   0.023452            0.020674
      grouped_prompt                  adj coverage@risk=0.30  0.310606 0.356061       0.356061   0.045455            0.045455
      grouped_prompt                  adj risk@coverage=0.50  0.439394 0.393939       0.393939   0.045455            0.045455
      grouped_prompt                  adj                ECE  0.118130 0.088385       0.087227   0.029745            0.030903
      grouped_prompt                 rank           log_loss  0.511038 0.409274       0.412376   0.101764            0.098662
      grouped_prompt                 rank              brier  0.153992 0.120957       0.121796   0.033035            0.032196
      grouped_prompt                 rank               AURC  0.469461 0.449982       0.451665   0.019479            0.017795
      grouped_prompt                 rank coverage@risk=0.30  0.310606 0.356061       0.356061   0.045455            0.045455
      grouped_prompt                 rank risk@coverage=0.50  0.439394 0.393939       0.409091   0.045455            0.030303
      grouped_prompt                 rank                ECE  0.118130 0.075509       0.066289   0.042621            0.051841
      grouped_prompt           percentile           log_loss  0.511038 0.415083       0.417482   0.095955            0.093556
      grouped_prompt           percentile              brier  0.153992 0.122556       0.123246   0.031436            0.030746
      grouped_prompt           percentile               AURC  0.469461 0.451665       0.451665   0.017795            0.017795
      grouped_prompt           percentile coverage@risk=0.30  0.310606 0.356061       0.356061   0.045455            0.045455
      grouped_prompt           percentile risk@coverage=0.50  0.439394 0.409091       0.409091   0.030303            0.030303
      grouped_prompt           percentile                ECE  0.118130 0.066506       0.072147   0.051624            0.045983
      grouped_prompt z_model_layer_family           log_loss  0.511038 0.419688       0.421695   0.091350            0.089343
      grouped_prompt z_model_layer_family              brier  0.153992 0.123897       0.124496   0.030094            0.029496
      grouped_prompt z_model_layer_family               AURC  0.469461 0.451665       0.451665   0.017795            0.017795
      grouped_prompt z_model_layer_family coverage@risk=0.30  0.310606 0.356061       0.356061   0.045455            0.045455
      grouped_prompt z_model_layer_family risk@coverage=0.50  0.439394 0.409091       0.409091   0.030303            0.030303
      grouped_prompt z_model_layer_family                ECE  0.118130 0.070745       0.070839   0.047384            0.047290
 leave_one_model_out             absolute           log_loss  0.788570 0.590796       0.598380   0.197773            0.190189
 leave_one_model_out             absolute              brier  0.191133 0.156869       0.157838   0.034264            0.033294
 leave_one_model_out             absolute               AURC  0.498974 0.476083       0.476083   0.022890            0.022890
 leave_one_model_out             absolute coverage@risk=0.30  0.280303 0.356061       0.356061   0.075758            0.075758
 leave_one_model_out             absolute risk@coverage=0.50  0.500000 0.454545       0.454545   0.045455            0.045455
 leave_one_model_out             absolute                ECE  0.195151 0.125261       0.127883   0.069890            0.067268
 leave_one_model_out                  adj           log_loss  0.788570 0.601118       0.602384   0.187452            0.186186
 leave_one_model_out                  adj              brier  0.191133 0.158212       0.158400   0.032921            0.032732
 leave_one_model_out                  adj               AURC  0.498974 0.476083       0.476083   0.022890            0.022890
 leave_one_model_out                  adj coverage@risk=0.30  0.280303 0.356061       0.356061   0.075758            0.075758
 leave_one_model_out                  adj risk@coverage=0.50  0.500000 0.454545       0.454545   0.045455            0.045455
 leave_one_model_out                  adj                ECE  0.195151 0.127933       0.127935   0.067218            0.067217
 leave_one_model_out                 rank           log_loss  0.788570 0.603099       0.603523   0.185471            0.185047
 leave_one_model_out                 rank              brier  0.191133 0.158518       0.158599   0.032614            0.032534
 leave_one_model_out                 rank               AURC  0.498974 0.477767       0.477767   0.021207            0.021207
 leave_one_model_out                 rank coverage@risk=0.30  0.280303 0.356061       0.356061   0.075758            0.075758
 leave_one_model_out                 rank risk@coverage=0.50  0.500000 0.469697       0.469697   0.030303            0.030303
 leave_one_model_out                 rank                ECE  0.195151 0.127913       0.140038   0.067238            0.055113
 leave_one_model_out           percentile           log_loss  0.788570 0.603873       0.604092   0.184696            0.184477
 leave_one_model_out           percentile              brier  0.191133 0.158659       0.158703   0.032474            0.032430
 leave_one_model_out           percentile               AURC  0.498974 0.477767       0.477767   0.021207            0.021207
 leave_one_model_out           percentile coverage@risk=0.30  0.280303 0.356061       0.356061   0.075758            0.075758
 leave_one_model_out           percentile risk@coverage=0.50  0.500000 0.469697       0.469697   0.030303            0.030303
 leave_one_model_out           percentile                ECE  0.195151 0.129421       0.129399   0.065730            0.065752
 leave_one_model_out z_model_layer_family           log_loss  0.788570 0.604199       0.604335   0.184370            0.184234
 leave_one_model_out z_model_layer_family              brier  0.191133 0.158735       0.158768   0.032397            0.032365
 leave_one_model_out z_model_layer_family               AURC  0.498974 0.477767       0.477767   0.021207            0.021207
 leave_one_model_out z_model_layer_family coverage@risk=0.30  0.280303 0.356061       0.356061   0.075758            0.075758
 leave_one_model_out z_model_layer_family risk@coverage=0.50  0.500000 0.469697       0.469697   0.030303            0.030303
 leave_one_model_out z_model_layer_family                ECE  0.195151 0.129385       0.129369   0.065766            0.065782
leave_one_source_out             absolute           log_loss  1.851963 3.091247       3.432283  -1.239285           -1.580321
leave_one_source_out             absolute              brier  0.355386 0.385107       0.387255  -0.029722           -0.031869
leave_one_source_out             absolute               AURC  0.636755 0.639532       0.640581  -0.002778           -0.003826
leave_one_source_out             absolute coverage@risk=0.30  0.000000 0.000000       0.000000   0.000000            0.000000
leave_one_source_out             absolute risk@coverage=0.50  0.560606 0.560606       0.560606   0.000000            0.000000
leave_one_source_out             absolute                ECE  0.336919 0.376763       0.379360  -0.039844           -0.042441
leave_one_source_out                  adj           log_loss  1.851963 3.569829       3.645551  -1.717867           -1.793588
leave_one_source_out                  adj              brier  0.355386 0.388430       0.389435  -0.033044           -0.034049
leave_one_source_out                  adj               AURC  0.636755 0.642164       0.638048  -0.005409           -0.001294
leave_one_source_out                  adj coverage@risk=0.30  0.000000 0.000000       0.000000   0.000000            0.000000
leave_one_source_out                  adj risk@coverage=0.50  0.560606 0.560606       0.560606   0.000000            0.000000
leave_one_source_out                  adj                ECE  0.336919 0.381048       0.382459  -0.044129           -0.045540
leave_one_source_out                 rank           log_loss  1.851963 3.684113       3.706313  -1.832150           -1.854350
leave_one_source_out                 rank              brier  0.355386 0.390300       0.391058  -0.034914           -0.035672
leave_one_source_out                 rank               AURC  0.636755 0.633933       0.626751   0.002821            0.010004
leave_one_source_out                 rank coverage@risk=0.30  0.000000 0.000000       0.000000   0.000000            0.000000
leave_one_source_out                 rank risk@coverage=0.50  0.560606 0.560606       0.560606   0.000000            0.000000
leave_one_source_out                 rank                ECE  0.336919 0.383637       0.384633  -0.046718           -0.047714
leave_one_source_out           percentile           log_loss  1.851963 3.719871       3.729958  -1.867909           -1.877995
leave_one_source_out           percentile              brier  0.355386 0.391736       0.392359  -0.036350           -0.036973
leave_one_source_out           percentile               AURC  0.636755 0.630866       0.630866   0.005888            0.005888
leave_one_source_out           percentile coverage@risk=0.30  0.000000 0.000000       0.000000   0.000000            0.000000
leave_one_source_out           percentile risk@coverage=0.50  0.560606 0.560606       0.560606   0.000000            0.000000
leave_one_source_out           percentile                ECE  0.336919 0.379494       0.380346  -0.042575           -0.043427
leave_one_source_out z_model_layer_family           log_loss  1.851963 3.738186       3.745019  -1.886224           -1.893056
leave_one_source_out z_model_layer_family              brier  0.355386 0.392912       0.393416  -0.037527           -0.038031
leave_one_source_out z_model_layer_family               AURC  0.636755 0.629477       0.629477   0.007277            0.007277
leave_one_source_out z_model_layer_family coverage@risk=0.30  0.000000 0.000000       0.000000   0.000000            0.000000
leave_one_source_out z_model_layer_family risk@coverage=0.50  0.560606 0.560606       0.560606   0.000000            0.000000
leave_one_source_out z_model_layer_family                ECE  0.336919 0.381091       0.381753  -0.044172           -0.044834
```

## Leave-One-Source Breakdown
```text
         rho_feature      heldout_source  n  base_correct_rate  delta_log_loss  delta_brier  delta_AURC  delta_risk50
            absolute     decoder_battery 24           0.250000        0.006208     0.000377    0.000000      0.000000
            absolute masked_full_battery 63           0.365079       -0.022720    -0.006491    0.000000      0.000000
            absolute         masked_topk 45           0.311111       -3.606738    -0.078297   -0.029944     -0.086957
                 adj     decoder_battery 24           0.250000        0.058650     0.002515    0.000000      0.000000
                 adj masked_full_battery 63           0.365079       -0.045474    -0.012790    0.000000      0.000000
                 adj         masked_topk 45           0.311111       -5.006692    -0.080363    0.067396      0.000000
                rank     decoder_battery 24           0.250000        0.088897     0.003870    0.007407      0.000000
                rank masked_full_battery 63           0.365079       -0.062105    -0.017218    0.000000      0.000000
                rank         masked_topk 45           0.311111       -5.334773    -0.080375    0.064060      0.086957
          percentile     decoder_battery 24           0.250000        0.110152     0.004880    0.007407      0.000000
          percentile masked_full_battery 63           0.365079       -0.075195    -0.020609    0.000000      0.000000
          percentile         masked_topk 45           0.311111       -5.432674    -0.080376    0.083037      0.130435
z_model_layer_family     decoder_battery 24           0.250000        0.126160     0.005673    0.007407      0.000000
z_model_layer_family masked_full_battery 63           0.365079       -0.086072    -0.023376    0.000000      0.000000
z_model_layer_family         masked_topk 45           0.311111       -5.479707    -0.080377    0.083037      0.130435
```

## Policy Metrics
```text
       policy   n  accuracy    brier  log_loss    auroc  average_precision  aurc_mean_risk  coverage_at_risk_030  risk_at_50pct_coverage  risk_at_80pct_coverage      ece  high_pred_reliable_error_rate  high_pred_reliable_coverage  top20pct_error_rate                split          rho_feature
     baseline 132  0.325758 0.153992  0.511038 0.863601           0.758355        0.469461              0.310606                0.439394                0.594340 0.118130                       0.200000                     0.265152             0.148148       grouped_prompt             absolute
         +rho 132  0.325758 0.115138  0.382033 0.911158           0.825349        0.446009              0.356061                0.393939                0.594340 0.105338                       0.125000                     0.242424             0.148148       grouped_prompt             absolute
+shuffled_rho 132  0.325758 0.117320  0.393605 0.908283           0.815189        0.446009              0.356061                0.393939                0.594340 0.095382                       0.125000                     0.242424             0.148148       grouped_prompt             absolute
     baseline 132  0.325758 0.153992  0.511038 0.863601           0.758355        0.469461              0.310606                0.439394                0.594340 0.118130                       0.200000                     0.265152             0.148148       grouped_prompt                  adj
         +rho 132  0.325758 0.118837  0.400652 0.906715           0.808089        0.446009              0.356061                0.393939                0.594340 0.088385                       0.129032                     0.234848             0.148148       grouped_prompt                  adj
+shuffled_rho 132  0.325758 0.119996  0.405523 0.905148           0.805169        0.448787              0.356061                0.393939                0.594340 0.087227                       0.129032                     0.234848             0.148148       grouped_prompt                  adj
     baseline 132  0.325758 0.153992  0.511038 0.863601           0.758355        0.469461              0.310606                0.439394                0.594340 0.118130                       0.200000                     0.265152             0.148148       grouped_prompt                 rank
         +rho 132  0.325758 0.120957  0.409274 0.904364           0.804600        0.449982              0.356061                0.393939                0.594340 0.075509                       0.156250                     0.242424             0.148148       grouped_prompt                 rank
+shuffled_rho 132  0.325758 0.121796  0.412376 0.902535           0.801384        0.451665              0.356061                0.409091                0.594340 0.066289                       0.156250                     0.242424             0.148148       grouped_prompt                 rank
     baseline 132  0.325758 0.153992  0.511038 0.863601           0.758355        0.469461              0.310606                0.439394                0.594340 0.118130                       0.200000                     0.265152             0.148148       grouped_prompt           percentile
         +rho 132  0.325758 0.122556  0.415083 0.900967           0.799595        0.451665              0.356061                0.409091                0.594340 0.066506                       0.156250                     0.242424             0.148148       grouped_prompt           percentile
+shuffled_rho 132  0.325758 0.123246  0.417482 0.900183           0.798982        0.451665              0.356061                0.409091                0.594340 0.072147                       0.156250                     0.242424             0.148148       grouped_prompt           percentile
     baseline 132  0.325758 0.153992  0.511038 0.863601           0.758355        0.469461              0.310606                0.439394                0.594340 0.118130                       0.200000                     0.265152             0.148148       grouped_prompt z_model_layer_family
         +rho 132  0.325758 0.123897  0.419688 0.898876           0.796889        0.451665              0.356061                0.409091                0.594340 0.070745                       0.156250                     0.242424             0.148148       grouped_prompt z_model_layer_family
+shuffled_rho 132  0.325758 0.124496  0.421695 0.897831           0.795733        0.451665              0.356061                0.409091                0.594340 0.070839                       0.156250                     0.242424             0.148148       grouped_prompt z_model_layer_family
     baseline 132  0.325758 0.191133  0.788570 0.813953           0.667542        0.498974              0.280303                0.500000                0.603774 0.195151                       0.257143                     0.265152             0.222222  leave_one_model_out             absolute
         +rho 132  0.325758 0.156869  0.590796 0.858375           0.743106        0.476083              0.356061                0.454545                0.594340 0.125261                       0.225806                     0.234848             0.222222  leave_one_model_out             absolute
+shuffled_rho 132  0.325758 0.157838  0.598380 0.857330           0.740972        0.476083              0.356061                0.454545                0.594340 0.127883                       0.225806                     0.234848             0.222222  leave_one_model_out             absolute
     baseline 132  0.325758 0.191133  0.788570 0.813953           0.667542        0.498974              0.280303                0.500000                0.603774 0.195151                       0.257143                     0.265152             0.222222  leave_one_model_out                  adj
         +rho 132  0.325758 0.158212  0.601118 0.857068           0.739281        0.476083              0.356061                0.454545                0.594340 0.127933                       0.225806                     0.234848             0.222222  leave_one_model_out                  adj
+shuffled_rho 132  0.325758 0.158400  0.602384 0.857068           0.738937        0.476083              0.356061                0.454545                0.594340 0.127935                       0.225806                     0.234848             0.222222  leave_one_model_out                  adj
     baseline 132  0.325758 0.191133  0.788570 0.813953           0.667542        0.498974              0.280303                0.500000                0.603774 0.195151                       0.257143                     0.265152             0.222222  leave_one_model_out                 rank
         +rho 132  0.325758 0.158518  0.603099 0.856546           0.738559        0.477767              0.356061                0.469697                0.594340 0.127913                       0.225806                     0.234848             0.222222  leave_one_model_out                 rank
+shuffled_rho 132  0.325758 0.158599  0.603523 0.856546           0.738559        0.477767              0.356061                0.469697                0.594340 0.140038                       0.250000                     0.242424             0.222222  leave_one_model_out                 rank
     baseline 132  0.325758 0.191133  0.788570 0.813953           0.667542        0.498974              0.280303                0.500000                0.603774 0.195151                       0.257143                     0.265152             0.222222  leave_one_model_out           percentile
         +rho 132  0.325758 0.158659  0.603873 0.856284           0.738112        0.477767              0.356061                0.469697                0.594340 0.129421                       0.250000                     0.242424             0.222222  leave_one_model_out           percentile
+shuffled_rho 132  0.325758 0.158703  0.604092 0.855762           0.736438        0.477767              0.356061                0.469697                0.594340 0.129399                       0.250000                     0.242424             0.222222  leave_one_model_out           percentile
     baseline 132  0.325758 0.191133  0.788570 0.813953           0.667542        0.498974              0.280303                0.500000                0.603774 0.195151                       0.257143                     0.265152             0.222222  leave_one_model_out z_model_layer_family
         +rho 132  0.325758 0.158735  0.604199 0.855762           0.736438        0.477767              0.356061                0.469697                0.594340 0.129385                       0.250000                     0.242424             0.222222  leave_one_model_out z_model_layer_family
+shuffled_rho 132  0.325758 0.158768  0.604335 0.855762           0.736438        0.477767              0.356061                0.469697                0.594340 0.129369                       0.250000                     0.242424             0.222222  leave_one_model_out z_model_layer_family
     baseline 132  0.325758 0.355386  1.851963 0.581918           0.357679        0.636755              0.000000                0.560606                0.650943 0.336919                       0.658537                     0.310606             0.703704 leave_one_source_out             absolute
         +rho 132  0.325758 0.385107  3.091247 0.574602           0.347045        0.639532              0.000000                0.560606                0.650943 0.376763                       0.688889                     0.340909             0.703704 leave_one_source_out             absolute
+shuffled_rho 132  0.325758 0.387255  3.432283 0.579044           0.352945        0.640581              0.000000                0.560606                0.641509 0.379360                       0.688889                     0.340909             0.703704 leave_one_source_out             absolute
     baseline 132  0.325758 0.355386  1.851963 0.581918           0.357679        0.636755              0.000000                0.560606                0.650943 0.336919                       0.658537                     0.310606             0.703704 leave_one_source_out                  adj
         +rho 132  0.325758 0.388430  3.569829 0.582441           0.361415        0.642164              0.000000                0.560606                0.641509 0.381048                       0.688889                     0.340909             0.703704 leave_one_source_out                  adj
+shuffled_rho 132  0.325758 0.389435  3.645551 0.580350           0.356779        0.638048              0.000000                0.560606                0.641509 0.382459                       0.688889                     0.340909             0.666667 leave_one_source_out                  adj
     baseline 132  0.325758 0.355386  1.851963 0.581918           0.357679        0.636755              0.000000                0.560606                0.650943 0.336919                       0.658537                     0.310606             0.703704 leave_one_source_out                 rank
         +rho 132  0.325758 0.390300  3.684113 0.580089           0.356632        0.633933              0.000000                0.560606                0.641509 0.383637                       0.688889                     0.340909             0.629630 leave_one_source_out                 rank
+shuffled_rho 132  0.325758 0.391058  3.706313 0.585054           0.362065        0.626751              0.000000                0.560606                0.650943 0.384633                       0.688889                     0.340909             0.555556 leave_one_source_out                 rank
     baseline 132  0.325758 0.355386  1.851963 0.581918           0.357679        0.636755              0.000000                0.560606                0.650943 0.336919                       0.658537                     0.310606             0.703704 leave_one_source_out           percentile
         +rho 132  0.325758 0.391736  3.719871 0.582963           0.359566        0.630866              0.000000                0.560606                0.650943 0.379494                       0.688889                     0.340909             0.592593 leave_one_source_out           percentile
+shuffled_rho 132  0.325758 0.392359  3.729958 0.582963           0.359479        0.630866              0.000000                0.560606                0.650943 0.380346                       0.688889                     0.340909             0.592593 leave_one_source_out           percentile
     baseline 132  0.325758 0.355386  1.851963 0.581918           0.357679        0.636755              0.000000                0.560606                0.650943 0.336919                       0.658537                     0.310606             0.703704 leave_one_source_out z_model_layer_family
         +rho 132  0.325758 0.392912  3.738186 0.583224           0.359652        0.629477              0.000000                0.560606                0.650943 0.381091                       0.688889                     0.340909             0.592593 leave_one_source_out z_model_layer_family
+shuffled_rho 132  0.325758 0.393416  3.745019 0.582963           0.359548        0.629477              0.000000                0.560606                0.650943 0.381753                       0.688889                     0.340909             0.592593 leave_one_source_out z_model_layer_family
```

## Baseline Vs Baseline+rho Deltas
```text
         split rho_feature                                          metric    value
grouped_prompt    absolute                         delta_brier_improvement 0.038853
grouped_prompt    absolute                      delta_log_loss_improvement 0.129006
grouped_prompt    absolute                          delta_aurc_improvement 0.023452
grouped_prompt    absolute          delta_coverage_at_risk_030_improvement 0.045455
grouped_prompt    absolute                        delta_risk50_improvement 0.045455
grouped_prompt    absolute                           delta_ece_improvement 0.012792
grouped_prompt    absolute                shuffled_delta_brier_improvement 0.036671
grouped_prompt    absolute             shuffled_delta_log_loss_improvement 0.117434
grouped_prompt    absolute                 shuffled_delta_aurc_improvement 0.023452
grouped_prompt    absolute shuffled_delta_coverage_at_risk_030_improvement 0.045455
grouped_prompt    absolute               shuffled_delta_risk50_improvement 0.045455
grouped_prompt    absolute                  shuffled_delta_ece_improvement 0.022748
grouped_prompt         adj                         delta_brier_improvement 0.035155
grouped_prompt         adj                      delta_log_loss_improvement 0.110387
grouped_prompt         adj                          delta_aurc_improvement 0.023452
grouped_prompt         adj          delta_coverage_at_risk_030_improvement 0.045455
grouped_prompt         adj                        delta_risk50_improvement 0.045455
grouped_prompt         adj                           delta_ece_improvement 0.029745
grouped_prompt         adj                shuffled_delta_brier_improvement 0.033996
grouped_prompt         adj             shuffled_delta_log_loss_improvement 0.105515
```

## Bootstrap CI
```text
         split rho_feature                                          metric    ci_low   median  ci_high
grouped_prompt    absolute                          delta_aurc_improvement  0.010256 0.026198 0.046714
grouped_prompt    absolute                         delta_brier_improvement  0.024502 0.038572 0.054906
grouped_prompt    absolute          delta_coverage_at_risk_030_improvement  0.000000 0.051095 0.151731
grouped_prompt    absolute                           delta_ece_improvement -0.032781 0.013802 0.056060
grouped_prompt    absolute                      delta_log_loss_improvement  0.081843 0.127824 0.183186
grouped_prompt    absolute                        delta_risk50_improvement  0.000000 0.046875 0.106088
grouped_prompt    absolute                 shuffled_delta_aurc_improvement  0.009254 0.025215 0.045682
grouped_prompt    absolute                shuffled_delta_brier_improvement  0.022062 0.036467 0.053176
grouped_prompt    absolute shuffled_delta_coverage_at_risk_030_improvement  0.000000 0.049238 0.135403
grouped_prompt    absolute                  shuffled_delta_ece_improvement -0.022779 0.017304 0.058718
grouped_prompt    absolute             shuffled_delta_log_loss_improvement  0.073261 0.116295 0.169520
grouped_prompt    absolute               shuffled_delta_risk50_improvement  0.000000 0.044118 0.098592
grouped_prompt         adj                          delta_aurc_improvement  0.008797 0.025080 0.044383
grouped_prompt         adj                         delta_brier_improvement  0.020740 0.034865 0.051480
grouped_prompt         adj          delta_coverage_at_risk_030_improvement  0.000000 0.048000 0.134343
grouped_prompt         adj                           delta_ece_improvement -0.018753 0.023365 0.064170
grouped_prompt         adj                      delta_log_loss_improvement  0.063464 0.109999 0.159482
grouped_prompt         adj                        delta_risk50_improvement  0.000000 0.042254 0.092308
grouped_prompt         adj                 shuffled_delta_aurc_improvement  0.007737 0.024113 0.042257
grouped_prompt         adj                shuffled_delta_brier_improvement  0.019468 0.033719 0.050440
grouped_prompt         adj shuffled_delta_coverage_at_risk_030_improvement  0.000000 0.047619 0.133858
grouped_prompt         adj                  shuffled_delta_ece_improvement -0.015661 0.026206 0.066168
grouped_prompt         adj             shuffled_delta_log_loss_improvement  0.058729 0.105153 0.155630
grouped_prompt         adj               shuffled_delta_risk50_improvement  0.000000 0.042254 0.092308
grouped_prompt        rank                          delta_aurc_improvement  0.006850 0.023291 0.044217
grouped_prompt        rank                         delta_brier_improvement  0.017797 0.032694 0.050890
grouped_prompt        rank          delta_coverage_at_risk_030_improvement  0.000000 0.047244 0.131794
grouped_prompt        rank                           delta_ece_improvement -0.012469 0.030226 0.073062
grouped_prompt        rank                      delta_log_loss_improvement  0.058340 0.100830 0.152139
grouped_prompt        rank                        delta_risk50_improvement  0.000000 0.041667 0.095276
grouped_prompt        rank                 shuffled_delta_aurc_improvement  0.005117 0.021845 0.042530
grouped_prompt        rank                shuffled_delta_brier_improvement  0.017003 0.031972 0.050033
grouped_prompt        rank shuffled_delta_coverage_at_risk_030_improvement  0.000000 0.046154 0.130098
grouped_prompt        rank                  shuffled_delta_ece_improvement -0.007488 0.032296 0.074207
grouped_prompt        rank             shuffled_delta_log_loss_improvement  0.054550 0.098183 0.148558
grouped_prompt        rank               shuffled_delta_risk50_improvement -0.012512 0.032787 0.090944
grouped_prompt  percentile                          delta_aurc_improvement  0.004238 0.020619 0.041523
grouped_prompt  percentile                         delta_brier_improvement  0.015768 0.031102 0.047560
grouped_prompt  percentile          delta_coverage_at_risk_030_improvement  0.000000 0.042857 0.126119
grouped_prompt  percentile                           delta_ece_improvement -0.008024 0.032545 0.077759
```

## Risk-Coverage
```text
         split          rho_feature  requested_coverage     +rho  +shuffled_rho  baseline  risk_improvement
grouped_prompt             absolute                 0.2 0.148148       0.148148  0.148148          0.000000
grouped_prompt             absolute                 0.3 0.200000       0.200000  0.275000          0.075000
grouped_prompt             absolute                 0.4 0.339623       0.339623  0.396226          0.056604
grouped_prompt             absolute                 0.5 0.393939       0.393939  0.439394          0.045455
grouped_prompt             absolute                 0.6 0.487500       0.487500  0.500000          0.012500
grouped_prompt             absolute                 0.7 0.537634       0.537634  0.559140          0.021505
grouped_prompt             absolute                 0.8 0.594340       0.594340  0.594340          0.000000
grouped_prompt             absolute                 0.9 0.638655       0.638655  0.638655          0.000000
grouped_prompt             absolute                 1.0 0.674242       0.674242  0.674242          0.000000
grouped_prompt                  adj                 0.2 0.148148       0.148148  0.148148          0.000000
grouped_prompt                  adj                 0.3 0.200000       0.225000  0.275000          0.075000
grouped_prompt                  adj                 0.4 0.339623       0.339623  0.396226          0.056604
grouped_prompt                  adj                 0.5 0.393939       0.393939  0.439394          0.045455
grouped_prompt                  adj                 0.6 0.487500       0.487500  0.500000          0.012500
grouped_prompt                  adj                 0.7 0.537634       0.537634  0.559140          0.021505
grouped_prompt                  adj                 0.8 0.594340       0.594340  0.594340          0.000000
grouped_prompt                  adj                 0.9 0.638655       0.638655  0.638655          0.000000
grouped_prompt                  adj                 1.0 0.674242       0.674242  0.674242          0.000000
grouped_prompt           percentile                 0.2 0.148148       0.148148  0.148148          0.000000
grouped_prompt           percentile                 0.3 0.225000       0.225000  0.275000          0.050000
grouped_prompt           percentile                 0.4 0.339623       0.339623  0.396226          0.056604
grouped_prompt           percentile                 0.5 0.409091       0.409091  0.439394          0.030303
grouped_prompt           percentile                 0.6 0.487500       0.487500  0.500000          0.012500
grouped_prompt           percentile                 0.7 0.548387       0.548387  0.559140          0.010753
grouped_prompt           percentile                 0.8 0.594340       0.594340  0.594340          0.000000
grouped_prompt           percentile                 0.9 0.638655       0.638655  0.638655          0.000000
grouped_prompt           percentile                 1.0 0.674242       0.674242  0.674242          0.000000
grouped_prompt                 rank                 0.2 0.148148       0.148148  0.148148          0.000000
grouped_prompt                 rank                 0.3 0.225000       0.225000  0.275000          0.050000
grouped_prompt                 rank                 0.4 0.339623       0.339623  0.396226          0.056604
grouped_prompt                 rank                 0.5 0.393939       0.409091  0.439394          0.045455
grouped_prompt                 rank                 0.6 0.487500       0.487500  0.500000          0.012500
grouped_prompt                 rank                 0.7 0.548387       0.548387  0.559140          0.010753
grouped_prompt                 rank                 0.8 0.594340       0.594340  0.594340          0.000000
grouped_prompt                 rank                 0.9 0.638655       0.638655  0.638655          0.000000
grouped_prompt                 rank                 1.0 0.674242       0.674242  0.674242          0.000000
grouped_prompt z_model_layer_family                 0.2 0.148148       0.148148  0.148148          0.000000
grouped_prompt z_model_layer_family                 0.3 0.225000       0.225000  0.275000          0.050000
grouped_prompt z_model_layer_family                 0.4 0.339623       0.339623  0.396226          0.056604
grouped_prompt z_model_layer_family                 0.5 0.409091       0.409091  0.439394          0.030303
```

## Rho-Guided Action Mix
```text
               action  n  coverage  error_rate  rho_action_mean  predicted_correct_mean
     abstain_or_route 28  0.212121    1.000000         4.892857                0.015406
               accept 40  0.303030    0.200000        20.700000                0.891345
intervene_or_retrieve 43  0.325758    0.837209        30.697674                0.200282
  pass_to_other_model 21  0.159091    0.809524         5.619048                0.245445
```

## Files
```text
selective_reliability_dataset.csv
selective_reliability_predictions.csv
selective_reliability_metrics.csv
selective_reliability_final_table.csv
leave_one_source_breakdown.csv
selective_reliability_deltas.csv
selective_reliability_bootstrap_ci.csv
risk_coverage_curve.csv
rho_guided_action_mix.csv
report.md
```
