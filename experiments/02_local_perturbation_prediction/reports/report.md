# DistilBERT Geometry Intervention and Ablation Tests

Model: distilbert-base-uncased. Dataset: generated factual cloze prompts from the existing factual-error benchmark. Labels are observed model correctness: top-1 target match vs error.

## Prompt Counts
```text
       topic  correct  error
animal_class        4     12
     antonym        1     19
     capital       97     11
      common        6     10
   continent        0     24
     culture        9      7
    currency       10      6
    language       30      2
        math        1     15
      plural        6     14
        role       12      8
     science        7      9
```

## Feature-Set AUC Summary
```text
 split_kind                      feature_set  n_splits  auc_mean  auc_std  auc_min  auc_max
 fact_group     scalar_plus_rho_fisher_delta        80  0.888444 0.034251 0.790276 0.961031
 fact_group  scalar_plus_rho_euclidean_delta        80  0.884594 0.035966 0.780496 0.950725
 fact_group                 rho_fisher_delta        80  0.883562 0.026893 0.802612 0.946286
 fact_group             scalar_layer_profile        80  0.874250 0.037456 0.769149 0.950403
 fact_group scalar_plus_matched_random_delta        80  0.870622 0.037381 0.776950 0.949758
 fact_group               rho_random_u_delta        80  0.791944 0.037360 0.688112 0.872401
 fact_group   rho_matched_random_trace_delta        80  0.771065 0.043540 0.654184 0.861464
 fact_group      rho_high_trace_random_delta        80  0.763822 0.040228 0.676550 0.866023
 fact_group              rho_euclidean_delta        80  0.756897 0.038176 0.665257 0.841868
 fact_group             rho_shuffled_u_delta        80  0.732094 0.038121 0.631034 0.802773
 fact_group                 rho_fisher_state        80  0.732085 0.044874 0.615862 0.824231
leave_topic                 rho_fisher_delta        11  0.754205 0.170626 0.315789 1.000000
leave_topic     scalar_plus_rho_fisher_delta        11  0.664491 0.248840 0.133333 0.953140
leave_topic               rho_random_u_delta        11  0.641117 0.196554 0.368421 0.904762
leave_topic   rho_matched_random_trace_delta        11  0.626720 0.185483 0.312500 0.866667
leave_topic             scalar_layer_profile        11  0.604199 0.205732 0.315789 0.965323
leave_topic  scalar_plus_rho_euclidean_delta        11  0.602349 0.211535 0.263158 0.959700
leave_topic scalar_plus_matched_random_delta        11  0.595771 0.223619 0.263158 0.956888
leave_topic      rho_high_trace_random_delta        11  0.574755 0.202826 0.266667 0.873016
leave_topic                 rho_fisher_state        11  0.535171 0.209360 0.125000 0.866667
leave_topic              rho_euclidean_delta        11  0.520899 0.218132 0.052632 0.857143
leave_topic             rho_shuffled_u_delta        11  0.513719 0.223230 0.066667 0.856607
```

## Paired Feature Deltas
```text
 split_kind                             left                right  left_auc_mean  right_auc_mean  delta_auc_mean
 fact_group                 rho_fisher_delta scalar_layer_profile       0.883562        0.874250        0.009312
 fact_group              rho_euclidean_delta     rho_fisher_delta       0.756897        0.883562       -0.126664
 fact_group             rho_shuffled_u_delta     rho_fisher_delta       0.732094        0.883562       -0.151467
 fact_group               rho_random_u_delta     rho_fisher_delta       0.791944        0.883562       -0.091617
 fact_group   rho_matched_random_trace_delta     rho_fisher_delta       0.771065        0.883562       -0.112496
 fact_group     scalar_plus_rho_fisher_delta scalar_layer_profile       0.888444        0.874250        0.014194
 fact_group  scalar_plus_rho_euclidean_delta scalar_layer_profile       0.884594        0.874250        0.010344
 fact_group scalar_plus_matched_random_delta scalar_layer_profile       0.870622        0.874250       -0.003628
leave_topic                 rho_fisher_delta scalar_layer_profile       0.754205        0.604199        0.150006
leave_topic              rho_euclidean_delta     rho_fisher_delta       0.520899        0.754205       -0.233306
leave_topic             rho_shuffled_u_delta     rho_fisher_delta       0.513719        0.754205       -0.240486
leave_topic               rho_random_u_delta     rho_fisher_delta       0.641117        0.754205       -0.113088
leave_topic   rho_matched_random_trace_delta     rho_fisher_delta       0.626720        0.754205       -0.127485
leave_topic     scalar_plus_rho_fisher_delta scalar_layer_profile       0.664491        0.604199        0.060292
leave_topic  scalar_plus_rho_euclidean_delta scalar_layer_profile       0.602349        0.604199       -0.001851
leave_topic scalar_plus_matched_random_delta scalar_layer_profile       0.595771        0.604199       -0.008428
```

## Controlled Latent Intervention Summary
```text
           direction  epsilon    n  delta_entropy_mean  delta_entropy_median  abs_delta_varentropy_mean  delta_varentropy_mean  top1_changed_rate  delta_target_prob_mean  rho_direction_mean  trace_fim_direction_mean
            delta_1d     0.25 1920            0.011957              0.004297                   0.021170              -0.017157           0.031771               -0.001392            0.185820                  0.018984
            delta_1d     0.50 1920            0.023193              0.008018                   0.042283              -0.033889           0.060417               -0.002819            0.185820                  0.018984
            delta_1d     1.00 1920            0.043175              0.013896                   0.085216              -0.065230           0.123438               -0.005759            0.185820                  0.018984
   high_trace_random     0.25 1920            0.021808              0.011662                   0.048489              -0.044011           0.068750               -0.001965            0.231391                  0.059053
   high_trace_random     0.50 1920            0.039773              0.018964                   0.091401              -0.080433           0.138021               -0.004029            0.231391                  0.059053
   high_trace_random     1.00 1920            0.063878              0.020490                   0.164965              -0.128776           0.308333               -0.008421            0.231391                  0.059053
matched_random_trace     0.25 1920            0.008828              0.003867                   0.017337              -0.014906           0.026562               -0.000952            0.117140                  0.019510
matched_random_trace     0.50 1920            0.017128              0.007006                   0.034606              -0.029499           0.057813               -0.001965            0.117140                  0.019510
matched_random_trace     1.00 1920            0.031911              0.012120                   0.069485              -0.057147           0.103646               -0.004165            0.117140                  0.019510
```

## Delta-Direction Intervention by Rho Quartile
```text
 epsilon rho_bin   n  rho_mean  trace_fim_mean  delta_entropy_mean  abs_delta_varentropy_mean  top1_changed_rate
    0.25  q1_low 480  0.004985        0.020712            0.000881                   0.003783           0.047917
    0.25      q2 480  0.037726        0.017591            0.003860                   0.010482           0.043750
    0.25      q3 480  0.152789        0.015502            0.008769                   0.019478           0.031250
    0.25 q4_high 480  0.547779        0.022133            0.034316                   0.050936           0.004167
    0.50  q1_low 480  0.004985        0.020712            0.000686                   0.008045           0.085417
    0.50      q2 480  0.037726        0.017591            0.006920                   0.019842           0.085417
    0.50      q3 480  0.152789        0.015502            0.016773                   0.037659           0.060417
    0.50 q4_high 480  0.547779        0.022133            0.068394                   0.103587           0.010417
    1.00  q1_low 480  0.004985        0.020712           -0.003081                   0.021565           0.179167
    1.00      q2 480  0.037726        0.017591            0.010452                   0.037013           0.164583
    1.00      q3 480  0.152789        0.015502            0.030241                   0.069543           0.125000
    1.00 q4_high 480  0.547779        0.022133            0.135086                   0.212742           0.025000
```

## Intervention Residual Correlations
```text
 epsilon                  outcome    n  pearson_rho_outcome  partial_corr_after_entropy_varentropy_trace
    0.25            delta_entropy 1920             0.811310                                     0.781382
    0.25     abs_delta_varentropy 1920             0.629444                                     0.512775
    0.25 top1_changed_within_topk 1920            -0.095886                                    -0.035494
    0.50            delta_entropy 1920             0.816230                                     0.774126
    0.50     abs_delta_varentropy 1920             0.640610                                     0.517881
    0.50 top1_changed_within_topk 1920            -0.127213                                    -0.052253
    1.00            delta_entropy 1920             0.824878                                     0.760555
    1.00     abs_delta_varentropy 1920             0.656155                                     0.523118
    1.00 top1_changed_within_topk 1920            -0.176825                                    -0.049175
```

## Random-Subspace Saturation
```text
  k observed_condition    n  rho_mean      rho_std  rho_p10  rho_p50  rho_p90  trace_fim_mean
  1            correct 1281  0.111854 1.647329e-01 0.000709 0.039544 0.378078        0.036088
  1              error  959  0.095185 1.306175e-01 0.001385 0.041634 0.285033        0.037002
  2            correct 1281  0.219798 1.981095e-01 0.032557 0.163409 0.497089        0.081449
  2              error  959  0.163432 1.472345e-01 0.020499 0.115497 0.375103        0.087519
  4            correct 1281  0.363558 2.375814e-01 0.087677 0.323813 0.717427        0.128979
  4              error  959  0.289037 1.911139e-01 0.085718 0.247998 0.569840        0.142364
  8            correct 1281  0.557142 2.105122e-01 0.265030 0.559521 0.859786        0.288472
  8              error  959  0.499850 1.836102e-01 0.251689 0.511983 0.754094        0.313239
 16            correct 1281  0.764372 1.525179e-01 0.561338 0.795229 0.950347        0.589066
 16              error  959  0.732071 1.379460e-01 0.551610 0.750075 0.904073        0.634085
 32            correct 1281  1.000000 8.434548e-15 1.000000 1.000000 1.000000        1.181232
 32              error  959  1.000000 3.399380e-13 1.000000 1.000000 1.000000        1.273824
 64            correct 1281  1.000000 4.786026e-16 1.000000 1.000000 1.000000        2.362675
 64              error  959  1.000000 3.939301e-16 1.000000 1.000000 1.000000        2.529956
128            correct 1281  1.000000 1.220591e-15 1.000000 1.000000 1.000000        4.551434
128              error  959  1.000000 4.521429e-16 1.000000 1.000000 1.000000        4.910785
256            correct 1281  1.000000 3.704909e-15 1.000000 1.000000 1.000000        9.283240
256              error  959  1.000000 1.328239e-15 1.000000 1.000000 1.000000        9.958915
768            correct 1281  1.000000 4.754163e-15 1.000000 1.000000 1.000000       27.913721
768              error  959  1.000000 4.121732e-15 1.000000 1.000000 1.000000       29.936937
```

## Directional Token Shift Examples
```text
 prompt_id           kind  layer  rho_delta                                    prompt    target final_top1_token                                                      top_up_tokens                                                    top_down_tokens
       121 high_rho_layer      5   0.956680           Nepal's capital city is [MASK]. kathmandu        kathmandu      bethlehem santiago geneva beijing delhi lima nairobi florence                kathmandu bow excel ##raz city ##sit ##pala ##oping
       336 high_rho_layer      5   0.953844 Sherlock [MASK] is a fictional detective.    holmes           holmes                       hyde jones newton fox moss knox watson burns      holmes ##cend ##partisan ##rcle ##mber ##cite ##culture ##vao
        58 high_rho_layer      5   0.944666          Serbia's capital city is [MASK].  belgrade         belgrade            bergen sofia wrocław lima athens budapest geneva moscow                  belgrade hague serbian ##oja spa novi ##cle ##end
        55 high_rho_layer      5   0.942562        Bulgaria's capital city is [MASK].     sofia            sofia             tallinn lima santiago kiev moscow athens tirana bergen                  sofia ruse ##rov capital ##sit ##ning ##rel takes
        88 high_rho_layer      5   0.941947            Peru's capital city is [MASK].      lima             lima santiago toledo concepcion bogota mendoza montevideo sofia caracas         lima lap ##asian ##hair ##raphic ##actic ##thic ##national
        60 high_rho_layer      5   0.941945         The capital of Croatia is [MASK].    zagreb           zagreb       wrocław bergen croatia orange tirana acre sarajevo bethlehem                   zagreb belgrade novi hr croatian ##je serbs city
       122 high_rho_layer      5   0.940015     In Nepal, the capital city is [MASK]. kathmandu        kathmandu              santiago jaipur jakarta delhi geneva pune kabul hanoi               kathmandu doha belgrade ##nagar anand patna city bam
        54 high_rho_layer      5   0.935913        The capital of Bulgaria is [MASK].     sofia            sofia             tallinn tirana santiago skopje kiev bergen petra brest              sofia ruse ##grad ##lub ##kov ##rov capital bulgarian
       120 high_rho_layer      5   0.935051           The capital of Nepal is [MASK]. kathmandu        kathmandu          santiago bethlehem dax jakarta kabul wrocław beijing nara                kathmandu anand bam patna dharma baku rai bangalore
        77 high_rho_layer      5   0.934441    In Jordan, the capital city is [MASK].     amman            amman    beirut bethlehem damascus nazareth jerusalem tunis kabul tirana                      amman doha hussein bled ras ##amen tyre ##lab
        76 high_rho_layer      5   0.931844          Jordan's capital city is [MASK].     amman            amman      bethlehem beirut damascus jerusalem lima acre athens nazareth                    amman bled doha hussein ##amen tyre ##lab sabha
        37 high_rho_layer      5   0.931617         Finland's capital city is [MASK].  helsinki         helsinki           tallinn perth athens hamilton moscow geneva orange sofia              helsinki hague radio finnish ##poo ##oping det ##ning
        61 high_rho_layer      5   0.928864         Croatia's capital city is [MASK].    zagreb           zagreb       wrocław vienna orange bergen perth sarajevo tirana jerusalem                 zagreb split belgrade ##cle novi den ##je croatian
        67 high_rho_layer      5   0.928423            Iran's capital city is [MASK].    tehran           tehran               beirut bethlehem delhi pune damascus hanoi goa haifa                tehran kerman ##fahan hague baku pest ankara munich
        56 high_rho_layer      5   0.921588  In Bulgaria, the capital city is [MASK].     sofia            sofia         santiago tirana tallinn bergen kiev lima jerusalem wrocław            sofia ruse skopje pest bucharest petersburg ##rov ##lev
        82 high_rho_layer      5   0.921534        Thailand's capital city is [MASK].   bangkok          bangkok     geneva perth jerusalem nassau orange saigon wellington tallinn      bangkok moscow munich hague ankara kathmandu guangzhou manila
       335 high_rho_layer      6   0.918327       The title pair is Romeo and [MASK].    juliet           juliet           juliette venus delilah julia cleopatra judy lucia hamlet                  juliet fauna viola yang virgil cora tobago marlon
        69 high_rho_layer      5   0.917699            The capital of Iraq is [MASK].   baghdad          baghdad       beirut toledo tehran kuwait damascus bethlehem bahrain derby                 baghdad amman iraq aleppo iraqi abu arabic kurdish
        62 high_rho_layer      5   0.917531   In Croatia, the capital city is [MASK].    zagreb           zagreb           wrocław tirana bergen vienna hobart perth croatia orange                 zagreb belgrade novi city ##je croatian ##nik serb
        57 high_rho_layer      5   0.912863          The capital of Serbia is [MASK].  belgrade         belgrade         wrocław bergen sofia dax tirana florence jakarta bethlehem        belgrade serbia hague novi serbs helsinki serbian rotterdam
        25 high_rho_layer      5   0.910083          Greece's capital city is [MASK].    athens           athens       aurora geneva jerusalem perth orange sparta bethlehem beirut   athens thessaloniki olympia helsinki belgrade hague sofia skopje
        22 high_rho_layer      5   0.909620           Egypt's capital city is [MASK].     cairo            cairo     hobart perth bethlehem geneva kingston medina beirut jerusalem                  cairo alexandria acre mecca tyre oslo gaza hassan
        59 high_rho_layer      5   0.908606    In Serbia, the capital city is [MASK].  belgrade         belgrade            wrocław bergen tirana jakarta sofia perth hobart geneva            belgrade hague oslo novi pest helsinki rotterdam zagreb
        92 high_rho_layer      5   0.906418     In Chile, the capital city is [MASK].  santiago         santiago      lima concepcion havana caracas sofia darwin aurora montevideo          santiago anal ##sio ##sit ##ivar especially primera ##ras
        70 high_rho_layer      5   0.906129            Iraq's capital city is [MASK].   baghdad          baghdad      beirut toledo jerusalem tehran bethlehem damascus derby delhi                   baghdad amman doha abu iraqi hussein male ##sham
       251 high_rho_layer      5   0.904170   Yes has the opposite meaning of [MASK].        no              yes     heaven ghosts treason everything ours earthquakes god miracles            yes aye rejection yeah denial thanks skepticism divorce
        19 high_rho_layer      5   0.898901          Russia's capital city is [MASK].    moscow           moscow         tallinn perth jerusalem lima geneva orange istanbul vienna moscow kazan helsinki kiev vladimir petersburg leningrad stockholm
        90 high_rho_layer      5   0.895888           The capital of Chile is [MASK].  santiago         santiago         aurora lima mendoza havana darwin dax montevideo concordia     santiago concepcion dante santos universidad chilean juan anal
        75 high_rho_layer      5   0.894057          The capital of Jordan is [MASK].     amman            amman        bethlehem beirut bahrain nazareth wells qatar nassau bergen                     amman damascus doha hal hussein dir ras jordan
        81 high_rho_layer      5   0.893216        The capital of Thailand is [MASK].   bangkok          bangkok            geneva bethlehem nassau saigon aurora macau bahrain goa             bangkok ankara moscow hague rai kathmandu hanoi munich
```
