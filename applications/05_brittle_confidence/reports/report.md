# Application 5: Brittle Confidence

This application asks whether high-confidence predictions with low uncertainty accessibility are more fragile under prompt perturbations that should be semantically innocuous.

Matched groups control for model, task, observed correctness, layer, subspace family, confidence, entropy, and margin as much as the checked-in prompt pool allows.

## Matched Group Counts
```text
                            model           task rho_group  size
                bert-base-uncased   factual_deep  high_rho    16
                bert-base-uncased   factual_deep   low_rho    16
                bert-base-uncased factual_simple  high_rho     7
                bert-base-uncased factual_simple   low_rho     7
          distilbert-base-uncased   factual_deep  high_rho    10
          distilbert-base-uncased   factual_deep   low_rho    10
          distilbert-base-uncased factual_simple  high_rho     7
          distilbert-base-uncased factual_simple   low_rho     7
google/bert_uncased_L-2_H-128_A-2   factual_deep  high_rho     4
google/bert_uncased_L-2_H-128_A-2   factual_deep   low_rho     4
google/bert_uncased_L-2_H-128_A-2 factual_simple  high_rho     4
google/bert_uncased_L-2_H-128_A-2 factual_simple   low_rho     4
```

## Perturbation Summary
```text
rho_group         variant_type  n  rho_mean  confidence_mean  entropy_mean  margin_mean  answer_flip_rate  correctness_loss_rate  answer_prob_drop_mean  entropy_increase_mean  candidate_kl_mean  top10_jaccard_drop_mean  brier_degradation_mean  fragility_score_mean
 high_rho       neutral_prefix 48  0.657621         0.228007      2.887036     0.613631          0.291667               0.041667               0.008947               0.031588           0.164328                 0.394696               -0.014878              1.023643
 high_rho       neutral_suffix 48  0.657621         0.228007      2.887036     0.613631          0.562500               0.145833               0.031014               0.083118           0.423484                 0.558656                0.003097              1.892817
 high_rho          quiz_format 48  0.657621         0.228007      2.887036     0.613631          0.500000               0.125000               0.120939               0.280012           0.695410                 0.694694                0.085916              2.566338
 high_rho synonym_substitution 14  0.669601         0.188451      2.991708     0.497778          0.357143               0.214286               0.046591              -0.113415           0.183041                 0.384352               -0.041458              1.234295
 high_rho     template_rewrite 27  0.661151         0.243015      2.811391     0.613938          0.370370               0.148148              -0.035581              -0.018485           0.203341                 0.385838               -0.054552              1.245542
  low_rho       neutral_prefix 48  0.428902         0.188761      3.009274     0.520115          0.291667               0.041667              -0.018643              -0.063582           0.242038                 0.387849               -0.043990              1.078704
  low_rho       neutral_suffix 48  0.428902         0.188761      3.009274     0.520115          0.395833               0.083333              -0.013798              -0.040870           0.440935                 0.580347               -0.041548              1.665373
  low_rho          quiz_format 48  0.428902         0.188761      3.009274     0.520115          0.500000               0.145833               0.073184               0.062934           0.823696                 0.715235               -0.025114              2.582487
  low_rho synonym_substitution 11  0.396616         0.152050      3.089622     0.330619          0.272727               0.090909               0.013800              -0.183806           0.216437                 0.511849               -0.008424              1.141952
  low_rho     template_rewrite 28  0.472735         0.196137      2.974042     0.548484          0.250000               0.107143              -0.029511              -0.077032           0.226681                 0.442595               -0.076330              1.152938
```

## Low-Rho Minus High-Rho Matched Contrasts
```text
          metric  n_pairs  low_minus_high_mean  low_greater_than_high_rate  low_mean  high_mean
 fragility_score       48            -0.036358                    0.500000  1.661429   1.697787
answer_flip_rate       48            -0.065972                    0.375000  0.355556   0.421528
answer_prob_drop       48            -0.034499                    0.437500  0.006086   0.040585
entropy_increase       48            -0.125506                    0.354167 -0.041003   0.084504
    candidate_kl       48             0.078173                    0.562500  0.465059   0.386886
      top10_drop       48             0.017898                    0.500000  0.549195   0.531296
```

## Files
```text
matched_high_confidence_groups.csv
brittle_perturbation_records.csv
brittle_summary.csv
brittle_matched_contrasts.csv
report.md
```
