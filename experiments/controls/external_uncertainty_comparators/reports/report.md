# External Uncertainty Comparator Diagnostics

This diagnostic compares accessible-varentropy quantities with token-level approximations of Semantic Entropy, Semantic Density, and a HaloScope-style unlabeled consistency risk.

Semantic Entropy is approximated by entropy over embedding clusters of top-m next-token candidates. Semantic Density is an embedding-kernel concentration score converted to uncertainty. HaloScope is not implemented as the full NeurIPS framework; the checked-in column is a HaloScope-style consistency proxy over unlabeled candidate clusters.

The purpose is not to beat these methods on hallucination detection. The purpose is to show that they measure different objects from route-specific accessible varentropy.

## Metric Correlations
```text
                        metric_a                         metric_b   n  spearman
                             rho                       varentropy 108  0.224319
                             rho           semantic_entropy_proxy 108 -0.625497
                             rho     semantic_density_uncertainty 108 -0.629242
                         entropy                              rho 108 -0.650971
                         entropy                       varentropy 108 -0.564221
                         entropy           semantic_entropy_proxy 108  0.941570
                         entropy     semantic_density_uncertainty 108  0.968597
                         entropy haloscope_style_consistency_risk 108  0.941570
                      confidence                              rho 108  0.654830
                      confidence                          entropy 108 -0.954955
                      confidence                       varentropy 108  0.601287
                      confidence           semantic_entropy_proxy 108 -0.902188
                      confidence     semantic_density_uncertainty 108 -0.926898
                      confidence haloscope_style_consistency_risk 108 -0.922780
          semantic_entropy_proxy                       varentropy 108 -0.497812
    semantic_density_uncertainty                       varentropy 108 -0.546976
    semantic_density_uncertainty           semantic_entropy_proxy 108  0.940541
haloscope_style_consistency_risk                              rho 108 -0.644366
haloscope_style_consistency_risk                       varentropy 108 -0.515315
haloscope_style_consistency_risk           semantic_entropy_proxy 108  0.991248
haloscope_style_consistency_risk     semantic_density_uncertainty 108  0.933333
```

## Steering Sensitivity
```text
                          metric              outcome  n  spearman
                             rho    abs_delta_entropy 72  0.646440
                             rho abs_delta_varentropy 72  0.483214
                         entropy    abs_delta_entropy 72 -0.813535
                         entropy abs_delta_varentropy 72 -0.426196
                      varentropy    abs_delta_entropy 72  0.886294
                      varentropy abs_delta_varentropy 72  0.238541
                      confidence    abs_delta_entropy 72  0.828782
                      confidence abs_delta_varentropy 72  0.551127
          semantic_entropy_proxy    abs_delta_entropy 72 -0.737431
          semantic_entropy_proxy abs_delta_varentropy 72 -0.401943
    semantic_density_uncertainty    abs_delta_entropy 72 -0.783171
    semantic_density_uncertainty abs_delta_varentropy 72 -0.411335
haloscope_style_consistency_risk    abs_delta_entropy 72 -0.753128
haloscope_style_consistency_risk abs_delta_varentropy 72 -0.447489
```

## Files
```text
comparator_scores.csv
comparator_base_metrics.csv
comparator_metric_correlations.csv
comparator_steering_sensitivity.csv
report.md
```
