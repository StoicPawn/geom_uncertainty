# Scaled Confidence Repair And Selective Abstention Summary

## Model Coverage

```text
 model_run   n  accuracy  wrong_high_n  max_wrong_confidence
 qwen_0_5b 216  0.703704             0              0.451736
distilgpt2 216  0.287037           149              0.979478
```

## Qwen/Qwen2.5-0.5B

Qwen produced no true overconfident-wrong cases at the absolute threshold `confidence >= 0.70`.

```text
outcome_group  n  confidence_mean  accuracy
 correct_high 85         0.897703       1.0
  correct_low 36         0.350955       1.0
  correct_mid 31         0.594638       1.0
    wrong_low 63         0.342502       0.0
    wrong_mid  1         0.451736       0.0
```

Calibration improved after the accessible-rho intervention on this model, but this does not validate overconfident-wrong repair because the relevant group is empty.

```text
condition                         accuracy  mean_confidence  overconfident_wrong_rate      nll    brier      ece
before                            0.703704         0.599085                       0.0 0.690242 0.359072 0.112621
accessible_rho_after_eps_median   0.703704         0.712706                       0.0 0.573760 0.314581 0.080437
```

Selective risk did not improve over confidence or margin:

```text
policy                  cov=.25  cov=.50  cov=.75  cov=1.0
confidence              0.0000   0.0000   0.1667   0.2963
margin                  0.0000   0.0000   0.1605   0.2963
rho                     0.1111   0.1389   0.1790   0.2963
accessible_policy       0.0000   0.0185   0.1852   0.2963
```

## distilgpt2

distilgpt2 produced many true overconfident-wrong cases at the absolute threshold.

```text
outcome_group   n  confidence_mean  accuracy
 correct_high  61         0.870319       1.0
  correct_mid   1         0.689113       1.0
   wrong_high 149         0.869671       0.0
    wrong_mid   5         0.646642       0.0
```

On wrong-high cases, accessible-rho routes reduced wrong confidence while preserving the predicted answer and inducing very low output-distribution drift. The effect was smaller than projected gradients.

```text
method              eps   answer_preserved  confidence_delta  wrong_conf_reduced  semantic_drift
accessible_rho      0.40          1.000000         -0.002314            0.721477        0.000024
low_rho             0.40          1.000000         -0.000961            0.677852        0.000006
projected_gradient  0.40          1.000000         -0.033006            1.000000        0.002446
random_activation   0.40          1.000000          0.000017            0.271812        0.000001
```

Calibration worsened after accessible-rho intervention on distilgpt2:

```text
condition                         accuracy  mean_confidence  overconfident_wrong_rate      nll    brier      ece
before                            0.287037         0.863855                  0.689815 2.364256 1.195672 0.576818
accessible_rho_after_eps_median   0.287037         0.993037                  0.712963 4.956855 1.412166 0.706000
```

Selective risk was best for rho at 25 percent coverage, but accessible_policy did not beat confidence overall:

```text
policy                  cov=.25  cov=.50  cov=.75  cov=1.0
confidence              0.7037   0.7037   0.7099   0.7130
margin                  0.7407   0.7222   0.7160   0.7130
rho                     0.5926   0.7037   0.7160   0.7130
accessible_policy       0.7037   0.7407   0.7099   0.7130
```

## Bottom Line

This scaled run does not validate the strong claim.

It supports a narrower observation: high-rho routes can make small, answer-preserving, low-drift confidence adjustments. But in the model with real overconfident-wrong cases, projected gradients were much stronger for reducing wrong confidence, and the accessible-rho repair did not improve calibration or selective abstention policy.

The next required step is a less degenerate benchmark/model pair: an instruction-tuned model that produces true high-confidence errors while still having meaningful answer accuracy. distilgpt2 gives many wrong-high cases but is too poor and too miscalibrated; Qwen 0.5B is accurate enough but does not produce wrong-high cases under the absolute threshold.
