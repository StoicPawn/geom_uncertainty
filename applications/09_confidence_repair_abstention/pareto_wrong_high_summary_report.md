# Wrong-High Confidence Repair Pareto Summary

## Setup

This run tests true overconfident-wrong cases using the global absolute threshold `confidence >= 0.70`.

The Pareto objective is:

- X axis: output drift proxy, lower is better.
- Y axis: reduction of confidence on the originally wrong predicted answer, higher is better.
- Constraint: answer preservation.
- Group: only true `wrong_high` examples.
- Bootstrap: enabled for confidence intervals on the distilgpt2 Pareto run.

Compared methods:

```text
accessible_rho
low_rho
random_activation
standard_delta_pca
projected_gradient
wrong_logit_gradient_normed
temperature_scaling
```

The plotted curve is saved at:

```text
applications/09_confidence_repair_abstention/pareto_wrong_high_distilgpt2.png
```

## Model Coverage

Only distilgpt2 produced true wrong-high cases at the absolute threshold.

```text
model_run                       n   wrong_high_n  max_wrong_confidence
distilgpt2_normal             216            149             0.979478
qwen_0_5b_misleading          216              0             0.574371
qwen_1_5b_instruct_misleading 216              0             0.250000
```

Qwen/Qwen2.5-0.5B with the harder misleading prompt produced errors, but none were high-confidence by the global threshold.

```text
outcome_group   n  confidence_mean  accuracy
 correct_high  49         0.833933       1.0
  correct_low  27         0.393488       1.0
  correct_mid  43         0.583028       1.0
    wrong_low  73         0.375844       0.0
    wrong_mid  24         0.487760       0.0
```

Qwen/Qwen2.5-1.5B-Instruct with the misleading prompt was degenerate for this exact forced-choice scoring setup: the answer distribution was uniform over A/B/C/D.

```text
outcome_group   n  confidence_mean  accuracy
  correct_low  62             0.25       1.0
    wrong_low 154             0.25       0.0
```

Therefore the only valid Pareto comparison over true wrong-high cases is distilgpt2.

## distilgpt2 Pareto Results

All listed distilgpt2 Pareto points had `answer_preserved_rate = 1.0` on the wrong-high group.

```text
method                         eps       drift_mean  confidence_reduction  reduced_rate
accessible_rho               0.025        0.0000007              0.000180        0.3054
low_rho                      0.025        0.0000005              0.000140        0.1913
random_activation            0.025        0.0000005              0.000028        0.1577
standard_delta_pca           0.025        0.0000013              0.000360        0.5772
wrong_logit_gradient_normed  0.025        0.0000010              0.000201        0.4698

accessible_rho               0.100        0.0000022              0.000628        0.5201
low_rho                      0.100        0.0000013              0.000313        0.3792
random_activation            0.100        0.0000009              0.000057        0.3154
standard_delta_pca           0.100        0.0000047              0.001140        0.6443
wrong_logit_gradient_normed  0.100        0.0000037              0.001020        0.6678

accessible_rho               0.400        0.0000240              0.002314        0.7215
low_rho                      0.400        0.0000125              0.001246        0.5604
random_activation            0.400        0.0000041              0.000145        0.4262
standard_delta_pca           0.400        0.0000645              0.004660        0.5671
wrong_logit_gradient_normed  0.400        0.0000451              0.004052        0.9195
projected_gradient           0.400        0.0024458              0.033006        1.0000
temperature_1.25             0.000        0.0034730              0.038933        1.0000
```

Bootstrap 95 percent intervals for representative points:

```text
method                         eps      drift_95ci                  reduction_95ci
accessible_rho               0.025  [0.0000006, 0.0000009]   [0.000120, 0.000250]
accessible_rho               0.400  [0.0000201, 0.0000289]   [0.002053, 0.002662]
standard_delta_pca           0.025  [0.0000011, 0.0000020]   [0.000299, 0.000425]
wrong_logit_gradient_normed  0.025  [0.0000009, 0.0000010]   [0.000153, 0.000263]
wrong_logit_gradient_normed  0.400  [0.0000425, 0.0000480]   [0.003815, 0.004338]
projected_gradient           0.025  [0.0000079, 0.0000093]   [0.001580, 0.001800]
projected_gradient           0.400  [0.0023185, 0.0025951]   [0.031338, 0.035062]
temperature_1.25             0.000  [0.0033834, 0.0035819]   [0.037573, 0.040438]
```

## Interpretation

The strong claim is not validated by this test.

What is supported:

- `accessible_rho` reduces wrong-answer confidence more than random activation and low-rho routes at comparable intervention scales.
- The effect is answer-preserving and extremely low-drift on distilgpt2.
- At very low drift, `accessible_rho` is a real confidence-repair direction, not just noise.

What is not supported:

- `accessible_rho` does not beat standard PCA in this Pareto run.
- `accessible_rho` does not beat normalized wrong-logit gradients at matched or nearby drift.
- `accessible_rho` is far weaker than projected gradients and temperature scaling when more output drift is allowed.
- The multi-model requirement is not satisfied empirically because both Qwen variants produced no true wrong-high cases at the absolute threshold.

Best objective conclusion:

```text
rho identifies low-drift, answer-preserving directions that can reduce wrong confidence better than random and low-rho baselines, but this run does not show that rho beats PCA or normalized gradient directions on the Pareto frontier.
```

The next valid stress test needs a decoder/model-prompt pair that produces real high-confidence errors while retaining non-degenerate forced-choice probabilities. The current Qwen prompts were harder, but they did not create the required wrong-high group.
