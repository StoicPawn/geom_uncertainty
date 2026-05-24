# Uncertainty Steering via Accessible vs Inaccessible Varentropy

## Setup

- Model: `distilbert-base-uncased`
- Layer: `5`
- Output top-k lens: `32`
- Latent subspace: top PCA basis of layer-delta hidden states, dim `16`
- Prompts: `64` factual cloze cases

For each prompt, the run computes `w = F^{1/2}u`, projects it onto `Im(F^{1/2}JB)`, and applies a unit latent perturbation in the accessible least-squares direction. The strict orthogonal least-squares target is also recorded; because it is outside the accessible image, its solution is expected to be near zero. The empirical equal-energy comparison therefore uses an entropy-gradient-orthogonal latent control and a random latent control with the same unit latent norm.

The semantic uncertainty column is an embedding-quantile entropy proxy over the fixed top-k token set, not a full semantic-entropy estimator. Accuracy is measured as exact-token top-1 stability within the same monitored top-k set.

## Prompt Counts
```text
       topic  correct  error
animal_class        0      3
     antonym        0      6
     capital        4      1
      common        2      4
   continent        0      6
     culture        4      2
    currency        2      2
    language        6      0
        math        0      6
      plural        1      5
        role        4      2
     science        3      1
```

## Strict Projection Diagnostics
```text
index  rho_accessible  v_access  v_inaccess  accessible_ls_norm  strict_orth_ls_norm  accessible_target_residual  strict_orth_target_residual
count       64.000000 64.000000   64.000000           64.000000         6.400000e+01                6.400000e+01                    64.000000
 mean        0.778085  0.909440    0.178725           10.155945         1.462339e-14                1.774967e-15                     0.410989
  std        0.127370  0.705102    0.083296            5.654258         9.978394e-15                2.122785e-15                     0.099843
  min        0.316856  0.055021    0.029757            2.274565         3.388597e-15                3.391953e-16                     0.172501
  25%        0.705282  0.362468    0.115598            5.801696         8.364638e-15                7.649466e-16                     0.339997
  50%        0.796038  0.809920    0.173209            8.491467         1.191546e-14                1.254283e-15                     0.416181
  75%        0.869803  1.186942    0.222874           13.677305         1.814548e-14                1.786175e-15                     0.472080
  max        0.973947  3.640977    0.381113           27.553352         6.271849e-14                1.401104e-14                     0.617343
```

## Steering Summary
```text
              direction     sign  epsilon  n  delta_entropy_mean  abs_delta_entropy_mean  delta_varentropy_mean  abs_delta_varentropy_mean  delta_semantic_entropy_proxy_mean  abs_delta_semantic_entropy_proxy_mean  selected_top1_changed_rate  target_correct_changed_rate  target_in_topk_rate  delta_target_prob_topk_mean  first_order_entropy_mean  first_order_logit_norm_mean
          accessible_ls decrease      0.1 64           -0.008731                0.008731           1.494735e-02               1.709370e-02                          -0.002207                               0.002207                    0.000000                     0.000000             0.578125                     0.000411             -8.698914e-02                     0.728631
grad_orthogonal_control decrease      0.1 64           -0.000023                0.000038          -3.079422e-04               1.974672e-03                          -0.000045                               0.000492                    0.000000                     0.000000             0.578125                    -0.000042             -3.069647e-18                     0.833899
         random_control decrease      0.1 64           -0.001292                0.003127           2.657466e-03               7.503755e-03                          -0.000346                               0.001074                    0.000000                     0.000000             0.578125                     0.000011             -1.242158e-02                     0.901655
   strict_orthogonal_ls decrease      0.1 64            0.000000                0.000000          -1.062518e-17               6.960578e-17                           0.000000                               0.000000                    0.000000                     0.000000             0.578125                     0.000000              0.000000e+00                     0.000000
          accessible_ls increase      0.1 64            0.008666                0.008666          -1.489036e-02               1.701320e-02                           0.002187                               0.002187                    0.000000                     0.000000             0.578125                    -0.000415              8.698914e-02                     0.728631
grad_orthogonal_control increase      0.1 64           -0.000023                0.000038           3.313178e-04               1.942923e-03                           0.000044                               0.000490                    0.015625                     0.015625             0.578125                     0.000040              3.069647e-18                     0.833899
         random_control increase      0.1 64            0.001192                0.003106          -2.489539e-03               7.417673e-03                           0.000329                               0.001072                    0.000000                     0.000000             0.578125                    -0.000014              1.242158e-02                     0.901655
   strict_orthogonal_ls increase      0.1 64            0.000000                0.000000          -1.062518e-17               6.960578e-17                           0.000000                               0.000000                    0.000000                     0.000000             0.578125                     0.000000              0.000000e+00                     0.000000
          accessible_ls decrease      0.2 64           -0.017526                0.017526           2.994392e-02               3.425750e-02                          -0.004434                               0.004434                    0.000000                     0.000000             0.578125                     0.000817             -8.698914e-02                     0.728631
grad_orthogonal_control decrease      0.2 64           -0.000091                0.000152          -5.921897e-04               4.007067e-03                          -0.000092                               0.000988                    0.015625                     0.000000             0.578125                    -0.000087             -3.069647e-18                     0.833899
         random_control decrease      0.2 64           -0.002682                0.006283           5.481234e-03               1.510591e-02                          -0.000708                               0.002149                    0.000000                     0.000000             0.578125                     0.000019             -1.242158e-02                     0.901655
   strict_orthogonal_ls decrease      0.2 64            0.000000                0.000000          -1.062518e-17               6.960578e-17                           0.000000                               0.000000                    0.000000                     0.000000             0.578125                     0.000000              0.000000e+00                     0.000000
          accessible_ls increase      0.2 64            0.017264                0.017264          -2.971730e-02               3.393637e-02                           0.004353                               0.004353                    0.000000                     0.000000             0.578125                    -0.000834              8.698914e-02                     0.728631
grad_orthogonal_control increase      0.2 64           -0.000090                0.000153           6.861693e-04               3.887496e-03                           0.000087                               0.000976                    0.015625                     0.015625             0.578125                     0.000077              3.069647e-18                     0.833899
         random_control increase      0.2 64            0.002286                0.006195          -4.810684e-03               1.476888e-02                           0.000642                               0.002142                    0.000000                     0.000000             0.578125                    -0.000032              1.242158e-02                     0.901655
   strict_orthogonal_ls increase      0.2 64            0.000000                0.000000          -1.062518e-17               6.960578e-17                           0.000000                               0.000000                    0.000000                     0.000000             0.578125                     0.000000              0.000000e+00                     0.000000
          accessible_ls decrease      0.4 64           -0.035296                0.035296           6.005905e-02               6.875854e-02                          -0.008947                               0.008947                    0.000000                     0.000000             0.578125                     0.001616             -8.698914e-02                     0.728631
grad_orthogonal_control decrease      0.4 64           -0.000363                0.000606          -1.087277e-03               8.332662e-03                          -0.000189                               0.001986                    0.015625                     0.000000             0.578125                    -0.000183             -3.069647e-18                     0.833899
         random_control decrease      0.4 64           -0.005759                0.012707           1.162262e-02               3.065441e-02                          -0.001484                               0.004309                    0.046875                     0.015625             0.578125                     0.000026             -1.242158e-02                     0.901655
   strict_orthogonal_ls decrease      0.4 64            0.000000                0.000000          -1.062518e-17               6.960578e-17                           0.000000                               0.000000                    0.000000                     0.000000             0.578125                     0.000000              0.000000e+00                     0.000000
          accessible_ls increase      0.4 64            0.034248                0.034248          -5.915495e-02               6.747569e-02                           0.008621                               0.008621                    0.000000                     0.000000             0.578125                    -0.001686              8.698914e-02                     0.728631
grad_orthogonal_control increase      0.4 64           -0.000361                0.000612           1.464167e-03               7.826600e-03                           0.000169                               0.001941                    0.062500                     0.015625             0.578125                     0.000144              3.069647e-18                     0.833899
         random_control increase      0.4 64            0.004173                0.012332          -8.943116e-03               2.929655e-02                           0.001219                               0.004280                    0.046875                     0.000000             0.578125                    -0.000075              1.242158e-02                     0.901655
   strict_orthogonal_ls increase      0.4 64            0.000000                0.000000          -1.062518e-17               6.960578e-17                           0.000000                               0.000000                    0.000000                     0.000000             0.578125                     0.000000              0.000000e+00                     0.000000
```

## Accessible vs Control Contrasts
```text
    sign  epsilon                 control                      metric  accessible  control_value  accessible_minus_control  accessible_over_control
decrease      0.1    strict_orthogonal_ls      abs_delta_entropy_mean    0.008731   0.000000e+00                  0.008731                      inf
decrease      0.1    strict_orthogonal_ls   abs_delta_varentropy_mean    0.017094   6.960578e-17                  0.017094                      inf
decrease      0.1    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.1    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.1 grad_orthogonal_control      abs_delta_entropy_mean    0.008731   3.811339e-05                  0.008693               229.090538
decrease      0.1 grad_orthogonal_control   abs_delta_varentropy_mean    0.017094   1.974672e-03                  0.015119                 8.656476
decrease      0.1 grad_orthogonal_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.1 grad_orthogonal_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.1          random_control      abs_delta_entropy_mean    0.008731   3.127441e-03                  0.005604                 2.791872
decrease      0.1          random_control   abs_delta_varentropy_mean    0.017094   7.503755e-03                  0.009590                 2.278020
decrease      0.1          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.1          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.2    strict_orthogonal_ls      abs_delta_entropy_mean    0.017526   0.000000e+00                  0.017526                      inf
decrease      0.2    strict_orthogonal_ls   abs_delta_varentropy_mean    0.034257   6.960578e-17                  0.034257                      inf
decrease      0.2    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.2    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.2 grad_orthogonal_control      abs_delta_entropy_mean    0.017526   1.520027e-04                  0.017374               115.300590
decrease      0.2 grad_orthogonal_control   abs_delta_varentropy_mean    0.034257   4.007067e-03                  0.030250                 8.549271
decrease      0.2 grad_orthogonal_control  selected_top1_changed_rate    0.000000   1.562500e-02                 -0.015625                 0.000000
decrease      0.2 grad_orthogonal_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.2          random_control      abs_delta_entropy_mean    0.017526   6.282718e-03                  0.011243                 2.789557
decrease      0.2          random_control   abs_delta_varentropy_mean    0.034257   1.510591e-02                  0.019152                 2.267820
decrease      0.2          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.2          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.4    strict_orthogonal_ls      abs_delta_entropy_mean    0.035296   0.000000e+00                  0.035296                      inf
decrease      0.4    strict_orthogonal_ls   abs_delta_varentropy_mean    0.068759   6.960578e-17                  0.068759                      inf
decrease      0.4    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.4    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.4 grad_orthogonal_control      abs_delta_entropy_mean    0.035296   6.058340e-04                  0.034690                58.260161
decrease      0.4 grad_orthogonal_control   abs_delta_varentropy_mean    0.068759   8.332662e-03                  0.060426                 8.251690
decrease      0.4 grad_orthogonal_control  selected_top1_changed_rate    0.000000   1.562500e-02                 -0.015625                 0.000000
decrease      0.4 grad_orthogonal_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease      0.4          random_control      abs_delta_entropy_mean    0.035296   1.270718e-02                  0.022589                 2.777642
decrease      0.4          random_control   abs_delta_varentropy_mean    0.068759   3.065441e-02                  0.038104                 2.243023
decrease      0.4          random_control  selected_top1_changed_rate    0.000000   4.687500e-02                 -0.046875                 0.000000
decrease      0.4          random_control target_correct_changed_rate    0.000000   1.562500e-02                 -0.015625                 0.000000
increase      0.1    strict_orthogonal_ls      abs_delta_entropy_mean    0.008666   0.000000e+00                  0.008666                      inf
increase      0.1    strict_orthogonal_ls   abs_delta_varentropy_mean    0.017013   6.960578e-17                  0.017013                      inf
increase      0.1    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.1    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.1 grad_orthogonal_control      abs_delta_entropy_mean    0.008666   3.819304e-05                  0.008627               226.891147
increase      0.1 grad_orthogonal_control   abs_delta_varentropy_mean    0.017013   1.942923e-03                  0.015070                 8.756495
increase      0.1 grad_orthogonal_control  selected_top1_changed_rate    0.000000   1.562500e-02                 -0.015625                 0.000000
increase      0.1 grad_orthogonal_control target_correct_changed_rate    0.000000   1.562500e-02                 -0.015625                 0.000000
increase      0.1          random_control      abs_delta_entropy_mean    0.008666   3.106208e-03                  0.005559                 2.789789
increase      0.1          random_control   abs_delta_varentropy_mean    0.017013   7.417673e-03                  0.009596                 2.293603
increase      0.1          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.1          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.2    strict_orthogonal_ls      abs_delta_entropy_mean    0.017264   0.000000e+00                  0.017264                      inf
increase      0.2    strict_orthogonal_ls   abs_delta_varentropy_mean    0.033936   6.960578e-17                  0.033936                      inf
increase      0.2    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.2    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.2 grad_orthogonal_control      abs_delta_entropy_mean    0.017264   1.525565e-04                  0.017111               113.163093
increase      0.2 grad_orthogonal_control   abs_delta_varentropy_mean    0.033936   3.887496e-03                  0.030049                 8.729621
increase      0.2 grad_orthogonal_control  selected_top1_changed_rate    0.000000   1.562500e-02                 -0.015625                 0.000000
increase      0.2 grad_orthogonal_control target_correct_changed_rate    0.000000   1.562500e-02                 -0.015625                 0.000000
increase      0.2          random_control      abs_delta_entropy_mean    0.017264   6.194564e-03                  0.011069                 2.786922
increase      0.2          random_control   abs_delta_varentropy_mean    0.033936   1.476888e-02                  0.019167                 2.297829
increase      0.2          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.2          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.4    strict_orthogonal_ls      abs_delta_entropy_mean    0.034248   0.000000e+00                  0.034248                      inf
increase      0.4    strict_orthogonal_ls   abs_delta_varentropy_mean    0.067476   6.960578e-17                  0.067476                      inf
increase      0.4    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.4    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase      0.4 grad_orthogonal_control      abs_delta_entropy_mean    0.034248   6.115853e-04                  0.033637                55.999105
increase      0.4 grad_orthogonal_control   abs_delta_varentropy_mean    0.067476   7.826600e-03                  0.059649                 8.621329
increase      0.4 grad_orthogonal_control  selected_top1_changed_rate    0.000000   6.250000e-02                 -0.062500                 0.000000
increase      0.4 grad_orthogonal_control target_correct_changed_rate    0.000000   1.562500e-02                 -0.015625                 0.000000
increase      0.4          random_control      abs_delta_entropy_mean    0.034248   1.233250e-02                  0.021916                 2.777071
increase      0.4          random_control   abs_delta_varentropy_mean    0.067476   2.929655e-02                  0.038179                 2.303196
increase      0.4          random_control  selected_top1_changed_rate    0.000000   4.687500e-02                 -0.046875                 0.000000
increase      0.4          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
```

## Files
```text
prompts.csv
projection_diagnostics.csv
steering_records.csv
steering_summary.csv
steering_contrasts.csv
report.md
```
