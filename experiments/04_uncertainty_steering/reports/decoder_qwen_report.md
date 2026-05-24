# Decoder Uncertainty Steering

## Setup

- Model: `Qwen/Qwen2.5-0.5B`
- Local files only: `True`
- Layer: `23`
- Output top-m lens: `24`
- Latent subspace: top PCA basis of forward hidden deltas, dim `16`
- Prompts: `63` causal factual completions
- Torch dtype: `float32`

This is the decoder-only version of the uncertainty steering test. It computes `w = F^{1/2}u` on next-token logit-lens probabilities, projects it onto `Im(F^{1/2}JB)`, and compares equal-latent-energy accessible steering against strict orthogonal, entropy-gradient-orthogonal, and random controls.

The strict orthogonal least-squares target should be nearly zero because it asks the decoder latent map to realize a Fisher-whitened output vector outside its local image. The empirical equal-energy test is therefore the accessible direction versus the gradient-orthogonal and random controls.

Semantic uncertainty is an embedding-quantile entropy proxy over the fixed top-m next-token set. Accuracy is exact-token top-1 stability within that monitored top-m set.

## Prompt Counts
```text
       topic  correct  error
animal_class        3      2
     antonym        0      5
     capital        1      6
      common        2      3
   continent        0      6
     culture        0      4
    currency        2      3
    language        2      3
        math        0      5
      plural        2      3
        role        3      3
     science        3      2
```

## Strict Projection Diagnostics
```text
index  rho_accessible  v_access  v_inaccess  accessible_ls_norm  strict_orth_ls_norm  accessible_target_residual  strict_orth_target_residual
count       63.000000 63.000000   63.000000           63.000000         6.300000e+01                6.300000e+01                    63.000000
 mean        0.900680  1.398374    0.120841           92.771044         1.041708e-13                3.010121e-15                     0.325842
  std        0.083381  0.688842    0.085730           47.482895         7.282733e-14                4.693126e-15                     0.122083
  min        0.577062  0.261969    0.005670           26.688449         2.757754e-14                6.058334e-16                     0.075299
  25%        0.854802  0.866061    0.063590           58.250316         5.948572e-14                1.358751e-15                     0.252169
  50%        0.927991  1.313750    0.097272           79.169304         8.614766e-14                1.822738e-15                     0.311884
  75%        0.965811  1.921674    0.179847          106.497060         1.217928e-13                2.679346e-15                     0.424074
  max        0.993792  2.954250    0.361511          246.741533         3.474968e-13                3.578982e-14                     0.601258
```

## Steering Summary
```text
              direction     sign  epsilon  n  delta_entropy_mean  abs_delta_entropy_mean  delta_varentropy_mean  abs_delta_varentropy_mean  delta_semantic_entropy_proxy_mean  abs_delta_semantic_entropy_proxy_mean  selected_top1_changed_rate  target_correct_changed_rate  target_in_topm_rate  delta_target_prob_topm_mean  first_order_entropy_mean  first_order_logit_norm_mean
          accessible_ls decrease     0.05 63       -8.397303e-04            8.397303e-04           8.040917e-04               1.032034e-03                      -3.151238e-04                           3.151238e-04                         0.0                          0.0             0.873016                 8.197780e-05             -1.680006e-02                     0.233707
grad_orthogonal_control decrease     0.05 63       -3.794128e-08            8.763828e-07           9.586649e-06               3.407739e-04                       1.555728e-05                           1.565444e-04                         0.0                          0.0             0.873016                -2.153279e-05             -9.362000e-19                     0.241000
         random_control decrease     0.05 63       -1.752013e-04            5.757916e-04           1.831686e-04               9.928456e-04                      -1.151008e-04                           3.327442e-04                         0.0                          0.0             0.873016                -1.498750e-05             -3.504203e-03                     0.243218
   strict_orthogonal_ls decrease     0.05 63        2.447595e-09            1.286846e-08          -2.647858e-09               2.562852e-08                      -4.494562e-10                           1.211845e-08                         0.0                          0.0             0.873016                -7.088806e-10              0.000000e+00                     0.000000
          accessible_ls increase     0.05 63        8.403109e-04            8.403109e-04          -8.057193e-04               1.033448e-03                       3.152893e-04                           3.152893e-04                         0.0                          0.0             0.873016                -8.207997e-05              1.680006e-02                     0.233707
grad_orthogonal_control increase     0.05 63       -6.170422e-08            8.696294e-07          -9.783448e-06               3.400423e-04                      -1.540828e-05                           1.565273e-04                         0.0                          0.0             0.873016                 2.139592e-05              9.362000e-19                     0.241000
         random_control increase     0.05 63        1.750968e-04            5.761720e-04          -1.844149e-04               9.925967e-04                       1.150288e-04                           3.329170e-04                         0.0                          0.0             0.873016                 1.482074e-05              3.504203e-03                     0.243218
   strict_orthogonal_ls increase     0.05 63        2.447595e-09            1.286846e-08          -2.647858e-09               2.562852e-08                      -4.494562e-10                           1.211845e-08                         0.0                          0.0             0.873016                -7.088806e-10              0.000000e+00                     0.000000
          accessible_ls decrease     0.10 63       -1.678945e-03            1.678945e-03           1.606410e-03               2.062239e-03                      -6.301164e-04                           6.301164e-04                         0.0                          0.0             0.873016                 1.638056e-04             -1.680006e-02                     0.233707
grad_orthogonal_control decrease     0.10 63       -1.410150e-07            3.008209e-06           1.877954e-05               6.817762e-04                       3.123031e-05                           3.129595e-04                         0.0                          0.0             0.873016                -4.321754e-05             -9.362000e-19                     0.241000
         random_control decrease     0.10 63       -3.511393e-04            1.151656e-03           3.655692e-04               1.985718e-03                      -2.304870e-04                           6.654619e-04                         0.0                          0.0             0.873016                -3.010231e-05             -3.504203e-03                     0.243218
   strict_orthogonal_ls decrease     0.10 63        2.447595e-09            1.286846e-08          -2.647858e-09               2.562852e-08                      -4.494562e-10                           1.211845e-08                         0.0                          0.0             0.873016                -7.088806e-10              0.000000e+00                     0.000000
          accessible_ls increase     0.10 63        1.681085e-03            1.681085e-03          -1.613212e-03               2.068683e-03                       6.307162e-04                           6.307162e-04                         0.0                          0.0             0.873016                -1.642255e-04              1.680006e-02                     0.233707
grad_orthogonal_control increase     0.10 63       -1.330300e-07            2.959281e-06          -1.992560e-05               6.792327e-04                      -3.066760e-05                           3.130974e-04                         0.0                          0.0             0.873016                 4.267435e-05              9.362000e-19                     0.241000
         random_control increase     0.10 63        3.497814e-04            1.152434e-03          -3.699631e-04               1.985131e-03                       2.298754e-04                           6.659666e-04                         0.0                          0.0             0.873016                 2.955270e-05              3.504203e-03                     0.243218
   strict_orthogonal_ls increase     0.10 63        2.447595e-09            1.286846e-08          -2.647858e-09               2.562852e-08                      -4.494562e-10                           1.211845e-08                         0.0                          0.0             0.873016                -7.088806e-10              0.000000e+00                     0.000000
          accessible_ls decrease     0.20 63       -3.355887e-03            3.355887e-03           3.206206e-03               4.118332e-03                      -1.259705e-03                           1.259705e-03                         0.0                          0.0             0.873016                 3.273124e-04             -1.680006e-02                     0.233707
grad_orthogonal_control decrease     0.20 63       -7.335481e-07            1.166978e-05           3.644206e-05               1.365241e-03                       6.297508e-05                           6.258085e-04                         0.0                          0.0             0.873016                -8.686700e-05             -9.362000e-19                     0.241000
         random_control decrease     0.20 63       -7.036853e-04            2.302853e-03           7.267772e-04               3.971827e-03                      -4.616508e-04                           1.330440e-03                         0.0                          0.0             0.873016                -6.059997e-05             -3.504203e-03                     0.243218
   strict_orthogonal_ls decrease     0.20 63        2.447595e-09            1.286846e-08          -2.647858e-09               2.562852e-08                      -4.494562e-10                           1.211845e-08                         0.0                          0.0             0.873016                -7.088806e-10              0.000000e+00                     0.000000
          accessible_ls increase     0.20 63        3.364083e-03            3.364083e-03          -3.232507e-03               4.143121e-03                       1.261961e-03                           1.261961e-03                         0.0                          0.0             0.873016                -3.287703e-04              1.680006e-02                     0.233707
grad_orthogonal_control increase     0.20 63       -8.840040e-07            1.154696e-05          -4.060733e-05               1.356877e-03                      -6.085601e-05                           6.262228e-04                         0.0                          0.0             0.873016                 8.484640e-05              9.362000e-19                     0.241000
         random_control increase     0.20 63        6.979528e-04            2.305482e-03          -7.435597e-04               3.969638e-03                       4.590534e-04                           1.332303e-03                         0.0                          0.0             0.873016                 5.869561e-05              3.504203e-03                     0.243218
   strict_orthogonal_ls increase     0.20 63        2.447595e-09            1.286846e-08          -2.647858e-09               2.562852e-08                      -4.494562e-10                           1.211845e-08                         0.0                          0.0             0.873016                -7.088806e-10              0.000000e+00                     0.000000
```

## Accessible vs Control Contrasts
```text
    sign  epsilon                 control                      metric  accessible  control_value  accessible_minus_control  accessible_over_control
decrease     0.05    strict_orthogonal_ls      abs_delta_entropy_mean    0.000840   1.286846e-08                  0.000840             6.525494e+04
decrease     0.05    strict_orthogonal_ls   abs_delta_varentropy_mean    0.001032   2.562852e-08                  0.001032             4.026898e+04
decrease     0.05    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.05    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.05 grad_orthogonal_control      abs_delta_entropy_mean    0.000840   8.763828e-07                  0.000839             9.581775e+02
decrease     0.05 grad_orthogonal_control   abs_delta_varentropy_mean    0.001032   3.407739e-04                  0.000691             3.028502e+00
decrease     0.05 grad_orthogonal_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.05 grad_orthogonal_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.05          random_control      abs_delta_entropy_mean    0.000840   5.757916e-04                  0.000264             1.458393e+00
decrease     0.05          random_control   abs_delta_varentropy_mean    0.001032   9.928456e-04                  0.000039             1.039471e+00
decrease     0.05          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.05          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.10    strict_orthogonal_ls      abs_delta_entropy_mean    0.001679   1.286846e-08                  0.001679             1.304698e+05
decrease     0.10    strict_orthogonal_ls   abs_delta_varentropy_mean    0.002062   2.562852e-08                  0.002062             8.046659e+04
decrease     0.10    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.10    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.10 grad_orthogonal_control      abs_delta_entropy_mean    0.001679   3.008209e-06                  0.001676             5.581211e+02
decrease     0.10 grad_orthogonal_control   abs_delta_varentropy_mean    0.002062   6.817762e-04                  0.001380             3.024804e+00
decrease     0.10 grad_orthogonal_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.10 grad_orthogonal_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.10          random_control      abs_delta_entropy_mean    0.001679   1.151656e-03                  0.000527             1.457852e+00
decrease     0.10          random_control   abs_delta_varentropy_mean    0.002062   1.985718e-03                  0.000077             1.038536e+00
decrease     0.10          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.10          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.20    strict_orthogonal_ls      abs_delta_entropy_mean    0.003356   1.286846e-08                  0.003356             2.607839e+05
decrease     0.20    strict_orthogonal_ls   abs_delta_varentropy_mean    0.004118   2.562852e-08                  0.004118             1.606933e+05
decrease     0.20    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.20    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.20 grad_orthogonal_control      abs_delta_entropy_mean    0.003356   1.166978e-05                  0.003344             2.875707e+02
decrease     0.20 grad_orthogonal_control   abs_delta_varentropy_mean    0.004118   1.365241e-03                  0.002753             3.016560e+00
decrease     0.20 grad_orthogonal_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.20 grad_orthogonal_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.20          random_control      abs_delta_entropy_mean    0.003356   2.302853e-03                  0.001053             1.457273e+00
decrease     0.20          random_control   abs_delta_varentropy_mean    0.004118   3.971827e-03                  0.000147             1.036886e+00
decrease     0.20          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
decrease     0.20          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.05    strict_orthogonal_ls      abs_delta_entropy_mean    0.000840   1.286846e-08                  0.000840             6.530005e+04
increase     0.05    strict_orthogonal_ls   abs_delta_varentropy_mean    0.001033   2.562852e-08                  0.001033             4.032415e+04
increase     0.05    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.05    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.05 grad_orthogonal_control      abs_delta_entropy_mean    0.000840   8.696294e-07                  0.000839             9.662861e+02
increase     0.05 grad_orthogonal_control   abs_delta_varentropy_mean    0.001033   3.400423e-04                  0.000693             3.039176e+00
increase     0.05 grad_orthogonal_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.05 grad_orthogonal_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.05          random_control      abs_delta_entropy_mean    0.000840   5.761720e-04                  0.000264             1.458438e+00
increase     0.05          random_control   abs_delta_varentropy_mean    0.001033   9.925967e-04                  0.000041             1.041156e+00
increase     0.05          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.05          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.10    strict_orthogonal_ls      abs_delta_entropy_mean    0.001681   1.286846e-08                  0.001681             1.306361e+05
increase     0.10    strict_orthogonal_ls   abs_delta_varentropy_mean    0.002069   2.562852e-08                  0.002069             8.071802e+04
increase     0.10    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.10    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.10 grad_orthogonal_control      abs_delta_entropy_mean    0.001681   2.959281e-06                  0.001678             5.680722e+02
increase     0.10 grad_orthogonal_control   abs_delta_varentropy_mean    0.002069   6.792327e-04                  0.001389             3.045618e+00
increase     0.10 grad_orthogonal_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.10 grad_orthogonal_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.10          random_control      abs_delta_entropy_mean    0.001681   1.152434e-03                  0.000529             1.458726e+00
increase     0.10          random_control   abs_delta_varentropy_mean    0.002069   1.985131e-03                  0.000084             1.042089e+00
increase     0.10          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.10          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.20    strict_orthogonal_ls      abs_delta_entropy_mean    0.003364   1.286846e-08                  0.003364             2.614208e+05
increase     0.20    strict_orthogonal_ls   abs_delta_varentropy_mean    0.004143   2.562852e-08                  0.004143             1.616606e+05
increase     0.20    strict_orthogonal_ls  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.20    strict_orthogonal_ls target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.20 grad_orthogonal_control      abs_delta_entropy_mean    0.003364   1.154696e-05                  0.003353             2.913392e+02
increase     0.20 grad_orthogonal_control   abs_delta_varentropy_mean    0.004143   1.356877e-03                  0.002786             3.053424e+00
increase     0.20 grad_orthogonal_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.20 grad_orthogonal_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.20          random_control      abs_delta_entropy_mean    0.003364   2.305482e-03                  0.001059             1.459167e+00
increase     0.20          random_control   abs_delta_varentropy_mean    0.004143   3.969638e-03                  0.000173             1.043703e+00
increase     0.20          random_control  selected_top1_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
increase     0.20          random_control target_correct_changed_rate    0.000000   0.000000e+00                  0.000000                      inf
```

## Files
```text
prompt_features.csv
projection_diagnostics.csv
steering_records.csv
steering_summary.csv
steering_contrasts.csv
report.md
```
