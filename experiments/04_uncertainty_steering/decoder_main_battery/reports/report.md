# Decoder-Only LLM Main Battery

This is the main decoder-only extension of the accessible-varentropy steering protocol. It uses token-level QA, factual completion, and short generative-open prompts with next-token logit-lens geometry.

Qwen and Phi are executed from the local cache. Llama and Mistral are recorded as skipped when absent from the local Hugging Face cache; no network download is attempted.

## Prompt Counts
```text
            model               task  size
Qwen/Qwen2.5-0.5B factual_completion     6
Qwen/Qwen2.5-0.5B    generative_open     6
Qwen/Qwen2.5-0.5B     token_level_qa     6
  microsoft/phi-2 factual_completion     6
  microsoft/phi-2    generative_open     6
  microsoft/phi-2     token_level_qa     6
```

## Skipped Models
```text
                    model              reason
  meta-llama/Llama-3.2-1B missing local cache
mistralai/Mistral-7B-v0.1 missing local cache
```

## Accessibility Summary
```text
            model subspace_family  layer  n  rho_mean  rho_std  v_access_mean  grad_entropy_proj_norm_mean
Qwen/Qwen2.5-0.5B       delta_pca     12  9  0.660159 0.213701       0.604035                     0.183370
Qwen/Qwen2.5-0.5B       delta_pca     23  9  0.784450 0.153465       1.102944                     0.043128
Qwen/Qwen2.5-0.5B          random     12  9  0.749980 0.145763       0.631813                     0.184833
Qwen/Qwen2.5-0.5B          random     23  9  0.788789 0.101908       1.080109                     0.042631
Qwen/Qwen2.5-0.5B       state_pca     12  9  0.759758 0.134360       0.666404                     0.194085
Qwen/Qwen2.5-0.5B       state_pca     23  9  0.774197 0.136364       1.062397                     0.053732
  microsoft/phi-2       delta_pca     16  9  0.762801 0.171853       0.757488                     0.032422
  microsoft/phi-2       delta_pca     31  9  0.905187 0.087387       0.862686                     0.012504
  microsoft/phi-2          random     16  9  0.699665 0.092091       0.659550                     0.022658
  microsoft/phi-2          random     31  9  0.882564 0.113116       0.852482                     0.008703
  microsoft/phi-2       state_pca     16  9  0.817387 0.108107       0.783552                     0.040690
  microsoft/phi-2       state_pca     31  9  0.891431 0.098683       0.845431                     0.012634
```

## Steering Summary
```text
            model subspace_family               direction     sign  n  abs_delta_entropy_mean  abs_delta_varentropy_mean  selected_top1_changed_rate  top10_jaccard_mean  delta_confidence_mean
Qwen/Qwen2.5-0.5B       delta_pca           accessible_ls decrease 18                0.042844                   0.060401                    0.000000            0.939394               0.014187
Qwen/Qwen2.5-0.5B       delta_pca           accessible_ls increase 18                0.043608                   0.062974                    0.055556            0.929293              -0.014714
Qwen/Qwen2.5-0.5B       delta_pca grad_orthogonal_control decrease 18                0.001557                   0.011827                    0.111111            0.919192              -0.001986
Qwen/Qwen2.5-0.5B       delta_pca grad_orthogonal_control increase 18                0.001511                   0.010422                    0.000000            0.909091               0.001043
Qwen/Qwen2.5-0.5B       delta_pca          random_control decrease 18                0.019512                   0.030341                    0.055556            0.861953              -0.001519
Qwen/Qwen2.5-0.5B       delta_pca          random_control increase 18                0.020025                   0.029730                    0.000000            0.959596               0.000685
Qwen/Qwen2.5-0.5B          random           accessible_ls decrease 18                0.043679                   0.058858                    0.000000            0.949495               0.013575
Qwen/Qwen2.5-0.5B          random           accessible_ls increase 18                0.044473                   0.061983                    0.055556            0.919192              -0.014125
Qwen/Qwen2.5-0.5B          random grad_orthogonal_control decrease 18                0.001259                   0.010454                    0.000000            0.959596               0.001801
Qwen/Qwen2.5-0.5B          random grad_orthogonal_control increase 18                0.001269                   0.011700                    0.055556            0.900673              -0.002712
Qwen/Qwen2.5-0.5B          random          random_control decrease 18                0.021129                   0.029884                    0.055556            0.949495               0.001215
Qwen/Qwen2.5-0.5B          random          random_control increase 18                0.022037                   0.031133                    0.000000            0.888889              -0.001838
Qwen/Qwen2.5-0.5B       state_pca           accessible_ls decrease 18                0.043528                   0.058649                    0.000000            0.949495               0.013139
Qwen/Qwen2.5-0.5B       state_pca           accessible_ls increase 18                0.044987                   0.062512                    0.055556            0.929293              -0.013899
Qwen/Qwen2.5-0.5B       state_pca grad_orthogonal_control decrease 18                0.001306                   0.013165                    0.055556            0.880471               0.000655
Qwen/Qwen2.5-0.5B       state_pca grad_orthogonal_control increase 18                0.001362                   0.013035                    0.000000            0.919192              -0.001623
Qwen/Qwen2.5-0.5B       state_pca          random_control decrease 18                0.017353                   0.028813                    0.111111            0.919192               0.002944
Qwen/Qwen2.5-0.5B       state_pca          random_control increase 18                0.017459                   0.030125                    0.000000            0.909091              -0.003663
  microsoft/phi-2       delta_pca           accessible_ls decrease 18                0.040475                   0.063576                    0.000000            0.939394               0.012761
  microsoft/phi-2       delta_pca           accessible_ls increase 18                0.046956                   0.078728                    0.000000            0.979798              -0.014359
  microsoft/phi-2       delta_pca grad_orthogonal_control decrease 18                0.010757                   0.033396                    0.000000            0.959596              -0.001991
  microsoft/phi-2       delta_pca grad_orthogonal_control increase 18                0.006807                   0.035600                    0.055556            0.941077              -0.002113
  microsoft/phi-2       delta_pca          random_control decrease 18                0.020878                   0.047515                    0.000000            0.920875              -0.002391
  microsoft/phi-2       delta_pca          random_control increase 18                0.022707                   0.041232                    0.000000            0.979798               0.000563
  microsoft/phi-2          random           accessible_ls decrease 18                0.036935                   0.055512                    0.000000            0.930976               0.011779
  microsoft/phi-2          random           accessible_ls increase 18                0.048868                   0.084642                    0.000000            0.969697              -0.015036
  microsoft/phi-2          random grad_orthogonal_control decrease 18                0.008092                   0.040984                    0.000000            0.949495              -0.000003
  microsoft/phi-2          random grad_orthogonal_control increase 18                0.007464                   0.026075                    0.000000            0.942761              -0.003698
  microsoft/phi-2          random          random_control decrease 18                0.015394                   0.029177                    0.055556            0.929293              -0.000358
  microsoft/phi-2          random          random_control increase 18                0.018186                   0.039542                    0.000000            0.951178              -0.002073
  microsoft/phi-2       state_pca           accessible_ls decrease 18                0.041414                   0.066878                    0.000000            0.959596               0.013530
  microsoft/phi-2       state_pca           accessible_ls increase 18                0.046790                   0.079132                    0.000000            0.979798              -0.014799
  microsoft/phi-2       state_pca grad_orthogonal_control decrease 18                0.005059                   0.019277                    0.055556            0.949495              -0.004006
  microsoft/phi-2       state_pca grad_orthogonal_control increase 18                0.004980                   0.024641                    0.000000            0.922559               0.001397
  microsoft/phi-2       state_pca          random_control decrease 18                0.024703                   0.052247                    0.055556            0.939394              -0.001419
  microsoft/phi-2       state_pca          random_control increase 18                0.024162                   0.049384                    0.000000            0.961279              -0.000326
```

## Accessible Vs Controls
```text
            model subspace_family     sign                 control                    metric  accessible  control_value  accessible_minus_control  accessible_over_control
Qwen/Qwen2.5-0.5B       delta_pca decrease          random_control    abs_delta_entropy_mean    0.042844       0.019512                  0.023332                 2.195728
Qwen/Qwen2.5-0.5B       delta_pca decrease          random_control abs_delta_varentropy_mean    0.060401       0.030341                  0.030060                 1.990746
Qwen/Qwen2.5-0.5B       delta_pca decrease grad_orthogonal_control    abs_delta_entropy_mean    0.042844       0.001557                  0.041287                27.514174
Qwen/Qwen2.5-0.5B       delta_pca decrease grad_orthogonal_control abs_delta_varentropy_mean    0.060401       0.011827                  0.048574                 5.107015
Qwen/Qwen2.5-0.5B       delta_pca increase          random_control    abs_delta_entropy_mean    0.043608       0.020025                  0.023583                 2.177716
Qwen/Qwen2.5-0.5B       delta_pca increase          random_control abs_delta_varentropy_mean    0.062974       0.029730                  0.033244                 2.118208
Qwen/Qwen2.5-0.5B       delta_pca increase grad_orthogonal_control    abs_delta_entropy_mean    0.043608       0.001511                  0.042097                28.862389
Qwen/Qwen2.5-0.5B       delta_pca increase grad_orthogonal_control abs_delta_varentropy_mean    0.062974       0.010422                  0.052552                 6.042338
Qwen/Qwen2.5-0.5B          random decrease          random_control    abs_delta_entropy_mean    0.043679       0.021129                  0.022550                 2.067266
Qwen/Qwen2.5-0.5B          random decrease          random_control abs_delta_varentropy_mean    0.058858       0.029884                  0.028974                 1.969551
Qwen/Qwen2.5-0.5B          random decrease grad_orthogonal_control    abs_delta_entropy_mean    0.043679       0.001259                  0.042420                34.692714
Qwen/Qwen2.5-0.5B          random decrease grad_orthogonal_control abs_delta_varentropy_mean    0.058858       0.010454                  0.048404                 5.630109
Qwen/Qwen2.5-0.5B          random increase          random_control    abs_delta_entropy_mean    0.044473       0.022037                  0.022436                 2.018115
Qwen/Qwen2.5-0.5B          random increase          random_control abs_delta_varentropy_mean    0.061983       0.031133                  0.030850                 1.990916
Qwen/Qwen2.5-0.5B          random increase grad_orthogonal_control    abs_delta_entropy_mean    0.044473       0.001269                  0.043204                35.038746
Qwen/Qwen2.5-0.5B          random increase grad_orthogonal_control abs_delta_varentropy_mean    0.061983       0.011700                  0.050282                 5.297468
Qwen/Qwen2.5-0.5B       state_pca decrease          random_control    abs_delta_entropy_mean    0.043528       0.017353                  0.026175                 2.508371
Qwen/Qwen2.5-0.5B       state_pca decrease          random_control abs_delta_varentropy_mean    0.058649       0.028813                  0.029836                 2.035520
Qwen/Qwen2.5-0.5B       state_pca decrease grad_orthogonal_control    abs_delta_entropy_mean    0.043528       0.001306                  0.042223                33.331052
Qwen/Qwen2.5-0.5B       state_pca decrease grad_orthogonal_control abs_delta_varentropy_mean    0.058649       0.013165                  0.045485                 4.455113
Qwen/Qwen2.5-0.5B       state_pca increase          random_control    abs_delta_entropy_mean    0.044987       0.017459                  0.027528                 2.576675
Qwen/Qwen2.5-0.5B       state_pca increase          random_control abs_delta_varentropy_mean    0.062512       0.030125                  0.032387                 2.075077
Qwen/Qwen2.5-0.5B       state_pca increase grad_orthogonal_control    abs_delta_entropy_mean    0.044987       0.001362                  0.043625                33.024538
Qwen/Qwen2.5-0.5B       state_pca increase grad_orthogonal_control abs_delta_varentropy_mean    0.062512       0.013035                  0.049477                 4.795790
  microsoft/phi-2       delta_pca decrease          random_control    abs_delta_entropy_mean    0.040475       0.020878                  0.019597                 1.938645
  microsoft/phi-2       delta_pca decrease          random_control abs_delta_varentropy_mean    0.063576       0.047515                  0.016061                 1.338029
  microsoft/phi-2       delta_pca decrease grad_orthogonal_control    abs_delta_entropy_mean    0.040475       0.010757                  0.029717                 3.762539
  microsoft/phi-2       delta_pca decrease grad_orthogonal_control abs_delta_varentropy_mean    0.063576       0.033396                  0.030180                 1.903702
  microsoft/phi-2       delta_pca increase          random_control    abs_delta_entropy_mean    0.046956       0.022707                  0.024249                 2.067927
  microsoft/phi-2       delta_pca increase          random_control abs_delta_varentropy_mean    0.078728       0.041232                  0.037496                 1.909403
  microsoft/phi-2       delta_pca increase grad_orthogonal_control    abs_delta_entropy_mean    0.046956       0.006807                  0.040149                 6.898510
  microsoft/phi-2       delta_pca increase grad_orthogonal_control abs_delta_varentropy_mean    0.078728       0.035600                  0.043128                 2.211449
  microsoft/phi-2          random decrease          random_control    abs_delta_entropy_mean    0.036935       0.015394                  0.021541                 2.399313
  microsoft/phi-2          random decrease          random_control abs_delta_varentropy_mean    0.055512       0.029177                  0.026335                 1.902574
  microsoft/phi-2          random decrease grad_orthogonal_control    abs_delta_entropy_mean    0.036935       0.008092                  0.028843                 4.564365
  microsoft/phi-2          random decrease grad_orthogonal_control abs_delta_varentropy_mean    0.055512       0.040984                  0.014528                 1.354489
  microsoft/phi-2          random increase          random_control    abs_delta_entropy_mean    0.048868       0.018186                  0.030682                 2.687173
  microsoft/phi-2          random increase          random_control abs_delta_varentropy_mean    0.084642       0.039542                  0.045100                 2.140580
  microsoft/phi-2          random increase grad_orthogonal_control    abs_delta_entropy_mean    0.048868       0.007464                  0.041404                 6.547250
  microsoft/phi-2          random increase grad_orthogonal_control abs_delta_varentropy_mean    0.084642       0.026075                  0.058567                 3.246145
  microsoft/phi-2       state_pca decrease          random_control    abs_delta_entropy_mean    0.041414       0.024703                  0.016711                 1.676483
  microsoft/phi-2       state_pca decrease          random_control abs_delta_varentropy_mean    0.066878       0.052247                  0.014632                 1.280049
  microsoft/phi-2       state_pca decrease grad_orthogonal_control    abs_delta_entropy_mean    0.041414       0.005059                  0.036355                 8.186190
  microsoft/phi-2       state_pca decrease grad_orthogonal_control abs_delta_varentropy_mean    0.066878       0.019277                  0.047602                 3.469406
  microsoft/phi-2       state_pca increase          random_control    abs_delta_entropy_mean    0.046790       0.024162                  0.022628                 1.936526
  microsoft/phi-2       state_pca increase          random_control abs_delta_varentropy_mean    0.079132       0.049384                  0.029748                 1.602387
  microsoft/phi-2       state_pca increase grad_orthogonal_control    abs_delta_entropy_mean    0.046790       0.004980                  0.041810                 9.395528
  microsoft/phi-2       state_pca increase grad_orthogonal_control abs_delta_varentropy_mean    0.079132       0.024641                  0.054491                 3.211340
```

## Files
```text
prompt_table.csv
decoder_scores.csv
decoder_steering_records.csv
decoder_score_summary.csv
decoder_steering_summary.csv
decoder_control_contrasts.csv
skipped_models.csv
report.md
```
