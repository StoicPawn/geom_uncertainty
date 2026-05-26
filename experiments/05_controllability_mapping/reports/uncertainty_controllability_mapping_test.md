# Uncertainty Controllability Mapping Test

## Objective

Map whether high-rho regions/routes correspond to stronger local uncertainty controllability after controlling for scalar uncertainty, gradient and route/model/layer/task features.

```json
{
  "status": "completed",
  "decoder_models": [
    "distilgpt2",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/phi-2"
  ],
  "masked_models_requested": [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "prajjwal1/bert-tiny"
  ],
  "max_items": 18,
  "top_m": 8,
  "layers": "auto3",
  "subspace_dims": "1,2,4",
  "eps": "0.01,0.025,0.05,0.1,0.2",
  "movement_threshold": 0.01,
  "drift_cap": 0.00025,
  "top_k": 3,
  "bootstrap": 300,
  "seed": 20260608,
  "controls": [
    "entropy",
    "confidence",
    "margin",
    "varentropy",
    "projected_gradient_norm",
    "fisher_output_energy",
    "unit_fisher_output_energy",
    "jacobian_norm",
    "layer",
    "model",
    "task",
    "route_family",
    "subspace_dim",
    "budget_mode"
  ],
  "rho_added": "rho"
}
```

## Model Status

```text
                     model            status                                                                                                         reason      layers
   distilbert-base-uncased skipped_not_local We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.         NaN
         bert-base-uncased skipped_not_local We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.         NaN
       prajjwal1/bert-tiny skipped_not_local We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.         NaN
                distilgpt2         completed                                                                                                            NaN   [1, 3, 4]
         Qwen/Qwen2.5-0.5B         completed                                                                                                            NaN [6, 12, 18]
Qwen/Qwen2.5-1.5B-Instruct         completed                                                                                                            NaN [7, 14, 21]
           microsoft/phi-2         completed                                                                                                            NaN [8, 16, 24]
```

## Target Summary

```text
                     model                  task route_family  subspace_dim               budget_mode  n  rho_mean  movement_mean  safe_rate  minimal_energy_median
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    delta_pca             1              same_epsilon 12  0.248181       0.009905   0.416667               0.000603
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    delta_pca             1 same_fisher_output_energy 12  0.248181       0.097463   0.416667               0.002500
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    delta_pca             2              same_epsilon 12  0.708659       0.014513   0.500000               0.000344
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    delta_pca             2 same_fisher_output_energy 12  0.708659       0.112966   0.500000               0.001563
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    delta_pca             4              same_epsilon 12  0.928416       0.013585   0.500000               0.000246
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    delta_pca             4 same_fisher_output_energy 12  0.928416       0.150179   0.666667               0.000625
         Qwen/Qwen2.5-0.5B      ambiguous_prompt       random             1              same_epsilon 12  0.187272       0.003880   0.083333               0.000185
         Qwen/Qwen2.5-0.5B      ambiguous_prompt       random             1 same_fisher_output_energy 12  0.187272       0.082833   0.250000               0.002500
         Qwen/Qwen2.5-0.5B      ambiguous_prompt       random             2              same_epsilon 12  0.651829       0.008214   0.250000               0.000328
         Qwen/Qwen2.5-0.5B      ambiguous_prompt       random             2 same_fisher_output_energy 12  0.651829       0.106110   0.583333               0.000625
         Qwen/Qwen2.5-0.5B      ambiguous_prompt       random             4              same_epsilon 12  0.875324       0.011712   0.416667               0.000388
         Qwen/Qwen2.5-0.5B      ambiguous_prompt       random             4 same_fisher_output_energy 12  0.875324       0.148449   0.750000               0.000625
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    state_pca             1              same_epsilon 12  0.464165       0.012395   0.583333               0.000238
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    state_pca             1 same_fisher_output_energy 12  0.464165       0.153662   0.750000               0.000625
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    state_pca             2              same_epsilon 12  0.711197       0.012622   0.500000               0.000193
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    state_pca             2 same_fisher_output_energy 12  0.711197       0.137270   0.583333               0.000625
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    state_pca             4              same_epsilon 12  0.884486       0.011897   0.500000               0.000350
         Qwen/Qwen2.5-0.5B      ambiguous_prompt    state_pca             4 same_fisher_output_energy 12  0.884486       0.138507   0.666667               0.000625
         Qwen/Qwen2.5-0.5B    decoder_completion    delta_pca             1              same_epsilon  6  0.247647       0.008940   0.166667               0.000278
         Qwen/Qwen2.5-0.5B    decoder_completion    delta_pca             1 same_fisher_output_energy  6  0.247647       0.081046   0.666667               0.000625
         Qwen/Qwen2.5-0.5B    decoder_completion    delta_pca             2              same_epsilon  6  0.371726       0.008707   0.333333               0.000450
         Qwen/Qwen2.5-0.5B    decoder_completion    delta_pca             2 same_fisher_output_energy  6  0.371726       0.101296   0.666667               0.000625
         Qwen/Qwen2.5-0.5B    decoder_completion    delta_pca             4              same_epsilon  6  0.695937       0.011738   0.333333               0.000342
         Qwen/Qwen2.5-0.5B    decoder_completion    delta_pca             4 same_fisher_output_energy  6  0.695937       0.086081   0.500000               0.005313
         Qwen/Qwen2.5-0.5B    decoder_completion       random             1              same_epsilon  6  0.329428       0.012422   0.500000               0.000497
         Qwen/Qwen2.5-0.5B    decoder_completion       random             1 same_fisher_output_energy  6  0.329428       0.089462   0.666667               0.000625
         Qwen/Qwen2.5-0.5B    decoder_completion       random             2              same_epsilon  6  0.614295       0.009378   0.500000               0.000311
         Qwen/Qwen2.5-0.5B    decoder_completion       random             2 same_fisher_output_energy  6  0.614295       0.115628   0.500000               0.000625
         Qwen/Qwen2.5-0.5B    decoder_completion       random             4              same_epsilon  6  0.792166       0.012267   0.333333               0.000264
         Qwen/Qwen2.5-0.5B    decoder_completion       random             4 same_fisher_output_energy  6  0.792166       0.091993   0.500000               0.000625
         Qwen/Qwen2.5-0.5B    decoder_completion    state_pca             1              same_epsilon  6  0.347889       0.018864   0.666667               0.000514
         Qwen/Qwen2.5-0.5B    decoder_completion    state_pca             1 same_fisher_output_energy  6  0.347889       0.100575   0.666667               0.000625
         Qwen/Qwen2.5-0.5B    decoder_completion    state_pca             2              same_epsilon  6  0.727668       0.017374   0.666667               0.000409
         Qwen/Qwen2.5-0.5B    decoder_completion    state_pca             2 same_fisher_output_energy  6  0.727668       0.098671   0.500000               0.000625
         Qwen/Qwen2.5-0.5B    decoder_completion    state_pca             4              same_epsilon  6  0.863725       0.014953   0.500000               0.000392
         Qwen/Qwen2.5-0.5B    decoder_completion    state_pca             4 same_fisher_output_energy  6  0.863725       0.110710   0.666667               0.000625
         Qwen/Qwen2.5-0.5B decoder_qa_completion    delta_pca             1              same_epsilon 12  0.277868       0.006995   0.166667               0.001162
         Qwen/Qwen2.5-0.5B decoder_qa_completion    delta_pca             1 same_fisher_output_energy 12  0.277868       0.087389   0.416667               0.002500
         Qwen/Qwen2.5-0.5B decoder_qa_completion    delta_pca             2              same_epsilon 12  0.666308       0.011068   0.416667               0.000536
         Qwen/Qwen2.5-0.5B decoder_qa_completion    delta_pca             2 same_fisher_output_energy 12  0.666308       0.117401   0.583333               0.000625
         Qwen/Qwen2.5-0.5B decoder_qa_completion    delta_pca             4              same_epsilon 12  0.886587       0.012101   0.500000               0.000382
         Qwen/Qwen2.5-0.5B decoder_qa_completion    delta_pca             4 same_fisher_output_energy 12  0.886587       0.142553   0.833333               0.000625
         Qwen/Qwen2.5-0.5B decoder_qa_completion       random             1              same_epsilon 12  0.438189       0.010070   0.333333               0.000281
         Qwen/Qwen2.5-0.5B decoder_qa_completion       random             1 same_fisher_output_energy 12  0.438189       0.117668   0.666667               0.000625
         Qwen/Qwen2.5-0.5B decoder_qa_completion       random             2              same_epsilon 12  0.678838       0.010201   0.416667               0.000500
         Qwen/Qwen2.5-0.5B decoder_qa_completion       random             2 same_fisher_output_energy 12  0.678838       0.105896   0.583333               0.000625
         Qwen/Qwen2.5-0.5B decoder_qa_completion       random             4              same_epsilon 12  0.868327       0.008113   0.166667               0.000638
         Qwen/Qwen2.5-0.5B decoder_qa_completion       random             4 same_fisher_output_energy 12  0.868327       0.116045   0.583333               0.000625
         Qwen/Qwen2.5-0.5B decoder_qa_completion    state_pca             1              same_epsilon 12  0.479862       0.013104   0.500000               0.000565
         Qwen/Qwen2.5-0.5B decoder_qa_completion    state_pca             1 same_fisher_output_energy 12  0.479862       0.110549   0.666667               0.000625
         Qwen/Qwen2.5-0.5B decoder_qa_completion    state_pca             2              same_epsilon 12  0.646418       0.010261   0.416667               0.000653
         Qwen/Qwen2.5-0.5B decoder_qa_completion    state_pca             2 same_fisher_output_energy 12  0.646418       0.108091   0.500000               0.001563
         Qwen/Qwen2.5-0.5B decoder_qa_completion    state_pca             4              same_epsilon 12  0.930071       0.013471   0.583333               0.000415
         Qwen/Qwen2.5-0.5B decoder_qa_completion    state_pca             4 same_fisher_output_energy 12  0.930071       0.094141   0.750000               0.000625
         Qwen/Qwen2.5-0.5B  factual_error_prompt    delta_pca             1              same_epsilon 12  0.231751       0.003909   0.083333               0.000360
         Qwen/Qwen2.5-0.5B  factual_error_prompt    delta_pca             1 same_fisher_output_energy 12  0.231751       0.252453   0.333333               0.002500
         Qwen/Qwen2.5-0.5B  factual_error_prompt    delta_pca             2              same_epsilon 12  0.688463       0.007501   0.333333               0.000236
         Qwen/Qwen2.5-0.5B  factual_error_prompt    delta_pca             2 same_fisher_output_energy 12  0.688463       0.179434   0.750000               0.000625
         Qwen/Qwen2.5-0.5B  factual_error_prompt    delta_pca             4              same_epsilon 12  0.939276       0.008706   0.416667               0.000374
         Qwen/Qwen2.5-0.5B  factual_error_prompt    delta_pca             4 same_fisher_output_energy 12  0.939276       0.128010   0.916667               0.000625
         Qwen/Qwen2.5-0.5B  factual_error_prompt       random             1              same_epsilon 12  0.431548       0.007672   0.166667               0.000928
         Qwen/Qwen2.5-0.5B  factual_error_prompt       random             1 same_fisher_output_energy 12  0.431548       0.097805   0.666667               0.000625
         Qwen/Qwen2.5-0.5B  factual_error_prompt       random             2              same_epsilon 12  0.599950       0.004410   0.166667               0.000338
         Qwen/Qwen2.5-0.5B  factual_error_prompt       random             2 same_fisher_output_energy 12  0.599950       0.185720   0.583333               0.000625
         Qwen/Qwen2.5-0.5B  factual_error_prompt       random             4              same_epsilon 12  0.938453       0.008839   0.250000               0.000673
         Qwen/Qwen2.5-0.5B  factual_error_prompt       random             4 same_fisher_output_energy 12  0.938453       0.157535   0.583333               0.002500
         Qwen/Qwen2.5-0.5B  factual_error_prompt    state_pca             1              same_epsilon 12  0.575745       0.011518   0.250000               0.000349
         Qwen/Qwen2.5-0.5B  factual_error_prompt    state_pca             1 same_fisher_output_energy 12  0.575745       0.171012   0.833333               0.000625
         Qwen/Qwen2.5-0.5B  factual_error_prompt    state_pca             2              same_epsilon 12  0.818773       0.009597   0.250000               0.000304
         Qwen/Qwen2.5-0.5B  factual_error_prompt    state_pca             2 same_fisher_output_energy 12  0.818773       0.162404   0.666667               0.000625
         Qwen/Qwen2.5-0.5B  factual_error_prompt    state_pca             4              same_epsilon 12  0.922159       0.010976   0.250000               0.000661
         Qwen/Qwen2.5-0.5B  factual_error_prompt    state_pca             4 same_fisher_output_energy 12  0.922159       0.187171   0.750000               0.000625
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    delta_pca             1              same_epsilon 12  0.257155       0.003638   0.083333               0.000724
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    delta_pca             1 same_fisher_output_energy 12  0.257155       0.119559   0.500000               0.000625
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    delta_pca             2              same_epsilon 12  0.553601       0.005299   0.166667               0.000339
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    delta_pca             2 same_fisher_output_energy 12  0.553601       0.150803   0.500000               0.001563
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    delta_pca             4              same_epsilon 12  0.840895       0.005973   0.166667               0.000479
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    delta_pca             4 same_fisher_output_energy 12  0.840895       0.140798   0.500000               0.001563
         Qwen/Qwen2.5-0.5B  masked_factual_cloze       random             1              same_epsilon 12  0.207802       0.003830   0.083333               0.000179
         Qwen/Qwen2.5-0.5B  masked_factual_cloze       random             1 same_fisher_output_energy 12  0.207802       0.101183   0.416667               0.001563
         Qwen/Qwen2.5-0.5B  masked_factual_cloze       random             2              same_epsilon 12  0.683879       0.003272   0.000000                    NaN
         Qwen/Qwen2.5-0.5B  masked_factual_cloze       random             2 same_fisher_output_energy 12  0.683879       0.185877   0.416667               0.002500
         Qwen/Qwen2.5-0.5B  masked_factual_cloze       random             4              same_epsilon 12  0.860476       0.008570   0.083333               0.000133
         Qwen/Qwen2.5-0.5B  masked_factual_cloze       random             4 same_fisher_output_energy 12  0.860476       0.083434   0.333333               0.002500
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    state_pca             1              same_epsilon 12  0.365618       0.006127   0.083333               0.000798
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    state_pca             1 same_fisher_output_energy 12  0.365618       0.163475   0.416667               0.000625
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    state_pca             2              same_epsilon 12  0.571179       0.009017   0.416667               0.000465
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    state_pca             2 same_fisher_output_energy 12  0.571179       0.098447   0.500000               0.000625
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    state_pca             4              same_epsilon 12  0.897977       0.005731   0.166667               0.001497
         Qwen/Qwen2.5-0.5B  masked_factual_cloze    state_pca             4 same_fisher_output_energy 12  0.897977       0.079501   0.583333               0.000625
Qwen/Qwen2.5-1.5B-Instruct      ambiguous_prompt    delta_pca             1              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct      ambiguous_prompt    delta_pca             2              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct      ambiguous_prompt    delta_pca             4              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct      ambiguous_prompt       random             1              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct      ambiguous_prompt       random             2              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct      ambiguous_prompt       random             4              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct      ambiguous_prompt    state_pca             1              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct      ambiguous_prompt    state_pca             2              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct      ambiguous_prompt    state_pca             4              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct    decoder_completion    delta_pca             1              same_epsilon  6  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct    decoder_completion    delta_pca             2              same_epsilon  6  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct    decoder_completion    delta_pca             4              same_epsilon  6  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct    decoder_completion       random             1              same_epsilon  6  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct    decoder_completion       random             2              same_epsilon  6  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct    decoder_completion       random             4              same_epsilon  6  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct    decoder_completion    state_pca             1              same_epsilon  6  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct    decoder_completion    state_pca             2              same_epsilon  6  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct    decoder_completion    state_pca             4              same_epsilon  6  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct decoder_qa_completion    delta_pca             1              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct decoder_qa_completion    delta_pca             2              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct decoder_qa_completion    delta_pca             4              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct decoder_qa_completion       random             1              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct decoder_qa_completion       random             2              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct decoder_qa_completion       random             4              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct decoder_qa_completion    state_pca             1              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct decoder_qa_completion    state_pca             2              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct decoder_qa_completion    state_pca             4              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct  factual_error_prompt    delta_pca             1              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct  factual_error_prompt    delta_pca             2              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
Qwen/Qwen2.5-1.5B-Instruct  factual_error_prompt    delta_pca             4              same_epsilon 12  0.000000       0.000000   0.000000                    NaN
```

## Controls vs Controls+rho

```text
                  target             model    metric    value
max_uncertainty_movement          controls       mae 0.045796
max_uncertainty_movement controls_plus_rho       mae 0.045305
    safe_movement_target          controls     auprc 0.673757
    safe_movement_target          controls     auroc 0.888090
    safe_movement_target          controls  log_loss 0.433265
    safe_movement_target controls_plus_rho     auprc 0.763993
    safe_movement_target controls_plus_rho     auroc 0.919315
    safe_movement_target controls_plus_rho  log_loss 0.372353
          minimal_energy          controls mae_log10 0.391496
          minimal_energy controls_plus_rho mae_log10 0.346423
```

## Bootstrap Delta CI

```text
                  target                     metric   ci_low   median  ci_high
max_uncertainty_movement      delta_mae_improvement 0.000120 0.000497 0.000854
          minimal_energy      delta_mae_improvement 0.034425 0.045135 0.054733
    safe_movement_target                delta_auprc 0.064484 0.088066 0.114117
    safe_movement_target delta_log_loss_improvement 0.043725 0.060430 0.080658
```

## Verdict

Supported on: safe_movement AUPRC, movement MAE, minimal_energy MAE
