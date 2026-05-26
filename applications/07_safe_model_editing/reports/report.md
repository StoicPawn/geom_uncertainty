# Safe Model Editing Diagnostic

Local representation-editing diagnostic for whether uncertainty accessibility predicts edit cost and side effects.

## Scope

This is not a persistent ROME/MEMIT-style weight edit. It applies a same-layer hidden-state delta in an internal route and measures the local edit plus side effects on other prompts.

## Interpretation

This first diagnostic does not support the strong claim that uncertainty accessibility alone predicts safe model editing. `rho` is weakly positive for safe-success, but target-specific edit-gradient and Fisher-output norms are stronger predictors of edit cost. The current result should be treated as a boundary case and as a scaffold for a stronger persistent weight-editing experiment.

## Setup

```json
{
  "status": "completed",
  "seed": 20260601,
  "models": [
    "distilbert-base-uncased",
    "google/bert_uncased_L-2_H-128_A-2"
  ],
  "top_k": 32,
  "subspace_k": 8,
  "layers": "auto",
  "max_prompts_per_task": 8,
  "random_subspaces": 1,
  "edit_margin": 0.0,
  "max_latent_norm": 8.0,
  "side_prompts_per_edit": 12,
  "n_prompts": 48,
  "n_edit_records": 240,
  "n_side_effect_records": 2880
}
```

## Edit Summary

```text
                            model subspace_family  n  rho_mean  actual_min_cost_mean  estimated_linear_min_cost_mean  edit_success_rate  safe_success_rate  side_top1_flip_rate_mean  side_kl_mean  side_top10_jaccard_mean
          distilbert-base-uncased       delta_pca 48  0.527547              5.097437                        6.232394           0.750000           0.000000                  0.614583      0.383550                 0.383416
          distilbert-base-uncased          random 48  0.502218              5.308267                        6.525692           0.645833           0.020833                  0.588542      0.328931                 0.399401
          distilbert-base-uncased       state_pca 48  0.540813              4.279729                        4.707798           0.875000           0.000000                  0.602431      0.319822                 0.410560
google/bert_uncased_L-2_H-128_A-2       delta_pca 32  0.496158              4.901525                        4.999275           0.750000           0.000000                  0.585938      0.345605                 0.466963
google/bert_uncased_L-2_H-128_A-2          random 32  0.556199              4.678882                        4.892827           0.875000           0.000000                  0.559896      0.448081                 0.436642
google/bert_uncased_L-2_H-128_A-2       state_pca 32  0.470640              5.093333                        5.468569           0.718750           0.000000                  0.578125      0.453784                 0.454702
```

## Rho Quartiles

```text
rho_quartile  n  rho_mean  actual_min_cost_mean  estimated_linear_min_cost_mean  edit_success_rate  side_top1_flip_rate_mean  side_kl_mean  side_top10_jaccard_mean
      q1_low 60  0.286931              4.788161                        5.304457           0.816667                  0.552778      0.282541                 0.457895
          q2 60  0.437195              5.457659                        6.276781           0.650000                  0.600000      0.388057                 0.390264
          q3 60  0.571453              4.777478                        5.451108           0.833333                  0.620833      0.437930                 0.392921
     q4_high 60  0.773149              4.551044                        5.132719           0.766667                  0.590278      0.382632                 0.438052
```

## Predictor Benchmark

```text
                           outcome                  model                                               predictor        metric     value   n
          negative_actual_min_cost       single_predictor                                                     rho      spearman  0.053557 240
          negative_actual_min_cost       single_predictor                                              confidence      spearman  0.227950 240
          negative_actual_min_cost       single_predictor                                                 entropy      spearman -0.136772 240
          negative_actual_min_cost       single_predictor                                              varentropy      spearman  0.089732 240
          negative_actual_min_cost       single_predictor                                                  margin      spearman -0.098481 240
          negative_actual_min_cost       single_predictor                                          edit_grad_norm      spearman  0.242927 240
          negative_actual_min_cost       single_predictor                                  grad_entropy_proj_norm      spearman  0.202604 240
          negative_actual_min_cost       single_predictor                                  fisher_output_fro_norm      spearman  0.354653 240
          negative_actual_min_cost        scalar_baseline                    confidence,entropy,varentropy,margin            r2  0.122199 240
          negative_actual_min_cost        scalar_baseline                    confidence,entropy,varentropy,margin pred_spearman  0.326837 240
          negative_actual_min_cost edit_gradient_baseline     confidence,entropy,varentropy,margin,edit_grad_norm            r2  0.227506 240
          negative_actual_min_cost edit_gradient_baseline     confidence,entropy,varentropy,margin,edit_grad_norm pred_spearman  0.479131 240
          negative_actual_min_cost      gradient_plus_rho confidence,entropy,varentropy,margin,edit_grad_norm,rho            r2  0.234195 240
          negative_actual_min_cost      gradient_plus_rho confidence,entropy,varentropy,margin,edit_grad_norm,rho pred_spearman  0.488964 240
negative_estimated_linear_min_cost       single_predictor                                                     rho      spearman  0.037321 240
negative_estimated_linear_min_cost       single_predictor                                              confidence      spearman  0.260671 240
negative_estimated_linear_min_cost       single_predictor                                                 entropy      spearman -0.163249 240
negative_estimated_linear_min_cost       single_predictor                                              varentropy      spearman  0.119219 240
negative_estimated_linear_min_cost       single_predictor                                                  margin      spearman -0.107496 240
negative_estimated_linear_min_cost       single_predictor                                          edit_grad_norm      spearman  0.250932 240
negative_estimated_linear_min_cost       single_predictor                                  grad_entropy_proj_norm      spearman  0.208125 240
negative_estimated_linear_min_cost       single_predictor                                  fisher_output_fro_norm      spearman  0.338377 240
negative_estimated_linear_min_cost        scalar_baseline                    confidence,entropy,varentropy,margin            r2  0.123977 240
negative_estimated_linear_min_cost        scalar_baseline                    confidence,entropy,varentropy,margin pred_spearman  0.364407 240
negative_estimated_linear_min_cost edit_gradient_baseline     confidence,entropy,varentropy,margin,edit_grad_norm            r2  0.274602 240
negative_estimated_linear_min_cost edit_gradient_baseline     confidence,entropy,varentropy,margin,edit_grad_norm pred_spearman  0.498708 240
negative_estimated_linear_min_cost      gradient_plus_rho confidence,entropy,varentropy,margin,edit_grad_norm,rho            r2  0.276529 240
negative_estimated_linear_min_cost      gradient_plus_rho confidence,entropy,varentropy,margin,edit_grad_norm,rho pred_spearman  0.501339 240
             negative_side_kl_mean       single_predictor                                                     rho      spearman -0.124082 240
             negative_side_kl_mean       single_predictor                                              confidence      spearman  0.104394 240
```

## Qualitative Examples

```text
                  model  prompt_id                        prompt           task   topic observed_condition  layer subspace_family  subspace_k  rep      rho  v_access  varentropy  entropy  confidence   margin  source_id source_token  edit_id edit_token             edit_target_kind  initial_target_id initial_target_token  initial_correct  rank_whitened_jb  current_logit_gap_source_minus_edit  required_logit_change  edit_grad_norm  estimated_linear_min_cost  actual_min_cost  edit_success  after_top1_id after_top1_token  after_edit_margin  source_prob_before  source_prob_after  edit_prob_before  edit_prob_after  edit_prob_gain  source_prob_drop  candidate_kl  top10_jaccard  grad_entropy_proj_norm  fisher_output_fro_norm  side_n  side_top1_flip_rate  side_target_correct_change_rate  side_kl_mean  side_top10_jaccard_mean  side_abs_entropy_delta_mean  side_target_prob_drop_mean
distilbert-base-uncased         11 The largest planet is [MASK]. factual_simple science            correct      6          random           8    0 0.749904  0.941654      1.2557 2.180608    0.220669 0.082664      13035      jupiter    11691      venus top_candidate_counterfactual              13035              jupiter                1                 8                             0.082664               0.082664        0.461381                   0.179166         0.178737             1          11691            venus           0.000003            0.220669           0.214176          0.203161         0.214177        0.011015          0.006493      0.000834            1.0                0.174595                0.472742    12.0                  0.0                              0.0      0.000626                 0.957071                     0.007769                    0.000733
```
