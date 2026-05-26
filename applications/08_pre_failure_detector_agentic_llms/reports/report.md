# Pre-Failure Detector For Agentic LLMs

Diagnostic for whether accessible-varentropy features predict future agentic failure before an action mistake occurs.

## Scope

The tasks are local synthetic multi-step agent-control traces with explicit action choices. This is not a live browser/tool-use benchmark, but it exercises retrieve, clarify, verify, final-answer, and defer decisions with decoder-only LLM logit-lens geometry.

## Setup

```json
{
  "status": "completed",
  "seed": 20260602,
  "models": [
    "Qwen/Qwen2.5-0.5B",
    "microsoft/phi-2"
  ],
  "local_files_only": false,
  "top_m": 6,
  "pca_dim": 4,
  "layers": "auto",
  "max_tasks": 10,
  "self_consistency_prompts": 4,
  "resume": true,
  "step_checkpoints": true,
  "checkpoint_dir": "/home/stoicpawn/projects/geom_uncertainty/applications/08_pre_failure_detector_agentic_llms/outputs/checkpoints",
  "n_agent_tasks": 10,
  "n_agent_steps": 41,
  "n_step_records": 246,
  "n_base_step_records": 41,
  "skipped_models": [
    {
      "model": "microsoft/phi-2",
      "reason": "CUDA out of memory. Tried to allocate 250.00 MiB. GPU 0 has a total capacity of 5.64 GiB of which 98.44 MiB is free. Including non-PyTorch memory, this process has 5.52 GiB memory in use. Of the allocated memory 4.95 GiB is allocated by PyTorch, and 464.02 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)"
    }
  ],
  "action_labels": {
    "A": "reason internally / compute next step",
    "B": "use a tool or retrieve external information",
    "C": "ask the user for clarification",
    "D": "verify or run a check before proceeding",
    "E": "give the final answer or take the final action",
    "F": "stop, defer, or refuse because the action is unsafe"
  }
}
```

## Trajectory Summary

```text
            model                task_id task_type  n_steps  trajectory_success  mean_confidence  mean_entropy  mean_self_consistency  first_failure_step
Qwen/Qwen2.5-0.5B  ambiguous_instruction  planning        3                   0         0.310093      1.624451               0.833333                   0
Qwen/Qwen2.5-0.5B     browser_fact_check   browser        4                   0         0.341211      1.558262               0.625000                   1
Qwen/Qwen2.5-0.5B code_execution_request    coding        4                   0         0.353593      1.548628               0.750000                   0
Qwen/Qwen2.5-0.5B           coding_patch    coding        5                   0         0.342152      1.605531               0.850000                   1
Qwen/Qwen2.5-0.5B         flight_booking  tool_use        5                   0         0.338307      1.578057               0.800000                   0
Qwen/Qwen2.5-0.5B         legal_question    safety        3                   0         0.362724      1.532193               0.666667                   0
Qwen/Qwen2.5-0.5B      math_word_problem  planning        3                   0         0.342253      1.577062               0.750000                   1
Qwen/Qwen2.5-0.5B       research_summary   browser        5                   0         0.387330      1.480350               0.600000                   2
Qwen/Qwen2.5-0.5B          shell_command    coding        4                   0         0.302143      1.629233               0.625000                   0
Qwen/Qwen2.5-0.5B         shopping_agent  tool_use        5                   0         0.312909      1.616400               0.600000                   0
```

## Route Summary

```text
            model subspace_family  layer  n  rho_mean  inaccessible_confidence_mean  future_failure_rate  pre_failure_after_correct_rate  immediate_error_rate  external_support_rate
Qwen/Qwen2.5-0.5B       delta_pca     12 41  0.999218                      0.000263                  1.0                        0.219512              0.780488               0.585366
Qwen/Qwen2.5-0.5B       delta_pca     23 41  0.988924                      0.003565                  1.0                        0.219512              0.780488               0.585366
Qwen/Qwen2.5-0.5B          random     12 41  0.994969                      0.001687                  1.0                        0.219512              0.780488               0.585366
Qwen/Qwen2.5-0.5B          random     23 41  0.986248                      0.004115                  1.0                        0.219512              0.780488               0.585366
Qwen/Qwen2.5-0.5B       state_pca     12 41  0.709590                      0.098778                  1.0                        0.219512              0.780488               0.585366
Qwen/Qwen2.5-0.5B       state_pca     23 41  0.946831                      0.017224                  1.0                        0.219512              0.780488               0.585366
```

## Rho Quartiles

```text
rho_quartile  n  rho_mean  inaccessible_confidence_mean  future_failure_rate  pre_failure_after_correct_rate  immediate_error_rate  external_support_rate
      q1_low 62  0.780270                      0.073678                  1.0                        0.193548              0.806452               0.564516
          q2 61  0.977001                      0.007779                  1.0                        0.245902              0.754098               0.590164
          q3 61  0.995390                      0.001567                  1.0                        0.229508              0.770492               0.639344
     q4_high 62  0.999425                      0.000207                  1.0                        0.209677              0.790323               0.548387
```

## Predictor Benchmark

```text
               outcome                 feature_set   metric     value   n  positive_rate            cv               predictor
     immediate_correct                      scalar    auroc  0.760417 246       0.219512 group_by_task                     NaN
     immediate_correct                      scalar    auprc  0.643033 246       0.219512 group_by_task                     NaN
     immediate_correct          scalar_self_report    auroc  0.701389 246       0.219512 group_by_task                     NaN
     immediate_correct          scalar_self_report    auprc  0.579535 246       0.219512 group_by_task                     NaN
     immediate_correct                    rho_only    auroc  0.682292 246       0.219512 group_by_task                     NaN
     immediate_correct                    rho_only    auprc  0.404384 246       0.219512 group_by_task                     NaN
     immediate_correct scalar_self_report_plus_rho    auroc  0.697724 246       0.219512 group_by_task                     NaN
     immediate_correct scalar_self_report_plus_rho    auprc  0.560476 246       0.219512 group_by_task                     NaN
     immediate_correct               all_geometric    auroc  0.686343 246       0.219512 group_by_task                     NaN
     immediate_correct               all_geometric    auprc  0.538679 246       0.219512 group_by_task                     NaN
     immediate_correct            single_predictor spearman  0.002628 246       0.219512          none                     rho
     immediate_correct            single_predictor spearman -0.002628 246       0.219512          none         inaccessibility
     immediate_correct            single_predictor spearman  0.023510 246       0.219512          none inaccessible_confidence
     immediate_correct            single_predictor spearman  0.388449 246       0.219512          none              confidence
     immediate_correct            single_predictor spearman -0.348608 246       0.219512          none                 entropy
     immediate_correct            single_predictor spearman  0.378489 246       0.219512          none                  margin
     immediate_correct            single_predictor spearman  0.133156 246       0.219512          none        self_consistency
     immediate_correct            single_predictor spearman -0.004980 246       0.219512          none   verbalized_confidence
needs_external_support                      scalar    auroc  0.362745 246       0.585366 group_by_task                     NaN
needs_external_support                      scalar    auprc  0.552070 246       0.585366 group_by_task                     NaN
needs_external_support          scalar_self_report    auroc  0.438725 246       0.585366 group_by_task                     NaN
needs_external_support          scalar_self_report    auprc  0.596006 246       0.585366 group_by_task                     NaN
needs_external_support                    rho_only    auroc  0.429330 246       0.585366 group_by_task                     NaN
needs_external_support                    rho_only    auprc  0.524963 246       0.585366 group_by_task                     NaN
needs_external_support scalar_self_report_plus_rho    auroc  0.437023 246       0.585366 group_by_task                     NaN
needs_external_support scalar_self_report_plus_rho    auprc  0.582348 246       0.585366 group_by_task                     NaN
needs_external_support               all_geometric    auroc  0.458606 246       0.585366 group_by_task                     NaN
needs_external_support               all_geometric    auprc  0.570241 246       0.585366 group_by_task                     NaN
needs_external_support            single_predictor spearman -0.005693 246       0.585366          none                     rho
needs_external_support            single_predictor spearman  0.005693 246       0.585366          none         inaccessibility
needs_external_support            single_predictor spearman  0.008133 246       0.585366          none inaccessible_confidence
needs_external_support            single_predictor spearman  0.079499 246       0.585366          none              confidence
needs_external_support            single_predictor spearman -0.087867 246       0.585366          none                 entropy
needs_external_support            single_predictor spearman  0.079499 246       0.585366          none                  margin
needs_external_support            single_predictor spearman -0.068494 246       0.585366          none        self_consistency
needs_external_support            single_predictor spearman -0.196663 246       0.585366          none   verbalized_confidence
       should_retrieve                      scalar    auroc  0.678030 246       0.195122 group_by_task                     NaN
       should_retrieve                      scalar    auprc  0.482818 246       0.195122 group_by_task                     NaN
       should_retrieve          scalar_self_report    auroc  0.742424 246       0.195122 group_by_task                     NaN
       should_retrieve          scalar_self_report    auprc  0.522496 246       0.195122 group_by_task                     NaN
```

## Policy Need Summary

```text
            model gold_action  n  chosen_correct_rate  confidence_mean  rho_mean  inaccessible_confidence_mean  future_failure_rate
Qwen/Qwen2.5-0.5B           A 42             0.428571         0.347244  0.931347                      0.023091                  1.0
Qwen/Qwen2.5-0.5B           B 48             0.750000         0.368508  0.941864                      0.021314                  1.0
Qwen/Qwen2.5-0.5B           C 30             0.000000         0.315146  0.927362                      0.022383                  1.0
Qwen/Qwen2.5-0.5B           D 60             0.000000         0.338874  0.940190                      0.020265                  1.0
Qwen/Qwen2.5-0.5B           E 60             0.000000         0.330254  0.939769                      0.019786                  1.0
Qwen/Qwen2.5-0.5B           F  6             0.000000         0.290700  0.952092                      0.013927                  1.0
```

## Qualitative Pre-Failure Examples

```text
            model                task_id task_type  step_id                                                                            state gold_action chosen_action  future_failure  pre_failure_after_correct  confidence  entropy  self_consistency  verbalized_confidence      rho  inaccessible_confidence subspace_family  layer                    hazard
Qwen/Qwen2.5-0.5B       research_summary   browser        1                      The abstract is available but the claim depends on a table.           B             B               1                          1    0.417978 1.443742              0.50               0.467537 0.663276                 0.140743       state_pca     12 needs more source content
Qwen/Qwen2.5-0.5B      math_word_problem  planning        0                    A word problem gives quantities and asks for the final total.           A             A               1                          1    0.354456 1.568228              0.75               0.492294 0.611319                 0.137770       state_pca     12               can compute
Qwen/Qwen2.5-0.5B         shopping_agent  tool_use        1                    Budget and use case are known but current prices are unknown.           B             B               1                          1    0.358938 1.554220              0.75               0.472667 0.644210                 0.127706       state_pca     12           needs retrieval
Qwen/Qwen2.5-0.5B       research_summary   browser        0                  The user references a paper by title but no text is in context.           B             B               1                          1    0.390568 1.447391              0.75               0.470697 0.702490                 0.116198       state_pca     12    needs source retrieval
Qwen/Qwen2.5-0.5B code_execution_request    coding        1                       The script only performs pure arithmetic after inspection.           A             A               1                          1    0.400332 1.474724              0.75               0.476383 0.720735                 0.111799       state_pca     12                can reason
Qwen/Qwen2.5-0.5B         flight_booking  tool_use        1             Origin, destination, and date are known. Current prices are unknown.           B             B               1                          1    0.370504 1.533128              1.00               0.471566 0.699744                 0.111246       state_pca     12     needs external search
Qwen/Qwen2.5-0.5B     browser_fact_check   browser        0                                  The user asks for the current CEO of a company.           B             B               1                          1    0.330600 1.580593              0.50               0.487822 0.717104                 0.093525       state_pca     12       time-sensitive fact
Qwen/Qwen2.5-0.5B           coding_patch    coding        0     A test failed with a stack trace, but no source file has been inspected yet.           B             B               1                          1    0.419736 1.488685              1.00               0.494741 0.787103                 0.089361       state_pca     12     needs file inspection
Qwen/Qwen2.5-0.5B  ambiguous_instruction  planning        1 The user clarifies they want a concise rewrite of a paragraph that is available.           A             A               1                          1    0.285029 1.643776              0.75               0.493497 0.688435                 0.088805       state_pca     12       local edit possible
Qwen/Qwen2.5-0.5B  ambiguous_instruction  planning        1 The user clarifies they want a concise rewrite of a paragraph that is available.           A             A               1                          1    0.285029 1.643776              0.75               0.493497 0.847535                 0.043457       state_pca     23       local edit possible
Qwen/Qwen2.5-0.5B      math_word_problem  planning        0                    A word problem gives quantities and asks for the final total.           A             A               1                          1    0.354456 1.568228              0.75               0.492294 0.940894                 0.020951       state_pca     23               can compute
Qwen/Qwen2.5-0.5B  ambiguous_instruction  planning        1 The user clarifies they want a concise rewrite of a paragraph that is available.           A             A               1                          1    0.285029 1.643776              0.75               0.493497 0.942598                 0.016361          random     23       local edit possible
Qwen/Qwen2.5-0.5B     browser_fact_check   browser        0                                  The user asks for the current CEO of a company.           B             B               1                          1    0.330600 1.580593              0.50               0.487822 0.952446                 0.015721       state_pca     23       time-sensitive fact
```
