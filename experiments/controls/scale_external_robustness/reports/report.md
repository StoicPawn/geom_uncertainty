# Scale Robustness Matrix

This report aggregates the checked-in evidence for broad model coverage, top-k robustness, and controllability mapping.

It is intentionally lightweight: it does not rerun model inference. New RoBERTa, Llama, or Mistral runs are picked up automatically from the standard output CSVs.

## Completed Family Coverage
```text
    family  n_sources  rows                                               models
      BERT          2  3475 bert-base-uncased, google/bert_uncased_L-2_H-128_A-2
DistilBERT          2  2108                              distilbert-base-uncased
      Qwen          2   189        Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B-Instruct
       Phi          2   144                                      microsoft/phi-2
     GPT-2          1    90                                           distilgpt2
```

## Model Status Matrix
```text
                 source                             model     family                            status  rows                                                                            reason
controllability_mapping                 bert-base-uncased       BERT historical_skip_auto_download_now     0 previous checked-in run skipped; current reproduce config downloads automatically
controllability_mapping               prajjwal1/bert-tiny       BERT historical_skip_auto_download_now     0 previous checked-in run skipped; current reproduce config downloads automatically
controllability_mapping           distilbert-base-uncased DistilBERT historical_skip_auto_download_now     0 previous checked-in run skipped; current reproduce config downloads automatically
controllability_mapping                        distilgpt2      GPT-2                         completed    90                                                                               NaN
controllability_mapping           meta-llama/Llama-3.2-1B      Llama             requested_pending_run     0                  listed in reproduce config but not present in checked-in outputs
controllability_mapping         mistralai/Mistral-7B-v0.1    Mistral             requested_pending_run     0                  listed in reproduce config but not present in checked-in outputs
controllability_mapping                   microsoft/phi-2        Phi                         completed    90                                                                               NaN
controllability_mapping                 Qwen/Qwen2.5-0.5B       Qwen                         completed    90                                                                               NaN
controllability_mapping        Qwen/Qwen2.5-1.5B-Instruct       Qwen                         completed    45                                                                               NaN
controllability_mapping                      roberta-base    RoBERTa             requested_pending_run     0                  listed in reproduce config but not present in checked-in outputs
        decoder_battery                        distilgpt2      GPT-2             requested_pending_run     0                  listed in reproduce config but not present in checked-in outputs
        decoder_battery           meta-llama/Llama-3.2-1B      Llama historical_skip_auto_download_now     0 previous checked-in run skipped; current reproduce config downloads automatically
        decoder_battery         mistralai/Mistral-7B-v0.1    Mistral historical_skip_auto_download_now     0 previous checked-in run skipped; current reproduce config downloads automatically
        decoder_battery                   microsoft/phi-2        Phi                         completed    54                                                                               NaN
        decoder_battery                 Qwen/Qwen2.5-0.5B       Qwen                         completed    54                                                                               NaN
        decoder_battery        Qwen/Qwen2.5-1.5B-Instruct       Qwen             requested_pending_run     0                  listed in reproduce config but not present in checked-in outputs
    masked_full_battery                 bert-base-uncased       BERT                         completed  1188                                                                               NaN
    masked_full_battery google/bert_uncased_L-2_H-128_A-2       BERT                         completed   792                                                                               NaN
    masked_full_battery           distilbert-base-uncased DistilBERT                         completed  1188                                                                               NaN
    masked_full_battery                      roberta-base    RoBERTa             requested_pending_run     0                  listed in reproduce config but not present in checked-in outputs
            masked_topk                 bert-base-uncased       BERT                         completed   920                                                                               NaN
            masked_topk google/bert_uncased_L-2_H-128_A-2       BERT                         completed   575                                                                               NaN
            masked_topk           distilbert-base-uncased DistilBERT                         completed   920                                                                               NaN
            masked_topk                      roberta-base    RoBERTa             requested_pending_run     0                  listed in reproduce config but not present in checked-in outputs
```

## Top-k Robustness
```text
                        claim    n   median  minimum
   rho_rank_stability_vs_topk 2415 0.843911 0.766992
layer_trend_stability_vs_topk  105 1.000000 0.500000
```

## Controllability Mapping Bootstrap
```text
                  target                     metric   ci_low   median  ci_high  supported
max_uncertainty_movement      delta_mae_improvement 0.000120 0.000497 0.000854       True
          minimal_energy      delta_mae_improvement 0.034425 0.045135 0.054733       True
    safe_movement_target                delta_auprc 0.064484 0.088066 0.114117       True
    safe_movement_target delta_log_loss_improvement 0.043725 0.060430 0.080658       True
```

## Reproduce
```bash
python scripts/run_scale_external_robustness_matrix.py
```
