# Model Inventory

This file lists the models relevant to the clean paper branch.

| Model | Type | Role | Status |
|---|---|---|---|
| `distilbert-base-uncased` | masked LM | Experiments 1-4 and controls | main masked-LM evidence |
| `bert-base-uncased` | masked LM | Experiments 1 and 4, top-k robustness, gradient/regression controls | main masked-LM replication |
| `roberta-base` | masked LM | steering full battery, top-k/gradient/regression controls, controllability availability audit | included in default test set; downloaded automatically when absent unless `--local-files-only` is set |
| `google/bert_uncased_L-2_H-128_A-2` | small masked LM | tiny model, full-vocabulary sanity, random-init control | small-model replication |
| `distilgpt2` | decoder-only LM | decoder steering and controllability mapping | local decoder evidence |
| `Qwen/Qwen2.5-0.5B` | decoder-only LM | decoder steering and controllability mapping | local decoder evidence |
| `Qwen/Qwen2.5-1.5B-Instruct` | decoder-only LM | controllability mapping | local decoder evidence |
| `microsoft/phi-2` | decoder-only LM | decoder steering and controllability mapping | local decoder replication |
| `meta-llama/Llama-3.2-1B` | decoder-only LM | decoder steering and controllability mapping | included in default test set; downloaded automatically when absent unless `--local-files-only` is set |
| `mistralai/Mistral-7B-v0.1` | decoder-only LM | decoder steering and controllability mapping | included in default test set; downloaded automatically when absent unless `--local-files-only` is set |

## Output Lenses

- Masked-LM main experiments use selected top-k MLM-head logits.
- Experiment 4 full battery uses top-k `16`.
- Top-k robustness evaluates `16, 32, 64, 128, 256`.
- Full-vocabulary sanity uses the tiny model and computes full-vocabulary Fisher-kernel rho without materializing full `F^{1/2}`.
- Decoder-only steering uses selected top-m next-token logits.

Exploratory application model records were moved to `archive_applications_exploratory`.
