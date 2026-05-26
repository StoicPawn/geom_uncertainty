# Model Inventory

This file lists the models relevant to the clean paper branch.

| Model | Type | Role | Status |
|---|---|---|---|
| `distilbert-base-uncased` | masked LM | Experiments 1-4 and controls | main masked-LM evidence |
| `bert-base-uncased` | masked LM | Experiments 1 and 4, top-k robustness, gradient/regression controls | main masked-LM replication |
| `google/bert_uncased_L-2_H-128_A-2` | small masked LM | tiny model, full-vocabulary sanity, random-init control | small-model replication |
| `distilgpt2` | decoder-only LM | decoder steering and controllability mapping | local decoder evidence |
| `Qwen/Qwen2.5-0.5B` | decoder-only LM | decoder steering and controllability mapping | local decoder evidence |
| `Qwen/Qwen2.5-1.5B-Instruct` | decoder-only LM | controllability mapping | local decoder evidence |
| `microsoft/phi-2` | decoder-only LM | decoder steering and controllability mapping | local decoder replication |
| RoBERTa, Llama, Mistral families | masked/decoder LMs | intended future replication | not available locally; no network download attempted |

## Output Lenses

- Masked-LM main experiments use selected top-k MLM-head logits.
- Experiment 4 full battery uses top-k `16`.
- Top-k robustness evaluates `16, 32, 64, 128, 256`.
- Full-vocabulary sanity uses the tiny model and computes full-vocabulary Fisher-kernel rho without materializing full `F^{1/2}`.
- Decoder-only steering uses selected top-m next-token logits.

Exploratory application model records were moved to `archive_applications_exploratory`.
