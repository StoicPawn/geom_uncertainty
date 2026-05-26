# Model Inventory

This repository separates model coverage from experiment logic so the paper can state exactly which architectures support each claim.

| Model | Type | Role in repository | Status |
|---|---|---|---|
| `distilbert-base-uncased` | masked LM | Experiments 1-4, top-k robustness, gradient/regression controls | main |
| `bert-base-uncased` | masked LM | Experiments 1 and 4, top-k robustness, gradient/regression controls, out-of-sample routes, random-init control | main |
| `google/bert_uncased_L-2_H-128_A-2` | small masked LM | Experiments 1 and 4, top-k robustness, gradient/regression controls, full-vocabulary sanity, random-init control | main small-model replication |
| `Qwen/Qwen2.5-0.5B` | decoder-only LM | Decoder logit-lens uncertainty steering inside Experiment 4, comparator diagnostics, local confidence-control application | main decoder evidence |
| `microsoft/phi-2` | decoder-only LM | Decoder logit-lens uncertainty steering inside Experiment 4, comparator diagnostics, local confidence-control application | main decoder replication |
| ResNet-18 CIFAR | vision classifier | Hidden-fragility CIFAR-10/CIFAR-10-C application | 5-epoch CPU pilot completed; full resumable run pending |
| DistilBERT / Tiny BERT MLM heads | local masked-LM logit-lens editors | Safe model-editing diagnostic | completed as local representation edits; not persistent weight editing |
| Llama family | decoder-only LM | Intended stronger decoder replication | not available locally; no network download attempted |
| Mistral family | decoder-only LM | Intended stronger decoder replication | not available locally; no network download attempted |
| RoBERTa family | masked LM | Intended replication | not available locally; no network download attempted |

## Output Lenses

- Masked-LM main experiments use selected top-k MLM-head logits.
- Experiment 4 full battery uses top-k `16`.
- Top-k robustness explicitly evaluates `16, 32, 64, 128, 256`.
- Full-vocabulary sanity uses the tiny model and computes full-vocabulary Fisher-kernel rho without materializing full `F^{1/2}`.
- Decoder-only steering uses selected top-m next-token logits. The main decoder battery uses top-m `16`.
- The CIFAR hidden-fragility protocol uses the full 10-class output distribution, so no top-k approximation is needed.

## Locality

All checked-in runs were executed from local model availability. The repository does not require network access to inspect outputs or regenerate figures.
