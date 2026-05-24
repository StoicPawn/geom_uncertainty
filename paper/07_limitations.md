# Limitations

- RoBERTa was not present in the local cache and was not downloaded.
- Decoder-only evidence is included as a separate Qwen logit-lens steering run, not yet merged into the full MLM battery.
- Semantic uncertainty is represented by an embedding proxy, not full generative semantic entropy.
- Most intervention evidence is local logit-lens steering, not end-to-end generation-time intervention.
- External hallucination benchmarks are not included yet.
