# Limitations

- RoBERTa was not present in the local cache and was not downloaded.
- Decoder-only evidence is part of Experiment 4 but remains smaller than the masked-LM battery.
- Semantic uncertainty is represented by an embedding proxy, not full generative semantic entropy.
- Top-k Fisher geometry is a local output-lens approximation. The tiny full-vocabulary sanity check suggests larger top-k lenses move closer to full-vocabulary rho, but top-k should not be presented as invariant.
- Direct projected gradients are stronger raw infinitesimal predictors than `rho`; the central claim is geometric decomposition and route accessibility, not predictor dominance.
- Most intervention evidence is local logit-lens steering, not end-to-end generation-time intervention.
- External hallucination benchmarks are not included yet.
