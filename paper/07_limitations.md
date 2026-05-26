# Limitations

- RoBERTa was not present in the local cache and was not downloaded.
- Decoder-only evidence uses local next-token logit-lens interventions rather than end-to-end generation-time interventions.
- Semantic Entropy, Semantic Density, and HaloScope are represented by token-level/proxy diagnostics for object comparison, not full implementations on original hallucination-detection tasks.
- Top-k Fisher geometry is a local output-lens approximation. The tiny full-vocabulary sanity check suggests larger top-k lenses move closer to full-vocabulary rho, but top-k should not be presented as invariant.
- Direct projected gradients are stronger raw infinitesimal predictors than `rho`; the central claim is geometric decomposition and route accessibility, not predictor dominance.
- Minimal intervention-energy results are strongest in the MLM top-k local-linear setting; the broader multi-epsilon grid is more mixed.
- The controllability mapping test currently covers local decoder models; requested masked-LM models were not present in the local cache and were skipped without network download.
- Rho should be framed as a local controllability map, not as a generic predictor of model error.
- Most intervention evidence is local logit-lens steering, not persistent model editing or deployment-time generation intervention.
- Exploratory applications are archived outside main and should not be treated as paper claims.
