# Limitations

- RoBERTa, Llama, and Mistral are included in the requested default test sets and are downloaded automatically unless `--local-files-only` is set.
- Decoder-only evidence uses local next-token logit-lens interventions rather than end-to-end generation-time interventions.
- Top-k Fisher geometry is a local output-lens approximation. Top-k robustness is retained on main, but top-k should not be presented as invariant.
- Direct projected gradients are stronger raw infinitesimal predictors than `rho`; the central claim is geometric decomposition and route accessibility, not predictor dominance.
- Minimal intervention-energy results are strongest in the MLM top-k local-linear setting; the broader multi-epsilon grid is more mixed.
- The checked-in controllability mapping results currently cover local decoder models; broader requested model runs depend on download availability and hardware budget.
- Rho should be framed as a local controllability map, not as a generic predictor of model error.
- Most intervention evidence is local logit-lens steering, not persistent model editing or deployment-time generation intervention.
- Exploratory side experiments are archived on `archive_disruptive_experiments_full` and should not be treated as paper claims.
