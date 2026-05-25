# Limitations

- RoBERTa was not present in the local cache and was not downloaded.
- Decoder-only evidence is now part of Experiment 4 on Qwen and Phi, but still uses local next-token logit-lens interventions rather than end-to-end generation-time interventions.
- Semantic Entropy, Semantic Density, and HaloScope are represented by token-level/proxy diagnostics for object comparison. They are not full implementations of those methods on their original hallucination-detection tasks.
- Top-k Fisher geometry is a local output-lens approximation. The tiny full-vocabulary sanity check suggests larger top-k lenses move closer to full-vocabulary rho, but top-k should not be presented as invariant.
- Direct projected gradients are stronger raw infinitesimal predictors than `rho`; the central claim is geometric decomposition and route accessibility, not predictor dominance.
- The minimal intervention-energy test supports a local cost interpretation on MLM top-k 32 records, but the multi-epsilon grid and decoder slices are more mixed; this should be framed as operational evidence with gradient-baseline caveats.
- The calibration-diagnosis application shows high-rho NLL/Brier/target-probability steerability, but not ECE improvement; the exact-token baseline is low-confidence and low-accuracy.
- The uncertainty-circuit application supports causal route localization, but the high-rho advantage over low-rho/random routes is modest in the first checked-in battery.
- The brittle-confidence application is a mixed/negative boundary case under the current template perturbations, not a positive claim.
- The CIFAR-10/CIFAR-10-C hidden-fragility application is prepared as a cross-domain protocol but was not executed because the local datasets were absent.
- Train-estimated route generalization is preliminary: train routes match held-out oracle routes in this small run, but random routes remain competitive.
- Random-init controls show learned weights change accessibility structure, but random architectures still have nonzero accessibility.
- Most intervention evidence is local logit-lens steering, not end-to-end generation-time intervention.
- External hallucination leaderboards are not included yet.
