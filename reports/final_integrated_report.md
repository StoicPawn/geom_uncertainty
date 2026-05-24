# Final Integrated Report

This report is the single paper-facing synthesis for the organized repository. The underlying artifacts are kept in the experiment folders, with controls separated under `experiments/controls/`.

## Mathematical Framework

Accessible varentropy decomposes uncertainty into a Fisher-whitened component reachable by a latent subspace and an orthogonal component that is locally inaccessible:

```text
Var = V_access + V_inaccess,
rho(B) = V_access / Var.
```

## Experiment 1: Scalar Uncertainty Does Not Determine Accessibility

Matched scalar-uncertainty pairs in `experiments/01_matched_scalar_uncertainty/outputs/` show that entropy and varentropy do not determine accessibility.

Main artifacts:

- `same_uncertainty_different_rho_pairs.csv`
- `same_uncertainty_different_accessibility_pairs.csv`
- `matched_error_correct_summary.csv`

## Experiment 2: Accessibility Predicts Local Uncertainty Movement

The local perturbation outputs in `experiments/02_local_perturbation_prediction/` test whether accessibility predicts local `Delta H` and `Delta Var` after scalar controls.

Main artifacts:

- `intervention_summary.csv`
- `intervention_by_rho_quartile.csv`
- `intervention_residual_correlations.csv`
- `intervention_records.csv`

## Experiment 3: Layerwise And K-Dimensional Structure

The layerwise and k-dimensional outputs in `experiments/03_layerwise_k_structure/` show structured variation across layers, subspace dimension, and architecture families.

Main artifacts:

- `layerwise_k_summary.csv`
- `layerwise_k_trajectory.csv`
- `compressibility_summary.csv`
- `multiarch_summary.csv`

## Experiment 4: Uncertainty Steering

The five-check steering battery in `experiments/04_uncertainty_steering/` supports directional and specific local uncertainty control.

Key aggregate results:

```text
Directional control:
decrease directional_rate = 0.9857
increase directional_rate = 0.9581

Equal Fisher-output-energy vs gradient-orthogonal control:
|Delta H| ratio = 5.88x to 32.40x
|Delta Var| ratio = 3.40x to 5.23x

Equal Fisher-output-energy vs random control:
|Delta H| ratio = 1.74x to 1.83x
|Delta Var| ratio = 1.60x to 1.67x

Rho dependency after controls:
partial corr rho -> |Delta H|   = 0.3837
partial corr rho -> |Delta Var| = 0.3245
```

The decoder-only Qwen logit-lens steering run is included in `experiments/04_uncertainty_steering/reports/decoder_qwen_report.md`.

The steering battery includes the requested five checks:

- directional control;
- equal Fisher-output-energy controls;
- specificity of uncertainty movement relative to answer content;
- rho dependency after scalar and geometric controls;
- replication across local models, tasks, layers, dimensions, and controls.

## Ablations And Boundary Cases

Controls are organized in `experiments/controls/`: random subspaces, Euclidean ablations, shuffled-surprisal controls, and equal Fisher-output-energy controls.

## Limitations

RoBERTa was not available locally. Semantic uncertainty is an embedding-based proxy. Decoder evidence is present but separate from the full MLM battery. External hallucination benchmarks and end-to-end generation interventions remain future work.
