# Final Reproducibility And Results Report

This cleaned main branch contains the paper-facing scientific story:

1. core Fisher-output accessible-varentropy theory;
2. scalar-matched evidence that uncertainty scalars do not determine accessibility;
3. local perturbation evidence that accessibility predicts uncertainty movement;
4. layerwise and k-dimensional structure;
5. uncertainty steering and equal-output-movement/minimal-energy controls;
6. uncertainty controllability mapping as the diagnostic application.

Exploratory application tests were moved to `archive_applications_exploratory`.

## Core Definition

For an internal route or subspace `B`, `rho(B)` measures the fraction of the local varentropy direction accessible through `B` in Fisher-output geometry. The point of the measure is geometric controllability, not generic error prediction.

## Main Results

### Experiment 1

Scalar entropy, varentropy, confidence, and margin do not determine accessibility. Scalar-matched examples can have large rho differences.

Primary artifacts:

- `experiments/01_matched_scalar_uncertainty/outputs/same_uncertainty_different_rho_pairs.csv`
- `experiments/01_matched_scalar_uncertainty/outputs/same_uncertainty_different_accessibility_pairs.csv`

### Experiment 2

High-accessibility routes predict larger local entropy/varentropy movement under small interventions.

Primary artifacts:

- `experiments/02_local_perturbation_prediction/outputs/intervention_by_rho_quartile.csv`
- `experiments/02_local_perturbation_prediction/outputs/intervention_residual_correlations.csv`

### Experiment 3

Accessibility has structure over layer and subspace dimension. The layer heatmaps and k-compressibility curves show that rho is not route noise.

Primary artifacts:

- `experiments/03_layerwise_k_structure/outputs/layerwise_k_summary.csv`
- `experiments/03_layerwise_k_structure/outputs/compressibility_summary.csv`

### Experiment 4

Accessible directions steer uncertainty more efficiently than random and orthogonal controls, with answer/top-k preservation at small epsilon. Equal-output-movement controls address the objection that accessible directions merely move logits more.

Primary artifacts:

- `experiments/04_uncertainty_steering/outputs/directionality_summary.csv`
- `experiments/04_uncertainty_steering/outputs/equal_output_energy_contrasts.csv`
- `experiments/04_uncertainty_steering/01_minimal_intervention_energy/`
- `experiments/04_uncertainty_steering/02_equal_output_movement/`

### Experiment 5

The uncertainty controllability mapping test asks whether rho helps predict where uncertainty is controllable after controlling for scalar uncertainty, gradient norms, Fisher-output energy, Jacobian norm, model, layer, task, route family, subspace dimension, and budget mode.

Primary result:

```text
safe_movement AUPRC:       controls 0.673757 -> controls+rho 0.763993
safe_movement log-loss:    controls 0.433265 -> controls+rho 0.372353
minimal_energy MAE log10:  controls 0.391496 -> controls+rho 0.346423
```

Bootstrap deltas are positive for safe movement AUPRC, log-loss improvement, movement MAE improvement, and minimal-energy MAE improvement.

Primary artifacts:

- `experiments/05_controllability_mapping/outputs/mapping_metrics.csv`
- `experiments/05_controllability_mapping/outputs/mapping_bootstrap_ci.csv`
- `experiments/05_controllability_mapping/reports/uncertainty_controllability_mapping_test.md`

## Required Controls

Core controls retained on main:

- gradient baselines: `experiments/controls/gradient_baselines/`;
- full regression: `experiments/controls/full_regression/`;
- semantic/top-k preservation: `experiments/controls/semantic_preservation/`;
- top-k robustness: `experiments/controls/topk_robustness/`;
- equal Fisher-output movement: `experiments/controls/fisher_output_energy_control/`;
- bootstrap and matched scalar-gradient diagnostics: `experiments/controls/statistical_diagnostics/`.

Appendix controls retained on main:

- `experiments/controls/random_subspaces/`;
- `experiments/controls/euclidean_ablation/`;
- `experiments/controls/shuffled_surprisal/`;
- `experiments/controls/full_vocab_sanity/`;
- `experiments/controls/out_of_sample_generalization/`;
- `experiments/controls/random_init_vs_pretrained/`;
- `experiments/controls/external_uncertainty_comparators/`.

## Reproduction Commands

Regenerate figures:

```bash
python scripts/make_paper_figures.py
```

Run steering cost/equal-output controls:

```bash
python scripts/run_control_cost_and_equal_output_tests.py --tau-entropy 0.02 --tau-varentropy 0.04 --top-k-output 32 --seed 20260529
```

Run controllability mapping:

```bash
python scripts/run_uncertainty_controllability_mapping_test.py --out-dir experiments/05_controllability_mapping --decoder-models distilgpt2,Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B-Instruct,microsoft/phi-2 --max-items 18 --top-m 8 --layers auto3 --subspace-dims 1,2,4 --eps 0.01,0.025,0.05,0.1,0.2 --movement-threshold 0.01 --drift-cap 0.00025 --top-k 3 --bootstrap 300 --torch-dtype float16 --device cuda --trust-remote-code --seed 20260608
```

## Limitations

Rho should be claimed as a local controllability geometry, not a generic error predictor. Gradient baselines remain strong for raw local movement. Minimal-energy results are strongest in the top-k local-linear setting and should be stated with that caveat. The mapping test currently covers local decoder models; masked-LM variants requested in that test were not present in the local cache and were skipped without download.
