# Final Reproducibility And Results Report

This cleaned main branch contains the paper-facing scientific story:

1. core Fisher-output accessible-varentropy theory;
2. scalar-matched evidence that uncertainty scalars do not determine accessibility;
3. local perturbation evidence that accessibility predicts uncertainty movement;
4. uncertainty steering and equal-output-movement/minimal-energy controls;
5. uncertainty controllability mapping as the diagnostic application;
6. rho-guided selective reliability as the non-oracle decision test.

Side experiments and exploratory ablations were moved to `archive_disruptive_experiments_full`.

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

Accessible directions steer uncertainty more efficiently than random and orthogonal controls, with answer/top-k preservation at small epsilon. Equal-output-movement controls address the objection that accessible directions merely move logits more. The choose-the-route intervention test makes this decision-facing: for each example, choose the high-rho route and compare it against low-rho, gradient-selected, random, and equal-Fisher-output controls.

Primary choose-route result:

```text
abs Delta entropy:     high-rho 0.048148 vs low-rho 0.027510 vs random 0.037249
abs Delta varentropy:  high-rho 0.078786 vs low-rho 0.050399 vs random 0.063799
Fisher-output drift:   all matched at 0.05
```

Cluster bootstrap CIs are positive for high-rho over low-rho, random-route, gradient-orthogonal, and equal-output-energy controls on entropy and varentropy movement. High-rho is roughly tied with the gradient-selected route on raw movement, which should be presented as a caveat rather than a failure.

Primary artifacts:

- `experiments/04_uncertainty_steering/outputs/directionality_summary.csv`
- `experiments/04_uncertainty_steering/outputs/equal_output_energy_contrasts.csv`
- `experiments/04_uncertainty_steering/01_minimal_intervention_energy/`
- `experiments/04_uncertainty_steering/02_equal_output_movement/`
- `experiments/04_uncertainty_steering/03_choose_route_intervention/`

### Experiment 4

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

### Experiment 5

The rho-guided selective reliability test asks whether rho improves a non-oracle accept/abstain/route/intervene policy. Correctness is not used as a decision feature; it is used only for out-of-fold training and evaluation. The retained test compares a strong B2 baseline against B2+rho variants and B2+shuffled-rho variants under grouped-prompt, leave-one-model-out, and leave-one-source-out splits.

Primary result:

```text
AURC mean risk:  baseline 0.469461 -> baseline+rho_abs 0.446009
Brier:           baseline 0.153992 -> baseline+rho_abs 0.115138
Log-loss:        baseline 0.511038 -> baseline+rho_abs 0.382033
```

Bootstrap CIs are positive for AURC, Brier, and log-loss improvements versus B2. The shuffled-rho control remains competitive in grouped-prompt and leave-one-model-out splits, so this result should be treated as secondary reliability evidence for rho-family features rather than the cleanest mechanistic isolation of row-aligned rho.

Leave-one-model-out keeps the same qualitative pattern, while leave-one-source-out does not. Absolute, adjusted, rank, percentile, and model/layer/route-family z-scored rho variants do not rescue leave-one-source-out on log-loss, Brier, or ECE. The source breakdown shows the failure is concentrated when `masked_topk` is held out. This is treated as a limitation and boundary condition: current evidence supports rho as useful outside prompt groups and across held-out models, but not yet as a general reliability method across source/protocol shifts.

Primary artifacts:

- `experiments/controls/rho_guided_selective_reliability/outputs/selective_reliability_final_table.csv`
- `experiments/controls/rho_guided_selective_reliability/outputs/selective_reliability_metrics.csv`
- `experiments/controls/rho_guided_selective_reliability/outputs/selective_reliability_bootstrap_ci.csv`
- `experiments/controls/rho_guided_selective_reliability/reports/report.md`

## Required Controls

Core controls retained on main:

- gradient baselines: `experiments/controls/gradient_baselines/`;
- full regression: `experiments/controls/full_regression/`;
- semantic/top-k preservation: `experiments/controls/semantic_preservation/`;
- top-k robustness: `experiments/controls/topk_robustness/`;
- equal Fisher-output movement: `experiments/controls/fisher_output_energy_control/`;
- bootstrap and matched scalar-gradient diagnostics: `experiments/controls/statistical_diagnostics/`;
- rho-guided selective reliability: `experiments/controls/rho_guided_selective_reliability/`.

Scale and model-family robustness retained on main:

- `experiments/controls/scale_external_robustness/`.

## Reproduction Commands

Regenerate figures:

```bash
python scripts/make_paper_figures.py
```

Run steering cost/equal-output controls:

```bash
python scripts/run_control_cost_and_equal_output_tests.py --tau-entropy 0.02 --tau-varentropy 0.04 --top-k-output 32 --seed 20260529
```

Run choose-the-route intervention:

```bash
python scripts/run_choose_route_intervention_test.py --bootstrap 1000 --seed 20260610
```

Run controllability mapping:

```bash
python scripts/run_uncertainty_controllability_mapping_test.py --out-dir experiments/05_controllability_mapping --decoder-models distilgpt2,Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B-Instruct,microsoft/phi-2,meta-llama/Llama-3.2-1B,mistralai/Mistral-7B-v0.1 --masked-models distilbert-base-uncased,bert-base-uncased,roberta-base,prajjwal1/bert-tiny --max-items 18 --top-m 8 --layers auto3 --subspace-dims 1,2,4 --eps 0.01,0.025,0.05,0.1,0.2 --movement-threshold 0.01 --drift-cap 0.00025 --top-k 3 --bootstrap 300 --torch-dtype float16 --device cuda --trust-remote-code --seed 20260608
```

Aggregate scale and external robustness evidence:

```bash
python scripts/run_scale_external_robustness_matrix.py
```

Run rho-guided selective reliability:

```bash
python scripts/run_rho_guided_selective_reliability.py --bootstrap 1000 --seed 20260609
```

## Limitations

Rho should be claimed as a local controllability geometry, not a generic error predictor. Gradient baselines remain strong for raw local movement. Minimal-energy results are strongest in the top-k local-linear setting and should be stated with that caveat. Selective reliability improves under grouped-prompt and leave-one-model-out evaluation, but not under leave-one-source-out evaluation; the current source count is too small to sell "general reliability." RoBERTa, Llama, and Mistral are now part of the requested default test sets and are downloaded automatically when absent unless `--local-files-only` is set.
