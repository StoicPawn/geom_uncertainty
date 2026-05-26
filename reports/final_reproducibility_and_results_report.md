# Final Reproducibility And Results Report

This is the single paper-facing reproducibility report for the repository. It links each claim to the experiment, CSV artifacts, paper-ready figures, and minimal commands needed to regenerate the analysis.

## Repository Map

```text
experiments/
  01_matched_scalar_uncertainty/
  02_local_perturbation_prediction/
  03_layerwise_k_structure/
  04_uncertainty_steering/
    01_minimal_intervention_energy/
    02_equal_output_movement/
    decoder_main_battery/
  controls/
applications/
  03_calibration_diagnosis/
  local_confidence_control/
  04_uncertainty_circuits/
  05_brittle_confidence/
  06_hidden_fragility_cifar_c/
  07_safe_model_editing/
reports/
  final_reproducibility_and_results_report.md
  figures/
scripts/
src/accessible_varentropy/
```

The claim-to-artifact map is in `CLAIM_MAP.md`. Figure sources are in `scripts/make_paper_figures.py`, and rendered SVG/PNG files are in `reports/figures/`.

## Core Mathematical Object

For a top-k output distribution `p`, define centered surprisal `u = -log p - H(p)` and categorical Fisher matrix `F = diag(p) - pp^T`. For a latent subspace `B` and local logit Jacobian `J`, accessibility is measured in Fisher-whitened output space:

```text
w = F^{1/2} u
rho(B) = ||P_Im(F^{1/2} J B) w||^2 / ||w||^2
Var = V_access + V_inaccess
```

The empirical question is whether `V_access` predicts and enables local uncertainty movement better than scalar uncertainty alone.

## Models

Main masked-LM runs:

- `distilbert-base-uncased`
- `bert-base-uncased`
- `google/bert_uncased_L-2_H-128_A-2`

Decoder evidence:

- `Qwen/Qwen2.5-0.5B`, included as decoder-only logit-lens steering evidence inside Experiment 4.
- `microsoft/phi-2`, included as a stronger decoder-only replication inside Experiment 4.

Unavailable or incomplete:

- RoBERTa was not available in the local Hugging Face cache. No network download was attempted for the consolidated run.
- Llama and Mistral were requested as stronger decoder replications, but they were not available in the local Hugging Face cache. No network download was attempted.

## Tasks And Prompt Counts

Experiment 1 uses the factual-error/deep cloze suite and scalar-matched diagnostics. Its checked-in prompt table has 960 model-prompt rows.

Experiments 2 and 3 use 320 DistilBERT masked-cloze prompts covering factual and error cases across topics including capital, science, language, math, common knowledge, plural, role, culture, currency, continent, antonym, and animal class.

Experiment 4 uses 99 model-prompt rows across three local masked-LM models:

```text
bert-base-uncased: 12 ambiguous_open, 12 factual_deep, 9 factual_simple
distilbert-base-uncased: 12 ambiguous_open, 12 factual_deep, 9 factual_simple
google/bert_uncased_L-2_H-128_A-2: 12 ambiguous_open, 12 factual_deep, 9 factual_simple
```

The exact per-model correctness counts are in `experiments/04_uncertainty_steering/data/prompt_tables.csv`.

The decoder-only main battery uses token-level QA, factual completion, and generative-open prompts. It contributes 18 prompt-layer rows for Qwen and 18 prompt-layer rows for Phi, using two layers per model and three tasks.

The uncertainty-circuit application uses the same three local masked-LM models with top-k 32 route scoring and 1,440 Fisher-normalized route interventions. The brittle-confidence application audits 96 matched high/low-rho rows and 368 prompt-perturbation records across neutral context, format, template, and synonym changes.

The hidden-fragility vision application is configured for CIFAR-10 clean test predictions and CIFAR-10-C corruptions across standard severities 1-5. A small local CPU pilot has been executed: 2,000 clean test examples, 12 matched low/high-rho pairs, 288 corruption records, and a 5-epoch ResNet-18 checkpoint. This is a pipeline validation, not paper evidence.

The calibration-diagnosis application uses 57 factual model-prompt rows across the three local masked-LM models, full-vocabulary calibration metrics, top-k 32 accessibility geometry, auto-selected layers, and Fisher-output-equal interventions.

The safe model-editing diagnostic uses DistilBERT and tiny BERT local masked-LM logit-lens edits: 48 prompt rows, 240 route-specific edit records, and 2,880 side-effect prompt evaluations.

## Layers, Dimensions, Epsilons, And Subspaces

Layer coverage:

- Experiment 2: all DistilBERT hidden layers available to the intervention run.
- Experiment 3: DistilBERT layers `0..6`.
- Experiment 4: auto-selected layer thirds and final layer per model. The resulting matrix is in `experiments/04_uncertainty_steering/outputs/replication_matrix.csv`.
- Decoder main battery: Qwen layers `12` and `23`; Phi layers `16` and `31`.
- Applications 4 and 5: auto-selected masked-LM layers from each model.
- Application 6: ResNet-18 CIFAR penultimate embedding route.
- Application 3: auto-selected masked-LM layers from each model.
- Application 7: auto-selected masked-LM layers for local representation edits.

Subspace dimensions:

- Experiment 3: `k = 1, 2, 4, 8, 16, 32`.
- Experiment 4: `k = 1, 4, 8, 16`, filtered by hidden dimension and sample count.
- Decoder main battery: PCA/random subspace dimension `8`.
- Applications 4 and 5: route subspace dimension `8`.
- Application 7: route subspace dimension `8`.
- Application 6: PCA route dimension `64` by default, plus full 10-class output Fisher geometry.
- Application 3: route subspace dimension `8`.

Epsilon values:

- Experiment 2 latent perturbations: `epsilon = 0.25, 0.5, 1.0`.
- Experiment 4 latent-equal pilot/full controls: `epsilon = 0.05, 0.1`.
- Experiment 4 Fisher-output-equal controls: `epsilon = 0.02, 0.05, 0.1`.
- DistilBERT pilot steering: `epsilon = 0.1, 0.2, 0.4`.
- Decoder steering: `epsilon = 0.05, 0.1, 0.2`.
- Decoder main battery: equal Fisher-output-energy `epsilon = 0.05`.
- Applications 4 and 5: equal Fisher-output-energy `epsilon = 0.05`.
- Application 6: no steering epsilon; it evaluates corruption severity thresholds and robustness under label-preserving shifts.
- Application 3: equal Fisher-output-energy `epsilon = 0.05`.

Subspace types:

- `state_pca`: PCA basis of hidden states at a layer.
- `delta_pca`: PCA basis of forward hidden-state deltas.
- `delta_forward_1d`: one-dimensional forward delta direction.
- `random`: random orthonormal subspaces.
- `accessible_ls`: least-squares accessible steering direction.
- `grad_orthogonal_control`: latent control orthogonal to the entropy gradient.
- `random_control`: random latent control.

## Jacobian And Fisher Details

Masked-LM Jacobian:

- The code computes selected-token MLM-head logits from the `[MASK]` hidden vector.
- `torch.autograd.functional.jacobian(..., vectorize=True)` gives `J = d logits_topk / d hidden`.
- The top-k token set is selected from the full-vocabulary logits, then renormalized into a top-k distribution for Fisher geometry.

Decoder Jacobian:

- The decoder run uses a top-m next-token logit lens at the selected layer and computes the local Jacobian with respect to the layer hidden state.
- The decoder main battery applies the same Jacobian protocol to Qwen and Phi, with `state_pca`, `delta_pca`, and random decoder subspaces.

Fisher stabilization:

- Probabilities are clipped with `eps = 1e-12` and renormalized.
- `F = diag(p) - pp^T`.
- `F^{1/2}` is computed by eigendecomposition.
- Negative numerical eigenvalues are clipped to zero before the square root.
- Projection rank uses an SVD cutoff of approximately `1e-10 * max(shape) * sigma_max`.
- Least-squares steering uses pseudoinverse tolerance `rcond = 1e-10`.

Output lens:

- Experiments 1, 2, and 3 use top-k output lenses with `top-k = 32`.
- Experiment 4 full battery uses `top-k = 16`.
- Top-k robustness controls explicitly evaluate `top-k = 16, 32, 64, 128, 256`.
- Full-vocabulary sanity check evaluates full-vocabulary Fisher-kernel accessibility on `google/bert_uncased_L-2_H-128_A-2`.
- The DistilBERT pilot uses `top-k = 32`.
- The decoder run uses `top-m = 16`.
- The decoder main battery also uses `top-m = 16`.
- The computations are therefore top-k/top-m local Fisher geometries, not full-vocabulary Fisher matrices.

## Seeds

```text
123        factual-error benchmark defaults
7          multi-architecture layerwise controls
20260522   local perturbation/intervention run
20260523   layerwise k-dimensional adjusted-rho run
20260524   steering full battery, pilot steering, decoder steering
20260525   top-k robustness, gradient baselines, full regression, preservation controls
20260526   bootstrap diagnostics, prompt audit, failure modes, tiny full-vocabulary sanity
20260527   out-of-sample route generalization, random-init versus pretrained controls
20260528   decoder-only main battery, external uncertainty comparators, local confidence-control application
20260529   fixed-target intervention cost and equal-output-movement efficiency tests
20260530   uncertainty-circuit route localization and brittle-confidence perturbation applications
20260531   CIFAR-10/CIFAR-10-C hidden-fragility protocol, CPU pilot completed; full run resumable
20260532   calibration-diagnosis application for internal steerability of miscalibration
20260601   safe model-editing local representation diagnostic
```

## Environment

Observed environment for the consolidated repository:

```text
OS: Windows-10-10.0.26200-SP0
Python: 3.11.9
NumPy: 2.2.5
pandas: 2.3.3
PyTorch: 2.7.0+cpu
Transformers: 5.9.0
scikit-learn: 1.7.2
Matplotlib: 3.10.1
CUDA available: False
CPU count: 16
RAM: 16.72 GB
Torch threads: 12
```

Hardware details were gathered from Python runtime introspection. Windows WMI hardware queries were blocked by local permissions.

## Experiment 1: Scalar Uncertainty Does Not Determine Accessibility

Claim: entropy and varentropy do not determine accessibility.

Primary artifacts:

- `experiments/01_matched_scalar_uncertainty/outputs/same_uncertainty_different_rho_pairs.csv`
- `experiments/01_matched_scalar_uncertainty/outputs/same_uncertainty_different_accessibility_pairs.csv`
- `experiments/01_matched_scalar_uncertainty/figures/fig02_scalar_uncertainty_vs_rho.svg`
- `experiments/01_matched_scalar_uncertainty/figures/fig03_scalar_matched_pairs.svg`

Key result:

- 300 scalar-matched rho pairs were retained.
- Maximum `rho_abs_diff` is `0.9771`.
- Median entropy difference inside matched pairs is `0.0337`.
- Median varentropy difference inside matched pairs is `0.0852`.
- Adjusted-accessibility pairs reach `rho_adjusted_abs_diff = 1.0507`.

Interpretation: scalar uncertainty can be nearly matched while accessible uncertainty differs sharply.

## Experiment 2: Accessibility Predicts Local Uncertainty Movement

Claim: higher accessibility predicts larger local movement in entropy and varentropy.

Primary artifacts:

- `experiments/02_local_perturbation_prediction/outputs/intervention_by_rho_quartile.csv`
- `experiments/02_local_perturbation_prediction/outputs/intervention_residual_correlations.csv`
- `experiments/02_local_perturbation_prediction/figures/fig03_accessibility_predicts_movement.svg`

Key result:

```text
epsilon=0.25: q1 Delta H=0.0009, q4 Delta H=0.0343; q1 |Delta Var|=0.0038, q4 |Delta Var|=0.0509
epsilon=0.50: q1 Delta H=0.0007, q4 Delta H=0.0684; q1 |Delta Var|=0.0080, q4 |Delta Var|=0.1036
epsilon=1.00: q1 Delta H=-0.0031, q4 Delta H=0.1351; q1 |Delta Var|=0.0216, q4 |Delta Var|=0.2127
```

Interpretation: movement increases by rho quartile and scales with perturbation size.

## Experiment 3: Layerwise And K-Dimensional Structure

Claim: accessibility is structured across layer and subspace dimension, rather than being a scalar-output artifact.

Primary artifacts:

- `experiments/03_layerwise_k_structure/outputs/layerwise_k_summary.csv`
- `experiments/03_layerwise_k_structure/outputs/layerwise_k_scores.csv`
- `experiments/03_layerwise_k_structure/outputs/compressibility_summary.csv`
- `experiments/03_layerwise_k_structure/figures/fig05_layerwise_heatmap.svg`
- `experiments/03_layerwise_k_structure/figures/fig05_compressibility_curves.svg`

Key result:

- Layerwise run covers layers `0..6`.
- Dimensions are `k = 1, 2, 4, 8, 16, 32`.
- Subspaces are `delta_forward_1d`, `delta_pca`, and `state_pca`.
- Maximum mean adjusted accessibility is `0.3192` for `delta_forward_1d`, layer `5`, correct examples.
- Mean `k50` is `11.80`; mean `k80` is `23.85`.

Interpretation: accessibility accumulates with dimension and varies systematically by layer.

## Experiment 4: Uncertainty Steering

Claim: accessible directions steer uncertainty more than matched controls while mostly preserving answer content.

Primary artifacts:

- `experiments/04_uncertainty_steering/outputs/directionality_summary.csv`
- `experiments/04_uncertainty_steering/outputs/equal_output_energy_contrasts.csv`
- `experiments/04_uncertainty_steering/outputs/specificity_summary.csv`
- `experiments/04_uncertainty_steering/outputs/rho_dependency.csv`
- `experiments/04_uncertainty_steering/outputs/replication_matrix.csv`
- `experiments/04_uncertainty_steering/01_minimal_intervention_energy/outputs/minimal_energy_predictor_benchmark.csv`
- `experiments/04_uncertainty_steering/02_equal_output_movement/outputs/equal_output_control_contrasts.csv`
- `experiments/04_uncertainty_steering/decoder_main_battery/outputs/decoder_control_contrasts.csv`
- `experiments/04_uncertainty_steering/decoder_main_battery/outputs/decoder_steering_records.csv`
- `experiments/04_uncertainty_steering/figures/fig06_steering_vs_controls_ci.svg`
- `experiments/04_uncertainty_steering/figures/fig06_uncertainty_steering_main.svg`

Five checks:

0. Minimal intervention energy: estimate the latent cost required to hit fixed uncertainty targets.
1. Directional control: `+delta_acc` increases entropy and `-delta_acc` decreases entropy.
2. Equal Fisher-output-energy controls: comparisons are matched on `||F^{1/2} J delta z||`.
3. Specificity: uncertainty changes are tracked separately from top-1, correctness, semantic-proxy, and ranking shifts.
4. Rho dependency: rho predicts movement after scalar and geometric controls.
5. Replication: local MLMs, decoder-only LMs, tasks, layers, dimensions, and random controls are included.

Key result:

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

Decoder-only Qwen/Phi equal Fisher-output-energy:
|Delta H| accessible/random ratio = 1.68x to 2.69x
|Delta H| accessible/grad-orthogonal ratio = 3.76x to 35.04x

Operational fixed-target cost, MLM top-k 32 local-linear:
rho -> -log C_tau(H): Spearman = 0.7571, AUROC = 0.8723
rho -> -log C_tau(Var): Spearman = 0.6036, AUROC = 0.7860
```

Specificity:

- At `epsilon=0.02`, mean top-1 change rate is `0.0142`; target-correctness change rate is `0.0003`.
- At `epsilon=0.05`, mean top-1 change rate is `0.0345`; target-correctness change rate is `0.0010`.

Interpretation: accessible steering changes uncertainty more efficiently than controls at equal Fisher-output energy and largely preserves monitored answer identity at small epsilons. The fixed-target cost test reframes the same geometry as operational controllability: high-accessibility MLM cases need less estimated intervention energy, although direct projected gradients remain stronger raw cost predictors.

## Confirmatory Vs Exploratory Analyses

Confirmatory analyses are the claim-supporting tests used in the main paper narrative:

- scalar-matched uncertainty with different accessibility;
- accessibility predicting local movement;
- layerwise and k-dimensional structure;
- uncertainty steering against equal Fisher-output-energy controls;
- fixed-target minimal intervention energy for uncertainty movement;
- bootstrap CIs/effect sizes for the main steering and rho-quartile contrasts;
- same scalar uncertainty plus same projected-gradient magnitude but different `rho`;
- tiny-model full-vocabulary sanity check;
- train-route held-out prompt generalization;
- random-init versus pretrained learned-structure control;
- decoder-only Qwen/Phi steering on token-level QA, factual completion, and generative-open prompts;
- calibration diagnosis: high-rho routes as internally steerable NLL/Brier/target-probability repair paths;
- local confidence-control application with answer-neighborhood preservation;
- uncertainty-circuit route localization: high-rho routes as causal control points;
- answer-preserving qualitative steering examples.

Exploratory and diagnostic analyses are used to bound the claim:

- direct projected-gradient baselines;
- decoder cost cases where projected gradients dominate `rho` as a raw cost predictor;
- residual `rho` effects after scalar, gradient, and geometric controls;
- top-k robustness beyond the main lens;
- Semantic Entropy, Semantic Density, and HaloScope-style comparator diagnostics;
- ECE caveat in calibration diagnosis, where the exact-token baseline is already low-confidence and low-accuracy;
- brittle-confidence perturbation battery, currently a mixed/negative boundary case;
- safe model-editing diagnostic, where target-specific edit gradients dominate `rho` as edit-cost predictors;
- random-route competitiveness in held-out route generalization;
- prompt duplicate and near-duplicate audit;
- failure-mode table;
- Euclidean, shuffled-surprisal, and random-subspace ablations.

Claim boundary: `rho` should not be described as beating direct projected gradients as a raw infinitesimal predictor. The supported claim is that `rho` is a geometric accessibility coefficient: it decomposes varentropy with respect to an internal route, remains informative after controls, and identifies routes along which uncertainty can be changed while mostly preserving the answer neighborhood.

## Controls, Ablations, And Boundary Cases

Control folders:

- `experiments/controls/statistical_diagnostics/`
- `experiments/controls/full_vocab_sanity/`
- `experiments/controls/out_of_sample_generalization/`
- `experiments/controls/random_init_vs_pretrained/`
- `experiments/controls/external_uncertainty_comparators/`
- `experiments/controls/topk_robustness/`
- `experiments/controls/gradient_baselines/`
- `experiments/controls/full_regression/`
- `experiments/controls/semantic_preservation/`
- `experiments/controls/random_subspaces/`
- `experiments/controls/euclidean_ablation/`
- `experiments/controls/shuffled_surprisal/`
- `experiments/controls/fisher_output_energy_control/`

What they address:

- Statistical diagnostics add bootstrap CIs, effect sizes, matched scalar/gradient pairs, prompt audits, and failure modes.
- Full-vocabulary sanity tests whether top-k accessibility tracks full-vocabulary accessibility on a tiny model.
- Out-of-sample generalization tests whether train-estimated routes transfer to held-out prompts.
- Random-init versus pretrained tests whether learned weights change accessibility structure.
- External uncertainty comparators test whether accessible varentropy measures the same object as Semantic Entropy, Semantic Density, or HaloScope-style consistency.
- Random subspaces test whether the result is a generic subspace effect.
- Euclidean ablation tests whether Fisher geometry is needed.
- Shuffled surprisal tests whether the true centered-surprisal direction is needed.
- Equal Fisher-output-energy controls test whether accessible steering wins after matching total output movement.
- Top-k robustness tests whether rho rankings, layer trends, and steering effects survive output-lens changes.
- Gradient baselines compare rho to `||Pi_B grad H||`, `||Pi_B grad Var||`, `||F^{1/2}JB||`, and `||JB||`.
- Full regression tests rho after scalar, gradient, geometric, model, layer, and prompt controls.
- Semantic preservation extends top-1 checks to top-5/top-10 Jaccard, KL, candidate-set mass, and embedding-cluster proxies.

Application folder:

- `applications/03_calibration_diagnosis/`: internal steerability diagnostic for calibration errors.
- `applications/local_confidence_control/`: selective local confidence control with answer-neighborhood preservation.
- `applications/04_uncertainty_circuits/`: route-interpretability test for causal localization of uncertainty accessibility.
- `applications/05_brittle_confidence/`: high-confidence matched prompt-perturbation fragility test and boundary-case diagnostic.
- `applications/06_hidden_fragility_cifar_c/`: CIFAR-10 to CIFAR-10-C hidden-fragility protocol for confident correct predictions.
- `applications/07_safe_model_editing/`: local representation-editing diagnostic for edit cost and side effects.

## Added Control A: Top-k Robustness

Primary artifacts:

- `experiments/controls/topk_robustness/outputs/topk_rank_stability.csv`
- `experiments/controls/topk_robustness/outputs/topk_layer_trend_stability.csv`
- `experiments/controls/topk_robustness/outputs/topk_steering_summary.csv`
- `experiments/controls/topk_robustness/figures/fig07_topk_robustness.svg`

Coverage:

- Models: `distilbert-base-uncased`, `bert-base-uncased`, `google/bert_uncased_L-2_H-128_A-2`.
- Top-k values: `16, 32, 64, 128, 256`.
- Score rows: `2415`.
- Steering rows: `14490`.
- Fisher-output epsilon: `0.05`.

Rank stability against top-k 32:

```text
k=16:  Spearman rho = 0.7670
k=32:  Spearman rho = 1.0000
k=64:  Spearman rho = 0.9133
k=128: Spearman rho = 0.8439
k=256: Spearman rho = 0.7882
```

Interpretation: the signal does not collapse under top-k changes, but it is not invariant. The most stable comparison is 32 to 64; 16 and 256 remain directionally consistent but should be described as moderately shifted output lenses.

## Added Confirmatory Diagnostic: Bootstrap CIs And Effect Sizes

Primary artifacts:

- `experiments/controls/statistical_diagnostics/outputs/bootstrap_steering_contrasts.csv`
- `experiments/controls/statistical_diagnostics/outputs/bootstrap_rho_quartiles.csv`
- `experiments/controls/statistical_diagnostics/reports/report.md`

At top-k 32 and equal Fisher-output energy, accessible steering versus controls has bootstrap CIs and large paired effect sizes:

```text
|Delta H| accessible - random:
decrease = 0.02047 [0.01940, 0.02159], ratio = 2.50 [2.35, 2.66], d = 1.62 [1.52, 1.74]
increase = 0.02033 [0.01923, 0.02148], ratio = 2.61 [2.45, 2.82], d = 1.62 [1.53, 1.73]

|Delta H| accessible - grad-orthogonal:
decrease = 0.03302 [0.03163, 0.03455], ratio = 30.43 [27.32, 34.06], d = 2.02 [1.92, 2.12]
increase = 0.03180 [0.03016, 0.03332], ratio = 29.13 [26.37, 32.39], d = 1.78 [1.70, 1.88]
```

Rho quartile contrasts also remain large:

```text
epsilon=0.25: q4-q1 |Delta H| = 0.03848 [0.03710, 0.03973], d = 2.06 [1.98, 2.15]
epsilon=0.50: q4-q1 |Delta H| = 0.07458 [0.07206, 0.07732], d = 2.03 [1.96, 2.11]
epsilon=1.00: q4-q1 |Delta H| = 0.13606 [0.13070, 0.14126], d = 1.90 [1.83, 1.98]
```

## Added Confirmatory Diagnostic: Same Scalar + Same Projected Gradients + Different Rho

Primary artifacts:

- `experiments/controls/statistical_diagnostics/outputs/same_scalar_gradient_different_rho_pairs.csv`
- `experiments/controls/statistical_diagnostics/outputs/same_scalar_gradient_different_rho_summary.csv`

The matched-pair diagnostic searches within the same model, prompt, layer, and top-k lens, so scalar uncertainty is exactly matched. It then matches projected-gradient norms and asks whether `rho` can still differ:

```text
n pairs = 86
mean rho absolute difference = 0.2487
max rho absolute difference = 0.4935
median entropy difference = 0.0000
median varentropy difference = 0.0000
median |Delta projected grad H norm| = 0.0271
median |Delta projected grad Var norm| = 0.0526
```

Interpretation: this directly addresses the gradient-baseline novelty concern. Even at identical scalar uncertainty and similar projected-gradient magnitudes, the geometric accessibility coefficient can vary materially across internal routes.

## Added Control B: Direct Gradient Baselines

Primary artifacts:

- `experiments/controls/gradient_baselines/outputs/gradient_baseline_correlations.csv`
- `experiments/controls/gradient_baselines/outputs/gradient_baseline_scores.csv`
- `experiments/controls/gradient_baselines/figures/fig08_gradient_baselines.svg`

Accessible-only Spearman correlations:

```text
Predicting |Delta H|:
rho                         0.6277
||Pi_B grad H||             0.8827
||Pi_B grad Var||           0.7870
||F^{1/2}JB||              -0.0466
||JB||                      0.1600

Predicting |Delta Var|:
rho                         0.3544
||Pi_B grad H||             0.6815
||Pi_B grad Var||           0.8127
||F^{1/2}JB||              -0.0462
||JB||                      0.1853
```

Interpretation: direct projected gradients are stronger immediate predictors of infinitesimal local movement. This is expected and narrows the claim: rho is not a replacement for the gradient; it is a geometric accessibility coefficient that remains informative after controls and is useful for decomposing accessible versus inaccessible uncertainty.

## Added Control C: Full Regression

Primary artifact:

- `experiments/controls/full_regression/outputs/full_regression_ols.csv`

Controls:

`entropy`, `varentropy`, `confidence`, `margin`, `jacobian_fro_norm`, `fisher_output_energy`, `grad_entropy_proj_norm`, `grad_varentropy_proj_norm`, `top_k_output`, `layer`, `model`, `task`, `topic`, `subspace_family`, `subspace_k`, `direction`, `sign`.

Key result:

```text
Outcome              rho beta   t       p
|Delta H|            0.00343   19.00   1.55e-79
|Delta Var|          0.00723   17.71   1.84e-69
directional success  0.00419    0.80   0.424
```

Interpretation: rho remains significant for movement magnitudes after strong controls, including direct gradient baselines. It does not independently explain directional success once direction type and sign are included; directional success is mostly a property of how the steering direction is constructed.

## Added Control D: Semantic And Top-k Preservation

Primary artifacts:

- `experiments/controls/semantic_preservation/outputs/semantic_topk_preservation.csv`
- `experiments/controls/semantic_preservation/outputs/semantic_topk_preservation_summary.csv`

At top-k 32 and Fisher-output epsilon `0.05`, accessible steering has:

```text
selected top-1 changed rate: 0.0207 to 0.0248
selected top-5 Jaccard:      0.9593 to 0.9614
full-vocab top-10 Jaccard:   0.9484 to 0.9529
candidate-set KL:            about 0.00125
candidate mass retention:    about 0.978 to 1.020
semantic cluster L1:         about 0.022
embedding centroid shift:    about 0.013
```

Interpretation: top-1 preservation is not the only evidence. The selected and full-vocabulary candidate neighborhoods remain mostly stable at the steering scale used for the control battery.

Qualitative answer-preserving examples are stored in:

- `experiments/04_uncertainty_steering/outputs/answer_preserving_qualitative_examples.csv`
- `experiments/04_uncertainty_steering/reports/answer_preserving_qualitative_examples.md`

Example pattern: the prompt `The meeting happened near the [MASK] in the school.` preserves the answer token `library` while entropy moves from `0.9675` to `1.0590` under an increasing accessible intervention, with full-vocabulary top-10 Jaccard equal to `1.000`.

## Added Test 1: Minimal Intervention Energy

Primary artifacts:

- `experiments/04_uncertainty_steering/01_minimal_intervention_energy/outputs/minimal_energy_cost_records.csv`
- `experiments/04_uncertainty_steering/01_minimal_intervention_energy/outputs/minimal_energy_predictor_benchmark.csv`
- `experiments/04_uncertainty_steering/01_minimal_intervention_energy/outputs/minimal_energy_residual_effects.csv`
- `experiments/04_uncertainty_steering/01_minimal_intervention_energy/reports/report.md`

This test asks the inverse controllability question: for fixed targets `Delta H >= 0.02` and `|Delta Var| >= 0.04`, how much latent intervention cost is needed?

Key result:

```text
MLM top-k 32 local-linear:
rho -> -log C_tau(H):   Spearman = 0.7571, AUROC = 0.8723
rho -> -log C_tau(Var): Spearman = 0.6036, AUROC = 0.7860
rho residual effect after scalar, gradient, Fisher/Jacobian, model/task/layer controls:
  H   residual Spearman = 0.5863, standardized beta = 0.4337
  Var residual Spearman = 0.4061, standardized beta = 0.3013

MLM full multi-epsilon grid:
rho -> -log C_tau(H):   Spearman = 0.2132, AUROC = 0.6012
rho -> -log C_tau(Var): Spearman = 0.1206, AUROC = 0.5697
```

Interpretation: the operational cost test is strongest in the matched top-k 32 local-linear setting, where `rho` remains predictive after strong controls. The multi-epsilon grid is weaker once direct gradient and Fisher controls are included. The paper should therefore present this as evidence for local controllability cost, not as a universal replacement for projected-gradient predictors.

## Added Test 2: Equal-Output-Movement Steering

Primary artifacts:

- `experiments/04_uncertainty_steering/02_equal_output_movement/outputs/equal_output_efficiency_records.csv`
- `experiments/04_uncertainty_steering/02_equal_output_movement/outputs/equal_output_control_contrasts.csv`
- `experiments/04_uncertainty_steering/02_equal_output_movement/outputs/equal_output_preservation_summary.csv`
- `experiments/04_uncertainty_steering/02_equal_output_movement/reports/report.md`

This test asks whether accessible directions move uncertainty more per unit matched Fisher-output movement.

Key result:

```text
MLM top-k 32 entropy-efficiency accessible/random ratio: 2.44x to 2.64x
MLM top-k 32 entropy-efficiency accessible/grad-orthogonal ratio: 27.40x to 32.09x
Decoder entropy-efficiency accessible/random ratio: 2.02x to 2.32x
Decoder entropy-efficiency accessible/grad-orthogonal ratio: 6.77x to 14.47x
```

Preservation:

```text
MLM accessible top-1 change: 0.0207 to 0.0248; top-10 Jaccard about 0.95
Decoder accessible top-1 change: 0.0000 to 0.0278; top-10 Jaccard about 0.95
```

Interpretation: at matched `||F^{1/2}J delta z||`, accessible directions produce more uncertainty movement than random and gradient-orthogonal controls while preserving the local answer neighborhood.

## Added Control E: Decoder-only Main Evidence

Decoder-only evidence is now a main Experiment 4 battery rather than only an appendix. It runs Qwen and Phi on token-level QA, factual completion, and generative-open next-token prompts:

- `experiments/04_uncertainty_steering/decoder_main_battery/README.md`
- `experiments/04_uncertainty_steering/decoder_main_battery/outputs/decoder_scores.csv`
- `experiments/04_uncertainty_steering/decoder_main_battery/outputs/decoder_steering_records.csv`
- `experiments/04_uncertainty_steering/decoder_main_battery/outputs/decoder_control_contrasts.csv`
- `experiments/04_uncertainty_steering/decoder_main_battery/reports/report.md`

Key result:

```text
Models completed: Qwen/Qwen2.5-0.5B, microsoft/phi-2
Skipped due to local cache absence: Llama, Mistral
rho mean range across decoder layers/subspaces: 0.6602 to 0.9052
|Delta H| accessible/random ratio: 1.68x to 2.69x
|Delta H| accessible/grad-orthogonal ratio: 3.76x to 35.04x
```

Interpretation: decoder-only logit-lens steering shows the same qualitative accessible-versus-control pattern on two local decoder LMs. Llama/Mistral remain planned replications because they were not locally available.

## Added Control E2: External Uncertainty Comparator Diagnostics

Primary artifacts:

- `experiments/controls/external_uncertainty_comparators/outputs/comparator_scores.csv`
- `experiments/controls/external_uncertainty_comparators/outputs/comparator_metric_correlations.csv`
- `experiments/controls/external_uncertainty_comparators/outputs/comparator_steering_sensitivity.csv`
- `experiments/controls/external_uncertainty_comparators/reports/report.md`

This diagnostic compares accessible varentropy to token-level approximations of Semantic Entropy, Semantic Density, and a HaloScope-style unlabeled consistency risk. It is not a hallucination leaderboard claim; it asks whether the metrics track the same object.

Key result for accessible steering sensitivity to `|Delta H|`:

```text
rho                                  Spearman =  0.6464
semantic_entropy_proxy               Spearman = -0.7374
semantic_density_uncertainty         Spearman = -0.7832
haloscope_style_consistency_risk      Spearman = -0.7531
```

Interpretation: the semantic/density/consistency proxies are strongly related to scalar uncertainty structure, while `rho` is route-specific. They are not interchangeable measurements, which is exactly the distinction the paper should make.

## Added Application 3: Calibration Diagnosis

Primary artifacts:

- `applications/03_calibration_diagnosis/outputs/clean_calibration_records.csv`
- `applications/03_calibration_diagnosis/outputs/route_scores.csv`
- `applications/03_calibration_diagnosis/outputs/calibration_intervention_records.csv`
- `applications/03_calibration_diagnosis/outputs/calibration_summary.csv`
- `applications/03_calibration_diagnosis/outputs/calibration_effects.csv`
- `applications/03_calibration_diagnosis/outputs/calibration_route_contrasts.csv`
- `applications/03_calibration_diagnosis/reports/report.md`

Protocol: for factual masked-LM prompts with known target tokens, compute full-vocabulary confidence, correctness, NLL, Brier score, and ECE. Score candidate internal routes by top-k 32 accessible varentropy. For each prompt, select high-rho, low-rho, and random routes, then apply the same Fisher-output-energy intervention. The calibration policy decreases entropy for correct examples and increases entropy for incorrect examples.

Key result:

```text
clean baseline: accuracy 0.0175, ECE 0.0310, NLL 8.9147, Brier 1.0090
high-rho route: accuracy 0.3860, ECE 0.0923, NLL 4.3609, Brier 0.6809
low-rho route:  accuracy 0.0526, ECE 0.0320, NLL 7.0463, Brier 0.9815
random route:   accuracy 0.3860, ECE 0.1212, NLL 4.2607, Brier 0.6980
```

High-rho versus low-rho paired route contrasts:

```text
after NLL:          high - low = -2.6854, high better rate = 0.7018
after Brier:        high - low = -0.3006, high better rate = 0.5789
target-prob delta:  high - low = +0.0052, high better rate = 0.7018
full-vocab KL:      high - low = -0.0002, high better rate = 0.4912
```

Interpretation: high-rho routes make target-probability, NLL, Brier, and exact-target accuracy much more steerable than low-rho routes, with similar full-vocabulary KL and top-10 preservation. ECE does not improve in this run because the exact-token baseline is already low-confidence and low-accuracy, so it is nearly calibrated in a pessimistic sense. This application should be framed as an internal calibration-steerability diagnostic with an ECE caveat, not as a pure ECE-improvement result.

## Added Application: Local Confidence Control

Primary artifacts:

- `applications/local_confidence_control/outputs/selective_confidence_control_summary.csv`
- `applications/local_confidence_control/outputs/selective_confidence_control_examples.csv`
- `applications/local_confidence_control/reports/report.md`

Success requires unchanged top-1, top-10 Jaccard at least `0.8`, unchanged target correctness when a target exists, and above-median entropy movement.

Key result:

```text
Qwen decrease: success_rate 0.4722, answer_preserved_rate 1.0000
Qwen increase: success_rate 0.4444, answer_preserved_rate 0.9444
Phi decrease:  success_rate 0.4444, answer_preserved_rate 1.0000
Phi increase:  success_rate 0.6389, answer_preserved_rate 1.0000
```

Interpretation: this is the clearest killer-application slice: local uncertainty/confidence can be moved while preserving the monitored answer neighborhood.

## Added Application 4: Uncertainty Circuits / Route Interpretability

Primary artifacts:

- `applications/04_uncertainty_circuits/outputs/route_scores.csv`
- `applications/04_uncertainty_circuits/outputs/circuit_intervention_records.csv`
- `applications/04_uncertainty_circuits/outputs/circuit_route_contrasts.csv`
- `applications/04_uncertainty_circuits/outputs/rho_causal_effect_correlations.csv`
- `applications/04_uncertainty_circuits/reports/report.md`

Protocol: compute `rho(B)` across model, prompt, layer, and subspace routes; select high-rho, low-rho, random, entropy-gradient, and varentropy-gradient routes; apply the same Fisher-output-normalized intervention budget to each route; measure uncertainty movement and answer-neighborhood preservation.

Key result:

```text
High-rho route mean rho: 0.5986
Low-rho route mean rho:  0.5152

High-rho vs low-rho:
|Delta H|  +0.00260, ratio 1.11x, high better rate 0.5903
|Delta Var| +0.00542, ratio 1.12x, high better rate 0.5486

High-rho vs random:
|Delta H|  +0.00166, ratio 1.08x, high better rate 0.6181
|Delta Var| +0.00393, ratio 1.09x, high better rate 0.6042

High-rho vs entropy-gradient route:
|Delta H|  +0.00560, ratio 1.21x, high better rate 1.0000
|Delta Var| +0.00904, ratio 2.05x, high better rate 0.8785
```

Rho-causal correlations:

```text
all routes:               rho vs |Delta H| = 0.8041; rho vs |Delta Var| = 0.7241
accessible routes only:   rho vs |Delta H| = 0.8389; rho vs |Delta Var| = 0.7373
rho vs top-1 changed:    -0.1427 overall
rho vs candidate-set KL:  0.0013 overall
```

Interpretation: the route-localization result is stronger than a pure descriptive geometry claim. In this battery, high-accessibility routes are better causal control points for uncertainty movement, while top-1 changes and candidate-set KL do not rise with `rho`. The high-rho advantage over low-rho/random routes is modest but consistent; the advantage over gradient-selected routes is larger.

## Added Application 5: Brittle Confidence

Primary artifacts:

- `applications/05_brittle_confidence/outputs/matched_high_confidence_groups.csv`
- `applications/05_brittle_confidence/outputs/brittle_perturbation_records.csv`
- `applications/05_brittle_confidence/outputs/brittle_summary.csv`
- `applications/05_brittle_confidence/outputs/brittle_matched_contrasts.csv`
- `applications/05_brittle_confidence/reports/report.md`

Protocol: build high-confidence matched groups controlling as much as the checked-in prompt pool allows for model, task, correctness, layer, subspace family, confidence, entropy, and margin; split into high-rho and low-rho groups; apply neutral prefixes/suffixes, quiz-format rewrites, template rewrites, and synonym substitutions; measure answer flips, probability drops, entropy increases, KL drift, top-10 Jaccard drop, correctness loss, and a composite fragility score.

Matched coverage:

```text
BERT:       46 matched high/low rows across factual_deep and factual_simple
DistilBERT: 34 matched high/low rows across factual_deep and factual_simple
Tiny BERT:  16 matched high/low rows across factual_deep and factual_simple
```

Low-rho minus high-rho matched contrasts:

```text
fragility score:     -0.0364, low-greater rate 0.5000
answer flip rate:    -0.0660, low-greater rate 0.3750
answer prob drop:    -0.0345, low-greater rate 0.4375
entropy increase:    -0.1255, low-greater rate 0.3542
candidate KL:        +0.0782, low-greater rate 0.5625
top-10 drop:         +0.0179, low-greater rate 0.5000
```

Interpretation: this application is a useful boundary case rather than a positive claim. The current matched perturbation battery does not support the hypothesis that high-confidence low-rho predictions are generally more brittle. Low-rho cases show slightly more KL and top-10 drift, but high-rho cases show more answer flips, probability drop, entropy increase, and overall fragility. This should stay in diagnostics unless a larger paraphrase dataset reverses the result.

## Added Application 6: Hidden Fragility In Confident Predictions

Primary artifacts:

- `applications/06_hidden_fragility_cifar_c/README.md`
- `applications/06_hidden_fragility_cifar_c/config/reproduce.json`
- `applications/06_hidden_fragility_cifar_c/reports/report.md`
- `applications/06_hidden_fragility_cifar_c/pilot_k2/outputs/predictor_benchmark.csv`
- `applications/06_hidden_fragility_cifar_c/pilot_k2/reports/report.md`
- `scripts/run_hidden_fragility_cifar_c.py`

Protocol: train or load a ResNet-18 CIFAR-10 classifier, compute `rho(B)` on the penultimate embedding using a PCA route, restrict to correctly classified high-confidence clean CIFAR-10 examples, greedily match high-rho and low-rho examples within class on confidence, entropy, margin, loss, entropy-gradient norm, and projected-gradient norm, then evaluate label-preserving CIFAR-10-C corruptions over severities 1-5.

Planned metrics:

```text
S_flip: minimum corruption severity that changes the clean prediction, or 6 if no flip
true-probability collapse: p_true(clean) - p_true(corrupted)
robust accuracy under CIFAR-10-C
entropy increase, margin drop, correctness loss
incremental AUROC/AUPRC and regression value of rho beyond confidence, entropy, margin, loss, and gradient norms
```

Status in this workspace:

```text
partial CPU pilot completed
checkpoint: applications/06_hidden_fragility_cifar_c/models/resnet18_cifar10_pilot_5ep_10k.pt
clean scored rows: 2000
matched low/high-rho pairs: 12
corruption records: 288
route: penultimate PCA with pca_dim=2
torchvision was not installed, so the runner includes a local CIFAR ResNet-18 and direct CIFAR batch loaders
```

Pilot result:

```text
any_flip AUROC:
scalar baseline      = 0.8472
gradient baseline    = 0.8681
gradient + rho       = 0.8611
rho only             = 0.6528

mean true-probability drop:
scalar baseline R2   = 0.2859
gradient baseline R2 = 0.3343
gradient + rho R2    = 0.4560
rho only R2          = 0.0615
```

Interpretation: this is a clean cross-domain protocol with a completed smoke test, not a completed paper result. The pilot model is undertrained and the matched set is small. It does not support a strong hidden-fragility claim yet, but it shows the pipeline works, that large PCA dimensions saturate the 10-class Fisher geometry, and that the full run should use `pca_dim` in `{1,2,4}` plus resumable training.

## Added Application 7: Safe Model Editing Diagnostic

Primary artifacts:

- `applications/07_safe_model_editing/outputs/edit_records.csv`
- `applications/07_safe_model_editing/outputs/side_effect_records.csv`
- `applications/07_safe_model_editing/outputs/edit_summary.csv`
- `applications/07_safe_model_editing/outputs/rho_quartile_summary.csv`
- `applications/07_safe_model_editing/outputs/predictor_benchmark.csv`
- `applications/07_safe_model_editing/outputs/qualitative_safe_edit_examples.csv`
- `applications/07_safe_model_editing/reports/report.md`

Protocol: for factual masked-LM prompts, choose a correction target when the model is wrong and a same-topic/top-candidate counterfactual when it is right. For each route, compute `rho`, construct the least-norm route direction that increases the edit target over the source token, binary-search the actual local edit cost, then apply the same hidden-state delta to unrelated prompts to measure side effects.

Coverage:

```text
models: distilbert-base-uncased; google/bert_uncased_L-2_H-128_A-2
prompt rows: 48
edit records: 240
side-effect records: 2880
top-k lens: 32
subspace k: 8
```

Key result:

```text
rho Spearman with lower actual edit cost:          0.0536
edit-gradient norm Spearman with lower cost:       0.2429
Fisher-output norm Spearman with lower cost:       0.3547
rho Spearman with safe-success:                    0.0798

cost model R2:
scalar baseline             0.1222
edit-gradient baseline      0.2275
edit-gradient + rho         0.2342
```

Interpretation: this is a useful diagnostic but not a positive model-editing claim. `rho` adds a small amount beyond scalar and edit-gradient controls, but target-specific edit gradients and Fisher-output norms are stronger cost predictors. The result should be framed as evidence that a serious safe-editing claim needs persistent weight-editing experiments or a target-specific accessibility metric, not as a headline result.

## Added Control F: Full-Vocabulary Tiny-Model Sanity

Primary artifacts:

- `experiments/controls/full_vocab_sanity/outputs/full_vocab_rho_scores.csv`
- `experiments/controls/full_vocab_sanity/outputs/topk_vs_full_vocab_summary.csv`
- `experiments/controls/full_vocab_sanity/reports/report.md`

The tiny-model run computes full-vocabulary accessibility via the Fisher-kernel identity:

```text
rho = u^T F C (C^T F C)^+ C^T F u / (u^T F u), where C = J B
```

Key result:

```text
k=16:  mean |rho_topk-rho_full| = 0.3793, Spearman = 0.3523
k=32:  mean |rho_topk-rho_full| = 0.1807, Spearman = 0.5066
k=64:  mean |rho_topk-rho_full| = 0.1174, Spearman = 0.4792
k=128: mean |rho_topk-rho_full| = 0.0915, Spearman = 0.4977
k=256: mean |rho_topk-rho_full| = 0.0829, Spearman = 0.4952
```

Interpretation: small top-k lenses overestimate absolute full-vocabulary accessibility, while larger top-k lenses move closer to full-vocabulary rho. This supports the top-k lens as a useful local approximation but not as an invariant substitute for full-vocabulary geometry.

## Added Control G: Out-of-Sample Route Generalization

Primary artifacts:

- `experiments/controls/out_of_sample_generalization/outputs/oos_scores.csv`
- `experiments/controls/out_of_sample_generalization/outputs/oos_steering_records.csv`
- `experiments/controls/out_of_sample_generalization/outputs/oos_contrasts.csv`
- `experiments/controls/out_of_sample_generalization/reports/report.md`

Protocol: estimate `state_pca` and `delta_pca` routes on train prompts, then evaluate accessibility and accessible steering on held-out prompts. Held-out oracle routes are computed as a diagnostic upper-bound, not as a deployable route.

Key held-out train-route versus held-out oracle result:

```text
BERT delta rho:      train route 0.5394 vs oracle 0.5380; |Delta H| 0.03714 vs 0.03699
BERT state rho:      train route 0.6246 vs oracle 0.6241; |Delta H| 0.04347 vs 0.04370
DistilBERT delta rho: train route 0.4807 vs oracle 0.4986; |Delta H| 0.02781 vs 0.02786
DistilBERT state rho: train route 0.5420 vs oracle 0.5504; |Delta H| 0.03334 vs 0.03373
Tiny delta rho:      train route 0.3742 vs oracle 0.4524; |Delta H| 0.02540 vs 0.02786
Tiny state rho:      train route 0.4203 vs oracle 0.3605; |Delta H| 0.02713 vs 0.02488
```

Random-route caveat:

```text
BERT state train route beats pooled random rho by +0.0342, but BERT delta is below pooled random by -0.0510.
DistilBERT state is near pooled random (+0.0122), while delta is below pooled random (-0.0490).
Tiny routes are mixed.
```

Interpretation: train-estimated routes transfer surprisingly well to held-out oracle routes for BERT and DistilBERT, especially for state routes. This supports route reusability, but random routes remain competitive in this small run, so the claim should remain preliminary rather than a strong global reusable-route theorem.

## Added Control H: Random-Init Versus Pretrained

Primary artifacts:

- `experiments/controls/random_init_vs_pretrained/outputs/random_init_scores.csv`
- `experiments/controls/random_init_vs_pretrained/outputs/pretrained_vs_random_contrasts.csv`
- `experiments/controls/random_init_vs_pretrained/reports/report.md`

Key result:

```text
BERT state_pca: pretrained rho 0.6367 vs random-init 0.2802, ratio 2.27, rank Spearman -0.105
BERT delta_pca: pretrained rho 0.4900 vs random-init 0.2937, ratio 1.67, rank Spearman -0.025
Tiny state_pca: pretrained rho 0.4169 vs random-init 0.2983, ratio 1.40, rank Spearman 0.219
Tiny delta_pca: pretrained rho 0.3858 vs random-init 0.2941, ratio 1.31, rank Spearman -0.655
```

Interpretation: random-init models retain nonzero accessibility, so architecture/Jacobian geometry contributes. But pretrained models have substantially higher accessibility and weak or unstable rank alignment with random-init models, supporting the narrower claim that accessible varentropy reflects learned representational organization in addition to architecture.

## Added Diagnostic: Prompt Audit And Failure Modes

Primary artifacts:

- `experiments/controls/statistical_diagnostics/outputs/prompt_duplicate_summary.csv`
- `experiments/controls/statistical_diagnostics/outputs/prompt_near_duplicates.csv`
- `experiments/controls/statistical_diagnostics/outputs/failure_modes.csv`
- `experiments/controls/statistical_diagnostics/outputs/bootstrap_residual_effects.csv`

Prompt audit:

```text
prompt rows audited = 1779
unique exact prompts = 342
unique normalized prompts = 342
near-duplicate pairs at threshold 0.92 = 103
```

Failure-mode table:

```text
rho unstable across top-k = 80
top-10 degradation cases = 40
top-1 changed cases = 22
high-rho / low-movement cases = 1
```

Residual diagnostic after scalar, gradient, and geometric controls:

```text
rho residual effect on |Delta H|   = 0.1366 [0.1207, 0.1522]
rho residual effect on |Delta Var| = 0.1444 [0.1286, 0.1609]
```

## Figures

```text
Fig. 1: reports/figures/fig01_conceptual_accessible_varentropy.svg
Fig. 2: reports/figures/fig02_scalar_uncertainty_vs_rho.svg
Fig. 3: reports/figures/fig03_scalar_matched_pairs.svg
Fig. 4: reports/figures/fig04_rho_vs_delta_controls.svg
Fig. 5: reports/figures/fig05_layerwise_heatmap.svg
Fig. 5b: reports/figures/fig05_compressibility_curves.svg
Fig. 6: reports/figures/fig06_steering_vs_controls_ci.svg
Fig. 6b: reports/figures/fig06_uncertainty_steering_main.svg
Fig. 7: reports/figures/fig07_topk_robustness.svg
Fig. 8: reports/figures/fig08_gradient_baselines.svg
```

Regenerate with:

```powershell
python scripts\make_paper_figures.py
```

## Minimal Reproduction Commands

Set the local package path:

```powershell
$env:PYTHONPATH='src'
```

Experiment 1 scalar diagnostics:

```powershell
python scripts\run_factual_error_deep_benchmark.py --model bert-base-uncased --out-dir results\factual_error_deep_bert_top32 --top-k 32 --random-seed 123
python scripts\run_factual_error_deep_benchmark.py --model distilbert-base-uncased --out-dir results\factual_error_deep_distilbert_top32 --top-k 32 --random-seed 123
python scripts\run_factual_error_deep_benchmark.py --model google/bert_uncased_L-2_H-128_A-2 --out-dir results\factual_error_deep_google_tiny_top32 --top-k 32 --random-seed 123
python scripts\analyze_geometric_diagnostics_from_outputs.py --out-dir results\geometric_diagnostics_existing
```

Experiment 2 local perturbation:

```powershell
python scripts\run_distilbert_geometry_interventions.py --out-dir results\distilbert_geometry_interventions --top-k 32 --random-seed 20260522
```

Experiment 3 layerwise and k structure:

```powershell
python scripts\run_layerwise_k_adjusted_rho.py --out-dir results\layerwise_k_adjusted_rho --top-k-output 32 --subspace-ks 1,2,4,8,16,32 --random-seed 20260523
python scripts\analyze_layerwise_k_compressibility.py --in-dir results\layerwise_k_adjusted_rho --out-dir results\layerwise_k_compressibility
python scripts\run_multiarch_layerwise_accessibility.py --out-dir results\multiarch_layerwise_accessibility --seed 7
```

Experiment 4 steering:

```powershell
python scripts\run_uncertainty_steering_full_battery.py --out-dir results\uncertainty_steering_full_battery_v1 --top-k 16 --seed 20260524
python scripts\run_uncertainty_steering.py --out-dir results\uncertainty_steering_distilbert_pca16 --top-k 32 --layer 5 --pca-dim 16 --seed 20260524
python scripts\run_decoder_uncertainty_steering.py --out-dir results\decoder_uncertainty_steering_qwen2_5_0_5b --max-prompts 64 --top-m 16 --layer 23 --pca-dim 16 --torch-dtype float32 --local-files-only --seed 20260524
```

Decoder-only main battery, comparator diagnostics, and local confidence-control application:

```powershell
python scripts\run_decoder_llm_main_and_comparators.py --local-files-only --trust-remote-code --max-prompts-per-task 3 --top-m 16 --pca-dim 8 --output-eps 0.05 --seed 20260528
```

Operational controllability tests:

```powershell
python scripts\run_control_cost_and_equal_output_tests.py --tau-entropy 0.02 --tau-varentropy 0.04 --top-k-output 32 --seed 20260529
```

Calibration-diagnosis application:

```powershell
python scripts\run_calibration_diagnosis.py --max-prompts-per-task 10 --top-k 32 --subspace-ks 8 --output-eps 0.05 --seed 20260532
```

Uncertainty-circuit and brittle-confidence applications:

```powershell
python scripts\run_brittle_confidence_and_circuit_applications.py --max-prompts-per-task 6 --top-k 32 --subspace-k 8 --output-eps 0.05 --seed 20260530
```

CIFAR-10/CIFAR-10-C hidden-fragility application:

```powershell
python scripts\run_hidden_fragility_cifar_c.py --cifar10-dir data\cifar-10-batches-py --cifar10c-dir data\CIFAR-10-C --checkpoint applications\06_hidden_fragility_cifar_c\models\resnet18_cifar10_full.pt --train-if-missing --epochs 40 --batch-size 256 --pca-dim 2 --confidence-quantile 0.70 --seed 20260531
```

Safe model-editing diagnostic:

```powershell
python scripts\run_safe_model_editing_application.py --models distilbert-base-uncased,google/bert_uncased_L-2_H-128_A-2 --max-prompts-per-task 8 --top-k 32 --subspace-k 8 --random-subspaces 1 --side-prompts-per-edit 12 --seed 20260601
```

Paper figures:

```powershell
python scripts\make_paper_figures.py
```

Extended robustness, gradient, regression, and preservation controls:

```powershell
python scripts\run_topk_gradient_regression_controls.py --top-k-values 16,32,64,128,256 --subspace-ks 8 --max-prompts-per-task 8 --random-subspaces 1 --output-eps 0.05 --seed 20260525
python scripts\make_paper_figures.py
```

Statistical diagnostics and full-vocabulary tiny-model sanity:

```powershell
python scripts\run_statistical_diagnostics.py --bootstrap 1000 --seed 20260526
python scripts\run_tiny_full_vocab_sanity.py --max-prompts-per-task 4 --top-k-values 16,32,64,128,256 --subspace-ks 8 --random-subspaces 1 --seed 20260526
```

Out-of-sample and random-init reviewer controls:

```powershell
python scripts\run_generalization_random_init_controls.py --top-k 32 --subspace-ks 8 --max-prompts-per-task 8 --random-subspaces 1 --output-eps 0.05 --seed 20260527
```

The commands regenerate raw outputs under `results/`. The checked-in paper-ready artifacts are curated copies under `experiments/`.

## Limitations

- The Fisher geometry is computed on top-k/top-m selected output sets, not the full vocabulary.
- Semantic uncertainty comparators are represented by token-level embedding/cluster proxies, not full generative Semantic Entropy, Semantic Density, or HaloScope implementations on their original hallucination-detection tasks.
- Decoder-only evidence is present in the main steering experiment on Qwen and Phi, but remains a local next-token logit-lens intervention.
- Direct projected gradients are stronger immediate predictors than rho for raw local movement; rho's stronger claim is residual geometric information after controls and decompositional interpretability.
- In the minimal intervention-energy test, `rho` predicts lower cost in the local top-k 32 setting, but projected gradients remain the strongest raw cost predictors and the full multi-epsilon grid is more mixed.
- The calibration-diagnosis application improves NLL/Brier/target probability through high-rho routes, but ECE worsens because the exact-token baseline is low-confidence and low-accuracy; do not frame it as a pure ECE win.
- The uncertainty-circuit application supports route localization, but high-rho improvements over low-rho/random routes are modest in this first battery and still use local logit-lens interventions.
- The brittle-confidence application is mixed/negative: low-rho high-confidence cases do not show uniformly higher fragility under the current template perturbations.
- The CIFAR-10/CIFAR-10-C hidden-fragility application has only a small CPU pilot so far. The pilot is mixed and underpowered; the full 40-epoch run is resumable but still pending.
- The safe model-editing diagnostic is local representation editing, not persistent weight editing. In the current run, `rho` is weak for edit cost while target-specific edit gradients and Fisher-output norms are stronger.
- Top-k robustness is good but not perfect; full-vocabulary tiny-model sanity shows that top-k can overestimate absolute rho at small k.
- Out-of-sample route transfer is encouraging but preliminary because pooled random routes remain competitive in this small run.
- Random-init controls show learned weights reshape accessibility, but nonzero random-init accessibility means architectural geometry is also part of the measurement.
- RoBERTa, Llama, and Mistral were unavailable locally and were not downloaded.
- The steering claim is local: it concerns hidden-state/logit-lens perturbations, not end-to-end generation behavior under arbitrary prompts.
