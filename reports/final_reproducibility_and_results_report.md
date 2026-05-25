# Final Reproducibility And Results Report

This is the single paper-facing reproducibility report for the repository. It links each claim to the experiment, CSV artifacts, paper-ready figures, and minimal commands needed to regenerate the analysis.

## Repository Map

```text
experiments/
  01_matched_scalar_uncertainty/
  02_local_perturbation_prediction/
  03_layerwise_k_structure/
  04_uncertainty_steering/
  controls/
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

Unavailable or incomplete:

- RoBERTa was not available in the local Hugging Face cache. No network download was attempted for the consolidated run.

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

## Layers, Dimensions, Epsilons, And Subspaces

Layer coverage:

- Experiment 2: all DistilBERT hidden layers available to the intervention run.
- Experiment 3: DistilBERT layers `0..6`.
- Experiment 4: auto-selected layer thirds and final layer per model. The resulting matrix is in `experiments/04_uncertainty_steering/outputs/replication_matrix.csv`.

Subspace dimensions:

- Experiment 3: `k = 1, 2, 4, 8, 16, 32`.
- Experiment 4: `k = 1, 4, 8, 16`, filtered by hidden dimension and sample count.

Epsilon values:

- Experiment 2 latent perturbations: `epsilon = 0.25, 0.5, 1.0`.
- Experiment 4 latent-equal pilot/full controls: `epsilon = 0.05, 0.1`.
- Experiment 4 Fisher-output-equal controls: `epsilon = 0.02, 0.05, 0.1`.
- DistilBERT pilot steering: `epsilon = 0.1, 0.2, 0.4`.
- Decoder steering: `epsilon = 0.05, 0.1, 0.2`.

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
- `experiments/04_uncertainty_steering/figures/fig06_steering_vs_controls_ci.svg`
- `experiments/04_uncertainty_steering/figures/fig06_uncertainty_steering_main.svg`

Five checks:

1. Directional control: `+delta_acc` increases entropy and `-delta_acc` decreases entropy.
2. Equal Fisher-output-energy controls: comparisons are matched on `||F^{1/2} J delta z||`.
3. Specificity: uncertainty changes are tracked separately from top-1, correctness, semantic-proxy, and ranking shifts.
4. Rho dependency: rho predicts movement after scalar and geometric controls.
5. Replication: local MLMs, tasks, layers, dimensions, and random controls are included; decoder evidence is separate.

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
```

Specificity:

- At `epsilon=0.02`, mean top-1 change rate is `0.0142`; target-correctness change rate is `0.0003`.
- At `epsilon=0.05`, mean top-1 change rate is `0.0345`; target-correctness change rate is `0.0010`.

Interpretation: accessible steering changes uncertainty more efficiently than controls at equal Fisher-output energy and largely preserves monitored answer identity at small epsilons.

## Confirmatory Vs Exploratory Analyses

Confirmatory analyses are the claim-supporting tests used in the main paper narrative:

- scalar-matched uncertainty with different accessibility;
- accessibility predicting local movement;
- layerwise and k-dimensional structure;
- uncertainty steering against equal Fisher-output-energy controls;
- bootstrap CIs/effect sizes for the main steering and rho-quartile contrasts;
- same scalar uncertainty plus same projected-gradient magnitude but different `rho`;
- tiny-model full-vocabulary sanity check;
- train-route held-out prompt generalization;
- random-init versus pretrained learned-structure control;
- answer-preserving qualitative steering examples.

Exploratory and diagnostic analyses are used to bound the claim:

- direct projected-gradient baselines;
- residual `rho` effects after scalar, gradient, and geometric controls;
- top-k robustness beyond the main lens;
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
- Random subspaces test whether the result is a generic subspace effect.
- Euclidean ablation tests whether Fisher geometry is needed.
- Shuffled surprisal tests whether the true centered-surprisal direction is needed.
- Equal Fisher-output-energy controls test whether accessible steering wins after matching total output movement.
- Top-k robustness tests whether rho rankings, layer trends, and steering effects survive output-lens changes.
- Gradient baselines compare rho to `||Pi_B grad H||`, `||Pi_B grad Var||`, `||F^{1/2}JB||`, and `||JB||`.
- Full regression tests rho after scalar, gradient, geometric, model, layer, and prompt controls.
- Semantic preservation extends top-1 checks to top-5/top-10 Jaccard, KL, candidate-set mass, and embedding-cluster proxies.

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

## Added Control E: Decoder-only Main Evidence

Decoder-only evidence is now documented inside Experiment 4 rather than treated only as an external appendix:

- `experiments/04_uncertainty_steering/decoder_main/README.md`
- `experiments/04_uncertainty_steering/outputs/decoder_qwen_steering_summary.csv`
- `experiments/04_uncertainty_steering/outputs/decoder_qwen_steering_contrasts.csv`
- `experiments/04_uncertainty_steering/reports/decoder_qwen_report.md`

Interpretation: the decoder evidence is still smaller than the masked-LM full battery, but it is part of the main uncertainty-steering experiment and supports the claim that the geometry is not exclusive to masked-LM heads.

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
- Semantic uncertainty is represented by embedding/cluster proxies, not a full semantic-entropy estimator.
- Decoder-only evidence is present in the main steering experiment but remains smaller than the masked-LM battery.
- Direct projected gradients are stronger immediate predictors than rho for raw local movement; rho's stronger claim is residual geometric information after controls and decompositional interpretability.
- Top-k robustness is good but not perfect; full-vocabulary tiny-model sanity shows that top-k can overestimate absolute rho at small k.
- Out-of-sample route transfer is encouraging but preliminary because pooled random routes remain competitive in this small run.
- Random-init controls show learned weights reshape accessibility, but nonzero random-init accessibility means architectural geometry is also part of the measurement.
- RoBERTa was unavailable locally and was not downloaded.
- The steering claim is local: it concerns hidden-state/logit-lens perturbations, not end-to-end generation behavior under arbitrary prompts.
