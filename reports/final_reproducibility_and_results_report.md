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

- `Qwen/Qwen2.5-0.5B`, included as a separate logit-lens steering run.

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
- `experiments/01_matched_scalar_uncertainty/figures/fig02_scalar_matched_pairs.svg`

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
- `experiments/03_layerwise_k_structure/figures/fig04_layerwise_heatmap.svg`
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

## Controls, Ablations, And Boundary Cases

Control folders:

- `experiments/controls/random_subspaces/`
- `experiments/controls/euclidean_ablation/`
- `experiments/controls/shuffled_surprisal/`
- `experiments/controls/fisher_output_energy_control/`

What they address:

- Random subspaces test whether the result is a generic subspace effect.
- Euclidean ablation tests whether Fisher geometry is needed.
- Shuffled surprisal tests whether the true centered-surprisal direction is needed.
- Equal Fisher-output-energy controls test whether accessible steering wins after matching total output movement.

## Figures

```text
Fig. 1: reports/figures/fig01_conceptual_accessible_varentropy.svg
Fig. 2: reports/figures/fig02_scalar_matched_pairs.svg
Fig. 3: reports/figures/fig03_accessibility_predicts_movement.svg
Fig. 4: reports/figures/fig04_layerwise_heatmap.svg
Fig. 5: reports/figures/fig05_compressibility_curves.svg
Fig. 6: reports/figures/fig06_uncertainty_steering_main.svg
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

The commands regenerate raw outputs under `results/`. The checked-in paper-ready artifacts are curated copies under `experiments/`.

## Limitations

- The Fisher geometry is computed on top-k/top-m selected output sets, not the full vocabulary.
- Semantic uncertainty is represented by embedding/cluster proxies, not a full semantic-entropy estimator.
- Decoder-only evidence is present but separate from the masked-LM full battery.
- RoBERTa was unavailable locally and was not downloaded.
- The steering claim is local: it concerns hidden-state/logit-lens perturbations, not end-to-end generation behavior under arbitrary prompts.
