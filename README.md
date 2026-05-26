# Geometry Of Uncertainty

Paper-facing repository for accessible varentropy: a Fisher-output geometric measure of where uncertainty can be controlled inside a network.

The main branch is intentionally narrow. It keeps the scientific spine, the required controls, and one diagnostic application that closes the story. Exploratory application tests live on the archive branch `archive_applications_exploratory`.

## Core Claim

For an internal route or subspace `B`, `rho(B)` measures the fraction of local varentropy direction that is accessible through `B` in Fisher-output geometry. It is not just entropy, confidence, margin, or gradient magnitude rewritten.

## Main Experiments

1. `experiments/01_matched_scalar_uncertainty/`
   Scalar uncertainty does not determine accessibility. Scalar-matched pairs can have sharply different rho.

2. `experiments/02_local_perturbation_prediction/`
   Accessibility predicts local uncertainty movement under small interventions.

3. `experiments/03_layerwise_k_structure/`
   Rho has layerwise and k-dimensional structure, including compressibility and heatmap evidence.

4. `experiments/04_uncertainty_steering/`
   High-accessibility directions support uncertainty steering, with equal-output-movement and minimal-energy controls.

5. `experiments/05_controllability_mapping/`
   Diagnostic mapping test: controls plus rho better predict safe uncertainty movement and minimal energy than controls alone.

## Required Controls

Curated controls are under `experiments/controls/`:

- `gradient_baselines/`
- `full_regression/`
- `semantic_preservation/`
- `topk_robustness/`
- `fisher_output_energy_control/`
- `statistical_diagnostics/`
- appendix-style ablations: `random_subspaces/`, `euclidean_ablation/`, `shuffled_surprisal/`, `full_vocab_sanity/`, `out_of_sample_generalization/`, `random_init_vs_pretrained/`, `external_uncertainty_comparators/`

## Paper Files

- `paper/`: paper section drafts.
- `CLAIM_MAP.md`: claim-to-artifact map.
- `reports/final_reproducibility_and_results_report.md`: integrated results and limitations.
- `reports/figure_index.md`: canonical figure list.
- `models/model_inventory.md`: model coverage.
- `configs/paper_runs.json`: central run index.

## Reproduction

Regenerate paper figures:

```bash
python scripts/make_paper_figures.py
```

Run the main steering controls:

```bash
python scripts/run_control_cost_and_equal_output_tests.py --tau-entropy 0.02 --tau-varentropy 0.04 --top-k-output 32 --seed 20260529
```

Run the diagnostic controllability mapping:

```bash
python scripts/run_uncertainty_controllability_mapping_test.py --out-dir experiments/05_controllability_mapping --decoder-models distilgpt2,Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B-Instruct,microsoft/phi-2 --max-items 18 --top-m 8 --layers auto3 --subspace-dims 1,2,4 --eps 0.01,0.025,0.05,0.1,0.2 --movement-threshold 0.01 --drift-cap 0.00025 --top-k 3 --bootstrap 300 --torch-dtype float16 --device cuda --trust-remote-code --seed 20260608
```

The paper-ready artifacts are checked in under `experiments/`; raw exploratory results are archived off main.
