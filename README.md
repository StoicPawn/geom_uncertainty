# Geometry Of Uncertainty

Paper-facing repository for accessible varentropy: a Fisher-output geometric measure of where uncertainty can be controlled inside a network.

The main branch is intentionally narrow. It keeps the disruptive scientific spine plus the statistical robustness controls needed to defend it. Exploratory and side-branch experiments live on `archive_disruptive_experiments_full`.

## Core Claim

For an internal route or subspace `B`, `rho(B)` measures the fraction of local varentropy direction that is accessible through `B` in Fisher-output geometry. It is not just entropy, confidence, margin, or gradient magnitude rewritten.

## Main Experiments

1. `experiments/01_matched_scalar_uncertainty/`
   Scalar uncertainty does not determine accessibility. Scalar-matched pairs can have sharply different rho.

2. `experiments/02_local_perturbation_prediction/`
   Accessibility predicts local uncertainty movement under small interventions.

3. `experiments/04_uncertainty_steering/`
   High-accessibility directions support uncertainty steering, with equal-output-movement, minimal-energy, and choose-the-route intervention controls.

4. `experiments/05_controllability_mapping/`
   Diagnostic mapping test: controls plus rho better predict safe uncertainty movement and minimal energy than controls alone.

5. `experiments/controls/rho_guided_selective_reliability/`
   Non-oracle selective reliability test: baseline+rho improves risk-coverage/calibration tradeoffs.

6. `experiments/06_non_nlp_safety_policies/`
   Non-NLP safety-policy tests on medical diagnostic classification and vision perception surrogates.

## Required Controls

Curated controls are under `experiments/controls/`:

- `gradient_baselines/`
- `full_regression/`
- `semantic_preservation/`
- `topk_robustness/`
- `scale_external_robustness/`
- `rho_guided_selective_reliability/`
- `fisher_output_energy_control/`
- `statistical_diagnostics/`

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

Run the choose-the-route intervention test:

```bash
python scripts/run_choose_route_intervention_test.py --bootstrap 1000 --seed 20260610
```

Run the diagnostic controllability mapping:

```bash
python scripts/run_uncertainty_controllability_mapping_test.py --out-dir experiments/05_controllability_mapping --decoder-models distilgpt2,Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B-Instruct,microsoft/phi-2,meta-llama/Llama-3.2-1B,mistralai/Mistral-7B-v0.1 --masked-models distilbert-base-uncased,bert-base-uncased,roberta-base,prajjwal1/bert-tiny --max-items 18 --top-m 8 --layers auto3 --subspace-dims 1,2,4 --eps 0.01,0.025,0.05,0.1,0.2 --movement-threshold 0.01 --drift-cap 0.00025 --top-k 3 --bootstrap 300 --torch-dtype float16 --device cuda --trust-remote-code --seed 20260608
```

Aggregate the broad robustness matrix:

```bash
python scripts/run_scale_external_robustness_matrix.py
```

Run rho-guided selective reliability:

```bash
python scripts/run_rho_guided_selective_reliability.py --bootstrap 1000 --seed 20260609
```

Run non-NLP rho-guided safety policies:

```bash
conda run -n arca python scripts/run_non_nlp_rho_safety_tests.py --bootstrap 1000 --seed 20260611 --device cuda --fixed-review-costs 0.17,0.19
```

The paper-ready artifacts are checked in under `experiments/`; archived experiments are on `archive_disruptive_experiments_full`.
