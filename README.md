# Geometry Of Uncertainty

A paper-ready repository for the accessible-varentropy / uncertainty-geometry experiments. The project is organized around four main experiments, explicit controls, reproducibility configs, checked-in CSV outputs, and paper-ready figures.

## Claim-First Map

Start here:

- `CLAIM_MAP.md`: paper claim -> experiment -> CSV -> figure -> config.
- `reports/final_reproducibility_and_results_report.md`: complete methods/results/reproducibility report.
- `reports/figure_index.md`: canonical figure list.
- `models/model_inventory.md`: model coverage and status.
- `configs/paper_runs.json`: central config index.

## Paper Structure

The paper-facing material is in `paper/`:

1. Mathematical framework
2. Experiment 1: scalar uncertainty does not determine accessibility
3. Experiment 2: accessibility predicts local uncertainty movement
4. Experiment 3: layerwise and k-dimensional structure
5. Experiment 4: uncertainty steering
6. Ablations and boundary cases
7. Limitations

The integrated reproducibility and results report is `reports/final_reproducibility_and_results_report.md`.

## GitHub Structure

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
    statistical_diagnostics/
    full_vocab_sanity/
    out_of_sample_generalization/
    random_init_vs_pretrained/
    external_uncertainty_comparators/
    topk_robustness/
    gradient_baselines/
    full_regression/
    semantic_preservation/
    random_subspaces/
    euclidean_ablation/
    shuffled_surprisal/
    fisher_output_energy_control/
applications/
  03_calibration_diagnosis/
  local_confidence_control/
  04_uncertainty_circuits/
  05_brittle_confidence/
  06_hidden_fragility_cifar_c/
  07_safe_model_editing/
models/
configs/
reports/
scripts/
src/
```

Each experiment contains:

- `README.md`: short purpose and navigation.
- `config/reproduce.json`: minimal reproduction metadata and commands.
- `data/`: prompt tables or source data.
- `outputs/`: clean CSV outputs.
- `figures/`: paper-ready SVG/PNG figures.
- `reports/`: short summary plus source reports.

## Where To Start

Read `reports/final_reproducibility_and_results_report.md` for the paper-level synthesis. Then inspect `CLAIM_MAP.md` to connect claims to the exact CSVs and figures.

## Reproducibility

Core code lives in `src/accessible_varentropy`. Curated runner scripts live in `scripts/`.

Regenerate paper figures from checked-in CSV outputs:

```powershell
python scripts\make_paper_figures.py
```

Run the extended robustness/control suite:

```powershell
python scripts\run_topk_gradient_regression_controls.py --top-k-values 16,32,64,128,256 --subspace-ks 8 --max-prompts-per-task 8 --random-subspaces 1 --output-eps 0.05 --seed 20260525
```

Run the statistical/diagnostic package and tiny-model full-vocabulary sanity check:

```powershell
python scripts\run_statistical_diagnostics.py --bootstrap 1000 --seed 20260526
python scripts\run_tiny_full_vocab_sanity.py --max-prompts-per-task 4 --top-k-values 16,32,64,128,256 --subspace-ks 8 --random-subspaces 1 --seed 20260526
```

Run the out-of-sample and random-init reviewer controls:

```powershell
python scripts\run_generalization_random_init_controls.py --top-k 32 --subspace-ks 8 --max-prompts-per-task 8 --random-subspaces 1 --output-eps 0.05 --seed 20260527
```

Run the decoder-only main battery, external uncertainty comparators, and local confidence-control application:

```powershell
python scripts\run_decoder_llm_main_and_comparators.py --local-files-only --trust-remote-code --max-prompts-per-task 3 --top-m 16 --pca-dim 8 --output-eps 0.05 --seed 20260528
```

Run the operational controllability tests:

```powershell
python scripts\run_control_cost_and_equal_output_tests.py --tau-entropy 0.02 --tau-varentropy 0.04 --top-k-output 32 --seed 20260529
```

Run the calibration-diagnosis application:

```powershell
python scripts\run_calibration_diagnosis.py --max-prompts-per-task 10 --top-k 32 --subspace-ks 8 --output-eps 0.05 --seed 20260532
```

Run the route-interpretability and brittle-confidence applications:

```powershell
python scripts\run_brittle_confidence_and_circuit_applications.py --max-prompts-per-task 6 --top-k 32 --subspace-k 8 --output-eps 0.05 --seed 20260530
```

Run the CIFAR-10/CIFAR-10-C hidden-fragility application after placing local data. The runner saves a resumable checkpoint after every epoch:

```powershell
python scripts\run_hidden_fragility_cifar_c.py --cifar10-dir data\cifar-10-batches-py --cifar10c-dir data\CIFAR-10-C --checkpoint applications\06_hidden_fragility_cifar_c\models\resnet18_cifar10_full.pt --train-if-missing --epochs 40 --batch-size 256 --pca-dim 2 --confidence-quantile 0.70 --seed 20260531
```

Run the safe model-editing diagnostic:

```powershell
python scripts\run_safe_model_editing_application.py --models distilbert-base-uncased,google/bert_uncased_L-2_H-128_A-2 --max-prompts-per-task 8 --top-k 32 --subspace-k 8 --random-subspaces 1 --side-prompts-per-edit 12 --seed 20260601
```

Minimal full-run commands are listed in `reports/final_reproducibility_and_results_report.md`. Raw reruns write under `results/`; the paper-ready artifacts are curated into `experiments/`.
