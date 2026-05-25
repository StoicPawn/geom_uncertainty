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
  controls/
    topk_robustness/
    gradient_baselines/
    full_regression/
    semantic_preservation/
    random_subspaces/
    euclidean_ablation/
    shuffled_surprisal/
    fisher_output_energy_control/
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

Minimal full-run commands are listed in `reports/final_reproducibility_and_results_report.md`. Raw reruns write under `results/`; the paper-ready artifacts are curated into `experiments/`.
