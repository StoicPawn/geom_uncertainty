# Accessible Varentropy

A clean GitHub project for the Accessible Varentropy paper. The repository is organized around four experiments, explicit controls, and one integrated final report.

## Paper Structure

The paper-facing material is in `paper/`:

1. Mathematical framework
2. Experiment 1: scalar uncertainty does not determine accessibility
3. Experiment 2: accessibility predicts local uncertainty movement
4. Experiment 3: layerwise and k-dimensional structure
5. Experiment 4: uncertainty steering
6. Ablations and boundary cases
7. Limitations

The integrated paper report is `reports/final_integrated_report.md`.

## GitHub Structure

```text
experiments/
  01_matched_scalar_uncertainty/
  02_local_perturbation_prediction/
  03_layerwise_k_structure/
  04_uncertainty_steering/
  controls/
    random_subspaces/
    euclidean_ablation/
    shuffled_surprisal/
    fisher_output_energy_control/
```

Each experiment contains `data/`, `outputs/`, `reports/`, and a local `README.md`.

## Where To Start

Read `reports/final_integrated_report.md` for the paper-level synthesis. Then inspect the experiment folders in order:

- `experiments/01_matched_scalar_uncertainty/`
- `experiments/02_local_perturbation_prediction/`
- `experiments/03_layerwise_k_structure/`
- `experiments/04_uncertainty_steering/`

The controls live separately under `experiments/controls/` so the main experimental narrative stays clean while the ablations remain auditable.

## Reproducibility

Core code lives in `src/accessible_varentropy`. Curated runner scripts live in `scripts/`. The old raw `results/` and `paper_artifacts/` trees have been consolidated into the experiment folders.
