# Geometry Of Uncertainty

Paper-facing repository for accessible varentropy: a Fisher-output geometric measure of where uncertainty can be controlled inside a network.

## Core Claim

For an internal route or subspace `B`, `rho(B)` measures the fraction of the local varentropy direction that is accessible through `B` in Fisher-output geometry.

## Main Paper Spine

1. `experiments/01_matched_scalar_uncertainty/`
   Non-redundancy: scalar uncertainty does not determine accessibility.

2. `experiments/02_local_perturbation_prediction/`
   Local validity: accessibility predicts local uncertainty movement under small interventions.

3. `experiments/04_uncertainty_steering/`
   Intervention validity: high-accessibility routes move uncertainty at matched Fisher-output movement. The paper-facing emphasis is equal-output steering and choose-the-route intervention.

4. `experiments/05_controllability_mapping/budgeted_allocation_policy/`
   Functional value: rho improves budgeted controllability-aware route allocation under fixed intervention budget and matched Fisher-output energy.

## Main Results Snapshot

- Scalar uncertainty is not enough: matched scalar-uncertainty cases can have substantially different rho values.
- Local perturbations: higher-rho quartiles show larger local entropy and varentropy movement under controlled interventions.
- Route intervention: high-rho routes move uncertainty more than low-rho and random routes at matched Fisher-output movement.
- Budgeted allocation: at 20% budget, controls+rho improves SafeMove over controls-only by +0.014195, gradient+rho improves over gradient-only by +0.027604, and rho-only beats within-example shuffled rho by +0.034033.

## Appendix Controls

Curated controls and supporting artifacts are documented in `APPENDIX_ARTIFACTS.md`. These controls support the paper but should not be framed as independent main claims.

## Paper Files

- `paper/`: paper section drafts.
- `CLAIM_MAP.md`: main claim-to-artifact map.
- `APPENDIX_ARTIFACTS.md`: appendix and boundary artifact map.
- `reports/final_reproducibility_and_results_report.md`: integrated results and limitations.
- `reports/figure_index.md`: canonical figure list.
- `models/model_inventory.md`: model coverage.
- `configs/paper_runs.json`: central run index.

## Reproduction

```bash
python scripts/make_paper_figures.py
python scripts/run_control_cost_and_equal_output_tests.py --tau-entropy 0.02 --tau-varentropy 0.04 --top-k-output 32 --seed 20260529
python scripts/run_choose_route_intervention_test.py --bootstrap 1000 --seed 20260610
python scripts/run_budgeted_controllability_allocation_test.py --bootstrap 1000 --permutations 5000 --seed 20260613
```

## Branch Policy

- `main`: paper-facing spine plus required appendix controls.
- `experiments-full`: complete snapshot of all experiments and side explorations.
- `paper-main-clean`: proposed cleaned paper-facing branch before merging into `main`.
