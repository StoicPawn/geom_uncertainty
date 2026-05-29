# Claim Map

This map links the main paper claims to the maintained paper-spine artifacts.

| Claim | Location | Primary artifacts |
|---|---|---|
| Core definition: `rho(B)` is the Fisher-output accessible fraction of the local varentropy direction through an internal route/subspace `B`. | `paper/01_mathematical_framework.md`; `src/accessible_varentropy/` | Mathematical framework and core implementation |
| Non-redundancy: scalar uncertainty does not determine accessibility. | `experiments/01_matched_scalar_uncertainty/`; `experiments/controls/statistical_diagnostics/` | Scalar-matched pairs; same scalar/gradient diagnostics; scalar uncertainty vs rho figures |
| Local validity: accessibility predicts local uncertainty movement. | `experiments/02_local_perturbation_prediction/` | `outputs/intervention_by_rho_quartile.csv`; `outputs/intervention_residual_correlations.csv`; `outputs/intervention_records.csv`; movement figures |
| Intervention validity: high-rho routes move uncertainty more than low-rho, random, and equal-output controls at matched Fisher-output movement. | `experiments/04_uncertainty_steering/`; `experiments/04_uncertainty_steering/02_equal_output_movement/`; `experiments/04_uncertainty_steering/03_choose_route_intervention/`; `experiments/controls/fisher_output_energy_control/` | Equal-output contrasts; choose-route summary/contrasts/bootstrap CI; preservation summaries |
| Functional value: rho improves budgeted controllability-aware route allocation under fixed intervention budget and matched Fisher-output energy. | `experiments/05_controllability_mapping/budgeted_allocation_policy/` | `budgeted_allocation_summary.csv`; `budgeted_allocation_fold_summary.csv`; `budgeted_allocation_permutation_fdr.csv`; `budgeted_allocation_selected_records_sample.csv`; `report.md`; `config/reproduce.json` |

Appendix controls and boundary experiments are listed in `APPENDIX_ARTIFACTS.md`.
