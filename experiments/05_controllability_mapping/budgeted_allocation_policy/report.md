# Budgeted Controllability Allocation Test

This test asks whether rho has decision value under a fixed intervention budget: each policy may select only a fraction of held-out candidate routes, and all candidates come from the `same_fisher_output_energy` intervention panel unless otherwise configured.

No learned selector is fit on its evaluation fold. Learned controls use only training-fold pre-intervention features and the training-fold observed safe-movement labels; held-out scoring uses pre-intervention features only. The oracle is reported only as an upper bound.

Rows evaluated: 7290. Budget mode: `same_fisher_output_energy`. Drift cap: 0.00025. Budgets: 0.05, 0.1, 0.2, 0.3.

## Main Grouped-Prompt Panel
```text
Policy                         SafeMove@10%   Move/Drift   Regret   TopK
random                            0.094143       20.170 0.905857 0.9506
highest_entropy                   0.042361       15.152 0.957639 0.9621
highest_gradient_norm             0.149088       15.103 0.850912 0.8994
highest_fisher_output_energy      0.127286       27.858 0.872714 0.9798
controls_only_learned             0.173173       21.332 0.826827 0.8921
rho_only                          0.161079       17.093 0.838921 0.9224
controls_plus_rho_learned         0.186064       18.383 0.813936 0.8820
gradient_plus_rho_learned         0.153076       16.599 0.846924 0.8964
oracle                            1.000000      196.831 0.000000 1.0000
```

## Key Paired Cluster Bootstrap Contrasts
```text
Contrast                                           Metric              Delta      CI low     CI high   p        q
controls_plus_rho_learned_vs_controls_only_learned safe_movement_rate  0.012892 -0.010713  0.041709  0.01000  0.01671
controls_plus_rho_learned_vs_controls_only_learned movement_per_drift -2.949550 -15.734814  1.724289  0.03459  0.04999
controls_plus_rho_learned_vs_controls_only_learned regret_vs_oracle   -0.012892 -0.041264  0.009877  0.01140  0.01877
gradient_plus_rho_learned_vs_gradient_only_learned safe_movement_rate  0.003988 -0.020778  0.034450  0.00380  0.00686
gradient_plus_rho_learned_vs_gradient_only_learned movement_per_drift  1.495980 -1.600854  3.604597  0.00040  0.00090
gradient_plus_rho_learned_vs_gradient_only_learned regret_vs_oracle   -0.003988 -0.034216  0.025490  0.00520  0.00917
rho_only_vs_within_example_shuffled_rho            safe_movement_rate  0.021633 -0.002423  0.048388  0.03299  0.04830
rho_only_vs_within_example_shuffled_rho            movement_per_drift -0.863873 -13.328525  6.030900  0.03399  0.04944
rho_only_vs_within_example_shuffled_rho            regret_vs_oracle   -0.021633 -0.047774  0.001299  0.04279  0.05991
```

## SafeMove Budget Sensitivity
```text
Budget  Contrast                                           Delta      CI low     CI high
  0.05  controls_plus_rho_learned_vs_controls_only_learned  0.007809 -0.023019  0.050189
  0.05  gradient_plus_rho_learned_vs_gradient_only_learned  0.001224 -0.036497  0.037056
  0.05  rho_only_vs_within_example_shuffled_rho             0.027279 -0.003314  0.067471
  0.05  rho_only_vs_random                                  0.041836 -0.005946  0.084650
  0.05  rho_only_vs_highest_entropy                         0.120619  0.057918  0.163017
  0.05  rho_only_vs_highest_fisher_output_energy            0.024214 -0.048108  0.080988
  0.10  controls_plus_rho_learned_vs_controls_only_learned  0.012892 -0.010713  0.041709
  0.10  gradient_plus_rho_learned_vs_gradient_only_learned  0.003988 -0.020778  0.034450
  0.10  rho_only_vs_within_example_shuffled_rho             0.021633 -0.002423  0.048388
  0.10  rho_only_vs_random                                  0.066936  0.036170  0.094961
  0.10  rho_only_vs_highest_entropy                         0.118719  0.081021  0.145562
  0.10  rho_only_vs_highest_fisher_output_energy            0.033793 -0.013017  0.073799
  0.20  controls_plus_rho_learned_vs_controls_only_learned  0.014195  0.005509  0.026392
  0.20  gradient_plus_rho_learned_vs_gradient_only_learned  0.027604  0.011319  0.055714
  0.20  rho_only_vs_within_example_shuffled_rho             0.034033  0.020348  0.051609
  0.20  rho_only_vs_random                                  0.048560  0.025288  0.070638
  0.20  rho_only_vs_highest_entropy                         0.099102  0.070566  0.122383
  0.20  rho_only_vs_highest_fisher_output_energy            0.032847  0.003804  0.060665
  0.30  controls_plus_rho_learned_vs_controls_only_learned  0.015679  0.008587  0.023878
  0.30  gradient_plus_rho_learned_vs_gradient_only_learned  0.028781  0.012638  0.045962
  0.30  rho_only_vs_within_example_shuffled_rho             0.041293  0.028425  0.056738
  0.30  rho_only_vs_random                                  0.047794  0.033105  0.065913
  0.30  rho_only_vs_highest_entropy                         0.095627  0.073939  0.116617
  0.30  rho_only_vs_highest_fisher_output_energy            0.038467  0.018210  0.056895
```

## Negative Controls

The test includes global shuffled rho, within-example shuffled rho, and rho shuffled within entropy, varentropy, gradient, and Fisher-energy bins. The within-example shuffle is the strictest negative control because it permutes rho only among candidate routes belonging to the same example/intervention setting.

## Files

Committed:
- `budgeted_allocation_selected_records_sample.csv`
- `budgeted_allocation_fold_summary.csv`
- `budgeted_allocation_summary.csv`
- `budgeted_allocation_contrasts.csv`
- `budgeted_allocation_bootstrap_ci.csv`
- `budgeted_allocation_permutation_fdr.csv`
- `report.md`
- `config/reproduce.json`

Generated but omitted from git due to GitHub file-size limits:
- `budgeted_allocation_selected_records.csv`

The sample file contains 5,000 deterministic inspection rows plus header. It is not used to compute the reported aggregate metrics; those are computed from the full generated records.
