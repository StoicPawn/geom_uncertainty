# Rho Route-Selection Policy On Controllability Mapping

This derived test asks whether `rho` selects the internal route that should be used for a local uncertainty intervention.
Each case has multiple candidate layer/route/subspace interventions. Policies choose a route using only pre-intervention geometry; `safe_movement` is used only for evaluation.

## Verdict

High-rho route selection strongly beats low-rho and deterministic random routes for safe movement and uncertainty movement. It is competitive with high-gradient and high-Jacobian selectors, but it does not uniformly dominate high-Fisher-energy route selection. This supports rho as an intervention-route feature, not as a universal replacement for all local geometric selectors.

## Key Results

```text
safe_movement: high-rho 0.095238 vs random 0.047619 vs low-rho 0.004762
safe_movement strong controls: high-gradient 0.084127, high-Fisher-energy 0.104762
mean uncertainty movement: high-rho 0.021195 vs random 0.019687 vs low-rho 0.010965
top1 preservation: high-rho 0.993651 vs random 0.980952 vs low-rho 0.958730
```

## Files
```text
route_selection_records.csv
route_selection_summary.csv
route_selection_contrasts.csv
route_selection_bootstrap_ci.csv
report.md
```
