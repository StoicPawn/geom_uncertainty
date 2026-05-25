# Control: Full Regression

This control tests whether `rho` remains informative after scalar, gradient, geometric, model, layer, and prompt controls.

## Regression Results

```text
             outcome     n  n_parameters  rho_beta_standardized  rho_t_stat  rho_p_value       r2                                                                                                                                                                                                   controls
   abs_delta_entropy 14490            40               0.003433   19.001116 1.553694e-79 0.755310 entropy,varentropy,confidence,margin,jacobian_fro_norm,fisher_output_energy,grad_entropy_proj_norm,grad_varentropy_proj_norm,top_k_output,layer,model,task,topic,subspace_family,subspace_k,direction,sign
abs_delta_varentropy 14490            40               0.007226   17.711758 1.842447e-69 0.707135 entropy,varentropy,confidence,margin,jacobian_fro_norm,fisher_output_energy,grad_entropy_proj_norm,grad_varentropy_proj_norm,top_k_output,layer,model,task,topic,subspace_family,subspace_k,direction,sign
 directional_success 14490            40               0.004194    0.800335 4.235301e-01 0.422862 entropy,varentropy,confidence,margin,jacobian_fro_norm,fisher_output_energy,grad_entropy_proj_norm,grad_varentropy_proj_norm,top_k_output,layer,model,task,topic,subspace_family,subspace_k,direction,sign
```

## Files

```text
full_regression_ols.csv
report.md
```