# Figure Index

The canonical paper-ready figure set is in `reports/figures/`. SVG files are preferred for the paper; PNG files are included for quick inspection.

| Figure | Narrative role | Source script | Primary data |
|---|---|---|---|
| `fig01_conceptual_accessible_varentropy.svg` | Conceptual map from `p` to `u`, `F^{1/2}u`, projection, `V_access`, `V_inaccess` | `scripts/make_paper_figures.py` | mathematical schematic |
| `fig02_scalar_uncertainty_vs_rho.svg` | Scatter of entropy/varentropy against rho | `scripts/make_paper_figures.py` | Experiment 4 subspace scores |
| `fig03_scalar_matched_pairs.svg` | Scalar-matched examples with different accessibility | `scripts/make_paper_figures.py` | Experiment 1 matched-pair CSVs |
| `fig04_rho_vs_delta_controls.svg` | Rho versus `|Delta H|` and `|Delta Var|`, split by controls | `scripts/make_paper_figures.py` | Top-k robustness steering records |
| `fig05_layerwise_heatmap.svg` | Layer by k heatmap of adjusted accessibility | `scripts/make_paper_figures.py` | Experiment 3 layerwise summary |
| `fig05_compressibility_curves.svg` | Accessibility accumulation as k grows | `scripts/make_paper_figures.py` | Experiment 3 layerwise summary |
| `fig06_steering_vs_controls_ci.svg` | Steering versus controls with 95% confidence intervals | `scripts/make_paper_figures.py` | Top-k robustness steering records |
| `fig06_uncertainty_steering_main.svg` | Supplementary equal-energy dose response and preservation panel | `scripts/make_paper_figures.py` | Experiment 4 steering summaries and records |
| `fig07_topk_robustness.svg` | Top-k robustness for rho ranking, steering movement, and top-10 preservation | `scripts/make_paper_figures.py` | Top-k robustness outputs |
| `fig08_gradient_baselines.svg` | Direct gradient/Jacobian baselines against rho | `scripts/make_paper_figures.py` | Gradient baseline correlations |
