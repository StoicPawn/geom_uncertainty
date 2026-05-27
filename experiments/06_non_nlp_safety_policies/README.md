# Non-NLP Rho-Guided Safety Policies

This experiment runs two real, non-NLP decision-policy tests:

- `medical_breast_cancer`: Wisconsin breast-cancer diagnostic classification.
- `vision_digits_perception`: sklearn handwritten-digits image classification as a compact vision-perception surrogate.

Each fold trains a small MLP from scratch, fits hidden-state PCA routes on the training split, calibrates entropy/gradient/Jacobian/rho thresholds on the training split, and evaluates held-out policy outcomes. Correctness labels are not used to choose defer/refine actions. The report also includes equal-review-cost selectors at 17% and 19%.

Reproduce:

```bash
conda run -n arca python scripts/run_non_nlp_rho_safety_tests.py --bootstrap 1000 --seed 20260611 --device cuda --fixed-review-costs 0.17,0.19
```

Primary report: `reports/report.md`.
