# Choose-The-Route Intervention

Derived paper-facing test for the decision question: if several local routes are available, does choosing the high-rho route move uncertainty more efficiently?

The test reuses checked-in equal-Fisher-output and minimal-energy intervention records. It compares high-rho, low-rho, gradient-selected, deterministic random, gradient-orthogonal, and equal-output-energy controls without using correctness labels to choose routes.

Run:

```bash
python scripts/run_choose_route_intervention_test.py --bootstrap 1000 --seed 20260610
```
