# Ablations And Boundary Cases

The main branch keeps controls that directly support the paper claims and moves exploratory applications to `archive_applications_exploratory`.

Core boundary cases:

- projected-gradient baselines are strong raw predictors of movement;
- top-k geometry is a local approximation, not a full-vocabulary identity;
- train-estimated routes transfer imperfectly to held-out prompts;
- external semantic uncertainty metrics measure different objects;
- high rho does not guarantee safe movement at every epsilon or layer;
- minimal-energy results are strongest in the top-k local-linear setting.

Appendix ablations cover random subspaces, Euclidean projection, shuffled surprisal, full-vocabulary sanity checks, out-of-sample route reuse, random initialization, and external uncertainty comparators.
