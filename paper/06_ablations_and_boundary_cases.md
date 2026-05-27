# Ablations And Boundary Cases

The main branch keeps controls that directly support the paper claims and moves exploratory side experiments to `archive_disruptive_experiments_full`.

Core boundary cases:

- projected-gradient baselines are strong raw predictors of movement;
- top-k geometry is a local approximation, not a full-vocabulary identity;
- selective reliability improves only over part of the risk-coverage curve;
- high rho does not guarantee safe movement at every epsilon or layer;
- minimal-energy results are strongest in the top-k local-linear setting.

Archived ablations cover random subspaces, Euclidean projection, shuffled surprisal, full-vocabulary sanity checks, out-of-sample route reuse, random initialization, and external uncertainty comparators.
