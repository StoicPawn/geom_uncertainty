# Mathematical Framework

The central quantity is the Fisher-geometric accessibility coefficient:

```text
rho(B) = ||P_Im(F^{1/2} J B) F^{1/2}u||^2 / ||F^{1/2}u||^2.
```

Here `u = -log p - H(p)` is centered surprisal, `F = diag(p) - pp^T` is the categorical Fisher metric, `J` is the local logit Jacobian, and `B` is the latent subspace.

The decomposition is:

```text
Var = V_access + V_inaccess.
```

The experiments test whether `V_access`, not scalar uncertainty alone, predicts controllable local uncertainty movement.
