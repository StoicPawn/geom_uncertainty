from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from accessible_varentropy.metrics import fisher_scores_for_sample


def gram_accessible(probs: np.ndarray, jacobian: np.ndarray) -> float:
    p = probs / probs.sum()
    log_p = np.log(p)
    entropy = -np.sum(p * log_p)
    u = -log_p - entropy
    fisher = np.diag(p) - np.outer(p, p)
    fisher_u = fisher @ u
    gram = jacobian.T @ fisher @ jacobian
    rhs = jacobian.T @ fisher_u
    return float(rhs @ np.linalg.pinv(gram, hermitian=True, rcond=1e-10) @ rhs)


def main() -> None:
    rng = np.random.default_rng(123)
    max_error = 0.0
    for classes in [3, 5, 10]:
        for latent_dim in [1, 2, 4, 16, 64]:
            for _ in range(100):
                raw = rng.lognormal(size=classes)
                probs = raw / raw.sum()
                jacobian = rng.normal(size=(classes, latent_dim))
                svd_value = fisher_scores_for_sample(probs, jacobian).accessible_varentropy
                gram_value = gram_accessible(probs, jacobian)
                max_error = max(max_error, abs(svd_value - gram_value))

    print(f"max_abs_error={max_error:.3e}")
    if max_error > 1e-8:
        raise SystemExit("projection identity check failed")


if __name__ == "__main__":
    main()
