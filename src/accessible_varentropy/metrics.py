from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FisherScores:
    entropy: float
    varentropy: float
    accessible_varentropy: float
    rho: float
    grad_norm_sq: float
    trace_fim: float


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def fisher_scores_for_sample(
    probs: np.ndarray,
    jacobian: np.ndarray,
    *,
    eps: float = 1e-12,
) -> FisherScores:
    """Compute entropy, varentropy, rho, grad norm, and trace-FIM.

    `jacobian` has shape `(num_classes, latent_dim)` and is the local derivative
    of logits with respect to the chosen latent coordinates.
    """
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()

    log_p = np.log(p)
    entropy = float(-np.sum(p * log_p))
    u = -log_p - entropy
    fisher = np.diag(p) - np.outer(p, p)
    fisher_u = fisher @ u
    varentropy = float(u @ fisher_u)

    j = np.asarray(jacobian, dtype=np.float64)
    rhs = j.T @ fisher_u

    if j.size == 0 or varentropy <= eps:
        accessible = 0.0
        rho = 0.0
    else:
        eigvals, eigvecs = np.linalg.eigh(fisher)
        sqrt_eigvals = np.sqrt(np.clip(eigvals, 0.0, None))
        sqrt_fisher = (eigvecs * sqrt_eigvals) @ eigvecs.T
        a = sqrt_fisher @ u
        fisher_whitened_j = sqrt_fisher @ j
        left, singular_values, _right = np.linalg.svd(fisher_whitened_j, full_matrices=False)
        if singular_values.size == 0:
            accessible = 0.0
        else:
            rank_cutoff = 1e-10 * max(fisher_whitened_j.shape) * singular_values[0]
            active = singular_values > rank_cutoff
            if not np.any(active):
                accessible = 0.0
            else:
                basis = left[:, active]
                accessible = float(np.sum((basis.T @ a) ** 2))
        accessible = min(max(accessible, 0.0), max(varentropy, 0.0))
        rho = float(accessible / varentropy)
        rho = min(max(rho, 0.0), 1.0)

    grad_norm_sq = float(rhs @ rhs)
    trace_fim = float(np.trace(j.T @ fisher @ j))

    return FisherScores(
        entropy=entropy,
        varentropy=varentropy,
        accessible_varentropy=accessible,
        rho=rho,
        grad_norm_sq=grad_norm_sq,
        trace_fim=trace_fim,
    )


def fisher_scores_batch(
    logits: np.ndarray,
    head_weight: np.ndarray,
    basis: np.ndarray,
) -> list[FisherScores]:
    """Compute Fisher scores for each sample and one global latent subspace.

    `head_weight` has shape `(num_classes, hidden_dim)`.
    `basis` has shape `(hidden_dim, subspace_dim)`.
    """
    probs = softmax_np(logits)
    jacobian = head_weight @ basis
    return [fisher_scores_for_sample(row, jacobian) for row in probs]
