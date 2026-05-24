from __future__ import annotations

import numpy as np


def _orthonormalize(matrix: np.ndarray, k: int) -> np.ndarray:
    q, _r = np.linalg.qr(matrix)
    return q[:, :k]


def pca_basis(features: np.ndarray, k: int, *, top: bool = True) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    if top:
        components = vt[:k]
    else:
        components = vt[-k:]
    return components.T.copy()


def class_mean_basis(features: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    means = []
    for label in sorted(np.unique(labels)):
        means.append(features[labels == label].mean(axis=0))
    mean_matrix = np.vstack(means)
    centered = mean_matrix - mean_matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k]
    if components.shape[0] < k:
        pad = np.zeros((k - components.shape[0], features.shape[1]))
        components = np.vstack([components, pad])
    return _orthonormalize(components.T, k)


def random_basis(feature_dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return _orthonormalize(rng.normal(size=(feature_dim, k)), k)


def coordinate_basis(feature_dim: int) -> np.ndarray:
    return np.eye(feature_dim)
