from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SemanticOODSplit:
    name: str
    id_classes: tuple[int, ...]
    ood_classes: tuple[int, ...]
    class_map: dict[int, int]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_eval: np.ndarray
    y_eval: np.ndarray
    is_ood: np.ndarray
    original_y_eval: np.ndarray


def load_digits_semantic_ood(
    *,
    name: str,
    id_classes: Sequence[int],
    ood_classes: Sequence[int],
    seed: int,
    test_size: float = 0.35,
    val_size: float = 0.20,
) -> SemanticOODSplit:
    """Load a public handwritten-digits semantic OOD split.

    The model is trained only on `id_classes`; evaluation combines held-out ID
    examples with held-out examples from `ood_classes`.
    """
    digits = load_digits()
    x = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    all_classes = tuple(id_classes) + tuple(ood_classes)
    mask = np.isin(y, all_classes)
    x = x[mask]
    y = y[mask]

    x_pool, x_test, y_pool, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    id_pool_mask = np.isin(y_pool, id_classes)
    x_id_pool = x_pool[id_pool_mask]
    y_id_pool_original = y_pool[id_pool_mask]

    x_train, x_val, y_train_original, y_val_original = train_test_split(
        x_id_pool,
        y_id_pool_original,
        test_size=val_size,
        random_state=seed + 10_000,
        stratify=y_id_pool_original,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    class_map = {klass: idx for idx, klass in enumerate(id_classes)}
    y_train = np.array([class_map[int(v)] for v in y_train_original], dtype=np.int64)
    y_val = np.array([class_map[int(v)] for v in y_val_original], dtype=np.int64)

    eval_id_mask = np.isin(y_test, id_classes)
    eval_ood_mask = np.isin(y_test, ood_classes)
    eval_mask = eval_id_mask | eval_ood_mask

    x_eval = x_test[eval_mask]
    original_y_eval = y_test[eval_mask].astype(np.int64)
    is_ood = np.isin(original_y_eval, ood_classes)

    y_eval = np.full_like(original_y_eval, fill_value=-1)
    for original, remapped in class_map.items():
        y_eval[original_y_eval == original] = remapped

    return SemanticOODSplit(
        name=name,
        id_classes=tuple(id_classes),
        ood_classes=tuple(ood_classes),
        class_map=class_map,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_eval=x_eval,
        y_eval=y_eval,
        is_ood=is_ood.astype(bool),
        original_y_eval=original_y_eval,
    )


def default_digits_splits(seed: int) -> list[SemanticOODSplit]:
    return [
        load_digits_semantic_ood(
            name="digits_low_vs_high",
            id_classes=(0, 1, 2, 3, 4),
            ood_classes=(5, 6, 7, 8, 9),
            seed=seed,
        ),
        load_digits_semantic_ood(
            name="digits_even_vs_odd",
            id_classes=(0, 2, 4, 6, 8),
            ood_classes=(1, 3, 5, 7, 9),
            seed=seed,
        ),
    ]
