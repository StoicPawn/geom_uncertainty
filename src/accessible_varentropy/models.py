from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 64,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


@dataclass(frozen=True)
class TrainedModel:
    model: MLPClassifier
    best_val_accuracy: float
    final_train_accuracy: float


def _accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    return float(np.mean(pred == y))


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    num_classes: int,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    hidden_dim: int = 64,
    feature_dim: int = 64,
) -> TrainedModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    model = MLPClassifier(
        x_train.shape[1],
        num_classes,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    best_state = None
    best_val_accuracy = -1.0

    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_accuracy = _accuracy(model, x_val, y_val)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainedModel(
        model=model,
        best_val_accuracy=best_val_accuracy,
        final_train_accuracy=_accuracy(model, x_train, y_train),
    )


def extract_logits_features(model: MLPClassifier, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(x)
        features = model.features(tensor_x).cpu().numpy()
        logits = model(tensor_x).cpu().numpy()
    return logits.astype(np.float64), features.astype(np.float64)


def head_weight_np(model: MLPClassifier) -> np.ndarray:
    return model.head.weight.detach().cpu().numpy().astype(np.float64)
