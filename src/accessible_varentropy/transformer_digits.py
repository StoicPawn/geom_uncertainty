from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _weights = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class PatchTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        hidden_dim: int = 48,
        num_layers: int = 3,
        num_heads: int = 4,
        mlp_dim: int = 96,
        patch_size: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size
        self.num_patches = (8 // patch_size) * (8 // patch_size)

        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, mlp_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        image = x.view(batch, 1, 8, 8)
        patches = image.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(batch, 1, -1, self.patch_dim)
        return patches.squeeze(1)

    def initial_sequence(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(self.patchify(x))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, patches], dim=1) + self.pos_embed

    def forward_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        states = []
        h = self.initial_sequence(x)
        states.append(h)
        for block in self.blocks:
            h = block(h)
            states.append(h)
        return states

    def logits_from_state(self, state: torch.Tensor, start_layer: int) -> torch.Tensor:
        h = state
        for block in self.blocks[start_layer:]:
            h = block(h)
        cls = self.norm(h[:, 0, :])
        return self.head(cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_from_state(self.forward_states(x)[-1], self.num_layers)


@dataclass(frozen=True)
class TrainedTransformer:
    model: PatchTransformerClassifier
    best_val_accuracy: float
    final_train_accuracy: float


def _accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(torch.from_numpy(x)), dim=1).cpu().numpy()
    return float(np.mean(pred == y))


def train_patch_transformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    num_classes: int,
    epochs: int = 80,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    hidden_dim: int = 48,
    num_layers: int = 3,
    num_heads: int = 4,
    mlp_dim: int = 96,
) -> TrainedTransformer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    model = PatchTransformerClassifier(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
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

    return TrainedTransformer(
        model=model,
        best_val_accuracy=best_val_accuracy,
        final_train_accuracy=_accuracy(model, x_train, y_train),
    )


def collect_cls_states(
    model: PatchTransformerClassifier,
    x: np.ndarray,
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, list[np.ndarray]]:
    model.eval()
    logits_batches: list[np.ndarray] = []
    states_by_layer: list[list[np.ndarray]] = [[] for _ in range(model.num_layers + 1)]
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size])
            states = model.forward_states(xb)
            logits = model.logits_from_state(states[-1], model.num_layers)
            logits_batches.append(logits.cpu().numpy())
            for layer, state in enumerate(states):
                states_by_layer[layer].append(state[:, 0, :].cpu().numpy())
    logits_np = np.vstack(logits_batches).astype(np.float64)
    cls_states = [np.vstack(parts).astype(np.float64) for parts in states_by_layer]
    return logits_np, cls_states


def jacobian_logits_wrt_cls(
    model: PatchTransformerClassifier,
    state: torch.Tensor,
    *,
    layer: int,
) -> np.ndarray:
    """Jacobian of logits wrt the CLS vector at one intermediate layer.

    Other token states at that layer are held fixed. The returned matrix has
    shape `(num_classes, hidden_dim)`.
    """
    model.eval()
    state = state.detach()
    cls = state[0, 0, :].detach().clone().requires_grad_(True)
    fixed_state = state.clone()

    def logits_from_cls(cls_vector: torch.Tensor) -> torch.Tensor:
        local_state = fixed_state.clone()
        local_state[0, 0, :] = cls_vector
        return model.logits_from_state(local_state, layer).squeeze(0)

    jac = torch.autograd.functional.jacobian(
        logits_from_cls,
        cls,
        create_graph=False,
        strict=False,
        vectorize=True,
    )
    return jac.detach().cpu().numpy().astype(np.float64)
