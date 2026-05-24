from __future__ import annotations

import numpy as np
import torch


def mlm_logits_from_hidden(model, hidden: torch.Tensor) -> torch.Tensor:
    """Apply a BERT/DistilBERT masked-LM head to hidden states."""
    if hasattr(model, "cls"):
        return model.cls(hidden)
    if all(
        hasattr(model, name)
        for name in ["vocab_transform", "vocab_layer_norm", "vocab_projector"]
    ):
        x = model.vocab_transform(hidden)
        activation = getattr(model, "activation", None)
        if activation is not None:
            x = activation(x)
        x = model.vocab_layer_norm(x)
        return model.vocab_projector(x)
    raise TypeError(f"Unsupported MLM head for {type(model).__name__}")


def mlm_head_jacobian(model, hidden_vector: torch.Tensor, token_ids: np.ndarray) -> np.ndarray:
    """Jacobian of selected MLM-head logits with respect to one hidden vector."""
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=hidden_vector.device)
    base = hidden_vector.detach().clone().requires_grad_(True)

    def selected_logits(vec: torch.Tensor) -> torch.Tensor:
        logits = mlm_logits_from_hidden(model, vec.view(1, 1, -1))[0, 0, ids]
        return logits

    jac = torch.autograd.functional.jacobian(
        selected_logits,
        base,
        create_graph=False,
        strict=False,
        vectorize=True,
    )
    return jac.detach().cpu().numpy().astype(np.float64)
