from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in [SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from accessible_varentropy.data import default_digits_splits
from accessible_varentropy.metrics import fisher_scores_for_sample, softmax_np
from accessible_varentropy.transformer_digits import (
    PatchTransformerClassifier,
    jacobian_logits_wrt_cls,
    train_patch_transformer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "multiarch_layerwise_accessibility")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--eval-per-status", type=int, default=45)
    parser.add_argument("--random-reps", type=int, default=12)
    parser.add_argument("--subspace-ks", default="1,2,4,8")
    parser.add_argument("--intervention-eps", type=float, default=0.5)
    return parser.parse_args()


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v, dtype=np.float64)
    return (v / norm).astype(np.float64)


def orthonormalize(matrix: np.ndarray, k: int) -> np.ndarray:
    q, _r = np.linalg.qr(matrix)
    return q[:, :k].astype(np.float64)


def pca_basis(matrix: np.ndarray, max_k: int) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:max_k].T.astype(np.float64)


def entropy_varentropy_from_logits(logits: np.ndarray) -> tuple[float, float]:
    probs = softmax_np(logits[None, :])[0]
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    entropy = float(-np.sum(probs * log_p))
    surprisal = -log_p
    return entropy, float(np.sum(probs * (surprisal - entropy) ** 2))


class MLPStack(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 48)
        self.fc2 = nn.Linear(48, 32)
        self.head = nn.Linear(32, num_classes)

    def forward_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return [x, h1, h2]

    def logits_from_state(self, state: torch.Tensor, layer: int) -> torch.Tensor:
        if layer == 0:
            return self.head(torch.relu(self.fc2(torch.relu(self.fc1(state)))))
        if layer == 1:
            return self.head(torch.relu(self.fc2(state)))
        if layer == 2:
            return self.head(state)
        raise ValueError(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_from_state(self.forward_states(x)[-1], 2)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 8 * 8, 48)
        self.head = nn.Linear(48, num_classes)

    def forward_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        image = x.view(-1, 1, 8, 8)
        h1 = torch.relu(self.conv1(image))
        h2 = torch.relu(self.conv2(h1))
        h3 = torch.relu(self.fc(h2.flatten(1)))
        return [h1.flatten(1), h2.flatten(1), h3]

    def logits_from_state(self, state: torch.Tensor, layer: int) -> torch.Tensor:
        if layer == 0:
            h1 = state.view(-1, 4, 8, 8)
            h2 = torch.relu(self.conv2(h1))
            h3 = torch.relu(self.fc(h2.flatten(1)))
            return self.head(h3)
        if layer == 1:
            h2 = state.view(-1, 8, 8, 8)
            h3 = torch.relu(self.fc(h2.flatten(1)))
            return self.head(h3)
        if layer == 2:
            return self.head(state)
        raise ValueError(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_from_state(self.forward_states(x)[-1], 2)


@dataclass
class ArchRun:
    architecture: str
    model: nn.Module
    num_layers: int


def train_torch_model(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, seed: int, epochs: int) -> tuple[nn.Module, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state = None
    best_acc = -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x_val)).argmax(dim=1).cpu().numpy()
        acc = float(np.mean(pred == y_val))
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def balanced_indices(is_ood: np.ndarray, per_status: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    id_idx = np.flatnonzero(~is_ood)
    ood_idx = np.flatnonzero(is_ood)
    keep = np.concatenate(
        [
            rng.choice(id_idx, size=min(per_status, len(id_idx)), replace=False),
            rng.choice(ood_idx, size=min(per_status, len(ood_idx)), replace=False),
        ]
    )
    rng.shuffle(keep)
    return keep


def collect_states_generic(model: nn.Module, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    model.eval()
    logits_parts = []
    states_by_layer: list[list[np.ndarray]] | None = None
    with torch.no_grad():
        for start in range(0, len(x), 128):
            xb = torch.from_numpy(x[start : start + 128])
            states = model.forward_states(xb)
            logits = model.logits_from_state(states[-1], len(states) - 1)
            logits_parts.append(logits.cpu().numpy())
            if states_by_layer is None:
                states_by_layer = [[] for _ in states]
            for layer, state in enumerate(states):
                states_by_layer[layer].append(state.cpu().numpy())
    assert states_by_layer is not None
    return np.vstack(logits_parts).astype(np.float64), [np.vstack(parts).astype(np.float64) for parts in states_by_layer]


def jacobian_generic(model: nn.Module, state_vec: np.ndarray, layer: int) -> np.ndarray:
    base = torch.as_tensor(state_vec, dtype=torch.float32).detach().clone().requires_grad_(True)

    def fn(vec: torch.Tensor) -> torch.Tensor:
        return model.logits_from_state(vec.view(1, -1), layer).squeeze(0)

    jac = torch.autograd.functional.jacobian(fn, base, create_graph=False, strict=False, vectorize=True)
    return jac.detach().cpu().numpy().astype(np.float64)


def transformer_states(model: PatchTransformerClassifier, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[torch.Tensor]]:
    model.eval()
    logits_parts = []
    cls_by_layer: list[list[np.ndarray]] = [[] for _ in range(model.num_layers + 1)]
    full_states: list[torch.Tensor] = []
    with torch.no_grad():
        for idx in range(len(x)):
            xb = torch.from_numpy(x[idx : idx + 1])
            states = model.forward_states(xb)
            logits = model.logits_from_state(states[-1], model.num_layers)
            logits_parts.append(logits.cpu().numpy())
            full_states.append(torch.cat([s.detach().cpu() for s in states], dim=0))
            for layer, state in enumerate(states):
                cls_by_layer[layer].append(state[:, 0, :].cpu().numpy())
    return np.vstack(logits_parts).astype(np.float64), [np.vstack(parts).astype(np.float64) for parts in cls_by_layer], full_states


def fisher_row(probs: np.ndarray, jac: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    score = fisher_scores_for_sample(probs, jac @ basis)
    return {
        "rho": score.rho,
        "entropy": score.entropy,
        "varentropy": score.varentropy,
        "trace_fim": score.trace_fim,
        "grad_norm_sq": score.grad_norm_sq,
    }


def intervention_delta(model, state_vec: np.ndarray, layer: int, jac: np.ndarray, probs: np.ndarray, direction: np.ndarray, eps: float) -> tuple[float, float]:
    p = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(p * np.log(p)))
    u = -np.log(p) - entropy
    fisher = np.diag(p) - np.outer(p, p)
    sign = 1.0 if float((jac @ direction) @ (fisher @ u)) >= 0.0 else -1.0
    before_entropy, before_var = entropy_varentropy_from_logits(model.logits_from_state(torch.as_tensor(state_vec, dtype=torch.float32).view(1, -1), layer).detach().cpu().numpy()[0])
    after_state = state_vec + sign * eps * direction
    after_logits = model.logits_from_state(torch.as_tensor(after_state, dtype=torch.float32).view(1, -1), layer).detach().cpu().numpy()[0]
    after_entropy, after_var = entropy_varentropy_from_logits(after_logits)
    return after_entropy - before_entropy, abs(after_var - before_var)


def intervention_delta_transformer(
    model: PatchTransformerClassifier,
    full_state: torch.Tensor,
    layer: int,
    jac: np.ndarray,
    probs: np.ndarray,
    direction: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    p = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(p * np.log(p)))
    u = -np.log(p) - entropy
    fisher = np.diag(p) - np.outer(p, p)
    sign = 1.0 if float((jac @ direction) @ (fisher @ u)) >= 0.0 else -1.0
    before_logits = model.logits_from_state(full_state, layer).detach().cpu().numpy()[0]
    before_entropy, before_var = entropy_varentropy_from_logits(before_logits)
    after_state = full_state.detach().clone()
    step = torch.as_tensor(sign * eps * direction, dtype=after_state.dtype)
    after_state[0, 0, :] = after_state[0, 0, :] + step
    after_logits = model.logits_from_state(after_state, layer).detach().cpu().numpy()[0]
    after_entropy, after_var = entropy_varentropy_from_logits(after_logits)
    return after_entropy - before_entropy, abs(after_var - before_var)


def run_architecture(
    arch: ArchRun,
    split,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict]]:
    keep = balanced_indices(split.is_ood, args.eval_per_status, args.seed + 101)
    x_eval = split.x_eval[keep]
    y_eval = split.y_eval[keep]
    is_ood = split.is_ood[keep]
    original_y = split.original_y_eval[keep]
    if arch.architecture == "transformer":
        _train_logits, train_states, _train_full = transformer_states(arch.model, split.x_train[: min(len(split.x_train), 256)])
        eval_logits, eval_states, eval_full_states = transformer_states(arch.model, x_eval)
    else:
        _train_logits, train_states = collect_states_generic(arch.model, split.x_train)
        eval_logits, eval_states = collect_states_generic(arch.model, x_eval)
        eval_full_states = []

    ks = [int(v) for v in args.subspace_ks.split(",") if v.strip()]
    rows: list[dict] = []
    intervention_rows: list[dict] = []
    run_rows = [
        {
            "architecture": arch.architecture,
            "split": split.name,
            "n_eval": len(x_eval),
            "n_ood": int(is_ood.sum()),
            "num_classes": len(split.id_classes),
        }
    ]
    for layer, train_layer_states in enumerate(train_states):
        hidden_dim = train_layer_states.shape[1]
        local_ks = [k for k in ks if k <= hidden_dim]
        max_k = max(local_ks)
        state_basis_full = pca_basis(train_layer_states, max_k)
        random_bases = {
            k: [orthonormalize(rng.normal(size=(hidden_dim, k)), k) for _ in range(args.random_reps)]
            for k in local_ks
        }
        for sample_idx in range(len(x_eval)):
            if arch.architecture == "transformer":
                state_vec = eval_states[layer][sample_idx]
                jac = jacobian_logits_wrt_cls(arch.model, eval_full_states[sample_idx][layer : layer + 1], layer=layer)
                logits_layer = arch.model.logits_from_state(eval_full_states[sample_idx][layer : layer + 1], layer).detach().cpu().numpy()[0]
            else:
                state_vec = eval_states[layer][sample_idx]
                jac = jacobian_generic(arch.model, state_vec, layer)
                logits_layer = arch.model.logits_from_state(torch.as_tensor(state_vec, dtype=torch.float32).view(1, -1), layer).detach().cpu().numpy()[0]
            probs = softmax_np(logits_layer[None, :])[0]
            random_cache: dict[int, tuple[float, float]] = {}
            for k in local_ks:
                vals = [fisher_row(probs, jac, basis)["rho"] for basis in random_bases[k]]
                random_cache[k] = (float(np.mean(vals)), float(np.std(vals)))
            for k in local_ks:
                basis = state_basis_full[:, :k]
                values = fisher_row(probs, jac, basis)
                random_mean, random_std = random_cache[k]
                rows.append(
                    {
                        "architecture": arch.architecture,
                        "split": split.name,
                        "sample_index": sample_idx,
                        "layer": layer,
                        "subspace": "state_pca",
                        "k": k,
                        "is_ood": bool(is_ood[sample_idx]),
                        "original_label": int(original_y[sample_idx]),
                        "mapped_label": int(y_eval[sample_idx]) if y_eval[sample_idx] >= 0 else -1,
                        "rho_structured": values["rho"],
                        "rho_random_mean": random_mean,
                        "rho_adjusted": values["rho"] - random_mean,
                        "entropy": values["entropy"],
                        "varentropy": values["varentropy"],
                        "trace_fim": values["trace_fim"],
                    }
                )
            if layer < len(eval_states) - 1 and eval_states[layer + 1].shape[1] == eval_states[layer].shape[1]:
                direction = normalize(eval_states[layer + 1][sample_idx] - eval_states[layer][sample_idx])[:, None]
                values = fisher_row(probs, jac, direction)
                random_mean, random_std = random_cache[1]
                rows.append(
                    {
                        "architecture": arch.architecture,
                        "split": split.name,
                        "sample_index": sample_idx,
                        "layer": layer,
                        "subspace": "delta_forward_1d",
                        "k": 1,
                        "is_ood": bool(is_ood[sample_idx]),
                        "original_label": int(original_y[sample_idx]),
                        "mapped_label": int(y_eval[sample_idx]) if y_eval[sample_idx] >= 0 else -1,
                        "rho_structured": values["rho"],
                        "rho_random_mean": random_mean,
                        "rho_adjusted": values["rho"] - random_mean,
                        "entropy": values["entropy"],
                        "varentropy": values["varentropy"],
                        "trace_fim": values["trace_fim"],
                    }
                )
            for intervention_name, direction in [("state_pca_1", state_basis_full[:, 0])]:
                direction = normalize(direction)
                if arch.architecture == "transformer":
                    delta_entropy, abs_delta_var = intervention_delta_transformer(
                        arch.model,
                        eval_full_states[sample_idx][layer : layer + 1],
                        layer,
                        jac,
                        probs,
                        direction,
                        args.intervention_eps,
                    )
                else:
                    delta_entropy, abs_delta_var = intervention_delta(
                        arch.model,
                        state_vec,
                        layer,
                        jac,
                        probs,
                        direction,
                        args.intervention_eps,
                    )
                intervention_rows.append(
                    {
                        "architecture": arch.architecture,
                        "split": split.name,
                        "sample_index": sample_idx,
                        "layer": layer,
                        "direction": intervention_name,
                        "is_ood": bool(is_ood[sample_idx]),
                        "delta_entropy": delta_entropy,
                        "abs_delta_varentropy": abs_delta_var,
                    }
                )
    return rows, run_rows, intervention_rows


def summarize(scores: pd.DataFrame, interventions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = scores.groupby(["architecture", "split", "subspace", "k", "layer"], dropna=False).agg(
        n=("sample_index", "count"),
        rho_structured_mean=("rho_structured", "mean"),
        rho_random_mean=("rho_random_mean", "mean"),
        rho_adjusted_mean=("rho_adjusted", "mean"),
        saturation_rate=("rho_structured", lambda s: float((s >= 0.99).mean())),
    ).reset_index()
    curve_rows = []
    for keys, group in scores[scores["subspace"].eq("state_pca")].groupby(["architecture", "split", "sample_index", "layer"]):
        architecture, split, sample_index, layer = keys
        group = group.sort_values("k")
        for threshold in [0.5, 0.8]:
            hit = group[group["rho_structured"] >= threshold]
            curve_rows.append(
                {
                    "architecture": architecture,
                    "split": split,
                    "sample_index": int(sample_index),
                    "layer": int(layer),
                    "threshold": threshold,
                    "k_threshold": float(hit.iloc[0]["k"]) if not hit.empty else float("nan"),
                }
            )
    compress = pd.DataFrame(curve_rows).groupby(["architecture", "split", "layer", "threshold"], dropna=False).agg(
        n=("sample_index", "count"),
        k_mean=("k_threshold", "mean"),
        missing_rate=("k_threshold", lambda s: float(s.isna().mean())),
    ).reset_index()
    auc_rows = []
    for keys, group in scores.groupby(["architecture", "split", "subspace", "k"]):
        architecture, split, subspace, k = keys
        prompt_values = group.groupby("sample_index").agg(
            rho_adjusted_mean=("rho_adjusted", "mean"),
            rho_structured_mean=("rho_structured", "mean"),
            entropy_mean=("entropy", "mean"),
            trace_fim_mean=("trace_fim", "mean"),
        )
        labels = group.groupby("sample_index")["is_ood"].first().astype(int).loc[prompt_values.index].to_numpy()
        if len(np.unique(labels)) < 2:
            continue
        for metric in prompt_values.columns:
            values = prompt_values[metric].to_numpy(dtype=float)
            auc_rows.append(
                {
                    "architecture": architecture,
                    "split": split,
                    "subspace": subspace,
                    "k": int(k),
                    "metric": metric,
                    "best_auc": float(max(roc_auc_score(labels, values), roc_auc_score(labels, -values))),
                }
            )
    intervention_summary = interventions.groupby(["architecture", "split", "layer", "direction"], dropna=False).agg(
        n=("sample_index", "count"),
        delta_entropy_mean=("delta_entropy", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
    ).reset_index()
    return summary, compress, pd.concat([pd.DataFrame(auc_rows), intervention_summary.assign(metric="intervention")], ignore_index=True, sort=False)


def write_report(out_dir: Path, run_info: pd.DataFrame, summary: pd.DataFrame, compress: pd.DataFrame, combined: pd.DataFrame) -> None:
    def table(df: pd.DataFrame, max_rows: int = 60) -> str:
        if df.empty:
            return "```text\n(empty)\n```"
        return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"

    aucs = combined[combined["metric"].ne("intervention")].sort_values(["architecture", "best_auc"], ascending=[True, False])
    interventions = combined[combined["metric"].eq("intervention")]
    lines = [
        "# Multi-Architecture Layerwise Accessibility",
        "",
        "Dataset: sklearn digits semantic OOD splits. Architectures: MLP, CNN, patch transformer.",
        "Each layer has its own tail `g_ell` to the classifier output. MLP/CNN layers have different dimensions; transformer CLS layers share dimensions and include `delta_forward_1d`.",
        "",
        "## Run Info",
        table(run_info),
        "",
        "## Structured vs Random-Adjusted Summary",
        table(summary, max_rows=120),
        "",
        "## Compressibility k50/k80",
        table(compress, max_rows=120),
        "",
        "## Prompt-Level Best AUCs",
        table(aucs, max_rows=120),
        "",
        "## Intervention Summary",
        table(interventions, max_rows=120),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_rows: list[dict] = []
    all_runs: list[dict] = []
    all_interventions: list[dict] = []
    for split in default_digits_splits(args.seed):
        num_classes = len(split.id_classes)
        mlp, mlp_acc = train_torch_model(
            MLPStack(split.x_train.shape[1], num_classes),
            split.x_train,
            split.y_train,
            split.x_val,
            split.y_val,
            seed=args.seed,
            epochs=args.epochs,
        )
        cnn, cnn_acc = train_torch_model(
            SmallCNN(num_classes),
            split.x_train,
            split.y_train,
            split.x_val,
            split.y_val,
            seed=args.seed + 1,
            epochs=args.epochs,
        )
        trained_transformer = train_patch_transformer(
            split.x_train,
            split.y_train,
            split.x_val,
            split.y_val,
            seed=args.seed + 2,
            num_classes=num_classes,
            epochs=args.epochs,
            hidden_dim=32,
            num_layers=3,
            num_heads=4,
            mlp_dim=64,
        )
        archs = [
            ArchRun("mlp", mlp, 3),
            ArchRun("cnn", cnn, 3),
            ArchRun("transformer", trained_transformer.model, trained_transformer.model.num_layers + 1),
        ]
        val_accs = {"mlp": mlp_acc, "cnn": cnn_acc, "transformer": trained_transformer.best_val_accuracy}
        for arch in archs:
            print(f"running_arch {arch.architecture} split {split.name}", flush=True)
            rows, runs, interventions = run_architecture(arch, split, args, rng)
            for run in runs:
                run["val_accuracy"] = val_accs[arch.architecture]
            all_rows.extend(rows)
            all_runs.extend(runs)
            all_interventions.extend(interventions)
    scores = pd.DataFrame(all_rows)
    run_info = pd.DataFrame(all_runs)
    interventions = pd.DataFrame(all_interventions)
    scores.to_csv(args.out_dir / "multiarch_scores.csv", index=False)
    run_info.to_csv(args.out_dir / "multiarch_run_info.csv", index=False)
    interventions.to_csv(args.out_dir / "multiarch_interventions.csv", index=False)
    summary, compress, combined = summarize(scores, interventions)
    summary.to_csv(args.out_dir / "multiarch_summary.csv", index=False)
    compress.to_csv(args.out_dir / "multiarch_compressibility.csv", index=False)
    combined.to_csv(args.out_dir / "multiarch_auc_and_intervention_summary.csv", index=False)
    write_report(args.out_dir, run_info, summary, compress, combined)
    report = (args.out_dir / "report.md").read_text(encoding="utf-8")
    print(report.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
