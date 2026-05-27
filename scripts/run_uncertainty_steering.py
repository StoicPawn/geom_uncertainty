from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, BertTokenizer

from accessible_varentropy.bert_mlm import encode_prompt, hidden_states_and_logits, token_id_for_word
from accessible_varentropy.metrics import softmax_np
from accessible_varentropy.mlm_heads import mlm_head_jacobian, mlm_logits_from_hidden
from accessible_varentropy.subspaces import pca_basis
from run_factual_error_deep_benchmark import FactualCase, factual_error_cases


@dataclass(frozen=True)
class FisherGeometry:
    probs: np.ndarray
    fisher: np.ndarray
    sqrt_fisher: np.ndarray
    u: np.ndarray
    w: np.ndarray
    fisher_u: np.ndarray
    entropy: float
    varentropy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "uncertainty_steering_distilbert_pca16")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--eps", default="0.1,0.2,0.4")
    parser.add_argument("--semantic-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v, dtype=np.float64)
    return (v / norm).astype(np.float64)


def topk_distribution(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    values = logits[idx]
    return idx.astype(np.int64), softmax_np(values[None, :])[0].astype(np.float64)


def entropy_varentropy(probs: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    log_p = np.log(p)
    entropy = float(-np.sum(p * log_p))
    varentropy = float(np.sum(p * ((-log_p - entropy) ** 2)))
    return entropy, varentropy


def fisher_geometry(probs: np.ndarray, eps: float = 1e-12) -> FisherGeometry:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    log_p = np.log(p)
    entropy = float(-np.sum(p * log_p))
    u = -log_p - entropy
    fisher = np.diag(p) - np.outer(p, p)
    fisher_u = fisher @ u
    varentropy = float(u @ fisher_u)
    eigvals, eigvecs = np.linalg.eigh(fisher)
    sqrt_vals = np.sqrt(np.clip(eigvals, 0.0, None))
    sqrt_fisher = (eigvecs * sqrt_vals) @ eigvecs.T
    w = sqrt_fisher @ u
    return FisherGeometry(p, fisher, sqrt_fisher, u, w, fisher_u, entropy, varentropy)


def orthogonal_basis_projection(matrix: np.ndarray, vector: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, int]:
    if matrix.size == 0:
        return np.zeros_like(vector), np.zeros((len(vector), 0), dtype=np.float64), 0
    left, singular_values, _right = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros_like(vector), np.zeros((len(vector), 0), dtype=np.float64), 0
    cutoff = 1e-10 * max(matrix.shape) * singular_values[0]
    active = singular_values > max(cutoff, eps)
    if not np.any(active):
        return np.zeros_like(vector), np.zeros((len(vector), 0), dtype=np.float64), 0
    basis = left[:, active]
    projected = basis @ (basis.T @ vector)
    return projected.astype(np.float64), basis.astype(np.float64), int(active.sum())


def least_squares(matrix: np.ndarray, target: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[1],), dtype=np.float64)
    return np.linalg.pinv(matrix, rcond=rcond) @ target


def semantic_entropy_proxy(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    masses = []
    for label in sorted(set(labels.tolist())):
        masses.append(float(probs[labels == label].sum()))
    return entropy_varentropy(np.asarray(masses, dtype=np.float64), eps=eps)[0]


def token_cluster_labels(
    embeddings: np.ndarray,
    token_ids: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    x = embeddings[token_ids]
    k = max(1, min(int(n_clusters), len(token_ids)))
    if k == 1:
        return np.zeros((len(token_ids),), dtype=np.int64)
    centered = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[0]
    order = np.argsort(scores)
    labels = np.zeros((len(token_ids),), dtype=np.int64)
    for rank, idx in enumerate(order):
        labels[int(idx)] = min(k - 1, int(rank * k / len(token_ids)))
    return labels


def layer_logits(model, hidden: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = mlm_logits_from_hidden(model, hidden.view(1, 1, -1))[0, 0, :]
    return logits.detach().cpu().numpy().astype(np.float64)


def selected_logits_from_hidden(model, hidden: torch.Tensor, token_ids: np.ndarray) -> np.ndarray:
    required = ["vocab_transform", "vocab_layer_norm", "vocab_projector"]
    if not all(hasattr(model, name) for name in required):
        full = layer_logits(model, hidden)
        return full[token_ids]

    with torch.no_grad():
        x = model.vocab_transform(hidden.view(1, 1, -1))
        activation = getattr(model, "activation", None)
        if activation is not None:
            x = activation(x)
        x = model.vocab_layer_norm(x)[0, 0, :].detach().float()
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=x.device)
        rows = model.vocab_projector.weight.detach().float()[ids]
        logits = rows @ x
        if model.vocab_projector.bias is not None:
            logits = logits + model.vocab_projector.bias.detach().float()[ids]
    return logits.detach().cpu().numpy().astype(np.float64)


def head_jacobian(model, hidden: torch.Tensor, token_ids: np.ndarray) -> np.ndarray:
    """Fast exact Jacobian for DistilBERT MLM heads, autograd fallback otherwise."""
    required = ["vocab_transform", "vocab_layer_norm", "vocab_projector"]
    if not all(hasattr(model, name) for name in required):
        return mlm_head_jacobian(model, hidden, token_ids)

    h = hidden.detach().float()
    transform = model.vocab_transform
    layer_norm = model.vocab_layer_norm
    projector = model.vocab_projector

    weight1 = transform.weight.detach().float()
    bias1 = transform.bias.detach().float() if transform.bias is not None else 0.0
    pre = (weight1 @ h + bias1).detach().requires_grad_(True)
    activation = getattr(model, "activation", None)
    activated = activation(pre) if activation is not None else pre
    activation_grad = torch.autograd.grad(
        activated,
        pre,
        grad_outputs=torch.ones_like(activated),
        retain_graph=False,
        create_graph=False,
    )[0].detach()

    x = activated.detach()
    mean = x.mean()
    centered = x - mean
    variance = torch.mean(centered * centered)
    std = torch.sqrt(variance + float(layer_norm.eps))
    xhat = centered / std

    ids = torch.as_tensor(token_ids, dtype=torch.long, device=projector.weight.device)
    rows = projector.weight.detach().float()[ids]
    gamma = layer_norm.weight.detach().float()
    upstream = rows * gamma
    upstream_mean = upstream.mean(dim=1, keepdim=True)
    upstream_xhat_mean = (upstream * xhat).mean(dim=1, keepdim=True)
    after_layer_norm = (upstream - upstream_mean - xhat * upstream_xhat_mean) / std
    after_activation = after_layer_norm * activation_grad
    jac = after_activation @ weight1
    return jac.detach().cpu().numpy().astype(np.float64)


def logits_after_step(model, hidden: torch.Tensor, direction: np.ndarray, epsilon: float) -> np.ndarray:
    direction_t = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
    with torch.no_grad():
        stepped = hidden + float(epsilon) * direction_t
        logits = mlm_logits_from_hidden(model, stepped.view(1, 1, -1))[0, 0, :]
    return logits.detach().cpu().numpy().astype(np.float64)


def selected_logits_after_step(
    model,
    hidden: torch.Tensor,
    direction: np.ndarray,
    epsilon: float,
    token_ids: np.ndarray,
) -> np.ndarray:
    direction_t = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
    stepped = hidden + float(epsilon) * direction_t
    return selected_logits_from_hidden(model, stepped, token_ids)


def rank_of_token(logits: np.ndarray, token_id: int) -> int:
    return int(1 + np.sum(logits > logits[int(token_id)]))


def token_probability(logits: np.ndarray, token_id: int) -> float:
    shifted = logits - float(np.max(logits))
    return float(np.exp(shifted[int(token_id)]) / np.sum(np.exp(shifted)))


def build_prompt_records(
    tokenizer,
    model,
    cases: list[FactualCase],
    max_prompts: int,
    top_k: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for source_id, case in enumerate(cases):
        if max_prompts > 0 and len(rows) >= max_prompts:
            break
        target_id = token_id_for_word(tokenizer, case.target)
        if target_id is None:
            continue
        encoded, mask_index = encode_prompt(tokenizer, case.prompt.replace("[MASK]", tokenizer.mask_token))
        hidden_states, final_logits = hidden_states_and_logits(model, encoded)
        final_mask_logits = final_logits[0, mask_index, :].detach().cpu().numpy().astype(np.float64)
        top_ids, top_probs = topk_distribution(final_mask_logits, top_k)
        top1_id = int(top_ids[0])
        rows.append(
            {
                "prompt_id": len(rows),
                "source_case_id": source_id,
                "prompt": case.prompt,
                "target": case.target,
                "target_id": int(target_id),
                "topic": case.topic,
                "fact_id": case.fact_id,
                "template_id": case.template_id,
                "final_observed_condition": "correct" if top1_id == int(target_id) else "error",
                "final_top1_token": tokenizer.convert_ids_to_tokens([top1_id])[0],
                "final_top1_prob_topk": float(top_probs[0]),
                "final_target_rank": rank_of_token(final_mask_logits, int(target_id)),
                "final_top8_tokens": " ".join(tokenizer.convert_ids_to_tokens([int(v) for v in top_ids[:8]])),
            }
        )
        records.append(
            {
                "prompt_id": len(rows) - 1,
                "mask_index": int(mask_index),
                "hidden_states": hidden_states,
            }
        )
    return pd.DataFrame(rows), records


def interleaved_case_order(cases: list[FactualCase], seed: int) -> list[FactualCase]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[FactualCase]] = {}
    for case in cases:
        groups.setdefault(case.topic, []).append(case)
    for topic_cases in groups.values():
        rng.shuffle(topic_cases)
    ordered: list[FactualCase] = []
    topics = sorted(groups)
    while any(groups[topic] for topic in topics):
        for topic in topics:
            if groups[topic]:
                ordered.append(groups[topic].pop())
    return ordered


def build_pca_subspace(records: list[dict[str, object]], layer: int, k: int) -> np.ndarray:
    vectors = []
    for record in records:
        hidden_states = record["hidden_states"]
        current = hidden_states[layer][0, int(record["mask_index"]), :].detach().cpu().numpy().astype(np.float64)
        previous = hidden_states[layer - 1][0, int(record["mask_index"]), :].detach().cpu().numpy().astype(np.float64)
        vectors.append(current - previous)
    features = np.vstack(vectors)
    dim = min(k, features.shape[0] - 1, features.shape[1])
    if dim < 1:
        raise RuntimeError("Need at least two prompts to build a PCA subspace.")
    return pca_basis(features, dim, top=True).astype(np.float64)


def control_direction_orthogonal_to_entropy_gradient(
    rng: np.random.Generator,
    grad_z: np.ndarray,
    k: int,
    eps: float = 1e-12,
) -> np.ndarray:
    for _ in range(100):
        raw = rng.normal(size=k).astype(np.float64)
        denom = float(grad_z @ grad_z)
        if denom > eps:
            raw = raw - grad_z * float(raw @ grad_z) / denom
        norm = float(np.linalg.norm(raw))
        if norm > eps:
            return raw / norm
    return np.zeros((k,), dtype=np.float64)


def compute_steering(
    tokenizer,
    model,
    prompts: pd.DataFrame,
    records: list[dict[str, object]],
    basis: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    eps_values = [float(part.strip()) for part in args.eps.split(",") if part.strip()]
    prompt_lookup = prompts.set_index("prompt_id")
    steering_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    B = basis
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()

    for idx, record in enumerate(records):
        if idx % 10 == 0:
            print(f"processing_prompt {idx + 1}/{len(records)}", flush=True)
        meta = prompt_lookup.loc[int(record["prompt_id"])]
        mask_index = int(record["mask_index"])
        hidden = record["hidden_states"][args.layer][0, mask_index, :].detach()
        hidden_dim = int(hidden.shape[-1])
        logits_before = layer_logits(model, hidden)
        token_ids, probs = topk_distribution(logits_before, args.top_k)
        top_logits = logits_before[token_ids]
        geometry = fisher_geometry(probs)
        jac = head_jacobian(model, hidden, token_ids)
        jb = jac @ B
        whitened_jb = geometry.sqrt_fisher @ jb
        w_acc, _basis_out, rank = orthogonal_basis_projection(whitened_jb, geometry.w)
        w_orth = geometry.w - w_acc
        rho = float((w_acc @ w_acc) / max(float(geometry.w @ geometry.w), 1e-12))

        z_acc = least_squares(whitened_jb, w_acc)
        z_orth_strict = least_squares(whitened_jb, w_orth)
        grad_z = whitened_jb.T @ geometry.w
        z_grad_orth = control_direction_orthogonal_to_entropy_gradient(rng, grad_z, B.shape[1])
        z_random = normalize(rng.normal(size=B.shape[1]))

        z_acc_norm = float(np.linalg.norm(z_acc))
        z_orth_norm = float(np.linalg.norm(z_orth_strict))
        if z_acc_norm < 1e-12:
            continue
        directions = {
            "accessible_ls": z_acc / z_acc_norm,
            "strict_orthogonal_ls": z_orth_strict / z_orth_norm if z_orth_norm > 1e-12 else np.zeros_like(z_acc),
            "grad_orthogonal_control": z_grad_orth,
            "random_control": z_random,
        }

        target_id = int(meta["target_id"])
        cluster_labels = token_cluster_labels(embeddings, token_ids, args.semantic_clusters)
        sem_before = semantic_entropy_proxy(probs, cluster_labels)
        base_target_rank = rank_of_token(logits_before, target_id)
        target_positions = np.where(token_ids == target_id)[0]
        target_in_topk = int(len(target_positions) > 0)
        target_prob_topk_before = float(probs[int(target_positions[0])]) if target_in_topk else 0.0
        selected_target_top1_before = int(int(token_ids[0]) == target_id)

        diagnostic_rows.append(
            {
                "prompt_id": int(record["prompt_id"]),
                "layer": int(args.layer),
                "pca_dim": int(B.shape[1]),
                "output_top_k": int(args.top_k),
                "rank_whitened_jb": int(rank),
                "entropy_before": geometry.entropy,
                "varentropy_before": geometry.varentropy,
                "rho_accessible": rho,
                "v_access": float(w_acc @ w_acc),
                "v_inaccess": float(w_orth @ w_orth),
                "strict_orth_ls_norm": z_orth_norm,
                "accessible_ls_norm": z_acc_norm,
                "strict_orth_target_residual": float(np.linalg.norm(whitened_jb @ z_orth_strict - w_orth)),
                "accessible_target_residual": float(np.linalg.norm(whitened_jb @ z_acc - w_acc)),
            }
        )

        for direction_name, z_dir in directions.items():
            hidden_direction = B @ z_dir
            direction_norm = float(np.linalg.norm(hidden_direction))
            first_order_entropy = float(grad_z @ z_dir)
            first_order_logit_norm = float(np.linalg.norm(jb @ z_dir))
            for sign_name, sign in [("increase", 1.0), ("decrease", -1.0)]:
                signed_direction = sign * hidden_direction
                for epsilon in eps_values:
                    selected_after = selected_logits_after_step(model, hidden, signed_direction, epsilon, token_ids)
                    probs_after = softmax_np(selected_after[None, :])[0]
                    entropy_after, varentropy_after = entropy_varentropy(probs_after)
                    sem_after = semantic_entropy_proxy(probs_after, cluster_labels)
                    top1_after_selected = int(token_ids[int(np.argmax(probs_after))])
                    target_prob_topk_after = float(probs_after[int(target_positions[0])]) if target_in_topk else 0.0
                    selected_target_top1_after = int(top1_after_selected == target_id)
                    steering_rows.append(
                        {
                            "prompt_id": int(record["prompt_id"]),
                            "prompt": meta["prompt"],
                            "target": meta["target"],
                            "topic": meta["topic"],
                            "fact_id": meta["fact_id"],
                            "final_observed_condition": meta["final_observed_condition"],
                            "layer": int(args.layer),
                            "pca_dim": int(B.shape[1]),
                            "direction": direction_name,
                            "sign": sign_name,
                            "epsilon": float(epsilon),
                            "rho_accessible": rho,
                            "v_access": float(w_acc @ w_acc),
                            "v_inaccess": float(w_orth @ w_orth),
                            "latent_direction_norm": float(np.linalg.norm(z_dir)),
                            "hidden_direction_norm": direction_norm,
                            "first_order_entropy": sign * first_order_entropy,
                            "first_order_logit_norm": first_order_logit_norm,
                            "entropy_before": geometry.entropy,
                            "entropy_after": entropy_after,
                            "delta_entropy": entropy_after - geometry.entropy,
                            "abs_delta_entropy": abs(entropy_after - geometry.entropy),
                            "varentropy_before": geometry.varentropy,
                            "varentropy_after": varentropy_after,
                            "delta_varentropy": varentropy_after - geometry.varentropy,
                            "abs_delta_varentropy": abs(varentropy_after - geometry.varentropy),
                            "semantic_entropy_proxy_before": sem_before,
                            "semantic_entropy_proxy_after": sem_after,
                            "delta_semantic_entropy_proxy": sem_after - sem_before,
                            "abs_delta_semantic_entropy_proxy": abs(sem_after - sem_before),
                            "selected_top1_before": int(token_ids[0]),
                            "selected_top1_after": top1_after_selected,
                            "selected_top1_changed": int(top1_after_selected != int(token_ids[0])),
                            "target_in_topk": target_in_topk,
                            "target_top1_before_topk": selected_target_top1_before,
                            "target_top1_after_topk": selected_target_top1_after,
                            "target_correct_changed": int(selected_target_top1_before != selected_target_top1_after),
                            "target_rank_before": base_target_rank,
                            "target_prob_topk_before": target_prob_topk_before,
                            "target_prob_topk_after": target_prob_topk_after,
                            "delta_target_prob_topk": target_prob_topk_after - target_prob_topk_before,
                        }
                    )
    return pd.DataFrame(steering_rows), pd.DataFrame(diagnostic_rows)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    return (
        records.groupby(["direction", "sign", "epsilon"], as_index=False)
        .agg(
            n=("prompt_id", "count"),
            delta_entropy_mean=("delta_entropy", "mean"),
            abs_delta_entropy_mean=("abs_delta_entropy", "mean"),
            delta_varentropy_mean=("delta_varentropy", "mean"),
            abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
            delta_semantic_entropy_proxy_mean=("delta_semantic_entropy_proxy", "mean"),
            abs_delta_semantic_entropy_proxy_mean=("abs_delta_semantic_entropy_proxy", "mean"),
            selected_top1_changed_rate=("selected_top1_changed", "mean"),
            target_correct_changed_rate=("target_correct_changed", "mean"),
            target_in_topk_rate=("target_in_topk", "mean"),
            delta_target_prob_topk_mean=("delta_target_prob_topk", "mean"),
            first_order_entropy_mean=("first_order_entropy", "mean"),
            first_order_logit_norm_mean=("first_order_logit_norm", "mean"),
        )
        .sort_values(["epsilon", "sign", "direction"])
    )


def contrast_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame()
    controls = ["strict_orthogonal_ls", "grad_orthogonal_control", "random_control"]
    indexed = summary.set_index(["direction", "sign", "epsilon"])
    for sign in sorted(summary["sign"].unique()):
        for epsilon in sorted(summary["epsilon"].unique()):
            key = ("accessible_ls", sign, epsilon)
            if key not in indexed.index:
                continue
            acc = indexed.loc[key]
            for control in controls:
                control_key = (control, sign, epsilon)
                if control_key not in indexed.index:
                    continue
                ctrl = indexed.loc[control_key]
                for metric in [
                    "abs_delta_entropy_mean",
                    "abs_delta_varentropy_mean",
                    "abs_delta_semantic_entropy_proxy_mean",
                    "selected_top1_changed_rate",
                    "target_correct_changed_rate",
                ]:
                    denom = float(ctrl[metric])
                    ratio = float(acc[metric] / denom) if abs(denom) > 1e-12 else math.inf
                    rows.append(
                        {
                            "sign": sign,
                            "epsilon": float(epsilon),
                            "control": control,
                            "metric": metric,
                            "accessible": float(acc[metric]),
                            "control_value": float(ctrl[metric]),
                            "accessible_minus_control": float(acc[metric] - ctrl[metric]),
                            "accessible_over_control": ratio,
                        }
                    )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def write_report(
    args: argparse.Namespace,
    prompts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    counts = prompts.groupby(["topic", "final_observed_condition"]).size().unstack(fill_value=0).reset_index()
    strict = diagnostics[
        [
            "rho_accessible",
            "v_access",
            "v_inaccess",
            "accessible_ls_norm",
            "strict_orth_ls_norm",
            "accessible_target_residual",
            "strict_orth_target_residual",
        ]
    ].describe()
    key_contrasts = contrasts[
        contrasts["metric"].isin(
            [
                "abs_delta_entropy_mean",
                "abs_delta_varentropy_mean",
                "selected_top1_changed_rate",
                "target_correct_changed_rate",
            ]
        )
    ] if not contrasts.empty else contrasts
    lines = [
        "# Uncertainty Steering via Accessible vs Inaccessible Varentropy",
        "",
        "## Setup",
        "",
        f"- Model: `{args.model}`",
        f"- Layer: `{args.layer}`",
        f"- Output top-k lens: `{args.top_k}`",
        f"- Latent subspace: top PCA basis of layer-delta hidden states, dim `{args.pca_dim}`",
        f"- Prompts: `{len(prompts)}` factual cloze cases",
        "",
        "For each prompt, the run computes `w = F^{1/2}u`, projects it onto `Im(F^{1/2}JB)`, and applies a unit latent perturbation in the accessible least-squares direction. The strict orthogonal least-squares target is also recorded; because it is outside the accessible image, its solution is expected to be near zero. The empirical equal-energy comparison therefore uses an entropy-gradient-orthogonal latent control and a random latent control with the same unit latent norm.",
        "",
        "The semantic uncertainty column is an embedding-quantile entropy proxy over the fixed top-k token set, not a full semantic-entropy estimator. Accuracy is measured as exact-token top-1 stability within the same monitored top-k set.",
        "",
        "## Prompt Counts",
        markdown_table(counts, 40),
        "",
        "## Strict Projection Diagnostics",
        markdown_table(strict.reset_index(), 20),
        "",
        "## Steering Summary",
        markdown_table(summary, 80),
        "",
        "## Accessible vs Control Contrasts",
        markdown_table(key_contrasts, 80),
        "",
        "## Files",
        "```text",
        "prompts.csv",
        "projection_diagnostics.csv",
        "steering_records.csv",
        "steering_summary.csv",
        "steering_contrasts.csv",
        "report.md",
        "```",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    tokenizer = BertTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )
    model.eval()

    cases = interleaved_case_order(factual_error_cases(), args.seed)
    prompts, records = build_prompt_records(tokenizer, model, cases, args.max_prompts, args.top_k)
    if prompts.empty:
        raise RuntimeError("No prompts could be evaluated.")
    num_layers = len(records[0]["hidden_states"]) - 1
    if args.layer <= 0 or args.layer > num_layers:
        raise ValueError(f"Layer must be in [1, {num_layers}], got {args.layer}.")
    basis = build_pca_subspace(records, args.layer, args.pca_dim)

    steering, diagnostics = compute_steering(tokenizer, model, prompts, records, basis, args)
    summary = summarize(steering)
    contrasts = contrast_summary(summary)

    prompts.to_csv(args.out_dir / "prompts.csv", index=False)
    diagnostics.to_csv(args.out_dir / "projection_diagnostics.csv", index=False)
    steering.to_csv(args.out_dir / "steering_records.csv", index=False)
    summary.to_csv(args.out_dir / "steering_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "steering_contrasts.csv", index=False)
    write_report(args, prompts, diagnostics, summary, contrasts)
    print(args.out_dir)
    print((args.out_dir / "report.md").read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
