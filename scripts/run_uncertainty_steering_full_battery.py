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
from sklearn.linear_model import LinearRegression
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertTokenizer

from accessible_varentropy.bert_mlm import PromptCase, extended_prompt_cases
from accessible_varentropy.metrics import softmax_np
from accessible_varentropy.mlm_heads import mlm_logits_from_hidden
from run_factual_error_deep_benchmark import FactualCase, factual_error_cases


@dataclass(frozen=True)
class CaseSpec:
    prompt: str
    task: str
    target: str | None
    topic: str
    fact_id: str
    template_id: str


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
    parser.add_argument(
        "--models",
        default="distilbert-base-uncased,bert-base-uncased,roberta-base,google/bert_uncased_L-2_H-128_A-2",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "uncertainty_steering_full_battery")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-prompts-per-task", type=int, default=24)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--subspace-ks", default="1,4,8,16")
    parser.add_argument("--random-subspaces", type=int, default=3)
    parser.add_argument("--latent-eps", default="0.05,0.1")
    parser.add_argument("--output-eps", default="0.02,0.05,0.1")
    parser.add_argument("--semantic-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
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
    return vt[:max_k].T.copy().astype(np.float64)


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


def projection(matrix: np.ndarray, vector: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, int]:
    if matrix.size == 0:
        return np.zeros_like(vector), 0
    left, singular_values, _right = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros_like(vector), 0
    cutoff = 1e-10 * max(matrix.shape) * singular_values[0]
    active = singular_values > max(cutoff, eps)
    if not np.any(active):
        return np.zeros_like(vector), 0
    basis = left[:, active]
    return (basis @ (basis.T @ vector)).astype(np.float64), int(active.sum())


def least_squares(matrix: np.ndarray, target: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[1],), dtype=np.float64)
    return np.linalg.pinv(matrix, rcond=rcond) @ target


def token_id_for_word(tokenizer, word: str | None) -> int | None:
    if not word:
        return None
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        return None
    return int(ids[0])


def encode_prompt(tokenizer, prompt: str) -> tuple[dict[str, torch.Tensor], int]:
    encoded = tokenizer(prompt.replace("[MASK]", tokenizer.mask_token), return_tensors="pt")
    mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    if len(mask_positions) != 1:
        raise ValueError(f"Expected exactly one mask token: {prompt}")
    return encoded, int(mask_positions[0, 0].item())


def interleave_cases(cases: list[CaseSpec], max_per_task: int, seed: int) -> list[CaseSpec]:
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[CaseSpec]] = {}
    for case in cases:
        by_task.setdefault(case.task, []).append(case)
    sampled: dict[str, list[CaseSpec]] = {}
    for task, task_cases in by_task.items():
        by_topic: dict[str, list[CaseSpec]] = {}
        for case in task_cases:
            by_topic.setdefault(case.topic, []).append(case)
        chunks = []
        per_topic = max(1, max_per_task // max(1, len(by_topic)))
        for topic_cases in by_topic.values():
            rng.shuffle(topic_cases)
            chunks.extend(topic_cases[:per_topic])
        if len(chunks) < max_per_task:
            used = {case.prompt for case in chunks}
            remaining = [case for case in task_cases if case.prompt not in used]
            rng.shuffle(remaining)
            chunks.extend(remaining[: max_per_task - len(chunks)])
        rng.shuffle(chunks)
        sampled[task] = chunks[:max_per_task]
    ordered: list[CaseSpec] = []
    tasks = sorted(sampled)
    while any(sampled[task] for task in tasks):
        for task in tasks:
            if sampled[task]:
                ordered.append(sampled[task].pop())
    return ordered


def build_cases(seed: int, max_per_task: int) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    seen: set[tuple[str, str]] = set()
    for case in factual_error_cases():
        key = ("factual_deep", case.prompt)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            CaseSpec(
                prompt=case.prompt,
                task="factual_deep",
                target=case.target,
                topic=case.topic,
                fact_id=case.fact_id,
                template_id=case.template_id,
            )
        )
    for idx, case in enumerate(extended_prompt_cases()):
        task = "ambiguous_open" if case.condition == "ambiguous" else "factual_simple"
        key = (task, case.prompt)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            CaseSpec(
                prompt=case.prompt,
                task=task,
                target=case.target,
                topic=case.topic or "unknown",
                fact_id=f"{task}:{idx}",
                template_id="extended",
            )
        )
    return interleave_cases(cases, max_per_task, seed)


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "auto":
        return sorted({max(1, n_layers // 3), max(1, (2 * n_layers) // 3), n_layers})
    return sorted({int(part) for part in spec.split(",") if part.strip()})


def mlm_selected_logits(model, hidden: torch.Tensor, token_ids: np.ndarray) -> np.ndarray:
    parts = _distilbert_head_parts(model) or _bert_head_parts(model)
    if parts is not None:
        dense, activation, layer_norm, decoder = parts
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=hidden.device)
        with torch.no_grad():
            x = dense(hidden.view(1, 1, -1))
            if activation is not None:
                x = activation(x)
            x = layer_norm(x)[0, 0, :].detach().float()
            rows = decoder.weight.detach().float()[ids]
            logits = rows @ x
            bias = getattr(decoder, "bias", None)
            if bias is not None:
                logits = logits + bias.detach().float()[ids]
        return logits.detach().cpu().numpy().astype(np.float64)
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=hidden.device)
    with torch.no_grad():
        logits = mlm_logits_from_hidden(model, hidden.view(1, 1, -1))[0, 0, ids]
    return logits.detach().cpu().numpy().astype(np.float64)


def _bert_head_parts(model):
    if hasattr(model, "cls") and hasattr(model.cls, "predictions"):
        pred = model.cls.predictions
        return (
            pred.transform.dense,
            getattr(pred.transform, "transform_act_fn", None),
            pred.transform.LayerNorm,
            pred.decoder,
        )
    return None


def _distilbert_head_parts(model):
    if all(hasattr(model, name) for name in ["vocab_transform", "vocab_layer_norm", "vocab_projector"]):
        return (
            model.vocab_transform,
            getattr(model, "activation", None),
            model.vocab_layer_norm,
            model.vocab_projector,
        )
    return None


def mlm_head_jacobian_fast(model, hidden: torch.Tensor, token_ids: np.ndarray) -> np.ndarray:
    parts = _distilbert_head_parts(model) or _bert_head_parts(model)
    if parts is None:
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=hidden.device)
        base = hidden.detach().clone().requires_grad_(True)

        def selected(vec: torch.Tensor) -> torch.Tensor:
            return mlm_logits_from_hidden(model, vec.view(1, 1, -1))[0, 0, ids]

        jac = torch.autograd.functional.jacobian(selected, base, create_graph=False, strict=False, vectorize=True)
        return jac.detach().cpu().numpy().astype(np.float64)

    dense, activation, layer_norm, decoder = parts
    h = hidden.detach().float()
    weight1 = dense.weight.detach().float()
    bias1 = dense.bias.detach().float() if dense.bias is not None else 0.0
    pre = (weight1 @ h + bias1).detach().requires_grad_(True)
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

    ids = torch.as_tensor(token_ids, dtype=torch.long, device=decoder.weight.device)
    rows = decoder.weight.detach().float()[ids]
    gamma = layer_norm.weight.detach().float()
    upstream = rows * gamma
    upstream_mean = upstream.mean(dim=1, keepdim=True)
    upstream_xhat_mean = (upstream * xhat).mean(dim=1, keepdim=True)
    after_layer_norm = (upstream - upstream_mean - xhat * upstream_xhat_mean) / std
    after_activation = after_layer_norm * activation_grad
    jac = after_activation @ weight1
    return jac.detach().cpu().numpy().astype(np.float64)


def full_logits(model, hidden: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = mlm_logits_from_hidden(model, hidden.view(1, 1, -1))[0, 0, :]
    return logits.detach().cpu().numpy().astype(np.float64)


def rank_of_token(logits: np.ndarray, token_id: int | None) -> float:
    if token_id is None:
        return float("nan")
    return float(1 + np.sum(logits > logits[int(token_id)]))


def token_cluster_labels(embeddings: np.ndarray, token_ids: np.ndarray, n_clusters: int) -> np.ndarray:
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


def cluster_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    masses = np.array([float(probs[labels == label].sum()) for label in sorted(set(labels.tolist()))], dtype=np.float64)
    return entropy_varentropy(masses)[0]


def cluster_l1(probs_before: np.ndarray, probs_after: np.ndarray, labels: np.ndarray) -> float:
    before = np.array([float(probs_before[labels == label].sum()) for label in sorted(set(labels.tolist()))], dtype=np.float64)
    after = np.array([float(probs_after[labels == label].sum()) for label in sorted(set(labels.tolist()))], dtype=np.float64)
    return float(np.sum(np.abs(after - before)))


def rank_displacement(probs_before: np.ndarray, probs_after: np.ndarray) -> float:
    before_order = np.argsort(-probs_before)
    after_order = np.argsort(-probs_after)
    before_rank = np.empty_like(before_order)
    after_rank = np.empty_like(after_order)
    before_rank[before_order] = np.arange(len(before_order))
    after_rank[after_order] = np.arange(len(after_order))
    return float(np.mean(np.abs(after_rank - before_rank)) / max(1, len(before_order) - 1))


def top_jaccard(probs_before: np.ndarray, probs_after: np.ndarray, k: int = 5) -> float:
    kk = min(k, len(probs_before))
    left = set(np.argsort(-probs_before)[:kk].tolist())
    right = set(np.argsort(-probs_after)[:kk].tolist())
    return float(len(left & right) / max(1, len(left | right)))


def build_records(tokenizer, model, cases: list[CaseSpec], top_k: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for prompt_id, case in enumerate(cases):
        target_id = token_id_for_word(tokenizer, case.target)
        if case.target and target_id is None:
            continue
        try:
            encoded, mask_index = encode_prompt(tokenizer, case.prompt)
        except ValueError:
            continue
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)
            hidden_states = tuple(state.detach() for state in output.hidden_states)
            logits = output.logits[0, mask_index, :].detach().cpu().numpy().astype(np.float64)
        token_ids, probs = topk_distribution(logits, top_k)
        top1_id = int(token_ids[0])
        if case.task.startswith("ambiguous"):
            observed = "ambiguous"
        else:
            observed = "correct" if target_id is not None and top1_id == int(target_id) else "error"
        rows.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": case.prompt,
                "task": case.task,
                "topic": case.topic,
                "fact_id": case.fact_id,
                "template_id": case.template_id,
                "target": case.target,
                "target_id": target_id,
                "observed_condition": observed,
                "final_top1_token": tokenizer.convert_ids_to_tokens([top1_id])[0],
                "final_top1_prob_topk": float(probs[0]),
                "target_rank_final": rank_of_token(logits, target_id),
            }
        )
        records.append({"prompt_id": int(prompt_id), "mask_index": int(mask_index), "hidden_states": hidden_states})
    return pd.DataFrame(rows), records


def model_subspaces(
    records: list[dict[str, object]],
    layer: int,
    ks: list[int],
    random_reps: int,
    seed: int,
) -> list[tuple[str, int, int, np.ndarray]]:
    hidden_dim = int(records[0]["hidden_states"][layer].shape[-1])
    max_k = min(max(ks), hidden_dim, max(1, len(records) - 1))
    states = []
    deltas = []
    for record in records:
        current = record["hidden_states"][layer][0, int(record["mask_index"]), :].detach().cpu().numpy().astype(np.float64)
        previous = record["hidden_states"][layer - 1][0, int(record["mask_index"]), :].detach().cpu().numpy().astype(np.float64)
        states.append(current)
        deltas.append(current - previous)
    state_pca = pca_basis(np.vstack(states), max_k)
    delta_pca = pca_basis(np.vstack(deltas), max_k)
    rng = np.random.default_rng(seed + 1009 * layer)
    result: list[tuple[str, int, int, np.ndarray]] = []
    for k in ks:
        if k > max_k:
            continue
        result.append(("state_pca", k, 0, state_pca[:, :k]))
        result.append(("delta_pca", k, 0, delta_pca[:, :k]))
        for rep in range(random_reps):
            result.append(("random", k, rep, orthonormalize(rng.normal(size=(hidden_dim, k)), k)))
    return result


def z_control_orthogonal(rng: np.random.Generator, grad_z: np.ndarray, k: int) -> np.ndarray:
    for _ in range(100):
        raw = rng.normal(size=k)
        denom = float(grad_z @ grad_z)
        if denom > 1e-12:
            raw = raw - grad_z * float(raw @ grad_z) / denom
        norm = float(np.linalg.norm(raw))
        if norm > 1e-12:
            return (raw / norm).astype(np.float64)
    return np.zeros((k,), dtype=np.float64)


def intervention_metrics(
    *,
    model,
    hidden: torch.Tensor,
    token_ids: np.ndarray,
    target_id: int | None,
    embeddings: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    geometry: FisherGeometry,
    selected_logits: np.ndarray,
    hidden_direction: np.ndarray,
    epsilon: float,
    sign: float,
) -> dict[str, float | int]:
    after_logits = mlm_selected_logits(model, hidden + torch.as_tensor(sign * epsilon * hidden_direction, dtype=hidden.dtype), token_ids)
    probs_after = softmax_np(after_logits[None, :])[0]
    entropy_after, varentropy_after = entropy_varentropy(probs_after)
    top1_before = int(token_ids[0])
    top1_after = int(token_ids[int(np.argmax(probs_after))])
    target_positions = np.where(token_ids == int(target_id))[0] if target_id is not None else np.array([], dtype=np.int64)
    target_in_topk = int(len(target_positions) > 0)
    target_top1_before = int(target_id is not None and top1_before == int(target_id))
    target_top1_after = int(target_id is not None and top1_after == int(target_id))
    target_prob_before = float(probs[int(target_positions[0])]) if target_in_topk else float("nan")
    target_prob_after = float(probs_after[int(target_positions[0])]) if target_in_topk else float("nan")
    centroid_before = probs @ embeddings[token_ids]
    centroid_after = probs_after @ embeddings[token_ids]
    kl = float(np.sum(probs * (np.log(np.clip(probs, 1e-12, 1.0)) - np.log(np.clip(probs_after, 1e-12, 1.0)))))
    return {
        "entropy_after": entropy_after,
        "delta_entropy": entropy_after - geometry.entropy,
        "abs_delta_entropy": abs(entropy_after - geometry.entropy),
        "varentropy_after": varentropy_after,
        "delta_varentropy": varentropy_after - geometry.varentropy,
        "abs_delta_varentropy": abs(varentropy_after - geometry.varentropy),
        "semantic_entropy_proxy_after": cluster_entropy(probs_after, labels),
        "delta_semantic_entropy_proxy": cluster_entropy(probs_after, labels) - cluster_entropy(probs, labels),
        "cluster_distribution_l1": cluster_l1(probs, probs_after, labels),
        "embedding_centroid_delta_norm": float(np.linalg.norm(centroid_after - centroid_before)),
        "selected_logit_l2": float(np.linalg.norm(after_logits - selected_logits)),
        "selected_prob_l1": float(np.sum(np.abs(probs_after - probs))),
        "selected_kl_before_after": kl,
        "rank_displacement_mean": rank_displacement(probs, probs_after),
        "top5_jaccard": top_jaccard(probs, probs_after, 5),
        "selected_top1_changed": int(top1_after != top1_before),
        "target_in_topk": target_in_topk,
        "target_correct_changed": int(target_top1_before != target_top1_after),
        "delta_target_prob_topk": target_prob_after - target_prob_before if target_in_topk else float("nan"),
        "non_target_mass_delta": -(target_prob_after - target_prob_before) if target_in_topk else float("nan"),
    }


def run_model(args: argparse.Namespace, model_name: str, cases: list[CaseSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"loading_model {model_name}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
    except ValueError:
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )
    model.eval()
    prompt_table, records = build_records(tokenizer, model, cases, args.top_k)
    if prompt_table.empty:
        return prompt_table, pd.DataFrame(), pd.DataFrame()
    n_layers = len(records[0]["hidden_states"]) - 1
    layers = [layer for layer in parse_layers(args.layers, n_layers) if 1 <= layer <= n_layers]
    ks = [int(part) for part in args.subspace_ks.split(",") if part.strip()]
    latent_eps = [float(part) for part in args.latent_eps.split(",") if part.strip()]
    output_eps = [float(part) for part in args.output_eps.split(",") if part.strip()]
    rng = np.random.default_rng(args.seed)
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    steering_rows: list[dict[str, object]] = []
    subspace_rows: list[dict[str, object]] = []
    prompt_lookup = prompt_table.set_index("prompt_id")

    for layer in layers:
        print(f"model_layer {model_name} {layer}", flush=True)
        subspaces = model_subspaces(records, layer, ks, args.random_subspaces, args.seed)
        for rec_idx, record in enumerate(records):
            if rec_idx % 20 == 0:
                print(f"model_layer_prompt {model_name} {layer} {rec_idx + 1}/{len(records)}", flush=True)
            meta = prompt_lookup.loc[int(record["prompt_id"])]
            hidden = record["hidden_states"][layer][0, int(record["mask_index"]), :].detach()
            logits = full_logits(model, hidden)
            token_ids, probs = topk_distribution(logits, args.top_k)
            selected_logits = logits[token_ids]
            geometry = fisher_geometry(probs)
            jac = mlm_head_jacobian_fast(model, hidden, token_ids)
            labels = token_cluster_labels(embeddings, token_ids, args.semantic_clusters)
            semantic_before = cluster_entropy(probs, labels)
            target_id = None if pd.isna(meta["target_id"]) else int(meta["target_id"])
            logit_norm = float(np.linalg.norm(selected_logits))
            for family, k, rep, basis in subspaces:
                jb = jac @ basis
                whitened = geometry.sqrt_fisher @ jb
                w_acc, rank = projection(whitened, geometry.w)
                w_orth = geometry.w - w_acc
                rho = float((w_acc @ w_acc) / max(float(geometry.w @ geometry.w), 1e-12))
                grad_z = whitened.T @ geometry.w
                trace_fim = float(np.trace(jb.T @ geometry.fisher @ jb))
                grad_norm_sq = float(grad_z @ grad_z)
                z_acc = least_squares(whitened, w_acc)
                z_acc_unit = normalize(z_acc)
                z_strict = least_squares(whitened, w_orth)
                z_strict_unit = normalize(z_strict)
                z_grad_orth = z_control_orthogonal(rng, grad_z, basis.shape[1])
                z_random = normalize(rng.normal(size=basis.shape[1]))
                directions = {
                    "accessible_ls": z_acc_unit,
                    "grad_orthogonal_control": z_grad_orth,
                    "random_control": z_random,
                    "strict_orthogonal_ls": z_strict_unit,
                }
                subspace_rows.append(
                    {
                        "model": model_name,
                        "prompt_id": int(record["prompt_id"]),
                        "task": meta["task"],
                        "topic": meta["topic"],
                        "observed_condition": meta["observed_condition"],
                        "layer": layer,
                        "subspace_family": family,
                        "k": k,
                        "rep": rep,
                        "rank_whitened_jb": rank,
                        "rho": rho,
                        "v_access": float(w_acc @ w_acc),
                        "v_inaccess": float(w_orth @ w_orth),
                        "entropy_before": geometry.entropy,
                        "varentropy_before": geometry.varentropy,
                        "semantic_entropy_proxy_before": semantic_before,
                        "trace_fim": trace_fim,
                        "grad_norm_sq": grad_norm_sq,
                        "logit_norm": logit_norm,
                        "strict_orth_ls_norm": float(np.linalg.norm(z_strict)),
                        "accessible_ls_norm": float(np.linalg.norm(z_acc)),
                    }
                )
                for direction_name, z_unit in directions.items():
                    if float(np.linalg.norm(z_unit)) < 1e-12:
                        continue
                    output_energy_unit = float(np.linalg.norm(whitened @ z_unit))
                    first_order = float(grad_z @ z_unit)
                    sign_base = 1.0 if first_order >= 0.0 else -1.0
                    hidden_direction = basis @ z_unit
                    for mode, eps_values in [("latent_equal", latent_eps), ("fisher_output_equal", output_eps)]:
                        for eps in eps_values:
                            if mode == "latent_equal":
                                step_scale = eps
                                realized_output_energy = eps * output_energy_unit
                            else:
                                if output_energy_unit < 1e-12:
                                    continue
                                step_scale = eps / output_energy_unit
                                realized_output_energy = eps
                            for sign_name, sign_multiplier in [("increase", 1.0), ("decrease", -1.0)]:
                                signed = sign_multiplier * sign_base
                                metrics = intervention_metrics(
                                    model=model,
                                    hidden=hidden,
                                    token_ids=token_ids,
                                    target_id=target_id,
                                    embeddings=embeddings,
                                    labels=labels,
                                    probs=probs,
                                    geometry=geometry,
                                    selected_logits=selected_logits,
                                    hidden_direction=hidden_direction,
                                    epsilon=step_scale,
                                    sign=signed,
                                )
                                steering_rows.append(
                                    {
                                        "model": model_name,
                                        "prompt_id": int(record["prompt_id"]),
                                        "task": meta["task"],
                                        "topic": meta["topic"],
                                        "observed_condition": meta["observed_condition"],
                                        "layer": layer,
                                        "subspace_family": family,
                                        "k": k,
                                        "rep": rep,
                                        "mode": mode,
                                        "direction": direction_name,
                                        "sign": sign_name,
                                        "epsilon": eps,
                                        "latent_step_norm": step_scale,
                                        "fisher_output_energy": realized_output_energy,
                                        "direction_output_energy_unit": output_energy_unit,
                                        "rho": rho,
                                        "entropy_before": geometry.entropy,
                                        "varentropy_before": geometry.varentropy,
                                        "trace_fim": trace_fim,
                                        "grad_norm_sq": grad_norm_sq,
                                        "logit_norm": logit_norm,
                                        "first_order_entropy": signed * step_scale * first_order,
                                        **metrics,
                                    }
                                )
    return prompt_table.assign(model=model_name), pd.DataFrame(subspace_rows), pd.DataFrame(steering_rows)


def summarize_directionality(steering: pd.DataFrame) -> pd.DataFrame:
    frame = steering[(steering["direction"].eq("accessible_ls")) & (steering["mode"].eq("fisher_output_equal"))].copy()
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["model", "task", "layer", "subspace_family", "k", "prompt_id", "sign"]):
        group = group.sort_values("epsilon")
        deltas = group["delta_entropy"].to_numpy(dtype=float)
        eps = group["epsilon"].to_numpy(dtype=float)
        expected_sign = 1.0 if keys[-1] == "increase" else -1.0
        directional_ok = bool(np.all(expected_sign * deltas > 0.0))
        monotonic_ok = bool(np.all(np.diff(np.abs(deltas)) >= -1e-10))
        slope = float(np.polyfit(eps, deltas, 1)[0]) if len(group) >= 2 else float("nan")
        rows.append(
            {
                "model": keys[0],
                "task": keys[1],
                "layer": int(keys[2]),
                "subspace_family": keys[3],
                "k": int(keys[4]),
                "prompt_id": int(keys[5]),
                "sign": keys[6],
                "directional_ok": directional_ok,
                "monotonic_abs_ok": monotonic_ok,
                "entropy_slope": slope,
            }
        )
    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return detailed
    return detailed.groupby(["model", "task", "layer", "subspace_family", "k", "sign"], as_index=False).agg(
        n=("prompt_id", "count"),
        directional_rate=("directional_ok", "mean"),
        monotonic_abs_rate=("monotonic_abs_ok", "mean"),
        entropy_slope_mean=("entropy_slope", "mean"),
    )


def summarize_equal_output(steering: pd.DataFrame) -> pd.DataFrame:
    frame = steering[steering["mode"].eq("fisher_output_equal")].copy()
    summary = frame.groupby(["direction", "sign", "epsilon"], as_index=False).agg(
        n=("prompt_id", "count"),
        fisher_output_energy_mean=("fisher_output_energy", "mean"),
        latent_step_norm_mean=("latent_step_norm", "mean"),
        abs_delta_entropy_mean=("abs_delta_entropy", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
        selected_top1_changed_rate=("selected_top1_changed", "mean"),
        target_correct_changed_rate=("target_correct_changed", "mean"),
        embedding_centroid_delta_norm_mean=("embedding_centroid_delta_norm", "mean"),
        cluster_distribution_l1_mean=("cluster_distribution_l1", "mean"),
        rank_displacement_mean=("rank_displacement_mean", "mean"),
    )
    rows = []
    indexed = summary.set_index(["direction", "sign", "epsilon"])
    for sign in sorted(summary["sign"].unique()):
        for eps in sorted(summary["epsilon"].unique()):
            acc_key = ("accessible_ls", sign, eps)
            if acc_key not in indexed.index:
                continue
            acc = indexed.loc[acc_key]
            for control in ["grad_orthogonal_control", "random_control", "strict_orthogonal_ls"]:
                key = (control, sign, eps)
                if key not in indexed.index:
                    continue
                ctrl = indexed.loc[key]
                for metric in ["abs_delta_entropy_mean", "abs_delta_varentropy_mean"]:
                    denom = float(ctrl[metric])
                    rows.append(
                        {
                            "sign": sign,
                            "epsilon": eps,
                            "control": control,
                            "metric": metric,
                            "accessible": float(acc[metric]),
                            "control_value": denom,
                            "accessible_minus_control": float(acc[metric] - denom),
                            "accessible_over_control": float(acc[metric] / denom) if abs(denom) > 1e-12 else math.inf,
                            "accessible_output_energy": float(acc["fisher_output_energy_mean"]),
                            "control_output_energy": float(ctrl["fisher_output_energy_mean"]),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_specificity(steering: pd.DataFrame) -> pd.DataFrame:
    frame = steering[(steering["direction"].eq("accessible_ls")) & (steering["mode"].eq("fisher_output_equal"))].copy()
    return frame.groupby(["model", "task", "sign", "epsilon"], as_index=False).agg(
        n=("prompt_id", "count"),
        abs_delta_entropy_mean=("abs_delta_entropy", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
        selected_top1_changed_rate=("selected_top1_changed", "mean"),
        target_in_topk_rate=("target_in_topk", "mean"),
        target_correct_changed_rate=("target_correct_changed", "mean"),
        embedding_centroid_delta_norm_mean=("embedding_centroid_delta_norm", "mean"),
        cluster_distribution_l1_mean=("cluster_distribution_l1", "mean"),
        rank_displacement_mean=("rank_displacement_mean", "mean"),
        top5_jaccard_mean=("top5_jaccard", "mean"),
        selected_prob_l1_mean=("selected_prob_l1", "mean"),
        selected_kl_mean=("selected_kl_before_after", "mean"),
        abs_non_target_mass_delta_mean=("non_target_mass_delta", lambda s: float(np.nanmean(np.abs(s)))),
    )


def partial_corr(frame: pd.DataFrame, outcome: str, controls: list[str]) -> dict[str, float | int | str]:
    cols = ["rho", outcome] + controls
    local = frame[cols + ["model", "task", "layer", "subspace_family"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(local) < 20 or local["rho"].nunique() < 3 or local[outcome].nunique() < 3:
        return {"outcome": outcome, "n": int(len(local)), "partial_corr": float("nan"), "rho_beta": float("nan")}
    dummies = pd.get_dummies(local[["model", "task", "layer", "subspace_family"]].astype(str), drop_first=True)
    x_controls = pd.concat([local[controls].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1).to_numpy(dtype=float)
    y = local[outcome].to_numpy(dtype=float)
    rho = local["rho"].to_numpy(dtype=float)
    y_resid = y - LinearRegression().fit(x_controls, y).predict(x_controls)
    rho_resid = rho - LinearRegression().fit(x_controls, rho).predict(x_controls)
    corr = float(np.corrcoef(rho_resid, y_resid)[0, 1])
    full_x = np.column_stack([rho, x_controls])
    beta = float(LinearRegression().fit(full_x, y).coef_[0])
    return {"outcome": outcome, "n": int(len(local)), "partial_corr": corr, "rho_beta": beta}


def summarize_rho_dependency(steering: pd.DataFrame) -> pd.DataFrame:
    frame = steering[
        steering["mode"].eq("fisher_output_equal")
        & steering["direction"].eq("accessible_ls")
        & steering["sign"].isin(["increase", "decrease"])
    ].copy()
    controls = [
        "entropy_before",
        "varentropy_before",
        "trace_fim",
        "grad_norm_sq",
        "logit_norm",
        "k",
        "fisher_output_energy",
    ]
    rows = []
    for label, local in [("all", frame)] + [(f"model:{m}", g) for m, g in frame.groupby("model")]:
        for outcome in ["abs_delta_entropy", "abs_delta_varentropy", "embedding_centroid_delta_norm", "selected_top1_changed"]:
            row = partial_corr(local, outcome, controls)
            row["subset"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_replication(prompt_tables: pd.DataFrame, subspaces: pd.DataFrame, steering: pd.DataFrame) -> pd.DataFrame:
    coverage = subspaces.groupby(["model", "layer", "subspace_family", "k"], as_index=False).agg(
        n_subspace_prompt_rows=("prompt_id", "count"),
        rho_mean=("rho", "mean"),
        rho_std=("rho", "std"),
    )
    interventions = steering.groupby(["model", "mode"], as_index=False).agg(
        n_interventions=("prompt_id", "count"),
    )
    coverage = coverage.merge(interventions, on="model", how="left")
    return coverage


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def write_report(
    out_dir: Path,
    prompt_tables: pd.DataFrame,
    directionality: pd.DataFrame,
    equal_output: pd.DataFrame,
    specificity: pd.DataFrame,
    rho_dependency: pd.DataFrame,
    replication: pd.DataFrame,
    skipped: pd.DataFrame,
) -> None:
    prompt_counts = prompt_tables.groupby(["model", "task", "observed_condition"], as_index=False).size()
    lines = [
        "# Uncertainty Steering Full Battery",
        "",
        "This suite targets five checks: directional monotonicity, equal Fisher-output-energy controls, specificity, rho dependency after scalar controls, and replication across available local MLM models/tasks/layers/subspace dimensions/random seeds.",
        "",
        "The default model set includes RoBERTa. Missing models are downloaded automatically unless `--local-files-only` is set; unavailable models are recorded in `skipped_models.csv`.",
        "",
        "## Skipped Models",
        markdown_table(skipped, 20),
        "",
        "## Prompt Counts",
        markdown_table(prompt_counts, 80),
        "",
        "## 1. Directional And Monotonic Control",
        markdown_table(directionality.sort_values(["model", "task", "layer", "subspace_family", "k"]), 80),
        "",
        "## 2. Equal Fisher-Output-Energy Contrasts",
        markdown_table(equal_output.sort_values(["epsilon", "sign", "control", "metric"]), 80),
        "",
        "## 3. Specificity",
        markdown_table(specificity.sort_values(["model", "task", "epsilon", "sign"]), 80),
        "",
        "## 4. Rho Dependency After Controls",
        markdown_table(rho_dependency.sort_values(["subset", "outcome"]), 80),
        "",
        "## 5. Replication Matrix",
        markdown_table(replication.sort_values(["model", "layer", "subspace_family", "k"]), 120),
        "",
        "## Files",
        "```text",
        "prompt_tables.csv",
        "subspace_scores.csv",
        "steering_records.csv",
        "directionality_summary.csv",
        "equal_output_energy_contrasts.csv",
        "specificity_summary.csv",
        "rho_dependency.csv",
        "replication_matrix.csv",
        "report.md",
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    cases = build_cases(args.seed, args.max_prompts_per_task)
    all_prompts: list[pd.DataFrame] = []
    all_subspaces: list[pd.DataFrame] = []
    all_steering: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, str]] = []
    for model_name in [part.strip() for part in args.models.split(",") if part.strip()]:
        try:
            prompts, subspaces, steering = run_model(args, model_name, cases)
        except Exception as exc:  # noqa: BLE001
            print(f"skipping_model {model_name}: {exc}", flush=True)
            skipped_rows.append({"model": model_name, "reason": str(exc)})
            continue
        all_prompts.append(prompts)
        all_subspaces.append(subspaces)
        all_steering.append(steering)
    if not all_steering:
        raise RuntimeError("No model completed successfully.")
    prompt_tables = pd.concat(all_prompts, ignore_index=True)
    subspace_scores = pd.concat(all_subspaces, ignore_index=True)
    steering_records = pd.concat(all_steering, ignore_index=True)

    directionality = summarize_directionality(steering_records)
    equal_output = summarize_equal_output(steering_records)
    specificity = summarize_specificity(steering_records)
    rho_dependency = summarize_rho_dependency(steering_records)
    replication = summarize_replication(prompt_tables, subspace_scores, steering_records)
    skipped = pd.DataFrame(skipped_rows)

    prompt_tables.to_csv(args.out_dir / "prompt_tables.csv", index=False)
    subspace_scores.to_csv(args.out_dir / "subspace_scores.csv", index=False)
    steering_records.to_csv(args.out_dir / "steering_records.csv", index=False)
    directionality.to_csv(args.out_dir / "directionality_summary.csv", index=False)
    equal_output.to_csv(args.out_dir / "equal_output_energy_contrasts.csv", index=False)
    specificity.to_csv(args.out_dir / "specificity_summary.csv", index=False)
    rho_dependency.to_csv(args.out_dir / "rho_dependency.csv", index=False)
    replication.to_csv(args.out_dir / "replication_matrix.csv", index=False)
    skipped.to_csv(args.out_dir / "skipped_models.csv", index=False)
    write_report(args.out_dir, prompt_tables, directionality, equal_output, specificity, rho_dependency, replication, skipped)
    print(args.out_dir)
    print((args.out_dir / "report.md").read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
