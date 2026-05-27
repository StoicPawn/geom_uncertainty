from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from accessible_varentropy.metrics import softmax_np


ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt-csv", type=Path, default=ROOT / "results" / "factual_error_deep_distilbert_top32" / "prompts.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "decoder_uncertainty_steering_qwen2_5_0_5b")
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--eps", default="0.05,0.1,0.2")
    parser.add_argument("--semantic-clusters", type=int, default=4)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args()


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v, dtype=np.float64)
    return (v / norm).astype(np.float64)


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


def orthogonal_basis_projection(matrix: np.ndarray, vector: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, int]:
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


def semantic_entropy_proxy(probs: np.ndarray, labels: np.ndarray) -> float:
    masses = []
    for label in sorted(set(labels.tolist())):
        masses.append(float(probs[labels == label].sum()))
    return entropy_varentropy(np.asarray(masses, dtype=np.float64))[0]


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


def causal_prefix(mask_prompt: str) -> str:
    return mask_prompt.split("[MASK]")[0].rstrip()


def candidate_token_ids(tokenizer, target: str) -> set[int]:
    variants = {
        target,
        target.lower(),
        target.capitalize(),
        target.title(),
        " " + target,
        " " + target.capitalize(),
        " " + target.title(),
    }
    ids: set[int] = set()
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded:
            ids.add(int(encoded[0]))
    return ids


def choose_prompts(path: Path, max_prompts: int, seed: int) -> pd.DataFrame:
    prompts = pd.read_csv(path)
    prompts = prompts[prompts["prompt"].str.contains(r"\[MASK\]", regex=True)].copy()
    prompts = prompts.drop_duplicates(["fact_id", "template_id"]).reset_index(drop=True)
    if len(prompts) <= max_prompts:
        return prompts
    per_topic = max(1, max_prompts // max(1, prompts["topic"].nunique()))
    chunks = []
    for _topic, group in prompts.groupby("topic", sort=True):
        chunks.append(group.sample(n=min(per_topic, len(group)), random_state=seed))
    sampled = pd.concat(chunks).drop_duplicates("prompt_id")
    if len(sampled) < max_prompts:
        remaining = prompts[~prompts["prompt_id"].isin(sampled["prompt_id"])]
        sampled = pd.concat(
            [
                sampled,
                remaining.sample(n=min(max_prompts - len(sampled), len(remaining)), random_state=seed + 1),
            ]
        )
    return sampled.sort_values("prompt_id").head(max_prompts).reset_index(drop=True)


def get_final_norm(model):
    candidates = [
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    ]
    for parent_name, child_name in candidates:
        parent = getattr(model, parent_name, None)
        if parent is not None and hasattr(parent, child_name):
            return getattr(parent, child_name)
    return torch.nn.Identity()


def lens_logits_from_hidden(hidden: torch.Tensor, norm, lm_head) -> torch.Tensor:
    x = norm(hidden.reshape(1, 1, -1))[0, 0, :]
    return lm_head(x)


def selected_lens_logits(hidden: torch.Tensor, norm, lm_head, token_ids: torch.Tensor) -> torch.Tensor:
    x = norm(hidden.reshape(1, 1, -1))[0, 0, :]
    weight = lm_head.weight[token_ids]
    logits = weight @ x
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[token_ids]
    return logits


def jacobian_selected(hidden: torch.Tensor, norm, lm_head, token_ids: torch.Tensor) -> np.ndarray:
    base = hidden.detach().float().requires_grad_(True)

    def fn(vec: torch.Tensor) -> torch.Tensor:
        return selected_lens_logits(vec, norm, lm_head, token_ids).float()

    jac = torch.autograd.functional.jacobian(
        fn,
        base,
        create_graph=False,
        strict=False,
        vectorize=True,
    )
    return jac.detach().cpu().numpy().astype(np.float64)


def selected_logits_after_step(
    hidden: torch.Tensor,
    direction: np.ndarray,
    epsilon: float,
    norm,
    lm_head,
    token_ids: torch.Tensor,
) -> np.ndarray:
    direction_t = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
    stepped = hidden + float(epsilon) * direction_t
    with torch.no_grad():
        logits = selected_lens_logits(stepped, norm, lm_head, token_ids).float()
    return logits.detach().cpu().numpy().astype(np.float64)


def rank_of_token(logits: np.ndarray, token_id: int) -> int:
    return int(1 + np.sum(logits > logits[int(token_id)]))


def build_prompt_records(
    tokenizer,
    model,
    norm,
    lm_head,
    prompts: pd.DataFrame,
    layer: int,
    top_m: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for idx, row in prompts.iterrows():
        if idx % 10 == 0:
            print(f"encoding_prompt {idx + 1}/{len(prompts)}", flush=True)
        prefix = causal_prefix(str(row["prompt"]))
        if not prefix:
            continue
        target_ids = candidate_token_ids(tokenizer, str(row["target"]))
        encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        if encoded["input_ids"].numel() == 0 or not target_ids:
            continue
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            if layer + 1 >= len(hidden_states):
                raise ValueError(f"Layer {layer} needs hidden_states[{layer + 1}], but only {len(hidden_states)} states exist.")
            hidden = hidden_states[layer][0, -1, :].detach().float()
            next_hidden = hidden_states[layer + 1][0, -1, :].detach().float()
            logits = lens_logits_from_hidden(hidden, norm, lm_head).detach().float()
            top = torch.topk(logits, k=min(top_m, logits.numel()))
            top_ids = top.indices.detach().cpu().numpy().astype(np.int64)
            probs = softmax_np(top.values.detach().cpu().numpy()[None, :])[0]
        top1_id = int(top_ids[0])
        observed = "correct" if top1_id in target_ids else "error"
        rows.append(
            {
                "prompt_id": int(row["prompt_id"]),
                "prompt": row["prompt"],
                "causal_prefix": prefix,
                "target": row["target"],
                "target_token_ids": " ".join(str(tid) for tid in sorted(target_ids)),
                "topic": row["topic"],
                "fact_id": row["fact_id"],
                "template_id": row.get("template_id", ""),
                "observed_condition": observed,
                "top1_token_id": top1_id,
                "top1_token": tokenizer.decode([top1_id]).strip(),
                "top1_prob_topm": float(probs[0]),
                "entropy_topm": entropy_varentropy(probs)[0],
                "varentropy_topm": entropy_varentropy(probs)[1],
                "target_rank_lens": min(rank_of_token(logits.detach().cpu().numpy().astype(np.float64), tid) for tid in target_ids),
            }
        )
        records.append(
            {
                "prompt_id": int(row["prompt_id"]),
                "hidden": hidden,
                "next_hidden": next_hidden,
                "top_ids": top_ids,
                "target_ids": sorted(target_ids),
            }
        )
    return pd.DataFrame(rows), records


def build_pca_subspace(records: list[dict[str, object]], k: int) -> np.ndarray:
    deltas = []
    for record in records:
        delta = (record["next_hidden"] - record["hidden"]).detach().cpu().numpy().astype(np.float64)
        deltas.append(delta)
    features = np.vstack(deltas)
    dim = min(k, features.shape[0] - 1, features.shape[1])
    if dim < 1:
        raise RuntimeError("Need at least two prompts to build a PCA subspace.")
    centered = features - features.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:dim].T.copy().astype(np.float64)


def compute_steering(
    tokenizer,
    model,
    norm,
    lm_head,
    prompts: pd.DataFrame,
    records: list[dict[str, object]],
    basis: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    eps_values = [float(part.strip()) for part in args.eps.split(",") if part.strip()]
    prompt_lookup = prompts.set_index("prompt_id")
    embedding_weight = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    steering_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for idx, record in enumerate(records):
        if idx % 10 == 0:
            print(f"steering_prompt {idx + 1}/{len(records)}", flush=True)
        meta = prompt_lookup.loc[int(record["prompt_id"])]
        hidden = record["hidden"].detach()
        top_ids = np.asarray(record["top_ids"], dtype=np.int64)
        target_ids = set(int(tid) for tid in record["target_ids"])
        top_ids_t = torch.as_tensor(top_ids, dtype=torch.long, device=hidden.device)

        with torch.no_grad():
            selected_before = selected_lens_logits(hidden, norm, lm_head, top_ids_t).float()
        probs = softmax_np(selected_before.detach().cpu().numpy()[None, :])[0]
        geometry = fisher_geometry(probs)
        jac = jacobian_selected(hidden, norm, lm_head, top_ids_t)
        jb = jac @ basis
        whitened_jb = geometry.sqrt_fisher @ jb
        w_acc, rank = orthogonal_basis_projection(whitened_jb, geometry.w)
        w_orth = geometry.w - w_acc
        rho = float((w_acc @ w_acc) / max(float(geometry.w @ geometry.w), 1e-12))

        z_acc = least_squares(whitened_jb, w_acc)
        z_orth_strict = least_squares(whitened_jb, w_orth)
        grad_z = whitened_jb.T @ geometry.w
        z_grad_orth = control_direction_orthogonal_to_entropy_gradient(rng, grad_z, basis.shape[1])
        z_random = normalize(rng.normal(size=basis.shape[1]))
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
        target_positions = [idx for idx, token_id in enumerate(top_ids.tolist()) if int(token_id) in target_ids]
        target_in_topm = int(len(target_positions) > 0)
        target_prob_before = float(sum(probs[pos] for pos in target_positions)) if target_in_topm else 0.0
        target_top1_before = int(int(top_ids[0]) in target_ids)
        cluster_labels = token_cluster_labels(embedding_weight, top_ids, args.semantic_clusters)
        semantic_before = semantic_entropy_proxy(probs, cluster_labels)

        diagnostic_rows.append(
            {
                "prompt_id": int(record["prompt_id"]),
                "layer": int(args.layer),
                "pca_dim": int(basis.shape[1]),
                "output_top_m": int(args.top_m),
                "rank_whitened_jb": int(rank),
                "entropy_before": geometry.entropy,
                "varentropy_before": geometry.varentropy,
                "rho_accessible": rho,
                "v_access": float(w_acc @ w_acc),
                "v_inaccess": float(w_orth @ w_orth),
                "accessible_ls_norm": z_acc_norm,
                "strict_orth_ls_norm": z_orth_norm,
                "accessible_target_residual": float(np.linalg.norm(whitened_jb @ z_acc - w_acc)),
                "strict_orth_target_residual": float(np.linalg.norm(whitened_jb @ z_orth_strict - w_orth)),
            }
        )

        for direction_name, z_dir in directions.items():
            hidden_direction = basis @ z_dir
            first_order_entropy = float(grad_z @ z_dir)
            first_order_logit_norm = float(np.linalg.norm(jb @ z_dir))
            for sign_name, sign in [("increase", 1.0), ("decrease", -1.0)]:
                signed_direction = sign * hidden_direction
                for epsilon in eps_values:
                    selected_after = selected_logits_after_step(hidden, signed_direction, epsilon, norm, lm_head, top_ids_t)
                    probs_after = softmax_np(selected_after[None, :])[0]
                    entropy_after, varentropy_after = entropy_varentropy(probs_after)
                    semantic_after = semantic_entropy_proxy(probs_after, cluster_labels)
                    top1_after = int(top_ids[int(np.argmax(probs_after))])
                    target_prob_after = float(sum(probs_after[pos] for pos in target_positions)) if target_in_topm else 0.0
                    target_top1_after = int(top1_after in target_ids)
                    steering_rows.append(
                        {
                            "prompt_id": int(record["prompt_id"]),
                            "prompt": meta["prompt"],
                            "causal_prefix": meta["causal_prefix"],
                            "target": meta["target"],
                            "topic": meta["topic"],
                            "fact_id": meta["fact_id"],
                            "observed_condition": meta["observed_condition"],
                            "layer": int(args.layer),
                            "pca_dim": int(basis.shape[1]),
                            "direction": direction_name,
                            "sign": sign_name,
                            "epsilon": float(epsilon),
                            "rho_accessible": rho,
                            "v_access": float(w_acc @ w_acc),
                            "v_inaccess": float(w_orth @ w_orth),
                            "latent_direction_norm": float(np.linalg.norm(z_dir)),
                            "hidden_direction_norm": float(np.linalg.norm(hidden_direction)),
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
                            "semantic_entropy_proxy_before": semantic_before,
                            "semantic_entropy_proxy_after": semantic_after,
                            "delta_semantic_entropy_proxy": semantic_after - semantic_before,
                            "abs_delta_semantic_entropy_proxy": abs(semantic_after - semantic_before),
                            "selected_top1_before": int(top_ids[0]),
                            "selected_top1_after": top1_after,
                            "selected_top1_changed": int(top1_after != int(top_ids[0])),
                            "target_in_topm": target_in_topm,
                            "target_top1_before_topm": target_top1_before,
                            "target_top1_after_topm": target_top1_after,
                            "target_correct_changed": int(target_top1_before != target_top1_after),
                            "target_prob_topm_before": target_prob_before,
                            "target_prob_topm_after": target_prob_after,
                            "delta_target_prob_topm": target_prob_after - target_prob_before,
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
            target_in_topm_rate=("target_in_topm", "mean"),
            delta_target_prob_topm_mean=("delta_target_prob_topm", "mean"),
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
    counts = prompts.groupby(["topic", "observed_condition"]).size().unstack(fill_value=0).reset_index()
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
        "# Decoder Uncertainty Steering",
        "",
        "## Setup",
        "",
        f"- Model: `{args.model}`",
        f"- Local files only: `{args.local_files_only}`",
        f"- Layer: `{args.layer}`",
        f"- Output top-m lens: `{args.top_m}`",
        f"- Latent subspace: top PCA basis of forward hidden deltas, dim `{args.pca_dim}`",
        f"- Prompts: `{len(prompts)}` causal factual completions",
        f"- Torch dtype: `{args.torch_dtype}`",
        "",
        "This is the decoder-only version of the uncertainty steering test. It computes `w = F^{1/2}u` on next-token logit-lens probabilities, projects it onto `Im(F^{1/2}JB)`, and compares equal-latent-energy accessible steering against strict orthogonal, entropy-gradient-orthogonal, and random controls.",
        "",
        "The strict orthogonal least-squares target should be nearly zero because it asks the decoder latent map to realize a Fisher-whitened output vector outside its local image. The empirical equal-energy test is therefore the accessible direction versus the gradient-orthogonal and random controls.",
        "",
        "Semantic uncertainty is an embedding-quantile entropy proxy over the fixed top-m next-token set. Accuracy is exact-token top-1 stability within that monitored top-m set.",
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
        "prompt_features.csv",
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
    if args.out_dir is None:
        args.out_dir = ROOT / "results" / f"decoder_uncertainty_steering_{safe_name(args.model)}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    torch.set_grad_enabled(True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    model.eval()
    model.config.use_cache = False
    norm = get_final_norm(model)
    lm_head = model.get_output_embeddings()

    prompts_source = choose_prompts(args.prompt_csv, args.max_prompts, args.seed)
    prompts, records = build_prompt_records(tokenizer, model, norm, lm_head, prompts_source, args.layer, args.top_m)
    if prompts.empty:
        raise RuntimeError("No prompts could be evaluated.")
    basis = build_pca_subspace(records, args.pca_dim)
    steering, diagnostics = compute_steering(tokenizer, model, norm, lm_head, prompts, records, basis, args)
    summary = summarize(steering)
    contrasts = contrast_summary(summary)

    prompts.to_csv(args.out_dir / "prompt_features.csv", index=False)
    diagnostics.to_csv(args.out_dir / "projection_diagnostics.csv", index=False)
    steering.to_csv(args.out_dir / "steering_records.csv", index=False)
    summary.to_csv(args.out_dir / "steering_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "steering_contrasts.csv", index=False)
    write_report(args, prompts, diagnostics, summary, contrasts)
    print(args.out_dir)
    print((args.out_dir / "report.md").read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
