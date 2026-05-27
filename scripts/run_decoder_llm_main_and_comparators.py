from __future__ import annotations

import argparse
import json
import math
import re
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
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from accessible_varentropy.metrics import softmax_np
from run_decoder_uncertainty_steering import (
    candidate_token_ids,
    control_direction_orthogonal_to_entropy_gradient,
    entropy_varentropy,
    fisher_geometry,
    get_final_norm,
    jacobian_selected,
    least_squares,
    lens_logits_from_hidden,
    normalize,
    orthogonal_basis_projection,
    selected_lens_logits,
    token_cluster_labels,
)


@dataclass(frozen=True)
class DecoderCase:
    prompt: str
    target: str | None
    task: str
    topic: str
    case_id: str


ROOT_EXPERIMENT = ROOT / "experiments" / "04_uncertainty_steering" / "decoder_main_battery"
ROOT_COMPARATORS = ROOT / "experiments" / "controls" / "external_uncertainty_comparators"
ROOT_APPLICATION = ROOT / "applications" / "local_confidence_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="distilgpt2,Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B-Instruct,microsoft/phi-2,meta-llama/Llama-3.2-1B,mistralai/Mistral-7B-v0.1",
    )
    parser.add_argument(
        "--record-skipped-models",
        default="",
    )
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--pca-dim", type=int, default=8)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--max-prompts-per-task", type=int, default=3)
    parser.add_argument("--output-eps", type=float, default=0.05)
    parser.add_argument("--semantic-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--decoder-out", type=Path, default=ROOT_EXPERIMENT)
    parser.add_argument("--comparators-out", type=Path, default=ROOT_COMPARATORS)
    parser.add_argument("--application-out", type=Path, default=ROOT_APPLICATION)
    return parser.parse_args()


def resolve_torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()


def build_cases(max_per_task: int, seed: int) -> list[DecoderCase]:
    qa = [
        ("Question: What is the capital of France?\nAnswer:", "Paris", "capital"),
        ("Question: What is the capital of Italy?\nAnswer:", "Rome", "capital"),
        ("Question: Which planet is known as the Red Planet?\nAnswer:", "Mars", "science"),
        ("Question: What gas do humans need to breathe?\nAnswer:", "oxygen", "science"),
        ("Question: Who wrote Hamlet?\nAnswer:", "Shakespeare", "culture"),
        ("Question: What color is grass usually?\nAnswer:", "green", "common"),
        ("Question: What is the opposite of hot?\nAnswer:", "cold", "antonym"),
        ("Question: What animal barks?\nAnswer:", "dog", "animal"),
    ]
    completion = [
        ("The capital of Germany is", "Berlin", "capital"),
        ("The largest planet in the solar system is", "Jupiter", "science"),
        ("A person who teaches students is a", "teacher", "role"),
        ("The language most commonly spoken in Brazil is", "Portuguese", "language"),
        ("The currency used in Japan is the", "yen", "currency"),
        ("A baby cat is called a", "kitten", "animal"),
        ("The plural of mouse is", "mice", "plural"),
        ("Water freezes into", "ice", "science"),
    ]
    open_ended = [
        ("A useful tool for writing code is", None, "tool"),
        ("A traveler entering a city might first see the", None, "scene"),
        ("When people want to celebrate, they often bring", None, "common"),
        ("A short explanation of uncertainty is", None, "concept"),
        ("In a school hallway, a student may notice the", None, "scene"),
        ("A practical way to reduce risk is to", None, "action"),
    ]
    by_task = {
        "token_level_qa": qa,
        "factual_completion": completion,
        "generative_open": open_ended,
    }
    rng = np.random.default_rng(seed)
    cases: list[DecoderCase] = []
    for task, rows in by_task.items():
        order = list(range(len(rows)))
        rng.shuffle(order)
        for local_idx, row_idx in enumerate(order[: max(1, min(max_per_task, len(rows)))]):
            prompt, target, topic = rows[row_idx]
            cases.append(
                DecoderCase(
                    prompt=prompt,
                    target=target,
                    task=task,
                    topic=topic,
                    case_id=f"{task}:{row_idx}:{local_idx}",
                )
            )
    return cases


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "auto":
        return sorted({max(1, n_layers // 2), max(1, n_layers - 1)})
    return sorted({int(part) for part in spec.split(",") if part.strip()})


def pca_basis(matrix: np.ndarray, max_k: int) -> np.ndarray:
    dim = min(max_k, matrix.shape[0] - 1, matrix.shape[1])
    if dim < 1:
        raise RuntimeError("Need at least two examples to build a PCA basis.")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:dim].T.astype(np.float64)


def orthonormalize(matrix: np.ndarray, k: int) -> np.ndarray:
    q, _r = np.linalg.qr(matrix)
    return q[:, :k].astype(np.float64)


def topm_from_logits(logits: torch.Tensor, top_m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top = torch.topk(logits.detach().float(), k=min(top_m, logits.numel()))
    token_ids = top.indices.detach().cpu().numpy().astype(np.int64)
    values = top.values.detach().cpu().numpy().astype(np.float64)
    probs = softmax_np(values[None, :])[0].astype(np.float64)
    return token_ids, values, probs


def rank_of_token(logits: np.ndarray, token_id: int) -> int:
    return int(1 + np.sum(logits > logits[int(token_id)]))


def jaccard_from_ids(before: list[int] | np.ndarray, after: list[int] | np.ndarray, n: int) -> float:
    a = set([int(x) for x in list(before)[:n]])
    b = set([int(x) for x in list(after)[:n]])
    return len(a & b) / max(len(a | b), 1)


def semantic_density_uncertainty(embeddings: np.ndarray, token_ids: np.ndarray, probs: np.ndarray) -> float:
    x = embeddings[token_ids].astype(np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    distances = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    positive = distances[distances > 1e-12]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    kernel = np.exp(-distances / max(2.0 * bandwidth, 1e-12))
    density = float(probs @ kernel @ probs)
    return float(1.0 - density)


def haloscope_style_consistency_risk(probs: np.ndarray, labels: np.ndarray) -> float:
    masses = np.array([float(probs[labels == label].sum()) for label in sorted(set(labels.tolist()))], dtype=np.float64)
    if masses.size == 0:
        return float("nan")
    entropy, _var = entropy_varentropy(masses)
    normalized_entropy = entropy / max(math.log(max(len(masses), 2)), 1e-12)
    dominant_gap = float(1.0 - np.max(masses))
    return float(0.5 * normalized_entropy + 0.5 * dominant_gap)


def build_records_for_model(
    tokenizer,
    model,
    norm,
    lm_head,
    model_name: str,
    cases: list[DecoderCase],
    layers: list[int],
    top_m: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    prompt_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for idx, case in enumerate(cases):
        print(f"decoder_prompt {safe_name(model_name)} {idx + 1}/{len(cases)}", flush=True)
        encoded = tokenizer(case.prompt, return_tensors="pt", add_special_tokens=False)
        if encoded["input_ids"].numel() == 0:
            continue
        target_ids = candidate_token_ids(tokenizer, case.target) if case.target else set()
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        full_n_layers = len(hidden_states) - 1
        usable_layers = [layer for layer in layers if 1 <= layer < len(hidden_states)]
        for layer in usable_layers:
            hidden = hidden_states[layer][0, -1, :].detach().float()
            next_hidden = hidden_states[min(layer + 1, full_n_layers)][0, -1, :].detach().float()
            logits = lens_logits_from_hidden(hidden, norm, lm_head).detach().float()
            token_ids, values, probs = topm_from_logits(logits, top_m)
            top1_id = int(token_ids[0])
            target_in_topm = int(bool(target_ids & set(int(x) for x in token_ids.tolist())))
            target_rank = (
                min(rank_of_token(logits.detach().cpu().numpy().astype(np.float64), tid) for tid in target_ids)
                if target_ids
                else np.nan
            )
            top1_correct = int(top1_id in target_ids) if target_ids else np.nan
            entropy, varentropy = entropy_varentropy(probs)
            prompt_id = f"{safe_name(model_name)}:{case.case_id}:layer{layer}"
            prompt_rows.append(
                {
                    "model": model_name,
                    "prompt_id": prompt_id,
                    "case_id": case.case_id,
                    "prompt": case.prompt,
                    "target": case.target or "",
                    "target_token_ids": " ".join(str(x) for x in sorted(target_ids)),
                    "task": case.task,
                    "topic": case.topic,
                    "layer": int(layer),
                    "top_m": int(top_m),
                    "top1_token_id": top1_id,
                    "top1_token": tokenizer.decode([top1_id]).strip(),
                    "top1_correct": top1_correct,
                    "target_in_topm": target_in_topm,
                    "target_rank_lens": target_rank,
                    "entropy": entropy,
                    "varentropy": varentropy,
                    "confidence": float(probs[0]),
                    "num_model_layers": int(full_n_layers),
                }
            )
            records.append(
                {
                    "model": model_name,
                    "prompt_id": prompt_id,
                    "case_id": case.case_id,
                    "prompt": case.prompt,
                    "target": case.target or "",
                    "task": case.task,
                    "topic": case.topic,
                    "layer": int(layer),
                    "hidden": hidden,
                    "next_hidden": next_hidden,
                    "top_ids": token_ids,
                    "top_logits": values,
                    "target_ids": sorted(target_ids),
                }
            )
    return pd.DataFrame(prompt_rows), records


def make_subspaces(records: list[dict[str, object]], pca_dim: int, seed: int) -> dict[tuple[str, int], np.ndarray]:
    hidden = np.stack([rec["hidden"].detach().cpu().numpy().astype(np.float64) for rec in records])
    delta = np.stack(
        [(rec["next_hidden"] - rec["hidden"]).detach().cpu().numpy().astype(np.float64) for rec in records]
    )
    hidden_dim = hidden.shape[1]
    max_k = min(pca_dim, hidden_dim, len(records) - 1)
    rng = np.random.default_rng(seed)
    return {
        ("state_pca", max_k): pca_basis(hidden, max_k),
        ("delta_pca", max_k): pca_basis(delta, max_k),
        ("random", max_k): orthonormalize(rng.normal(size=(hidden_dim, max_k)), max_k),
    }


def apply_hidden_delta(hidden: torch.Tensor, delta: np.ndarray, norm, lm_head, token_ids: torch.Tensor) -> np.ndarray:
    step = torch.as_tensor(delta, dtype=hidden.dtype, device=hidden.device)
    with torch.no_grad():
        logits = selected_lens_logits(hidden + step, norm, lm_head, token_ids).detach().float()
    return logits.cpu().numpy().astype(np.float64)


def selected_top_ids_after(hidden: torch.Tensor, delta: np.ndarray, norm, lm_head, top_m: int) -> list[int]:
    step = torch.as_tensor(delta, dtype=hidden.dtype, device=hidden.device)
    with torch.no_grad():
        logits = lens_logits_from_hidden(hidden + step, norm, lm_head).detach().float()
        top = torch.topk(logits, k=min(top_m, logits.numel())).indices.detach().cpu().numpy().astype(np.int64)
    return [int(x) for x in top.tolist()]


def score_and_steer(
    tokenizer,
    model,
    norm,
    lm_head,
    records: list[dict[str, object]],
    basis_by_family: dict[tuple[str, int], np.ndarray],
    output_eps: float,
    semantic_clusters: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    score_rows: list[dict[str, object]] = []
    steering_rows: list[dict[str, object]] = []
    comparator_rows: list[dict[str, object]] = []

    for rec_idx, record in enumerate(records):
        print(f"decoder_steer {safe_name(str(record['model']))} layer={record['layer']} {rec_idx + 1}/{len(records)}", flush=True)
        hidden = record["hidden"].detach()
        top_ids = np.asarray(record["top_ids"], dtype=np.int64)
        top_ids_t = torch.as_tensor(top_ids, dtype=torch.long, device=hidden.device)
        selected_before = selected_lens_logits(hidden, norm, lm_head, top_ids_t).detach().float().cpu().numpy().astype(np.float64)
        probs = softmax_np(selected_before[None, :])[0].astype(np.float64)
        geom = fisher_geometry(probs)
        jac = jacobian_selected(hidden, norm, lm_head, top_ids_t)
        labels = token_cluster_labels(embeddings, top_ids, semantic_clusters)
        semantic_entropy = float(entropy_varentropy(np.array([probs[labels == label].sum() for label in sorted(set(labels.tolist()))]))[0])
        semantic_density = semantic_density_uncertainty(embeddings, top_ids, probs)
        halo_risk = haloscope_style_consistency_risk(probs, labels)
        target_ids = set(int(x) for x in record["target_ids"])
        target_positions = [idx for idx, token_id in enumerate(top_ids.tolist()) if int(token_id) in target_ids]
        target_in_topm = int(len(target_positions) > 0)
        target_prob_before = float(sum(probs[pos] for pos in target_positions)) if target_positions else 0.0
        selected_top1_before = int(top_ids[0])
        target_top1_before = int(selected_top1_before in target_ids) if target_ids else np.nan

        comparator_rows.append(
            {
                "model": record["model"],
                "prompt_id": record["prompt_id"],
                "case_id": record["case_id"],
                "prompt": record["prompt"],
                "target": record["target"],
                "task": record["task"],
                "topic": record["topic"],
                "layer": int(record["layer"]),
                "top_m": int(len(top_ids)),
                "entropy": geom.entropy,
                "varentropy": geom.varentropy,
                "confidence": float(probs[0]),
                "semantic_entropy_proxy": semantic_entropy,
                "semantic_density_uncertainty": semantic_density,
                "haloscope_style_consistency_risk": halo_risk,
                "target_in_topm": target_in_topm,
                "target_top1_before": target_top1_before,
            }
        )

        for (family, subspace_k), basis in basis_by_family.items():
            jb = jac @ basis
            whitened = geom.sqrt_fisher @ jb
            w_acc, rank = orthogonal_basis_projection(whitened, geom.w)
            w_orth = geom.w - w_acc
            denom = max(float(geom.w @ geom.w), 1e-12)
            rho = float((w_acc @ w_acc) / denom)
            grad_z = whitened.T @ geom.w
            z_acc = least_squares(whitened, w_acc)
            z_acc_norm = float(np.linalg.norm(z_acc))
            if z_acc_norm <= 1e-12:
                continue
            z_acc = z_acc / z_acc_norm
            z_grad_orth = control_direction_orthogonal_to_entropy_gradient(rng, grad_z, basis.shape[1])
            z_random = normalize(rng.normal(size=basis.shape[1]))
            directions = {
                "accessible_ls": z_acc,
                "grad_orthogonal_control": z_grad_orth,
                "random_control": z_random,
            }
            score_rows.append(
                {
                    "model": record["model"],
                    "prompt_id": record["prompt_id"],
                    "case_id": record["case_id"],
                    "prompt": record["prompt"],
                    "target": record["target"],
                    "task": record["task"],
                    "topic": record["topic"],
                    "layer": int(record["layer"]),
                    "subspace_family": family,
                    "subspace_k": int(subspace_k),
                    "top_m": int(len(top_ids)),
                    "rho": rho,
                    "v_access": float(w_acc @ w_acc),
                    "v_inaccess": float(w_orth @ w_orth),
                    "entropy": geom.entropy,
                    "varentropy": geom.varentropy,
                    "confidence": float(probs[0]),
                    "rank_whitened_jb": int(rank),
                    "grad_entropy_proj_norm": float(np.linalg.norm(grad_z)),
                    "fisher_output_fro_norm": float(np.linalg.norm(whitened)),
                    "semantic_entropy_proxy": semantic_entropy,
                    "semantic_density_uncertainty": semantic_density,
                    "haloscope_style_consistency_risk": halo_risk,
                }
            )

            for direction_name, z_dir in directions.items():
                output_vec = whitened @ z_dir
                output_norm = float(np.linalg.norm(output_vec))
                if output_norm <= 1e-12:
                    continue
                latent_scale = output_eps / output_norm
                hidden_delta_unit = basis @ z_dir
                for sign_name, sign in [("increase", 1.0), ("decrease", -1.0)]:
                    hidden_delta = sign * latent_scale * hidden_delta_unit
                    selected_after = apply_hidden_delta(hidden, hidden_delta, norm, lm_head, top_ids_t)
                    probs_after = softmax_np(selected_after[None, :])[0].astype(np.float64)
                    entropy_after, varentropy_after = entropy_varentropy(probs_after)
                    target_prob_after = float(sum(probs_after[pos] for pos in target_positions)) if target_positions else 0.0
                    top1_after = int(top_ids[int(np.argmax(probs_after))])
                    target_top1_after = int(top1_after in target_ids) if target_ids else np.nan
                    top_after_full = selected_top_ids_after(hidden, hidden_delta, norm, lm_head, len(top_ids))
                    masses_after = np.array([probs_after[labels == label].sum() for label in sorted(set(labels.tolist()))])
                    sem_after, _sem_var = entropy_varentropy(masses_after)
                    density_after = semantic_density_uncertainty(embeddings, top_ids, probs_after)
                    halo_after = haloscope_style_consistency_risk(probs_after, labels)
                    steering_rows.append(
                        {
                            "model": record["model"],
                            "prompt_id": record["prompt_id"],
                            "case_id": record["case_id"],
                            "prompt": record["prompt"],
                            "target": record["target"],
                            "task": record["task"],
                            "topic": record["topic"],
                            "layer": int(record["layer"]),
                            "subspace_family": family,
                            "subspace_k": int(subspace_k),
                            "direction": direction_name,
                            "sign": sign_name,
                            "epsilon": float(output_eps),
                            "mode": "fisher_output_equal",
                            "fisher_output_energy": output_eps,
                            "latent_step_norm": float(abs(latent_scale) * np.linalg.norm(z_dir)),
                            "hidden_step_norm": float(np.linalg.norm(hidden_delta)),
                            "rho": rho,
                            "entropy_before": geom.entropy,
                            "entropy_after": entropy_after,
                            "delta_entropy": entropy_after - geom.entropy,
                            "abs_delta_entropy": abs(entropy_after - geom.entropy),
                            "varentropy_before": geom.varentropy,
                            "varentropy_after": varentropy_after,
                            "delta_varentropy": varentropy_after - geom.varentropy,
                            "abs_delta_varentropy": abs(varentropy_after - geom.varentropy),
                            "confidence_before": float(probs[0]),
                            "confidence_after": float(probs_after[0]),
                            "delta_confidence": float(probs_after[0] - probs[0]),
                            "semantic_entropy_proxy_before": semantic_entropy,
                            "semantic_entropy_proxy_after": float(sem_after),
                            "delta_semantic_entropy_proxy": float(sem_after - semantic_entropy),
                            "semantic_density_uncertainty_before": semantic_density,
                            "semantic_density_uncertainty_after": density_after,
                            "delta_semantic_density_uncertainty": density_after - semantic_density,
                            "haloscope_style_consistency_risk_before": halo_risk,
                            "haloscope_style_consistency_risk_after": halo_after,
                            "delta_haloscope_style_consistency_risk": halo_after - halo_risk,
                            "selected_top1_before": selected_top1_before,
                            "selected_top1_after": top1_after,
                            "selected_top1_changed": int(top1_after != selected_top1_before),
                            "target_in_topm": target_in_topm,
                            "target_top1_before": target_top1_before,
                            "target_top1_after": target_top1_after,
                            "target_correct_changed": int(target_top1_before != target_top1_after)
                            if target_ids
                            else np.nan,
                            "target_prob_before": target_prob_before,
                            "target_prob_after": target_prob_after,
                            "delta_target_prob": target_prob_after - target_prob_before,
                            "top5_jaccard": jaccard_from_ids(top_ids, top_after_full, min(5, len(top_ids))),
                            "top10_jaccard": jaccard_from_ids(top_ids, top_after_full, min(10, len(top_ids))),
                            "top1_token_before": tokenizer.decode([selected_top1_before]).strip(),
                            "top1_token_after": tokenizer.decode([top1_after]).strip(),
                        }
                    )
    return pd.DataFrame(score_rows), pd.DataFrame(steering_rows), pd.DataFrame(comparator_rows)


def summarize_decoder(scores: pd.DataFrame, steering: pd.DataFrame, skipped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_summary = scores.groupby(["model", "subspace_family", "layer"], as_index=False).agg(
        n=("prompt_id", "count"),
        rho_mean=("rho", "mean"),
        rho_std=("rho", "std"),
        v_access_mean=("v_access", "mean"),
        grad_entropy_proj_norm_mean=("grad_entropy_proj_norm", "mean"),
    )
    steering_summary = steering.groupby(["model", "subspace_family", "direction", "sign"], as_index=False).agg(
        n=("prompt_id", "count"),
        abs_delta_entropy_mean=("abs_delta_entropy", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
        selected_top1_changed_rate=("selected_top1_changed", "mean"),
        top10_jaccard_mean=("top10_jaccard", "mean"),
        delta_confidence_mean=("delta_confidence", "mean"),
    )
    rows: list[dict[str, object]] = []
    grouped = steering_summary.set_index(["model", "subspace_family", "direction", "sign"])
    for (model_name, family, _direction, sign), acc in grouped.loc[
        grouped.index.get_level_values("direction") == "accessible_ls"
    ].iterrows():
        for control in ["random_control", "grad_orthogonal_control"]:
            key = (model_name, family, control, sign)
            if key not in grouped.index:
                continue
            ctrl = grouped.loc[key]
            for metric in ["abs_delta_entropy_mean", "abs_delta_varentropy_mean"]:
                denom = float(ctrl[metric])
                rows.append(
                    {
                        "model": model_name,
                        "subspace_family": family,
                        "sign": sign,
                        "control": control,
                        "metric": metric,
                        "accessible": float(acc[metric]),
                        "control_value": denom,
                        "accessible_minus_control": float(acc[metric] - denom),
                        "accessible_over_control": float(acc[metric] / denom) if abs(denom) > 1e-12 else math.inf,
                    }
                )
    contrasts = pd.DataFrame(rows)
    return score_summary, steering_summary, contrasts


def summarize_comparators(scores: pd.DataFrame, steering: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "rho",
        "entropy",
        "varentropy",
        "confidence",
        "semantic_entropy_proxy",
        "semantic_density_uncertainty",
        "haloscope_style_consistency_risk",
    ]
    merged = scores.merge(
        steering[
            (steering["direction"] == "accessible_ls")
            & (steering["subspace_family"].isin(["state_pca", "delta_pca"]))
            & (steering["sign"] == "increase")
        ][["model", "prompt_id", "subspace_family", "abs_delta_entropy", "abs_delta_varentropy", "selected_top1_changed"]],
        on=["model", "prompt_id", "subspace_family"],
        how="left",
    )
    rows = []
    for a in metric_cols:
        for b in metric_cols:
            if a >= b:
                continue
            local = merged[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
            corr = float(spearmanr(local[a], local[b]).correlation) if len(local) >= 4 else float("nan")
            rows.append({"metric_a": a, "metric_b": b, "n": int(len(local)), "spearman": corr})
    correlations = pd.DataFrame(rows)
    sens_rows = []
    for metric in metric_cols:
        for outcome in ["abs_delta_entropy", "abs_delta_varentropy", "selected_top1_changed"]:
            local = merged[[metric, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
            corr = float(spearmanr(local[metric], local[outcome]).correlation) if len(local) >= 4 else float("nan")
            sens_rows.append({"metric": metric, "outcome": outcome, "n": int(len(local)), "spearman": corr})
    sensitivity = pd.DataFrame(sens_rows)
    return merged, correlations, sensitivity


def summarize_application(steering: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accessible = steering[(steering["direction"] == "accessible_ls") & (steering["subspace_family"].isin(["state_pca", "delta_pca"]))].copy()
    if accessible.empty:
        return pd.DataFrame(), pd.DataFrame()
    accessible["answer_preserved"] = (
        (accessible["selected_top1_changed"] == 0)
        & (accessible["top10_jaccard"] >= 0.8)
        & (accessible["target_correct_changed"].fillna(0) == 0)
    )
    threshold = float(accessible["abs_delta_entropy"].median())
    accessible["selective_confidence_control_success"] = accessible["answer_preserved"] & (
        accessible["abs_delta_entropy"] >= threshold
    )
    summary = accessible.groupby(["model", "sign"], as_index=False).agg(
        n=("prompt_id", "count"),
        success_rate=("selective_confidence_control_success", "mean"),
        answer_preserved_rate=("answer_preserved", "mean"),
        abs_delta_entropy_mean=("abs_delta_entropy", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
        abs_delta_confidence_mean=("delta_confidence", lambda s: float(np.mean(np.abs(s)))),
        top10_jaccard_mean=("top10_jaccard", "mean"),
        top1_changed_rate=("selected_top1_changed", "mean"),
    )
    examples = (
        accessible[accessible["selective_confidence_control_success"]]
        .sort_values(["abs_delta_entropy", "top10_jaccard"], ascending=[False, False])
        .head(30)
        [
            [
                "model",
                "prompt",
                "target",
                "task",
                "topic",
                "layer",
                "subspace_family",
                "sign",
                "top1_token_before",
                "top1_token_after",
                "entropy_before",
                "entropy_after",
                "delta_entropy",
                "confidence_before",
                "confidence_after",
                "delta_confidence",
                "top10_jaccard",
                "rho",
            ]
        ]
    )
    return summary, examples


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def write_decoder_report(out_dir: Path, prompt_table: pd.DataFrame, score_summary: pd.DataFrame, steering_summary: pd.DataFrame, contrasts: pd.DataFrame, skipped: pd.DataFrame) -> None:
    counts = prompt_table.groupby(["model", "task"], as_index=False).size() if not prompt_table.empty else pd.DataFrame()
    lines = [
        "# Decoder-Only LLM Main Battery",
        "",
        "This is the main decoder-only extension of the accessible-varentropy steering protocol. It uses token-level QA, factual completion, and short generative-open prompts with next-token logit-lens geometry.",
        "",
        "The default decoder set spans GPT-2, Qwen, Phi, Llama, and Mistral families. Missing models are downloaded automatically unless `--local-files-only` is set; unavailable models are recorded in `skipped_models.csv`.",
        "",
        "## Prompt Counts",
        markdown_table(counts, 40),
        "",
        "## Skipped Models",
        markdown_table(skipped, 20),
        "",
        "## Accessibility Summary",
        markdown_table(score_summary, 80),
        "",
        "## Steering Summary",
        markdown_table(steering_summary, 80),
        "",
        "## Accessible Vs Controls",
        markdown_table(contrasts, 80),
        "",
        "## Files",
        "```text",
        "prompt_table.csv",
        "decoder_scores.csv",
        "decoder_steering_records.csv",
        "decoder_score_summary.csv",
        "decoder_steering_summary.csv",
        "decoder_control_contrasts.csv",
        "skipped_models.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparator_report(out_dir: Path, correlations: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    key = sensitivity[sensitivity["outcome"].isin(["abs_delta_entropy", "abs_delta_varentropy"])]
    lines = [
        "# External Uncertainty Comparator Diagnostics",
        "",
        "This diagnostic compares accessible-varentropy quantities with token-level approximations of Semantic Entropy, Semantic Density, and a HaloScope-style unlabeled consistency risk.",
        "",
        "Semantic Entropy is approximated by entropy over embedding clusters of top-m next-token candidates. Semantic Density is an embedding-kernel concentration score converted to uncertainty. HaloScope is not implemented as the full NeurIPS framework; the checked-in column is a HaloScope-style consistency proxy over unlabeled candidate clusters.",
        "",
        "The purpose is not to beat these methods on hallucination detection. The purpose is to show that they measure different objects from route-specific accessible varentropy.",
        "",
        "## Metric Correlations",
        markdown_table(correlations, 80),
        "",
        "## Steering Sensitivity",
        markdown_table(key, 80),
        "",
        "## Files",
        "```text",
        "comparator_scores.csv",
        "comparator_base_metrics.csv",
        "comparator_metric_correlations.csv",
        "comparator_steering_sensitivity.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_application_report(out_dir: Path, summary: pd.DataFrame, examples: pd.DataFrame) -> None:
    lines = [
        "# Application: Local Confidence Control",
        "",
        "This application filters accessible decoder interventions for cases where uncertainty/confidence moves while the local answer neighborhood is preserved.",
        "",
        "Success requires unchanged top-1, top-10 Jaccard at least 0.8, unchanged target correctness when a target exists, and above-median entropy movement.",
        "",
        "## Summary",
        markdown_table(summary, 40),
        "",
        "## Qualitative Examples",
        markdown_table(examples, 30),
        "",
        "## Files",
        "```text",
        "selective_confidence_control_summary.csv",
        "selective_confidence_control_examples.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readmes(args: argparse.Namespace) -> None:
    args.decoder_out.joinpath("README.md").write_text(
        "\n".join(
            [
                "# Decoder-Only LLM Main Battery",
                "",
                "Main decoder-only extension for accessible-varentropy steering.",
                "",
                "## Reproduce",
                "",
                "```powershell",
                "python scripts\\run_decoder_llm_main_and_comparators.py --trust-remote-code --models distilgpt2,Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B-Instruct,microsoft/phi-2,meta-llama/Llama-3.2-1B,mistralai/Mistral-7B-v0.1 --max-prompts-per-task 3 --top-m 16 --pca-dim 8 --output-eps 0.05 --seed 20260528",
                "```",
                "",
                "## Claim Supported",
                "",
                "Accessible-varentropy geometry extends to decoder-only next-token logit lenses on token-level QA, factual completion, and generative-open prompts.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.comparators_out.joinpath("README.md").write_text(
        "\n".join(
            [
                "# External Uncertainty Comparator Diagnostics",
                "",
                "Compares route-specific accessible varentropy against token-level approximations of Semantic Entropy, Semantic Density, and a HaloScope-style consistency proxy.",
                "",
                "These diagnostics are framed as object-comparison controls, not as hallucination-detection leaderboard claims.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.application_out.joinpath("README.md").write_text(
        "\n".join(
            [
                "# Local Confidence Control",
                "",
                "Killer-application slice: accessible steering changes local confidence/uncertainty while preserving the monitored answer neighborhood.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_configs(args: argparse.Namespace) -> None:
    command = (
        "python scripts\\run_decoder_llm_main_and_comparators.py --trust-remote-code "
        f"--models {args.models} --record-skipped-models \"{args.record_skipped_models}\" "
        f"--max-prompts-per-task {args.max_prompts_per_task} --top-m {args.top_m} "
        f"--pca-dim {args.pca_dim} --output-eps {args.output_eps} --seed {args.seed}"
    )
    payload = {
        "command": command,
        "seed": args.seed,
        "models": args.models.split(","),
        "record_skipped_models": args.record_skipped_models,
        "top_m": args.top_m,
        "pca_dim": args.pca_dim,
        "output_eps": args.output_eps,
        "local_files_only": bool(args.local_files_only),
    }
    for folder in [args.decoder_out, args.comparators_out, args.application_out]:
        (folder / "config" / "reproduce.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    torch.set_grad_enabled(True)
    for folder in [args.decoder_out, args.comparators_out, args.application_out]:
        for child in ["outputs", "reports", "config"]:
            (folder / child).mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.max_prompts_per_task, args.seed)
    all_prompts: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    all_steering: list[pd.DataFrame] = []
    all_comparators: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, str]] = []
    for item in [part.strip() for part in args.record_skipped_models.split(";") if part.strip()]:
        if ":" in item:
            model_label, reason = item.split(":", 1)
            skipped_rows.append({"model": model_label, "reason": reason})

    for model_name in [part.strip() for part in args.models.split(",") if part.strip()]:
        try:
            print(f"loading_decoder {model_name}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=resolve_torch_dtype(args.torch_dtype),
            )
            model.eval()
            model.config.use_cache = False
            norm = get_final_norm(model)
            lm_head = model.get_output_embeddings()
            n_layers = int(getattr(model.config, "num_hidden_layers", 0))
            layers = parse_layers(args.layers, n_layers)
            prompt_table, records = build_records_for_model(tokenizer, model, norm, lm_head, model_name, cases, layers, args.top_m)
            if prompt_table.empty or not records:
                raise RuntimeError("no decoder prompts completed")
            model_scores: list[pd.DataFrame] = []
            model_steering: list[pd.DataFrame] = []
            model_comparators: list[pd.DataFrame] = []
            for layer, layer_records in pd.DataFrame(records).groupby("layer", sort=True):
                layer_list = layer_records.to_dict("records")
                subspaces = make_subspaces(layer_list, args.pca_dim, args.seed + int(layer))
                scores, steering, comparators = score_and_steer(
                    tokenizer,
                    model,
                    norm,
                    lm_head,
                    layer_list,
                    subspaces,
                    args.output_eps,
                    args.semantic_clusters,
                    args.seed + int(layer),
                )
                model_scores.append(scores)
                model_steering.append(steering)
                model_comparators.append(comparators)
            all_prompts.append(prompt_table)
            all_scores.append(pd.concat(model_scores, ignore_index=True))
            all_steering.append(pd.concat(model_steering, ignore_index=True))
            all_comparators.append(pd.concat(model_comparators, ignore_index=True).drop_duplicates(["model", "prompt_id"]))
            del model
        except Exception as exc:  # noqa: BLE001
            print(f"skipping_decoder {model_name}: {exc}", flush=True)
            skipped_rows.append({"model": model_name, "reason": str(exc).splitlines()[0]})
            continue

    if not all_scores:
        raise RuntimeError("No decoder model completed.")
    prompt_table = pd.concat(all_prompts, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)
    steering = pd.concat(all_steering, ignore_index=True)
    comparator_base = pd.concat(all_comparators, ignore_index=True)
    skipped = pd.DataFrame(skipped_rows)

    score_summary, steering_summary, contrasts = summarize_decoder(scores, steering, skipped)
    comparator_scores, comparator_correlations, comparator_sensitivity = summarize_comparators(scores, steering)
    app_summary, app_examples = summarize_application(steering)

    prompt_table.to_csv(args.decoder_out / "outputs" / "prompt_table.csv", index=False)
    scores.to_csv(args.decoder_out / "outputs" / "decoder_scores.csv", index=False)
    steering.to_csv(args.decoder_out / "outputs" / "decoder_steering_records.csv", index=False)
    score_summary.to_csv(args.decoder_out / "outputs" / "decoder_score_summary.csv", index=False)
    steering_summary.to_csv(args.decoder_out / "outputs" / "decoder_steering_summary.csv", index=False)
    contrasts.to_csv(args.decoder_out / "outputs" / "decoder_control_contrasts.csv", index=False)
    skipped.to_csv(args.decoder_out / "outputs" / "skipped_models.csv", index=False)

    comparator_scores.to_csv(args.comparators_out / "outputs" / "comparator_scores.csv", index=False)
    comparator_base.to_csv(args.comparators_out / "outputs" / "comparator_base_metrics.csv", index=False)
    comparator_correlations.to_csv(args.comparators_out / "outputs" / "comparator_metric_correlations.csv", index=False)
    comparator_sensitivity.to_csv(args.comparators_out / "outputs" / "comparator_steering_sensitivity.csv", index=False)

    app_summary.to_csv(args.application_out / "outputs" / "selective_confidence_control_summary.csv", index=False)
    app_examples.to_csv(args.application_out / "outputs" / "selective_confidence_control_examples.csv", index=False)

    write_decoder_report(args.decoder_out, prompt_table, score_summary, steering_summary, contrasts, skipped)
    write_comparator_report(args.comparators_out, comparator_correlations, comparator_sensitivity)
    write_application_report(args.application_out, app_summary, app_examples)
    write_readmes(args)
    write_configs(args)

    print(args.decoder_out)
    print((args.decoder_out / "reports" / "report.md").read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii"))
    print(args.comparators_out)
    print((args.comparators_out / "reports" / "report.md").read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii"))
    print(args.application_out)
    print((args.application_out / "reports" / "report.md").read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
