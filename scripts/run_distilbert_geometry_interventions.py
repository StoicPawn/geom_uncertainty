from __future__ import annotations

import argparse
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForMaskedLM, BertTokenizer

from accessible_varentropy.bert_mlm import encode_prompt, hidden_states_and_logits, token_id_for_word
from accessible_varentropy.metrics import softmax_np
from accessible_varentropy.mlm_heads import mlm_head_jacobian, mlm_logits_from_hidden
from run_factual_error_deep_benchmark import FactualCase, factual_error_cases


@dataclass(frozen=True)
class FisherParts:
    probs: np.ndarray
    fisher: np.ndarray
    sqrt_fisher: np.ndarray
    u: np.ndarray
    fisher_u: np.ndarray
    entropy: float
    varentropy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "distilbert_geometry_interventions")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=20260522)
    parser.add_argument("--random-candidates", type=int, default=48)
    parser.add_argument("--saturation-reps", type=int, default=3)
    parser.add_argument("--splits", type=int, default=80)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--logreg-c", type=float, default=0.1)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--intervention-eps", default="0.25,0.5,1.0")
    return parser.parse_args()


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v, dtype=np.float64)
    return (v / norm).astype(np.float64)


def orthonormalize(matrix: np.ndarray, k: int) -> np.ndarray:
    q, _r = np.linalg.qr(matrix)
    return q[:, :k].astype(np.float64)


def topk_distribution(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    values = logits[idx]
    exp = np.exp(values - values.max())
    return idx.astype(np.int64), (exp / exp.sum()).astype(np.float64)


def entropy_varentropy(probs: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    log_p = np.log(p)
    entropy = float(-np.sum(p * log_p))
    surprisals = -log_p
    varentropy = float(np.sum(p * (surprisals - entropy) ** 2))
    return entropy, varentropy


def fisher_parts(probs: np.ndarray, eps: float = 1e-12) -> FisherParts:
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
    return FisherParts(p, fisher, sqrt_fisher, u, fisher_u, entropy, varentropy)


def projection_ratio(vector: np.ndarray, matrix: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(vector @ vector)
    if denom <= eps or matrix.size == 0:
        return 0.0
    left, singular_values, _right = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return 0.0
    cutoff = 1e-10 * max(matrix.shape) * singular_values[0]
    active = singular_values > cutoff
    if not np.any(active):
        return 0.0
    basis = left[:, active]
    accessible = float(np.sum((basis.T @ vector) ** 2))
    return float(np.clip(accessible / denom, 0.0, 1.0))


def fisher_scores_for_basis(
    parts: FisherParts,
    jacobian: np.ndarray,
    basis: np.ndarray,
    *,
    u_override: np.ndarray | None = None,
) -> dict[str, float]:
    j_basis = jacobian @ basis
    u = parts.u if u_override is None else np.asarray(u_override, dtype=np.float64)
    fisher_u = parts.fisher @ u
    varentropy = float(u @ fisher_u)
    a = parts.sqrt_fisher @ u
    whitened = parts.sqrt_fisher @ j_basis
    rho = projection_ratio(a, whitened) if varentropy > 1e-12 else 0.0
    accessible = float(rho * varentropy)
    grad = j_basis.T @ fisher_u
    trace_fim = float(np.trace(j_basis.T @ parts.fisher @ j_basis))
    return {
        "rho": rho,
        "accessible_varentropy": accessible,
        "varentropy": varentropy,
        "grad_norm_sq": float(grad @ grad),
        "trace_fim": trace_fim,
    }


def euclidean_rho(parts: FisherParts, jacobian: np.ndarray, basis: np.ndarray) -> float:
    return projection_ratio(parts.u, jacobian @ basis)


def layer_logits(model, hidden: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = mlm_logits_from_hidden(model, hidden.view(1, 1, -1))[0, 0, :]
    return logits.detach().cpu().numpy().astype(np.float64)


def topk_probs_after_step(
    model,
    hidden: torch.Tensor,
    token_ids: np.ndarray,
    direction: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    direction_t = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
    with torch.no_grad():
        stepped = hidden + float(eps) * direction_t
        logits = mlm_logits_from_hidden(model, stepped.view(1, 1, -1))[0, 0, :]
    full = logits.detach().cpu().numpy().astype(np.float64)
    selected = full[token_ids]
    probs = softmax_np(selected[None, :])[0]
    return full, probs


def sign_for_entropy_increase(parts: FisherParts, jacobian: np.ndarray, direction: np.ndarray) -> float:
    directional_grad = float((jacobian @ direction) @ parts.fisher_u)
    if directional_grad >= 0.0:
        return 1.0
    return -1.0


def build_prompt_table(tokenizer, model, cases: list[FactualCase], top_k: int) -> tuple[pd.DataFrame, list[dict]]:
    prompt_rows: list[dict] = []
    records: list[dict] = []
    for prompt_id, case in enumerate(cases):
        target_id = token_id_for_word(tokenizer, case.target)
        if target_id is None:
            continue
        encoded, mask_index = encode_prompt(tokenizer, case.prompt.replace("[MASK]", tokenizer.mask_token))
        hidden_states, final_logits = hidden_states_and_logits(model, encoded)
        final_mask_logits = final_logits[0, mask_index, :].detach().cpu().numpy().astype(np.float64)
        top_ids, top_probs = topk_distribution(final_mask_logits, top_k)
        sorted_all = np.argsort(final_mask_logits)[::-1]
        target_rank = int(np.where(sorted_all == target_id)[0][0] + 1)
        top1_id = int(top_ids[0])
        observed = "correct" if top1_id == int(target_id) else "error"
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": case.prompt,
                "target": case.target,
                "target_id": int(target_id),
                "topic": case.topic,
                "fact_id": case.fact_id,
                "template_id": case.template_id,
                "observed_condition": observed,
                "target_rank_final": target_rank,
                "final_top1_token": tokenizer.convert_ids_to_tokens([top1_id])[0],
                "final_top1_prob_topk": float(top_probs[0]),
                "final_top8_tokens": " ".join(tokenizer.convert_ids_to_tokens([int(v) for v in top_ids[:8]])),
            }
        )
        records.append(
            {
                "prompt_id": prompt_id,
                "mask_index": mask_index,
                "hidden_states": hidden_states,
            }
        )
    return pd.DataFrame(prompt_rows), records


def scalar_full_scores(parts: FisherParts, jacobian: np.ndarray) -> dict[str, float]:
    grad = jacobian.T @ parts.fisher_u
    return {
        "entropy": parts.entropy,
        "varentropy": parts.varentropy,
        "trace_fim": float(np.trace(jacobian.T @ parts.fisher @ jacobian)),
        "grad_norm_sq": float(grad @ grad),
    }


def conditionally_center_random_u(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    u = rng.normal(size=len(probs))
    return u - float(probs @ u)


def compute_records(
    tokenizer,
    model,
    prompts: pd.DataFrame,
    records: list[dict],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eps_values = [float(part.strip()) for part in args.intervention_eps.split(",") if part.strip()]
    rng = np.random.default_rng(args.random_seed)
    num_layers = len(records[0]["hidden_states"]) - 1
    hidden_dim = int(records[0]["hidden_states"][0].shape[-1])
    random_dirs = {
        layer: np.stack([normalize(rng.normal(size=hidden_dim)) for _ in range(args.random_candidates)], axis=1)
        for layer in range(num_layers + 1)
    }
    saturation_dims = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    saturation_dims = [k for k in saturation_dims if k < hidden_dim] + [hidden_dim]
    saturation_bases: dict[tuple[int, int, int], np.ndarray] = {}
    for layer in range(num_layers + 1):
        for k in saturation_dims:
            reps = 1 if k == hidden_dim else args.saturation_reps
            for rep in range(reps):
                if k == hidden_dim:
                    basis = np.eye(hidden_dim, dtype=np.float64)
                else:
                    basis = orthonormalize(rng.normal(size=(hidden_dim, k)), k)
                saturation_bases[(layer, k, rep)] = basis

    metric_rows: list[dict] = []
    feature_rows: list[dict] = []
    saturation_rows: list[dict] = []
    intervention_rows: list[dict] = []
    token_shift_rows: list[dict] = []

    prompt_lookup = prompts.set_index("prompt_id")
    for idx, record in enumerate(records):
        if idx % 20 == 0:
            print(f"processing_prompt {idx + 1}/{len(records)}", flush=True)
        meta = prompt_lookup.loc[int(record["prompt_id"])]
        hidden_states = record["hidden_states"]
        mask_index = int(record["mask_index"])
        feature: dict[str, object] = {
            "prompt_id": int(record["prompt_id"]),
            "observed_condition": meta["observed_condition"],
            "topic": meta["topic"],
            "fact_id": meta["fact_id"],
            "template_id": meta["template_id"],
        }
        best_token_shift_candidates: list[dict] = []
        for layer in range(num_layers + 1):
            hidden = hidden_states[layer][0, mask_index, :].detach()
            hidden_vec = hidden.cpu().numpy().astype(np.float64)
            logits = layer_logits(model, hidden)
            token_ids, probs = topk_distribution(logits, args.top_k)
            parts = fisher_parts(probs)
            jac = mlm_head_jacobian(model, hidden, token_ids)
            scalar = scalar_full_scores(parts, jac)
            feature[f"entropy_topk_l{layer}"] = scalar["entropy"]
            feature[f"varentropy_topk_l{layer}"] = scalar["varentropy"]
            feature[f"trace_fim_l{layer}"] = scalar["trace_fim"]
            feature[f"grad_norm_sq_l{layer}"] = scalar["grad_norm_sq"]
            feature[f"top1_prob_topk_layer_l{layer}"] = float(probs[0])

            bases: dict[str, np.ndarray] = {"state_1d": normalize(hidden_vec)[:, None]}
            if layer > 0:
                prev_vec = hidden_states[layer - 1][0, mask_index, :].detach().cpu().numpy().astype(np.float64)
                delta_vec = hidden_vec - prev_vec
                bases["delta_1d"] = normalize(delta_vec)[:, None]
            else:
                delta_vec = np.zeros_like(hidden_vec)

            random_scores: list[dict[str, float]] = []
            for cand_idx in range(args.random_candidates):
                direction = random_dirs[layer][:, cand_idx : cand_idx + 1]
                score = fisher_scores_for_basis(parts, jac, direction)
                random_scores.append(
                    {
                        "candidate": cand_idx,
                        "rho": score["rho"],
                        "trace_fim": score["trace_fim"],
                    }
                )

            for name, basis in bases.items():
                fisher_score = fisher_scores_for_basis(parts, jac, basis)
                euclid = euclidean_rho(parts, jac, basis)
                shuffled_u = rng.permutation(parts.u)
                shuffled_score = fisher_scores_for_basis(parts, jac, basis, u_override=shuffled_u)
                random_u = conditionally_center_random_u(rng, parts.probs)
                random_u_score = fisher_scores_for_basis(parts, jac, basis, u_override=random_u)
                row = {
                    "prompt_id": int(record["prompt_id"]),
                    "layer": layer,
                    "subspace": name,
                    "observed_condition": meta["observed_condition"],
                    "topic": meta["topic"],
                    "fact_id": meta["fact_id"],
                    "template_id": meta["template_id"],
                    "entropy_topk": parts.entropy,
                    "varentropy_topk": parts.varentropy,
                    "top1_prob_topk_layer": float(probs[0]),
                    "rho_fisher": fisher_score["rho"],
                    "accessible_varentropy": fisher_score["accessible_varentropy"],
                    "trace_fim": fisher_score["trace_fim"],
                    "grad_norm_sq": fisher_score["grad_norm_sq"],
                    "rho_euclidean": euclid,
                    "rho_shuffled_u": shuffled_score["rho"],
                    "rho_random_u": random_u_score["rho"],
                }
                if name == "delta_1d":
                    random_frame = pd.DataFrame(random_scores)
                    random_frame["trace_gap"] = np.abs(random_frame["trace_fim"] - fisher_score["trace_fim"])
                    matched = random_frame.sort_values(["trace_gap", "candidate"]).iloc[0]
                    high_trace = random_frame.sort_values(["trace_fim", "candidate"], ascending=[False, True]).iloc[0]
                    row["rho_matched_random_trace"] = float(matched["rho"])
                    row["matched_random_trace_gap"] = float(matched["trace_gap"])
                    row["matched_random_trace_fim"] = float(matched["trace_fim"])
                    row["rho_high_trace_random"] = float(high_trace["rho"])
                    row["high_trace_random_trace_fim"] = float(high_trace["trace_fim"])
                metric_rows.append(row)
                prefix = f"{name}_l{layer}"
                feature[f"rho_fisher_{prefix}"] = fisher_score["rho"]
                feature[f"rho_euclidean_{prefix}"] = euclid
                feature[f"rho_shuffled_u_{prefix}"] = shuffled_score["rho"]
                feature[f"rho_random_u_{prefix}"] = random_u_score["rho"]
                if name == "delta_1d":
                    feature[f"rho_matched_random_trace_{prefix}"] = row["rho_matched_random_trace"]
                    feature[f"rho_high_trace_random_{prefix}"] = row["rho_high_trace_random"]

            for k in saturation_dims:
                reps = 1 if k == hidden_dim else args.saturation_reps
                for rep in range(reps):
                    sat_basis = saturation_bases[(layer, k, rep)]
                    sat_score = fisher_scores_for_basis(parts, jac, sat_basis)
                    saturation_rows.append(
                        {
                            "prompt_id": int(record["prompt_id"]),
                            "layer": layer,
                            "k": k,
                            "rep": rep,
                            "observed_condition": meta["observed_condition"],
                            "topic": meta["topic"],
                            "rho": sat_score["rho"],
                            "trace_fim": sat_score["trace_fim"],
                            "entropy_topk": parts.entropy,
                            "varentropy_topk": parts.varentropy,
                        }
                    )

            if layer > 0:
                delta_dir = normalize(delta_vec)
                random_frame = pd.DataFrame(random_scores)
                delta_score = fisher_scores_for_basis(parts, jac, delta_dir[:, None])
                random_frame["trace_gap"] = np.abs(random_frame["trace_fim"] - delta_score["trace_fim"])
                matched_idx = int(random_frame.sort_values(["trace_gap", "candidate"]).iloc[0]["candidate"])
                high_idx = int(random_frame.sort_values(["trace_fim", "candidate"], ascending=[False, True]).iloc[0]["candidate"])
                directions = {
                    "delta_1d": delta_dir,
                    "matched_random_trace": random_dirs[layer][:, matched_idx],
                    "high_trace_random": random_dirs[layer][:, high_idx],
                }
                original_entropy = parts.entropy
                original_varentropy = parts.varentropy
                original_top1_id = int(token_ids[0])
                original_target_prob = float(probs[np.where(token_ids == int(meta["target_id"]))[0][0]]) if int(meta["target_id"]) in set(token_ids.tolist()) else 0.0
                for direction_name, direction in directions.items():
                    sign = sign_for_entropy_increase(parts, jac, direction)
                    signed_direction = sign * direction
                    dir_score = fisher_scores_for_basis(parts, jac, signed_direction[:, None])
                    for eps in eps_values:
                        full_logits_after, probs_after = topk_probs_after_step(
                            model,
                            hidden,
                            token_ids,
                            signed_direction,
                            eps,
                        )
                        entropy_after, varentropy_after = entropy_varentropy(probs_after)
                        top1_after_id = int(token_ids[int(np.argmax(probs_after))])
                        target_positions = np.where(token_ids == int(meta["target_id"]))[0]
                        target_prob_after = float(probs_after[int(target_positions[0])]) if len(target_positions) else 0.0
                        intervention_rows.append(
                            {
                                "prompt_id": int(record["prompt_id"]),
                                "layer": layer,
                                "direction": direction_name,
                                "epsilon": eps,
                                "observed_condition": meta["observed_condition"],
                                "topic": meta["topic"],
                                "fact_id": meta["fact_id"],
                                "rho_direction": dir_score["rho"],
                                "trace_fim_direction": dir_score["trace_fim"],
                                "entropy_before": original_entropy,
                                "entropy_after": entropy_after,
                                "delta_entropy": entropy_after - original_entropy,
                                "varentropy_before": original_varentropy,
                                "varentropy_after": varentropy_after,
                                "delta_varentropy": varentropy_after - original_varentropy,
                                "abs_delta_varentropy": abs(varentropy_after - original_varentropy),
                                "top1_changed_within_topk": int(top1_after_id != original_top1_id),
                                "target_prob_before_topk": original_target_prob,
                                "target_prob_after_topk": target_prob_after,
                                "delta_target_prob_topk": target_prob_after - original_target_prob,
                            }
                        )
                        if direction_name == "delta_1d" and eps == max(eps_values):
                            full_before = logits
                            probs_full_before = softmax_np(full_before[None, :])[0]
                            probs_full_after = softmax_np(full_logits_after[None, :])[0]
                            delta_probs = probs_full_after - probs_full_before
                            up = np.argsort(delta_probs)[-8:][::-1]
                            down = np.argsort(delta_probs)[:8]
                            best_token_shift_candidates.append(
                                {
                                    "layer": layer,
                                    "rho": dir_score["rho"],
                                    "epsilon": eps,
                                    "up_ids": up,
                                    "down_ids": down,
                                    "delta_probs": delta_probs,
                                }
                            )

        if best_token_shift_candidates:
            ordered = sorted(best_token_shift_candidates, key=lambda row: row["rho"])
            selected = [ordered[0], ordered[-1]]
            for kind, candidate in zip(["low_rho_layer", "high_rho_layer"], selected):
                up_tokens = tokenizer.convert_ids_to_tokens([int(v) for v in candidate["up_ids"]])
                down_tokens = tokenizer.convert_ids_to_tokens([int(v) for v in candidate["down_ids"]])
                token_shift_rows.append(
                    {
                        "prompt_id": int(record["prompt_id"]),
                        "prompt": meta["prompt"],
                        "observed_condition": meta["observed_condition"],
                        "topic": meta["topic"],
                        "target": meta["target"],
                        "final_top1_token": meta["final_top1_token"],
                        "kind": kind,
                        "layer": int(candidate["layer"]),
                        "rho_delta": float(candidate["rho"]),
                        "epsilon": float(candidate["epsilon"]),
                        "top_up_tokens": " ".join(up_tokens),
                        "top_up_delta_probs": " ".join(f"{float(candidate['delta_probs'][int(i)]):.6g}" for i in candidate["up_ids"]),
                        "top_down_tokens": " ".join(down_tokens),
                        "top_down_delta_probs": " ".join(f"{float(candidate['delta_probs'][int(i)]):.6g}" for i in candidate["down_ids"]),
                    }
                )
        feature_rows.append(feature)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(feature_rows),
        pd.DataFrame(saturation_rows),
        pd.DataFrame(intervention_rows),
        pd.DataFrame(token_shift_rows),
    )


def classifier(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=c),
    )


def build_feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    scalar_cols = [
        c
        for c in features.columns
        if c.startswith(("entropy_topk_l", "varentropy_topk_l", "trace_fim_l", "grad_norm_sq_l", "top1_prob_topk_layer_l"))
    ]
    sets: dict[str, list[str]] = {"scalar_layer_profile": scalar_cols}
    ablation_prefixes = {
        "rho_fisher_delta": "rho_fisher_delta_1d_l",
        "rho_euclidean_delta": "rho_euclidean_delta_1d_l",
        "rho_shuffled_u_delta": "rho_shuffled_u_delta_1d_l",
        "rho_random_u_delta": "rho_random_u_delta_1d_l",
        "rho_matched_random_trace_delta": "rho_matched_random_trace_delta_1d_l",
        "rho_high_trace_random_delta": "rho_high_trace_random_delta_1d_l",
        "rho_fisher_state": "rho_fisher_state_1d_l",
    }
    for name, prefix in ablation_prefixes.items():
        sets[name] = [c for c in features.columns if c.startswith(prefix)]
    sets["scalar_plus_rho_fisher_delta"] = sets["scalar_layer_profile"] + sets["rho_fisher_delta"]
    sets["scalar_plus_rho_euclidean_delta"] = sets["scalar_layer_profile"] + sets["rho_euclidean_delta"]
    sets["scalar_plus_matched_random_delta"] = sets["scalar_layer_profile"] + sets["rho_matched_random_trace_delta"]
    return {name: cols for name, cols in sets.items() if cols}


def evaluate_feature_sets(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fsets = build_feature_sets(features)
    y = features["observed_condition"].eq("error").astype(int).to_numpy()
    groups = features["fact_id"].astype(str).to_numpy()
    rows: list[dict] = []
    split_rows: list[dict] = []
    gss = GroupShuffleSplit(n_splits=args.splits, test_size=args.test_size, random_state=args.random_seed)
    for feature_set, cols in fsets.items():
        cols = [col for col in cols if not features[col].isna().all()]
        frame = features[cols].fillna(0.0)
        aucs: list[float] = []
        for split_idx, (train_idx, test_idx) in enumerate(gss.split(frame, y, groups)):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            clf = classifier(args.logreg_c)
            clf.fit(frame.iloc[train_idx].values, y[train_idx])
            pred = clf.predict_proba(frame.iloc[test_idx].values)[:, 1]
            auc = float(roc_auc_score(y[test_idx], pred))
            aucs.append(auc)
            split_rows.append({"split_kind": "fact_group", "split": split_idx, "feature_set": feature_set, "auc": auc})
        if aucs:
            rows.append(
                {
                    "split_kind": "fact_group",
                    "feature_set": feature_set,
                    "n_splits": len(aucs),
                    "auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs)),
                    "auc_min": float(np.min(aucs)),
                    "auc_max": float(np.max(aucs)),
                }
            )

    leave_rows: list[dict] = []
    logo = LeaveOneGroupOut()
    topic_groups = features["topic"].astype(str).to_numpy()
    if len(np.unique(topic_groups)) >= 2:
        for feature_set, cols in fsets.items():
            cols = [col for col in cols if not features[col].isna().all()]
            frame = features[cols].fillna(0.0)
            aucs = []
            for train_idx, test_idx in logo.split(frame, y, topic_groups):
                topic = str(topic_groups[test_idx][0])
                if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                    continue
                clf = classifier(args.logreg_c)
                clf.fit(frame.iloc[train_idx].values, y[train_idx])
                pred = clf.predict_proba(frame.iloc[test_idx].values)[:, 1]
                auc = float(roc_auc_score(y[test_idx], pred))
                aucs.append(auc)
                leave_rows.append(
                    {
                        "held_out_topic": topic,
                        "feature_set": feature_set,
                        "n_test": int(len(test_idx)),
                        "test_errors": int(y[test_idx].sum()),
                        "auc": auc,
                    }
                )
            if aucs:
                rows.append(
                    {
                        "split_kind": "leave_topic",
                        "feature_set": feature_set,
                        "n_splits": len(aucs),
                        "auc_mean": float(np.mean(aucs)),
                        "auc_std": float(np.std(aucs)),
                        "auc_min": float(np.min(aucs)),
                        "auc_max": float(np.max(aucs)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(split_rows), pd.DataFrame(leave_rows)


def summarize_interventions(interventions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = interventions.copy()
    summaries = frame.groupby(["direction", "epsilon"], dropna=False).agg(
        n=("prompt_id", "count"),
        delta_entropy_mean=("delta_entropy", "mean"),
        delta_entropy_median=("delta_entropy", "median"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
        delta_varentropy_mean=("delta_varentropy", "mean"),
        top1_changed_rate=("top1_changed_within_topk", "mean"),
        delta_target_prob_mean=("delta_target_prob_topk", "mean"),
        rho_direction_mean=("rho_direction", "mean"),
        trace_fim_direction_mean=("trace_fim_direction", "mean"),
    ).reset_index()

    delta_only = frame[frame["direction"].eq("delta_1d")].copy()
    if delta_only.empty:
        return summaries, pd.DataFrame()
    delta_only["rho_bin"] = delta_only.groupby("epsilon")["rho_direction"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
    )
    binned = delta_only.groupby(["epsilon", "rho_bin"], observed=True).agg(
        n=("prompt_id", "count"),
        rho_mean=("rho_direction", "mean"),
        trace_fim_mean=("trace_fim_direction", "mean"),
        delta_entropy_mean=("delta_entropy", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
        top1_changed_rate=("top1_changed_within_topk", "mean"),
    ).reset_index()
    return summaries, binned


def intervention_residual_correlations(interventions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    frame = interventions[interventions["direction"].eq("delta_1d")].copy()
    controls = ["entropy_before", "varentropy_before", "trace_fim_direction"]
    for eps, group in frame.groupby("epsilon"):
        for outcome in ["delta_entropy", "abs_delta_varentropy", "top1_changed_within_topk"]:
            if group[outcome].nunique() < 2:
                continue
            x_controls = group[controls].to_numpy(dtype=float)
            y = group[outcome].to_numpy(dtype=float)
            rho = group["rho_direction"].to_numpy(dtype=float)
            control_model = LinearRegression().fit(x_controls, y)
            residual_y = y - control_model.predict(x_controls)
            rho_model = LinearRegression().fit(x_controls, rho)
            residual_rho = rho - rho_model.predict(x_controls)
            corr = float(np.corrcoef(residual_rho, residual_y)[0, 1])
            rows.append(
                {
                    "epsilon": float(eps),
                    "outcome": outcome,
                    "n": int(len(group)),
                    "pearson_rho_outcome": float(np.corrcoef(rho, y)[0, 1]),
                    "partial_corr_after_entropy_varentropy_trace": corr,
                }
            )
    return pd.DataFrame(rows)


def summarize_saturation(saturation: pd.DataFrame) -> pd.DataFrame:
    return saturation.groupby(["k", "observed_condition"], dropna=False).agg(
        n=("prompt_id", "count"),
        rho_mean=("rho", "mean"),
        rho_std=("rho", "std"),
        rho_p10=("rho", lambda s: float(np.quantile(s, 0.10))),
        rho_p50=("rho", "median"),
        rho_p90=("rho", lambda s: float(np.quantile(s, 0.90))),
        trace_fim_mean=("trace_fim", "mean"),
    ).reset_index()


def paired_feature_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("rho_fisher_delta", "scalar_layer_profile"),
        ("rho_euclidean_delta", "rho_fisher_delta"),
        ("rho_shuffled_u_delta", "rho_fisher_delta"),
        ("rho_random_u_delta", "rho_fisher_delta"),
        ("rho_matched_random_trace_delta", "rho_fisher_delta"),
        ("scalar_plus_rho_fisher_delta", "scalar_layer_profile"),
        ("scalar_plus_rho_euclidean_delta", "scalar_layer_profile"),
        ("scalar_plus_matched_random_delta", "scalar_layer_profile"),
    ]
    rows: list[dict] = []
    for split_kind, group in summary.groupby("split_kind"):
        auc = group.set_index("feature_set")["auc_mean"]
        for left, right in pairs:
            if left in auc.index and right in auc.index:
                rows.append(
                    {
                        "split_kind": split_kind,
                        "left": left,
                        "right": right,
                        "left_auc_mean": float(auc[left]),
                        "right_auc_mean": float(auc[right]),
                        "delta_auc_mean": float(auc[left] - auc[right]),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    prompts: pd.DataFrame,
    feature_summary: pd.DataFrame,
    feature_deltas: pd.DataFrame,
    intervention_summary: pd.DataFrame,
    intervention_bins: pd.DataFrame,
    residual_corr: pd.DataFrame,
    saturation_summary: pd.DataFrame,
    token_shifts: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame, max_rows: int = 30) -> str:
        if df.empty:
            return "```text\n(empty)\n```"
        return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"

    counts = prompts.groupby(["topic", "observed_condition"]).size().unstack(fill_value=0).reset_index()
    lines = [
        "# DistilBERT Geometry Intervention and Ablation Tests",
        "",
        "Model: distilbert-base-uncased. Dataset: generated factual cloze prompts from the existing factual-error benchmark. Labels are observed model correctness: top-1 target match vs error.",
        "",
        "## Prompt Counts",
        table(counts, max_rows=40),
        "",
        "## Feature-Set AUC Summary",
        table(feature_summary.sort_values(["split_kind", "auc_mean"], ascending=[True, False]), max_rows=40),
        "",
        "## Paired Feature Deltas",
        table(feature_deltas, max_rows=40),
        "",
        "## Controlled Latent Intervention Summary",
        table(intervention_summary, max_rows=30),
        "",
        "## Delta-Direction Intervention by Rho Quartile",
        table(intervention_bins, max_rows=40),
        "",
        "## Intervention Residual Correlations",
        table(residual_corr, max_rows=30),
        "",
        "## Random-Subspace Saturation",
        table(saturation_summary, max_rows=40),
        "",
        "## Directional Token Shift Examples",
        table(
            token_shifts.sort_values("rho_delta", ascending=False)[
                [
                    "prompt_id",
                    "kind",
                    "layer",
                    "rho_delta",
                    "prompt",
                    "target",
                    "final_top1_token",
                    "top_up_tokens",
                    "top_down_tokens",
                ]
            ],
            max_rows=30,
        ),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.eval()

    cases = factual_error_cases()
    if args.max_prompts and args.max_prompts > 0:
        cases = cases[: args.max_prompts]
    prompts, records = build_prompt_table(tokenizer, model, cases, args.top_k)
    prompts.to_csv(args.out_dir / "prompts.csv", index=False)
    metrics, features, saturation, interventions, token_shifts = compute_records(tokenizer, model, prompts, records, args)
    metrics.to_csv(args.out_dir / "metric_ablation_scores.csv", index=False)
    features.to_csv(args.out_dir / "prompt_features_ablation.csv", index=False)
    saturation.to_csv(args.out_dir / "saturation_scores.csv", index=False)
    interventions.to_csv(args.out_dir / "intervention_records.csv", index=False)
    token_shifts.to_csv(args.out_dir / "directional_token_shifts.csv", index=False)

    feature_summary, split_scores, leave_topic = evaluate_feature_sets(features, args)
    feature_summary.to_csv(args.out_dir / "feature_set_ablation_summary.csv", index=False)
    split_scores.to_csv(args.out_dir / "feature_set_ablation_split_scores.csv", index=False)
    leave_topic.to_csv(args.out_dir / "feature_set_ablation_leave_topic.csv", index=False)
    feature_deltas = paired_feature_deltas(feature_summary)
    feature_deltas.to_csv(args.out_dir / "feature_set_ablation_deltas.csv", index=False)

    intervention_summary, intervention_bins = summarize_interventions(interventions)
    intervention_summary.to_csv(args.out_dir / "intervention_summary.csv", index=False)
    intervention_bins.to_csv(args.out_dir / "intervention_by_rho_quartile.csv", index=False)
    residual_corr = intervention_residual_correlations(interventions)
    residual_corr.to_csv(args.out_dir / "intervention_residual_correlations.csv", index=False)
    saturation_summary = summarize_saturation(saturation)
    saturation_summary.to_csv(args.out_dir / "saturation_summary.csv", index=False)

    write_report(
        args.out_dir,
        prompts,
        feature_summary,
        feature_deltas,
        intervention_summary,
        intervention_bins,
        residual_corr,
        saturation_summary,
        token_shifts,
    )
    report_text = (args.out_dir / "report.md").read_text(encoding="utf-8")
    print(report_text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
