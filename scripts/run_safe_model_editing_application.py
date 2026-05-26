from __future__ import annotations

import argparse
import json
import sys
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForMaskedLM

from accessible_varentropy.metrics import softmax_np
from accessible_varentropy.mlm_heads import mlm_logits_from_hidden
from run_topk_gradient_regression_controls import safe_tokenizer
from run_uncertainty_steering_full_battery import (
    build_cases,
    build_records,
    entropy_varentropy,
    fisher_geometry,
    mlm_head_jacobian_fast,
    model_subspaces,
    parse_layers,
    projection,
    topk_distribution,
)


OUT = ROOT / "applications" / "07_safe_model_editing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="distilbert-base-uncased,google/bert_uncased_L-2_H-128_A-2",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--subspace-k", type=int, default=8)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--max-prompts-per-task", type=int, default=8)
    parser.add_argument("--random-subspaces", type=int, default=1)
    parser.add_argument("--edit-margin", type=float, default=0.0)
    parser.add_argument("--max-latent-norm", type=float, default=8.0)
    parser.add_argument("--side-prompts-per-edit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    for rel in ["config", "outputs", "reports"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def load_mlm(model_name: str):
    tokenizer = safe_tokenizer(model_name)
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            local_files_only=True,
            attn_implementation="eager",
        )
    except Exception:
        model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
    model.eval()
    return tokenizer, model


def full_logits(model, hidden: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = mlm_logits_from_hidden(model, hidden.view(1, 1, -1))[0, 0, :]
    return logits.detach().float().cpu().numpy().astype(np.float64)


def apply_delta_logits(model, hidden: torch.Tensor, hidden_direction: np.ndarray, scale: float) -> np.ndarray:
    direction = torch.as_tensor(scale * hidden_direction, dtype=hidden.dtype, device=hidden.device)
    with torch.no_grad():
        logits = mlm_logits_from_hidden(model, (hidden + direction).view(1, 1, -1))[0, 0, :]
    return logits.detach().float().cpu().numpy().astype(np.float64)


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.convert_ids_to_tokens([int(token_id)])[0]


def single_token_id(tokenizer, text: object) -> int | None:
    if text is None or pd.isna(text):
        return None
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    if len(ids) != 1:
        return None
    return int(ids[0])


def is_readable_token(tokenizer, token_id: int) -> bool:
    token = token_text(tokenizer, token_id)
    stripped = token.replace("##", "")
    return stripped.isalpha() and len(stripped) > 1


def choose_edit_target(
    *,
    tokenizer,
    logits: np.ndarray,
    prompt_meta: pd.Series,
    prompt_table: pd.DataFrame,
    source_id: int,
) -> tuple[int | None, str]:
    target_id = None if pd.isna(prompt_meta.get("target_id")) else int(prompt_meta["target_id"])
    if target_id is not None and target_id != source_id:
        return target_id, "correction_to_target"

    same_topic = prompt_table[
        (prompt_table["task"].astype(str).str.startswith("factual"))
        & (prompt_table["topic"] == prompt_meta["topic"])
    ].copy()
    alternatives: list[int] = []
    for target in same_topic["target"].dropna().unique().tolist():
        candidate = single_token_id(tokenizer, target)
        if candidate is not None and candidate != source_id:
            alternatives.append(candidate)
    if alternatives:
        best = max(sorted(set(alternatives)), key=lambda token_id: float(logits[int(token_id)]))
        return int(best), "same_topic_counterfactual"

    for token_id in np.argsort(logits)[-128:][::-1]:
        token_id = int(token_id)
        if token_id != source_id and token_id != target_id and is_readable_token(tokenizer, token_id):
            return token_id, "top_candidate_counterfactual"
    return None, "no_target"


def jaccard_topn(logits_before: np.ndarray, logits_after: np.ndarray, n: int) -> float:
    before = set(np.argsort(logits_before)[-n:].tolist())
    after = set(np.argsort(logits_after)[-n:].tolist())
    return float(len(before & after) / max(len(before | after), 1))


def candidate_kl(logits_before: np.ndarray, logits_after: np.ndarray, token_ids: np.ndarray) -> float:
    p = softmax_np(logits_before[token_ids][None, :])[0]
    q = softmax_np(logits_after[token_ids][None, :])[0]
    return float(np.sum(p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(q, 1e-12, 1.0)))))


def edit_margin(logits: np.ndarray, source_id: int, edit_id: int) -> float:
    return float(logits[int(edit_id)] - logits[int(source_id)])


def binary_search_edit_cost(
    *,
    model,
    hidden: torch.Tensor,
    hidden_direction: np.ndarray,
    source_id: int,
    edit_id: int,
    margin: float,
    estimated_cost: float,
    max_norm: float,
) -> tuple[float, int, np.ndarray]:
    if not np.isfinite(estimated_cost) or estimated_cost <= 0.0:
        estimated_cost = 0.25
    hi = min(max_norm, max(0.05, 1.5 * estimated_cost))
    logits_hi = apply_delta_logits(model, hidden, hidden_direction, hi)
    while edit_margin(logits_hi, source_id, edit_id) < margin and hi < max_norm:
        hi = min(max_norm, hi * 1.75)
        logits_hi = apply_delta_logits(model, hidden, hidden_direction, hi)
    if edit_margin(logits_hi, source_id, edit_id) < margin:
        return max_norm, 0, logits_hi

    lo = 0.0
    best_logits = logits_hi
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        logits_mid = apply_delta_logits(model, hidden, hidden_direction, mid)
        if edit_margin(logits_mid, source_id, edit_id) >= margin:
            hi = mid
            best_logits = logits_mid
        else:
            lo = mid
    return hi, 1, best_logits


def score_side_effects(
    *,
    model,
    prompt_lookup: pd.DataFrame,
    records: list[dict[str, object]],
    layer: int,
    edited_prompt_id: int,
    hidden_direction: np.ndarray,
    edit_cost: float,
    top_k: int,
    side_limit: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float]]:
    candidates = [record for record in records if int(record["prompt_id"]) != int(edited_prompt_id)]
    if not candidates:
        return pd.DataFrame(), {
            "side_n": 0.0,
            "side_top1_flip_rate": float("nan"),
            "side_target_correct_change_rate": float("nan"),
            "side_kl_mean": float("nan"),
            "side_top10_jaccard_mean": float("nan"),
            "side_abs_entropy_delta_mean": float("nan"),
            "side_target_prob_drop_mean": float("nan"),
        }
    order = rng.permutation(len(candidates))[: min(len(candidates), int(side_limit))]
    rows: list[dict[str, object]] = []
    for idx in order:
        record = candidates[int(idx)]
        meta = prompt_lookup.loc[int(record["prompt_id"])]
        hidden = record["hidden_states"][layer][0, int(record["mask_index"]), :].detach()
        before = full_logits(model, hidden)
        after = apply_delta_logits(model, hidden, hidden_direction, edit_cost)
        token_ids, probs_before = topk_distribution(before, top_k)
        probs_after = softmax_np(after[token_ids][None, :])[0]
        entropy_before, _var_before = entropy_varentropy(probs_before)
        entropy_after, _var_after = entropy_varentropy(probs_after)
        top1_before = int(np.argmax(before))
        top1_after = int(np.argmax(after))
        target_id = None if pd.isna(meta["target_id"]) else int(meta["target_id"])
        target_prob_before = float(softmax_np(before[None, :])[0][target_id]) if target_id is not None else float("nan")
        target_prob_after = float(softmax_np(after[None, :])[0][target_id]) if target_id is not None else float("nan")
        rows.append(
            {
                "side_prompt_id": int(record["prompt_id"]),
                "side_task": meta["task"],
                "side_topic": meta["topic"],
                "side_observed_condition": meta["observed_condition"],
                "side_top1_before": top1_before,
                "side_top1_after": top1_after,
                "side_top1_changed": int(top1_before != top1_after),
                "side_target_id": target_id,
                "side_target_correct_before": int(target_id is not None and top1_before == target_id),
                "side_target_correct_after": int(target_id is not None and top1_after == target_id),
                "side_target_correct_changed": int(target_id is not None and (top1_before == target_id) != (top1_after == target_id)),
                "side_target_prob_drop": target_prob_before - target_prob_after if target_id is not None else float("nan"),
                "side_candidate_kl": candidate_kl(before, after, token_ids),
                "side_top10_jaccard": jaccard_topn(before, after, 10),
                "side_entropy_delta": entropy_after - entropy_before,
                "side_abs_entropy_delta": abs(entropy_after - entropy_before),
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "side_n": float(len(df)),
        "side_top1_flip_rate": float(df["side_top1_changed"].mean()) if not df.empty else float("nan"),
        "side_target_correct_change_rate": float(df["side_target_correct_changed"].mean()) if not df.empty else float("nan"),
        "side_kl_mean": float(df["side_candidate_kl"].mean()) if not df.empty else float("nan"),
        "side_top10_jaccard_mean": float(df["side_top10_jaccard"].mean()) if not df.empty else float("nan"),
        "side_abs_entropy_delta_mean": float(df["side_abs_entropy_delta"].mean()) if not df.empty else float("nan"),
        "side_target_prob_drop_mean": float(df["side_target_prob_drop"].dropna().mean()) if df["side_target_prob_drop"].notna().any() else float("nan"),
    }
    return df, summary


def run_model(args: argparse.Namespace, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"load_model {model_name}", flush=True)
    tokenizer, model = load_mlm(model_name)
    cases = build_cases(args.seed, args.max_prompts_per_task)
    prompt_table, records = build_records(tokenizer, model, cases, args.top_k)
    if prompt_table.empty or not records:
        return prompt_table.assign(model=model_name), pd.DataFrame(), pd.DataFrame()
    n_layers = len(records[0]["hidden_states"]) - 1
    layers = [layer for layer in parse_layers(args.layers, n_layers) if 1 <= layer <= n_layers]
    prompt_table = prompt_table.assign(model=model_name)
    prompt_lookup = prompt_table.set_index("prompt_id")
    factual_prompt_ids = set(
        prompt_table[prompt_table["task"].astype(str).str.startswith("factual")]["prompt_id"].astype(int).tolist()
    )
    stable_model_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(model_name))
    rng = np.random.default_rng(args.seed + stable_model_offset)
    edit_rows: list[dict[str, object]] = []
    side_rows: list[pd.DataFrame] = []

    for layer in layers:
        print(f"editing_layer {model_name} {layer}", flush=True)
        subspaces = model_subspaces(records, layer, [args.subspace_k], args.random_subspaces, args.seed)
        for record in records:
            prompt_id = int(record["prompt_id"])
            if prompt_id not in factual_prompt_ids:
                continue
            meta = prompt_lookup.loc[prompt_id]
            hidden = record["hidden_states"][layer][0, int(record["mask_index"]), :].detach()
            logits_before = full_logits(model, hidden)
            source_id = int(np.argmax(logits_before))
            edit_id, target_kind = choose_edit_target(
                tokenizer=tokenizer,
                logits=logits_before,
                prompt_meta=meta,
                prompt_table=prompt_table,
                source_id=source_id,
            )
            if edit_id is None or edit_id == source_id:
                continue
            top_ids, _top_probs = topk_distribution(logits_before, args.top_k)
            candidate_ids = list(dict.fromkeys([int(x) for x in top_ids.tolist()] + [source_id, int(edit_id)]))
            token_ids = np.asarray(candidate_ids, dtype=np.int64)
            selected_logits = logits_before[token_ids]
            probs = softmax_np(selected_logits[None, :])[0].astype(np.float64)
            geometry = fisher_geometry(probs)
            jac = mlm_head_jacobian_fast(model, hidden, token_ids)
            source_pos = int(np.where(token_ids == source_id)[0][0])
            edit_pos = int(np.where(token_ids == int(edit_id))[0][0])
            current_required = max(
                0.0,
                float(logits_before[source_id] - logits_before[int(edit_id)] + args.edit_margin),
            )

            for family, subspace_k, rep, basis in subspaces:
                jb = jac @ basis
                whitened = geometry.sqrt_fisher @ jb
                w_acc, rank = projection(whitened, geometry.w)
                denom = max(float(geometry.w @ geometry.w), 1e-12)
                rho = float((w_acc @ w_acc) / denom)
                edit_grad = (jac[edit_pos] - jac[source_pos]) @ basis
                edit_grad_norm = float(np.linalg.norm(edit_grad))
                if edit_grad_norm < 1e-12:
                    estimated_cost = float("inf")
                    edit_success = 0
                    actual_cost = float(args.max_latent_norm)
                    logits_after = logits_before.copy()
                    hidden_direction = np.zeros((basis.shape[0],), dtype=np.float64)
                else:
                    z_unit = edit_grad / edit_grad_norm
                    hidden_direction = (basis @ z_unit).astype(np.float64)
                    estimated_cost = current_required / edit_grad_norm
                    actual_cost, edit_success, logits_after = binary_search_edit_cost(
                        model=model,
                        hidden=hidden,
                        hidden_direction=hidden_direction,
                        source_id=source_id,
                        edit_id=int(edit_id),
                        margin=args.edit_margin,
                        estimated_cost=estimated_cost,
                        max_norm=args.max_latent_norm,
                    )
                full_probs_before = softmax_np(logits_before[None, :])[0]
                full_probs_after = softmax_np(logits_after[None, :])[0]
                source_prob_before = float(full_probs_before[source_id])
                source_prob_after = float(full_probs_after[source_id])
                edit_prob_before = float(full_probs_before[int(edit_id)])
                edit_prob_after = float(full_probs_after[int(edit_id)])
                after_top1 = int(np.argmax(logits_after))
                side_df, side_summary = score_side_effects(
                    model=model,
                    prompt_lookup=prompt_lookup,
                    records=records,
                    layer=layer,
                    edited_prompt_id=prompt_id,
                    hidden_direction=hidden_direction,
                    edit_cost=float(actual_cost),
                    top_k=args.top_k,
                    side_limit=args.side_prompts_per_edit,
                    rng=rng,
                )
                if not side_df.empty:
                    side_df = side_df.assign(
                        model=model_name,
                        prompt_id=prompt_id,
                        layer=layer,
                        subspace_family=family,
                        subspace_k=subspace_k,
                        rep=rep,
                    )
                    side_rows.append(side_df)
                edit_rows.append(
                    {
                        "model": model_name,
                        "prompt_id": prompt_id,
                        "prompt": meta["prompt"],
                        "task": meta["task"],
                        "topic": meta["topic"],
                        "observed_condition": meta["observed_condition"],
                        "layer": layer,
                        "subspace_family": family,
                        "subspace_k": subspace_k,
                        "rep": rep,
                        "rho": rho,
                        "v_access": float(w_acc @ w_acc),
                        "varentropy": float(geometry.varentropy),
                        "entropy": float(geometry.entropy),
                        "confidence": source_prob_before,
                        "margin": float(np.sort(logits_before)[-1] - np.sort(logits_before)[-2]),
                        "source_id": source_id,
                        "source_token": token_text(tokenizer, source_id),
                        "edit_id": int(edit_id),
                        "edit_token": token_text(tokenizer, int(edit_id)),
                        "edit_target_kind": target_kind,
                        "initial_target_id": None if pd.isna(meta["target_id"]) else int(meta["target_id"]),
                        "initial_target_token": meta["target"],
                        "initial_correct": int(meta["observed_condition"] == "correct"),
                        "rank_whitened_jb": rank,
                        "current_logit_gap_source_minus_edit": float(logits_before[source_id] - logits_before[int(edit_id)]),
                        "required_logit_change": current_required,
                        "edit_grad_norm": edit_grad_norm,
                        "estimated_linear_min_cost": estimated_cost,
                        "actual_min_cost": actual_cost,
                        "edit_success": int(edit_success),
                        "after_top1_id": after_top1,
                        "after_top1_token": token_text(tokenizer, after_top1),
                        "after_edit_margin": edit_margin(logits_after, source_id, int(edit_id)),
                        "source_prob_before": source_prob_before,
                        "source_prob_after": source_prob_after,
                        "edit_prob_before": edit_prob_before,
                        "edit_prob_after": edit_prob_after,
                        "edit_prob_gain": edit_prob_after - edit_prob_before,
                        "source_prob_drop": source_prob_before - source_prob_after,
                        "candidate_kl": candidate_kl(logits_before, logits_after, token_ids),
                        "top10_jaccard": jaccard_topn(logits_before, logits_after, 10),
                        "grad_entropy_proj_norm": float(np.linalg.norm(whitened.T @ geometry.w)),
                        "fisher_output_fro_norm": float(np.linalg.norm(whitened)),
                        **side_summary,
                    }
                )
    side = pd.concat(side_rows, ignore_index=True) if side_rows else pd.DataFrame()
    edits = pd.DataFrame(edit_rows)
    return prompt_table, edits, side


def predictor_benchmark(edits: pd.DataFrame) -> pd.DataFrame:
    if edits.empty:
        return pd.DataFrame()
    df = edits.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "rho",
            "actual_min_cost",
            "estimated_linear_min_cost",
            "side_kl_mean",
            "edit_success",
            "confidence",
            "entropy",
            "varentropy",
            "margin",
            "edit_grad_norm",
            "grad_entropy_proj_norm",
            "fisher_output_fro_norm",
        ]
    )
    if len(df) < 8:
        return pd.DataFrame()
    predictors = [
        "rho",
        "confidence",
        "entropy",
        "varentropy",
        "margin",
        "edit_grad_norm",
        "grad_entropy_proj_norm",
        "fisher_output_fro_norm",
    ]
    rows: list[dict[str, object]] = []
    outcomes = {
        "negative_actual_min_cost": -df["actual_min_cost"].to_numpy(dtype=float),
        "negative_estimated_linear_min_cost": -df["estimated_linear_min_cost"].clip(upper=1e6).to_numpy(dtype=float),
        "negative_side_kl_mean": -df["side_kl_mean"].to_numpy(dtype=float),
        "edit_success": df["edit_success"].to_numpy(dtype=float),
        "safe_success": (
            (df["edit_success"].eq(1))
            & (df["side_top1_flip_rate"].fillna(1.0) <= 0.05)
            & (df["side_top10_jaccard_mean"].fillna(0.0) >= 0.90)
        ).astype(float).to_numpy(),
    }
    for outcome_name, y in outcomes.items():
        binary = set(np.unique(y)).issubset({0.0, 1.0})
        for predictor in predictors:
            x = df[predictor].to_numpy(dtype=float)
            corr = spearmanr(x, y).correlation
            rows.append(
                {
                    "outcome": outcome_name,
                    "model": "single_predictor",
                    "predictor": predictor,
                    "metric": "spearman",
                    "value": float(corr) if corr == corr else float("nan"),
                    "n": len(df),
                }
            )
        for name, cols in [
            ("scalar_baseline", ["confidence", "entropy", "varentropy", "margin"]),
            ("edit_gradient_baseline", ["confidence", "entropy", "varentropy", "margin", "edit_grad_norm"]),
            ("gradient_plus_rho", ["confidence", "entropy", "varentropy", "margin", "edit_grad_norm", "rho"]),
        ]:
            xx = df[cols].to_numpy(dtype=float)
            xx = StandardScaler().fit_transform(xx)
            if binary:
                if len(np.unique(y)) < 2:
                    continue
                model = LogisticRegression(max_iter=2000, class_weight="balanced")
                model.fit(xx, y)
                score = model.predict_proba(xx)[:, 1]
                rows.append(
                    {
                        "outcome": outcome_name,
                        "model": name,
                        "predictor": ",".join(cols),
                        "metric": "auroc",
                        "value": float(roc_auc_score(y, score)),
                        "n": len(df),
                    }
                )
                rows.append(
                    {
                        "outcome": outcome_name,
                        "model": name,
                        "predictor": ",".join(cols),
                        "metric": "auprc",
                        "value": float(average_precision_score(y, score)),
                        "n": len(df),
                    }
                )
            else:
                model = LinearRegression()
                model.fit(xx, y)
                pred = model.predict(xx)
                rows.append(
                    {
                        "outcome": outcome_name,
                        "model": name,
                        "predictor": ",".join(cols),
                        "metric": "r2",
                        "value": float(model.score(xx, y)),
                        "n": len(df),
                    }
                )
                rows.append(
                    {
                        "outcome": outcome_name,
                        "model": name,
                        "predictor": ",".join(cols),
                        "metric": "pred_spearman",
                        "value": float(spearmanr(pred, y).correlation),
                        "n": len(df),
                    }
                )
    return pd.DataFrame(rows)


def summarize(edits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edits.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = (
        edits.groupby(["model", "subspace_family"], dropna=False)
        .agg(
            n=("prompt_id", "size"),
            rho_mean=("rho", "mean"),
            actual_min_cost_mean=("actual_min_cost", "mean"),
            estimated_linear_min_cost_mean=("estimated_linear_min_cost", "mean"),
            edit_success_rate=("edit_success", "mean"),
            safe_success_rate=(
                "edit_success",
                lambda s: float(
                    (
                        edits.loc[s.index, "edit_success"].eq(1)
                        & (edits.loc[s.index, "side_top1_flip_rate"].fillna(1.0) <= 0.05)
                        & (edits.loc[s.index, "side_top10_jaccard_mean"].fillna(0.0) >= 0.90)
                    ).mean()
                ),
            ),
            side_top1_flip_rate_mean=("side_top1_flip_rate", "mean"),
            side_kl_mean=("side_kl_mean", "mean"),
            side_top10_jaccard_mean=("side_top10_jaccard_mean", "mean"),
        )
        .reset_index()
    )
    qdf = edits.copy()
    qdf["rho_quartile"] = qdf.groupby(["model"])["rho"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"], duplicates="drop")
    )
    quartiles = (
        qdf.groupby(["rho_quartile"], observed=False)
        .agg(
            n=("prompt_id", "size"),
            rho_mean=("rho", "mean"),
            actual_min_cost_mean=("actual_min_cost", "mean"),
            estimated_linear_min_cost_mean=("estimated_linear_min_cost", "mean"),
            edit_success_rate=("edit_success", "mean"),
            side_top1_flip_rate_mean=("side_top1_flip_rate", "mean"),
            side_kl_mean=("side_kl_mean", "mean"),
            side_top10_jaccard_mean=("side_top10_jaccard_mean", "mean"),
        )
        .reset_index()
    )
    return summary, quartiles


def write_docs(out_dir: Path, args: argparse.Namespace, metadata: dict[str, object]) -> None:
    outputs = out_dir / "outputs"
    parts = ["# Safe Model Editing Diagnostic\n"]
    parts.append(
        "Local representation-editing diagnostic for whether uncertainty accessibility predicts edit cost and side effects.\n"
    )
    parts.append("## Scope\n")
    parts.append(
        "This is not a persistent ROME/MEMIT-style weight edit. It applies a same-layer hidden-state delta in an internal route and measures the local edit plus side effects on other prompts.\n"
    )
    parts.append("## Interpretation\n")
    parts.append(
        "This first diagnostic does not support the strong claim that uncertainty accessibility alone predicts safe model editing. `rho` is weakly positive for safe-success, but target-specific edit-gradient and Fisher-output norms are stronger predictors of edit cost. The current result should be treated as a boundary case and as a scaffold for a stronger persistent weight-editing experiment.\n"
    )
    parts.append("## Setup\n")
    parts.append("```json\n" + json.dumps(metadata, indent=2) + "\n```\n")
    for title, filename in [
        ("Edit Summary", "edit_summary.csv"),
        ("Rho Quartiles", "rho_quartile_summary.csv"),
        ("Predictor Benchmark", "predictor_benchmark.csv"),
        ("Qualitative Examples", "qualitative_safe_edit_examples.csv"),
    ]:
        path = outputs / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        parts.append(f"## {title}\n")
        if df.empty:
            parts.append("```text\n(empty)\n```\n")
        else:
            parts.append("```text\n" + df.head(30).to_string(index=False) + "\n```\n")
    (out_dir / "reports" / "report.md").write_text("\n".join(parts), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Safe Model Editing Diagnostic\n\n"
        "## Claim\n\n"
        "Accessible uncertainty may identify facts/beliefs that can be locally edited with smaller interventions and fewer side effects.\n\n"
        "## Scope\n\n"
        "The runner performs local representation edits in masked-LM logit-lens space. It is a safe model-editing diagnostic, not a persistent weight-editing method.\n\n"
        "## Result Status\n\n"
        "The first run is mixed/negative for the strong claim. `rho` is weakly positive for safe-success, but target-specific edit-gradient and Fisher-output norms are stronger predictors of edit cost. Keep this as a diagnostic unless a persistent weight-editing version or a larger edit suite reverses the result.\n\n"
        "## Artifacts\n\n"
        "- `outputs/edit_records.csv`\n"
        "- `outputs/side_effect_records.csv`\n"
        "- `outputs/edit_summary.csv`\n"
        "- `outputs/rho_quartile_summary.csv`\n"
        "- `outputs/predictor_benchmark.csv`\n"
        "- `outputs/qualitative_safe_edit_examples.csv`\n"
        "- `reports/report.md`\n\n"
        "## Reproduce\n\n"
        "```powershell\n"
        "python scripts\\run_safe_model_editing_application.py --models distilbert-base-uncased,google/bert_uncased_L-2_H-128_A-2 --max-prompts-per-task 8 --top-k 32 --subspace-k 8 --random-subspaces 1 --side-prompts-per-edit 12 --seed 20260601\n"
        "```\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    prompt_tables = []
    edit_tables = []
    side_tables = []
    for model_name in [part.strip() for part in args.models.split(",") if part.strip()]:
        try:
            prompts, edits, side = run_model(args, model_name)
        except Exception as exc:
            print(f"skip_model {model_name}: {exc}", flush=True)
            continue
        prompt_tables.append(prompts)
        edit_tables.append(edits)
        side_tables.append(side)

    prompts_df = pd.concat(prompt_tables, ignore_index=True) if prompt_tables else pd.DataFrame()
    edits_df = pd.concat(edit_tables, ignore_index=True) if edit_tables else pd.DataFrame()
    side_df = pd.concat(side_tables, ignore_index=True) if side_tables else pd.DataFrame()
    summary, quartiles = summarize(edits_df)
    benchmark = predictor_benchmark(edits_df)
    examples = (
        edits_df[
            (edits_df["edit_success"] == 1)
            & (edits_df["side_top1_flip_rate"].fillna(1.0) <= 0.05)
            & (edits_df["side_top10_jaccard_mean"].fillna(0.0) >= 0.90)
        ]
        .sort_values(["actual_min_cost", "side_kl_mean"], ascending=[True, True])
        .head(20)
        if not edits_df.empty
        else pd.DataFrame()
    )

    out = args.out_dir / "outputs"
    prompts_df.to_csv(out / "prompt_table.csv", index=False)
    edits_df.to_csv(out / "edit_records.csv", index=False)
    side_df.to_csv(out / "side_effect_records.csv", index=False)
    summary.to_csv(out / "edit_summary.csv", index=False)
    quartiles.to_csv(out / "rho_quartile_summary.csv", index=False)
    benchmark.to_csv(out / "predictor_benchmark.csv", index=False)
    examples.to_csv(out / "qualitative_safe_edit_examples.csv", index=False)

    metadata = {
        "status": "completed",
        "seed": args.seed,
        "models": [part.strip() for part in args.models.split(",") if part.strip()],
        "top_k": args.top_k,
        "subspace_k": args.subspace_k,
        "layers": args.layers,
        "max_prompts_per_task": args.max_prompts_per_task,
        "random_subspaces": args.random_subspaces,
        "edit_margin": args.edit_margin,
        "max_latent_norm": args.max_latent_norm,
        "side_prompts_per_edit": args.side_prompts_per_edit,
        "n_prompts": int(len(prompts_df)),
        "n_edit_records": int(len(edits_df)),
        "n_side_effect_records": int(len(side_df)),
    }
    (args.out_dir / "config" / "reproduce.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_docs(args.out_dir, args, metadata)


if __name__ == "__main__":
    main()
