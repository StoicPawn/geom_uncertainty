from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForMaskedLM, BertTokenizer

from accessible_varentropy.bert_mlm import encode_prompt, hidden_states_and_logits, token_id_for_word
from accessible_varentropy.mlm_heads import mlm_head_jacobian
from run_distilbert_geometry_interventions import (
    fisher_parts,
    fisher_scores_for_basis,
    layer_logits,
    normalize,
    orthonormalize,
    topk_distribution,
)
from run_factual_error_deep_benchmark import factual_error_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "layerwise_k_adjusted_rho")
    parser.add_argument("--top-k-output", type=int, default=32)
    parser.add_argument("--subspace-ks", default="1,2,4,8,16,32")
    parser.add_argument("--random-reps", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=20260523)
    parser.add_argument("--splits", type=int, default=80)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--logreg-c", type=float, default=0.1)
    parser.add_argument("--max-prompts", type=int, default=0)
    return parser.parse_args()


def parse_ks(text: str, hidden_dim: int) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    return [k for k in values if 1 <= k <= hidden_dim]


def pca_basis(matrix: np.ndarray, max_k: int) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:max_k].T.astype(np.float64)


def collect_hidden_table(tokenizer, model, top_k_output: int, max_prompts: int) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    cases = factual_error_cases()
    if max_prompts > 0:
        cases = cases[:max_prompts]
    prompt_rows: list[dict[str, object]] = []
    hidden_rows: list[np.ndarray] = []
    mask_indices: list[int] = []
    for prompt_id, case in enumerate(cases):
        target_id = token_id_for_word(tokenizer, case.target)
        if target_id is None:
            continue
        encoded, mask_index = encode_prompt(tokenizer, case.prompt.replace("[MASK]", tokenizer.mask_token))
        hidden_states, final_logits = hidden_states_and_logits(model, encoded)
        final_mask_logits = final_logits[0, mask_index, :].detach().cpu().numpy().astype(np.float64)
        top_ids, top_probs = topk_distribution(final_mask_logits, top_k_output)
        sorted_all = np.argsort(final_mask_logits)[::-1]
        top1_id = int(top_ids[0])
        prompt_rows.append(
            {
                "prompt_id": len(prompt_rows),
                "source_case_index": prompt_id,
                "prompt": case.prompt,
                "target": case.target,
                "target_id": int(target_id),
                "topic": case.topic,
                "fact_id": case.fact_id,
                "template_id": case.template_id,
                "observed_condition": "correct" if top1_id == int(target_id) else "error",
                "target_rank_final": int(np.where(sorted_all == target_id)[0][0] + 1),
                "final_top1_token": tokenizer.convert_ids_to_tokens([top1_id])[0],
                "final_top1_prob_topk": float(top_probs[0]),
                "final_top8_tokens": " ".join(tokenizer.convert_ids_to_tokens([int(v) for v in top_ids[:8]])),
            }
        )
        hidden_stack = np.stack(
            [
                state[0, mask_index, :].detach().cpu().numpy().astype(np.float64)
                for state in hidden_states
            ],
            axis=0,
        )
        hidden_rows.append(hidden_stack)
        mask_indices.append(mask_index)
    return pd.DataFrame(prompt_rows), np.stack(hidden_rows, axis=0), mask_indices


def make_bases(hidden: np.ndarray, ks: list[int], rng: np.random.Generator, random_reps: int) -> tuple[dict, dict]:
    n, num_layers, hidden_dim = hidden.shape
    max_k = max(ks)
    structured: dict[tuple[str, int, int], np.ndarray] = {}
    random_bases: dict[tuple[int, int], list[np.ndarray]] = {}
    for layer in range(num_layers):
        state_basis = pca_basis(hidden[:, layer, :], max_k)
        for k in ks:
            structured[("state_pca", layer, k)] = state_basis[:, :k]
            random_bases[(layer, k)] = [
                orthonormalize(rng.normal(size=(hidden_dim, k)), k)
                for _ in range(random_reps)
            ]
        if layer < num_layers - 1:
            delta = hidden[:, layer + 1, :] - hidden[:, layer, :]
            delta_basis = pca_basis(delta, max_k)
            for k in ks:
                structured[("delta_pca", layer, k)] = delta_basis[:, :k]
    return structured, random_bases


def classifier(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=c),
    )


def compute_layerwise_scores(
    tokenizer,
    model,
    prompts: pd.DataFrame,
    hidden: np.ndarray,
    ks: list[int],
    structured: dict,
    random_bases: dict,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    num_layers = hidden.shape[1]
    for prompt_idx, meta in prompts.reset_index(drop=True).iterrows():
        if prompt_idx % 20 == 0:
            print(f"layerwise_prompt {prompt_idx + 1}/{len(prompts)}", flush=True)
        feature: dict[str, object] = {
            "prompt_id": int(meta["prompt_id"]),
            "observed_condition": meta["observed_condition"],
            "topic": meta["topic"],
            "fact_id": meta["fact_id"],
            "template_id": meta["template_id"],
        }
        for layer in range(num_layers):
            hidden_vec_np = hidden[prompt_idx, layer, :]
            hidden_vec = torch.as_tensor(hidden_vec_np, dtype=next(model.parameters()).dtype)
            logits = layer_logits(model, hidden_vec)
            token_ids, probs = topk_distribution(logits, args.top_k_output)
            parts = fisher_parts(probs)
            jac = mlm_head_jacobian(model, hidden_vec, token_ids)
            full_grad = jac.T @ parts.fisher_u
            feature[f"entropy_topk_l{layer}"] = parts.entropy
            feature[f"varentropy_topk_l{layer}"] = parts.varentropy
            feature[f"trace_fim_full_l{layer}"] = float(np.trace(jac.T @ parts.fisher @ jac))
            feature[f"grad_norm_sq_full_l{layer}"] = float(full_grad @ full_grad)
            feature[f"top1_prob_topk_layer_l{layer}"] = float(probs[0])

            random_cache: dict[int, tuple[float, float]] = {}
            for k in ks:
                random_rhos = [
                    fisher_scores_for_basis(parts, jac, basis)["rho"]
                    for basis in random_bases[(layer, k)]
                ]
                random_cache[k] = (float(np.mean(random_rhos)), float(np.std(random_rhos)))

            for subspace_name in ["state_pca", "delta_pca"]:
                if subspace_name == "delta_pca" and layer >= num_layers - 1:
                    continue
                for k in ks:
                    basis = structured[(subspace_name, layer, k)]
                    score = fisher_scores_for_basis(parts, jac, basis)
                    random_mean, random_std = random_cache[k]
                    adjusted = score["rho"] - random_mean
                    rows.append(
                        {
                            "prompt_id": int(meta["prompt_id"]),
                            "layer": layer,
                            "k": k,
                            "subspace": subspace_name,
                            "observed_condition": meta["observed_condition"],
                            "topic": meta["topic"],
                            "fact_id": meta["fact_id"],
                            "rho_structured": score["rho"],
                            "rho_random_mean": random_mean,
                            "rho_random_std": random_std,
                            "rho_adjusted": adjusted,
                            "rho_adjusted_z": adjusted / (random_std + 1e-12),
                            "trace_fim_structured": score["trace_fim"],
                            "entropy_topk": parts.entropy,
                            "varentropy_topk": parts.varentropy,
                        }
                    )
                    prefix = f"{subspace_name}_k{k}_l{layer}"
                    feature[f"rho_{prefix}"] = score["rho"]
                    feature[f"rho_adj_{prefix}"] = adjusted

            if layer < num_layers - 1:
                delta_forward = normalize(hidden[prompt_idx, layer + 1, :] - hidden[prompt_idx, layer, :])[:, None]
                score = fisher_scores_for_basis(parts, jac, delta_forward)
                random_mean, random_std = random_cache[1]
                adjusted = score["rho"] - random_mean
                rows.append(
                    {
                        "prompt_id": int(meta["prompt_id"]),
                        "layer": layer,
                        "k": 1,
                        "subspace": "delta_forward_1d",
                        "observed_condition": meta["observed_condition"],
                        "topic": meta["topic"],
                        "fact_id": meta["fact_id"],
                        "rho_structured": score["rho"],
                        "rho_random_mean": random_mean,
                        "rho_random_std": random_std,
                        "rho_adjusted": adjusted,
                        "rho_adjusted_z": adjusted / (random_std + 1e-12),
                        "trace_fim_structured": score["trace_fim"],
                        "entropy_topk": parts.entropy,
                        "varentropy_topk": parts.varentropy,
                    }
                )
                feature[f"rho_delta_forward_1d_l{layer}"] = score["rho"]
                feature[f"rho_adj_delta_forward_1d_l{layer}"] = adjusted
        feature_rows.append(feature)
    return pd.DataFrame(rows), pd.DataFrame(feature_rows)


def summarize_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = scores.groupby(["subspace", "k", "layer", "observed_condition"], dropna=False).agg(
        n=("prompt_id", "nunique"),
        rho_structured_mean=("rho_structured", "mean"),
        rho_random_mean=("rho_random_mean", "mean"),
        rho_adjusted_mean=("rho_adjusted", "mean"),
        rho_adjusted_median=("rho_adjusted", "median"),
        rho_adjusted_z_mean=("rho_adjusted_z", "mean"),
        trace_fim_structured_mean=("trace_fim_structured", "mean"),
    ).reset_index()
    trajectory = scores.groupby(["subspace", "k", "layer"], dropna=False).agg(
        n=("prompt_id", "nunique"),
        rho_structured_mean=("rho_structured", "mean"),
        rho_random_mean=("rho_random_mean", "mean"),
        rho_adjusted_mean=("rho_adjusted", "mean"),
        rho_adjusted_std=("rho_adjusted", "std"),
        error_minus_correct_adjusted=(
            "rho_adjusted",
            lambda s: float(
                s[scores.loc[s.index, "observed_condition"].eq("error")].mean()
                - s[scores.loc[s.index, "observed_condition"].eq("correct")].mean()
            ),
        ),
    ).reset_index()
    return summary, trajectory


def feature_sets(features: pd.DataFrame, ks: list[int]) -> dict[str, list[str]]:
    scalar = [
        col
        for col in features.columns
        if col.startswith(
            (
                "entropy_topk_l",
                "varentropy_topk_l",
                "trace_fim_full_l",
                "grad_norm_sq_full_l",
                "top1_prob_topk_layer_l",
            )
        )
    ]
    sets: dict[str, list[str]] = {"scalar_layer_profile": scalar}
    for subspace in ["state_pca", "delta_pca"]:
        for k in ks:
            raw_prefix = f"rho_{subspace}_k{k}_l"
            adj_prefix = f"rho_adj_{subspace}_k{k}_l"
            raw_cols = [col for col in features.columns if col.startswith(raw_prefix)]
            adj_cols = [col for col in features.columns if col.startswith(adj_prefix)]
            if raw_cols:
                sets[f"rho_{subspace}_k{k}"] = raw_cols
            if adj_cols:
                sets[f"rho_adj_{subspace}_k{k}"] = adj_cols
                sets[f"scalar_plus_adj_{subspace}_k{k}"] = scalar + adj_cols
    delta_raw = [col for col in features.columns if col.startswith("rho_delta_forward_1d_l")]
    delta_adj = [col for col in features.columns if col.startswith("rho_adj_delta_forward_1d_l")]
    if delta_raw:
        sets["rho_delta_forward_1d"] = delta_raw
    if delta_adj:
        sets["rho_adj_delta_forward_1d"] = delta_adj
        sets["scalar_plus_adj_delta_forward_1d"] = scalar + delta_adj
    return {name: cols for name, cols in sets.items() if cols}


def evaluate_features(features: pd.DataFrame, ks: list[int], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fsets = feature_sets(features, ks)
    y = features["observed_condition"].eq("error").astype(int).to_numpy()
    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    groups = features["fact_id"].astype(str).to_numpy()
    gss = GroupShuffleSplit(n_splits=args.splits, test_size=args.test_size, random_state=args.random_seed)
    for name, cols in fsets.items():
        x = features[cols].fillna(0.0)
        aucs: list[float] = []
        for split_idx, (train_idx, test_idx) in enumerate(gss.split(x, y, groups)):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            clf = classifier(args.logreg_c)
            clf.fit(x.iloc[train_idx].to_numpy(), y[train_idx])
            pred = clf.predict_proba(x.iloc[test_idx].to_numpy())[:, 1]
            auc = float(roc_auc_score(y[test_idx], pred))
            aucs.append(auc)
            split_rows.append({"split_kind": "fact_group", "split": split_idx, "feature_set": name, "auc": auc})
        if aucs:
            rows.append(
                {
                    "split_kind": "fact_group",
                    "feature_set": name,
                    "n_splits": len(aucs),
                    "auc_mean": float(np.mean(aucs)),
                    "auc_std": float(np.std(aucs)),
                    "auc_min": float(np.min(aucs)),
                    "auc_max": float(np.max(aucs)),
                }
            )

    leave_rows: list[dict[str, object]] = []
    topic_groups = features["topic"].astype(str).to_numpy()
    if len(np.unique(topic_groups)) >= 2:
        logo = LeaveOneGroupOut()
        for name, cols in fsets.items():
            x = features[cols].fillna(0.0)
            aucs = []
            for train_idx, test_idx in logo.split(x, y, topic_groups):
                topic = str(topic_groups[test_idx][0])
                if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                    continue
                clf = classifier(args.logreg_c)
                clf.fit(x.iloc[train_idx].to_numpy(), y[train_idx])
                pred = clf.predict_proba(x.iloc[test_idx].to_numpy())[:, 1]
                auc = float(roc_auc_score(y[test_idx], pred))
                aucs.append(auc)
                leave_rows.append(
                    {
                        "held_out_topic": topic,
                        "feature_set": name,
                        "n_test": int(len(test_idx)),
                        "test_errors": int(y[test_idx].sum()),
                        "auc": auc,
                    }
                )
            if aucs:
                rows.append(
                    {
                        "split_kind": "leave_topic",
                        "feature_set": name,
                        "n_splits": len(aucs),
                        "auc_mean": float(np.mean(aucs)),
                        "auc_std": float(np.std(aucs)),
                        "auc_min": float(np.min(aucs)),
                        "auc_max": float(np.max(aucs)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(split_rows), pd.DataFrame(leave_rows)


def paired_deltas(eval_summary: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidate_pairs: list[tuple[str, str]] = [
        ("rho_adj_delta_forward_1d", "rho_delta_forward_1d"),
        ("rho_adj_delta_forward_1d", "scalar_layer_profile"),
        ("scalar_plus_adj_delta_forward_1d", "scalar_layer_profile"),
    ]
    for subspace in ["state_pca", "delta_pca"]:
        for k in ks:
            candidate_pairs.extend(
                [
                    (f"rho_adj_{subspace}_k{k}", f"rho_{subspace}_k{k}"),
                    (f"rho_adj_{subspace}_k{k}", "scalar_layer_profile"),
                    (f"scalar_plus_adj_{subspace}_k{k}", "scalar_layer_profile"),
                ]
            )
    for split_kind, group in eval_summary.groupby("split_kind"):
        aucs = group.set_index("feature_set")["auc_mean"]
        for left, right in candidate_pairs:
            if left in aucs.index and right in aucs.index:
                rows.append(
                    {
                        "split_kind": split_kind,
                        "left": left,
                        "right": right,
                        "left_auc_mean": float(aucs[left]),
                        "right_auc_mean": float(aucs[right]),
                        "delta_auc_mean": float(aucs[left] - aucs[right]),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    prompts: pd.DataFrame,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    trajectory: pd.DataFrame,
    eval_summary: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame, max_rows: int = 40, columns: list[str] | None = None) -> str:
        if df.empty:
            return "```text\n(empty)\n```"
        frame = df.copy()
        if columns is not None:
            frame = frame[[col for col in columns if col in frame.columns]]
        return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"

    counts = prompts.groupby(["topic", "observed_condition"]).size().unstack(fill_value=0).reset_index()
    best_eval = eval_summary.sort_values(["split_kind", "auc_mean"], ascending=[True, False])
    traj_view = trajectory[
        trajectory["subspace"].isin(["delta_forward_1d", "delta_pca", "state_pca"])
        & trajectory["k"].isin([1, 4, 16, 32])
    ].sort_values(["subspace", "k", "layer"])
    mean_adjusted = scores.groupby(["subspace", "k"], as_index=False).agg(
        rho_structured_mean=("rho_structured", "mean"),
        rho_random_mean=("rho_random_mean", "mean"),
        rho_adjusted_mean=("rho_adjusted", "mean"),
        rho_adjusted_z_mean=("rho_adjusted_z", "mean"),
    ).sort_values(["subspace", "k"])
    lines = [
        "# Layerwise k-Constrained Random-Adjusted Rho",
        "",
        "Definition used in this run:",
        "",
        "```text",
        "rho_{ell,k}(B) = ||P_Im(F_ell^{1/2} J_ell B_{ell,k}) F_ell^{1/2} u_ell||^2 / ||F_ell^{1/2} u_ell||^2",
        "rho_adjusted = rho_structured - mean_r rho_random(layer=ell, k)",
        "```",
        "",
        "Model: distilbert-base-uncased. Readout/Jacobian: MLM-head logit-lens top-k output lens with k_output=32.",
        "Structured subspaces: state PCA, forward-update PCA, and per-example forward delta direction `h_{ell+1}-h_ell` with k=1.",
        "",
        "## Prompt Counts",
        table(counts, max_rows=40),
        "",
        "## Mean Structured vs Random-Adjusted Rho",
        table(mean_adjusted, max_rows=80),
        "",
        "## Layer Trajectory Summary",
        table(
            traj_view,
            max_rows=120,
            columns=[
                "subspace",
                "k",
                "layer",
                "rho_structured_mean",
                "rho_random_mean",
                "rho_adjusted_mean",
                "error_minus_correct_adjusted",
            ],
        ),
        "",
        "## Predictive Feature AUCs",
        table(best_eval, max_rows=80),
        "",
        "## Paired AUC Deltas",
        table(deltas, max_rows=120),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.eval()

    prompts, hidden, _mask_indices = collect_hidden_table(tokenizer, model, args.top_k_output, args.max_prompts)
    ks = parse_ks(args.subspace_ks, hidden.shape[-1])
    structured, random_bases = make_bases(hidden, ks, rng, args.random_reps)
    prompts.to_csv(args.out_dir / "prompts.csv", index=False)
    np.save(args.out_dir / "hidden_mask_vectors.npy", hidden.astype(np.float32))
    scores, features = compute_layerwise_scores(tokenizer, model, prompts, hidden, ks, structured, random_bases, args)
    scores.to_csv(args.out_dir / "layerwise_k_scores.csv", index=False)
    features.to_csv(args.out_dir / "prompt_features_layerwise_k.csv", index=False)
    summary, trajectory = summarize_scores(scores)
    summary.to_csv(args.out_dir / "layerwise_k_summary.csv", index=False)
    trajectory.to_csv(args.out_dir / "layerwise_k_trajectory.csv", index=False)
    eval_summary, split_scores, leave_topic = evaluate_features(features, ks, args)
    eval_summary.to_csv(args.out_dir / "layerwise_k_feature_auc_summary.csv", index=False)
    split_scores.to_csv(args.out_dir / "layerwise_k_feature_split_scores.csv", index=False)
    leave_topic.to_csv(args.out_dir / "layerwise_k_feature_leave_topic.csv", index=False)
    deltas = paired_deltas(eval_summary, ks)
    deltas.to_csv(args.out_dir / "layerwise_k_feature_deltas.csv", index=False)
    write_report(args.out_dir, prompts, scores, summary, trajectory, eval_summary, deltas)
    report = (args.out_dir / "report.md").read_text(encoding="utf-8")
    print(report.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
