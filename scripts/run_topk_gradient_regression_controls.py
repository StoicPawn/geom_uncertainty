from __future__ import annotations

import argparse
import math
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
from scipy import stats
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertTokenizer

from accessible_varentropy.metrics import softmax_np
from accessible_varentropy.mlm_heads import mlm_logits_from_hidden
from run_uncertainty_steering_full_battery import (
    build_cases,
    build_records,
    fisher_geometry,
    full_logits,
    intervention_metrics,
    least_squares,
    mlm_head_jacobian_fast,
    normalize,
    parse_layers,
    projection,
    token_cluster_labels,
    topk_distribution,
    z_control_orthogonal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="distilbert-base-uncased,bert-base-uncased,google/bert_uncased_L-2_H-128_A-2",
    )
    parser.add_argument("--top-k-values", default="16,32,64,128,256")
    parser.add_argument("--subspace-ks", default="8")
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--max-prompts-per-task", type=int, default=8)
    parser.add_argument("--random-subspaces", type=int, default=1)
    parser.add_argument("--output-eps", type=float, default=0.05)
    parser.add_argument("--semantic-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "experiments" / "controls",
    )
    return parser.parse_args()


def safe_tokenizer(model_name: str):
    if "roberta" not in model_name.lower():
        try:
            return BertTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def pca_basis(matrix: np.ndarray, max_k: int) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:max_k].T.copy().astype(np.float64)


def orthonormalize(matrix: np.ndarray, k: int) -> np.ndarray:
    q, _r = np.linalg.qr(matrix)
    return q[:, :k].astype(np.float64)


def make_subspaces(records: list[dict[str, object]], layer: int, ks: list[int], random_reps: int, seed: int):
    hidden = np.stack(
        [
            rec["hidden_states"][layer][0, int(rec["mask_index"]), :].detach().cpu().numpy()
            for rec in records
        ]
    )
    hidden_dim = hidden.shape[1]
    max_k = min(max(ks), hidden_dim, max(1, len(records) - 1))
    bases: list[tuple[str, int, int, np.ndarray]] = []
    state = pca_basis(hidden, max_k)
    bases.extend(("state_pca", k, 0, state[:, :k]) for k in ks if k <= max_k)
    if 0 < layer < len(records[0]["hidden_states"]) - 1:
        delta = np.stack(
            [
                (
                    rec["hidden_states"][layer + 1][0, int(rec["mask_index"]), :]
                    - rec["hidden_states"][layer][0, int(rec["mask_index"]), :]
                )
                .detach()
                .cpu()
                .numpy()
                for rec in records
            ]
        )
        delta_basis = pca_basis(delta, max_k)
        bases.extend(("delta_pca", k, 0, delta_basis[:, :k]) for k in ks if k <= max_k)
    rng = np.random.default_rng(seed + 1009 * layer)
    for rep in range(random_reps):
        random_basis = orthonormalize(rng.normal(size=(hidden_dim, max_k)), max_k)
        bases.extend(("random", k, rep, random_basis[:, :k]) for k in ks if k <= max_k)
    return bases


def grad_var_logits(probs: np.ndarray, u: np.ndarray, varentropy: float) -> np.ndarray:
    return probs * (u**2 - 2.0 * u - varentropy)


def jaccard_topn(logits_before: np.ndarray, logits_after: np.ndarray, n: int) -> float:
    before = set(np.argsort(logits_before)[-n:])
    after = set(np.argsort(logits_after)[-n:])
    return len(before & after) / max(len(before | after), 1)


def full_kl_on_candidate_set(before_probs: np.ndarray, after_probs: np.ndarray, token_ids: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(before_probs[token_ids], eps, 1.0)
    q = np.clip(after_probs[token_ids], eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def apply_hidden_step(model, hidden: torch.Tensor, hidden_direction: np.ndarray, step_scale: float, sign: float) -> np.ndarray:
    with torch.no_grad():
        direction = torch.as_tensor(sign * step_scale * hidden_direction, dtype=hidden.dtype, device=hidden.device)
        stepped = hidden + direction
        logits = mlm_logits_from_hidden(model, stepped.view(1, 1, -1))[0, 0]
    return logits.detach().float().cpu().numpy()


def run_model(args: argparse.Namespace, model_name: str, top_k_values: list[int], subspace_ks: list[int]):
    print(f"load_model {model_name}", flush=True)
    tokenizer = safe_tokenizer(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
    model.eval()
    cases = build_cases(args.seed, args.max_prompts_per_task)
    prompt_table, records = build_records(tokenizer, model, cases, max(top_k_values))
    if not records:
        return prompt_table.assign(model=model_name), pd.DataFrame(), pd.DataFrame()

    n_layers = len(records[0]["hidden_states"]) - 1
    layers = [layer for layer in parse_layers(args.layers, n_layers) if 1 <= layer <= n_layers]
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    prompt_lookup = prompt_table.set_index("prompt_id")
    score_rows: list[dict[str, object]] = []
    steering_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)

    for layer in layers:
        print(f"topk_layer {model_name} {layer}", flush=True)
        subspaces = make_subspaces(records, layer, subspace_ks, args.random_subspaces, args.seed)
        for rec_idx, record in enumerate(records):
            if rec_idx % 12 == 0:
                print(f"topk_prompt {model_name} layer={layer} {rec_idx + 1}/{len(records)}", flush=True)
            meta = prompt_lookup.loc[int(record["prompt_id"])]
            hidden = record["hidden_states"][layer][0, int(record["mask_index"]), :].detach()
            logits_before = full_logits(model, hidden)
            full_probs_before = softmax_np(logits_before[None, :])[0].astype(np.float64)
            for top_k in top_k_values:
                token_ids, probs = topk_distribution(logits_before, top_k)
                selected_logits = logits_before[token_ids]
                geometry = fisher_geometry(probs)
                jac = mlm_head_jacobian_fast(model, hidden, token_ids)
                labels = token_cluster_labels(embeddings, token_ids, args.semantic_clusters)
                target_id = None if pd.isna(meta["target_id"]) else int(meta["target_id"])
                sorted_selected = np.sort(selected_logits)[::-1]
                confidence = float(probs[0])
                margin = float(sorted_selected[0] - sorted_selected[1]) if len(sorted_selected) > 1 else float("nan")

                for family, subspace_k, rep, basis in subspaces:
                    jb = jac @ basis
                    whitened = geometry.sqrt_fisher @ jb
                    w_acc, rank = projection(whitened, geometry.w)
                    w_orth = geometry.w - w_acc
                    denom = max(float(geometry.w @ geometry.w), 1e-12)
                    rho = float((w_acc @ w_acc) / denom)
                    grad_entropy_z = whitened.T @ geometry.w
                    grad_var_z = jb.T @ grad_var_logits(probs, geometry.u, geometry.varentropy)
                    trace_fim = float(np.trace(jb.T @ geometry.fisher @ jb))
                    jacobian_fro_norm = float(np.linalg.norm(jb))
                    fisher_output_fro_norm = math.sqrt(max(trace_fim, 0.0))
                    score_common = {
                        "model": model_name,
                        "prompt_id": int(record["prompt_id"]),
                        "task": meta["task"],
                        "topic": meta["topic"],
                        "observed_condition": meta["observed_condition"],
                        "layer": layer,
                        "subspace_family": family,
                        "subspace_k": subspace_k,
                        "rep": rep,
                        "top_k_output": top_k,
                        "rank_whitened_jb": rank,
                        "rho": rho,
                        "v_access": float(w_acc @ w_acc),
                        "v_inaccess": float(w_orth @ w_orth),
                        "entropy": geometry.entropy,
                        "varentropy": geometry.varentropy,
                        "confidence": confidence,
                        "margin": margin,
                        "trace_fim": trace_fim,
                        "jacobian_fro_norm": jacobian_fro_norm,
                        "fisher_output_fro_norm": fisher_output_fro_norm,
                        "grad_entropy_proj_norm": float(np.linalg.norm(grad_entropy_z)),
                        "grad_varentropy_proj_norm": float(np.linalg.norm(grad_var_z)),
                        "logit_norm": float(np.linalg.norm(selected_logits)),
                    }
                    score_rows.append(score_common)

                    z_acc = normalize(least_squares(whitened, w_acc))
                    if float(np.linalg.norm(z_acc)) < 1e-12:
                        continue
                    z_grad_orth = z_control_orthogonal(rng, grad_entropy_z, basis.shape[1])
                    z_random = normalize(rng.normal(size=basis.shape[1]))
                    directions = {
                        "accessible_ls": z_acc,
                        "grad_orthogonal_control": z_grad_orth,
                        "random_control": z_random,
                    }
                    for direction, z_unit in directions.items():
                        if float(np.linalg.norm(z_unit)) < 1e-12:
                            continue
                        unit_output_energy = float(np.linalg.norm(whitened @ z_unit))
                        if unit_output_energy < 1e-12:
                            continue
                        first_order = float(grad_entropy_z @ z_unit)
                        sign_base = 1.0 if first_order >= 0.0 else -1.0
                        hidden_direction = basis @ z_unit
                        step_scale = args.output_eps / unit_output_energy
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
                            logits_after = apply_hidden_step(model, hidden, hidden_direction, step_scale, signed)
                            full_probs_after = softmax_np(logits_after[None, :])[0].astype(np.float64)
                            candidate_mass_before = float(full_probs_before[token_ids].sum())
                            candidate_mass_after = float(full_probs_after[token_ids].sum())
                            expected = 1.0 if sign_name == "increase" else -1.0
                            steering_rows.append(
                                {
                                    **score_common,
                                    "direction": direction,
                                    "sign": sign_name,
                                    "epsilon": args.output_eps,
                                    "latent_step_norm": step_scale,
                                    "fisher_output_energy": args.output_eps,
                                    "direction_output_energy_unit": unit_output_energy,
                                    "first_order_entropy": signed * step_scale * first_order,
                                    "directional_success": int(expected * metrics["delta_entropy"] > 0.0),
                                    "candidate_mass_before_full_vocab": candidate_mass_before,
                                    "candidate_mass_after_full_vocab": candidate_mass_after,
                                    "candidate_mass_retention_ratio": candidate_mass_after / max(candidate_mass_before, 1e-12),
                                    "full_vocab_top5_jaccard": jaccard_topn(logits_before, logits_after, 5),
                                    "full_vocab_top10_jaccard": jaccard_topn(logits_before, logits_after, 10),
                                    "candidate_set_kl": full_kl_on_candidate_set(full_probs_before, full_probs_after, token_ids),
                                    **metrics,
                                }
                            )
    return prompt_table.assign(model=model_name), pd.DataFrame(score_rows), pd.DataFrame(steering_rows)


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    if valid.sum() < 3:
        return float("nan")
    if a[valid].nunique() < 2 or b[valid].nunique() < 2:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def rank_stability(scores: pd.DataFrame, reference_k: int = 32) -> pd.DataFrame:
    keys = ["model", "prompt_id", "layer", "subspace_family", "subspace_k", "rep"]
    ref = scores[scores["top_k_output"].eq(reference_k)][keys + ["rho"]].rename(columns={"rho": "rho_ref"})
    rows = []
    for top_k, group in scores.groupby("top_k_output"):
        merged = group.merge(ref, on=keys, how="inner")
        rows.append(
            {
                "top_k_output": int(top_k),
                "reference_top_k": reference_k,
                "n": int(len(merged)),
                "rho_spearman_vs_ref": safe_spearman(merged["rho"], merged["rho_ref"]),
                "rho_abs_diff_mean_vs_ref": float((merged["rho"] - merged["rho_ref"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def layer_trend_stability(scores: pd.DataFrame, reference_k: int = 32) -> pd.DataFrame:
    trend = scores.groupby(["top_k_output", "model", "subspace_family", "subspace_k", "layer"], as_index=False)["rho"].mean()
    ref = trend[trend["top_k_output"].eq(reference_k)].rename(columns={"rho": "rho_ref"})
    ref = ref.drop(columns=["top_k_output"])
    rows = []
    for keys, group in trend.groupby(["top_k_output", "model", "subspace_family", "subspace_k"]):
        top_k, model, family, subspace_k = keys
        merged = group.merge(ref, on=["model", "subspace_family", "subspace_k", "layer"], how="inner")
        rows.append(
            {
                "top_k_output": int(top_k),
                "model": model,
                "subspace_family": family,
                "subspace_k": int(subspace_k),
                "n_layers": int(len(merged)),
                "layer_trend_spearman_vs_ref": safe_spearman(merged["rho"], merged["rho_ref"]),
                "layer_mean_abs_diff_vs_ref": float((merged["rho"] - merged["rho_ref"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def steering_summary(steering: pd.DataFrame) -> pd.DataFrame:
    return steering.groupby(["top_k_output", "direction", "sign"], as_index=False).agg(
        n=("prompt_id", "count"),
        abs_delta_entropy_mean=("abs_delta_entropy", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
        directional_success_rate=("directional_success", "mean"),
        selected_top1_changed_rate=("selected_top1_changed", "mean"),
        selected_top5_jaccard_mean=("top5_jaccard", "mean"),
        full_vocab_top5_jaccard_mean=("full_vocab_top5_jaccard", "mean"),
        full_vocab_top10_jaccard_mean=("full_vocab_top10_jaccard", "mean"),
        selected_kl_mean=("selected_kl_before_after", "mean"),
        candidate_set_kl_mean=("candidate_set_kl", "mean"),
        selected_prob_l1_mean=("selected_prob_l1", "mean"),
        cluster_distribution_l1_mean=("cluster_distribution_l1", "mean"),
        embedding_centroid_delta_norm_mean=("embedding_centroid_delta_norm", "mean"),
        candidate_mass_retention_ratio_mean=("candidate_mass_retention_ratio", "mean"),
    )


def topk_summary(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.groupby(["top_k_output", "model", "subspace_family", "subspace_k"], as_index=False).agg(
        n=("prompt_id", "count"),
        rho_mean=("rho", "mean"),
        rho_std=("rho", "std"),
        entropy_mean=("entropy", "mean"),
        varentropy_mean=("varentropy", "mean"),
        grad_entropy_proj_norm_mean=("grad_entropy_proj_norm", "mean"),
        grad_varentropy_proj_norm_mean=("grad_varentropy_proj_norm", "mean"),
        fisher_output_fro_norm_mean=("fisher_output_fro_norm", "mean"),
    )


def baseline_correlations(steering: pd.DataFrame) -> pd.DataFrame:
    frames = {
        "accessible_only": steering[steering["direction"].eq("accessible_ls")].copy(),
        "all_directions": steering.copy(),
    }
    predictors = {
        "rho": "rho",
        "grad_entropy_projection": "grad_entropy_proj_norm",
        "grad_varentropy_projection": "grad_varentropy_proj_norm",
        "fisher_output_norm": "fisher_output_fro_norm",
        "jacobian_fro_norm": "jacobian_fro_norm",
        "trace_fim": "trace_fim",
    }
    outcomes = {
        "abs_delta_entropy": "abs_delta_entropy",
        "abs_delta_varentropy": "abs_delta_varentropy",
        "directional_success": "directional_success",
    }
    rows = []
    for subset, frame in frames.items():
        for outcome_name, outcome_col in outcomes.items():
            for pred_name, pred_col in predictors.items():
                rows.append(
                    {
                        "subset": subset,
                        "outcome": outcome_name,
                        "predictor": pred_name,
                        "n": int(frame[[outcome_col, pred_col]].dropna().shape[0]),
                        "spearman": safe_spearman(frame[outcome_col], frame[pred_col]),
                        "pearson": float(frame[[outcome_col, pred_col]].corr(method="pearson").iloc[0, 1]),
                    }
                )
    return pd.DataFrame(rows)


def ols_with_controls(frame: pd.DataFrame, outcome: str, base_predictor: str = "rho") -> dict[str, object]:
    controls = [
        "entropy",
        "varentropy",
        "confidence",
        "margin",
        "jacobian_fro_norm",
        "fisher_output_energy",
        "grad_entropy_proj_norm",
        "grad_varentropy_proj_norm",
    ]
    categorical = ["top_k_output", "layer", "model", "task", "topic", "subspace_family", "subspace_k", "direction", "sign"]
    cols = [outcome, base_predictor] + controls + categorical
    data = frame[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(data) < 20:
        return {"outcome": outcome, "n": int(len(data)), "rho_beta": float("nan"), "rho_p_value": float("nan")}

    x_numeric = data[[base_predictor] + controls].to_numpy(dtype=float)
    x_mean = x_numeric.mean(axis=0, keepdims=True)
    x_std = x_numeric.std(axis=0, keepdims=True)
    x_std[x_std < 1e-12] = 1.0
    x_numeric = (x_numeric - x_mean) / x_std
    dummies = pd.get_dummies(data[categorical].astype(str), drop_first=True, dtype=float)
    x = np.column_stack([np.ones(len(data)), x_numeric, dummies.to_numpy(dtype=float)])
    y = data[outcome].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = max(x.shape[0] - x.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0.0, None))
    rho_idx = 1
    t_stat = float(beta[rho_idx] / max(se[rho_idx], 1e-12))
    p_value = float(2.0 * stats.t.sf(abs(t_stat), dof))
    ss_tot = float(((y - y.mean()) @ (y - y.mean())))
    r2 = 1.0 - float(resid @ resid) / max(ss_tot, 1e-12)
    return {
        "outcome": outcome,
        "n": int(len(data)),
        "n_parameters": int(x.shape[1]),
        "rho_beta_standardized": float(beta[rho_idx]),
        "rho_t_stat": t_stat,
        "rho_p_value": p_value,
        "r2": r2,
        "controls": ",".join(controls + categorical),
    }


def full_regression(steering: pd.DataFrame) -> pd.DataFrame:
    frame = steering.copy()
    rows = [ols_with_controls(frame, outcome) for outcome in ["abs_delta_entropy", "abs_delta_varentropy", "directional_success"]]
    return pd.DataFrame(rows)


def code_table(frame: pd.DataFrame) -> list[str]:
    return ["```text", frame.to_string(index=False), "```"]


def write_topk_report(
    out_dir: Path,
    scores: pd.DataFrame,
    steering: pd.DataFrame,
    ranks: pd.DataFrame,
    layers: pd.DataFrame,
    topk_summary_frame: pd.DataFrame,
    steering_summary_frame: pd.DataFrame,
) -> None:
    lines = [
        "# Control: Top-k Robustness",
        "",
        "This control repeats the geometry and steering summaries for top-k output lenses `16, 32, 64, 128, 256`.",
        "",
        "## Coverage",
        "",
        f"- Score rows: `{len(scores)}`",
        f"- Steering rows: `{len(steering)}`",
        f"- Models: `{', '.join(sorted(scores['model'].dropna().unique()))}`",
        f"- Top-k values: `{', '.join(str(int(v)) for v in sorted(scores['top_k_output'].dropna().unique()))}`",
        "",
        "## Rank Stability",
        "",
        *code_table(ranks),
        "",
        "## Layer-Trend Stability",
        "",
        *code_table(layers),
        "",
        "## Rho Summary",
        "",
        *code_table(topk_summary_frame.head(30)),
        "",
        "## Steering Summary",
        "",
        *code_table(steering_summary_frame.head(30)),
        "",
        "## Files",
        "",
        "```text",
        "prompt_tables.csv",
        "topk_scores.csv",
        "topk_steering_records.csv",
        "topk_robustness_summary.csv",
        "topk_rank_stability.csv",
        "topk_layer_trend_stability.csv",
        "topk_steering_summary.csv",
        "skipped_models.csv",
        "report.md",
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_gradient_report(out_dir: Path, baseline_frame: pd.DataFrame, scores: pd.DataFrame) -> None:
    lines = [
        "# Control: Direct Gradient Baselines",
        "",
        "This control compares `rho(B)` against direct local gradient and Jacobian baselines.",
        "",
        "## Predictors",
        "",
        "- `||Pi_B grad_z H||`",
        "- `||Pi_B grad_z Var||`",
        "- `||F^{1/2} J B||`",
        "- `||J B||`",
        "",
        "## Correlations",
        "",
        *code_table(baseline_frame),
        "",
        "## Files",
        "",
        "```text",
        "gradient_baseline_scores.csv",
        "gradient_baseline_correlations.csv",
        "report.md",
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_regression_report(out_dir: Path, regressions: pd.DataFrame) -> None:
    lines = [
        "# Control: Full Regression",
        "",
        "This control tests whether `rho` remains informative after scalar, gradient, geometric, model, layer, and prompt controls.",
        "",
        "## Regression Results",
        "",
        *code_table(regressions),
        "",
        "## Files",
        "",
        "```text",
        "full_regression_ols.csv",
        "report.md",
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_preservation_report(out_dir: Path, steering_summary_frame: pd.DataFrame) -> None:
    lines = [
        "# Control: Semantic And Top-k Preservation",
        "",
        "This control extends answer-preservation checks beyond top-1 identity.",
        "",
        "## Metrics",
        "",
        "- selected top-1 changed rate",
        "- selected top-5 Jaccard",
        "- full-vocabulary top-5 and top-10 Jaccard",
        "- KL divergence on the original candidate set",
        "- probability mass retained on the original candidate set",
        "- semantic-cluster L1 shift",
        "- embedding-centroid shift",
        "",
        "## Summary",
        "",
        *code_table(steering_summary_frame),
        "",
        "## Files",
        "",
        "```text",
        "semantic_topk_preservation.csv",
        "semantic_topk_preservation_summary.csv",
        "report.md",
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    top_k_values = [int(part) for part in args.top_k_values.split(",") if part.strip()]
    subspace_ks = [int(part) for part in args.subspace_ks.split(",") if part.strip()]
    models = [part.strip() for part in args.models.split(",") if part.strip()]

    topk_dir = args.out_root / "topk_robustness"
    gradient_dir = args.out_root / "gradient_baselines"
    regression_dir = args.out_root / "full_regression"
    preservation_dir = args.out_root / "semantic_preservation"
    for path in [topk_dir, gradient_dir, regression_dir, preservation_dir]:
        (path / "outputs").mkdir(parents=True, exist_ok=True)
        (path / "reports").mkdir(parents=True, exist_ok=True)

    if args.analysis_only:
        prompt_table = pd.read_csv(topk_dir / "outputs" / "prompt_tables.csv")
        score_table = pd.read_csv(topk_dir / "outputs" / "topk_scores.csv")
        steering_table = pd.read_csv(topk_dir / "outputs" / "topk_steering_records.csv")
        skipped_path = topk_dir / "outputs" / "skipped_models.csv"
        try:
            skipped_table = pd.read_csv(skipped_path) if skipped_path.exists() and skipped_path.stat().st_size > 0 else pd.DataFrame()
        except pd.errors.EmptyDataError:
            skipped_table = pd.DataFrame()
    else:
        prompts = []
        scores = []
        steering = []
        skipped = []
        for model_name in models:
            try:
                prompt_table, score_frame, steering_frame = run_model(args, model_name, top_k_values, subspace_ks)
                prompts.append(prompt_table)
                scores.append(score_frame)
                steering.append(steering_frame)
            except Exception as exc:
                skipped.append({"model": model_name, "error": repr(exc)})
                print(f"skipped_model {model_name}: {exc}", flush=True)

        prompt_table = pd.concat(prompts, ignore_index=True) if prompts else pd.DataFrame()
        score_table = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
        steering_table = pd.concat(steering, ignore_index=True) if steering else pd.DataFrame()
        skipped_table = pd.DataFrame(skipped)

    if score_table.empty or steering_table.empty:
        skipped_table.to_csv(topk_dir / "outputs" / "skipped_models.csv", index=False)
        raise SystemExit("No top-k robustness rows were produced.")

    rank_table = rank_stability(score_table, reference_k=32)
    layer_table = layer_trend_stability(score_table, reference_k=32)
    topk_summary_table = topk_summary(score_table)
    steering_summary_table = steering_summary(steering_table)
    baseline_table = baseline_correlations(steering_table)
    regression_table = full_regression(steering_table)

    prompt_table.to_csv(topk_dir / "outputs" / "prompt_tables.csv", index=False)
    score_table.to_csv(topk_dir / "outputs" / "topk_scores.csv", index=False)
    steering_table.to_csv(topk_dir / "outputs" / "topk_steering_records.csv", index=False)
    topk_summary_table.to_csv(topk_dir / "outputs" / "topk_robustness_summary.csv", index=False)
    rank_table.to_csv(topk_dir / "outputs" / "topk_rank_stability.csv", index=False)
    layer_table.to_csv(topk_dir / "outputs" / "topk_layer_trend_stability.csv", index=False)
    steering_summary_table.to_csv(topk_dir / "outputs" / "topk_steering_summary.csv", index=False)
    skipped_table.to_csv(topk_dir / "outputs" / "skipped_models.csv", index=False)

    baseline_table.to_csv(gradient_dir / "outputs" / "gradient_baseline_correlations.csv", index=False)
    score_table[
        [
            "model",
            "prompt_id",
            "task",
            "topic",
            "layer",
            "subspace_family",
            "subspace_k",
            "top_k_output",
            "rho",
            "grad_entropy_proj_norm",
            "grad_varentropy_proj_norm",
            "fisher_output_fro_norm",
            "jacobian_fro_norm",
        ]
    ].to_csv(gradient_dir / "outputs" / "gradient_baseline_scores.csv", index=False)

    regression_table.to_csv(regression_dir / "outputs" / "full_regression_ols.csv", index=False)
    steering_table[
        [
            "model",
            "prompt_id",
            "task",
            "topic",
            "layer",
            "subspace_family",
            "subspace_k",
            "top_k_output",
            "direction",
            "sign",
            "epsilon",
            "rho",
            "abs_delta_entropy",
            "abs_delta_varentropy",
            "directional_success",
            "selected_top1_changed",
            "top5_jaccard",
            "full_vocab_top5_jaccard",
            "full_vocab_top10_jaccard",
            "selected_kl_before_after",
            "candidate_set_kl",
            "candidate_mass_retention_ratio",
            "cluster_distribution_l1",
            "embedding_centroid_delta_norm",
        ]
    ].to_csv(preservation_dir / "outputs" / "semantic_topk_preservation.csv", index=False)
    steering_summary_table.to_csv(preservation_dir / "outputs" / "semantic_topk_preservation_summary.csv", index=False)

    write_topk_report(
        topk_dir / "reports",
        score_table,
        steering_table,
        rank_table,
        layer_table,
        topk_summary_table,
        steering_summary_table,
    )
    write_gradient_report(gradient_dir / "reports", baseline_table, score_table)
    write_regression_report(regression_dir / "reports", regression_table)
    write_preservation_report(preservation_dir / "reports", steering_summary_table)


if __name__ == "__main__":
    main()
