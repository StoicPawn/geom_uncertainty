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
from sklearn.metrics import roc_auc_score

from run_factual_error_deep_benchmark import feature_sets


MODEL_DIRS = {
    "bert-base-uncased": ROOT / "results" / "factual_error_deep_bert_top32",
    "distilbert-base-uncased": ROOT / "results" / "factual_error_deep_distilbert_top32",
    "google/bert_uncased_L-2_H-128_A-2": ROOT / "results" / "factual_error_deep_google_tiny_top32",
}

SCALAR_SCORE_COLS = [
    "entropy_topk",
    "varentropy_topk",
    "trace_fim",
    "grad_norm_sq",
    "top1_prob_topk_layer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "geometric_diagnostics_existing")
    parser.add_argument("--max-pairs-per-group", type=int, default=15)
    parser.add_argument("--max-pairs-total", type=int, default=300)
    return parser.parse_args()


def safe_auc(labels: pd.Series, values: pd.Series) -> float:
    valid = values.notna()
    labels = labels[valid]
    values = values[valid]
    y = labels.astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, values.to_numpy(dtype=float)))


def zscore(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = frame[cols].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (values - mean) / std


def load_model_outputs(model: str, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prompts = pd.read_csv(directory / "prompts.csv")
    scores = pd.read_csv(directory / "sample_scores.csv")
    features = pd.read_csv(directory / "prompt_features.csv")
    prompts = prompts.rename(columns={"observed_condition": "prompt_observed_condition"})
    scores = scores.merge(
        prompts[
            [
                "prompt_id",
                "prompt",
                "target",
                "target_rank_final",
                "final_top1_token",
                "final_top8_tokens",
                "prompt_observed_condition",
            ]
        ],
        on="prompt_id",
        how="left",
    )
    scores.insert(0, "model", model)
    features.insert(0, "model", model)
    prompts.insert(0, "model", model)
    return prompts, scores, features


def same_uncertainty_pairs(scores: pd.DataFrame, max_per_group: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    structured = scores[scores["subspace"].isin(["state_1d", "delta_1d"])].copy()
    for (model, layer, subspace), group in structured.groupby(["model", "layer", "subspace"]):
        group = group.reset_index(drop=True)
        if len(group) < 8:
            continue
        x = zscore(group, SCALAR_SCORE_COLS)
        diff = x[:, None, :] - x[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        rho = group["rho"].to_numpy(dtype=float)
        rho_diff = np.abs(rho[:, None] - rho[None, :])
        upper = np.triu(np.ones_like(distances, dtype=bool), k=1)
        scalar_values = distances[upper]
        if scalar_values.size == 0:
            continue
        distance_cutoff = float(np.quantile(scalar_values, 0.05))
        candidates = np.argwhere(upper & (distances <= distance_cutoff))
        if candidates.size == 0:
            continue
        order = np.argsort(-rho_diff[candidates[:, 0], candidates[:, 1]])
        for idx in order[:max_per_group]:
            i, j = candidates[idx]
            left = group.iloc[int(i)]
            right = group.iloc[int(j)]
            row = {
                "model": model,
                "layer": int(layer),
                "subspace": subspace,
                "scalar_distance_z": float(distances[i, j]),
                "rho_abs_diff": float(rho_diff[i, j]),
                "left_prompt_id": int(left["prompt_id"]),
                "right_prompt_id": int(right["prompt_id"]),
                "left_condition": left["observed_condition"],
                "right_condition": right["observed_condition"],
                "left_rho": float(left["rho"]),
                "right_rho": float(right["rho"]),
                "left_prompt": left["prompt"],
                "right_prompt": right["prompt"],
                "left_target": left["target"],
                "right_target": right["target"],
                "left_final_top1": left["final_top1_token"],
                "right_final_top1": right["final_top1_token"],
                "left_final_top8": left["final_top8_tokens"],
                "right_final_top8": right["final_top8_tokens"],
            }
            for col in SCALAR_SCORE_COLS:
                row[f"left_{col}"] = float(left[col])
                row[f"right_{col}"] = float(right[col])
                row[f"absdiff_{col}"] = float(abs(left[col] - right[col]))
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["rho_abs_diff", "scalar_distance_z"], ascending=[False, True]).reset_index(drop=True)


def rho_family_columns(features: pd.DataFrame) -> dict[str, list[str]]:
    fsets = feature_sets(features)
    families = {
        "rho_delta": fsets.get("rho_delta", []),
        "rho_state": fsets.get("rho_state", []),
        "rho_state_delta": fsets.get("rho_state_delta", []),
        "rho_random": fsets.get("rho_random", []),
    }
    return {key: [col for col in cols if col in features.columns] for key, cols in families.items()}


def matched_error_correct_pairs(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_pair_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    for model, frame in features.groupby("model"):
        fsets = feature_sets(frame)
        match_specs = {
            "scalar_final": fsets.get("scalar_final", []),
            "scalar_layer_profile": fsets.get("scalar_layer_profile", []),
        }
        rho_families = rho_family_columns(frame)
        errors = frame[frame["observed_condition"].eq("error")].reset_index(drop=True)
        correct = frame[frame["observed_condition"].eq("correct")].reset_index(drop=True)
        if errors.empty or correct.empty:
            continue
        for match_name, cols in match_specs.items():
            cols = [col for col in cols if col in frame.columns and not frame[col].isna().all()]
            if not cols:
                continue
            stats = frame[cols].agg(["mean", "std"]).T
            std = stats["std"].to_numpy(dtype=float)
            std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
            std[std < 1e-12] = 1.0
            mean = stats["mean"].to_numpy(dtype=float)
            mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            err_x = (errors[cols].to_numpy(dtype=float) - mean) / std
            cor_x = (correct[cols].to_numpy(dtype=float) - mean) / std
            err_x = np.nan_to_num(err_x, nan=0.0, posinf=0.0, neginf=0.0)
            cor_x = np.nan_to_num(cor_x, nan=0.0, posinf=0.0, neginf=0.0)
            distances = np.sqrt(((err_x[:, None, :] - cor_x[None, :, :]) ** 2).sum(axis=2))
            nearest = distances.argmin(axis=1)
            pair_rows: list[dict[str, object]] = []
            for err_idx, cor_idx in enumerate(nearest):
                err = errors.iloc[int(err_idx)]
                cor = correct.iloc[int(cor_idx)]
                row: dict[str, object] = {
                    "model": model,
                    "match_controls": match_name,
                    "scalar_distance_z": float(distances[err_idx, cor_idx]),
                    "error_prompt_id": int(err["prompt_id"]),
                    "correct_prompt_id": int(cor["prompt_id"]),
                    "error_topic": err["topic"],
                    "correct_topic": cor["topic"],
                    "error_fact_id": err["fact_id"],
                    "correct_fact_id": cor["fact_id"],
                }
                for family, rho_cols in rho_families.items():
                    if not rho_cols:
                        continue
                    err_value = float(err[rho_cols].mean())
                    cor_value = float(cor[rho_cols].mean())
                    row[f"error_{family}_mean"] = err_value
                    row[f"correct_{family}_mean"] = cor_value
                    row[f"delta_{family}_error_minus_correct"] = err_value - cor_value
                pair_rows.append(row)
            pair_df = pd.DataFrame(pair_rows)
            all_pair_rows.extend(pair_rows)
            summary: dict[str, object] = {
                "model": model,
                "match_controls": match_name,
                "n_error_prompts": int(len(errors)),
                "n_correct_pool": int(len(correct)),
                "scalar_distance_z_mean": float(pair_df["scalar_distance_z"].mean()),
                "scalar_distance_z_median": float(pair_df["scalar_distance_z"].median()),
                "scalar_distance_z_p90": float(pair_df["scalar_distance_z"].quantile(0.9)),
            }
            for family in rho_families:
                delta_col = f"delta_{family}_error_minus_correct"
                if delta_col not in pair_df.columns:
                    continue
                deltas = pair_df[delta_col]
                summary[f"{family}_delta_mean"] = float(deltas.mean())
                summary[f"{family}_delta_median"] = float(deltas.median())
                summary[f"{family}_win_rate_error_gt_correct"] = float((deltas > 0).mean())
            all_summary_rows.append(summary)
    return pd.DataFrame(all_pair_rows), pd.DataFrame(all_summary_rows)


def layer_trajectories(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores[scores["subspace"].isin(["state_1d", "delta_1d", "full"]) | scores["subspace"].str.startswith("random_2_")].copy()
    frame["subspace_family"] = np.where(frame["subspace"].str.startswith("random_2_"), "random_2", frame["subspace"])
    grouped = frame.groupby(["model", "observed_condition", "layer", "subspace_family"], dropna=False)
    return grouped.agg(
        n=("prompt_id", "nunique"),
        rho_mean=("rho", "mean"),
        rho_std=("rho", "std"),
        entropy_mean=("entropy_topk", "mean"),
        varentropy_mean=("varentropy_topk", "mean"),
        trace_fim_mean=("trace_fim", "mean"),
        grad_norm_sq_mean=("grad_norm_sq", "mean"),
        top1_prob_mean=("top1_prob_topk_layer", "mean"),
    ).reset_index()


def existing_saturation(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores.copy()
    frame["subspace_family"] = np.where(frame["subspace"].str.startswith("random_2_"), "random_2", frame["subspace"])
    grouped = frame.groupby(["model", "observed_condition", "layer", "subspace_family", "subspace_dim"], dropna=False)
    return grouped.agg(
        n=("prompt_id", "nunique"),
        rho_mean=("rho", "mean"),
        rho_std=("rho", "std"),
        rho_min=("rho", "min"),
        rho_max=("rho", "max"),
        trace_fim_mean=("trace_fim", "mean"),
    ).reset_index()


def failure_mode_flags(features: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, frame in features.groupby("model"):
        y = frame["observed_condition"].eq("error")
        fsets = feature_sets(frame)
        aucs: dict[str, float] = {}
        for name, cols in fsets.items():
            cols = [col for col in cols if col in frame.columns and not frame[col].isna().all()]
            if not cols:
                continue
            values = frame[cols].mean(axis=1, skipna=True)
            aucs[name] = safe_auc(y, values)
        model_scores = scores[scores["model"].eq(model)]
        full = model_scores[model_scores["subspace"].eq("full")]
        rows.append(
            {
                "model": model,
                "full_rho_mean": float(full["rho"].mean()),
                "full_rho_std": float(full["rho"].std()),
                "full_rho_share_gt_0_99": float((full["rho"] > 0.99).mean()),
                "rho_delta_mean_feature_auc": aucs.get("rho_delta", float("nan")),
                "rho_state_delta_mean_feature_auc": aucs.get("rho_state_delta", float("nan")),
                "rho_random_mean_feature_auc": aucs.get("rho_random", float("nan")),
                "scalar_layer_mean_feature_auc": aucs.get("scalar_layer_profile", float("nan")),
                "hidden_all_mean_feature_auc": aucs.get("hidden_all", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    pairs: pd.DataFrame,
    matched_summary: pd.DataFrame,
    trajectories: pd.DataFrame,
    saturation: pd.DataFrame,
    failure: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame, max_rows: int = 30) -> str:
        if df.empty:
            return "```text\n(empty)\n```"
        return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"

    top_pairs_cols = [
        "model",
        "layer",
        "subspace",
        "scalar_distance_z",
        "rho_abs_diff",
        "left_condition",
        "right_condition",
        "left_rho",
        "right_rho",
        "left_prompt",
        "right_prompt",
    ]
    pair_view = pairs[[col for col in top_pairs_cols if col in pairs.columns]].copy()
    sat_view = saturation[
        saturation["subspace_family"].isin(["state_1d", "delta_1d", "random_2", "full"])
    ].copy()
    sat_view = sat_view.groupby(["model", "subspace_family", "subspace_dim"], dropna=False).agg(
        rho_mean=("rho_mean", "mean"),
        rho_std_mean=("rho_std", "mean"),
        trace_fim_mean=("trace_fim_mean", "mean"),
    ).reset_index()
    traj_view = trajectories[trajectories["subspace_family"].isin(["delta_1d", "state_1d"])].copy()
    lines = [
        "# Geometric Diagnostics from Existing Outputs",
        "",
        "Source: existing factual-error deep CSV outputs only. No model inference is run by this script.",
        "",
        "## Same-Uncertainty / Different-Rho Candidate Pairs",
        "",
        "Pairs are matched within model, layer, and subspace by low z-scored distance over entropy, varentropy, trace-FIM, gradient norm, and top-1 probability, then sorted by rho difference.",
        table(pair_view, max_rows=25),
        "",
        "## Error/Correct Matching Controlled by Scalars",
        "",
        "Each error prompt is nearest-neighbor matched to a correct prompt using either final-layer scalar controls or all scalar layer-profile controls.",
        table(matched_summary, max_rows=30),
        "",
        "## Layer Trajectories",
        "",
        "Layer-wise means by observed condition and structured subspace.",
        table(traj_view, max_rows=40),
        "",
        "## Existing Saturation / Full-Space Check",
        "",
        "This uses only already-computed subspaces. A true k-sweep is done in the intervention/ablation script.",
        table(sat_view, max_rows=40),
        "",
        "## Failure-Mode Flags",
        "",
        "Feature AUCs here are simple mean-feature AUCs, not cross-validated classifiers. They are only quick flags.",
        table(failure, max_rows=20),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_tables: list[pd.DataFrame] = []
    score_tables: list[pd.DataFrame] = []
    feature_tables: list[pd.DataFrame] = []
    for model, directory in MODEL_DIRS.items():
        if not directory.exists():
            continue
        prompts, scores, features = load_model_outputs(model, directory)
        prompt_tables.append(prompts)
        score_tables.append(scores)
        feature_tables.append(features)
    prompts_all = pd.concat(prompt_tables, ignore_index=True)
    scores_all = pd.concat(score_tables, ignore_index=True)
    features_all = pd.concat(feature_tables, ignore_index=True)

    pairs = same_uncertainty_pairs(scores_all, args.max_pairs_per_group)
    if len(pairs) > args.max_pairs_total:
        pairs = pairs.head(args.max_pairs_total).copy()
    matched_pairs, matched_summary = matched_error_correct_pairs(features_all)
    trajectories = layer_trajectories(scores_all)
    saturation = existing_saturation(scores_all)
    failure = failure_mode_flags(features_all, scores_all)

    prompts_all.to_csv(args.out_dir / "source_prompts.csv", index=False)
    pairs.to_csv(args.out_dir / "same_uncertainty_different_rho_pairs.csv", index=False)
    matched_pairs.to_csv(args.out_dir / "matched_error_correct_pairs.csv", index=False)
    matched_summary.to_csv(args.out_dir / "matched_error_correct_summary.csv", index=False)
    trajectories.to_csv(args.out_dir / "layer_trajectories.csv", index=False)
    saturation.to_csv(args.out_dir / "existing_saturation.csv", index=False)
    failure.to_csv(args.out_dir / "failure_mode_flags.csv", index=False)
    write_report(args.out_dir, pairs, matched_summary, trajectories, saturation, failure)

    report_text = (args.out_dir / "report.md").read_text(encoding="utf-8")
    print(report_text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
