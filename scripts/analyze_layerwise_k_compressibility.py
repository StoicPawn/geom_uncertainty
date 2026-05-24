from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=ROOT / "results" / "layerwise_k_adjusted_rho")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "layerwise_k_compressibility")
    parser.add_argument("--max-pairs-per-group", type=int, default=20)
    return parser.parse_args()


def curve_metrics(frame: pd.DataFrame) -> dict[str, float]:
    frame = frame.sort_values("k")
    k = frame["k"].to_numpy(dtype=float)
    rho = frame["rho_structured"].to_numpy(dtype=float)
    adjusted = frame["rho_adjusted"].to_numpy(dtype=float)
    random = frame["rho_random_mean"].to_numpy(dtype=float)

    def first_k_at(values: np.ndarray, threshold: float) -> float:
        hit = np.flatnonzero(values >= threshold)
        if len(hit) == 0:
            return float("nan")
        return float(k[int(hit[0])])

    log_k = np.log2(k)
    denom = float(log_k[-1] - log_k[0]) if len(log_k) > 1 else 1.0
    return {
        "k_min": float(k.min()),
        "k_max": float(k.max()),
        "rho_k1": float(rho[0]),
        "rho_kmax": float(rho[-1]),
        "rho_random_k1": float(random[0]),
        "rho_random_kmax": float(random[-1]),
        "rho_adjusted_k1": float(adjusted[0]),
        "rho_adjusted_kmax": float(adjusted[-1]),
        "k50": first_k_at(rho, 0.50),
        "k80": first_k_at(rho, 0.80),
        "random_k50": first_k_at(random, 0.50),
        "random_k80": first_k_at(random, 0.80),
        "initial_slope_log2": float((rho[1] - rho[0]) / (log_k[1] - log_k[0])) if len(k) > 1 else float("nan"),
        "adjusted_initial_slope_log2": float((adjusted[1] - adjusted[0]) / (log_k[1] - log_k[0])) if len(k) > 1 else float("nan"),
        "auc_logk": float(np.trapezoid(rho, log_k) / denom) if len(k) > 1 else float(rho[0]),
        "random_auc_logk": float(np.trapezoid(random, log_k) / denom) if len(k) > 1 else float(random[0]),
        "adjusted_auc_logk": float(np.trapezoid(adjusted, log_k) / denom) if len(k) > 1 else float(adjusted[0]),
    }


def compute_compressibility(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    curve_source = scores[scores["subspace"].isin(["state_pca", "delta_pca"])].copy()
    for keys, group in curve_source.groupby(["prompt_id", "observed_condition", "topic", "fact_id", "subspace", "layer"]):
        prompt_id, condition, topic, fact_id, subspace, layer = keys
        metrics = curve_metrics(group)
        curve_rows.append(
            {
                "prompt_id": int(prompt_id),
                "observed_condition": condition,
                "topic": topic,
                "fact_id": fact_id,
                "subspace": subspace,
                "layer": int(layer),
                **metrics,
            }
        )
    curves = pd.DataFrame(curve_rows)
    summary = curves.groupby(["subspace", "layer", "observed_condition"], dropna=False).agg(
        n=("prompt_id", "nunique"),
        k50_mean=("k50", "mean"),
        k80_mean=("k80", "mean"),
        k80_missing_rate=("k80", lambda s: float(s.isna().mean())),
        rho_k1_mean=("rho_k1", "mean"),
        rho_kmax_mean=("rho_kmax", "mean"),
        adjusted_auc_logk_mean=("adjusted_auc_logk", "mean"),
        adjusted_initial_slope_mean=("adjusted_initial_slope_log2", "mean"),
    ).reset_index()
    return curves, summary


def merge_layer_controls(scores: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_lookup = features.set_index("prompt_id")
    for _, row in scores.iterrows():
        prompt_id = int(row["prompt_id"])
        layer = int(row["layer"])
        feature = feature_lookup.loc[prompt_id]
        enriched = row.to_dict()
        for base in ["top1_prob_topk_layer", "grad_norm_sq_full", "trace_fim_full"]:
            col = f"{base}_l{layer}"
            if col in feature.index:
                enriched[base] = float(feature[col])
        rows.append(enriched)
    return pd.DataFrame(rows)


def zscore_matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = frame[cols].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return np.nan_to_num((values - mean) / std, nan=0.0)


def same_uncertainty_pairs(scores: pd.DataFrame, prompts: pd.DataFrame, max_per_group: int) -> pd.DataFrame:
    candidates = scores[
        (
            scores["subspace"].eq("delta_forward_1d")
            | (scores["subspace"].isin(["state_pca", "delta_pca"]) & scores["k"].isin([1, 2, 4, 8, 16]))
        )
    ].copy()
    prompts_small = prompts[["prompt_id", "prompt", "target", "final_top1_token", "final_top8_tokens"]]
    candidates = candidates.merge(prompts_small, on="prompt_id", how="left")
    controls = ["entropy_topk", "varentropy_topk", "top1_prob_topk_layer", "trace_fim_full"]
    controls = [col for col in controls if col in candidates.columns]
    rows: list[dict[str, object]] = []
    for keys, group in candidates.groupby(["subspace", "k", "layer"]):
        subspace, k, layer = keys
        group = group.reset_index(drop=True)
        if len(group) < 12:
            continue
        x = zscore_matrix(group, controls)
        distances = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
        rho = group["rho_adjusted"].to_numpy(dtype=float)
        rho_diff = np.abs(rho[:, None] - rho[None, :])
        upper = np.triu(np.ones_like(distances, dtype=bool), k=1)
        cutoff = float(np.quantile(distances[upper], 0.03))
        pair_idx = np.argwhere(upper & (distances <= cutoff))
        if len(pair_idx) == 0:
            continue
        order = np.argsort(-rho_diff[pair_idx[:, 0], pair_idx[:, 1]])
        for pos in order[:max_per_group]:
            i, j = pair_idx[pos]
            left = group.iloc[int(i)]
            right = group.iloc[int(j)]
            row = {
                "subspace": subspace,
                "k": int(k),
                "layer": int(layer),
                "scalar_distance_z": float(distances[i, j]),
                "rho_adjusted_abs_diff": float(rho_diff[i, j]),
                "left_prompt_id": int(left["prompt_id"]),
                "right_prompt_id": int(right["prompt_id"]),
                "left_condition": left["observed_condition"],
                "right_condition": right["observed_condition"],
                "left_rho_adjusted": float(left["rho_adjusted"]),
                "right_rho_adjusted": float(right["rho_adjusted"]),
                "left_prompt": left["prompt"],
                "right_prompt": right["prompt"],
                "left_target": left["target"],
                "right_target": right["target"],
                "left_final_top1": left["final_top1_token"],
                "right_final_top1": right["final_top1_token"],
            }
            for col in controls:
                row[f"left_{col}"] = float(left[col])
                row[f"right_{col}"] = float(right[col])
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["rho_adjusted_abs_diff", "scalar_distance_z"], ascending=[False, True]).reset_index(drop=True)


def accessibility_categories(scores: pd.DataFrame, prompts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = scores[
        (scores["subspace"].eq("delta_forward_1d"))
        | ((scores["subspace"].eq("state_pca")) & scores["k"].isin([2, 4]))
        | ((scores["subspace"].eq("delta_pca")) & scores["k"].isin([2, 4]))
    ].copy()
    rows: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    prompts_small = prompts[["prompt_id", "prompt", "target", "final_top1_token", "final_top8_tokens"]]
    for keys, group in selected.groupby(["subspace", "k", "layer"]):
        subspace, k, layer = keys
        threshold = float(group["rho_adjusted"].median())
        local = group.copy()
        local["accessibility_bin"] = np.where(local["rho_adjusted"] >= threshold, "high_accessibility", "low_accessibility")
        local["outcome_accessibility"] = local["observed_condition"] + "_" + local["accessibility_bin"]
        counts = local.groupby("outcome_accessibility").size().reset_index(name="count")
        for _, count_row in counts.iterrows():
            rows.append(
                {
                    "subspace": subspace,
                    "k": int(k),
                    "layer": int(layer),
                    "median_threshold": threshold,
                    "outcome_accessibility": count_row["outcome_accessibility"],
                    "count": int(count_row["count"]),
                }
            )
        for category in ["error_low_accessibility", "error_high_accessibility", "correct_low_accessibility", "correct_high_accessibility"]:
            hits = local[local["outcome_accessibility"].eq(category)].copy()
            if hits.empty:
                continue
            if "low" in category:
                hits = hits.sort_values("rho_adjusted", ascending=True)
            else:
                hits = hits.sort_values("rho_adjusted", ascending=False)
            for _, example in hits.head(5).merge(prompts_small, on="prompt_id", how="left").iterrows():
                examples.append(
                    {
                        "subspace": subspace,
                        "k": int(k),
                        "layer": int(layer),
                        "category": category,
                        "prompt_id": int(example["prompt_id"]),
                        "rho_adjusted": float(example["rho_adjusted"]),
                        "rho_structured": float(example["rho_structured"]),
                        "rho_random_mean": float(example["rho_random_mean"]),
                        "prompt": example["prompt"],
                        "target": example["target"],
                        "final_top1_token": example["final_top1_token"],
                        "final_top8_tokens": example["final_top8_tokens"],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(examples)


def write_report(
    out_dir: Path,
    curve_summary: pd.DataFrame,
    pairs: pd.DataFrame,
    categories: pd.DataFrame,
    examples: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame, max_rows: int = 40, cols: list[str] | None = None) -> str:
        if df.empty:
            return "```text\n(empty)\n```"
        frame = df.copy()
        if cols is not None:
            frame = frame[[col for col in cols if col in frame.columns]]
        return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"

    useful_summary = curve_summary[curve_summary["subspace"].isin(["state_pca", "delta_pca"])].copy()
    pair_cols = [
        "subspace",
        "k",
        "layer",
        "scalar_distance_z",
        "rho_adjusted_abs_diff",
        "left_condition",
        "right_condition",
        "left_rho_adjusted",
        "right_rho_adjusted",
        "left_prompt",
        "right_prompt",
    ]
    example_cols = [
        "subspace",
        "k",
        "layer",
        "category",
        "rho_adjusted",
        "rho_structured",
        "rho_random_mean",
        "prompt",
        "target",
        "final_top1_token",
    ]
    lines = [
        "# Layerwise k Compressibility and Accessibility Diagnostics",
        "",
        "This report is computed from `results/layerwise_k_adjusted_rho` without rerunning model inference.",
        "",
        "## Compressibility Summary",
        table(
            useful_summary.sort_values(["subspace", "layer", "observed_condition"]),
            max_rows=120,
            cols=[
                "subspace",
                "layer",
                "observed_condition",
                "n",
                "k50_mean",
                "k80_mean",
                "k80_missing_rate",
                "rho_k1_mean",
                "rho_kmax_mean",
                "adjusted_auc_logk_mean",
                "adjusted_initial_slope_mean",
            ],
        ),
        "",
        "## Same-Uncertainty / Different-Accessibility Pairs",
        table(pairs, max_rows=40, cols=pair_cols),
        "",
        "## Outcome x Accessibility Counts",
        table(categories, max_rows=120),
        "",
        "## Category Examples",
        table(examples, max_rows=80, cols=example_cols),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.in_dir / "layerwise_k_scores.csv")
    features = pd.read_csv(args.in_dir / "prompt_features_layerwise_k.csv")
    prompts = pd.read_csv(args.in_dir / "prompts.csv")
    enriched = merge_layer_controls(scores, features)
    curves, curve_summary = compute_compressibility(scores)
    pairs = same_uncertainty_pairs(enriched, prompts, args.max_pairs_per_group)
    categories, examples = accessibility_categories(scores, prompts)
    enriched.to_csv(args.out_dir / "layerwise_k_scores_enriched.csv", index=False)
    curves.to_csv(args.out_dir / "compressibility_by_prompt_layer.csv", index=False)
    curve_summary.to_csv(args.out_dir / "compressibility_summary.csv", index=False)
    pairs.to_csv(args.out_dir / "same_uncertainty_different_accessibility_pairs.csv", index=False)
    categories.to_csv(args.out_dir / "accessibility_category_counts.csv", index=False)
    examples.to_csv(args.out_dir / "accessibility_category_examples.csv", index=False)
    write_report(args.out_dir, curve_summary, pairs, categories, examples)
    report = (args.out_dir / "report.md").read_text(encoding="utf-8")
    print(report.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
