from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "controls" / "rho_guided_selective_reliability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--min-groups", type=int, default=5)
    parser.add_argument("--coverages", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(log_loss(y, np.clip(p, 1e-6, 1.0 - 1e-6), labels=[0, 1]))


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def aggregate_prefixed(
    frame: pd.DataFrame,
    keys: list[str],
    columns: list[str],
    prefix: str,
    funcs: tuple[str, ...] = ("mean", "max", "min"),
) -> pd.DataFrame:
    available = [col for col in columns if col in frame.columns]
    if not available:
        return frame[keys].drop_duplicates().reset_index(drop=True)
    agg = frame.groupby(keys, dropna=False)[available].agg(list(funcs))
    agg.columns = [f"{prefix}{name}_{func}" for name, func in agg.columns]
    return agg.reset_index()


def build_masked_dataset() -> pd.DataFrame:
    scores = read_csv(ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "topk_scores.csv")
    prompts = read_csv(ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "prompt_tables.csv")
    if scores.empty or prompts.empty:
        return pd.DataFrame()
    scores = scores[scores["top_k_output"].eq(32)].copy()
    prompts = prompts[~prompts["task"].astype(str).str.startswith("ambiguous")].copy()
    prompts = prompts[prompts["observed_condition"].isin(["correct", "error"])].copy()
    prompts["label_correct"] = prompts["observed_condition"].eq("correct").astype(int)
    keys = ["model", "prompt_id"]
    base_cols = [
        "entropy",
        "varentropy",
        "confidence",
        "margin",
        "trace_fim",
        "jacobian_fro_norm",
        "fisher_output_fro_norm",
        "grad_entropy_proj_norm",
        "grad_varentropy_proj_norm",
        "logit_norm",
        "rank_whitened_jb",
    ]
    rho_cols = ["rho"]
    base = aggregate_prefixed(scores, keys, base_cols, "base_")
    rho = aggregate_prefixed(scores, keys, rho_cols, "rho_")
    family_rho = (
        scores.pivot_table(index=keys, columns="subspace_family", values="rho", aggfunc="mean")
        .add_prefix("rho_family_")
        .reset_index()
    )
    merged = prompts.merge(base, on=keys, how="inner").merge(rho, on=keys, how="inner").merge(family_rho, on=keys, how="left")
    merged["source"] = "masked_topk"
    merged["group_id"] = "masked::" + merged["prompt"].astype(str)
    return merged


def build_masked_full_battery_dataset() -> pd.DataFrame:
    scores = read_csv(ROOT / "experiments" / "04_uncertainty_steering" / "outputs" / "subspace_scores.csv")
    if scores.empty:
        return pd.DataFrame()
    scores = scores[~scores["task"].astype(str).str.startswith("ambiguous")].copy()
    scores = scores[scores["observed_condition"].isin(["correct", "error"])].copy()
    scores["label_correct"] = scores["observed_condition"].eq("correct").astype(int)
    keys = ["model", "prompt_id"]
    meta_cols = ["model", "prompt_id", "task", "topic", "observed_condition", "label_correct"]
    meta = scores[meta_cols].drop_duplicates(keys)
    base_cols = [
        "entropy_before",
        "varentropy_before",
        "semantic_entropy_proxy_before",
        "trace_fim",
        "grad_norm_sq",
        "logit_norm",
        "strict_orth_ls_norm",
        "accessible_ls_norm",
        "rank_whitened_jb",
    ]
    rho_cols = ["rho"]
    base = aggregate_prefixed(scores, keys, base_cols, "base_")
    rho = aggregate_prefixed(scores, keys, rho_cols, "rho_")
    family_rho = (
        scores.pivot_table(index=keys, columns="subspace_family", values="rho", aggfunc="mean")
        .add_prefix("rho_family_")
        .reset_index()
    )
    merged = meta.merge(base, on=keys, how="inner").merge(rho, on=keys, how="inner").merge(family_rho, on=keys, how="left")
    merged["source"] = "masked_full_battery"
    merged["group_id"] = "masked_full::" + merged["model"].astype(str) + "::" + merged["prompt_id"].astype(str)
    return merged


def build_decoder_dataset() -> pd.DataFrame:
    scores = read_csv(ROOT / "experiments" / "04_uncertainty_steering" / "decoder_main_battery" / "outputs" / "decoder_scores.csv")
    prompts = read_csv(ROOT / "experiments" / "04_uncertainty_steering" / "decoder_main_battery" / "outputs" / "prompt_table.csv")
    if scores.empty or prompts.empty:
        return pd.DataFrame()
    prompts = prompts[prompts["target"].astype(str).str.len().gt(0)].copy()
    prompts = prompts[prompts["top1_correct"].notna()].copy()
    prompts["label_correct"] = prompts["top1_correct"].astype(int)
    keys = ["model", "prompt_id"]
    base_cols = [
        "entropy",
        "varentropy",
        "confidence",
        "rank_whitened_jb",
        "grad_entropy_proj_norm",
        "fisher_output_fro_norm",
        "semantic_entropy_proxy",
        "semantic_density_uncertainty",
        "haloscope_style_consistency_risk",
    ]
    rho_cols = ["rho"]
    base = aggregate_prefixed(scores, keys, base_cols, "base_")
    rho = aggregate_prefixed(scores, keys, rho_cols, "rho_")
    family_rho = (
        scores.pivot_table(index=keys, columns="subspace_family", values="rho", aggfunc="mean")
        .add_prefix("rho_family_")
        .reset_index()
    )
    meta_cols = [
        "model",
        "prompt_id",
        "prompt",
        "task",
        "topic",
        "target",
        "label_correct",
        "confidence",
        "entropy",
        "varentropy",
        "target_in_topm",
        "target_rank_lens",
    ]
    merged = prompts[meta_cols].drop_duplicates(keys).merge(base, on=keys, how="inner").merge(rho, on=keys, how="inner").merge(family_rho, on=keys, how="left")
    merged["source"] = "decoder_battery"
    merged["group_id"] = "decoder::" + merged["prompt"].astype(str)
    return merged


def build_dataset() -> pd.DataFrame:
    frames = [build_masked_dataset(), build_masked_full_battery_dataset(), build_decoder_dataset()]
    data = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if data.empty:
        return data
    data["row_id"] = np.arange(len(data))
    data["label_error"] = 1 - data["label_correct"].astype(int)
    return data


def feature_sets(data: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = {
        "row_id",
        "label_correct",
        "label_error",
        "observed_condition",
        "final_top1_token",
        "target",
        "target_id",
        "top1_correct",
        "target_in_topm",
        "target_rank_final",
        "target_rank_lens",
        "prompt",
        "prompt_id",
        "group_id",
    }
    numeric = [
        col
        for col in data.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(data[col]) and not col.startswith("rho_")
    ]
    rho_numeric = [col for col in data.columns if col.startswith("rho_") and pd.api.types.is_numeric_dtype(data[col])]
    categorical = [col for col in ["source", "model", "task", "topic"] if col in data.columns]
    return numeric, rho_numeric, categorical


def make_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
                        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )


def oof_predictions(data: pd.DataFrame, include_rho: bool, min_groups: int) -> np.ndarray:
    base_numeric, rho_numeric, categorical = feature_sets(data)
    numeric = base_numeric + (rho_numeric if include_rho else [])
    features = numeric + categorical
    y = data["label_correct"].to_numpy(dtype=int)
    groups = data["group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < min_groups:
        raise RuntimeError(f"Need at least {min_groups} prompt groups, got {len(unique_groups)}.")
    preds = np.zeros(len(data), dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(data, y, groups=groups):
        train_y = y[train_idx]
        if len(np.unique(train_y)) < 2:
            preds[test_idx] = float(train_y.mean())
            continue
        pipe = make_pipeline(numeric, categorical)
        pipe.fit(data.iloc[train_idx][features], train_y)
        preds[test_idx] = pipe.predict_proba(data.iloc[test_idx][features])[:, 1]
    return np.clip(preds, 1e-6, 1.0 - 1e-6)


def risk_at_coverage(y: np.ndarray, p: np.ndarray, coverage: float) -> dict[str, float]:
    n = len(y)
    k = max(1, min(n, int(math.ceil(float(coverage) * n))))
    order = np.argsort(-p)
    accepted = order[:k]
    deferred = order[k:]
    risk = float(1.0 - np.mean(y[accepted]))
    deferred_error_capture = float((1 - y[deferred]).sum() / max((1 - y).sum(), 1)) if len(deferred) else 0.0
    return {
        "coverage": float(k / n),
        "requested_coverage": float(coverage),
        "accepted_n": int(k),
        "risk": risk,
        "hallucination_rate_proxy": risk,
        "deferred_error_capture": deferred_error_capture,
    }


def risk_curve(data: pd.DataFrame, pred_col: str, coverages: list[float]) -> pd.DataFrame:
    y = data["label_correct"].to_numpy(dtype=int)
    p = data[pred_col].to_numpy(dtype=float)
    rows = [risk_at_coverage(y, p, c) for c in coverages]
    out = pd.DataFrame(rows)
    out["policy"] = pred_col
    return out


def policy_metrics(data: pd.DataFrame, pred_col: str, coverages: list[float], high_conf: float) -> dict[str, float | str | int]:
    y = data["label_correct"].to_numpy(dtype=int)
    p = data[pred_col].to_numpy(dtype=float)
    curve = risk_curve(data, pred_col, coverages)
    high = p >= high_conf
    high_error = float(1.0 - np.mean(y[high])) if np.any(high) else float("nan")
    top20 = risk_at_coverage(y, p, 0.2)["risk"]
    return {
        "policy": pred_col,
        "n": int(len(data)),
        "accuracy": float(np.mean(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": safe_log_loss(y, p),
        "auroc": safe_auc(y, p),
        "average_precision": safe_ap(y, p),
        "aurc_mean_risk": float(curve["risk"].mean()),
        "risk_at_50pct_coverage": float(risk_at_coverage(y, p, 0.5)["risk"]),
        "risk_at_80pct_coverage": float(risk_at_coverage(y, p, 0.8)["risk"]),
        "high_pred_reliable_error_rate": high_error,
        "high_pred_reliable_coverage": float(np.mean(high)),
        "top20pct_error_rate": float(top20),
    }


def action_table(data: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    work = data.copy()
    rho_cols = [col for col in work.columns if col.startswith("rho_") and pd.api.types.is_numeric_dtype(work[col])]
    work["_rho_action"] = work[rho_cols].max(axis=1) if rho_cols else 0.0
    high_reliable = work[pred_col] >= work[pred_col].quantile(0.7)
    low_reliable = work[pred_col] < work[pred_col].quantile(0.3)
    high_rho = work["_rho_action"] >= work["_rho_action"].quantile(0.66)
    mid_reliable = ~(high_reliable | low_reliable)
    work["action"] = "pass_to_other_model"
    work.loc[high_reliable, "action"] = "accept"
    work.loc[~high_reliable & high_rho, "action"] = "intervene_or_retrieve"
    work.loc[low_reliable & ~high_rho, "action"] = "abstain_or_route"
    work.loc[mid_reliable & ~high_rho, "action"] = "pass_to_other_model"
    return (
        work.groupby("action", as_index=False)
        .agg(n=("row_id", "size"), coverage=("row_id", lambda s: len(s) / len(work)), error_rate=("label_error", "mean"), rho_action_mean=("_rho_action", "mean"), predicted_correct_mean=(pred_col, "mean"))
        .sort_values("action")
    )


def paired_deltas(data: pd.DataFrame, coverages: list[float], high_conf: float) -> pd.DataFrame:
    base = policy_metrics(data, "pred_baseline", coverages, high_conf)
    rho = policy_metrics(data, "pred_baseline_plus_rho", coverages, high_conf)
    rows = []
    improvements = {
        "delta_brier_improvement": base["brier"] - rho["brier"],
        "delta_log_loss_improvement": base["log_loss"] - rho["log_loss"],
        "delta_aurc_improvement": base["aurc_mean_risk"] - rho["aurc_mean_risk"],
        "delta_risk50_improvement": base["risk_at_50pct_coverage"] - rho["risk_at_50pct_coverage"],
        "delta_risk80_improvement": base["risk_at_80pct_coverage"] - rho["risk_at_80pct_coverage"],
        "delta_high_conf_error_improvement": base["high_pred_reliable_error_rate"] - rho["high_pred_reliable_error_rate"],
    }
    for metric, value in improvements.items():
        rows.append({"metric": metric, "value": float(value)})
    return pd.DataFrame(rows)


def bootstrap_ci(data: pd.DataFrame, coverages: list[float], high_conf: float, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grouped = {group: frame for group, frame in data.groupby("group_id", sort=False)}
    groups = np.array(list(grouped), dtype=object)
    rows = []
    for _ in range(n_boot):
        sample = pd.concat([grouped[group] for group in rng.choice(groups, size=len(groups), replace=True)], ignore_index=True)
        rows.append(paired_deltas(sample, coverages, high_conf))
    boot = pd.concat(rows, ignore_index=True)
    return (
        boot.groupby("metric")["value"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_low", 0.5: "median", 0.975: "ci_high"})
    )


def code_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def write_report(
    out_dir: Path,
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    ci: pd.DataFrame,
    curve: pd.DataFrame,
    action: pd.DataFrame,
) -> None:
    coverage = data.groupby(["source", "model"], as_index=False).agg(n=("row_id", "size"), correct_rate=("label_correct", "mean"))
    risk_wide = curve.pivot_table(index="requested_coverage", columns="policy", values="risk").reset_index()
    if {"pred_baseline", "pred_baseline_plus_rho"}.issubset(risk_wide.columns):
        risk_wide["risk_improvement"] = risk_wide["pred_baseline"] - risk_wide["pred_baseline_plus_rho"]
    verdict = "Inconclusive"
    aurc = ci[ci["metric"].eq("delta_aurc_improvement")]
    logloss = ci[ci["metric"].eq("delta_log_loss_improvement")]
    risk50 = ci[ci["metric"].eq("delta_risk50_improvement")]
    positives = []
    for name, frame in [("AURC", aurc), ("log-loss", logloss), ("risk@50", risk50)]:
        if not frame.empty and float(frame["ci_low"].iloc[0]) > 0.0:
            positives.append(name)
    if positives:
        verdict = "Supported on " + ", ".join(positives)
    lines = [
        "# Rho-Guided Selective Reliability",
        "",
        "This disruptive test asks whether `rho` improves a non-oracle selective reliability policy: accept when predicted reliable, otherwise abstain/route or intervene/retrieve according to uncertainty and rho-actionability.",
        "",
        "Correctness is never used as a decision feature. It is used only for held-out evaluation and cross-validated training labels. The split is grouped by prompt identity/text so the policy is evaluated out of sample on prompt groups, not memorized rows.",
        "",
        "## Verdict",
        verdict,
        "",
        "## Dataset Coverage",
        code_table(coverage, 80),
        "",
        "## Metrics",
        code_table(metrics, 20),
        "",
        "## Baseline Vs Baseline+rho Deltas",
        code_table(deltas, 20),
        "",
        "## Bootstrap CI",
        code_table(ci, 40),
        "",
        "## Risk-Coverage",
        code_table(risk_wide, 40),
        "",
        "## Rho-Guided Action Mix",
        code_table(action, 20),
        "",
        "## Files",
        "```text",
        "selective_reliability_dataset.csv",
        "selective_reliability_predictions.csv",
        "selective_reliability_metrics.csv",
        "selective_reliability_deltas.csv",
        "selective_reliability_bootstrap_ci.csv",
        "risk_coverage_curve.csv",
        "rho_guided_action_mix.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for child in ["outputs", "reports", "config"]:
        (args.out_dir / child).mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    if data.empty:
        raise RuntimeError("No selective reliability rows were available.")
    coverages = [float(part) for part in args.coverages.split(",") if part.strip()]
    data["pred_baseline"] = oof_predictions(data, include_rho=False, min_groups=args.min_groups)
    data["pred_baseline_plus_rho"] = oof_predictions(data, include_rho=True, min_groups=args.min_groups)
    metrics = pd.DataFrame(
        [
            policy_metrics(data, "pred_baseline", coverages, args.high_confidence_threshold),
            policy_metrics(data, "pred_baseline_plus_rho", coverages, args.high_confidence_threshold),
        ]
    )
    deltas = paired_deltas(data, coverages, args.high_confidence_threshold)
    ci = bootstrap_ci(data, coverages, args.high_confidence_threshold, args.bootstrap, args.seed)
    curve = pd.concat(
        [
            risk_curve(data, "pred_baseline", coverages),
            risk_curve(data, "pred_baseline_plus_rho", coverages),
        ],
        ignore_index=True,
    )
    action = action_table(data, "pred_baseline_plus_rho")
    data.to_csv(args.out_dir / "outputs" / "selective_reliability_predictions.csv", index=False)
    data.drop(columns=["pred_baseline", "pred_baseline_plus_rho"]).to_csv(args.out_dir / "outputs" / "selective_reliability_dataset.csv", index=False)
    metrics.to_csv(args.out_dir / "outputs" / "selective_reliability_metrics.csv", index=False)
    deltas.to_csv(args.out_dir / "outputs" / "selective_reliability_deltas.csv", index=False)
    ci.to_csv(args.out_dir / "outputs" / "selective_reliability_bootstrap_ci.csv", index=False)
    curve.to_csv(args.out_dir / "outputs" / "risk_coverage_curve.csv", index=False)
    action.to_csv(args.out_dir / "outputs" / "rho_guided_action_mix.csv", index=False)
    metadata = {
        "command": "python scripts/run_rho_guided_selective_reliability.py",
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "coverages": coverages,
        "high_confidence_threshold": args.high_confidence_threshold,
        "policy_constraint": "No oracle correctness feature; correctness is used only for out-of-fold evaluation/training labels.",
    }
    (args.out_dir / "config" / "reproduce.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_report(args.out_dir, data, metrics, deltas, ci, curve, action)
    print(args.out_dir)


if __name__ == "__main__":
    main()
