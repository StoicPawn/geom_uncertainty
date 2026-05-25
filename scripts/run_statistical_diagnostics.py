from __future__ import annotations

import argparse
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments" / "controls" / "statistical_diagnostics",
    )
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    for child in ["outputs", "reports", "config"]:
        (out_dir / child).mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "04_uncertainty_steering" / "reports").mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        value = float(x[0])
        return value, value, value
    boots = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.choice(x, size=x.size, replace=True)
        boots[idx] = float(np.mean(sample))
    return float(np.mean(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap(
    acc: np.ndarray,
    ctrl: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, float]:
    x = np.asarray(acc, dtype=np.float64)
    y = np.asarray(ctrl, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return {
            "n": 0,
            "accessible_mean": float("nan"),
            "control_mean": float("nan"),
            "mean_difference": float("nan"),
            "mean_difference_ci_low": float("nan"),
            "mean_difference_ci_high": float("nan"),
            "ratio": float("nan"),
            "ratio_ci_low": float("nan"),
            "ratio_ci_high": float("nan"),
            "cohen_d_paired": float("nan"),
            "cohen_d_ci_low": float("nan"),
            "cohen_d_ci_high": float("nan"),
            "standardized_mean_difference": float("nan"),
            "standardized_mean_difference_ci_low": float("nan"),
            "standardized_mean_difference_ci_high": float("nan"),
        }
    diff = x - y

    def stats_for(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        d = a - b
        mean_diff = float(np.mean(d))
        ratio = float(np.mean(a) / max(abs(float(np.mean(b))), 1e-12))
        sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
        cohen = mean_diff / sd if sd > 1e-12 else float("nan")
        return mean_diff, ratio, cohen

    boots = np.empty((n_boot, 3), dtype=np.float64)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, x.size, size=x.size)
        boots[idx] = stats_for(x[sample_idx], y[sample_idx])
    mean_diff, ratio, cohen = stats_for(x, y)
    return {
        "n": int(x.size),
        "accessible_mean": float(np.mean(x)),
        "control_mean": float(np.mean(y)),
        "mean_difference": mean_diff,
        "mean_difference_ci_low": float(np.nanpercentile(boots[:, 0], 2.5)),
        "mean_difference_ci_high": float(np.nanpercentile(boots[:, 0], 97.5)),
        "ratio": ratio,
        "ratio_ci_low": float(np.nanpercentile(boots[:, 1], 2.5)),
        "ratio_ci_high": float(np.nanpercentile(boots[:, 1], 97.5)),
        "cohen_d_paired": cohen,
        "cohen_d_ci_low": float(np.nanpercentile(boots[:, 2], 2.5)),
        "cohen_d_ci_high": float(np.nanpercentile(boots[:, 2], 97.5)),
        "standardized_mean_difference": cohen,
        "standardized_mean_difference_ci_low": float(np.nanpercentile(boots[:, 2], 2.5)),
        "standardized_mean_difference_ci_high": float(np.nanpercentile(boots[:, 2], 97.5)),
    }


def independent_bootstrap(
    high: np.ndarray,
    low: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, float]:
    x = np.asarray(high, dtype=np.float64)
    y = np.asarray(low, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return {"n_high": int(x.size), "n_low": int(y.size)}

    def stats_for(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        diff = float(np.mean(a) - np.mean(b))
        ratio = float(np.mean(a) / max(abs(float(np.mean(b))), 1e-12))
        pooled = math.sqrt(
            max(
                ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1))
                / max(a.size + b.size - 2, 1),
                0.0,
            )
        )
        cohen = diff / pooled if pooled > 1e-12 else float("nan")
        return diff, ratio, cohen

    boots = np.empty((n_boot, 3), dtype=np.float64)
    for idx in range(n_boot):
        bx = rng.choice(x, size=x.size, replace=True)
        by = rng.choice(y, size=y.size, replace=True)
        boots[idx] = stats_for(bx, by)
    diff, ratio, cohen = stats_for(x, y)
    return {
        "n_high": int(x.size),
        "n_low": int(y.size),
        "high_mean": float(np.mean(x)),
        "low_mean": float(np.mean(y)),
        "mean_difference": diff,
        "mean_difference_ci_low": float(np.nanpercentile(boots[:, 0], 2.5)),
        "mean_difference_ci_high": float(np.nanpercentile(boots[:, 0], 97.5)),
        "ratio": ratio,
        "ratio_ci_low": float(np.nanpercentile(boots[:, 1], 2.5)),
        "ratio_ci_high": float(np.nanpercentile(boots[:, 1], 97.5)),
        "cohen_d": cohen,
        "cohen_d_ci_low": float(np.nanpercentile(boots[:, 2], 2.5)),
        "cohen_d_ci_high": float(np.nanpercentile(boots[:, 2], 97.5)),
        "standardized_mean_difference": cohen,
        "standardized_mean_difference_ci_low": float(np.nanpercentile(boots[:, 2], 2.5)),
        "standardized_mean_difference_ci_high": float(np.nanpercentile(boots[:, 2], 97.5)),
    }


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3:
        return float("nan"), float("nan"), float("nan")
    estimate = float(spearmanr(a, b).statistic)
    boots = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, a.size, size=a.size)
        boots[idx] = float(spearmanr(a[sample_idx], b[sample_idx]).statistic)
    return estimate, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def steering_contrast_bootstrap(steering: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    frame = steering[steering["top_k_output"].isin([16, 32, 64, 128, 256])].copy()
    key_cols = [
        "model",
        "prompt_id",
        "task",
        "topic",
        "layer",
        "subspace_family",
        "subspace_k",
        "rep",
        "top_k_output",
        "sign",
    ]
    rows: list[dict[str, object]] = []
    for metric in ["abs_delta_entropy", "abs_delta_varentropy"]:
        for (top_k, sign), group in frame.groupby(["top_k_output", "sign"], dropna=False):
            pivot = group.pivot_table(index=key_cols, columns="direction", values=metric, aggfunc="mean")
            for control in ["random_control", "grad_orthogonal_control"]:
                if "accessible_ls" not in pivot or control not in pivot:
                    continue
                stats = paired_bootstrap(pivot["accessible_ls"].to_numpy(), pivot[control].to_numpy(), rng, n_boot)
                rows.append(
                    {
                        "analysis": "steering_equal_fisher_output",
                        "top_k_output": int(top_k),
                        "sign": sign,
                        "control": control,
                        "metric": metric,
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def rho_quartile_bootstrap(interventions: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    frame = interventions.copy()
    frame["abs_delta_entropy"] = frame["delta_entropy"].abs()
    rows: list[dict[str, object]] = []
    for epsilon, group in frame.groupby("epsilon", dropna=False):
        if group["rho_direction"].nunique() < 4:
            continue
        group = group.assign(rho_quartile=pd.qcut(group["rho_direction"], 4, labels=False, duplicates="drop"))
        low = group[group["rho_quartile"] == group["rho_quartile"].min()]
        high = group[group["rho_quartile"] == group["rho_quartile"].max()]
        for metric in ["abs_delta_entropy", "abs_delta_varentropy"]:
            stats = independent_bootstrap(high[metric].to_numpy(), low[metric].to_numpy(), rng, n_boot)
            rows.append(
                {
                    "analysis": "rho_quartile_q4_minus_q1",
                    "epsilon": float(epsilon),
                    "metric": metric,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def topk_robustness_bootstrap(scores: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    key_cols = [
        "model",
        "prompt_id",
        "task",
        "topic",
        "observed_condition",
        "layer",
        "subspace_family",
        "subspace_k",
        "rep",
    ]
    pivot = scores.pivot_table(index=key_cols, columns="top_k_output", values="rho", aggfunc="mean")
    rows: list[dict[str, object]] = []
    if 32 not in pivot:
        return pd.DataFrame()
    ref = pivot[32].to_numpy()
    for top_k in sorted(col for col in pivot.columns if col != 32):
        values = pivot[top_k].to_numpy()
        mask = np.isfinite(ref) & np.isfinite(values)
        diff = np.abs(values[mask] - ref[mask])
        rho, rho_lo, rho_hi = bootstrap_spearman(values[mask], ref[mask], rng, n_boot)
        mean_diff, diff_lo, diff_hi = mean_ci(diff, rng, n_boot)
        rows.append(
            {
                "analysis": "topk_rank_stability_vs_32",
                "top_k_output": int(top_k),
                "n": int(mask.sum()),
                "rho_spearman": rho,
                "rho_spearman_ci_low": rho_lo,
                "rho_spearman_ci_high": rho_hi,
                "rho_abs_diff_mean": mean_diff,
                "rho_abs_diff_ci_low": diff_lo,
                "rho_abs_diff_ci_high": diff_hi,
            }
        )
    return pd.DataFrame(rows)


def design_matrix(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = [np.ones((len(frame), 1), dtype=np.float64)]
    for col in numeric:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        med = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
        values = np.where(np.isfinite(values), values, med)
        sd = float(np.std(values))
        parts.append(((values - float(np.mean(values))) / max(sd, 1e-12)).reshape(-1, 1))
    if categorical:
        dummies = pd.get_dummies(frame[categorical].astype(str), drop_first=True, dtype=float)
        if not dummies.empty:
            parts.append(dummies.to_numpy(dtype=np.float64))
    return np.hstack(parts)


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    beta = np.linalg.pinv(x) @ y
    return y - x @ beta


def residual_effect_bootstrap(steering: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    frame = steering.copy()
    frame = frame[np.isfinite(frame["rho"])].copy()
    numeric = [
        "entropy",
        "varentropy",
        "confidence",
        "margin",
        "jacobian_fro_norm",
        "fisher_output_energy",
        "grad_entropy_proj_norm",
        "grad_varentropy_proj_norm",
        "top_k_output",
        "layer",
    ]
    categorical = ["model", "task", "topic", "subspace_family", "direction", "sign"]
    x = design_matrix(frame, numeric, categorical)
    rho_resid = residualize(frame["rho"].to_numpy(dtype=np.float64), x)
    rows: list[dict[str, object]] = []
    for outcome in ["abs_delta_entropy", "abs_delta_varentropy"]:
        y = pd.to_numeric(frame[outcome], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(y)
        y_resid = residualize(y[mask], x[mask])
        r_resid = rho_resid[mask]
        rho, lo, hi = bootstrap_spearman(r_resid, y_resid, rng, n_boot)
        rows.append(
            {
                "analysis": "residual_rho_effect_after_scalar_gradient_geometric_controls",
                "outcome": outcome,
                "n": int(mask.sum()),
                "spearman_residual_effect": rho,
                "spearman_ci_low": lo,
                "spearman_ci_high": hi,
                "controls": ",".join(numeric + categorical),
            }
        )
    return pd.DataFrame(rows)


def matched_scalar_gradient_pairs(scores: pd.DataFrame, prompts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = scores[scores["top_k_output"] == 32].copy()
    prompt_cols = prompts[["model", "prompt_id", "prompt"]].drop_duplicates()
    frame = frame.merge(prompt_cols, on=["model", "prompt_id"], how="left")
    match_cols = [
        "grad_entropy_proj_norm",
        "grad_varentropy_proj_norm",
        "fisher_output_fro_norm",
        "jacobian_fro_norm",
    ]
    values = frame[match_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scales = values.std(ddof=0).replace(0, 1.0)
    rows: list[dict[str, object]] = []
    for (_model, _prompt_id, _layer), group in frame.groupby(["model", "prompt_id", "layer"], dropna=False):
        if len(group) < 2:
            continue
        local = group.reset_index(drop=False)
        z = (local[match_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) / scales).to_numpy(dtype=np.float64)
        rho = local["rho"].to_numpy(dtype=np.float64)
        for i in range(len(local)):
            for j in range(i + 1, len(local)):
                distance = float(np.sqrt(np.mean((z[i] - z[j]) ** 2)))
                rho_diff = float(abs(rho[i] - rho[j]))
                if rho_diff < 0.15 or distance > 0.75:
                    continue
                metric_diffs = {
                    f"{col}_abs_diff": float(abs(local.iloc[i][col] - local.iloc[j][col])) for col in match_cols
                }
                scalar_diffs = {
                    "entropy_abs_diff": float(abs(local.iloc[i]["entropy"] - local.iloc[j]["entropy"])),
                    "varentropy_abs_diff": float(abs(local.iloc[i]["varentropy"] - local.iloc[j]["varentropy"])),
                    "confidence_abs_diff": float(abs(local.iloc[i]["confidence"] - local.iloc[j]["confidence"])),
                    "margin_abs_diff": float(abs(local.iloc[i]["margin"] - local.iloc[j]["margin"])),
                }
                pair_score = rho_diff / (1.0 + distance)
                rows.append(
                    {
                        "row_a": int(i),
                        "row_b": int(j),
                        "pair_score": pair_score,
                        "match_distance_z": distance,
                        "rho_abs_diff": rho_diff,
                        "model_a": local.iloc[i]["model"],
                        "model_b": local.iloc[j]["model"],
                        "prompt_id_a": int(local.iloc[i]["prompt_id"]),
                        "prompt_id_b": int(local.iloc[j]["prompt_id"]),
                        "prompt_a": local.iloc[i].get("prompt", ""),
                        "prompt_b": local.iloc[j].get("prompt", ""),
                        "layer_a": int(local.iloc[i]["layer"]),
                        "layer_b": int(local.iloc[j]["layer"]),
                        "subspace_family_a": local.iloc[i]["subspace_family"],
                        "subspace_family_b": local.iloc[j]["subspace_family"],
                        "subspace_k_a": int(local.iloc[i]["subspace_k"]),
                        "subspace_k_b": int(local.iloc[j]["subspace_k"]),
                        "rho_a": float(local.iloc[i]["rho"]),
                        "rho_b": float(local.iloc[j]["rho"]),
                        "grad_entropy_proj_norm_a": float(local.iloc[i]["grad_entropy_proj_norm"]),
                        "grad_entropy_proj_norm_b": float(local.iloc[j]["grad_entropy_proj_norm"]),
                        "grad_varentropy_proj_norm_a": float(local.iloc[i]["grad_varentropy_proj_norm"]),
                        "grad_varentropy_proj_norm_b": float(local.iloc[j]["grad_varentropy_proj_norm"]),
                        "entropy_a": float(local.iloc[i]["entropy"]),
                        "entropy_b": float(local.iloc[j]["entropy"]),
                        "varentropy_a": float(local.iloc[i]["varentropy"]),
                        "varentropy_b": float(local.iloc[j]["varentropy"]),
                        **scalar_diffs,
                        **metric_diffs,
                    }
                )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs, pd.DataFrame()
    pairs = (
        pairs.sort_values(["rho_abs_diff", "match_distance_z"], ascending=[False, True])
        .drop_duplicates(["prompt_id_a", "prompt_id_b", "layer_a", "layer_b", "subspace_family_a", "subspace_family_b"])
        .head(200)
        .reset_index(drop=True)
    )
    summary = pd.DataFrame(
        [
            {
                "n_pairs": int(len(pairs)),
                "rho_abs_diff_mean": float(pairs["rho_abs_diff"].mean()),
                "rho_abs_diff_max": float(pairs["rho_abs_diff"].max()),
                "match_distance_z_median": float(pairs["match_distance_z"].median()),
                "entropy_abs_diff_median": float(pairs["entropy_abs_diff"].median()),
                "varentropy_abs_diff_median": float(pairs["varentropy_abs_diff"].median()),
                "grad_entropy_proj_norm_abs_diff_median": float(
                    pairs["grad_entropy_proj_norm_abs_diff"].median()
                ),
                "grad_varentropy_proj_norm_abs_diff_median": float(
                    pairs["grad_varentropy_proj_norm_abs_diff"].median()
                ),
            }
        ]
    )
    return pairs, summary


def normalize_prompt(text: str) -> str:
    text = str(text).lower().replace("[mask]", " mask ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prompt_duplicate_tables(threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "experiments").rglob("*.csv")):
        normalized_path = str(path.relative_to(ROOT)).replace("\\", "/")
        if not (
            "/data/" in normalized_path
            or normalized_path.endswith("/prompt_tables.csv")
            or normalized_path.endswith("/prompts.csv")
            or normalized_path.endswith("/source_prompts.csv")
        ):
            continue
        try:
            df = pd.read_csv(path, usecols=lambda c: c in {"model", "prompt_id", "prompt", "task", "topic", "fact_id"})
        except Exception:
            continue
        if "prompt" not in df.columns or df.empty:
            continue
        for idx, row in df.dropna(subset=["prompt"]).iterrows():
            rows.append(
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "row_index": int(idx),
                    "model": row.get("model", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "task": row.get("task", ""),
                    "topic": row.get("topic", ""),
                    "fact_id": row.get("fact_id", ""),
                    "prompt": row["prompt"],
                    "normalized_prompt": normalize_prompt(row["prompt"]),
                }
            )
    prompts = pd.DataFrame(rows).drop_duplicates(["source_file", "row_index", "normalized_prompt"])
    exact = prompts[prompts.duplicated("prompt", keep=False)].sort_values(["prompt", "source_file"])
    normalized = prompts[prompts.duplicated("normalized_prompt", keep=False)].sort_values(
        ["normalized_prompt", "source_file"]
    )
    exact_within_source = prompts[prompts.duplicated(["source_file", "prompt"], keep=False)]
    normalized_within_source = prompts[prompts.duplicated(["source_file", "normalized_prompt"], keep=False)]
    unique = prompts.drop_duplicates("normalized_prompt").reset_index(drop=True)
    near_rows: list[dict[str, object]] = []
    texts = unique["normalized_prompt"].tolist()
    for i, left in enumerate(texts):
        for j in range(i + 1, len(texts)):
            right = texts[j]
            if abs(len(left) - len(right)) > max(10, 0.4 * max(len(left), len(right))):
                continue
            ratio = SequenceMatcher(None, left, right).ratio()
            if threshold <= ratio < 1.0:
                near_rows.append(
                    {
                        "similarity": float(ratio),
                        "prompt_a": unique.iloc[i]["prompt"],
                        "prompt_b": unique.iloc[j]["prompt"],
                        "source_file_a": unique.iloc[i]["source_file"],
                        "source_file_b": unique.iloc[j]["source_file"],
                        "normalized_prompt_a": left,
                        "normalized_prompt_b": right,
                    }
                )
    near = pd.DataFrame(near_rows).sort_values("similarity", ascending=False) if near_rows else pd.DataFrame()
    summary = pd.DataFrame(
        [
            {
                "n_prompt_rows": int(len(prompts)),
                "n_unique_exact_prompts": int(prompts["prompt"].nunique()) if not prompts.empty else 0,
                "n_unique_normalized_prompts": int(prompts["normalized_prompt"].nunique()) if not prompts.empty else 0,
                "n_exact_duplicate_rows": int(len(exact)),
                "n_normalized_duplicate_rows": int(len(normalized)),
                "n_exact_duplicate_rows_within_source": int(len(exact_within_source)),
                "n_normalized_duplicate_rows_within_source": int(len(normalized_within_source)),
                "n_exact_duplicate_groups": int((prompts.groupby("prompt").size() > 1).sum()) if not prompts.empty else 0,
                "n_normalized_duplicate_groups": int((prompts.groupby("normalized_prompt").size() > 1).sum())
                if not prompts.empty
                else 0,
                "n_near_duplicate_pairs": int(len(near)),
                "near_duplicate_threshold": float(threshold),
            }
        ]
    )
    return summary, exact, normalized, near


def qualitative_examples(steering: pd.DataFrame, prompts: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    prompt_cols = prompts[
        ["model", "prompt_id", "prompt", "final_top1_token", "task", "topic", "observed_condition"]
    ].drop_duplicates()
    frame = steering.merge(prompt_cols, on=["model", "prompt_id", "task", "topic", "observed_condition"], how="left")
    frame = frame[
        (frame["top_k_output"] == 32)
        & (frame["direction"] == "accessible_ls")
        & (frame["directional_success"] == 1)
        & (frame["selected_top1_changed"] == 0)
        & (frame["full_vocab_top10_jaccard"] >= 0.9)
    ].copy()
    cols = [
        "model",
        "prompt_id",
        "prompt",
        "task",
        "topic",
        "observed_condition",
        "final_top1_token",
        "sign",
        "layer",
        "subspace_family",
        "subspace_k",
        "rho",
        "entropy",
        "entropy_after",
        "delta_entropy",
        "varentropy",
        "varentropy_after",
        "delta_varentropy",
        "full_vocab_top10_jaccard",
        "candidate_set_kl",
        "candidate_mass_retention_ratio",
        "selected_top1_changed",
    ]
    if frame.empty:
        return pd.DataFrame(columns=cols), "# Answer-Preserving Qualitative Examples\n\nNo examples passed filters.\n"
    selected = (
        frame.sort_values("abs_delta_entropy", ascending=False)
        .groupby(["sign"], group_keys=False)
        .head(12)
        .sort_values(["sign", "abs_delta_entropy"], ascending=[True, False])
    )[cols]
    lines = [
        "# Answer-Preserving Qualitative Examples",
        "",
        "Rows are accessible-steering interventions with unchanged selected top-1 token and full-vocabulary top-10 Jaccard at least 0.90.",
        "",
    ]
    for _, row in selected.head(16).iterrows():
        lines.extend(
            [
                f"## {row['sign']} | {row['model']} | prompt {int(row['prompt_id'])}",
                "",
                f"- Prompt: `{row['prompt']}`",
                f"- Preserved answer token: `{row['final_top1_token']}`",
                f"- Entropy: `{row['entropy']:.4f} -> {row['entropy_after']:.4f}` (`Delta={row['delta_entropy']:.4f}`)",
                f"- Varentropy: `{row['varentropy']:.4f} -> {row['varentropy_after']:.4f}` (`Delta={row['delta_varentropy']:.4f}`)",
                f"- rho: `{row['rho']:.4f}`; top-10 Jaccard: `{row['full_vocab_top10_jaccard']:.3f}`; candidate KL: `{row['candidate_set_kl']:.4g}`",
                "",
            ]
        )
    return selected.reset_index(drop=True), "\n".join(lines)


def failure_modes(steering: pd.DataFrame, scores: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    prompt_cols = prompts[["model", "prompt_id", "prompt", "final_top1_token"]].drop_duplicates()
    accessible = steering[(steering["top_k_output"] == 32) & (steering["direction"] == "accessible_ls")].copy()
    accessible = accessible.merge(prompt_cols, on=["model", "prompt_id"], how="left")
    rows: list[pd.DataFrame] = []
    if not accessible.empty:
        high_rho = accessible["rho"].quantile(0.8)
        low_rho = accessible["rho"].quantile(0.2)
        high_move = accessible["abs_delta_entropy"].quantile(0.8)
        low_move = accessible["abs_delta_entropy"].quantile(0.2)
        rows.append(
            accessible[(accessible["rho"] >= high_rho) & (accessible["abs_delta_entropy"] <= low_move)]
            .assign(failure_type="high_rho_low_delta_entropy")
            .sort_values(["rho", "abs_delta_entropy"], ascending=[False, True])
            .head(40)
        )
        rows.append(
            accessible[(accessible["rho"] <= low_rho) & (accessible["abs_delta_entropy"] >= high_move)]
            .assign(failure_type="low_rho_high_delta_entropy")
            .sort_values(["abs_delta_entropy", "rho"], ascending=[False, True])
            .head(40)
        )
        rows.append(
            accessible[accessible["selected_top1_changed"] == 1]
            .assign(failure_type="top1_changed")
            .sort_values("abs_delta_entropy", ascending=False)
            .head(40)
        )
        rows.append(
            accessible[accessible["full_vocab_top10_jaccard"] < 0.85]
            .assign(failure_type="top10_degradation")
            .sort_values("full_vocab_top10_jaccard")
            .head(40)
        )
    key_cols = [
        "model",
        "prompt_id",
        "task",
        "topic",
        "observed_condition",
        "layer",
        "subspace_family",
        "subspace_k",
        "rep",
    ]
    pivot = scores.pivot_table(index=key_cols, columns="top_k_output", values="rho", aggfunc="mean")
    if len(pivot) and 32 in pivot:
        pivot = pivot.assign(rho_topk_range=pivot.max(axis=1) - pivot.min(axis=1)).reset_index()
        unstable = (
            pivot.sort_values("rho_topk_range", ascending=False)
            .head(80)
            .merge(prompt_cols, on=["model", "prompt_id"], how="left")
            .assign(
                failure_type="rho_unstable_across_topk",
                direction="score_only",
                sign="score_only",
                rho=pivot.sort_values("rho_topk_range", ascending=False).head(80).get(32).to_numpy(),
                abs_delta_entropy=np.nan,
                full_vocab_top10_jaccard=np.nan,
                selected_top1_changed=np.nan,
            )
        )
        rows.append(unstable)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    keep = [
        "failure_type",
        "model",
        "prompt_id",
        "prompt",
        "task",
        "topic",
        "observed_condition",
        "final_top1_token",
        "layer",
        "subspace_family",
        "subspace_k",
        "top_k_output",
        "direction",
        "sign",
        "rho",
        "abs_delta_entropy",
        "abs_delta_varentropy",
        "full_vocab_top10_jaccard",
        "selected_top1_changed",
        "rho_topk_range",
        "candidate_set_kl",
        "candidate_mass_retention_ratio",
    ]
    return combined[[col for col in keep if col in combined.columns]].reset_index(drop=True)


def write_report(
    out_dir: Path,
    steering_ci: pd.DataFrame,
    quartile_ci: pd.DataFrame,
    topk_ci: pd.DataFrame,
    residual_ci: pd.DataFrame,
    matched_summary: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    failure_table: pd.DataFrame,
) -> None:
    lines = [
        "# Statistical And Diagnostic Package",
        "",
        "This package adds uncertainty estimates and paper-facing diagnostics on top of the curated experiment outputs.",
        "",
        "## Confirmatory Analyses",
        "",
        "- Bootstrap CIs for accessible-vs-control steering contrasts.",
        "- Bootstrap CIs for rho-quartile movement contrasts.",
        "- Bootstrap CIs for top-k robustness of rho rankings.",
        "- Same scalar uncertainty plus same projected-gradient controls but different rho.",
        "- Answer-preserving qualitative steering examples in Experiment 4.",
        "",
        "## Exploratory / Diagnostic Analyses",
        "",
        "- Residual rho effects after scalar, gradient, and geometric controls.",
        "- Prompt duplicate and near-duplicate audit.",
        "- Failure-mode table for high-rho/low-movement, low-rho/high-movement, top-k instability, and answer degradation.",
        "",
        "## Key Tables",
        "",
        "### Steering Contrasts",
        "",
        "```text",
        steering_ci.head(16).to_string(index=False),
        "```",
        "",
        "### Rho Quartiles",
        "",
        "```text",
        quartile_ci.to_string(index=False),
        "```",
        "",
        "### Top-k Robustness",
        "",
        "```text",
        topk_ci.to_string(index=False),
        "```",
        "",
        "### Residual Effects",
        "",
        "```text",
        residual_ci.to_string(index=False),
        "```",
        "",
        "### Same Scalar + Same Projected Gradients + Different Rho",
        "",
        "```text",
        matched_summary.to_string(index=False),
        "```",
        "",
        "### Prompt Duplicate Audit",
        "",
        "```text",
        duplicate_summary.to_string(index=False),
        "```",
        "",
        "### Failure Modes",
        "",
        "```text",
        failure_table["failure_type"].value_counts().rename_axis("failure_type").reset_index(name="n").to_string(index=False)
        if not failure_table.empty
        else "(empty)",
        "```",
        "",
        "## Claim Boundary",
        "",
        "`rho` should not be described as beating direct projected gradients as a raw infinitesimal predictor. The supported claim is narrower and cleaner: `rho` is a geometric accessibility coefficient that decomposes varentropy with respect to an internal route, remains informative after controls, and identifies directions where uncertainty can be moved while preserving the answer neighborhood.",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    rng = np.random.default_rng(args.seed)

    steering = read_csv("experiments/controls/topk_robustness/outputs/topk_steering_records.csv")
    scores = read_csv("experiments/controls/topk_robustness/outputs/topk_scores.csv")
    interventions = read_csv("experiments/02_local_perturbation_prediction/outputs/intervention_records.csv")
    prompts = read_csv("experiments/controls/topk_robustness/outputs/prompt_tables.csv")

    steering_ci = steering_contrast_bootstrap(steering, rng, args.bootstrap)
    quartile_ci = rho_quartile_bootstrap(interventions, rng, args.bootstrap)
    topk_ci = topk_robustness_bootstrap(scores, rng, args.bootstrap)
    residual_ci = residual_effect_bootstrap(steering, rng, args.bootstrap)
    matched_pairs, matched_summary = matched_scalar_gradient_pairs(scores, prompts)
    dup_summary, exact_dups, normalized_dups, near_dups = prompt_duplicate_tables(args.near_duplicate_threshold)
    examples, examples_md = qualitative_examples(steering, prompts)
    failures = failure_modes(steering, scores, prompts)

    out = args.out_dir / "outputs"
    steering_ci.to_csv(out / "bootstrap_steering_contrasts.csv", index=False)
    quartile_ci.to_csv(out / "bootstrap_rho_quartiles.csv", index=False)
    topk_ci.to_csv(out / "bootstrap_topk_robustness.csv", index=False)
    residual_ci.to_csv(out / "bootstrap_residual_effects.csv", index=False)
    matched_pairs.to_csv(out / "same_scalar_gradient_different_rho_pairs.csv", index=False)
    matched_summary.to_csv(out / "same_scalar_gradient_different_rho_summary.csv", index=False)
    dup_summary.to_csv(out / "prompt_duplicate_summary.csv", index=False)
    exact_dups.to_csv(out / "prompt_exact_duplicates.csv", index=False)
    normalized_dups.to_csv(out / "prompt_normalized_duplicates.csv", index=False)
    near_dups.to_csv(out / "prompt_near_duplicates.csv", index=False)
    failures.to_csv(out / "failure_modes.csv", index=False)

    steering_out = ROOT / "experiments" / "04_uncertainty_steering" / "outputs"
    reports_out = ROOT / "experiments" / "04_uncertainty_steering" / "reports"
    examples.to_csv(steering_out / "answer_preserving_qualitative_examples.csv", index=False)
    (reports_out / "answer_preserving_qualitative_examples.md").write_text(examples_md, encoding="utf-8")

    config = {
        "control_id": "statistical_diagnostics",
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "minimal_command": (
            "python scripts/run_statistical_diagnostics.py "
            f"--bootstrap {args.bootstrap} --seed {args.seed}"
        ),
    }
    pd.Series(config).to_json(args.out_dir / "config" / "reproduce.json", indent=2)
    write_report(args.out_dir, steering_ci, quartile_ci, topk_ci, residual_ci, matched_summary, dup_summary, failures)


if __name__ == "__main__":
    main()
