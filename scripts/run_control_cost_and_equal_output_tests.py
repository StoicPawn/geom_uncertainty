from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MINIMAL_OUT = ROOT / "experiments" / "04_uncertainty_steering" / "01_minimal_intervention_energy"
EQUAL_OUT = ROOT / "experiments" / "04_uncertainty_steering" / "02_equal_output_movement"
warnings.filterwarnings("ignore", category=FutureWarning, message="Downcasting behavior.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-entropy", type=float, default=0.02)
    parser.add_argument("--tau-varentropy", type=float, default=0.04)
    parser.add_argument("--top-k-output", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--minimal-out", type=Path, default=MINIMAL_OUT)
    parser.add_argument("--equal-out", type=Path, default=EQUAL_OUT)
    return parser.parse_args()


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    local = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(local) < 4 or local.iloc[:, 0].nunique() < 2 or local.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(spearmanr(local.iloc[:, 0], local.iloc[:, 1]).correlation)


def safe_auc(score: pd.Series, easy: pd.Series) -> tuple[float, float]:
    local = pd.concat([score, easy], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(local) < 4 or local.iloc[:, 1].nunique() < 2 or local.iloc[:, 0].nunique() < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(local.iloc[:, 1].astype(int), local.iloc[:, 0].astype(float)))
    return auc, max(auc, 1.0 - auc)


def local_linear_cost(step_norm: pd.Series, movement: pd.Series, tau: float) -> pd.Series:
    x = step_norm.astype(float)
    y = movement.astype(float)
    cost = x * float(tau) / y.where(y > 1e-12)
    return cost.replace([np.inf, -np.inf], np.nan)


def signed_entropy_movement(frame: pd.DataFrame) -> pd.Series:
    sign = frame["sign"].astype(str)
    delta = frame["delta_entropy"].astype(float)
    return pd.Series(np.where(sign.eq("increase"), delta, -delta), index=frame.index)


def add_random_adjusted_rho(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "prompt_id", "task", "topic", "layer", "subspace_k", "top_k_output"]
    random_mean = (
        frame[frame["subspace_family"].eq("random")]
        .groupby(keys, as_index=False)["rho"]
        .mean()
        .rename(columns={"rho": "rho_random_mean"})
    )
    out = frame.merge(random_mean, on=keys, how="left")
    out["rho_random_adjusted"] = out["rho"] - out["rho_random_mean"]
    return out


def build_minimal_energy_records(args: argparse.Namespace) -> pd.DataFrame:
    path = ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "topk_steering_records.csv"
    records = pd.read_csv(path)
    records = records[
        records["direction"].eq("accessible_ls")
        & records["top_k_output"].eq(args.top_k_output)
        & records["sign"].isin(["increase", "decrease"])
    ].copy()
    records["source"] = "mlm_topk32_local_linear"
    records["entropy_target_movement"] = signed_entropy_movement(records)
    records["varentropy_target_movement"] = records["abs_delta_varentropy"].astype(float)
    records["cost_entropy_tau"] = local_linear_cost(
        records["latent_step_norm"], records["entropy_target_movement"], args.tau_entropy
    )
    records["cost_varentropy_tau"] = local_linear_cost(
        records["latent_step_norm"], records["varentropy_target_movement"], args.tau_varentropy
    )
    records["observed_reached_entropy_tau"] = (records["entropy_target_movement"] >= args.tau_entropy).astype(int)
    records["observed_reached_varentropy_tau"] = (records["varentropy_target_movement"] >= args.tau_varentropy).astype(int)
    records["neg_log_cost_entropy_tau"] = -np.log(records["cost_entropy_tau"])
    records["neg_log_cost_varentropy_tau"] = -np.log(records["cost_varentropy_tau"])
    records = add_random_adjusted_rho(records)
    keep = [
        "source",
        "model",
        "prompt_id",
        "task",
        "topic",
        "observed_condition",
        "layer",
        "subspace_family",
        "subspace_k",
        "rep",
        "top_k_output",
        "sign",
        "rho",
        "rho_random_adjusted",
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
        "latent_step_norm",
        "fisher_output_energy",
        "entropy_target_movement",
        "varentropy_target_movement",
        "observed_reached_entropy_tau",
        "observed_reached_varentropy_tau",
        "cost_entropy_tau",
        "cost_varentropy_tau",
        "neg_log_cost_entropy_tau",
        "neg_log_cost_varentropy_tau",
        "selected_top1_changed",
        "target_correct_changed",
        "full_vocab_top10_jaccard",
    ]
    return records[[col for col in keep if col in records.columns]].copy()


def build_decoder_cost_records(args: argparse.Namespace) -> pd.DataFrame:
    steering_path = (
        ROOT
        / "experiments"
        / "04_uncertainty_steering"
        / "decoder_main_battery"
        / "outputs"
        / "decoder_steering_records.csv"
    )
    score_path = (
        ROOT
        / "experiments"
        / "04_uncertainty_steering"
        / "decoder_main_battery"
        / "outputs"
        / "decoder_scores.csv"
    )
    if not steering_path.exists() or not score_path.exists():
        return pd.DataFrame()
    records = pd.read_csv(steering_path)
    scores = pd.read_csv(score_path)
    records = records[
        records["direction"].eq("accessible_ls")
        & records["sign"].isin(["increase", "decrease"])
        & records["subspace_family"].isin(["state_pca", "delta_pca"])
    ].copy()
    records = records.merge(
        scores[
            [
                "model",
                "prompt_id",
                "subspace_family",
                "subspace_k",
                "rho",
                "v_access",
                "v_inaccess",
                "entropy",
                "varentropy",
                "confidence",
                "grad_entropy_proj_norm",
                "fisher_output_fro_norm",
                "semantic_entropy_proxy",
                "semantic_density_uncertainty",
                "haloscope_style_consistency_risk",
            ]
        ],
        on=["model", "prompt_id", "subspace_family", "subspace_k", "rho"],
        how="left",
        suffixes=("", "_score"),
    )
    records["source"] = "decoder_main_local_linear"
    records["top_k_output"] = records.get("top_m", 16)
    records["entropy_target_movement"] = signed_entropy_movement(records)
    records["varentropy_target_movement"] = records["abs_delta_varentropy"].astype(float)
    records["cost_entropy_tau"] = local_linear_cost(
        records["latent_step_norm"], records["entropy_target_movement"], args.tau_entropy
    )
    records["cost_varentropy_tau"] = local_linear_cost(
        records["latent_step_norm"], records["varentropy_target_movement"], args.tau_varentropy
    )
    records["observed_reached_entropy_tau"] = (records["entropy_target_movement"] >= args.tau_entropy).astype(int)
    records["observed_reached_varentropy_tau"] = (records["varentropy_target_movement"] >= args.tau_varentropy).astype(int)
    records["neg_log_cost_entropy_tau"] = -np.log(records["cost_entropy_tau"])
    records["neg_log_cost_varentropy_tau"] = -np.log(records["cost_varentropy_tau"])
    records["rho_random_adjusted"] = np.nan
    if "top10_jaccard" in records.columns:
        records["full_vocab_top10_jaccard"] = records["top10_jaccard"]
    return records


def interpolate_cost(step_norm: np.ndarray, movement: np.ndarray, tau: float) -> tuple[float, int]:
    order = np.argsort(step_norm)
    x = np.asarray(step_norm, dtype=float)[order]
    y = np.asarray(movement, dtype=float)[order]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return float("nan"), 0
    hits = np.where(y >= tau)[0]
    if hits.size == 0:
        return float("nan"), 0
    idx = int(hits[0])
    if idx == 0:
        if y[idx] <= 1e-12:
            return float(x[idx]), 1
        return float(x[idx] * tau / y[idx]), 1
    x0, x1 = float(x[idx - 1]), float(x[idx])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if abs(y1 - y0) <= 1e-12:
        return x1, 1
    alpha = (tau - y0) / (y1 - y0)
    alpha = min(1.0, max(0.0, alpha))
    return float(x0 + alpha * (x1 - x0)), 1


def build_full_grid_cost_records(args: argparse.Namespace) -> pd.DataFrame:
    path = ROOT / "experiments" / "04_uncertainty_steering" / "outputs" / "steering_records.csv"
    if not path.exists():
        return pd.DataFrame()
    records = pd.read_csv(path)
    records = records[
        records["mode"].eq("fisher_output_equal")
        & records["direction"].eq("accessible_ls")
        & records["sign"].isin(["increase", "decrease"])
    ].copy()
    if records.empty:
        return records
    records["entropy_target_movement"] = signed_entropy_movement(records)
    records["varentropy_target_movement"] = records["abs_delta_varentropy"].astype(float)
    group_cols = [
        "model",
        "prompt_id",
        "task",
        "topic",
        "observed_condition",
        "layer",
        "subspace_family",
        "k",
        "rep",
        "sign",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in records.groupby(group_cols, dropna=False):
        first = group.sort_values("latent_step_norm").iloc[0]
        entropy_cost, entropy_hit = interpolate_cost(
            group["latent_step_norm"].to_numpy(dtype=float),
            group["entropy_target_movement"].to_numpy(dtype=float),
            args.tau_entropy,
        )
        var_cost, var_hit = interpolate_cost(
            group["latent_step_norm"].to_numpy(dtype=float),
            group["varentropy_target_movement"].to_numpy(dtype=float),
            args.tau_varentropy,
        )
        row = {
            "source": "mlm_full_grid_interpolated",
            "model": first["model"],
            "prompt_id": first["prompt_id"],
            "task": first["task"],
            "topic": first["topic"],
            "observed_condition": first["observed_condition"],
            "layer": int(first["layer"]),
            "subspace_family": first["subspace_family"],
            "subspace_k": int(first["k"]),
            "rep": int(first["rep"]),
            "top_k_output": 16,
            "sign": first["sign"],
            "rho": float(first["rho"]),
            "rho_random_adjusted": np.nan,
            "entropy": float(first["entropy_before"]),
            "varentropy": float(first["varentropy_before"]),
            "confidence": np.nan,
            "margin": np.nan,
            "trace_fim": float(first["trace_fim"]),
            "jacobian_fro_norm": np.nan,
            "fisher_output_fro_norm": math.sqrt(max(float(first["trace_fim"]), 0.0)),
            "grad_entropy_proj_norm": math.sqrt(max(float(first["grad_norm_sq"]), 0.0)),
            "grad_varentropy_proj_norm": np.nan,
            "logit_norm": float(first["logit_norm"]),
            "latent_step_norm": float(group["latent_step_norm"].max()),
            "fisher_output_energy": float(group["fisher_output_energy"].max()),
            "entropy_target_movement": float(group["entropy_target_movement"].max()),
            "varentropy_target_movement": float(group["varentropy_target_movement"].max()),
            "observed_reached_entropy_tau": int(entropy_hit),
            "observed_reached_varentropy_tau": int(var_hit),
            "cost_entropy_tau": entropy_cost,
            "cost_varentropy_tau": var_cost,
            "neg_log_cost_entropy_tau": -math.log(entropy_cost) if entropy_cost > 0 else np.nan,
            "neg_log_cost_varentropy_tau": -math.log(var_cost) if var_cost > 0 else np.nan,
            "selected_top1_changed": float(group["selected_top1_changed"].max()),
            "target_correct_changed": float(group["target_correct_changed"].max()),
            "full_vocab_top10_jaccard": np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def predictor_benchmark(costs: pd.DataFrame, outcome: str) -> pd.DataFrame:
    predictors = [
        "rho",
        "rho_random_adjusted",
        "entropy",
        "varentropy",
        "confidence",
        "margin",
        "grad_entropy_proj_norm",
        "grad_varentropy_proj_norm",
        "fisher_output_fro_norm",
        "jacobian_fro_norm",
        "trace_fim",
        "logit_norm",
        "latent_step_norm",
    ]
    label_col = "easy_entropy_tau" if "entropy" in outcome else "easy_varentropy_tau"
    rows = []
    for source, source_frame in costs.groupby("source"):
        local = source_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome]).copy()
        if local.empty:
            continue
        local[label_col] = (local[outcome] >= local[outcome].median()).astype(int)
        for predictor in predictors:
            if predictor not in local.columns:
                continue
            auc, auc_abs = safe_auc(local[predictor], local[label_col])
            rows.append(
                {
                    "source": source,
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": int(local[[predictor, outcome]].dropna().shape[0]),
                    "spearman_with_easy_score": safe_spearman(local[predictor], local[outcome]),
                    "auroc_easy_raw": auc,
                    "auroc_easy_direction_adjusted": auc_abs,
                }
            )
    return pd.DataFrame(rows)


def residual_effects(costs: pd.DataFrame) -> pd.DataFrame:
    controls = [
        "entropy",
        "varentropy",
        "confidence",
        "margin",
        "grad_entropy_proj_norm",
        "grad_varentropy_proj_norm",
        "fisher_output_fro_norm",
        "jacobian_fro_norm",
        "trace_fim",
        "logit_norm",
    ]
    rows = []
    for source, source_frame in costs.groupby("source"):
        for outcome in ["neg_log_cost_entropy_tau", "neg_log_cost_varentropy_tau"]:
            available = [
                col
                for col in controls
                if col in source_frame.columns and source_frame[col].replace([np.inf, -np.inf], np.nan).notna().sum() >= 20
            ]
            cols = ["rho", outcome] + available + ["model", "task", "layer", "subspace_family", "sign"]
            local = source_frame[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
            if len(local) < 20 or local["rho"].nunique() < 3 or local[outcome].nunique() < 3:
                rows.append(
                    {
                        "source": source,
                        "outcome": outcome,
                        "n": int(len(local)),
                        "rho_residual_spearman": float("nan"),
                        "rho_beta_standardized": float("nan"),
                    }
                )
                continue
            dummies = pd.get_dummies(local[["model", "task", "layer", "subspace_family", "sign"]].astype(str), drop_first=True)
            x_controls = pd.concat([local[available].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
            x_controls = x_controls.astype(float).to_numpy()
            y = local[outcome].to_numpy(dtype=float)
            rho = local["rho"].to_numpy(dtype=float)
            y_resid = y - LinearRegression().fit(x_controls, y).predict(x_controls)
            rho_resid = rho - LinearRegression().fit(x_controls, rho).predict(x_controls)
            rho_z = (rho - rho.mean()) / max(rho.std(), 1e-12)
            y_z = (y - y.mean()) / max(y.std(), 1e-12)
            full_x = np.column_stack([rho_z, x_controls])
            beta = float(LinearRegression().fit(full_x, y_z).coef_[0])
            rows.append(
                {
                    "source": source,
                    "outcome": outcome,
                    "n": int(len(local)),
                    "rho_residual_spearman": safe_spearman(pd.Series(rho_resid), pd.Series(y_resid)),
                    "rho_beta_standardized": beta,
                    "controls": ",".join(available + ["model", "task", "layer", "subspace_family", "sign"]),
                }
            )
    return pd.DataFrame(rows)


def target_summary(costs: pd.DataFrame) -> pd.DataFrame:
    return costs.groupby(["source", "subspace_family", "sign"], as_index=False).agg(
        n=("prompt_id", "count"),
        rho_mean=("rho", "mean"),
        cost_entropy_tau_median=("cost_entropy_tau", "median"),
        cost_varentropy_tau_median=("cost_varentropy_tau", "median"),
        entropy_target_success_rate=("observed_reached_entropy_tau", "mean"),
        varentropy_target_success_rate=("observed_reached_varentropy_tau", "mean"),
        top1_changed_rate=("selected_top1_changed", "mean"),
        top10_jaccard_mean=("full_vocab_top10_jaccard", "mean"),
    )


def load_equal_output_records() -> pd.DataFrame:
    rows = []
    topk_path = ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "topk_steering_records.csv"
    topk = pd.read_csv(topk_path)
    topk = topk[topk["top_k_output"].eq(32)].copy()
    topk["source"] = "mlm_topk32"
    topk["top10_jaccard"] = topk["full_vocab_top10_jaccard"]
    rows.append(topk)
    decoder_path = (
        ROOT
        / "experiments"
        / "04_uncertainty_steering"
        / "decoder_main_battery"
        / "outputs"
        / "decoder_steering_records.csv"
    )
    if decoder_path.exists():
        decoder = pd.read_csv(decoder_path)
        decoder["source"] = "decoder_main"
        rows.append(decoder)
    frame = pd.concat(rows, ignore_index=True, sort=False)
    frame = frame[frame["fisher_output_energy"].astype(float) > 0].copy()
    frame["entropy_efficiency"] = frame["abs_delta_entropy"] / frame["fisher_output_energy"]
    frame["varentropy_efficiency"] = frame["abs_delta_varentropy"] / frame["fisher_output_energy"]
    return frame


def equal_output_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = frame.groupby(["source", "direction", "subspace_family", "sign"], as_index=False).agg(
        n=("prompt_id", "count"),
        fisher_output_energy_mean=("fisher_output_energy", "mean"),
        entropy_efficiency_mean=("entropy_efficiency", "mean"),
        varentropy_efficiency_mean=("varentropy_efficiency", "mean"),
        abs_delta_entropy_mean=("abs_delta_entropy", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
        selected_top1_changed_rate=("selected_top1_changed", "mean"),
        top10_jaccard_mean=("top10_jaccard", "mean"),
    )
    indexed = summary.set_index(["source", "subspace_family", "sign", "direction"])
    rows = []
    for key, acc in indexed[indexed.index.get_level_values("direction") == "accessible_ls"].iterrows():
        source, family, sign, _direction = key
        for control in ["random_control", "grad_orthogonal_control", "strict_orthogonal_ls"]:
            control_key = (source, family, sign, control)
            if control_key not in indexed.index:
                continue
            ctrl = indexed.loc[control_key]
            for metric in ["entropy_efficiency_mean", "varentropy_efficiency_mean"]:
                denom = float(ctrl[metric])
                rows.append(
                    {
                        "source": source,
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
    preservation = frame.groupby(["source", "direction", "sign"], as_index=False).agg(
        n=("prompt_id", "count"),
        selected_top1_changed_rate=("selected_top1_changed", "mean"),
        target_correct_changed_rate=("target_correct_changed", "mean"),
        top5_jaccard_mean=("top5_jaccard", "mean"),
        top10_jaccard_mean=("top10_jaccard", "mean"),
        selected_kl_mean=("selected_kl_before_after", "mean"),
    )
    return summary, contrasts, preservation


def baseline_availability() -> pd.DataFrame:
    rows = [
        ("accessible_ls", "implemented", "MLM and decoder equal Fisher-output movement"),
        ("random_control", "implemented", "MLM and decoder equal Fisher-output movement"),
        ("grad_orthogonal_control", "implemented", "MLM and decoder equal Fisher-output movement"),
        ("strict_orthogonal_ls", "implemented_partial", "available in the original MLM full battery, not in decoder main battery"),
        ("entropy_gradient_direction", "diagnostic_predictor", "projected gradient norm is benchmarked for minimal cost; intervention direction not stored in checked-in equal-output records"),
        ("varentropy_gradient_direction", "diagnostic_predictor", "projected gradient norm is benchmarked for minimal cost; intervention direction not stored in checked-in equal-output records"),
        ("max_fisher_output_direction", "not_materialized", "Fisher-output norm is a predictor/control, but max-energy intervention direction is not stored"),
        ("pca_direction", "implemented_as_subspace", "state_pca and delta_pca define routes; individual first PC intervention is not separately stored"),
    ]
    return pd.DataFrame(rows, columns=["baseline", "status", "note"])


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def write_minimal_report(out_dir: Path, args: argparse.Namespace, summary: pd.DataFrame, benchmark: pd.DataFrame, residuals: pd.DataFrame) -> None:
    key = benchmark[
        benchmark["predictor"].isin(
            [
                "rho",
                "rho_random_adjusted",
                "entropy",
                "confidence",
                "varentropy",
                "grad_entropy_proj_norm",
                "grad_varentropy_proj_norm",
                "fisher_output_fro_norm",
                "jacobian_fro_norm",
            ]
        )
    ].sort_values(["source", "outcome", "spearman_with_easy_score"], ascending=[True, True, False])
    lines = [
        "# Test 1: Minimal Intervention Energy",
        "",
        "This test asks the inverse controllability question: given a fixed target uncertainty movement, how much latent intervention energy is needed?",
        "",
        f"- Entropy target: `{args.tau_entropy}`",
        f"- Varentropy target: `{args.tau_varentropy}`",
        f"- Main MLM lens: top-k `{args.top_k_output}`",
        "",
        "Costs are estimated two ways: the main top-k 32 and decoder records use local-linear control costs from equal Fisher-output-energy interventions, while the original MLM full battery contributes an interpolated multi-epsilon grid estimate.",
        "",
        "## Target Summary",
        markdown_table(summary, 80),
        "",
        "## Predictor Benchmark",
        markdown_table(key, 120),
        "",
        "## Residual Rho Effects",
        markdown_table(residuals, 40),
        "",
        "## Files",
        "```text",
        "minimal_energy_cost_records.csv",
        "minimal_energy_predictor_benchmark.csv",
        "minimal_energy_residual_effects.csv",
        "minimal_energy_target_summary.csv",
        "baseline_availability.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_equal_report(out_dir: Path, summary: pd.DataFrame, contrasts: pd.DataFrame, preservation: pd.DataFrame) -> None:
    lines = [
        "# Test 2: Equal-Output-Movement Steering",
        "",
        "This test asks whether accessible directions change uncertainty more per unit Fisher-output movement than matched alternatives.",
        "",
        "All rows use matched `||F^{1/2}J delta z||`; the main readout is uncertainty movement divided by that output movement.",
        "",
        "## Direction Summary",
        markdown_table(summary.sort_values(["source", "direction", "subspace_family", "sign"]), 120),
        "",
        "## Accessible Vs Controls",
        markdown_table(contrasts.sort_values(["source", "subspace_family", "sign", "control", "metric"]), 120),
        "",
        "## Preservation",
        markdown_table(preservation.sort_values(["source", "direction", "sign"]), 80),
        "",
        "## Files",
        "```text",
        "equal_output_efficiency_records.csv",
        "equal_output_direction_summary.csv",
        "equal_output_control_contrasts.csv",
        "equal_output_preservation_summary.csv",
        "baseline_availability.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readmes_and_configs(args: argparse.Namespace) -> None:
    command = (
        "python scripts\\run_control_cost_and_equal_output_tests.py "
        f"--tau-entropy {args.tau_entropy} --tau-varentropy {args.tau_varentropy} "
        f"--top-k-output {args.top_k_output} --seed {args.seed}"
    )
    for folder, title in [
        (args.minimal_out, "Minimal Intervention Energy"),
        (args.equal_out, "Equal-Output-Movement Steering"),
    ]:
        (folder / "README.md").write_text(
            "\n".join(
                [
                    f"# {title}",
                    "",
                    "Experiment 4 sub-battery organized around operational controllability.",
                    "",
                    "## Reproduce",
                    "",
                    "```powershell",
                    command,
                    "```",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        payload = {
            "command": command,
            "seed": args.seed,
            "tau_entropy": args.tau_entropy,
            "tau_varentropy": args.tau_varentropy,
            "top_k_output": args.top_k_output,
        }
        (folder / "config" / "reproduce.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for folder in [args.minimal_out, args.equal_out]:
        for child in ["outputs", "reports", "config"]:
            (folder / child).mkdir(parents=True, exist_ok=True)

    minimal = pd.concat(
        [build_minimal_energy_records(args), build_full_grid_cost_records(args), build_decoder_cost_records(args)],
        ignore_index=True,
        sort=False,
    )
    minimal_summary = target_summary(minimal)
    entropy_bench = predictor_benchmark(minimal, "neg_log_cost_entropy_tau")
    var_bench = predictor_benchmark(minimal, "neg_log_cost_varentropy_tau")
    predictor_table = pd.concat([entropy_bench, var_bench], ignore_index=True)
    residuals = residual_effects(minimal)
    availability = baseline_availability()

    minimal.to_csv(args.minimal_out / "outputs" / "minimal_energy_cost_records.csv", index=False)
    predictor_table.to_csv(args.minimal_out / "outputs" / "minimal_energy_predictor_benchmark.csv", index=False)
    residuals.to_csv(args.minimal_out / "outputs" / "minimal_energy_residual_effects.csv", index=False)
    minimal_summary.to_csv(args.minimal_out / "outputs" / "minimal_energy_target_summary.csv", index=False)
    availability.to_csv(args.minimal_out / "outputs" / "baseline_availability.csv", index=False)
    write_minimal_report(args.minimal_out, args, minimal_summary, predictor_table, residuals)

    equal = load_equal_output_records()
    direction_summary, contrasts, preservation = equal_output_summaries(equal)
    equal.to_csv(args.equal_out / "outputs" / "equal_output_efficiency_records.csv", index=False)
    direction_summary.to_csv(args.equal_out / "outputs" / "equal_output_direction_summary.csv", index=False)
    contrasts.to_csv(args.equal_out / "outputs" / "equal_output_control_contrasts.csv", index=False)
    preservation.to_csv(args.equal_out / "outputs" / "equal_output_preservation_summary.csv", index=False)
    availability.to_csv(args.equal_out / "outputs" / "baseline_availability.csv", index=False)
    write_equal_report(args.equal_out, direction_summary, contrasts, preservation)

    write_readmes_and_configs(args)
    print(args.minimal_out)
    print((args.minimal_out / "reports" / "report.md").read_text(encoding="utf-8"))
    print(args.equal_out)
    print((args.equal_out / "reports" / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
