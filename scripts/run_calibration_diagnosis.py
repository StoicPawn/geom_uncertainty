from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accessible_varentropy.metrics import softmax_np
from run_uncertainty_steering_full_battery import (
    build_cases,
    build_records,
    fisher_geometry,
    full_logits,
    least_squares,
    mlm_head_jacobian_fast,
    model_subspaces,
    normalize,
    parse_layers,
    projection,
    topk_distribution,
)


OUT = ROOT / "applications" / "03_calibration_diagnosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="distilbert-base-uncased,bert-base-uncased,google/bert_uncased_L-2_H-128_A-2",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--subspace-ks", default="8")
    parser.add_argument("--random-subspaces", type=int, default=1)
    parser.add_argument("--max-prompts-per-task", type=int, default=10)
    parser.add_argument("--output-eps", type=float, default=0.05)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260532)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    for rel in ["config", "outputs", "reports"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def full_probs(logits: np.ndarray) -> np.ndarray:
    return softmax_np(np.asarray(logits, dtype=np.float64)[None, :])[0]


def topn_jaccard(p: np.ndarray, q: np.ndarray, n: int = 10) -> float:
    k = min(n, len(p), len(q))
    left = set(np.argsort(-p)[:k].tolist())
    right = set(np.argsort(-q)[:k].tolist())
    return float(len(left & right) / max(1, len(left | right)))


def calibration_metrics(frame: pd.DataFrame, bins: int) -> dict[str, float]:
    if frame.empty:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "ece": float("nan"),
            "nll": float("nan"),
            "brier": float("nan"),
            "confidence": float("nan"),
        }
    confidence = frame["confidence"].to_numpy(dtype=np.float64)
    correct = frame["correct"].to_numpy(dtype=np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        weight = float(np.mean(mask))
        ece += weight * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return {
        "n": int(len(frame)),
        "accuracy": float(np.mean(correct)),
        "ece": float(ece),
        "nll": float(np.mean(frame["nll"].to_numpy(dtype=np.float64))),
        "brier": float(np.mean(frame["brier"].to_numpy(dtype=np.float64))),
        "confidence": float(np.mean(confidence)),
    }


def single_metrics(logits: np.ndarray, target_id: int) -> dict[str, float | int]:
    probs = full_probs(logits)
    pred = int(np.argmax(probs))
    target_prob = float(probs[int(target_id)])
    confidence = float(probs[pred])
    brier = float(np.sum(probs * probs) - 2.0 * target_prob + 1.0)
    return {
        "pred_id": pred,
        "correct": int(pred == int(target_id)),
        "confidence": confidence,
        "target_prob": target_prob,
        "nll": float(-np.log(max(target_prob, 1e-12))),
        "brier": brier,
        "entropy_full": float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))),
    }


def compare_probs(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    return {
        "full_vocab_kl": float(np.sum(before * (np.log(np.clip(before, 1e-12, 1.0)) - np.log(np.clip(after, 1e-12, 1.0))))),
        "full_vocab_l1": float(np.sum(np.abs(after - before))),
        "full_vocab_top10_jaccard": topn_jaccard(before, after, 10),
        "full_vocab_top5_jaccard": topn_jaccard(before, after, 5),
    }


def row_key(row: pd.Series) -> tuple[int, int, str, int, int]:
    return (
        int(row["prompt_id"]),
        int(row["layer"]),
        str(row["subspace_family"]),
        int(row["subspace_k"]),
        int(row["rep"]),
    )


def select_routes(score_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if score_rows.empty:
        return pd.DataFrame()
    for (model, prompt_id), group in score_rows.groupby(["model", "prompt_id"], dropna=False):
        structured = group[group["subspace_family"].isin(["state_pca", "delta_pca"])].copy()
        random_group = group[group["subspace_family"].eq("random")].copy()
        if not structured.empty:
            high = structured.sort_values("rho", ascending=False).iloc[0].to_dict()
            high["route_role"] = "high_rho_route"
            rows.append(high)
            low = structured.sort_values("rho", ascending=True).iloc[0].to_dict()
            low["route_role"] = "low_rho_route"
            rows.append(low)
        if not random_group.empty:
            random_route = random_group.sort_values("rho", ascending=False).iloc[0].to_dict()
            random_route["route_role"] = "random_route"
            rows.append(random_route)
    return pd.DataFrame(rows)


def run_model(args: argparse.Namespace, model_name: str, cases) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"calibration_loading {model_name}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except ValueError:
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True, attn_implementation="eager")
    model.eval()
    prompt_table, records = build_records(tokenizer, model, cases, args.top_k)
    if prompt_table.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    prompt_table = prompt_table[prompt_table["target_id"].notna() & ~prompt_table["task"].eq("ambiguous_open")].copy()
    if prompt_table.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    prompt_ids = set(prompt_table["prompt_id"].astype(int).tolist())
    records = [record for record in records if int(record["prompt_id"]) in prompt_ids]
    prompt_lookup = prompt_table.set_index("prompt_id")
    n_layers = len(records[0]["hidden_states"]) - 1
    layers = [layer for layer in parse_layers(args.layers, n_layers) if 1 <= layer <= n_layers]
    ks = [int(part) for part in args.subspace_ks.split(",") if part.strip()]

    clean_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    route_payloads: dict[tuple[int, int, str, int, int], dict[str, object]] = {}

    for layer in layers:
        print(f"calibration_layer {model_name} {layer}", flush=True)
        subspaces = model_subspaces(records, layer, ks, args.random_subspaces, args.seed)
        for rec_idx, record in enumerate(records):
            meta = prompt_lookup.loc[int(record["prompt_id"])]
            target_id = int(meta["target_id"])
            hidden = record["hidden_states"][layer][0, int(record["mask_index"]), :].detach()
            logits = full_logits(model, hidden)
            probs_full = full_probs(logits)
            clean_metric = single_metrics(logits, target_id)
            token_ids, probs = topk_distribution(logits, args.top_k)
            selected_logits = logits[token_ids]
            geometry = fisher_geometry(probs)
            jac = mlm_head_jacobian_fast(model, hidden, token_ids)
            clean_rows.append(
                {
                    "model": model_name,
                    "prompt_id": int(record["prompt_id"]),
                    "prompt": meta["prompt"],
                    "task": meta["task"],
                    "topic": meta["topic"],
                    "layer": int(layer),
                    "target": meta["target"],
                    "target_id": target_id,
                    "rho_scope": "layer_prompt",
                    **clean_metric,
                }
            )
            for family, k, rep, basis in subspaces:
                jb = jac @ basis
                whitened = geometry.sqrt_fisher @ jb
                w_acc, rank = projection(whitened, geometry.w)
                rho = float((w_acc @ w_acc) / max(float(geometry.w @ geometry.w), 1e-12))
                z_acc = normalize(least_squares(whitened, w_acc))
                output_energy_unit = float(np.linalg.norm(whitened @ z_acc))
                grad_z = whitened.T @ geometry.w
                first_order = float(grad_z @ z_acc)
                route_key = (int(record["prompt_id"]), int(layer), family, int(k), int(rep))
                route_payloads[route_key] = {
                    "hidden": hidden,
                    "basis": basis,
                    "z_acc": z_acc,
                    "output_energy_unit": output_energy_unit,
                    "first_order": first_order,
                    "target_id": target_id,
                    "logits_before": logits,
                    "probs_full_before": probs_full,
                    "clean_metric": clean_metric,
                }
                score_rows.append(
                    {
                        "model": model_name,
                        "prompt_id": int(record["prompt_id"]),
                        "prompt": meta["prompt"],
                        "target": meta["target"],
                        "task": meta["task"],
                        "topic": meta["topic"],
                        "layer": int(layer),
                        "subspace_family": family,
                        "subspace_k": int(k),
                        "rep": int(rep),
                        "rho": rho,
                        "v_access": float(w_acc @ w_acc),
                        "v_inaccess": float((geometry.w - w_acc) @ (geometry.w - w_acc)),
                        "rank_whitened_jb": int(rank),
                        "output_energy_unit": output_energy_unit,
                        "first_order_entropy": first_order,
                        "clean_correct": int(clean_metric["correct"]),
                        "clean_confidence": float(clean_metric["confidence"]),
                        "clean_nll": float(clean_metric["nll"]),
                        "clean_brier": float(clean_metric["brier"]),
                    }
                )

    clean_frame = pd.DataFrame(clean_rows)
    score_frame = pd.DataFrame(score_rows)
    route_frame = select_routes(score_frame)
    intervention_rows: list[dict[str, object]] = []
    for _, route in route_frame.iterrows():
        key = row_key(route)
        payload = route_payloads[key]
        output_energy_unit = float(payload["output_energy_unit"])
        z_acc = payload["z_acc"]
        if output_energy_unit < 1e-12 or float(np.linalg.norm(z_acc)) < 1e-12:
            continue
        clean_metric = payload["clean_metric"]
        first_order = float(payload["first_order"])
        entropy_positive_sign = 1.0 if first_order >= 0.0 else -1.0
        calibration_sign = -entropy_positive_sign if int(clean_metric["correct"]) == 1 else entropy_positive_sign
        step_scale = float(args.output_eps) / output_energy_unit
        hidden_direction = payload["basis"] @ z_acc
        hidden_after = payload["hidden"] + torch.as_tensor(
            calibration_sign * step_scale * hidden_direction,
            dtype=payload["hidden"].dtype,
        )
        logits_after = full_logits(model, hidden_after)
        metric_after = single_metrics(logits_after, int(payload["target_id"]))
        probs_after = full_probs(logits_after)
        drift = compare_probs(payload["probs_full_before"], probs_after)
        intervention_rows.append(
            {
                "model": model_name,
                "prompt_id": int(route["prompt_id"]),
                "prompt": route["prompt"],
                "target": route["target"],
                "task": route["task"],
                "topic": route["topic"],
                "layer": int(route["layer"]),
                "subspace_family": route["subspace_family"],
                "subspace_k": int(route["subspace_k"]),
                "rep": int(route["rep"]),
                "route_role": route["route_role"],
                "rho": float(route["rho"]),
                "fisher_output_energy": float(args.output_eps),
                "latent_step_norm": step_scale,
                "calibration_policy": "correct_decrease_entropy_error_increase_entropy",
                "clean_correct": int(clean_metric["correct"]),
                "clean_confidence": float(clean_metric["confidence"]),
                "clean_target_prob": float(clean_metric["target_prob"]),
                "clean_nll": float(clean_metric["nll"]),
                "clean_brier": float(clean_metric["brier"]),
                "after_correct": int(metric_after["correct"]),
                "after_confidence": float(metric_after["confidence"]),
                "after_target_prob": float(metric_after["target_prob"]),
                "after_nll": float(metric_after["nll"]),
                "after_brier": float(metric_after["brier"]),
                "delta_confidence": float(metric_after["confidence"]) - float(clean_metric["confidence"]),
                "delta_target_prob": float(metric_after["target_prob"]) - float(clean_metric["target_prob"]),
                "delta_nll": float(metric_after["nll"]) - float(clean_metric["nll"]),
                "delta_brier": float(metric_after["brier"]) - float(clean_metric["brier"]),
                "accuracy_changed": int(int(metric_after["correct"]) != int(clean_metric["correct"])),
                "accuracy_loss": int(int(clean_metric["correct"]) == 1 and int(metric_after["correct"]) == 0),
                **drift,
            }
        )
    return clean_frame.assign(model=model_name), score_frame, route_frame, pd.DataFrame(intervention_rows)


def summarize(clean: pd.DataFrame, interventions: pd.DataFrame, bins: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if interventions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    clean_one = (
        clean.sort_values(["model", "prompt_id", "layer"])
        .drop_duplicates(["model", "prompt_id"])
        .rename(columns={"pred_id": "pred_id_clean"})
    )
    before = clean_one[["model", "prompt_id", "confidence", "correct", "nll", "brier"]].copy()
    before["route_role"] = "clean_baseline"
    base = calibration_metrics(before, bins)
    rows.append({"route_role": "clean_baseline", **base})
    for route_role, group in interventions.groupby("route_role"):
        after = pd.DataFrame(
            {
                "confidence": group["after_confidence"],
                "correct": group["after_correct"],
                "nll": group["after_nll"],
                "brier": group["after_brier"],
            }
        )
        metrics = calibration_metrics(after, bins)
        rows.append(
            {
                "route_role": route_role,
                **metrics,
                "mean_rho": float(group["rho"].mean()),
                "mean_full_vocab_kl": float(group["full_vocab_kl"].mean()),
                "mean_top10_jaccard": float(group["full_vocab_top10_jaccard"].mean()),
                "accuracy_loss_rate": float(group["accuracy_loss"].mean()),
                "mean_delta_nll": float(group["delta_nll"].mean()),
                "mean_delta_brier": float(group["delta_brier"].mean()),
                "mean_delta_target_prob": float(group["delta_target_prob"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    clean_metrics = summary[summary["route_role"].eq("clean_baseline")].iloc[0].to_dict()
    effect_rows = []
    for _, row in summary[~summary["route_role"].eq("clean_baseline")].iterrows():
        ece_improvement = float(clean_metrics["ece"]) - float(row["ece"])
        nll_improvement = float(clean_metrics["nll"]) - float(row["nll"])
        brier_improvement = float(clean_metrics["brier"]) - float(row["brier"])
        effect_rows.append(
            {
                "route_role": row["route_role"],
                "ece_improvement": ece_improvement,
                "nll_improvement": nll_improvement,
                "brier_improvement": brier_improvement,
                "accuracy_delta": float(row["accuracy"]) - float(clean_metrics["accuracy"]),
                "accuracy_loss_rate": float(row.get("accuracy_loss_rate", np.nan)),
                "calibration_steerability_score": ece_improvement - float(row.get("accuracy_loss_rate", 0.0)),
                "mean_full_vocab_kl": float(row.get("mean_full_vocab_kl", np.nan)),
                "mean_top10_jaccard": float(row.get("mean_top10_jaccard", np.nan)),
            }
        )
    effects = pd.DataFrame(effect_rows)
    contrast_rows = []
    if {"high_rho_route", "low_rho_route"}.issubset(set(interventions["route_role"])):
        merged = interventions.pivot_table(
            index=["model", "prompt_id"],
            columns="route_role",
            values=["after_nll", "after_brier", "after_correct", "full_vocab_kl", "full_vocab_top10_jaccard", "delta_target_prob"],
            aggfunc="mean",
        )
        merged.columns = [f"{metric}_{role}" for metric, role in merged.columns]
        merged = merged.reset_index()
        for metric, lower_is_better in [
            ("after_nll", True),
            ("after_brier", True),
            ("full_vocab_kl", True),
            ("delta_target_prob", False),
        ]:
            high = f"{metric}_high_rho_route"
            low = f"{metric}_low_rho_route"
            if high not in merged or low not in merged:
                continue
            diff = merged[high] - merged[low]
            high_better = diff < 0 if lower_is_better else diff > 0
            contrast_rows.append(
                {
                    "comparison": "high_rho_route_vs_low_rho_route",
                    "metric": metric,
                    "n_pairs": int(diff.notna().sum()),
                    "high_minus_low_mean": float(diff.mean()),
                    "high_better_rate": float(high_better.mean()),
                }
            )
    return summary, effects, pd.DataFrame(contrast_rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + df.head(max_rows).to_string(index=False) + "\n```"


def write_readme(out_dir: Path, status: str) -> None:
    (out_dir / "README.md").write_text(
        "# Application 3: Calibration Diagnosis\n\n"
        "Separate application for asking whether miscalibration is internally steerable through high-accessibility routes.\n\n"
        "## Question\n\n"
        "Classical calibration asks whether probabilities match empirical correctness. This test asks whether the calibration error can be moved through an internal route with low accuracy loss.\n\n"
        "## Main Metrics\n\n"
        "- ECE, NLL, Brier score, accuracy.\n"
        "- Full-vocabulary KL drift and top-k preservation.\n"
        "- Calibration steerability score: ECE improvement minus accuracy-loss rate.\n\n"
        f"## Status\n\n{status}\n\n"
        "## Reproduce\n\n"
        "```powershell\n"
        "python scripts\\run_calibration_diagnosis.py --max-prompts-per-task 10 --top-k 32 --subspace-ks 8 --output-eps 0.05 --seed 20260532\n"
        "```\n",
        encoding="utf-8",
    )


def write_report(out_dir: Path, skipped: pd.DataFrame) -> None:
    outputs = out_dir / "outputs"
    parts = ["# Application 3: Calibration Diagnosis\n"]
    parts.append("This application tests whether high-accessibility routes allow local calibration steering with lower accuracy loss and answer-neighborhood drift.\n")
    for title, filename in [
        ("Calibration Summary", "calibration_summary.csv"),
        ("Calibration Effects", "calibration_effects.csv"),
        ("High-Rho Route Contrasts", "calibration_route_contrasts.csv"),
        ("Route Scores", "route_scores.csv"),
    ]:
        path = outputs / filename
        parts.append(f"## {title}\n")
        if path.exists():
            parts.append(markdown_table(pd.read_csv(path)))
        else:
            parts.append("```text\n(missing)\n```")
    parts.append("## Skipped Models\n")
    parts.append(markdown_table(skipped))
    (out_dir / "reports" / "report.md").write_text("\n\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dirs(args.out_dir)
    config = {
        "application": "calibration diagnosis",
        "command": (
            "python scripts\\run_calibration_diagnosis.py --max-prompts-per-task 10 "
            "--top-k 32 --subspace-ks 8 --output-eps 0.05 --seed 20260532"
        ),
        "seed": args.seed,
        "models": [m.strip() for m in args.models.split(",") if m.strip()],
        "top_k": args.top_k,
        "subspace_ks": args.subspace_ks,
        "output_eps": args.output_eps,
        "calibration_policy": "correct examples decrease entropy; incorrect examples increase entropy",
    }
    (args.out_dir / "config" / "reproduce.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_readme(args.out_dir, "Completed on the local model cache when this script finishes successfully.")
    cases = build_cases(args.seed, args.max_prompts_per_task)
    all_clean, all_scores, all_routes, all_interventions, skipped_rows = [], [], [], [], []
    for model_name in config["models"]:
        try:
            clean, scores, routes, interventions = run_model(args, str(model_name), cases)
        except Exception as exc:  # noqa: BLE001
            skipped_rows.append({"model": model_name, "reason": repr(exc)})
            continue
        if not clean.empty:
            all_clean.append(clean)
        if not scores.empty:
            all_scores.append(scores)
        if not routes.empty:
            all_routes.append(routes)
        if not interventions.empty:
            all_interventions.append(interventions)
    clean_df = pd.concat(all_clean, ignore_index=True) if all_clean else pd.DataFrame()
    scores_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    routes_df = pd.concat(all_routes, ignore_index=True) if all_routes else pd.DataFrame()
    interventions_df = pd.concat(all_interventions, ignore_index=True) if all_interventions else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows)
    summary, effects, contrasts = summarize(clean_df, interventions_df, args.ece_bins)

    out = args.out_dir / "outputs"
    clean_df.to_csv(out / "clean_calibration_records.csv", index=False)
    scores_df.to_csv(out / "route_scores.csv", index=False)
    routes_df.to_csv(out / "selected_routes.csv", index=False)
    interventions_df.to_csv(out / "calibration_intervention_records.csv", index=False)
    summary.to_csv(out / "calibration_summary.csv", index=False)
    effects.to_csv(out / "calibration_effects.csv", index=False)
    contrasts.to_csv(out / "calibration_route_contrasts.csv", index=False)
    skipped.to_csv(out / "skipped_models.csv", index=False)
    write_report(args.out_dir, skipped)


if __name__ == "__main__":
    main()
