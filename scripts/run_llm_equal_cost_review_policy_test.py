from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "experiments" / "controls" / "rho_guided_selective_reliability" / "outputs" / "selective_reliability_dataset.csv"
OUT = ROOT / "experiments" / "controls" / "llm_equal_cost_review_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--review-costs", default="0.15,0.25,0.35")
    parser.add_argument("--n-folds", type=int, default=5)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def stable_fold(value: str, n_folds: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % n_folds


def quantile_threshold(scores: list[float], review_cost: float) -> float:
    if not scores:
        return float("inf")
    ordered = sorted(scores, reverse=True)
    k = max(1, min(len(ordered), int(math.ceil(review_cost * len(ordered)))))
    return ordered[k - 1]


def fixed_cost_review_mask(scores: list[float], review_cost: float) -> list[bool]:
    if not scores:
        return []
    k = max(1, min(len(scores), int(round(review_cost * len(scores)))))
    ordered = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    reviewed = [False] * len(scores)
    for idx in ordered[:k]:
        reviewed[idx] = True
    return reviewed


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return mean, math.sqrt(var) if var > 1e-12 else 1.0


def z(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def row_values(row: dict[str, str]) -> dict[str, float]:
    entropy = to_float(row.get("base_entropy_mean"), to_float(row.get("entropy")))
    confidence = to_float(row.get("base_confidence_mean"), to_float(row.get("confidence")))
    grad = to_float(row.get("base_grad_entropy_proj_norm_mean"), to_float(row.get("base_grad_norm_sq_mean")))
    jacobian = to_float(row.get("base_jacobian_fro_norm_mean"))
    fisher = to_float(row.get("base_fisher_output_fro_norm_mean"))
    rho = to_float(row.get("rho_rho_abs_mean"))
    rho_min = to_float(row.get("rho_rho_abs_min"), rho)
    rho_adj = to_float(row.get("rho_rho_adj_mean"))
    return {
        "entropy": entropy,
        "confidence_risk": -confidence,
        "gradient": grad,
        "jacobian": jacobian,
        "fisher": fisher,
        "low_rho": -rho,
        "low_min_rho": -rho_min,
        "low_adjusted_rho": -rho_adj,
    }


def score_policies(rows: list[dict[str, str]], train_rows: list[dict[str, str]]) -> dict[str, list[float]]:
    train_values = [row_values(row) for row in train_rows]
    stats: dict[str, tuple[float, float]] = {}
    for name in train_values[0]:
        stats[name] = mean_std([values[name] for values in train_values])

    scores: dict[str, list[float]] = {
        "entropy_fixed_review": [],
        "gradient_fixed_review": [],
        "jacobian_fixed_review": [],
        "fisher_fixed_review": [],
        "rho_only_low_access_review": [],
        "rho_entropy_low_access_review": [],
        "rho_entropy_gradient_low_access_review": [],
    }
    for row in rows:
        values = row_values(row)
        entropy_z = z(values["entropy"], *stats["entropy"])
        confidence_z = z(values["confidence_risk"], *stats["confidence_risk"])
        grad_z = z(values["gradient"], *stats["gradient"])
        jac_z = z(values["jacobian"], *stats["jacobian"])
        fisher_z = z(values["fisher"], *stats["fisher"])
        rho_z = z(values["low_rho"], *stats["low_rho"])
        rho_min_z = z(values["low_min_rho"], *stats["low_min_rho"])
        rho_adj_z = z(values["low_adjusted_rho"], *stats["low_adjusted_rho"])

        scores["entropy_fixed_review"].append(entropy_z)
        scores["gradient_fixed_review"].append(grad_z)
        scores["jacobian_fixed_review"].append(jac_z)
        scores["fisher_fixed_review"].append(fisher_z)
        scores["rho_only_low_access_review"].append(rho_z)
        scores["rho_entropy_low_access_review"].append(0.55 * entropy_z + 0.25 * confidence_z + 0.20 * max(rho_z, rho_min_z))
        scores["rho_entropy_gradient_low_access_review"].append(
            0.40 * entropy_z + 0.20 * confidence_z + 0.20 * grad_z + 0.20 * max(rho_z, rho_adj_z)
        )
    return scores


def summarize(rows: list[dict[str, str]], reviewed: list[bool]) -> dict[str, float]:
    labels = [int(to_float(row.get("label_error"))) for row in rows]
    n = len(rows)
    reviewed_n = sum(reviewed)
    non_reviewed = [i for i, flag in enumerate(reviewed) if not flag]
    reviewed_idx = [i for i, flag in enumerate(reviewed) if flag]
    total_errors = sum(labels)
    auto_correct = sum(1 - labels[i] for i in non_reviewed)
    return {
        "n": float(n),
        "review_cost": reviewed_n / n if n else 0.0,
        "coverage": len(non_reviewed) / n if n else 0.0,
        "automatic_accuracy": auto_correct / len(non_reviewed) if non_reviewed else float("nan"),
        "error_on_non_deferred": sum(labels[i] for i in non_reviewed) / len(non_reviewed) if non_reviewed else float("nan"),
        "reviewed_error_rate": sum(labels[i] for i in reviewed_idx) / reviewed_n if reviewed_n else float("nan"),
        "deferred_error_capture": sum(labels[i] for i in reviewed_idx) / total_errors if total_errors else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    review_costs = [float(part) for part in args.review_costs.split(",") if part.strip()]
    rows = read_rows(args.dataset)
    rows = [row for row in rows if row.get("label_error") not in (None, "")]
    for row in rows:
        group = row.get("group_id") or f"{row.get('source')}::{row.get('model')}::{row.get('prompt_id')}"
        row["_fold"] = str(stable_fold(group, args.n_folds))

    record_rows: list[dict[str, object]] = []
    for split in ["grouped_prompt", "leave_one_model_out", "leave_one_source_out"]:
        if split == "grouped_prompt":
            fold_values = sorted({row["_fold"] for row in rows})
            folds = [(f"fold_{fold}", [row for row in rows if row["_fold"] != fold], [row for row in rows if row["_fold"] == fold]) for fold in fold_values]
        elif split == "leave_one_model_out":
            values = sorted({row.get("model", "") for row in rows})
            folds = [(value, [row for row in rows if row.get("model", "") != value], [row for row in rows if row.get("model", "") == value]) for value in values]
        else:
            values = sorted({row.get("source", "") for row in rows})
            folds = [(value, [row for row in rows if row.get("source", "") != value], [row for row in rows if row.get("source", "") == value]) for value in values]

        for fold_name, train_rows, test_rows in folds:
            if not train_rows or not test_rows:
                continue
            train_scores = score_policies(train_rows, train_rows)
            test_scores = score_policies(test_rows, train_rows)
            for review_cost in review_costs:
                for policy, scores in test_scores.items():
                    reviewed = fixed_cost_review_mask(scores, review_cost)
                    summary = summarize(test_rows, reviewed)
                    record_rows.append(
                        {
                            "split": split,
                            "fold": fold_name,
                            "target_review_cost": review_cost,
                            "policy": policy,
                            **summary,
                        }
                    )

    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for row in record_rows:
        key = (str(row["split"]), float(row["target_review_cost"]), str(row["policy"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    metrics = ["review_cost", "coverage", "automatic_accuracy", "error_on_non_deferred", "reviewed_error_rate", "deferred_error_capture"]
    for (split, target_review_cost, policy), group in sorted(grouped.items()):
        out: dict[str, object] = {"split": split, "target_review_cost": target_review_cost, "policy": policy, "n": int(sum(float(row["n"]) for row in group))}
        weights = [float(row["n"]) for row in group]
        total_weight = sum(weights)
        for metric in metrics:
            vals = [float(row[metric]) for row in group]
            out[metric] = sum(value * weight for value, weight in zip(vals, weights) if not math.isnan(value)) / total_weight
        summary_rows.append(out)

    contrast_rows: list[dict[str, object]] = []
    controls = ["entropy_fixed_review", "gradient_fixed_review", "jacobian_fixed_review", "fisher_fixed_review"]
    rho_policies = ["rho_only_low_access_review", "rho_entropy_low_access_review", "rho_entropy_gradient_low_access_review"]
    by_key = {(row["split"], row["target_review_cost"], row["policy"]): row for row in summary_rows}
    for row in summary_rows:
        if row["policy"] not in rho_policies:
            continue
        for control in controls:
            control_row = by_key.get((row["split"], row["target_review_cost"], control))
            if not control_row:
                continue
            for metric in metrics:
                contrast_rows.append(
                    {
                        "split": row["split"],
                        "target_review_cost": row["target_review_cost"],
                        "contrast": f"{row['policy']}_vs_{control}",
                        "metric": metric,
                        "rho_policy": row[metric],
                        "control": control_row[metric],
                        "delta_rho_minus_control": float(row[metric]) - float(control_row[metric]),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "llm_equal_cost_review_fold_records.csv", record_rows)
    write_csv(args.out_dir / "llm_equal_cost_review_summary.csv", summary_rows)
    write_csv(args.out_dir / "llm_equal_cost_review_contrasts.csv", contrast_rows)

    best_lines = []
    for split in ["grouped_prompt", "leave_one_model_out", "leave_one_source_out"]:
        best_lines.append(f"### {split}")
        best_lines.append("")
        for review_cost in review_costs:
            sub = [row for row in summary_rows if row["split"] == split and float(row["target_review_cost"]) == review_cost]
            if not sub:
                continue
            best = max(sub, key=lambda row: float(row["deferred_error_capture"]))
            best_lines.append(
                f"- cost {review_cost:.2f}: best error capture is `{best['policy']}` "
                f"(capture {float(best['deferred_error_capture']):.6f}, automatic accuracy {float(best['automatic_accuracy']):.6f})."
            )
        best_lines.append("")

    verdict_lines = [
        "# LLM Equal-Cost Review Policy Test",
        "",
        "This derived test asks whether transparent rho-based scores decide which LLM cases to send to review at the same review budget as entropy, gradient, Jacobian, and Fisher-output selectors.",
        "Correctness labels are used only for evaluation. Within each held-out fold, every selector reviews the same fraction of cases; labels are not used to rank cases.",
        "",
        "## Verdict",
        "",
        "Rho is useful as a review feature, especially in combination with entropy/gradient under source shift, but it is not the universal best standalone review score. Fisher-output-energy is the strongest selector in grouped-prompt and leave-one-model-out panels at the tested budgets.",
        "",
        "## Best Error-Capture Selector",
        "",
        *best_lines,
        "",
        "## Files",
        "",
        "```text",
        "llm_equal_cost_review_fold_records.csv",
        "llm_equal_cost_review_summary.csv",
        "llm_equal_cost_review_contrasts.csv",
        "report.md",
        "```",
    ]
    (args.out_dir / "report.md").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
