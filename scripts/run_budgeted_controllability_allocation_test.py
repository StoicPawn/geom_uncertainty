from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "05_controllability_mapping" / "outputs" / "intervention_points.csv"
OUT = ROOT / "experiments" / "05_controllability_mapping" / "budgeted_allocation_policy"

NUMERIC_CONTROLS = [
    "entropy",
    "varentropy",
    "confidence",
    "margin",
    "projected_gradient_norm",
    "jacobian_norm",
    "fisher_output_energy",
    "unit_fisher_output_energy",
    "subspace_dim",
    "layer",
]
CATEGORICAL_CONTROLS = ["model", "task", "route_family"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--budget-mode", default="same_fisher_output_energy")
    parser.add_argument("--budgets", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--drift-cap", type=float, default=0.00025)
    parser.add_argument("--tau-varentropy", type=float, default=0.01)
    parser.add_argument("--topk-threshold", type=float, default=0.90)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def stable_float(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:15], 16) / float(16**15 - 1)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def example_id(row: dict[str, object]) -> str:
    return "::".join(
        [
            str(row["model"]),
            str(row["task"]),
            str(row["item_id"]),
            str(row["budget_mode"]),
            str(row["eps"]),
        ]
    )


def cluster_id(row: dict[str, object]) -> str:
    return "::".join([str(row["model"]), str(row["task"]), str(row["item_id"])])


def route_id(row: dict[str, object]) -> str:
    return "::".join(
        [
            str(row["layer"]),
            str(row["route_family"]),
            str(row["subspace_dim"]),
            str(row["eps"]),
        ]
    )


def read_rows(path: Path, budget_mode: str, drift_cap: float, tau_varentropy: float, topk_threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    numeric_fields = set(NUMERIC_CONTROLS) | {
        "rho",
        "rank",
        "eps",
        "applied_fisher_energy",
        "uncertainty_movement",
        "varentropy_movement",
        "delta_entropy",
        "drift",
        "top1_preserved",
        "topk_jaccard",
        "safe_movement",
        "base_confidence",
    }
    with path.open(newline="") as handle:
        for idx, raw in enumerate(csv.DictReader(handle)):
            if budget_mode != "all" and raw.get("budget_mode") != budget_mode:
                continue
            row: dict[str, object] = dict(raw)
            row["row_id"] = idx
            for field in numeric_fields:
                if field in row:
                    row[field] = to_float(str(row[field]))
            row["abs_delta_entropy"] = abs(float(row.get("delta_entropy", 0.0)))
            row["abs_delta_varentropy"] = abs(float(row.get("varentropy_movement", 0.0)))
            row["safe_movement"] = 1.0 if float(row.get("safe_movement", 0.0)) >= 0.5 else 0.0
            row["safe_varentropy_movement"] = (
                1.0
                if row["abs_delta_varentropy"] >= tau_varentropy
                and float(row.get("drift", 0.0)) <= drift_cap
                and float(row.get("topk_jaccard", 0.0)) >= topk_threshold
                else 0.0
            )
            row["example_id"] = example_id(row)
            row["cluster_id"] = cluster_id(row)
            row["route_id"] = route_id(row)
            rows.append(row)
    return rows


def make_grouped_folds(rows: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    clusters = sorted({str(row["cluster_id"]) for row in rows})
    fold_by_cluster = {cluster: int(stable_float(f"{seed}:fold:{cluster}") * 5) for cluster in clusters}
    folds = []
    for fold in range(5):
        train = [i for i, row in enumerate(rows) if fold_by_cluster[str(row["cluster_id"])] != fold]
        test = [i for i, row in enumerate(rows) if fold_by_cluster[str(row["cluster_id"])] == fold]
        if train and test:
            folds.append({"split": "grouped_example_5fold", "fold": f"fold_{fold}", "train": train, "test": test})
    return folds


def make_leave_one_folds(rows: list[dict[str, object]], field: str, split_name: str) -> list[dict[str, object]]:
    values = sorted({str(row[field]) for row in rows})
    folds = []
    for value in values:
        train = [i for i, row in enumerate(rows) if str(row[field]) != value]
        test = [i for i, row in enumerate(rows) if str(row[field]) == value]
        if train and test:
            folds.append({"split": split_name, "fold": value.replace("/", "__"), "train": train, "test": test})
    return folds


def mean_std(rows: list[dict[str, object]], indices: list[int], features: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = [float(rows[i].get(feature, 0.0)) for i in indices]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        stats[feature] = (mean, math.sqrt(var) or 1.0)
    return stats


def train_linear_safe_scorer(
    rows: list[dict[str, object]],
    train_indices: list[int],
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
) -> dict[str, object]:
    categorical_features = categorical_features or []
    stats = mean_std(rows, train_indices, numeric_features)
    safe = [i for i in train_indices if float(rows[i]["safe_movement"]) >= 0.5]
    unsafe = [i for i in train_indices if float(rows[i]["safe_movement"]) < 0.5]
    weights: dict[str, float] = {}
    for feature in numeric_features:
        mean, std = stats[feature]
        if not safe or not unsafe:
            weights[feature] = 0.0
            continue
        safe_mean = sum((float(rows[i].get(feature, 0.0)) - mean) / std for i in safe) / len(safe)
        unsafe_mean = sum((float(rows[i].get(feature, 0.0)) - mean) / std for i in unsafe) / len(unsafe)
        weights[feature] = safe_mean - unsafe_mean

    global_rate = len(safe) / len(train_indices)
    cat_effects: dict[tuple[str, str], float] = {}
    for feature in categorical_features:
        by_value: dict[str, list[int]] = defaultdict(list)
        for i in train_indices:
            by_value[str(rows[i].get(feature, ""))].append(i)
        for value, idxs in by_value.items():
            rate = sum(float(rows[i]["safe_movement"]) for i in idxs) / len(idxs)
            shrink = len(idxs) / (len(idxs) + 20.0)
            cat_effects[(feature, value)] = (rate - global_rate) * shrink
    return {"stats": stats, "weights": weights, "cat_effects": cat_effects, "categorical": categorical_features}


def score_linear(row: dict[str, object], model: dict[str, object]) -> float:
    total = 0.0
    stats: dict[str, tuple[float, float]] = model["stats"]  # type: ignore[assignment]
    weights: dict[str, float] = model["weights"]  # type: ignore[assignment]
    for feature, weight in weights.items():
        mean, std = stats[feature]
        total += weight * ((float(row.get(feature, 0.0)) - mean) / std)
    cat_effects: dict[tuple[str, str], float] = model["cat_effects"]  # type: ignore[assignment]
    for feature in model["categorical"]:  # type: ignore[index]
        total += cat_effects.get((feature, str(row.get(feature, ""))), 0.0)
    return total


def assign_quantile_bins(rows: list[dict[str, object]], field: str, bins: int = 10) -> dict[int, int]:
    ordered = sorted(range(len(rows)), key=lambda i: (float(rows[i].get(field, 0.0)), str(rows[i]["row_id"])))
    result: dict[int, int] = {}
    for rank, idx in enumerate(ordered):
        result[idx] = min(bins - 1, int(rank * bins / max(1, len(ordered))))
    return result


def shuffled_scores(rows: list[dict[str, object]], seed: int, group_field: str | None = None, bin_field: str | None = None) -> dict[int, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    bins = assign_quantile_bins(rows, bin_field) if bin_field else {}
    for idx, row in enumerate(rows):
        parts = []
        if group_field:
            parts.append(str(row[group_field]))
        if bin_field:
            parts.append(f"{bin_field}_bin_{bins[idx]}")
        key = "::".join(parts) if parts else "global"
        groups[key].append(idx)

    scores: dict[int, float] = {}
    for key, idxs in sorted(groups.items()):
        values = [float(rows[i]["rho"]) for i in idxs]
        rng = random.Random(f"{seed}:shuffle:{key}")
        rng.shuffle(values)
        for idx, value in zip(sorted(idxs), values):
            scores[idx] = value
    return scores


def select_indices(indices: list[int], scores: dict[int, float], budget: float, rows: list[dict[str, object]]) -> list[int]:
    k = max(1, math.ceil(len(indices) * budget))
    ordered = sorted(indices, key=lambda i: (scores[i], str(rows[i]["route_id"]), str(rows[i]["row_id"])), reverse=True)
    return ordered[:k]


def summarize_selection(
    rows: list[dict[str, object]],
    selected: list[int],
    oracle_safe_rate: float,
) -> dict[str, float]:
    n = len(selected)
    safe_rate = sum(float(rows[i]["safe_movement"]) for i in selected) / n
    safe_var_rate = sum(float(rows[i]["safe_varentropy_movement"]) for i in selected) / n
    mean_abs_entropy = sum(float(rows[i]["abs_delta_entropy"]) for i in selected) / n
    mean_abs_varentropy = sum(float(rows[i]["abs_delta_varentropy"]) for i in selected) / n
    mean_drift = sum(float(rows[i]["drift"]) for i in selected) / n
    return {
        "selected_n": float(n),
        "safe_movement_rate": safe_rate,
        "safe_varentropy_movement_rate": safe_var_rate,
        "mean_abs_delta_entropy": mean_abs_entropy,
        "mean_abs_delta_varentropy": mean_abs_varentropy,
        "mean_output_drift": mean_drift,
        "movement_per_drift": mean_abs_entropy / max(mean_drift, 1e-12),
        "top1_preserved_rate": sum(float(rows[i]["top1_preserved"]) for i in selected) / n,
        "topk_preservation": sum(float(rows[i]["topk_jaccard"]) for i in selected) / n,
        "failure_rate": 1.0 - safe_rate,
        "regret_vs_oracle": oracle_safe_rate - safe_rate,
        "minimal_energy_to_target": sum(float(rows[i]["applied_fisher_energy"]) for i in selected) / n,
    }


def policy_scores(
    rows: list[dict[str, object]],
    train_indices: list[int],
    seed: int,
    shuffle_maps: dict[str, dict[int, float]],
) -> dict[str, dict[int, float]]:
    all_indices = list(range(len(rows)))
    learned_controls = train_linear_safe_scorer(rows, train_indices, NUMERIC_CONTROLS, CATEGORICAL_CONTROLS)
    learned_controls_rho = train_linear_safe_scorer(rows, train_indices, NUMERIC_CONTROLS + ["rho"], CATEGORICAL_CONTROLS)
    learned_gradient = train_linear_safe_scorer(rows, train_indices, ["projected_gradient_norm"], [])
    learned_gradient_rho = train_linear_safe_scorer(rows, train_indices, ["projected_gradient_norm", "rho"], [])
    return {
        "random": {i: stable_float(f"{seed}:random:{rows[i]['row_id']}") for i in all_indices},
        "highest_entropy": {i: float(rows[i]["entropy"]) for i in all_indices},
        "highest_varentropy": {i: float(rows[i]["varentropy"]) for i in all_indices},
        "lowest_confidence": {i: -float(rows[i]["confidence"]) for i in all_indices},
        "highest_gradient_norm": {i: float(rows[i]["projected_gradient_norm"]) for i in all_indices},
        "highest_fisher_output_energy": {i: float(rows[i]["unit_fisher_output_energy"]) for i in all_indices},
        "highest_jacobian_norm": {i: float(rows[i]["jacobian_norm"]) for i in all_indices},
        "rho_only": {i: float(rows[i]["rho"]) for i in all_indices},
        "shuffled_rho": shuffle_maps["global"],
        "within_example_shuffled_rho": shuffle_maps["within_example"],
        "rho_shuffled_within_entropy_bin": shuffle_maps["entropy"],
        "rho_shuffled_within_varentropy_bin": shuffle_maps["varentropy"],
        "rho_shuffled_within_gradient_bin": shuffle_maps["gradient"],
        "rho_shuffled_within_fisher_bin": shuffle_maps["fisher"],
        "controls_only_learned": {i: score_linear(rows[i], learned_controls) for i in all_indices},
        "controls_plus_rho_learned": {i: score_linear(rows[i], learned_controls_rho) for i in all_indices},
        "gradient_only_learned": {i: score_linear(rows[i], learned_gradient) for i in all_indices},
        "gradient_plus_rho_learned": {i: score_linear(rows[i], learned_gradient_rho) for i in all_indices},
        "oracle": {
            i: float(rows[i]["safe_movement"]) * 10.0
            + float(rows[i]["abs_delta_entropy"])
            - float(rows[i]["drift"])
            for i in all_indices
        },
    }


def weighted_metric(selected_rows: list[dict[str, object]], metric: str) -> float:
    if not selected_rows:
        return 0.0
    if metric == "safe_movement_rate":
        return sum(float(row["safe_movement"]) for row in selected_rows) / len(selected_rows)
    if metric == "movement_per_drift":
        movement = sum(float(row["abs_delta_entropy"]) for row in selected_rows) / len(selected_rows)
        drift = sum(float(row["drift"]) for row in selected_rows) / len(selected_rows)
        return movement / max(drift, 1e-12)
    if metric == "regret_vs_oracle":
        return sum(float(row["regret_vs_oracle"]) for row in selected_rows) / len(selected_rows)
    if metric == "topk_preservation":
        return sum(float(row["topk_jaccard"]) for row in selected_rows) / len(selected_rows)
    raise ValueError(metric)


def paired_cluster_delta(
    selected_records: list[dict[str, object]],
    split: str,
    budget: float,
    policy: str,
    control: str,
    metric: str,
) -> tuple[list[str], dict[str, tuple[list[dict[str, object]], list[dict[str, object]]]]]:
    groups: dict[str, tuple[list[dict[str, object]], list[dict[str, object]]]] = {}
    relevant = [
        row
        for row in selected_records
        if row["split"] == split and float(row["budget"]) == budget and row["policy"] in (policy, control)
    ]
    by_cluster_policy: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in relevant:
        by_cluster_policy[(str(row["cluster_id"]), str(row["policy"]))].append(row)
    clusters = sorted({key[0] for key in by_cluster_policy})
    for cluster in clusters:
        groups[cluster] = (
            by_cluster_policy.get((cluster, policy), []),
            by_cluster_policy.get((cluster, control), []),
        )
    return clusters, groups


def bootstrap_ci(
    selected_records: list[dict[str, object]],
    split: str,
    budget: float,
    policy: str,
    control: str,
    metric: str,
    draws: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    invert = metric == "regret_vs_oracle"
    metric_for_delta = "safe_movement_rate" if invert else metric
    clusters, groups = paired_cluster_delta(selected_records, split, budget, policy, control, metric)
    if not clusters:
        return (0.0, 0.0, 0.0)
    values = []
    for _ in range(draws):
        sampled_policy: list[dict[str, object]] = []
        sampled_control: list[dict[str, object]] = []
        for _ in clusters:
            cluster = clusters[rng.randrange(len(clusters))]
            policy_rows, control_rows = groups[cluster]
            sampled_policy.extend(policy_rows)
            sampled_control.extend(control_rows)
        delta = weighted_metric(sampled_policy, metric_for_delta) - weighted_metric(sampled_control, metric_for_delta)
        values.append(-delta if invert else delta)
    return percentile(values, 0.025), percentile(values, 0.5), percentile(values, 0.975)


def permutation_pvalue(
    selected_records: list[dict[str, object]],
    split: str,
    budget: float,
    policy: str,
    control: str,
    metric: str,
    permutations: int,
    rng: random.Random,
) -> float:
    invert = metric == "regret_vs_oracle"
    metric_for_delta = "safe_movement_rate" if invert else metric
    clusters, groups = paired_cluster_delta(selected_records, split, budget, policy, control, metric)
    diffs = []
    for cluster in clusters:
        policy_rows, control_rows = groups[cluster]
        delta = weighted_metric(policy_rows, metric_for_delta) - weighted_metric(control_rows, metric_for_delta)
        diffs.append(-delta if invert else delta)
    if not diffs:
        return 1.0
    observed = abs(sum(diffs) / len(diffs))
    count = 0
    for _ in range(permutations):
        value = sum(diff if rng.random() < 0.5 else -diff for diff in diffs) / len(diffs)
        if abs(value) >= observed - 1e-15:
            count += 1
    return (count + 1.0) / (permutations + 1.0)


def add_bh_fdr(rows: list[dict[str, object]]) -> None:
    keyed = [(i, float(row["p_value"])) for i, row in enumerate(rows)]
    keyed.sort(key=lambda item: item[1])
    m = len(keyed)
    adjusted = [1.0] * m
    running = 1.0
    for rank, (idx, p_value) in reversed(list(enumerate(keyed, start=1))):
        running = min(running, p_value * m / rank)
        adjusted[idx] = min(1.0, running)
    for row, q_value in zip(rows, adjusted):
        row["bh_fdr_q_value"] = q_value


def main() -> None:
    args = parse_args()
    budgets = [float(value) for value in args.budgets.split(",") if value.strip()]
    rows = read_rows(args.input, args.budget_mode, args.drift_cap, args.tau_varentropy, args.topk_threshold)
    if not rows:
        raise RuntimeError("No rows available for the requested budget mode.")

    shuffle_maps = {
        "global": shuffled_scores(rows, args.seed),
        "within_example": shuffled_scores(rows, args.seed, group_field="example_id"),
        "entropy": shuffled_scores(rows, args.seed, bin_field="entropy"),
        "varentropy": shuffled_scores(rows, args.seed, bin_field="varentropy"),
        "gradient": shuffled_scores(rows, args.seed, bin_field="projected_gradient_norm"),
        "fisher": shuffled_scores(rows, args.seed, bin_field="unit_fisher_output_energy"),
    }

    folds = []
    folds.extend(make_grouped_folds(rows, args.seed))
    folds.extend(make_leave_one_folds(rows, "model", "leave_one_model_out"))
    folds.extend(make_leave_one_folds(rows, "task", "leave_one_task_out"))
    folds.extend(make_leave_one_folds(rows, "route_family", "leave_one_route_family_out"))

    selected_records: list[dict[str, object]] = []
    fold_summary_rows: list[dict[str, object]] = []
    policies = [
        "random",
        "highest_entropy",
        "highest_varentropy",
        "lowest_confidence",
        "highest_gradient_norm",
        "highest_fisher_output_energy",
        "highest_jacobian_norm",
        "rho_only",
        "shuffled_rho",
        "within_example_shuffled_rho",
        "rho_shuffled_within_entropy_bin",
        "rho_shuffled_within_varentropy_bin",
        "rho_shuffled_within_gradient_bin",
        "rho_shuffled_within_fisher_bin",
        "controls_only_learned",
        "controls_plus_rho_learned",
        "gradient_only_learned",
        "gradient_plus_rho_learned",
        "oracle",
    ]

    for fold in folds:
        scores = policy_scores(rows, fold["train"], args.seed, shuffle_maps)
        test_indices = fold["test"]
        for budget in budgets:
            oracle_selected = select_indices(test_indices, scores["oracle"], budget, rows)
            oracle_safe_rate = sum(float(rows[i]["safe_movement"]) for i in oracle_selected) / len(oracle_selected)
            for policy in policies:
                selected = select_indices(test_indices, scores[policy], budget, rows)
                summary = summarize_selection(rows, selected, oracle_safe_rate)
                fold_summary_rows.append(
                    {
                        "split": fold["split"],
                        "fold": fold["fold"],
                        "budget": budget,
                        "policy": policy,
                        "test_n": len(test_indices),
                        **summary,
                    }
                )
                for idx in selected:
                    selected_records.append(
                        {
                            "split": fold["split"],
                            "fold": fold["fold"],
                            "budget": budget,
                            "policy": policy,
                            "cluster_id": rows[idx]["cluster_id"],
                            "example_id": rows[idx]["example_id"],
                            "model": rows[idx]["model"],
                            "task": rows[idx]["task"],
                            "layer": rows[idx]["layer"],
                            "route_family": rows[idx]["route_family"],
                            "subspace_dim": rows[idx]["subspace_dim"],
                            "rho": rows[idx]["rho"],
                            "entropy": rows[idx]["entropy"],
                            "varentropy": rows[idx]["varentropy"],
                            "confidence": rows[idx]["confidence"],
                            "projected_gradient_norm": rows[idx]["projected_gradient_norm"],
                            "unit_fisher_output_energy": rows[idx]["unit_fisher_output_energy"],
                            "jacobian_norm": rows[idx]["jacobian_norm"],
                            "safe_movement": rows[idx]["safe_movement"],
                            "safe_varentropy_movement": rows[idx]["safe_varentropy_movement"],
                            "abs_delta_entropy": rows[idx]["abs_delta_entropy"],
                            "abs_delta_varentropy": rows[idx]["abs_delta_varentropy"],
                            "drift": rows[idx]["drift"],
                            "top1_preserved": rows[idx]["top1_preserved"],
                            "topk_jaccard": rows[idx]["topk_jaccard"],
                            "applied_fisher_energy": rows[idx]["applied_fisher_energy"],
                            "regret_vs_oracle": summary["regret_vs_oracle"],
                        }
                    )

    summary_rows: list[dict[str, object]] = []
    grouped_summary: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in fold_summary_rows:
        grouped_summary[(str(row["split"]), float(row["budget"]), str(row["policy"]))].append(row)
    metric_names = [
        "safe_movement_rate",
        "safe_varentropy_movement_rate",
        "mean_abs_delta_entropy",
        "mean_abs_delta_varentropy",
        "mean_output_drift",
        "movement_per_drift",
        "top1_preserved_rate",
        "topk_preservation",
        "failure_rate",
        "regret_vs_oracle",
        "minimal_energy_to_target",
    ]
    for (split, budget, policy), values in sorted(grouped_summary.items()):
        out: dict[str, object] = {
            "split": split,
            "budget": budget,
            "policy": policy,
            "folds": len(values),
            "mean_test_n": sum(float(row["test_n"]) for row in values) / len(values),
            "mean_selected_n": sum(float(row["selected_n"]) for row in values) / len(values),
        }
        for metric in metric_names:
            out[metric] = sum(float(row[metric]) for row in values) / len(values)
        summary_rows.append(out)

    summary_lookup = {(row["split"], float(row["budget"]), row["policy"]): row for row in summary_rows}
    contrasts = [
        ("controls_plus_rho_learned", "controls_only_learned"),
        ("gradient_plus_rho_learned", "gradient_only_learned"),
        ("rho_only", "within_example_shuffled_rho"),
        ("rho_only", "shuffled_rho"),
        ("rho_only", "random"),
        ("rho_only", "highest_entropy"),
        ("rho_only", "highest_fisher_output_energy"),
    ]
    contrast_metrics = ["safe_movement_rate", "movement_per_drift", "regret_vs_oracle", "topk_preservation"]
    contrast_rows: list[dict[str, object]] = []
    for split in sorted({str(row["split"]) for row in summary_rows}):
        for budget in budgets:
            for policy, control in contrasts:
                for metric in contrast_metrics:
                    left = float(summary_lookup[(split, budget, policy)][metric])
                    right = float(summary_lookup[(split, budget, control)][metric])
                    contrast_rows.append(
                        {
                            "split": split,
                            "budget": budget,
                            "contrast": f"{policy}_vs_{control}",
                            "metric": metric,
                            "policy": left,
                            "control": right,
                            "delta_policy_minus_control": left - right,
                        }
                    )

    rng = random.Random(args.seed)
    ci_rows: list[dict[str, object]] = []
    p_rows: list[dict[str, object]] = []
    for row in contrast_rows:
        split = str(row["split"])
        budget = float(row["budget"])
        policy, control = str(row["contrast"]).split("_vs_", 1)
        metric = str(row["metric"])
        ci_low, median, ci_high = bootstrap_ci(
            selected_records, split, budget, policy, control, metric, args.bootstrap, rng
        )
        ci_rows.append({**row, "ci_low": ci_low, "median": median, "ci_high": ci_high})
        p_rows.append(
            {
                **row,
                "p_value": permutation_pvalue(
                    selected_records, split, budget, policy, control, metric, args.permutations, rng
                ),
            }
        )
    add_bh_fdr(p_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "budgeted_allocation_selected_records.csv", selected_records)
    write_csv(args.out_dir / "budgeted_allocation_fold_summary.csv", fold_summary_rows)
    write_csv(args.out_dir / "budgeted_allocation_summary.csv", summary_rows)
    write_csv(args.out_dir / "budgeted_allocation_contrasts.csv", contrast_rows)
    write_csv(args.out_dir / "budgeted_allocation_bootstrap_ci.csv", ci_rows)
    write_csv(args.out_dir / "budgeted_allocation_permutation_fdr.csv", p_rows)

    final_split = "grouped_example_5fold"
    final_budget = 0.10 if 0.10 in budgets else budgets[0]
    table_policies = [
        "random",
        "highest_entropy",
        "highest_gradient_norm",
        "highest_fisher_output_energy",
        "controls_only_learned",
        "rho_only",
        "controls_plus_rho_learned",
        "gradient_plus_rho_learned",
        "oracle",
    ]
    final_rows = [
        summary_lookup[(final_split, final_budget, policy)]
        for policy in table_policies
        if (final_split, final_budget, policy) in summary_lookup
    ]
    key_contrasts = [
        row
        for row in ci_rows
        if row["split"] == final_split
        and float(row["budget"]) == final_budget
        and row["metric"] in ("safe_movement_rate", "movement_per_drift", "regret_vs_oracle")
        and row["contrast"]
        in (
            "controls_plus_rho_learned_vs_controls_only_learned",
            "gradient_plus_rho_learned_vs_gradient_only_learned",
            "rho_only_vs_within_example_shuffled_rho",
        )
    ]
    key_p = {
        (row["contrast"], row["metric"]): row
        for row in p_rows
        if row["split"] == final_split and float(row["budget"]) == final_budget
    }
    budget_sensitivity_rows = [
        row
        for row in ci_rows
        if row["split"] == final_split
        and row["metric"] == "safe_movement_rate"
        and row["contrast"]
        in (
            "controls_plus_rho_learned_vs_controls_only_learned",
            "gradient_plus_rho_learned_vs_gradient_only_learned",
            "rho_only_vs_within_example_shuffled_rho",
            "rho_only_vs_random",
            "rho_only_vs_highest_entropy",
            "rho_only_vs_highest_fisher_output_energy",
        )
    ]

    report_lines = [
        "# Budgeted Controllability Allocation Test",
        "",
        "This test asks whether rho has decision value under a fixed intervention budget: each policy may select only a fraction of held-out candidate routes, and all candidates come from the `same_fisher_output_energy` intervention panel unless otherwise configured.",
        "",
        "No learned selector is fit on its evaluation fold. Learned controls use only training-fold pre-intervention features and the training-fold observed safe-movement labels; held-out scoring uses pre-intervention features only. The oracle is reported only as an upper bound.",
        "",
        f"Rows evaluated: {len(rows)}. Budget mode: `{args.budget_mode}`. Drift cap: {args.drift_cap}. Budgets: {', '.join(str(b) for b in budgets)}.",
        "",
        "## Main Grouped-Prompt Panel",
        "```text",
        "Policy                         SafeMove@10%   Move/Drift   Regret   TopK",
    ]
    for row in final_rows:
        report_lines.append(
            f"{str(row['policy']):29s} {float(row['safe_movement_rate']):12.6f} "
            f"{float(row['movement_per_drift']):12.3f} {float(row['regret_vs_oracle']):8.6f} "
            f"{float(row['topk_preservation']):6.4f}"
        )
    report_lines.extend(["```", "", "## Key Paired Cluster Bootstrap Contrasts", "```text"])
    report_lines.append("Contrast                                           Metric              Delta      CI low     CI high   p        q")
    for row in key_contrasts:
        p_row = key_p[(row["contrast"], row["metric"])]
        report_lines.append(
            f"{str(row['contrast'])[:50]:50s} {str(row['metric'])[:18]:18s} "
            f"{float(row['delta_policy_minus_control']):9.6f} {float(row['ci_low']):9.6f} "
            f"{float(row['ci_high']):9.6f} {float(p_row['p_value']):8.5f} {float(p_row['bh_fdr_q_value']):8.5f}"
        )
    report_lines.extend(["```", "", "## SafeMove Budget Sensitivity", "```text"])
    report_lines.append("Budget  Contrast                                           Delta      CI low     CI high")
    for row in budget_sensitivity_rows:
        report_lines.append(
            f"{float(row['budget']):6.2f}  {str(row['contrast'])[:50]:50s} "
            f"{float(row['delta_policy_minus_control']):9.6f} {float(row['ci_low']):9.6f} {float(row['ci_high']):9.6f}"
        )
    report_lines.extend(
        [
            "```",
            "",
            "## Negative Controls",
            "",
            "The test includes global shuffled rho, within-example shuffled rho, and rho shuffled within entropy, varentropy, gradient, and Fisher-energy bins. The within-example shuffle is the strictest negative control because it permutes rho only among candidate routes belonging to the same example/intervention setting.",
            "",
            "## Files",
            "```text",
            "budgeted_allocation_selected_records.csv",
            "budgeted_allocation_fold_summary.csv",
            "budgeted_allocation_summary.csv",
            "budgeted_allocation_contrasts.csv",
            "budgeted_allocation_bootstrap_ci.csv",
            "budgeted_allocation_permutation_fdr.csv",
            "report.md",
            "```",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    config_dir = args.out_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "command": "python scripts/run_budgeted_controllability_allocation_test.py --bootstrap 1000 --permutations 5000 --seed 20260613",
        "input": str(args.input.relative_to(ROOT)),
        "outputs": str(args.out_dir.relative_to(ROOT)),
        "budget_mode": args.budget_mode,
        "budgets": budgets,
        "drift_cap": args.drift_cap,
        "tau_varentropy": args.tau_varentropy,
        "topk_threshold": args.topk_threshold,
        "seed": args.seed,
    }
    (config_dir / "reproduce.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
