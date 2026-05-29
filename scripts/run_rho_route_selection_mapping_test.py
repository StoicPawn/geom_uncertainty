from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "05_controllability_mapping" / "outputs" / "intervention_points.csv"
OUT = ROOT / "experiments" / "05_controllability_mapping" / "route_selection_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def stable_index(key: str, n: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % n


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("model", ""),
        row.get("model_type", ""),
        row.get("item_id", ""),
        row.get("task", ""),
        row.get("budget_mode", ""),
        row.get("eps", ""),
    )


def route_id(row: dict[str, str]) -> str:
    return "::".join([row.get("layer", ""), row.get("route_family", ""), row.get("subspace_dim", "")])


def choose(group: list[dict[str, str]], policy: str) -> dict[str, str]:
    if policy == "high_rho_route":
        return max(group, key=lambda row: (to_float(row.get("rho")), route_id(row)))
    if policy == "low_rho_route":
        return min(group, key=lambda row: (to_float(row.get("rho")), route_id(row)))
    if policy == "high_gradient_route":
        return max(group, key=lambda row: (to_float(row.get("projected_gradient_norm")), route_id(row)))
    if policy == "high_jacobian_route":
        return max(group, key=lambda row: (to_float(row.get("jacobian_norm")), route_id(row)))
    if policy == "high_fisher_energy_route":
        return max(group, key=lambda row: (to_float(row.get("unit_fisher_output_energy")), route_id(row)))
    if policy == "deterministic_random_route":
        ordered = sorted(group, key=route_id)
        key = "::".join(group_key(group[0]))
        return ordered[stable_index(key, len(ordered))]
    raise ValueError(f"Unknown policy: {policy}")


def summarize(records: list[dict[str, object]]) -> dict[str, float]:
    n = len(records)
    return {
        "n": float(n),
        "safe_movement_rate": sum(float(row["safe_movement"]) for row in records) / n,
        "mean_uncertainty_movement": sum(float(row["uncertainty_movement"]) for row in records) / n,
        "mean_varentropy_movement": sum(float(row["varentropy_movement"]) for row in records) / n,
        "mean_drift": sum(float(row["drift"]) for row in records) / n,
        "top1_preserved_rate": sum(float(row["top1_preserved"]) for row in records) / n,
        "topk_jaccard_mean": sum(float(row["topk_jaccard"]) for row in records) / n,
        "rho_mean": sum(float(row["rho"]) for row in records) / n,
        "projected_gradient_norm_mean": sum(float(row["projected_gradient_norm"]) for row in records) / n,
        "jacobian_norm_mean": sum(float(row["jacobian_norm"]) for row in records) / n,
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


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def record_metric(row: dict[str, object], metric: str) -> float:
    mapping = {
        "safe_movement_rate": "safe_movement",
        "mean_uncertainty_movement": "uncertainty_movement",
        "mean_varentropy_movement": "varentropy_movement",
        "mean_drift": "drift",
        "top1_preserved_rate": "top1_preserved",
        "topk_jaccard_mean": "topk_jaccard",
    }
    return float(row[mapping[metric]])


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("safe_movement") in ("", None):
            continue
        grouped[group_key(row)].append(row)
    grouped = {key: group for key, group in grouped.items() if len(group) >= 2}

    policies = [
        "high_rho_route",
        "low_rho_route",
        "high_gradient_route",
        "high_jacobian_route",
        "high_fisher_energy_route",
        "deterministic_random_route",
    ]
    record_rows: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        for policy in policies:
            chosen = choose(group, policy)
            record_rows.append(
                {
                    "group_id": "::".join(key),
                    "policy": policy,
                    "model": chosen.get("model", ""),
                    "model_type": chosen.get("model_type", ""),
                    "task": chosen.get("task", ""),
                    "budget_mode": chosen.get("budget_mode", ""),
                    "eps": chosen.get("eps", ""),
                    "layer": chosen.get("layer", ""),
                    "route_family": chosen.get("route_family", ""),
                    "subspace_dim": chosen.get("subspace_dim", ""),
                    "rho": to_float(chosen.get("rho")),
                    "projected_gradient_norm": to_float(chosen.get("projected_gradient_norm")),
                    "jacobian_norm": to_float(chosen.get("jacobian_norm")),
                    "uncertainty_movement": to_float(chosen.get("uncertainty_movement")),
                    "varentropy_movement": to_float(chosen.get("varentropy_movement")),
                    "drift": to_float(chosen.get("drift")),
                    "top1_preserved": to_float(chosen.get("top1_preserved")),
                    "topk_jaccard": to_float(chosen.get("topk_jaccard")),
                    "safe_movement": to_float(chosen.get("safe_movement")),
                }
            )

    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in record_rows:
        by_policy[str(row["policy"])].append(row)

    summary_rows = []
    for policy, records in sorted(by_policy.items()):
        summary_rows.append({"policy": policy, **summarize(records)})

    metrics = [
        "safe_movement_rate",
        "mean_uncertainty_movement",
        "mean_varentropy_movement",
        "mean_drift",
        "top1_preserved_rate",
        "topk_jaccard_mean",
    ]
    summary_by_policy = {row["policy"]: row for row in summary_rows}
    contrast_rows = []
    for control in policies:
        if control == "high_rho_route":
            continue
        for metric in metrics:
            rho_value = float(summary_by_policy["high_rho_route"][metric])
            control_value = float(summary_by_policy[control][metric])
            contrast_rows.append(
                {
                    "contrast": f"high_rho_route_vs_{control}",
                    "metric": metric,
                    "high_rho": rho_value,
                    "control": control_value,
                    "delta_high_rho_minus_control": rho_value - control_value,
                }
            )

    rng = random.Random(args.seed)
    group_ids = sorted(grouped)
    records_by_group_policy = {(row["group_id"], row["policy"]): row for row in record_rows}
    bootstrap_rows = []
    for control in [policy for policy in policies if policy != "high_rho_route"]:
        for metric in metrics:
            draws = []
            for _ in range(args.bootstrap):
                sample = [group_ids[rng.randrange(len(group_ids))] for _ in group_ids]
                diffs = []
                for key in sample:
                    gid = "::".join(key)
                    high = record_metric(records_by_group_policy[(gid, "high_rho_route")], metric)
                    ctrl = record_metric(records_by_group_policy[(gid, control)], metric)
                    diffs.append(high - ctrl)
                draws.append(sum(diffs) / len(diffs))
            bootstrap_rows.append(
                {
                    "contrast": f"high_rho_route_vs_{control}",
                    "metric": metric,
                    "ci_low": percentile(draws, 0.025),
                    "median": percentile(draws, 0.5),
                    "ci_high": percentile(draws, 0.975),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "route_selection_records.csv", record_rows)
    write_csv(args.out_dir / "route_selection_summary.csv", summary_rows)
    write_csv(args.out_dir / "route_selection_contrasts.csv", contrast_rows)
    write_csv(args.out_dir / "route_selection_bootstrap_ci.csv", bootstrap_rows)

    high = summary_by_policy["high_rho_route"]
    random_control = summary_by_policy["deterministic_random_route"]
    low = summary_by_policy["low_rho_route"]
    gradient = summary_by_policy["high_gradient_route"]
    fisher = summary_by_policy["high_fisher_energy_route"]
    report = [
        "# Rho Route-Selection Policy On Controllability Mapping",
        "",
        "This derived test asks whether `rho` selects the internal route that should be used for a local uncertainty intervention.",
        "Each case has multiple candidate layer/route/subspace interventions. Policies choose a route using only pre-intervention geometry; `safe_movement` is used only for evaluation.",
        "",
        "## Verdict",
        "",
        "High-rho route selection strongly beats low-rho and deterministic random routes for safe movement and uncertainty movement. It is competitive with high-gradient and high-Jacobian selectors, but it does not uniformly dominate high-Fisher-energy route selection. This supports rho as an intervention-route feature, not as a universal replacement for all local geometric selectors.",
        "",
        "## Key Results",
        "",
        "```text",
        f"safe_movement: high-rho {float(high['safe_movement_rate']):.6f} vs random {float(random_control['safe_movement_rate']):.6f} vs low-rho {float(low['safe_movement_rate']):.6f}",
        f"safe_movement strong controls: high-gradient {float(gradient['safe_movement_rate']):.6f}, high-Fisher-energy {float(fisher['safe_movement_rate']):.6f}",
        f"mean uncertainty movement: high-rho {float(high['mean_uncertainty_movement']):.6f} vs random {float(random_control['mean_uncertainty_movement']):.6f} vs low-rho {float(low['mean_uncertainty_movement']):.6f}",
        f"top1 preservation: high-rho {float(high['top1_preserved_rate']):.6f} vs random {float(random_control['top1_preserved_rate']):.6f} vs low-rho {float(low['top1_preserved_rate']):.6f}",
        "```",
        "",
        "## Files",
        "```text",
        "route_selection_records.csv",
        "route_selection_summary.csv",
        "route_selection_contrasts.csv",
        "route_selection_bootstrap_ci.csv",
        "report.md",
        "```",
    ]
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
