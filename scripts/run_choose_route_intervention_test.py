from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "04_uncertainty_steering" / "03_choose_route_intervention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--fisher-output-energy", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def code_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def route_cols(frame: pd.DataFrame) -> list[str]:
    return [col for col in ["source", "model", "prompt_id", "task", "topic", "observed_condition", "sign"] if col in frame.columns]


def candidate_cols(frame: pd.DataFrame) -> list[str]:
    return [col for col in ["layer", "subspace_family", "subspace_k", "k", "rep", "top_k_output"] if col in frame.columns]


def load_equal_output(energy: float) -> pd.DataFrame:
    frame = read_csv(ROOT / "experiments" / "04_uncertainty_steering" / "02_equal_output_movement" / "outputs" / "equal_output_efficiency_records.csv")
    if frame.empty:
        raise RuntimeError("Missing equal_output_efficiency_records.csv")
    frame = frame[frame["direction"].isin(["accessible_ls", "grad_orthogonal_control", "random_control"])].copy()
    frame = frame[np.isclose(frame["fisher_output_energy"].astype(float), energy)].copy()
    if frame.empty:
        raise RuntimeError(f"No equal-output rows at fisher_output_energy={energy}")
    frame["_unit_id"] = frame[route_cols(frame)].astype(str).agg("::".join, axis=1)
    frame["_route_id"] = frame[candidate_cols(frame)].astype(str).agg("::".join, axis=1)
    return frame


def deterministic_random_choice(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["_unit_id", "_route_id"]
    return frame.sort_values(sort_cols).groupby("_unit_id", as_index=False).head(1)


def choose_accessible_routes(frame: pd.DataFrame) -> pd.DataFrame:
    accessible = frame[frame["direction"].eq("accessible_ls")].copy()
    grad_col = "grad_entropy_proj_norm" if "grad_entropy_proj_norm" in accessible.columns else "rho"
    choices = [
        ("high_rho_route", accessible.sort_values(["_unit_id", "rho"], ascending=[True, False]).groupby("_unit_id", as_index=False).head(1)),
        ("low_rho_route", accessible.sort_values(["_unit_id", "rho"], ascending=[True, True]).groupby("_unit_id", as_index=False).head(1)),
        ("gradient_route", accessible.sort_values(["_unit_id", grad_col], ascending=[True, False]).groupby("_unit_id", as_index=False).head(1)),
        ("random_route", deterministic_random_choice(accessible)),
    ]
    rows = []
    for policy, chosen in choices:
        part = chosen.copy()
        part["candidate_policy"] = policy
        rows.append(part)
    selected = pd.concat(rows, ignore_index=True)

    high_keys = choices[0][1][["_unit_id", "_route_id"]].drop_duplicates()
    matched = frame.merge(high_keys, on=["_unit_id", "_route_id"], how="inner")
    for direction, policy in [
        ("grad_orthogonal_control", "gradient_orthogonal_route"),
        ("random_control", "equal_output_energy_control"),
    ]:
        part = matched[matched["direction"].eq(direction)].copy()
        part["candidate_policy"] = policy
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize_routes(chosen: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "n": ("_unit_id", "nunique"),
        "rho_mean": ("rho", "mean"),
        "abs_delta_entropy": ("abs_delta_entropy", "mean"),
        "abs_delta_varentropy": ("abs_delta_varentropy", "mean"),
        "output_drift": ("fisher_output_energy", "mean"),
    }
    optional = {
        "target_success_rate": ("directional_success", "mean"),
        "top10_preservation": ("top10_jaccard", "mean"),
        "top1_changed_rate": ("selected_top1_changed", "mean"),
        "semantic_drift": ("selected_kl_before_after", "mean"),
    }
    for key, value in optional.items():
        if value[0] in chosen.columns:
            aggregations[key] = value
    return chosen.groupby("candidate_policy", as_index=False).agg(**aggregations).sort_values("candidate_policy")


def metric_means(frame: pd.DataFrame) -> dict[str, float]:
    out = {
        "abs_delta_entropy": float(frame["abs_delta_entropy"].mean()),
        "abs_delta_varentropy": float(frame["abs_delta_varentropy"].mean()),
        "output_drift": float(frame["fisher_output_energy"].mean()),
    }
    if "directional_success" in frame.columns:
        out["target_success_rate"] = float(frame["directional_success"].mean())
    if "top10_jaccard" in frame.columns:
        out["top10_preservation"] = float(frame["top10_jaccard"].mean())
    if "selected_top1_changed" in frame.columns:
        out["top1_changed_rate"] = float(frame["selected_top1_changed"].mean())
    return out


def route_contrasts(chosen: pd.DataFrame) -> pd.DataFrame:
    base = chosen[chosen["candidate_policy"].eq("high_rho_route")]
    rows = []
    high = metric_means(base)
    for policy, group in chosen.groupby("candidate_policy", sort=True):
        if policy == "high_rho_route":
            continue
        metrics = metric_means(group)
        for metric, value in metrics.items():
            direction = -1 if metric in {"output_drift", "top1_changed_rate"} else 1
            delta = high[metric] - value if direction > 0 else value - high[metric]
            rows.append(
                {
                    "contrast": f"high_rho_route_vs_{policy}",
                    "metric": metric,
                    "high_rho": high[metric],
                    "control": value,
                    "delta_high_minus_control": float(delta),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_contrasts(chosen: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grouped = {unit: frame for unit, frame in chosen.groupby("_unit_id", sort=False)}
    units = np.array(list(grouped), dtype=object)
    rows = []
    for _ in range(n_boot):
        sample = pd.concat([grouped[unit] for unit in rng.choice(units, size=len(units), replace=True)], ignore_index=True)
        rows.append(route_contrasts(sample))
    boot = pd.concat(rows, ignore_index=True)
    return (
        boot.groupby(["contrast", "metric"])["delta_high_minus_control"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_low", 0.5: "median", 0.975: "ci_high"})
    )


def load_minimal_energy() -> pd.DataFrame:
    frame = read_csv(ROOT / "experiments" / "04_uncertainty_steering" / "01_minimal_intervention_energy" / "outputs" / "minimal_energy_cost_records.csv")
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["_unit_id"] = frame[route_cols(frame)].astype(str).agg("::".join, axis=1)
    frame["_route_id"] = frame[candidate_cols(frame)].astype(str).agg("::".join, axis=1)
    accessible = frame[frame.get("direction", "accessible_ls").eq("accessible_ls")].copy() if "direction" in frame.columns else frame
    grad_col = "grad_entropy_proj_norm" if "grad_entropy_proj_norm" in accessible.columns else "rho"
    choices = [
        ("high_rho_route", accessible.sort_values(["_unit_id", "rho"], ascending=[True, False]).groupby("_unit_id", as_index=False).head(1)),
        ("low_rho_route", accessible.sort_values(["_unit_id", "rho"], ascending=[True, True]).groupby("_unit_id", as_index=False).head(1)),
        ("gradient_route", accessible.sort_values(["_unit_id", grad_col], ascending=[True, False]).groupby("_unit_id", as_index=False).head(1)),
        ("random_route", deterministic_random_choice(accessible)),
    ]
    rows = []
    for policy, chosen in choices:
        part = chosen.copy()
        part["candidate_policy"] = policy
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize_minimal_energy(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby("candidate_policy", as_index=False)
        .agg(
            n=("_unit_id", "nunique"),
            minimal_energy_to_entropy_target=("cost_entropy_tau", "median"),
            minimal_energy_to_varentropy_target=("cost_varentropy_tau", "median"),
            entropy_target_success_rate=("observed_reached_entropy_tau", "mean"),
            varentropy_target_success_rate=("observed_reached_varentropy_tau", "mean"),
            semantic_top10_preservation=("top10_jaccard", "mean"),
            top1_changed_rate=("selected_top1_changed", "mean"),
        )
        .sort_values("candidate_policy")
    )


def write_report(out_dir: Path, summary: pd.DataFrame, contrasts: pd.DataFrame, ci: pd.DataFrame, minimal: pd.DataFrame) -> None:
    lines = [
        "# Choose-The-Route Intervention Test",
        "",
        "This derived intervention test asks whether choosing the high-rho route for each example moves uncertainty more efficiently than low-rho, gradient-selected, random, and equal-Fisher-output controls. It reuses checked-in intervention records; no correctness label is used to choose a route.",
        "",
        "## Route Summary",
        code_table(summary, 80),
        "",
        "## High-Rho Contrasts",
        code_table(contrasts, 120),
        "",
        "## Cluster Bootstrap CI",
        code_table(ci, 120),
        "",
        "## Minimal Energy To Target",
        code_table(minimal, 80),
        "",
        "## Files",
        "```text",
        "choose_route_records.csv",
        "choose_route_summary.csv",
        "choose_route_contrasts.csv",
        "choose_route_bootstrap_ci.csv",
        "choose_route_minimal_energy.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for child in ["outputs", "reports", "config"]:
        (args.out_dir / child).mkdir(parents=True, exist_ok=True)
    equal = load_equal_output(args.fisher_output_energy)
    chosen = choose_accessible_routes(equal)
    summary = summarize_routes(chosen)
    contrasts = route_contrasts(chosen)
    ci = bootstrap_contrasts(chosen, args.bootstrap, args.seed)
    minimal = summarize_minimal_energy(load_minimal_energy())

    chosen.to_csv(args.out_dir / "outputs" / "choose_route_records.csv", index=False)
    summary.to_csv(args.out_dir / "outputs" / "choose_route_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "outputs" / "choose_route_contrasts.csv", index=False)
    ci.to_csv(args.out_dir / "outputs" / "choose_route_bootstrap_ci.csv", index=False)
    minimal.to_csv(args.out_dir / "outputs" / "choose_route_minimal_energy.csv", index=False)
    metadata = {
        "command": "python scripts/run_choose_route_intervention_test.py",
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "fisher_output_energy": args.fisher_output_energy,
        "source_records": [
            "experiments/04_uncertainty_steering/02_equal_output_movement/outputs/equal_output_efficiency_records.csv",
            "experiments/04_uncertainty_steering/01_minimal_intervention_energy/outputs/minimal_energy_cost_records.csv",
        ],
        "policy_constraint": "Routes are selected from rho, gradient proxy, or deterministic route identity; correctness is not used.",
    }
    (args.out_dir / "config" / "reproduce.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_report(args.out_dir, summary, contrasts, ci, minimal)
    print(args.out_dir)


if __name__ == "__main__":
    main()
