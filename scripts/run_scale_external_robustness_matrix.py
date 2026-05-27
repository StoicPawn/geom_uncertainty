from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "controls" / "scale_external_robustness"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def family(model: str) -> str:
    lowered = str(model).lower()
    if "roberta" in lowered:
        return "RoBERTa"
    if "llama" in lowered:
        return "Llama"
    if "mistral" in lowered:
        return "Mistral"
    if "qwen" in lowered:
        return "Qwen"
    if "phi" in lowered:
        return "Phi"
    if "gpt2" in lowered:
        return "GPT-2"
    if "distilbert" in lowered:
        return "DistilBERT"
    if "bert" in lowered:
        return "BERT"
    return "Other"


def model_coverage() -> pd.DataFrame:
    sources = [
        ("masked_topk", ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "topk_scores.csv", "model"),
        ("masked_full_battery", ROOT / "experiments" / "04_uncertainty_steering" / "outputs" / "subspace_scores.csv", "model"),
        ("decoder_battery", ROOT / "experiments" / "04_uncertainty_steering" / "decoder_main_battery" / "outputs" / "decoder_scores.csv", "model"),
        ("controllability_mapping", ROOT / "experiments" / "05_controllability_mapping" / "outputs" / "target_summary.csv", "model"),
    ]
    rows: list[dict[str, object]] = []
    for source, path, col in sources:
        frame = read_csv(path)
        if frame.empty or col not in frame.columns:
            continue
        for model, group in frame.groupby(col, dropna=False):
            rows.append(
                {
                    "source": source,
                    "model": model,
                    "family": family(str(model)),
                    "status": "completed",
                    "rows": int(len(group)),
                }
            )
    skip_sources = [
        ("masked_topk", ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "skipped_models.csv"),
        ("masked_full_battery", ROOT / "experiments" / "04_uncertainty_steering" / "outputs" / "skipped_models.csv"),
        ("decoder_battery", ROOT / "experiments" / "04_uncertainty_steering" / "decoder_main_battery" / "outputs" / "skipped_models.csv"),
        ("controllability_mapping", ROOT / "experiments" / "05_controllability_mapping" / "outputs" / "model_status.csv"),
    ]
    configured = [
        ("masked_topk", ROOT / "experiments" / "controls" / "topk_robustness" / "config" / "reproduce.json", "models"),
        ("masked_full_battery", ROOT / "experiments" / "04_uncertainty_steering" / "config" / "reproduce.json", "models"),
        (
            "decoder_battery",
            ROOT / "experiments" / "04_uncertainty_steering" / "decoder_main_battery" / "config" / "reproduce.json",
            "models",
        ),
        ("controllability_mapping", ROOT / "experiments" / "05_controllability_mapping" / "config" / "reproduce.json", "decoder_models"),
        ("controllability_mapping", ROOT / "experiments" / "05_controllability_mapping" / "config" / "reproduce.json", "masked_models_requested"),
    ]
    auto_download_requested: set[tuple[str, str]] = set()
    for source, path, key in configured:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if bool(payload.get("local_files_only", False)):
            continue
        for model in payload.get(key, []):
            auto_download_requested.add((source, str(model)))
    for source, path in skip_sources:
        frame = read_csv(path)
        if frame.empty or "model" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            status = str(row.get("status", "skipped"))
            if status == "completed":
                continue
            reason = row.get("reason", row.get("error", ""))
            marker = (source, str(row["model"]))
            reason_text = str(reason).lower()
            if marker in auto_download_requested and (
                "missing local cache" in reason_text or "couldn't connect" in reason_text
            ):
                status = "historical_skip_auto_download_now"
                reason = "previous checked-in run skipped; current reproduce config downloads automatically"
            rows.append(
                {
                    "source": source,
                    "model": row["model"],
                    "family": family(str(row["model"])),
                    "status": status,
                    "rows": 0,
                    "reason": reason,
                }
            )
    present = {(str(row.get("source")), str(row.get("model"))) for row in rows}
    for source, path, key in configured:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for model in payload.get(key, []):
            marker = (source, str(model))
            if marker in present:
                continue
            rows.append(
                {
                    "source": source,
                    "model": model,
                    "family": family(str(model)),
                    "status": "requested_pending_run",
                    "rows": 0,
                    "reason": "listed in reproduce config but not present in checked-in outputs",
                }
            )
            present.add(marker)
    return pd.DataFrame(rows)


def topk_strength() -> pd.DataFrame:
    rank = read_csv(ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "topk_rank_stability.csv")
    layer = read_csv(ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "topk_layer_trend_stability.csv")
    rows = []
    if not rank.empty:
        rows.append(
            {
                "claim": "rho_rank_stability_vs_topk",
                "n": int(rank["n"].sum()) if "n" in rank else int(len(rank)),
                "median": float(rank["rho_spearman_vs_ref"].median()),
                "minimum": float(rank["rho_spearman_vs_ref"].min()),
            }
        )
    if not layer.empty:
        rows.append(
            {
                "claim": "layer_trend_stability_vs_topk",
                "n": int(layer["n_layers"].sum()) if "n_layers" in layer else int(len(layer)),
                "median": float(layer["layer_trend_spearman_vs_ref"].median()),
                "minimum": float(layer["layer_trend_spearman_vs_ref"].min()),
            }
        )
    return pd.DataFrame(rows)


def mapping_strength() -> pd.DataFrame:
    ci = read_csv(ROOT / "experiments" / "05_controllability_mapping" / "outputs" / "mapping_bootstrap_ci.csv")
    metrics = read_csv(ROOT / "experiments" / "05_controllability_mapping" / "outputs" / "mapping_metrics.csv")
    rows = []
    if not ci.empty:
        for _, row in ci.iterrows():
            rows.append(
                {
                    "target": row["target"],
                    "metric": row["metric"],
                    "ci_low": row["ci_low"],
                    "median": row["median"],
                    "ci_high": row["ci_high"],
                    "supported": bool(float(row["ci_low"]) > 0.0),
                }
            )
    if metrics.empty:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def code_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def write_report(coverage: pd.DataFrame, topk: pd.DataFrame, mapping: pd.DataFrame) -> None:
    completed = coverage[coverage["status"].eq("completed")] if not coverage.empty else pd.DataFrame()
    by_family = (
        completed.groupby("family", as_index=False)
        .agg(n_sources=("source", "nunique"), rows=("rows", "sum"), models=("model", lambda s: ", ".join(sorted(set(map(str, s))))))
        .sort_values(["n_sources", "rows"], ascending=[False, False])
        if not completed.empty
        else pd.DataFrame()
    )
    lines = [
        "# Scale Robustness Matrix",
        "",
        "This report aggregates the checked-in evidence for broad model coverage, top-k robustness, and controllability mapping.",
        "",
        "It is intentionally lightweight: it does not rerun model inference. New RoBERTa, Llama, or Mistral runs are picked up automatically from the standard output CSVs.",
        "",
        "## Completed Family Coverage",
        code_table(by_family),
        "",
        "## Model Status Matrix",
        code_table(coverage.sort_values(["source", "family", "model"]) if not coverage.empty else coverage, 120),
        "",
        "## Top-k Robustness",
        code_table(topk),
        "",
        "## Controllability Mapping Bootstrap",
        code_table(mapping, 80),
        "",
        "## Reproduce",
        "```bash",
        "python scripts/run_scale_external_robustness_matrix.py",
        "```",
    ]
    (OUT / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for child in ["outputs", "reports", "config"]:
        (OUT / child).mkdir(parents=True, exist_ok=True)
    coverage = model_coverage()
    topk = topk_strength()
    mapping = mapping_strength()
    coverage.to_csv(OUT / "outputs" / "model_coverage_matrix.csv", index=False)
    topk.to_csv(OUT / "outputs" / "topk_robustness_strength.csv", index=False)
    mapping.to_csv(OUT / "outputs" / "mapping_bootstrap_strength.csv", index=False)
    metadata = {
        "command": "python scripts/run_scale_external_robustness_matrix.py",
        "inputs": [
            "experiments/controls/topk_robustness/outputs/",
            "experiments/04_uncertainty_steering/outputs/",
            "experiments/04_uncertainty_steering/decoder_main_battery/outputs/",
            "experiments/05_controllability_mapping/outputs/",
        ],
        "purpose": "Aggregate scale, model-family, top-k, and controllability robustness evidence.",
    }
    (OUT / "config" / "reproduce.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_report(coverage, topk, mapping)
    print(OUT)


if __name__ == "__main__":
    main()
