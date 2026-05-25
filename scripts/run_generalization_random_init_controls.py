from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoConfig, AutoModelForMaskedLM


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accessible_varentropy.metrics import softmax_np
from run_topk_gradient_regression_controls import (
    apply_hidden_step,
    full_kl_on_candidate_set,
    grad_var_logits,
    jaccard_topn,
    pca_basis,
    safe_tokenizer,
)
from run_uncertainty_steering_full_battery import (
    build_cases,
    build_records,
    fisher_geometry,
    full_logits,
    intervention_metrics,
    least_squares,
    mlm_head_jacobian_fast,
    normalize,
    parse_layers,
    projection,
    token_cluster_labels,
    topk_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oos-models",
        default="distilbert-base-uncased,bert-base-uncased,google/bert_uncased_L-2_H-128_A-2",
    )
    parser.add_argument(
        "--random-init-models",
        default="bert-base-uncased,google/bert_uncased_L-2_H-128_A-2",
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--subspace-ks", default="8")
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--max-prompts-per-task", type=int, default=8)
    parser.add_argument("--random-subspaces", type=int, default=1)
    parser.add_argument("--output-eps", type=float, default=0.05)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--semantic-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "experiments" / "controls",
    )
    return parser.parse_args()


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        for child in ["outputs", "reports", "config"]:
            (directory / child).mkdir(parents=True, exist_ok=True)


def orthonormalize(matrix: np.ndarray, k: int) -> np.ndarray:
    q, _r = np.linalg.qr(matrix)
    return q[:, :k].astype(np.float64)


def split_prompt_ids(prompt_table: pd.DataFrame, train_frac: float, seed: int) -> tuple[set[int], set[int]]:
    rng = np.random.default_rng(seed)
    train_ids: set[int] = set()
    test_ids: set[int] = set()
    for _task, group in prompt_table.groupby("task", dropna=False):
        ids = sorted(group["prompt_id"].astype(int).unique().tolist())
        rng.shuffle(ids)
        cut = max(1, min(len(ids) - 1, int(round(len(ids) * train_frac)))) if len(ids) > 1 else len(ids)
        train_ids.update(ids[:cut])
        test_ids.update(ids[cut:])
    return train_ids, test_ids


def records_by_split(
    records: list[dict[str, object]], train_ids: set[int], test_ids: set[int]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [record for record in records if int(record["prompt_id"]) in train_ids]
    test = [record for record in records if int(record["prompt_id"]) in test_ids]
    return train, test


def route_subspaces(
    records: list[dict[str, object]],
    layer: int,
    ks: list[int],
    seed: int,
    route_source: str,
    random_reps: int = 0,
) -> list[dict[str, object]]:
    if not records:
        return []
    hidden_dim = int(records[0]["hidden_states"][layer].shape[-1])
    max_k = min(max(ks), hidden_dim, max(1, len(records) - 1))
    if max_k < 1:
        return []
    states = np.stack(
        [
            record["hidden_states"][layer][0, int(record["mask_index"]), :].detach().cpu().numpy()
            for record in records
        ]
    ).astype(np.float64)
    result: list[dict[str, object]] = []
    state_basis = pca_basis(states, max_k)
    for k in ks:
        if k <= max_k:
            result.append(
                {
                    "route_source": route_source,
                    "subspace_family": "state_pca",
                    "subspace_k": int(k),
                    "rep": 0,
                    "basis": state_basis[:, :k],
                }
            )
    if layer < len(records[0]["hidden_states"]) - 1:
        deltas = np.stack(
            [
                (
                    record["hidden_states"][layer + 1][0, int(record["mask_index"]), :]
                    - record["hidden_states"][layer][0, int(record["mask_index"]), :]
                )
                .detach()
                .cpu()
                .numpy()
                for record in records
            ]
        ).astype(np.float64)
        delta_basis = pca_basis(deltas, max_k)
        for k in ks:
            if k <= max_k:
                result.append(
                    {
                        "route_source": route_source,
                        "subspace_family": "delta_pca",
                        "subspace_k": int(k),
                        "rep": 0,
                        "basis": delta_basis[:, :k],
                    }
                )
    rng = np.random.default_rng(seed + 1009 * layer)
    for rep in range(random_reps):
        random_basis = orthonormalize(rng.normal(size=(hidden_dim, max_k)), max_k)
        for k in ks:
            if k <= max_k:
                result.append(
                    {
                        "route_source": "random_route",
                        "subspace_family": "random",
                        "subspace_k": int(k),
                        "rep": int(rep),
                        "basis": random_basis[:, :k],
                    }
                )
    return result


def random_subspaces_only(
    records: list[dict[str, object]],
    layer: int,
    ks: list[int],
    seed: int,
    random_reps: int,
) -> list[dict[str, object]]:
    if not records or random_reps <= 0:
        return []
    hidden_dim = int(records[0]["hidden_states"][layer].shape[-1])
    max_k = min(max(ks), hidden_dim)
    rng = np.random.default_rng(seed + 1009 * layer)
    result: list[dict[str, object]] = []
    for rep in range(random_reps):
        random_basis = orthonormalize(rng.normal(size=(hidden_dim, max_k)), max_k)
        for k in ks:
            if k <= max_k:
                result.append(
                    {
                        "route_source": "random_route",
                        "subspace_family": "random",
                        "subspace_k": int(k),
                        "rep": int(rep),
                        "basis": random_basis[:, :k],
                    }
                )
    return result


def score_record_routes(
    *,
    model,
    model_name: str,
    init_state: str,
    record: dict[str, object],
    meta: pd.Series,
    layer: int,
    subspaces: list[dict[str, object]],
    top_k: int,
    embeddings: np.ndarray,
    semantic_clusters: int,
    output_eps: float | None,
    split: str,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    hidden = record["hidden_states"][layer][0, int(record["mask_index"]), :].detach()
    logits_before = full_logits(model, hidden)
    full_probs_before = softmax_np(logits_before[None, :])[0].astype(np.float64)
    token_ids, probs = topk_distribution(logits_before, top_k)
    selected_logits = logits_before[token_ids]
    geometry = fisher_geometry(probs)
    jac = mlm_head_jacobian_fast(model, hidden, token_ids)
    labels = token_cluster_labels(embeddings, token_ids, semantic_clusters)
    target_id = None if pd.isna(meta["target_id"]) else int(meta["target_id"])
    sorted_selected = np.sort(selected_logits)[::-1]
    confidence = float(probs[0])
    margin = float(sorted_selected[0] - sorted_selected[1]) if len(sorted_selected) > 1 else float("nan")
    score_rows: list[dict[str, object]] = []
    steering_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed + int(record["prompt_id"]) + 7919 * layer)

    for subspace in subspaces:
        basis = np.asarray(subspace["basis"], dtype=np.float64)
        jb = jac @ basis
        whitened = geometry.sqrt_fisher @ jb
        w_acc, rank = projection(whitened, geometry.w)
        w_orth = geometry.w - w_acc
        denom = max(float(geometry.w @ geometry.w), 1e-12)
        grad_entropy_z = whitened.T @ geometry.w
        grad_var_z = jb.T @ grad_var_logits(probs, geometry.u, geometry.varentropy)
        trace_fim = float(np.trace(jb.T @ geometry.fisher @ jb))
        common = {
            "model": model_name,
            "init_state": init_state,
            "split": split,
            "prompt_id": int(record["prompt_id"]),
            "task": meta["task"],
            "topic": meta["topic"],
            "observed_condition": meta["observed_condition"],
            "layer": int(layer),
            "route_source": subspace["route_source"],
            "subspace_family": subspace["subspace_family"],
            "subspace_k": int(subspace["subspace_k"]),
            "rep": int(subspace["rep"]),
            "top_k_output": int(top_k),
            "rank_whitened_jb": int(rank),
            "rho": float((w_acc @ w_acc) / denom),
            "v_access": float(w_acc @ w_acc),
            "v_inaccess": float(w_orth @ w_orth),
            "entropy": geometry.entropy,
            "varentropy": geometry.varentropy,
            "confidence": confidence,
            "margin": margin,
            "trace_fim": trace_fim,
            "jacobian_fro_norm": float(np.linalg.norm(jb)),
            "fisher_output_fro_norm": math.sqrt(max(trace_fim, 0.0)),
            "grad_entropy_proj_norm": float(np.linalg.norm(grad_entropy_z)),
            "grad_varentropy_proj_norm": float(np.linalg.norm(grad_var_z)),
            "logit_norm": float(np.linalg.norm(selected_logits)),
        }
        score_rows.append(common)
        if output_eps is None:
            continue
        z_acc = normalize(least_squares(whitened, w_acc))
        if float(np.linalg.norm(z_acc)) < 1e-12:
            continue
        unit_output_energy = float(np.linalg.norm(whitened @ z_acc))
        if unit_output_energy < 1e-12:
            continue
        first_order = float(grad_entropy_z @ z_acc)
        sign_base = 1.0 if first_order >= 0.0 else -1.0
        hidden_direction = basis @ z_acc
        step_scale = output_eps / unit_output_energy
        for sign_name, sign_multiplier in [("increase", 1.0), ("decrease", -1.0)]:
            signed = sign_multiplier * sign_base
            metrics = intervention_metrics(
                model=model,
                hidden=hidden,
                token_ids=token_ids,
                target_id=target_id,
                embeddings=embeddings,
                labels=labels,
                probs=probs,
                geometry=geometry,
                selected_logits=selected_logits,
                hidden_direction=hidden_direction,
                epsilon=step_scale,
                sign=signed,
            )
            logits_after = apply_hidden_step(model, hidden, hidden_direction, step_scale, signed)
            full_probs_after = softmax_np(logits_after[None, :])[0].astype(np.float64)
            candidate_mass_before = float(full_probs_before[token_ids].sum())
            candidate_mass_after = float(full_probs_after[token_ids].sum())
            expected = 1.0 if sign_name == "increase" else -1.0
            steering_rows.append(
                {
                    **common,
                    "direction": "accessible_ls",
                    "sign": sign_name,
                    "epsilon": output_eps,
                    "latent_step_norm": step_scale,
                    "fisher_output_energy": output_eps,
                    "direction_output_energy_unit": unit_output_energy,
                    "first_order_entropy": signed * step_scale * first_order,
                    "directional_success": int(expected * metrics["delta_entropy"] > 0.0),
                    "candidate_mass_before_full_vocab": candidate_mass_before,
                    "candidate_mass_after_full_vocab": candidate_mass_after,
                    "candidate_mass_retention_ratio": candidate_mass_after / max(candidate_mass_before, 1e-12),
                    "full_vocab_top5_jaccard": jaccard_topn(logits_before, logits_after, 5),
                    "full_vocab_top10_jaccard": jaccard_topn(logits_before, logits_after, 10),
                    "candidate_set_kl": full_kl_on_candidate_set(full_probs_before, full_probs_after, token_ids),
                    **metrics,
                }
            )
    return score_rows, steering_rows


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    return (
        scores.groupby(
            ["model", "init_state", "split", "route_source", "subspace_family", "subspace_k", "layer"],
            dropna=False,
        )
        .agg(
            n=("rho", "size"),
            rho_mean=("rho", "mean"),
            rho_std=("rho", "std"),
            v_access_mean=("v_access", "mean"),
            v_inaccess_mean=("v_inaccess", "mean"),
            grad_entropy_proj_norm_mean=("grad_entropy_proj_norm", "mean"),
            grad_varentropy_proj_norm_mean=("grad_varentropy_proj_norm", "mean"),
        )
        .reset_index()
    )


def summarize_steering(steering: pd.DataFrame) -> pd.DataFrame:
    if steering.empty:
        return pd.DataFrame()
    return (
        steering.groupby(
            ["model", "init_state", "split", "route_source", "subspace_family", "subspace_k", "sign"],
            dropna=False,
        )
        .agg(
            n=("abs_delta_entropy", "size"),
            abs_delta_entropy_mean=("abs_delta_entropy", "mean"),
            abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
            directional_success_rate=("directional_success", "mean"),
            selected_top1_changed_rate=("selected_top1_changed", "mean"),
            full_vocab_top10_jaccard_mean=("full_vocab_top10_jaccard", "mean"),
        )
        .reset_index()
    )


def oos_contrasts(scores: pd.DataFrame, steering: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test_scores = scores[(scores["split"] == "test") & (scores["init_state"] == "pretrained")].copy()
    for metric, frame in [("rho", test_scores), ("abs_delta_entropy", steering[steering["split"] == "test"])]:
        if frame.empty:
            continue
        for (model, family), group in frame.groupby(["model", "subspace_family"], dropna=False):
            train_route = group[group["route_source"] == "train_route"][metric].dropna()
            oracle = group[group["route_source"] == "test_oracle"][metric].dropna()
            random = frame[
                (frame["model"] == model)
                & (frame["route_source"] == "random_route")
                & (frame["subspace_k"].isin(group["subspace_k"].unique()))
            ][metric].dropna()
            for control_name, control_values in [("random_route_pooled", random), ("test_oracle", oracle)]:
                if train_route.empty or control_values.empty:
                    continue
                rows.append(
                    {
                        "model": model,
                        "subspace_family": family,
                        "metric": metric,
                        "route": "train_route",
                        "control": control_name,
                        "route_mean": float(train_route.mean()),
                        "control_mean": float(control_values.mean()),
                        "route_minus_control": float(train_route.mean() - control_values.mean()),
                        "route_over_control": float(train_route.mean() / max(abs(control_values.mean()), 1e-12)),
                        "n_route": int(len(train_route)),
                        "n_control": int(len(control_values)),
                    }
                )
    return pd.DataFrame(rows)


def load_model(model_name: str, init_state: str, seed: int):
    tokenizer = safe_tokenizer(model_name)
    if init_state == "pretrained":
        model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
    elif init_state == "random_init":
        torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForMaskedLM.from_config(config)
    else:
        raise ValueError(f"Unknown init state: {init_state}")
    model.eval()
    return tokenizer, model


def run_oos_for_model(args: argparse.Namespace, model_name: str, ks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tokenizer, model = load_model(model_name, "pretrained", args.seed)
    cases = build_cases(args.seed, args.max_prompts_per_task)
    prompt_table, records = build_records(tokenizer, model, cases, args.top_k)
    if not records:
        return prompt_table.assign(model=model_name), pd.DataFrame(), pd.DataFrame()
    train_ids, test_ids = split_prompt_ids(prompt_table, args.train_frac, args.seed)
    train_records, test_records = records_by_split(records, train_ids, test_ids)
    n_layers = len(records[0]["hidden_states"]) - 1
    layers = [layer for layer in parse_layers(args.layers, n_layers) if 1 <= layer <= n_layers]
    prompt_lookup = prompt_table.set_index("prompt_id")
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    score_rows: list[dict[str, object]] = []
    steering_rows: list[dict[str, object]] = []
    for layer in layers:
        train_subspaces = route_subspaces(train_records, layer, ks, args.seed, "train_route")
        test_oracle = route_subspaces(test_records, layer, ks, args.seed + 17, "test_oracle")
        random_routes = random_subspaces_only(train_records or records, layer, ks, args.seed + 31, args.random_subspaces)
        for split, split_records, subspaces in [
            ("train", train_records, train_subspaces + random_routes),
            ("test", test_records, train_subspaces + test_oracle + random_routes),
        ]:
            for record in split_records:
                meta = prompt_lookup.loc[int(record["prompt_id"])]
                scores, steering = score_record_routes(
                    model=model,
                    model_name=model_name,
                    init_state="pretrained",
                    record=record,
                    meta=meta,
                    layer=layer,
                    subspaces=subspaces,
                    top_k=args.top_k,
                    embeddings=embeddings,
                    semantic_clusters=args.semantic_clusters,
                    output_eps=args.output_eps,
                    split=split,
                    seed=args.seed,
                )
                score_rows.extend(scores)
                steering_rows.extend(steering)
    return (
        prompt_table.assign(model=model_name),
        pd.DataFrame(score_rows),
        pd.DataFrame(steering_rows),
    )


def run_random_init_for_model(args: argparse.Namespace, model_name: str, ks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_scores: list[pd.DataFrame] = []
    prompt_tables: list[pd.DataFrame] = []
    cases = build_cases(args.seed, args.max_prompts_per_task)
    for init_state in ["pretrained", "random_init"]:
        tokenizer, model = load_model(model_name, init_state, args.seed)
        prompt_table, records = build_records(tokenizer, model, cases, args.top_k)
        prompt_tables.append(prompt_table.assign(model=model_name, init_state=init_state))
        if not records:
            continue
        n_layers = len(records[0]["hidden_states"]) - 1
        layers = [layer for layer in parse_layers(args.layers, n_layers) if 1 <= layer <= n_layers]
        prompt_lookup = prompt_table.set_index("prompt_id")
        embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
        rows: list[dict[str, object]] = []
        for layer in layers:
            subspaces = route_subspaces(records, layer, ks, args.seed, "within_model") + random_subspaces_only(
                records, layer, ks, args.seed + 53, args.random_subspaces
            )
            for record in records:
                meta = prompt_lookup.loc[int(record["prompt_id"])]
                scores, _steering = score_record_routes(
                    model=model,
                    model_name=model_name,
                    init_state=init_state,
                    record=record,
                    meta=meta,
                    layer=layer,
                    subspaces=subspaces,
                    top_k=args.top_k,
                    embeddings=embeddings,
                    semantic_clusters=args.semantic_clusters,
                    output_eps=None,
                    split="all",
                    seed=args.seed,
                )
                rows.extend(scores)
        all_scores.append(pd.DataFrame(rows))
    return pd.concat(prompt_tables, ignore_index=True), pd.concat(all_scores, ignore_index=True)


def random_init_contrasts(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if scores.empty:
        return pd.DataFrame()
    for (model, source, family), group in scores.groupby(["model", "route_source", "subspace_family"], dropna=False):
        pre = group[group["init_state"] == "pretrained"]["rho"].dropna()
        rnd = group[group["init_state"] == "random_init"]["rho"].dropna()
        if pre.empty or rnd.empty:
            continue
        # Match only by comparable row coordinates when possible.
        keys = ["prompt_id", "layer", "route_source", "subspace_family", "subspace_k", "rep"]
        pivot = group.pivot_table(index=keys, columns="init_state", values="rho", aggfunc="mean")
        corr = float("nan")
        if "pretrained" in pivot and "random_init" in pivot and len(pivot.dropna()) >= 3:
            corr = float(spearmanr(pivot.dropna()["pretrained"], pivot.dropna()["random_init"]).statistic)
        rows.append(
            {
                "model": model,
                "route_source": source,
                "subspace_family": family,
                "n_pretrained": int(len(pre)),
                "n_random_init": int(len(rnd)),
                "rho_pretrained_mean": float(pre.mean()),
                "rho_random_init_mean": float(rnd.mean()),
                "rho_pretrained_minus_random": float(pre.mean() - rnd.mean()),
                "rho_pretrained_over_random": float(pre.mean() / max(abs(rnd.mean()), 1e-12)),
                "rho_rank_spearman_pretrained_vs_random": corr,
            }
        )
    return pd.DataFrame(rows)


def write_oos_report(out_dir: Path, score_summary: pd.DataFrame, steering_summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    lines = [
        "# Control: Out-of-Sample Route Generalization",
        "",
        "Train-route subspaces are estimated on train prompts and evaluated on held-out prompts. Test-oracle subspaces are estimated on held-out prompts and serve as an upper-bound diagnostic, not a deployable route.",
        "",
        "## Score Summary",
        "",
        "```text",
        score_summary.head(40).to_string(index=False) if not score_summary.empty else "(empty)",
        "```",
        "",
        "## Steering Summary",
        "",
        "```text",
        steering_summary.head(40).to_string(index=False) if not steering_summary.empty else "(empty)",
        "```",
        "",
        "## Contrasts",
        "",
        "```text",
        contrasts.to_string(index=False) if not contrasts.empty else "(empty)",
        "```",
        "",
        "## Files",
        "",
        "```text",
        "prompt_tables.csv",
        "oos_scores.csv",
        "oos_steering_records.csv",
        "oos_score_summary.csv",
        "oos_steering_summary.csv",
        "oos_contrasts.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_random_report(out_dir: Path, summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    lines = [
        "# Control: Random-Init Versus Pretrained",
        "",
        "This control compares the same architecture before and after learned weights. If pretrained and random-init rho structure diverges, accessibility is less likely to be a purely architectural/Jacobian degree-of-freedom artifact.",
        "",
        "## Summary",
        "",
        "```text",
        summary.head(40).to_string(index=False) if not summary.empty else "(empty)",
        "```",
        "",
        "## Pretrained Vs Random Contrasts",
        "",
        "```text",
        contrasts.to_string(index=False) if not contrasts.empty else "(empty)",
        "```",
        "",
        "## Files",
        "",
        "```text",
        "prompt_tables.csv",
        "random_init_scores.csv",
        "random_init_summary.csv",
        "pretrained_vs_random_contrasts.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ks = [int(part) for part in args.subspace_ks.split(",") if part.strip()]
    oos_dir = args.out_root / "out_of_sample_generalization"
    random_dir = args.out_root / "random_init_vs_pretrained"
    ensure_dirs(oos_dir, random_dir)

    oos_prompts: list[pd.DataFrame] = []
    oos_scores: list[pd.DataFrame] = []
    oos_steering: list[pd.DataFrame] = []
    for model_name in [part.strip() for part in args.oos_models.split(",") if part.strip()]:
        print(f"oos_model {model_name}", flush=True)
        prompt_table, score_table, steering_table = run_oos_for_model(args, model_name, ks)
        oos_prompts.append(prompt_table)
        oos_scores.append(score_table)
        oos_steering.append(steering_table)
    oos_prompt_table = pd.concat(oos_prompts, ignore_index=True) if oos_prompts else pd.DataFrame()
    oos_score_table = pd.concat(oos_scores, ignore_index=True) if oos_scores else pd.DataFrame()
    oos_steering_table = pd.concat(oos_steering, ignore_index=True) if oos_steering else pd.DataFrame()
    oos_score_summary = summarize_scores(oos_score_table)
    oos_steering_summary = summarize_steering(oos_steering_table)
    oos_contrast_table = oos_contrasts(oos_score_table, oos_steering_table)
    oos_prompt_table.to_csv(oos_dir / "outputs" / "prompt_tables.csv", index=False)
    oos_score_table.to_csv(oos_dir / "outputs" / "oos_scores.csv", index=False)
    oos_steering_table.to_csv(oos_dir / "outputs" / "oos_steering_records.csv", index=False)
    oos_score_summary.to_csv(oos_dir / "outputs" / "oos_score_summary.csv", index=False)
    oos_steering_summary.to_csv(oos_dir / "outputs" / "oos_steering_summary.csv", index=False)
    oos_contrast_table.to_csv(oos_dir / "outputs" / "oos_contrasts.csv", index=False)
    write_oos_report(oos_dir, oos_score_summary, oos_steering_summary, oos_contrast_table)

    random_prompts: list[pd.DataFrame] = []
    random_scores: list[pd.DataFrame] = []
    for model_name in [part.strip() for part in args.random_init_models.split(",") if part.strip()]:
        print(f"random_init_model {model_name}", flush=True)
        prompt_table, score_table = run_random_init_for_model(args, model_name, ks)
        random_prompts.append(prompt_table)
        random_scores.append(score_table)
    random_prompt_table = pd.concat(random_prompts, ignore_index=True) if random_prompts else pd.DataFrame()
    random_score_table = pd.concat(random_scores, ignore_index=True) if random_scores else pd.DataFrame()
    random_summary = summarize_scores(random_score_table)
    random_contrast_table = random_init_contrasts(random_score_table)
    random_prompt_table.to_csv(random_dir / "outputs" / "prompt_tables.csv", index=False)
    random_score_table.to_csv(random_dir / "outputs" / "random_init_scores.csv", index=False)
    random_summary.to_csv(random_dir / "outputs" / "random_init_summary.csv", index=False)
    random_contrast_table.to_csv(random_dir / "outputs" / "pretrained_vs_random_contrasts.csv", index=False)
    write_random_report(random_dir, random_summary, random_contrast_table)

    for directory, control_id in [(oos_dir, "out_of_sample_generalization"), (random_dir, "random_init_vs_pretrained")]:
        config = {
            "control_id": control_id,
            "top_k": args.top_k,
            "subspace_ks": args.subspace_ks,
            "layers": args.layers,
            "max_prompts_per_task": args.max_prompts_per_task,
            "random_subspaces": args.random_subspaces,
            "output_eps": args.output_eps,
            "seed": args.seed,
            "minimal_command": "python scripts/run_generalization_random_init_controls.py",
        }
        pd.Series(config).to_json(directory / "config" / "reproduce.json", indent=2)


if __name__ == "__main__":
    main()
