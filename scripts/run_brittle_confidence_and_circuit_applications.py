from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM

from accessible_varentropy.metrics import softmax_np
from run_topk_gradient_regression_controls import (
    build_cases,
    build_records,
    full_kl_on_candidate_set,
    grad_var_logits,
    jaccard_topn,
    make_subspaces,
    parse_layers,
    safe_tokenizer,
)
from run_uncertainty_steering_full_battery import (
    fisher_geometry,
    full_logits,
    intervention_metrics,
    least_squares,
    mlm_head_jacobian_fast,
    normalize,
    projection,
    token_cluster_labels,
    topk_distribution,
)


BRITTLE_OUT = ROOT / "applications" / "05_brittle_confidence"
CIRCUITS_OUT = ROOT / "applications" / "04_uncertainty_circuits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="distilbert-base-uncased,bert-base-uncased,google/bert_uncased_L-2_H-128_A-2",
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--subspace-k", type=int, default=8)
    parser.add_argument("--max-prompts-per-task", type=int, default=6)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--output-eps", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--max-brittle-pairs", type=int, default=48)
    parser.add_argument("--brittle-out", type=Path, default=BRITTLE_OUT)
    parser.add_argument("--circuits-out", type=Path, default=CIRCUITS_OUT)
    return parser.parse_args()


def entropy_varentropy(probs: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    log_p = np.log(p)
    entropy = float(-np.sum(p * log_p))
    varentropy = float(np.sum(p * ((-log_p - entropy) ** 2)))
    return entropy, varentropy


def token_id_for_answer(tokenizer, target: str | float | None) -> int | None:
    if target is None or pd.isna(target):
        return None
    encoded = tokenizer.encode(str(target), add_special_tokens=False)
    if not encoded:
        return None
    return int(encoded[0])


def encode_masked(tokenizer, prompt: str):
    if prompt.count("[MASK]") != 1:
        raise ValueError("Prompt must contain exactly one [MASK].")
    encoded = tokenizer(prompt.replace("[MASK]", tokenizer.mask_token), return_tensors="pt")
    mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    if len(mask_positions) != 1:
        raise ValueError("Expected exactly one tokenizer mask token.")
    return encoded, int(mask_positions[0, 0].item())


def evaluate_masked_prompt(tokenizer, model, prompt: str, top_k: int, answer_id: int | None) -> dict[str, object]:
    encoded, mask_index = encode_masked(tokenizer, prompt)
    with torch.no_grad():
        output = model(**encoded)
    logits = output.logits[0, mask_index, :].detach().float().cpu().numpy().astype(np.float64)
    full_probs = softmax_np(logits[None, :])[0].astype(np.float64)
    token_ids = np.argpartition(logits, -top_k)[-top_k:]
    token_ids = token_ids[np.argsort(logits[token_ids])[::-1]].astype(np.int64)
    selected_probs = softmax_np(logits[token_ids][None, :])[0].astype(np.float64)
    entropy, varentropy = entropy_varentropy(selected_probs)
    sorted_logits = np.sort(logits[token_ids])[::-1]
    top1_id = int(token_ids[0])
    top10 = [int(x) for x in np.argsort(logits)[-10:][::-1].tolist()]
    answer_prob = float(full_probs[answer_id]) if answer_id is not None else float(full_probs[top1_id])
    correct = int(answer_id is not None and top1_id == answer_id)
    return {
        "logits": logits,
        "full_probs": full_probs,
        "candidate_ids": token_ids,
        "selected_probs": selected_probs,
        "top10": top10,
        "top1_id": top1_id,
        "top1_token": tokenizer.decode([top1_id]).strip(),
        "confidence_full": float(full_probs[top1_id]),
        "confidence_topk": float(selected_probs[0]),
        "entropy": entropy,
        "varentropy": varentropy,
        "margin": float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) > 1 else float("nan"),
        "answer_prob": answer_prob,
        "correct": correct,
    }


SYNONYMS = {
    "person": "individual",
    "child": "kid",
    "customer": "client",
    "coach": "trainer",
    "student": "pupil",
    "doctor": "physician",
    "walked": "moved",
    "looked": "glanced",
    "talked": "spoke",
    "studied": "examined",
}


def perturb_prompt(prompt: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = [
        ("neutral_prefix", "For reference, " + prompt),
        ("neutral_suffix", prompt + " This extra sentence is unrelated."),
        ("quiz_format", "In a short quiz, " + prompt),
    ]
    rewrites = [
        (r"^The capital of (.+) is \[MASK\]\.$", r"In geography, the capital of \1 is [MASK]."),
        (r"^The opposite of (.+) is \[MASK\]\.$", r"In ordinary usage, the opposite of \1 is [MASK]."),
        (r"^The plural of (.+) is \[MASK\]\.$", r"In English, the plural of \1 is [MASK]."),
        (r"^A person who (.+) is a \[MASK\]\.$", r"Someone who \1 is a [MASK]."),
        (r"^\[MASK\] painted (.+)\.$", r"The person who painted \1 was [MASK]."),
    ]
    for pattern, repl in rewrites:
        changed = re.sub(pattern, repl, prompt)
        if changed != prompt and changed.count("[MASK]") == 1:
            variants.append(("template_rewrite", changed))
            break
    tokens = prompt.split()
    changed_tokens = []
    changed = False
    for token in tokens:
        clean = re.sub(r"[^A-Za-z]", "", token).lower()
        if clean in SYNONYMS and not changed:
            changed_tokens.append(token.replace(clean, SYNONYMS[clean]))
            changed = True
        else:
            changed_tokens.append(token)
    synonym_prompt = " ".join(changed_tokens)
    if changed and synonym_prompt.count("[MASK]") == 1:
        variants.append(("synonym_substitution", synonym_prompt))
    seen = set()
    unique = []
    for kind, text in variants:
        if text not in seen and text.count("[MASK]") == 1:
            seen.add(text)
            unique.append((kind, text))
    return unique


def matched_brittle_pairs(scores: pd.DataFrame, prompts: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    base = scores[
        scores["top_k_output"].eq(32)
        & scores["subspace_family"].isin(["state_pca", "delta_pca"])
        & scores["observed_condition"].isin(["correct", "error"])
    ].copy()
    base = base.merge(
        prompts[["model", "prompt_id", "prompt", "target", "target_id", "final_top1_token"]],
        on=["model", "prompt_id"],
        how="left",
    )
    rows: list[dict[str, object]] = []
    for keys, group in base.groupby(["model", "task", "observed_condition", "layer", "subspace_family"], sort=True):
        if len(group) < 4:
            continue
        high_conf = group[
            (group["confidence"] >= group["confidence"].quantile(0.50))
            & (group["entropy"] <= group["entropy"].quantile(0.60))
            & (group["margin"] >= group["margin"].quantile(0.40))
        ].copy()
        if len(high_conf) < 4:
            high_conf = group.nlargest(min(len(group), max(4, len(group) // 2)), "confidence").copy()
        low_pool = high_conf[high_conf["rho"] <= high_conf["rho"].quantile(0.35)].copy()
        high_pool = high_conf[high_conf["rho"] >= high_conf["rho"].quantile(0.65)].copy()
        if low_pool.empty or high_pool.empty:
            continue
        scale_cols = ["confidence", "entropy", "margin"]
        means = high_conf[scale_cols].mean()
        stds = high_conf[scale_cols].std().replace(0.0, 1.0).fillna(1.0)
        used_high: set[int] = set()
        pair_counter = 0
        for low_idx, low in low_pool.sort_values("rho").iterrows():
            candidates = high_pool[~high_pool.index.isin(used_high)]
            if candidates.empty:
                break
            low_vec = ((low[scale_cols] - means) / stds).to_numpy(dtype=float)
            distances = []
            for high_idx, high in candidates.iterrows():
                high_vec = ((high[scale_cols] - means) / stds).to_numpy(dtype=float)
                distances.append((float(np.linalg.norm(low_vec - high_vec)), high_idx))
            distance, high_idx = min(distances)
            high = candidates.loc[high_idx]
            used_high.add(int(high_idx))
            pair_id = f"{keys[0]}:{keys[1]}:{keys[2]}:{keys[3]}:{keys[4]}:{pair_counter}"
            pair_counter += 1
            for label, item in [("low_rho", low), ("high_rho", high)]:
                rows.append(
                    {
                        "pair_id": pair_id,
                        "rho_group": label,
                        "match_distance": distance,
                        "model": keys[0],
                        "task": keys[1],
                        "observed_condition": keys[2],
                        "layer": int(keys[3]),
                        "subspace_family": keys[4],
                        "prompt_id": int(item["prompt_id"]),
                        "prompt": item["prompt"],
                        "target": item["target"],
                        "target_id": item["target_id"],
                        "rho": float(item["rho"]),
                        "confidence": float(item["confidence"]),
                        "entropy": float(item["entropy"]),
                        "varentropy": float(item["varentropy"]),
                        "margin": float(item["margin"]),
                        "final_top1_token": item["final_top1_token"],
                    }
                )
    matched = pd.DataFrame(rows)
    if matched.empty:
        return matched
    keep_pairs = matched[["pair_id", "match_distance"]].drop_duplicates().sort_values("match_distance").head(max_pairs)
    return matched[matched["pair_id"].isin(keep_pairs["pair_id"])].reset_index(drop=True)


def run_brittle_confidence(args: argparse.Namespace) -> None:
    out = args.brittle_out
    for child in ["outputs", "reports", "config"]:
        (out / child).mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "topk_scores.csv")
    prompts = pd.read_csv(ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "prompt_tables.csv")
    matched = matched_brittle_pairs(scores, prompts, args.max_brittle_pairs)
    matched.to_csv(out / "outputs" / "matched_high_confidence_groups.csv", index=False)
    rows: list[dict[str, object]] = []
    for model_name, model_group in matched.groupby("model", sort=True):
        print(f"brittle_load_model {model_name}", flush=True)
        tokenizer = safe_tokenizer(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
        model.eval()
        baseline_cache: dict[tuple[int, str], dict[str, object]] = {}
        for _idx, row in model_group.iterrows():
            prompt = str(row["prompt"])
            answer_id = token_id_for_answer(tokenizer, row["target"])
            if answer_id is None and not pd.isna(row["target_id"]):
                answer_id = int(row["target_id"])
            try:
                before = baseline_cache.get((int(row["prompt_id"]), prompt))
                if before is None:
                    before = evaluate_masked_prompt(tokenizer, model, prompt, args.top_k, answer_id)
                    baseline_cache[(int(row["prompt_id"]), prompt)] = before
            except Exception as exc:  # noqa: BLE001
                rows.append({**row.to_dict(), "variant_type": "baseline_error", "error": str(exc)})
                continue
            for variant_type, variant_prompt in perturb_prompt(prompt):
                try:
                    after = evaluate_masked_prompt(tokenizer, model, variant_prompt, args.top_k, answer_id)
                except Exception as exc:  # noqa: BLE001
                    rows.append({**row.to_dict(), "variant_type": variant_type, "variant_prompt": variant_prompt, "error": str(exc)})
                    continue
                candidate_ids = before["candidate_ids"]
                candidate_kl = full_kl_on_candidate_set(before["full_probs"], after["full_probs"], candidate_ids)
                top10_jaccard = len(set(before["top10"]) & set(after["top10"])) / max(
                    len(set(before["top10"]) | set(after["top10"])), 1
                )
                correctness_lost = int(before["correct"] == 1 and after["correct"] == 0)
                answer_flipped = int(before["top1_id"] != after["top1_id"])
                prob_drop = float(before["answer_prob"] - after["answer_prob"])
                entropy_delta = float(after["entropy"] - before["entropy"])
                before_brier = (float(before["confidence_full"]) - float(before["correct"])) ** 2
                after_brier = (float(after["confidence_full"]) - float(after["correct"])) ** 2
                fragility = (
                    answer_flipped
                    + correctness_lost
                    + max(0.0, prob_drop)
                    + max(0.0, entropy_delta)
                    + candidate_kl
                    + (1.0 - top10_jaccard)
                )
                rows.append(
                    {
                        **row.to_dict(),
                        "variant_type": variant_type,
                        "variant_prompt": variant_prompt,
                        "top1_before": before["top1_token"],
                        "top1_after": after["top1_token"],
                        "answer_flipped": answer_flipped,
                        "answer_prob_before": before["answer_prob"],
                        "answer_prob_after": after["answer_prob"],
                        "answer_prob_drop": prob_drop,
                        "entropy_before_eval": before["entropy"],
                        "entropy_after_eval": after["entropy"],
                        "entropy_increase": entropy_delta,
                        "varentropy_before_eval": before["varentropy"],
                        "varentropy_after_eval": after["varentropy"],
                        "varentropy_delta": after["varentropy"] - before["varentropy"],
                        "confidence_before_full": before["confidence_full"],
                        "confidence_after_full": after["confidence_full"],
                        "confidence_drop": before["confidence_full"] - after["confidence_full"],
                        "correct_before_eval": before["correct"],
                        "correct_after_eval": after["correct"],
                        "correctness_lost": correctness_lost,
                        "candidate_set_kl": candidate_kl,
                        "top10_jaccard": top10_jaccard,
                        "top10_jaccard_drop": 1.0 - top10_jaccard,
                        "brier_degradation": after_brier - before_brier,
                        "fragility_score": fragility,
                    }
                )
    records = pd.DataFrame(rows)
    records.to_csv(out / "outputs" / "brittle_perturbation_records.csv", index=False)
    summary = records.groupby(["rho_group", "variant_type"], as_index=False).agg(
        n=("prompt_id", "count"),
        rho_mean=("rho", "mean"),
        confidence_mean=("confidence", "mean"),
        entropy_mean=("entropy", "mean"),
        margin_mean=("margin", "mean"),
        answer_flip_rate=("answer_flipped", "mean"),
        correctness_loss_rate=("correctness_lost", "mean"),
        answer_prob_drop_mean=("answer_prob_drop", "mean"),
        entropy_increase_mean=("entropy_increase", "mean"),
        candidate_kl_mean=("candidate_set_kl", "mean"),
        top10_jaccard_drop_mean=("top10_jaccard_drop", "mean"),
        brier_degradation_mean=("brier_degradation", "mean"),
        fragility_score_mean=("fragility_score", "mean"),
    )
    summary.to_csv(out / "outputs" / "brittle_summary.csv", index=False)
    paired = (
        records.groupby(["pair_id", "rho_group"], as_index=False)
        .agg(
            fragility_score=("fragility_score", "mean"),
            answer_flip_rate=("answer_flipped", "mean"),
            answer_prob_drop=("answer_prob_drop", "mean"),
            entropy_increase=("entropy_increase", "mean"),
            candidate_kl=("candidate_set_kl", "mean"),
            top10_drop=("top10_jaccard_drop", "mean"),
        )
        .pivot(index="pair_id", columns="rho_group")
    )
    contrast_rows = []
    if not paired.empty and ("fragility_score", "low_rho") in paired.columns and ("fragility_score", "high_rho") in paired.columns:
        for metric in ["fragility_score", "answer_flip_rate", "answer_prob_drop", "entropy_increase", "candidate_kl", "top10_drop"]:
            diff = paired[(metric, "low_rho")] - paired[(metric, "high_rho")]
            contrast_rows.append(
                {
                    "metric": metric,
                    "n_pairs": int(diff.notna().sum()),
                    "low_minus_high_mean": float(diff.mean()),
                    "low_greater_than_high_rate": float((diff > 0).mean()),
                    "low_mean": float(paired[(metric, "low_rho")].mean()),
                    "high_mean": float(paired[(metric, "high_rho")].mean()),
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(out / "outputs" / "brittle_matched_contrasts.csv", index=False)
    write_brittle_report(out, matched, summary, contrasts)
    write_config(out, "brittle confidence", args)


def route_intervention(
    *,
    model,
    hidden: torch.Tensor,
    token_ids: np.ndarray,
    target_id: int | None,
    embeddings: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    geometry,
    selected_logits: np.ndarray,
    jb: np.ndarray,
    whitened: np.ndarray,
    basis: np.ndarray,
    z_unit: np.ndarray,
    sign_name: str,
    output_eps: float,
    logits_before: np.ndarray,
) -> dict[str, object] | None:
    output_unit = float(np.linalg.norm(whitened @ z_unit))
    if output_unit <= 1e-12:
        return None
    hidden_direction = basis @ z_unit
    grad_z = whitened.T @ geometry.w
    first_order = float(grad_z @ z_unit)
    sign_base = 1.0 if first_order >= 0.0 else -1.0
    signed = sign_base if sign_name == "increase" else -sign_base
    step_scale = output_eps / output_unit
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
    with torch.no_grad():
        stepped = hidden + torch.as_tensor(signed * step_scale * hidden_direction, dtype=hidden.dtype)
        logits_after = full_logits(model, stepped)
    full_probs_before = softmax_np(logits_before[None, :])[0].astype(np.float64)
    full_probs_after = softmax_np(logits_after[None, :])[0].astype(np.float64)
    return {
        "sign": sign_name,
        "latent_step_norm": step_scale,
        "fisher_output_energy": output_eps,
        "direction_output_energy_unit": output_unit,
        "first_order_entropy": signed * step_scale * first_order,
        "candidate_set_kl": full_kl_on_candidate_set(full_probs_before, full_probs_after, token_ids),
        "full_vocab_top10_jaccard": jaccard_topn(logits_before, logits_after, 10),
        **metrics,
    }


def run_uncertainty_circuits(args: argparse.Namespace) -> None:
    out = args.circuits_out
    for child in ["outputs", "reports", "config"]:
        (out / child).mkdir(parents=True, exist_ok=True)
    all_routes: list[dict[str, object]] = []
    all_interventions: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    cases = build_cases(args.seed, args.max_prompts_per_task)
    for model_name in [part.strip() for part in args.models.split(",") if part.strip()]:
        try:
            print(f"circuits_load_model {model_name}", flush=True)
            tokenizer = safe_tokenizer(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
            model.eval()
            prompt_table, records = build_records(tokenizer, model, cases, args.top_k)
            n_layers = len(records[0]["hidden_states"]) - 1
            layers = [layer for layer in parse_layers(args.layers, n_layers) if 1 <= layer <= n_layers]
            embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
            prompt_lookup = prompt_table.set_index("prompt_id")
            rng = np.random.default_rng(args.seed + len(all_routes))
            for layer in layers:
                subspaces = make_subspaces(records, layer, [args.subspace_k], 1, args.seed)
                for rec_idx, record in enumerate(records):
                    if rec_idx % 8 == 0:
                        print(f"circuits_prompt {model_name} layer={layer} {rec_idx + 1}/{len(records)}", flush=True)
                    meta = prompt_lookup.loc[int(record["prompt_id"])]
                    hidden = record["hidden_states"][layer][0, int(record["mask_index"]), :].detach()
                    logits_before = full_logits(model, hidden)
                    token_ids, probs = topk_distribution(logits_before, args.top_k)
                    selected_logits = logits_before[token_ids]
                    geometry = fisher_geometry(probs)
                    jac = mlm_head_jacobian_fast(model, hidden, token_ids)
                    labels = token_cluster_labels(embeddings, token_ids, 4)
                    target_id = None if pd.isna(meta["target_id"]) else int(meta["target_id"])
                    candidates = []
                    for family, subspace_k, rep, basis in subspaces:
                        jb = jac @ basis
                        whitened = geometry.sqrt_fisher @ jb
                        w_acc, rank = projection(whitened, geometry.w)
                        w_orth = geometry.w - w_acc
                        denom = max(float(geometry.w @ geometry.w), 1e-12)
                        rho = float((w_acc @ w_acc) / denom)
                        grad_entropy_z = whitened.T @ geometry.w
                        grad_var_z = jb.T @ grad_var_logits(probs, geometry.u, geometry.varentropy)
                        z_acc = normalize(least_squares(whitened, w_acc))
                        z_entropy = normalize(grad_entropy_z)
                        z_var = normalize(grad_var_z)
                        candidates.append(
                            {
                                "model": model_name,
                                "prompt_id": int(record["prompt_id"]),
                                "prompt": meta["prompt"],
                                "target": meta["target"],
                                "task": meta["task"],
                                "topic": meta["topic"],
                                "observed_condition": meta["observed_condition"],
                                "layer": int(layer),
                                "subspace_family": family,
                                "subspace_k": int(subspace_k),
                                "rep": int(rep),
                                "rho": rho,
                                "v_access": float(w_acc @ w_acc),
                                "v_inaccess": float(w_orth @ w_orth),
                                "entropy": geometry.entropy,
                                "varentropy": geometry.varentropy,
                                "confidence": float(probs[0]),
                                "rank_whitened_jb": int(rank),
                                "grad_entropy_proj_norm": float(np.linalg.norm(grad_entropy_z)),
                                "grad_varentropy_proj_norm": float(np.linalg.norm(grad_var_z)),
                                "basis": basis,
                                "jb": jb,
                                "whitened": whitened,
                                "z_accessible": z_acc,
                                "z_entropy_gradient": z_entropy,
                                "z_varentropy_gradient": z_var,
                            }
                        )
                    route_public = [
                        {k: v for k, v in cand.items() if k not in {"basis", "jb", "whitened", "z_accessible", "z_entropy_gradient", "z_varentropy_gradient"}}
                        for cand in candidates
                    ]
                    for row in route_public:
                        all_routes.append(row)
                    nonrandom = [c for c in candidates if c["subspace_family"] != "random"]
                    randoms = [c for c in candidates if c["subspace_family"] == "random"]
                    if not nonrandom:
                        continue
                    selected = [
                        ("high_rho_route", max(nonrandom, key=lambda c: c["rho"]), "accessible"),
                        ("low_rho_route", min(nonrandom, key=lambda c: c["rho"]), "accessible"),
                        (
                            "entropy_gradient_route",
                            max(nonrandom, key=lambda c: c["grad_entropy_proj_norm"]),
                            "entropy_gradient",
                        ),
                        (
                            "varentropy_gradient_route",
                            max(nonrandom, key=lambda c: c["grad_varentropy_proj_norm"]),
                            "varentropy_gradient",
                        ),
                    ]
                    if randoms:
                        selected.append(("random_route", randoms[0], "accessible"))
                    for route_role, cand, direction_kind in selected:
                        z = cand[f"z_{direction_kind}"] if direction_kind != "accessible" else cand["z_accessible"]
                        if float(np.linalg.norm(z)) <= 1e-12:
                            continue
                        for sign_name in ["increase", "decrease"]:
                            metrics = route_intervention(
                                model=model,
                                hidden=hidden,
                                token_ids=token_ids,
                                target_id=target_id,
                                embeddings=embeddings,
                                labels=labels,
                                probs=probs,
                                geometry=geometry,
                                selected_logits=selected_logits,
                                jb=cand["jb"],
                                whitened=cand["whitened"],
                                basis=cand["basis"],
                                z_unit=z,
                                sign_name=sign_name,
                                output_eps=args.output_eps,
                                logits_before=logits_before,
                            )
                            if metrics is None:
                                continue
                            all_interventions.append(
                                {
                                    **{k: v for k, v in cand.items() if k not in {"basis", "jb", "whitened", "z_accessible", "z_entropy_gradient", "z_varentropy_gradient"}},
                                    "route_role": route_role,
                                    "direction_kind": direction_kind,
                                    **metrics,
                                }
                            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"model": model_name, "reason": str(exc).splitlines()[0]})
    route_scores = pd.DataFrame(all_routes)
    interventions = pd.DataFrame(all_interventions)
    route_scores.to_csv(out / "outputs" / "route_scores.csv", index=False)
    interventions.to_csv(out / "outputs" / "circuit_intervention_records.csv", index=False)
    pd.DataFrame(skipped).to_csv(out / "outputs" / "skipped_models.csv", index=False)
    summary = interventions.groupby(["route_role", "direction_kind", "sign"], as_index=False).agg(
        n=("prompt_id", "count"),
        rho_mean=("rho", "mean"),
        abs_delta_entropy_mean=("abs_delta_entropy", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy", "mean"),
        selected_top1_changed_rate=("selected_top1_changed", "mean"),
        target_correct_changed_rate=("target_correct_changed", "mean"),
        full_vocab_top10_jaccard_mean=("full_vocab_top10_jaccard", "mean"),
        candidate_kl_mean=("candidate_set_kl", "mean"),
    )
    summary.to_csv(out / "outputs" / "circuit_route_summary.csv", index=False)
    contrasts = circuit_contrasts(interventions)
    contrasts.to_csv(out / "outputs" / "circuit_route_contrasts.csv", index=False)
    corr = circuit_correlations(interventions)
    corr.to_csv(out / "outputs" / "rho_causal_effect_correlations.csv", index=False)
    write_circuits_report(out, summary, contrasts, corr, pd.DataFrame(skipped))
    write_config(out, "uncertainty circuits", args)


def circuit_contrasts(interventions: pd.DataFrame) -> pd.DataFrame:
    if interventions.empty:
        return pd.DataFrame()
    rows = []
    summary = interventions.groupby(["model", "prompt_id", "layer", "sign", "route_role"], as_index=False).agg(
        abs_delta_entropy=("abs_delta_entropy", "mean"),
        abs_delta_varentropy=("abs_delta_varentropy", "mean"),
        selected_top1_changed=("selected_top1_changed", "mean"),
        full_vocab_top10_jaccard=("full_vocab_top10_jaccard", "mean"),
    )
    piv = summary.pivot_table(index=["model", "prompt_id", "layer", "sign"], columns="route_role")
    for control in ["low_rho_route", "random_route", "entropy_gradient_route", "varentropy_gradient_route"]:
        if ("abs_delta_entropy", "high_rho_route") not in piv.columns or ("abs_delta_entropy", control) not in piv.columns:
            continue
        for metric in ["abs_delta_entropy", "abs_delta_varentropy"]:
            diff = piv[(metric, "high_rho_route")] - piv[(metric, control)]
            ratio = piv[(metric, "high_rho_route")] / piv[(metric, control)].replace(0.0, np.nan)
            rows.append(
                {
                    "comparison": f"high_rho_route_vs_{control}",
                    "metric": metric,
                    "n": int(diff.notna().sum()),
                    "high_minus_control_mean": float(diff.mean()),
                    "high_over_control_mean": float(ratio.mean()),
                    "high_better_rate": float((diff > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def circuit_correlations(interventions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, frame in [
        ("all", interventions),
        ("accessible_routes_only", interventions[interventions["direction_kind"].eq("accessible")]),
    ]:
        for outcome in ["abs_delta_entropy", "abs_delta_varentropy", "selected_top1_changed", "candidate_set_kl"]:
            local = frame[["rho", outcome]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(local) < 4 or local["rho"].nunique() < 2 or local[outcome].nunique() < 2:
                corr = float("nan")
            else:
                corr = float(spearmanr(local["rho"], local[outcome]).correlation)
            rows.append({"subset": subset, "outcome": outcome, "n": int(len(local)), "spearman": corr})
    return pd.DataFrame(rows)


def write_brittle_report(out: Path, matched: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    lines = [
        "# Application 5: Brittle Confidence",
        "",
        "This application asks whether high-confidence predictions with low uncertainty accessibility are more fragile under prompt perturbations that should be semantically innocuous.",
        "",
        "Matched groups control for model, task, observed correctness, layer, subspace family, confidence, entropy, and margin as much as the checked-in prompt pool allows.",
        "",
        "## Matched Group Counts",
        markdown_table(matched.groupby(["model", "task", "rho_group"], as_index=False).size() if not matched.empty else pd.DataFrame(), 60),
        "",
        "## Perturbation Summary",
        markdown_table(summary, 120),
        "",
        "## Low-Rho Minus High-Rho Matched Contrasts",
        markdown_table(contrasts, 40),
        "",
        "## Files",
        "```text",
        "matched_high_confidence_groups.csv",
        "brittle_perturbation_records.csv",
        "brittle_summary.csv",
        "brittle_matched_contrasts.csv",
        "report.md",
        "```",
    ]
    (out / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_circuits_report(out: Path, summary: pd.DataFrame, contrasts: pd.DataFrame, corr: pd.DataFrame, skipped: pd.DataFrame) -> None:
    lines = [
        "# Application 4: Uncertainty Circuits",
        "",
        "This application asks whether high-accessibility layer/subspace routes are causally better control points for uncertainty than low-accessibility, random, or gradient-selected alternatives.",
        "",
        "All interventions are Fisher-output-normalized to the same `||F^{1/2}J delta z||` budget.",
        "",
        "## Skipped Models",
        markdown_table(skipped, 20),
        "",
        "## Route Summary",
        markdown_table(summary, 100),
        "",
        "## High-Rho Route Contrasts",
        markdown_table(contrasts, 40),
        "",
        "## Rho-Causal Effect Correlations",
        markdown_table(corr, 40),
        "",
        "## Files",
        "```text",
        "route_scores.csv",
        "circuit_intervention_records.csv",
        "circuit_route_summary.csv",
        "circuit_route_contrasts.csv",
        "rho_causal_effect_correlations.csv",
        "skipped_models.csv",
        "report.md",
        "```",
    ]
    (out / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def write_config(out: Path, name: str, args: argparse.Namespace) -> None:
    command = (
        "python scripts\\run_brittle_confidence_and_circuit_applications.py "
        f"--models {args.models} --top-k {args.top_k} --subspace-k {args.subspace_k} "
        f"--max-prompts-per-task {args.max_prompts_per_task} --output-eps {args.output_eps} --seed {args.seed}"
    )
    payload = {
        "application": name,
        "command": command,
        "seed": args.seed,
        "models": [part.strip() for part in args.models.split(",") if part.strip()],
        "top_k": args.top_k,
        "subspace_k": args.subspace_k,
        "max_prompts_per_task": args.max_prompts_per_task,
        "output_eps": args.output_eps,
    }
    (out / "config" / "reproduce.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "\n".join(
            [
                f"# {name.title()}",
                "",
                "Application-level test kept separate from the four main experiments.",
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    run_uncertainty_circuits(args)
    run_brittle_confidence(args)
    print(args.circuits_out)
    print((args.circuits_out / "reports" / "report.md").read_text(encoding="utf-8"))
    print(args.brittle_out)
    print((args.brittle_out / "reports" / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
