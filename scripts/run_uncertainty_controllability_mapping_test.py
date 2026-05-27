from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "applications" / "10_uncertainty_controllability_mapping"
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accessible_varentropy.metrics import softmax_np
from run_decoder_uncertainty_steering import entropy_varentropy, fisher_geometry, get_final_norm


@dataclass(frozen=True)
class PromptItem:
    item_id: int
    task: str
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument(
        "--decoder-models",
        default="distilgpt2,Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B-Instruct,microsoft/phi-2,meta-llama/Llama-3.2-1B,mistralai/Mistral-7B-v0.1",
    )
    parser.add_argument("--masked-models", default="distilbert-base-uncased,bert-base-uncased,roberta-base,prajjwal1/bert-tiny")
    parser.add_argument("--max-items", type=int, default=24)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--layers", default="auto3")
    parser.add_argument("--subspace-dims", default="1,2,4")
    parser.add_argument("--eps", default="0.01,0.025,0.05,0.1,0.2")
    parser.add_argument("--movement-threshold", type=float, default=0.01)
    parser.add_argument("--drift-cap", type=float, default=2.5e-4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=20260608)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    for rel in ["config", "outputs", "reports"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()


def resolve_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * (np.log(p) - np.log(m))) + 0.5 * np.sum(q * (np.log(q) - np.log(m))))


def topk_jaccard(p: np.ndarray, q: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), len(p)))
    a = set(np.argsort(p)[-k:])
    b = set(np.argsort(q)[-k:])
    return float(len(a & b) / len(a | b))


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v, dtype=np.float64)
    return (v / norm).astype(np.float64)


def pca_basis(matrix: np.ndarray, k: int) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    k = min(int(k), matrix.shape[1], max(1, matrix.shape[0] - 1))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        return vt[:k].T.astype(np.float64)
    except np.linalg.LinAlgError:
        q, _r = np.linalg.qr(centered.T)
        return q[:, :k].astype(np.float64)


def orthonormalize(matrix: np.ndarray, k: int) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    q, _r = np.linalg.qr(matrix)
    return q[:, : min(k, q.shape[1])].astype(np.float64)


def project_access(matrix: np.ndarray, vector: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, int]:
    if matrix.size == 0:
        return np.zeros_like(vector), 0
    try:
        left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        left = orthonormalize(matrix, min(matrix.shape))
        singular = np.ones((left.shape[1],), dtype=np.float64)
    if singular.size == 0:
        return np.zeros_like(vector), 0
    cutoff = 1e-10 * max(matrix.shape) * max(float(singular[0]), eps)
    active = singular > max(cutoff, eps)
    if not np.any(active):
        return np.zeros_like(vector), 0
    basis = left[:, active]
    return (basis @ (basis.T @ vector)).astype(np.float64), int(active.sum())


def prompt_items() -> list[PromptItem]:
    raw = [
        ("masked_factual_cloze", "The capital of France is"),
        ("masked_factual_cloze", "The chemical symbol for water is"),
        ("masked_factual_cloze", "The largest ocean on Earth is the"),
        ("masked_factual_cloze", "The author of Hamlet was"),
        ("ambiguous_prompt", "The bank was quiet, so the visitor sat near the"),
        ("ambiguous_prompt", "The crane was visible beside the"),
        ("ambiguous_prompt", "The bat was found in the"),
        ("ambiguous_prompt", "The seal was lying on the"),
        ("factual_error_prompt", "The capital of Australia is Sydney. Therefore the capital is"),
        ("factual_error_prompt", "The Sun orbits Earth. This means Earth is the center of the"),
        ("factual_error_prompt", "Water freezes at 100 degrees Celsius. At sea level it freezes at"),
        ("factual_error_prompt", "Shakespeare wrote The Odyssey. The author was"),
        ("decoder_qa_completion", "Question: What gas do plants take in for photosynthesis? Answer:"),
        ("decoder_qa_completion", "Question: If 3 notebooks cost 9 dollars, 5 notebooks cost"),
        ("decoder_qa_completion", "Question: Which planet is known as the Red Planet? Answer:"),
        ("decoder_qa_completion", "Question: A square has four equal"),
        ("decoder_completion", "To make tea, first boil the"),
        ("decoder_completion", "A compass is used to find"),
        ("decoder_completion", "After Monday comes"),
        ("decoder_completion", "A piano usually has black and white"),
        ("ambiguous_prompt", "The pitcher walked toward the"),
        ("factual_error_prompt", "Birds are mammals, so a bird is a kind of"),
        ("masked_factual_cloze", "The language mainly spoken in Brazil is"),
        ("decoder_qa_completion", "Question: What organ pumps blood through the body? Answer:"),
    ]
    return [PromptItem(i, task, prompt) for i, (task, prompt) in enumerate(raw)]


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "auto3":
        vals = [max(1, n_layers // 4), max(1, n_layers // 2), max(1, (3 * n_layers) // 4)]
        return sorted(set(min(n_layers, v) for v in vals))
    return [int(part) for part in spec.split(",") if part.strip()]


def selected_decoder_logits(hidden: torch.Tensor, norm, lm_head, token_ids: torch.Tensor) -> torch.Tensor:
    x = norm(hidden.reshape(1, 1, -1))[0, 0, :]
    weight = lm_head.weight[token_ids].to(device=x.device, dtype=x.dtype)
    logits = weight @ x
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[token_ids].to(device=x.device, dtype=x.dtype)
    return logits


def decoder_jacobian(hidden: torch.Tensor, norm, lm_head, token_ids: torch.Tensor) -> np.ndarray:
    base = hidden.detach().clone().requires_grad_(True)

    def fn(vec: torch.Tensor) -> torch.Tensor:
        return selected_decoder_logits(vec, norm, lm_head, token_ids).float()

    jac = torch.autograd.functional.jacobian(fn, base, create_graph=False, strict=False, vectorize=True)
    return np.nan_to_num(jac.detach().cpu().numpy().astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def eval_decoder(hidden: torch.Tensor, direction: np.ndarray, eps: float, norm, lm_head, token_ids: torch.Tensor) -> np.ndarray:
    step = torch.as_tensor(float(eps) * direction, dtype=hidden.dtype, device=hidden.device)
    with torch.no_grad():
        logits = selected_decoder_logits(hidden + step, norm, lm_head, token_ids).detach().float().cpu().numpy().astype(np.float64)
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    return softmax_np(logits[None, :])[0].astype(np.float64)


def route_bases_for_records(records: list[dict[str, object]], dims: list[int], seed: int) -> dict[tuple[int, str, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    by_layer: dict[int, list[dict[str, object]]] = {}
    for rec in records:
        by_layer.setdefault(int(rec["layer"]), []).append(rec)
    bases = {}
    for layer, layer_records in by_layer.items():
        hidden = np.stack([rec["hidden"].cpu().numpy().astype(np.float64) for rec in layer_records])
        delta = np.stack([(rec["next_hidden"] - rec["hidden"]).cpu().numpy().astype(np.float64) for rec in layer_records])
        dim_hidden = hidden.shape[1]
        for k in dims:
            kk = min(int(k), dim_hidden, max(1, len(layer_records) - 1))
            bases[(layer, "state_pca", kk)] = pca_basis(hidden, kk)
            bases[(layer, "delta_pca", kk)] = pca_basis(delta, kk)
            bases[(layer, "random", kk)] = orthonormalize(rng.normal(size=(dim_hidden, kk)), kk)
    return bases


def load_decoder(model_name: str, args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=resolve_dtype(args.torch_dtype),
        attn_implementation="eager",
    )
    model.to(resolve_device(args.device))
    model.eval()
    return tokenizer, model, get_final_norm(model), model.get_output_embeddings()


def build_decoder_records(model_name: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    tokenizer, model, norm, lm_head = load_decoder(model_name, args)
    device = next(model.parameters()).device
    with torch.no_grad():
        probe = model(**move_batch(tokenizer("probe", return_tensors="pt", add_special_tokens=False), device), output_hidden_states=True, use_cache=False)
    layers = parse_layers(args.layers, len(probe.hidden_states) - 1)
    rows = []
    records = []
    items = prompt_items()[: args.max_items]
    for item in items:
        encoded = move_batch(tokenizer(item.prompt, return_tensors="pt", add_special_tokens=False), device)
        if encoded["input_ids"].numel() == 0:
            continue
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True, use_cache=False)
            logits = out.logits[0, -1, :].detach().float()
            top = torch.topk(logits, k=min(args.top_m, logits.numel()))
        top_ids = top.indices.detach().cpu().numpy().astype(np.int64)
        probs = softmax_np(top.values.detach().cpu().numpy().astype(np.float64)[None, :])[0]
        entropy, varentropy = entropy_varentropy(probs)
        sorted_probs = np.sort(probs)
        rows.append(
            {
                "model": model_name,
                "model_type": "decoder",
                "item_id": item.item_id,
                "task": item.task,
                "prompt": item.prompt,
                "confidence": float(probs.max()),
                "margin": float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0,
                "entropy": entropy,
                "varentropy": varentropy,
                "top1_token": tokenizer.decode([int(top_ids[int(np.argmax(probs))])]).strip(),
            }
        )
        for layer in layers:
            hidden = out.hidden_states[layer][0, -1, :].detach()
            next_hidden = out.hidden_states[min(layer + 1, len(out.hidden_states) - 1)][0, -1, :].detach()
            records.append({"item_id": item.item_id, "task": item.task, "prompt": item.prompt, "layer": int(layer), "hidden": hidden, "next_hidden": next_hidden, "top_ids": top_ids})
    bases = route_bases_for_records(records, [int(x) for x in args.subspace_dims.split(",") if x.strip()], args.seed)
    point_rows = []
    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
    base_by_item = pd.DataFrame(rows).set_index("item_id")
    for idx, rec in enumerate(records):
        if idx % 20 == 0:
            print(f"mapping {model_name} record {idx + 1}/{len(records)}", flush=True)
        hidden = rec["hidden"].to(device)
        token_ids = torch.as_tensor(rec["top_ids"], dtype=torch.long, device=device)
        with torch.no_grad():
            logits_before = selected_decoder_logits(hidden, norm, lm_head, token_ids).detach().float().cpu().numpy().astype(np.float64)
        logits_before = np.nan_to_num(logits_before, nan=0.0, posinf=50.0, neginf=-50.0)
        probs = softmax_np(logits_before[None, :])[0]
        geom = fisher_geometry(probs)
        jac = decoder_jacobian(hidden, norm, lm_head, token_ids)
        jacobian_norm = float(np.linalg.norm(jac))
        projected_gradient_norm = float(np.linalg.norm(jac.T @ geom.u))
        top1_before = int(np.argmax(probs))
        for (layer, family, k), basis in bases.items():
            if layer != int(rec["layer"]):
                continue
            jb = jac @ basis
            whitened = np.nan_to_num(geom.sqrt_fisher @ jb, nan=0.0, posinf=0.0, neginf=0.0)
            w_acc, rank = project_access(whitened, geom.w)
            rho = float((w_acc @ w_acc) / max(float(geom.w @ geom.w), 1e-12))
            coeff = basis.T @ (jac.T @ geom.u)
            direction = normalize(basis @ coeff)
            fisher_output_energy = float(np.linalg.norm(geom.sqrt_fisher @ (jac @ direction)) ** 2)
            budgeted_directions = [("same_epsilon", direction, fisher_output_energy)]
            if fisher_output_energy > 1e-12:
                budgeted_directions.append(("same_fisher_output_energy", direction / math.sqrt(fisher_output_energy), 1.0))
            for budget_mode, budget_direction, unit_fisher_energy in budgeted_directions:
                for eps in eps_values:
                    after = eval_decoder(hidden, budget_direction, eps, norm, lm_head, token_ids)
                    h_after, var_after = entropy_varentropy(after)
                    drift = js_divergence(probs, after)
                    jacc = topk_jaccard(probs, after, args.top_k)
                    abs_dh = abs(h_after - geom.entropy)
                    point_rows.append(
                        {
                            "model": model_name,
                            "model_type": "decoder",
                            "item_id": rec["item_id"],
                            "task": rec["task"],
                            "layer": int(rec["layer"]),
                            "route_family": family,
                            "subspace_dim": int(k),
                            "budget_mode": budget_mode,
                            "eps": eps,
                            "rho": rho,
                            "rank": rank,
                            "entropy": geom.entropy,
                            "varentropy": geom.varentropy,
                            "confidence": float(probs.max()),
                            "margin": float(np.sort(probs)[-1] - np.sort(probs)[-2]),
                            "jacobian_norm": jacobian_norm,
                            "projected_gradient_norm": projected_gradient_norm,
                            "fisher_output_energy": fisher_output_energy,
                            "unit_fisher_output_energy": unit_fisher_energy,
                            "applied_fisher_energy": float(unit_fisher_energy * eps * eps),
                            "uncertainty_movement": abs_dh,
                            "varentropy_movement": abs(var_after - geom.varentropy),
                            "delta_entropy": h_after - geom.entropy,
                            "drift": drift,
                            "top1_preserved": int(int(np.argmax(after)) == top1_before),
                            "topk_jaccard": jacc,
                            "safe_movement": int(abs_dh >= args.movement_threshold and int(np.argmax(after)) == top1_before and jacc >= 0.8 and drift <= args.drift_cap),
                            "base_confidence": float(base_by_item.loc[rec["item_id"], "confidence"]),
                        }
                    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), pd.DataFrame(point_rows), {"model": model_name, "status": "completed", "layers": layers}


def try_masked_model(model_name: str, args: argparse.Namespace) -> dict[str, object]:
    try:
        _tok = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
        _model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=args.local_files_only)
        del _model
        return {"model": model_name, "status": "available_but_not_run", "reason": "masked-LM mapping path not enabled in this compact audit"}
    except Exception as exc:
        return {"model": model_name, "status": "skipped_not_local", "reason": str(exc).splitlines()[0][:300]}


def aggregate_targets(points: pd.DataFrame, movement_threshold: float) -> pd.DataFrame:
    keys = ["model", "model_type", "item_id", "task", "layer", "route_family", "subspace_dim", "budget_mode"]
    rows = []
    for key, group in points.groupby(keys, dropna=False):
        group = group.sort_values("eps")
        achieved = group[group["uncertainty_movement"] >= movement_threshold]
        row = group.iloc[0].to_dict()
        row.update(dict(zip(keys, key, strict=False)))
        row["max_uncertainty_movement"] = float(group["uncertainty_movement"].max())
        row["max_safe_movement"] = float(group.loc[group["safe_movement"] == 1, "uncertainty_movement"].max()) if np.any(group["safe_movement"] == 1) else 0.0
        row["safe_movement_target"] = int(np.any(group["safe_movement"] == 1))
        row["minimal_energy"] = float(achieved["applied_fisher_energy"].min()) if not achieved.empty else np.nan
        row["minimal_eps"] = float(achieved["eps"].min()) if not achieved.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def model_features(include_rho: bool) -> tuple[list[str], list[str]]:
    numeric = [
        "entropy",
        "confidence",
        "margin",
        "varentropy",
        "projected_gradient_norm",
        "fisher_output_energy",
        "unit_fisher_output_energy",
        "jacobian_norm",
        "layer",
        "subspace_dim",
    ]
    if include_rho:
        numeric = ["rho", *numeric]
    categorical = ["model", "model_type", "task", "route_family", "budget_mode"]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def cv_regression(df: pd.DataFrame, target: str, include_rho: bool, groups: pd.Series) -> np.ndarray:
    numeric, categorical = model_features(include_rho)
    features = numeric + categorical
    y = df[target].to_numpy(dtype=float)
    preds = np.zeros(len(df), dtype=np.float64)
    splitter = GroupKFold(n_splits=min(5, groups.nunique()))
    for train_idx, test_idx in splitter.split(df, y, groups=groups):
        pipe = Pipeline([("prep", make_preprocessor(numeric, categorical)), ("reg", Ridge(alpha=1.0))])
        pipe.fit(df.iloc[train_idx][features], y[train_idx])
        preds[test_idx] = pipe.predict(df.iloc[test_idx][features])
    return preds


def cv_classification(df: pd.DataFrame, target: str, include_rho: bool, groups: pd.Series) -> np.ndarray:
    numeric, categorical = model_features(include_rho)
    features = numeric + categorical
    y = df[target].to_numpy(dtype=int)
    preds = np.zeros(len(df), dtype=np.float64)
    splitter = GroupKFold(n_splits=min(5, groups.nunique()))
    for train_idx, test_idx in splitter.split(df, y, groups=groups):
        train_y = y[train_idx]
        if len(np.unique(train_y)) < 2:
            preds[test_idx] = float(train_y.mean())
            continue
        pipe = Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        )
        pipe.fit(df.iloc[train_idx][features], train_y)
        preds[test_idx] = pipe.predict_proba(df.iloc[test_idx][features])[:, 1]
    return np.clip(preds, 1e-6, 1.0 - 1e-6)


def evaluate_mapping(targets: pd.DataFrame, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = targets.replace([np.inf, -np.inf], np.nan).copy()
    groups = work["model"].astype(str) + "::" + work["item_id"].astype(str)
    rows = []
    predictions = {}
    for target in ["max_uncertainty_movement"]:
        local = work.dropna(subset=[target]).reset_index(drop=True)
        local_groups = local["model"].astype(str) + "::" + local["item_id"].astype(str)
        base_pred = cv_regression(local, target, False, local_groups)
        rho_pred = cv_regression(local, target, True, local_groups)
        predictions[target] = (local, base_pred, rho_pred, "regression")
        rows.append({"target": target, "model": "controls", "metric": "mae", "value": mean_absolute_error(local[target], base_pred)})
        rows.append({"target": target, "model": "controls_plus_rho", "metric": "mae", "value": mean_absolute_error(local[target], rho_pred)})
    for target in ["safe_movement_target"]:
        local = work.dropna(subset=[target]).reset_index(drop=True)
        if local[target].nunique() < 2:
            continue
        local_groups = local["model"].astype(str) + "::" + local["item_id"].astype(str)
        base_pred = cv_classification(local, target, False, local_groups)
        rho_pred = cv_classification(local, target, True, local_groups)
        predictions[target] = (local, base_pred, rho_pred, "classification")
        y = local[target].to_numpy(dtype=int)
        for name, pred in [("controls", base_pred), ("controls_plus_rho", rho_pred)]:
            rows.append({"target": target, "model": name, "metric": "auprc", "value": average_precision_score(y, pred)})
            rows.append({"target": target, "model": name, "metric": "auroc", "value": roc_auc_score(y, pred) if len(np.unique(y)) == 2 else np.nan})
            rows.append({"target": target, "model": name, "metric": "log_loss", "value": log_loss(y, pred, labels=[0, 1])})
    minimal = work.dropna(subset=["minimal_energy"]).reset_index(drop=True)
    if len(minimal) >= 10:
        local_groups = minimal["model"].astype(str) + "::" + minimal["item_id"].astype(str)
        y = np.log10(np.clip(minimal["minimal_energy"].to_numpy(dtype=float), 1e-12, None))
        minimal = minimal.assign(log_minimal_energy=y)
        base_pred = cv_regression(minimal, "log_minimal_energy", False, local_groups)
        rho_pred = cv_regression(minimal, "log_minimal_energy", True, local_groups)
        predictions["minimal_energy"] = (minimal, base_pred, rho_pred, "regression")
        rows.append({"target": "minimal_energy", "model": "controls", "metric": "mae_log10", "value": mean_absolute_error(y, base_pred)})
        rows.append({"target": "minimal_energy", "model": "controls_plus_rho", "metric": "mae_log10", "value": mean_absolute_error(y, rho_pred)})
    rng = np.random.default_rng(seed)
    boot_rows = []
    for target, (local, base_pred, rho_pred, kind) in predictions.items():
        local = local.copy()
        local["base_pred"] = base_pred
        local["rho_pred"] = rho_pred
        local["_unit"] = local["model"].astype(str) + "::" + local["item_id"].astype(str)
        grouped = {unit: group for unit, group in local.groupby("_unit", sort=False)}
        units = np.array(list(grouped), dtype=object)
        for _ in range(bootstrap):
            sample = pd.concat([grouped[u] for u in rng.choice(units, size=len(units), replace=True)], ignore_index=True)
            if kind == "classification":
                y = sample[target].to_numpy(dtype=int)
                if len(np.unique(y)) < 2:
                    continue
                boot_rows.append({"target": target, "metric": "delta_auprc", "value": average_precision_score(y, sample["rho_pred"]) - average_precision_score(y, sample["base_pred"])})
                boot_rows.append({"target": target, "metric": "delta_log_loss_improvement", "value": log_loss(y, sample["base_pred"], labels=[0, 1]) - log_loss(y, sample["rho_pred"], labels=[0, 1])})
            elif target == "minimal_energy":
                y = np.log10(np.clip(sample["minimal_energy"].to_numpy(dtype=float), 1e-12, None))
                boot_rows.append({"target": target, "metric": "delta_mae_improvement", "value": mean_absolute_error(y, sample["base_pred"]) - mean_absolute_error(y, sample["rho_pred"])})
            else:
                y = sample[target].to_numpy(dtype=float)
                boot_rows.append({"target": target, "metric": "delta_mae_improvement", "value": mean_absolute_error(y, sample["base_pred"]) - mean_absolute_error(y, sample["rho_pred"])})
    boot = pd.DataFrame(boot_rows)
    if not boot.empty:
        boot = boot.groupby(["target", "metric"])["value"].quantile([0.025, 0.5, 0.975]).unstack().reset_index().rename(columns={0.025: "ci_low", 0.5: "median", 0.975: "ci_high"})
    return pd.DataFrame(rows), boot


def write_report(out_dir: Path, metadata: dict[str, object], model_status: pd.DataFrame, summary: pd.DataFrame, metrics: pd.DataFrame, ci: pd.DataFrame) -> None:
    report = ["# Uncertainty Controllability Mapping Test\n"]
    report.append("## Objective\n")
    report.append("Map whether high-rho regions/routes correspond to stronger local uncertainty controllability after controlling for scalar uncertainty, gradient and route/model/layer/task features.\n")
    report.append("```json\n" + json.dumps(metadata, indent=2) + "\n```\n")
    for title, frame in [
        ("Model Status", model_status),
        ("Target Summary", summary),
        ("Controls vs Controls+rho", metrics),
        ("Bootstrap Delta CI", ci),
    ]:
        report.append(f"## {title}\n")
        report.append("```text\n" + (frame.to_string(index=False) if not frame.empty else "(empty)") + "\n```\n")
    verdict = "Inconclusive"
    if not ci.empty:
        safe = ci[(ci["target"] == "safe_movement_target") & (ci["metric"] == "delta_auprc")]
        movement = ci[(ci["target"] == "max_uncertainty_movement") & (ci["metric"] == "delta_mae_improvement")]
        minimal = ci[(ci["target"] == "minimal_energy") & (ci["metric"] == "delta_mae_improvement")]
        positives = []
        for name, row in [("safe_movement AUPRC", safe), ("movement MAE", movement), ("minimal_energy MAE", minimal)]:
            if not row.empty and float(row["ci_low"].iloc[0]) > 0.0:
                positives.append(name)
        verdict = "Supported on: " + ", ".join(positives) if positives else "Not supported by the preregistered controls+rho deltas"
    report.append("## Verdict\n")
    report.append(verdict + "\n")
    (out_dir / "reports" / "uncertainty_controllability_mapping_test.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "README.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    all_base = []
    all_points = []
    statuses = []
    for model_name in [m.strip() for m in args.masked_models.split(",") if m.strip()]:
        statuses.append(try_masked_model(model_name, args))
    for model_name in [m.strip() for m in args.decoder_models.split(",") if m.strip()]:
        try:
            base, points, status = build_decoder_records(model_name, args)
            all_base.append(base)
            all_points.append(points)
            statuses.append(status)
        except Exception as exc:
            statuses.append({"model": model_name, "status": "failed", "reason": str(exc).splitlines()[0][:500]})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    base_df = pd.concat(all_base, ignore_index=True) if all_base else pd.DataFrame()
    points_df = pd.concat(all_points, ignore_index=True) if all_points else pd.DataFrame()
    targets = aggregate_targets(points_df, args.movement_threshold) if not points_df.empty else pd.DataFrame()
    metrics, ci = evaluate_mapping(targets, args.bootstrap, args.seed + 1) if not targets.empty else (pd.DataFrame(), pd.DataFrame())
    summary = (
        targets.groupby(["model", "task", "route_family", "subspace_dim", "budget_mode"], dropna=False)
        .agg(n=("item_id", "size"), rho_mean=("rho", "mean"), movement_mean=("max_uncertainty_movement", "mean"), safe_rate=("safe_movement_target", "mean"), minimal_energy_median=("minimal_energy", "median"))
        .reset_index()
        if not targets.empty
        else pd.DataFrame()
    )
    metadata = {
        "status": "completed",
        "decoder_models": [m.strip() for m in args.decoder_models.split(",") if m.strip()],
        "masked_models_requested": [m.strip() for m in args.masked_models.split(",") if m.strip()],
        "max_items": args.max_items,
        "top_m": args.top_m,
        "layers": args.layers,
        "subspace_dims": args.subspace_dims,
        "eps": args.eps,
        "movement_threshold": args.movement_threshold,
        "drift_cap": args.drift_cap,
        "top_k": args.top_k,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "controls": ["entropy", "confidence", "margin", "varentropy", "projected_gradient_norm", "fisher_output_energy", "unit_fisher_output_energy", "jacobian_norm", "layer", "model", "task", "route_family", "subspace_dim", "budget_mode"],
        "rho_added": "rho",
    }
    out = args.out_dir / "outputs"
    base_df.to_csv(out / "base_records.csv", index=False)
    points_df.to_csv(out / "intervention_points.csv", index=False)
    targets.to_csv(out / "controllability_targets.csv", index=False)
    metrics.to_csv(out / "mapping_metrics.csv", index=False)
    ci.to_csv(out / "mapping_bootstrap_ci.csv", index=False)
    summary.to_csv(out / "target_summary.csv", index=False)
    model_status = pd.DataFrame(statuses)
    model_status.to_csv(out / "model_status.csv", index=False)
    (args.out_dir / "config" / "reproduce.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.out_dir, metadata, model_status, summary.head(120), metrics, ci)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
