from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from accessible_varentropy.metrics import softmax_np
from accessible_varentropy.mlm_heads import mlm_logits_from_hidden
from run_topk_gradient_regression_controls import make_subspaces, safe_tokenizer
from run_uncertainty_steering_full_battery import build_cases, build_records, full_logits, parse_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--top-k-values", default="16,32,64,128,256")
    parser.add_argument("--subspace-ks", default="8")
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--max-prompts-per-task", type=int, default=4)
    parser.add_argument("--random-subspaces", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments" / "controls" / "full_vocab_sanity",
    )
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    for child in ["outputs", "reports", "config"]:
        (out_dir / child).mkdir(parents=True, exist_ok=True)


def entropy_u_var(probs: np.ndarray, eps: float = 1e-12) -> tuple[float, np.ndarray, float]:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    log_p = np.log(p)
    entropy = float(-np.sum(p * log_p))
    u = -log_p - entropy
    varentropy = float(np.sum(p * (u**2)))
    return entropy, u, varentropy


def fisher_times(p: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    weighted_mean = p @ matrix
    return p[:, None] * (matrix - weighted_mean[None, :])


def rho_from_fisher_kernel(probs: np.ndarray, u: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float, int]:
    p = np.clip(probs.astype(np.float64), eps, 1.0)
    p = p / p.sum()
    centered_c = c - (p @ c)[None, :]
    fc = p[:, None] * centered_c
    gram = c.T @ fc
    fu = p * u
    q = c.T @ fu
    denom = max(float(u @ fu), eps)
    eigvals = np.linalg.eigvalsh(gram)
    rank = int(np.sum(eigvals > max(float(eigvals.max(initial=0.0)) * max(gram.shape) * 1e-10, eps)))
    access = float(q.T @ np.linalg.pinv(gram, rcond=1e-10) @ q)
    rho = max(0.0, min(1.0, access / denom))
    trace = float(np.trace(gram))
    return rho, access, denom - access, rank


def topk_distribution(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    return idx.astype(np.int64), softmax_np(logits[idx][None, :])[0].astype(np.float64)


def directional_logits_jvp(model, hidden: torch.Tensor, basis: np.ndarray) -> np.ndarray:
    hidden = hidden.detach().float()

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        return mlm_logits_from_hidden(model, x.view(1, 1, -1))[0, 0].float()

    cols: list[np.ndarray] = []
    for col in range(basis.shape[1]):
        tangent = torch.as_tensor(basis[:, col], dtype=hidden.dtype, device=hidden.device)
        _value, jvp = torch.autograd.functional.jvp(logits_fn, hidden, tangent, create_graph=False, strict=False)
        cols.append(jvp.detach().cpu().numpy().astype(np.float64))
    return np.stack(cols, axis=1)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tokenizer = safe_tokenizer(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, local_files_only=True)
    model.eval()
    cases = build_cases(args.seed, args.max_prompts_per_task)
    max_top_k = max(int(part) for part in args.top_k_values.split(",") if part.strip())
    prompt_table, records = build_records(tokenizer, model, cases, max_top_k)
    if not records:
        return prompt_table, pd.DataFrame(), pd.DataFrame()

    n_layers = len(records[0]["hidden_states"]) - 1
    layers = [layer for layer in parse_layers(args.layers, n_layers) if 1 <= layer <= n_layers]
    subspace_ks = [int(part) for part in args.subspace_ks.split(",") if part.strip()]
    top_k_values = [int(part) for part in args.top_k_values.split(",") if part.strip()]
    prompt_lookup = prompt_table.set_index("prompt_id")
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for layer in layers:
        subspaces = make_subspaces(records, layer, subspace_ks, args.random_subspaces, args.seed)
        for rec_idx, record in enumerate(records):
            hidden = record["hidden_states"][layer][0, int(record["mask_index"]), :].detach()
            logits = full_logits(model, hidden)
            probs = softmax_np(logits[None, :])[0].astype(np.float64)
            entropy_full, u_full, var_full = entropy_u_var(probs)
            meta = prompt_lookup.loc[int(record["prompt_id"])]
            for family, subspace_k, rep, basis in subspaces:
                c_full = directional_logits_jvp(model, hidden, basis)
                rho_full, v_access_full, v_inaccess_full, rank_full = rho_from_fisher_kernel(probs, u_full, c_full)
                common = {
                    "model": args.model,
                    "prompt_id": int(record["prompt_id"]),
                    "task": meta["task"],
                    "topic": meta["topic"],
                    "observed_condition": meta["observed_condition"],
                    "layer": int(layer),
                    "subspace_family": family,
                    "subspace_k": int(subspace_k),
                    "rep": int(rep),
                    "rho_full_vocab": rho_full,
                    "v_access_full_vocab": v_access_full,
                    "v_inaccess_full_vocab": v_inaccess_full,
                    "rank_full_vocab": rank_full,
                    "entropy_full_vocab": entropy_full,
                    "varentropy_full_vocab": var_full,
                }
                for top_k in top_k_values:
                    token_ids, p_top = topk_distribution(logits, top_k)
                    ent_top, u_top, var_top = entropy_u_var(p_top)
                    c_top = c_full[token_ids, :]
                    rho_top, v_access_top, v_inaccess_top, rank_top = rho_from_fisher_kernel(p_top, u_top, c_top)
                    rows.append(
                        {
                            **common,
                            "top_k_output": int(top_k),
                            "rho_topk": rho_top,
                            "v_access_topk": v_access_top,
                            "v_inaccess_topk": v_inaccess_top,
                            "rank_topk": rank_top,
                            "entropy_topk": ent_top,
                            "varentropy_topk": var_top,
                            "rho_abs_diff_topk_vs_full": abs(rho_top - rho_full),
                        }
                    )
    scores = pd.DataFrame(rows)
    if scores.empty:
        return prompt_table.assign(model=args.model), scores, pd.DataFrame()
    for top_k, group in scores.groupby("top_k_output"):
        rho_s = float(spearmanr(group["rho_topk"], group["rho_full_vocab"]).statistic)
        summary_rows.append(
            {
                "top_k_output": int(top_k),
                "n": int(len(group)),
                "rho_topk_mean": float(group["rho_topk"].mean()),
                "rho_full_vocab_mean": float(group["rho_full_vocab"].mean()),
                "spearman_rho_topk_vs_full": rho_s,
                "rho_abs_diff_mean": float(group["rho_abs_diff_topk_vs_full"].mean()),
                "rho_abs_diff_median": float(group["rho_abs_diff_topk_vs_full"].median()),
                "rho_abs_diff_max": float(group["rho_abs_diff_topk_vs_full"].max()),
            }
        )
    return prompt_table.assign(model=args.model), scores, pd.DataFrame(summary_rows)


def write_report(out_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Control: Full-Vocabulary Tiny-Model Sanity Check",
        "",
        "This sanity check computes full-vocabulary Fisher-kernel accessibility on `google/bert_uncased_L-2_H-128_A-2` and compares top-k rho against full-vocabulary rho.",
        "",
        "The implementation avoids forming a full `F^{1/2}` matrix. It uses the identity:",
        "",
        "```text",
        "rho = u^T F C (C^T F C)^+ C^T F u / (u^T F u), where C = J B",
        "```",
        "",
        "## Summary",
        "",
        "```text",
        summary.to_string(index=False) if not summary.empty else "(empty)",
        "```",
        "",
        "## Files",
        "",
        "```text",
        "full_vocab_rho_scores.csv",
        "topk_vs_full_vocab_summary.csv",
        "prompt_tables.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ensure_dirs(args.out_dir)
    prompt_table, scores, summary = run(args)
    prompt_table.to_csv(args.out_dir / "outputs" / "prompt_tables.csv", index=False)
    scores.to_csv(args.out_dir / "outputs" / "full_vocab_rho_scores.csv", index=False)
    summary.to_csv(args.out_dir / "outputs" / "topk_vs_full_vocab_summary.csv", index=False)
    config = {
        "control_id": "full_vocab_sanity",
        "model": args.model,
        "top_k_values": args.top_k_values,
        "subspace_ks": args.subspace_ks,
        "layers": args.layers,
        "max_prompts_per_task": args.max_prompts_per_task,
        "random_subspaces": args.random_subspaces,
        "seed": args.seed,
        "minimal_command": "python scripts/run_tiny_full_vocab_sanity.py",
    }
    pd.Series(config).to_json(args.out_dir / "config" / "reproduce.json", indent=2)
    write_report(args.out_dir, summary)


if __name__ == "__main__":
    main()
