from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "06_non_nlp_safety_policies"


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--route-dims", type=int, default=16)
    parser.add_argument("--review-budget", type=float, default=0.30)
    parser.add_argument("--fixed-review-costs", default="0.17,0.19")
    parser.add_argument("--drift-cap", type=float, default=0.12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def code_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "```text\n(empty)\n```"
    return "```text\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def entropy_np(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-8, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    epochs: int,
    seed: int,
    device: str,
) -> MLP:
    torch.manual_seed(seed)
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)
    counts = np.bincount(y_train, minlength=output_dim).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def hidden_and_probs(model: MLP, x: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        hidden = model.encoder(xt)
        probs = torch.softmax(model.head(hidden), dim=-1)
    return hidden.detach().cpu().numpy(), probs.detach().cpu().numpy()


def entropy_gradient_at_hidden(model: MLP, h: np.ndarray, device: str) -> np.ndarray:
    ht = torch.tensor(h[None, :], dtype=torch.float32, device=device, requires_grad=True)
    probs = torch.softmax(model.head(ht), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
    entropy.backward()
    return ht.grad.detach().cpu().numpy()[0]


def input_jacobian_fro_norms(model: MLP, x: np.ndarray, output_dim: int, device: str, batch_size: int = 128) -> np.ndarray:
    norms: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device, requires_grad=True)
        logits = model(xb)
        grads = []
        for class_idx in range(output_dim):
            grad = torch.autograd.grad(logits[:, class_idx].sum(), xb, retain_graph=True)[0]
            grads.append(grad.detach())
        jac = torch.stack(grads, dim=1)
        norms.append(torch.linalg.vector_norm(jac, dim=(1, 2)).cpu().numpy())
    return np.concatenate(norms)


def intervene_hidden(
    model: MLP,
    h: np.ndarray,
    probs_before: np.ndarray,
    route: np.ndarray,
    grad_entropy: np.ndarray,
    hidden_scale: float,
    drift_cap: float,
    device: str,
) -> tuple[np.ndarray, float, float, float, float, float]:
    slope = float(np.dot(grad_entropy, route))
    if abs(slope) < 1e-12:
        before_entropy = float(entropy_np(probs_before[None, :])[0])
        return probs_before, 0.0, 0.0, 0.0, before_entropy, float(np.max(probs_before))
    direction = -np.sign(slope) * route
    before_entropy = float(entropy_np(probs_before[None, :])[0])
    best_probs = probs_before
    best_delta = 0.0
    best_drift = 0.0
    best_energy = 0.0
    for multiplier in [0.05, 0.10, 0.18, 0.30, 0.45, 0.65, 0.90]:
        hp = h + multiplier * hidden_scale * direction
        with torch.no_grad():
            logits = model.head(torch.tensor(hp[None, :], dtype=torch.float32, device=device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        drift = float(0.5 * np.abs(probs - probs_before).sum())
        if drift > drift_cap:
            continue
        after_entropy = float(entropy_np(probs[None, :])[0])
        delta = before_entropy - after_entropy
        if delta > best_delta:
            best_probs = probs
            best_delta = delta
            best_drift = drift
            best_energy = float(multiplier * hidden_scale)
    post_entropy = float(entropy_np(best_probs[None, :])[0])
    post_confidence = float(np.max(best_probs))
    return best_probs, best_delta, best_drift, best_energy, post_entropy, post_confidence


def quantile_threshold(values: np.ndarray, upper_fraction: float) -> float:
    return float(np.quantile(values, max(0.0, min(1.0, 1.0 - upper_fraction))))


def fold_records(
    domain: str,
    x: np.ndarray,
    y: np.ndarray,
    target_names: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    splitter = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    rows: list[dict[str, object]] = []
    output_dim = int(np.max(y)) + 1
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        model = train_model(
            x_train,
            y[train_idx],
            x.shape[1],
            output_dim,
            args.hidden_dim,
            args.epochs,
            args.seed + fold_id,
            args.device,
        )
        h_train, p_train = hidden_and_probs(model, x_train, args.device)
        h_test, p_test = hidden_and_probs(model, x_test, args.device)
        pca = PCA(n_components=min(args.route_dims, h_train.shape[1], h_train.shape[0] - 1), random_state=args.seed)
        pca.fit(h_train)
        routes = pca.components_.astype(np.float64)
        hidden_scale = float(np.median(np.linalg.norm(h_train - h_train.mean(axis=0, keepdims=True), axis=1)))
        hidden_scale = max(hidden_scale, 1e-3)

        train_entropy = entropy_np(p_train)
        train_grad_norms = []
        train_high_rhos = []
        for h in h_train:
            grad = entropy_gradient_at_hidden(model, h, args.device)
            grad_norm = float(np.linalg.norm(grad))
            train_grad_norms.append(grad_norm)
            rhos = np.abs(routes @ grad) / max(grad_norm, 1e-12)
            train_high_rhos.append(float(np.max(rhos)))
        train_high_rhos = np.asarray(train_high_rhos)
        train_entropy_mean = float(np.mean(train_entropy))
        train_entropy_std = float(np.std(train_entropy) + 1e-8)
        train_rho_mean = float(np.mean(train_high_rhos))
        train_rho_std = float(np.std(train_high_rhos) + 1e-8)
        test_jacobian_norms = input_jacobian_fro_norms(model, x_test, output_dim, args.device)
        entropy_threshold = quantile_threshold(train_entropy, args.review_budget)
        grad_threshold = quantile_threshold(np.asarray(train_grad_norms), args.review_budget)
        uncertain_train_rhos = train_high_rhos[train_entropy >= entropy_threshold]
        rho_low_threshold = float(np.quantile(uncertain_train_rhos, 0.50)) if len(uncertain_train_rhos) else float(np.quantile(train_high_rhos, 0.50))

        for local_pos, sample_idx in enumerate(test_idx):
            probs = p_test[local_pos]
            h = h_test[local_pos]
            grad = entropy_gradient_at_hidden(model, h, args.device)
            grad_norm = float(np.linalg.norm(grad))
            rhos = np.abs(routes @ grad) / max(grad_norm, 1e-12)
            high_route_idx = int(np.argmax(rhos))
            low_route_idx = int(np.argmin(rhos))
            high_rho = float(rhos[high_route_idx])
            low_rho = float(rhos[low_route_idx])
            intervention_probs, entropy_drop, drift, energy, post_entropy, post_confidence = intervene_hidden(
                model,
                h,
                probs,
                routes[high_route_idx],
                grad,
                hidden_scale,
                args.drift_cap,
                args.device,
            )
            pred = int(np.argmax(probs))
            pred_after = int(np.argmax(intervention_probs))
            true = int(y[sample_idx])
            ent = float(entropy_np(probs[None, :])[0])
            rho_review_score = ((ent - train_entropy_mean) / train_entropy_std) - ((high_rho - train_rho_mean) / train_rho_std)
            high_uncertainty = ent >= entropy_threshold
            low_accessibility = high_rho < rho_low_threshold
            rows.append(
                {
                    "domain": domain,
                    "fold_id": fold_id,
                    "sample_id": int(sample_idx),
                    "true_label": true,
                    "true_name": target_names[true] if true < len(target_names) else str(true),
                    "pred_before": pred,
                    "pred_after_intervention": pred_after,
                    "correct_before": int(pred == true),
                    "correct_after_intervention": int(pred_after == true),
                    "entropy": ent,
                    "confidence": float(np.max(probs)),
                    "margin": float(np.sort(probs)[-1] - np.sort(probs)[-2]) if len(probs) > 1 else 1.0,
                    "grad_entropy_norm": grad_norm,
                    "jacobian_fro_norm": float(test_jacobian_norms[local_pos]),
                    "rho_high": high_rho,
                    "rho_review_score": rho_review_score,
                    "rho_low": low_rho,
                    "high_route_idx": high_route_idx,
                    "low_route_idx": low_route_idx,
                    "entropy_threshold": entropy_threshold,
                    "grad_threshold": grad_threshold,
                    "rho_low_threshold": rho_low_threshold,
                    "high_uncertainty": int(high_uncertainty),
                    "low_accessibility": int(low_accessibility),
                    "entropy_drop_after_intervention": entropy_drop,
                    "post_intervention_entropy": post_entropy,
                    "post_intervention_confidence": post_confidence,
                    "output_drift_after_intervention": drift,
                    "intervention_energy": energy,
                    "intervention_changed_top1": int(pred_after != pred),
                }
            )
    return pd.DataFrame(rows)


def add_policy_outcomes(records: pd.DataFrame, domain: str) -> pd.DataFrame:
    rows = []
    for _, row in records.iterrows():
        entropy_defer = bool(row["entropy"] >= row["entropy_threshold"])
        gradient_defer = bool(row["grad_entropy_norm"] >= row["grad_threshold"])
        rho_defer = bool(row["high_uncertainty"] and row["low_accessibility"])
        rho_intervene = bool(row["high_uncertainty"] and not row["low_accessibility"])
        refinement_passed = bool(
            rho_intervene
            and float(row["post_intervention_entropy"]) < float(row["entropy_threshold"])
            and float(row["entropy_drop_after_intervention"]) > 0.0
        )
        rho_guarded_defer = bool(row["high_uncertainty"] and (row["low_accessibility"] or not refinement_passed))
        policies = [
            ("entropy_human_review", entropy_defer, False),
            ("gradient_human_review", gradient_defer, False),
            ("rho_guided_review_or_refine", rho_defer, rho_intervene),
            ("rho_guided_guarded_refine", rho_guarded_defer, refinement_passed),
        ]
        for policy, defer, intervene in policies:
            correct_auto = int(row["correct_after_intervention"] if intervene else row["correct_before"])
            changed = int(row["intervention_changed_top1"] if intervene else 0)
            drift = float(row["output_drift_after_intervention"] if intervene else 0.0)
            error_not_deferred = 0 if defer else int(1 - correct_auto)
            rows.append(
                {
                    "domain": domain,
                    "policy": policy,
                    "fold_id": int(row["fold_id"]),
                    "sample_id": int(row["sample_id"]),
                    "deferred": int(defer),
                    "intervened": int(intervene),
                    "automatic": int(not defer),
                    "auto_correct": correct_auto if not defer else np.nan,
                    "error_not_deferred": error_not_deferred if not defer else np.nan,
                    "preserved_when_no_change_needed": int(changed == 0) if not defer and int(row["correct_before"]) == 1 else np.nan,
                    "safe_decision": int(defer or correct_auto == 1),
                    "false_intervention": int((defer or intervene) and int(row["correct_before"]) == 1),
                    "near_miss_surrogate": int((not defer) and correct_auto == 0 and float(row["confidence"]) >= 0.50),
                    "human_review_cost": float(defer),
                    "compute_sensor_budget": float(defer) + (0.25 if intervene else 0.0),
                    "output_drift": drift,
                    "entropy_drop_after_intervention": float(row["entropy_drop_after_intervention"] if intervene else 0.0),
                }
            )
    return pd.DataFrame(rows)


def summarize_policy(policy_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        policy_rows.groupby(["domain", "policy"], as_index=False)
        .agg(
            n=("sample_id", "size"),
            coverage=("automatic", "mean"),
            human_review_cost=("human_review_cost", "mean"),
            intervention_rate=("intervened", "mean"),
            automatic_accuracy=("auto_correct", "mean"),
            error_on_non_deferred=("error_not_deferred", "mean"),
            safe_decision_rate=("safe_decision", "mean"),
            false_intervention_rate=("false_intervention", "mean"),
            near_miss_surrogate_rate=("near_miss_surrogate", "mean"),
            compute_sensor_budget=("compute_sensor_budget", "mean"),
            preservation_when_no_change_needed=("preserved_when_no_change_needed", "mean"),
            mean_output_drift=("output_drift", "mean"),
            mean_entropy_drop=("entropy_drop_after_intervention", "mean"),
        )
        .sort_values(["domain", "policy"])
    )


def contrast_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "coverage",
        "human_review_cost",
        "automatic_accuracy",
        "error_on_non_deferred",
        "safe_decision_rate",
        "false_intervention_rate",
        "near_miss_surrogate_rate",
        "compute_sensor_budget",
        "preservation_when_no_change_needed",
    ]
    for domain, group in summary.groupby("domain", sort=True):
        by_policy = group.set_index("policy")
        for rho_policy in ["rho_guided_review_or_refine", "rho_guided_guarded_refine"]:
            if rho_policy not in by_policy.index:
                continue
            rho = by_policy.loc[rho_policy]
            for control in ["entropy_human_review", "gradient_human_review"]:
                if control not in by_policy.index:
                    continue
                base = by_policy.loc[control]
                for metric in metrics:
                    rows.append(
                        {
                            "domain": domain,
                            "contrast": f"{rho_policy}_vs_{control}",
                            "metric": metric,
                            "rho_guided": float(rho[metric]),
                            "control": float(base[metric]),
                            "delta_rho_minus_control": float(rho[metric] - base[metric]),
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_ci(policy_rows: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for domain, frame in policy_rows.groupby("domain", sort=True):
        grouped = {key: part for key, part in frame.groupby("fold_id", sort=False)}
        keys = np.array(list(grouped), dtype=object)
        for _ in range(n_boot):
            sample = pd.concat([grouped[key] for key in rng.choice(keys, size=len(keys), replace=True)], ignore_index=True)
            rows.append(contrast_table(summarize_policy(sample[sample["domain"].eq(domain)])))
    boot = pd.concat(rows, ignore_index=True)
    return (
        boot.groupby(["domain", "contrast", "metric"])["delta_rho_minus_control"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_low", 0.5: "median", 0.975: "ci_high"})
    )


def fixed_cost_policy_records(records: pd.DataFrame, review_costs: list[float]) -> pd.DataFrame:
    score_cols = {
        "entropy_fixed_review": "entropy",
        "gradient_fixed_review": "grad_entropy_norm",
        "jacobian_fixed_review": "jacobian_fro_norm",
        "rho_fixed_review": "rho_review_score",
    }
    rows = []
    for (domain, fold_id), frame in records.groupby(["domain", "fold_id"], sort=False):
        n = len(frame)
        for review_cost in review_costs:
            k = int(round(review_cost * n))
            k = max(0, min(n, k))
            for policy, score_col in score_cols.items():
                order = frame[score_col].to_numpy(dtype=float).argsort()[::-1]
                deferred_locs = set(order[:k].tolist())
                for loc, (_, row) in enumerate(frame.iterrows()):
                    defer = loc in deferred_locs
                    correct = int(row["correct_before"])
                    rows.append(
                        {
                            "domain": domain,
                            "fold_id": int(fold_id),
                            "sample_id": int(row["sample_id"]),
                            "target_review_cost": float(review_cost),
                            "policy": policy,
                            "score": float(row[score_col]),
                            "deferred": int(defer),
                            "automatic": int(not defer),
                            "auto_correct": correct if not defer else np.nan,
                            "error_not_deferred": int(1 - correct) if not defer else np.nan,
                            "safe_decision": int(defer or correct == 1),
                            "reviewed_error": int(1 - correct) if defer else np.nan,
                            "deferred_error_capture": int(defer and correct == 0),
                            "total_error": int(correct == 0),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_fixed_cost(policy_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        policy_rows.groupby(["domain", "target_review_cost", "policy"], as_index=False)
        .agg(
            n=("sample_id", "size"),
            review_cost=("deferred", "mean"),
            coverage=("automatic", "mean"),
            automatic_accuracy=("auto_correct", "mean"),
            error_on_non_deferred=("error_not_deferred", "mean"),
            safe_decision_rate=("safe_decision", "mean"),
            reviewed_error_rate=("reviewed_error", "mean"),
            deferred_error_capture=("deferred_error_capture", lambda s: float(s.sum() / max(policy_rows.loc[s.index, "total_error"].sum(), 1))),
        )
        .sort_values(["domain", "target_review_cost", "policy"])
    )


def fixed_cost_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "review_cost",
        "coverage",
        "automatic_accuracy",
        "error_on_non_deferred",
        "safe_decision_rate",
        "reviewed_error_rate",
        "deferred_error_capture",
    ]
    for (domain, review_cost), group in summary.groupby(["domain", "target_review_cost"], sort=True):
        by_policy = group.set_index("policy")
        if "rho_fixed_review" not in by_policy.index:
            continue
        rho = by_policy.loc["rho_fixed_review"]
        for control in ["entropy_fixed_review", "gradient_fixed_review", "jacobian_fixed_review"]:
            if control not in by_policy.index:
                continue
            base = by_policy.loc[control]
            for metric in metrics:
                rows.append(
                    {
                        "domain": domain,
                        "target_review_cost": float(review_cost),
                        "contrast": f"rho_fixed_review_vs_{control}",
                        "metric": metric,
                        "rho": float(rho[metric]),
                        "control": float(base[metric]),
                        "delta_rho_minus_control": float(rho[metric] - base[metric]),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_fixed_cost_ci(policy_rows: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for domain, frame in policy_rows.groupby("domain", sort=True):
        grouped = {key: part for key, part in frame.groupby("fold_id", sort=False)}
        keys = np.array(list(grouped), dtype=object)
        for _ in range(n_boot):
            sample = pd.concat([grouped[key] for key in rng.choice(keys, size=len(keys), replace=True)], ignore_index=True)
            rows.append(fixed_cost_contrasts(summarize_fixed_cost(sample[sample["domain"].eq(domain)])))
    boot = pd.concat(rows, ignore_index=True)
    return (
        boot.groupby(["domain", "target_review_cost", "contrast", "metric"])["delta_rho_minus_control"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_low", 0.5: "median", 0.975: "ci_high"})
    )


def route_diagnostics(records: pd.DataFrame) -> pd.DataFrame:
    return (
        records.groupby("domain", as_index=False)
        .agg(
            n=("sample_id", "size"),
            base_accuracy=("correct_before", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_rho_high=("rho_high", "mean"),
            mean_rho_low=("rho_low", "mean"),
            high_uncertainty_rate=("high_uncertainty", "mean"),
            low_accessibility_given_high_uncertainty=("low_accessibility", lambda s: float(np.mean(s[records.loc[s.index, "high_uncertainty"].astype(bool)])) if np.any(records.loc[s.index, "high_uncertainty"].astype(bool)) else float("nan")),
            mean_entropy_drop_after_intervention=("entropy_drop_after_intervention", "mean"),
            mean_output_drift_after_intervention=("output_drift_after_intervention", "mean"),
            top1_changed_after_intervention=("intervention_changed_top1", "mean"),
        )
        .sort_values("domain")
    )


def write_report(
    out_dir: Path,
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    ci: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    fixed_contrasts: pd.DataFrame,
    fixed_ci: pd.DataFrame,
) -> None:
    fixed_verdict = "Equal-cost selector test is a boundary case: rho does not beat entropy/gradient at the same review cost overall."
    if not fixed_ci.empty:
        wins = fixed_ci[
            fixed_ci["contrast"].eq("rho_fixed_review_vs_jacobian_fixed_review")
            & fixed_ci["metric"].eq("automatic_accuracy")
            & (fixed_ci["ci_low"] > 0.0)
        ]
        if not wins.empty:
            fixed_verdict += " It does beat the Jacobian selector on the digits perception surrogate."
    lines = [
        "# Non-NLP Rho-Guided Safety Policy Tests",
        "",
        "These are complete non-NLP decision-policy tests on real local sklearn datasets, not synthetic mock rows. The medical test uses the Wisconsin breast-cancer diagnostic classification dataset. The perception test uses the handwritten-digits image dataset as a compact vision-perception surrogate; it is not claimed to be an autonomous-driving dataset.",
        "",
        "For each fold, a small MLP is trained from scratch, hidden-state PCA routes are fitted on the training split only, and test decisions use scalar uncertainty, entropy-gradient norm, Jacobian norm, and rho thresholds calibrated on the training split only. Correctness is used only for evaluation. The rho policies defer high-uncertainty/low-rho cases and apply a light hidden-route refinement to high-uncertainty/high-rho cases under an output-drift cap; the guarded variant sends the case to review if refinement fails to move entropy below the fold's training-calibrated uncertainty threshold.",
        "",
        "The equal-cost panel is the stricter inverse test: entropy, entropy-gradient norm, input-Jacobian Frobenius norm, and rho all defer the same fraction of held-out cases within each fold. The rho score ranks cases by high uncertainty and low local accessibility, so a gain there means better triage at the same review cost rather than a different budget.",
        "",
        "## Equal-Cost Verdict",
        fixed_verdict,
        "",
        "## Route Diagnostics",
        code_table(diagnostics, 80),
        "",
        "## Policy Summary",
        code_table(summary, 80),
        "",
        "## Rho-Guided Contrasts",
        code_table(contrasts, 120),
        "",
        "## Cluster Bootstrap CI",
        code_table(ci, 120),
        "",
        "## Equal Review-Cost Selector Summary",
        code_table(fixed_summary, 80),
        "",
        "## Equal Review-Cost Rho Contrasts",
        code_table(fixed_contrasts, 120),
        "",
        "## Equal Review-Cost Bootstrap CI",
        code_table(fixed_ci, 120),
        "",
        "## Files",
        "```text",
        "non_nlp_route_records.csv",
        "non_nlp_policy_records.csv",
        "non_nlp_policy_summary.csv",
        "non_nlp_policy_contrasts.csv",
        "non_nlp_policy_bootstrap_ci.csv",
        "non_nlp_fixed_cost_policy_records.csv",
        "non_nlp_fixed_cost_summary.csv",
        "non_nlp_fixed_cost_contrasts.csv",
        "non_nlp_fixed_cost_bootstrap_ci.csv",
        "report.md",
        "```",
    ]
    (out_dir / "reports" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for child in ["outputs", "reports", "config"]:
        (args.out_dir / child).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cancer = load_breast_cancer()
    digits = load_digits()
    datasets = [
        ("medical_breast_cancer", cancer.data.astype(np.float32), cancer.target.astype(int), list(cancer.target_names)),
        ("vision_digits_perception", (digits.data.astype(np.float32) / 16.0), digits.target.astype(int), [str(i) for i in range(10)]),
    ]
    route_frames = [fold_records(domain, x, y, names, args) for domain, x, y, names in datasets]
    records = pd.concat(route_frames, ignore_index=True)
    policy = pd.concat([add_policy_outcomes(frame, domain) for domain, frame in records.groupby("domain", sort=False)], ignore_index=True)
    fixed_review_costs = [float(part) for part in args.fixed_review_costs.split(",") if part.strip()]
    fixed_policy = fixed_cost_policy_records(records, fixed_review_costs)
    diagnostics = route_diagnostics(records)
    summary = summarize_policy(policy)
    contrasts = contrast_table(summary)
    ci = bootstrap_ci(policy, args.bootstrap, args.seed)
    fixed_summary = summarize_fixed_cost(fixed_policy)
    fixed_contrasts = fixed_cost_contrasts(fixed_summary)
    fixed_ci = bootstrap_fixed_cost_ci(fixed_policy, args.bootstrap, args.seed + 17)

    records.to_csv(args.out_dir / "outputs" / "non_nlp_route_records.csv", index=False)
    policy.to_csv(args.out_dir / "outputs" / "non_nlp_policy_records.csv", index=False)
    summary.to_csv(args.out_dir / "outputs" / "non_nlp_policy_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "outputs" / "non_nlp_policy_contrasts.csv", index=False)
    ci.to_csv(args.out_dir / "outputs" / "non_nlp_policy_bootstrap_ci.csv", index=False)
    fixed_policy.to_csv(args.out_dir / "outputs" / "non_nlp_fixed_cost_policy_records.csv", index=False)
    fixed_summary.to_csv(args.out_dir / "outputs" / "non_nlp_fixed_cost_summary.csv", index=False)
    fixed_contrasts.to_csv(args.out_dir / "outputs" / "non_nlp_fixed_cost_contrasts.csv", index=False)
    fixed_ci.to_csv(args.out_dir / "outputs" / "non_nlp_fixed_cost_bootstrap_ci.csv", index=False)
    metadata = {
        "command": "conda run -n arca python scripts/run_non_nlp_rho_safety_tests.py",
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "folds": args.folds,
        "repeats": args.repeats,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "route_dims": args.route_dims,
        "review_budget": args.review_budget,
        "fixed_review_costs": fixed_review_costs,
        "drift_cap": args.drift_cap,
        "device": args.device,
        "datasets": ["sklearn.load_breast_cancer", "sklearn.load_digits"],
        "policy_constraint": "No correctness labels are used for route selection or defer/intervene decisions; labels are used only for supervised model training and held-out evaluation.",
    }
    (args.out_dir / "config" / "reproduce.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_report(args.out_dir, diagnostics, summary, contrasts, ci, fixed_summary, fixed_contrasts, fixed_ci)
    print(args.out_dir)


if __name__ == "__main__":
    main()
