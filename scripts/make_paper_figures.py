from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT_FIGURES = ROOT / "reports" / "figures"


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, experiment_path: Path, basename: str) -> None:
    for out_dir in [experiment_path / "figures", REPORT_FIGURES]:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{basename}.svg", bbox_inches="tight")
        fig.savefig(out_dir / f"{basename}.png", bbox_inches="tight")
    plt.close(fig)


def mean_ci(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    summary = frame.groupby(group_cols, as_index=False, observed=True).agg(
        mean=(value_col, "mean"),
        std=(value_col, "std"),
        n=(value_col, "count"),
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    return summary


def add_box(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 1.8) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        0.72,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.0,
        edgecolor="#1f2937",
        facecolor="#f8fafc",
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + 0.36, text, ha="center", va="center")


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color="#334155",
        )
    )


def conceptual_figure() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 2.6))
    ax.axis("off")
    add_box(ax, (0.0, 0.85), r"token distribution $p$", 1.75)
    add_box(ax, (2.2, 0.85), r"centered surprisal $u=s-H(p)$", 2.25)
    add_box(ax, (5.0, 0.85), r"Fisher whitening $w=F^{1/2}u$", 2.35)
    add_box(ax, (7.9, 1.25), r"projection onto $\mathrm{Im}(F^{1/2}JB)$", 2.7)
    add_box(ax, (7.9, 0.25), r"orthogonal complement", 2.7)
    add_arrow(ax, (1.78, 1.21), (2.18, 1.21))
    add_arrow(ax, (4.48, 1.21), (4.98, 1.21))
    add_arrow(ax, (7.35, 1.21), (7.88, 1.55))
    add_arrow(ax, (7.35, 1.21), (7.88, 0.55))
    ax.text(9.25, 2.12, r"$V_{\mathrm{access}}=\|P w\|^2$", ha="center", va="center", weight="bold")
    ax.text(9.25, -0.05, r"$V_{\mathrm{inaccess}}=\|(I-P)w\|^2$", ha="center", va="center", weight="bold")
    ax.text(5.7, 0.2, r"$\mathrm{Var}=V_{\mathrm{access}}+V_{\mathrm{inaccess}}$", ha="center", va="center")
    ax.set_xlim(-0.1, 10.8)
    ax.set_ylim(-0.35, 2.35)
    save_figure(fig, ROOT / "paper", "fig01_conceptual_accessible_varentropy")


def scalar_matched_pairs() -> None:
    df = read_csv("experiments/01_matched_scalar_uncertainty/outputs/same_uncertainty_different_rho_pairs.csv")
    adj = read_csv(
        "experiments/01_matched_scalar_uncertainty/outputs/same_uncertainty_different_accessibility_pairs.csv"
    )
    top = df.sort_values("rho_abs_diff", ascending=False).head(160)
    adj_top = adj.sort_values("rho_adjusted_abs_diff", ascending=False).head(160)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    sc = axes[0].scatter(
        top["absdiff_entropy_topk"],
        top["rho_abs_diff"],
        c=top["absdiff_varentropy_topk"],
        s=24,
        cmap="viridis",
        alpha=0.82,
    )
    axes[0].set_title("Matched scalar uncertainty, different accessibility")
    axes[0].set_xlabel(r"$|\Delta H|$ within matched pairs")
    axes[0].set_ylabel(r"$|\Delta \rho|$")
    fig.colorbar(sc, ax=axes[0], label=r"$|\Delta \mathrm{Var}|$")

    axes[1].scatter(
        adj_top["left_rho_adjusted"],
        adj_top["right_rho_adjusted"],
        c=adj_top["rho_adjusted_abs_diff"],
        s=24,
        cmap="magma",
        alpha=0.82,
    )
    lo = min(adj_top["left_rho_adjusted"].min(), adj_top["right_rho_adjusted"].min())
    hi = max(adj_top["left_rho_adjusted"].max(), adj_top["right_rho_adjusted"].max())
    axes[1].plot([lo, hi], [lo, hi], color="#64748b", linewidth=1.0, linestyle="--")
    axes[1].set_title("Adjusted accessibility mismatch")
    axes[1].set_xlabel(r"left $\rho_{\mathrm{adj}}$")
    axes[1].set_ylabel(r"right $\rho_{\mathrm{adj}}$")

    save_figure(fig, ROOT / "experiments" / "01_matched_scalar_uncertainty", "fig03_scalar_matched_pairs")


def scalar_uncertainty_vs_rho() -> None:
    df = read_csv("experiments/04_uncertainty_steering/outputs/subspace_scores.csv")
    df = df[df["subspace_family"].isin(["state_pca", "delta_pca"])].copy()
    sample = df.sample(min(len(df), 2500), random_state=20260525)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    axes[0].scatter(sample["entropy_before"], sample["rho"], s=14, alpha=0.45, color="#2563eb")
    axes[0].set_title(r"Entropy does not determine $\rho$")
    axes[0].set_xlabel(r"entropy $H$")
    axes[0].set_ylabel(r"accessibility $\rho$")
    axes[1].scatter(sample["varentropy_before"], sample["rho"], s=14, alpha=0.45, color="#be123c")
    axes[1].set_title(r"Varentropy does not determine $\rho$")
    axes[1].set_xlabel(r"varentropy")
    axes[1].set_ylabel(r"accessibility $\rho$")
    save_figure(fig, ROOT / "experiments" / "01_matched_scalar_uncertainty", "fig02_scalar_uncertainty_vs_rho")


def accessibility_predicts_movement() -> None:
    df = read_csv("experiments/02_local_perturbation_prediction/outputs/intervention_by_rho_quartile.csv")
    order = ["q1_low", "q2", "q3", "q4_high"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), constrained_layout=True)
    for eps, group in df.groupby("epsilon"):
        group = group.set_index("rho_bin").loc[order].reset_index()
        axes[0].plot(order, group["delta_entropy_mean"], marker="o", label=fr"$\epsilon={eps:g}$")
        axes[1].plot(order, group["abs_delta_varentropy_mean"], marker="o", label=fr"$\epsilon={eps:g}$")
    axes[0].set_title(r"Entropy movement by $\rho$ quartile")
    axes[0].set_xlabel(r"$\rho$ quartile")
    axes[0].set_ylabel(r"mean $\Delta H$")
    axes[1].set_title(r"Varentropy movement by $\rho$ quartile")
    axes[1].set_xlabel(r"$\rho$ quartile")
    axes[1].set_ylabel(r"mean $|\Delta \mathrm{Var}|$")
    axes[1].legend(frameon=False)
    save_figure(fig, ROOT / "experiments" / "02_local_perturbation_prediction", "fig03_accessibility_predicts_movement")


def rho_vs_delta_controls() -> None:
    path = ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "topk_steering_records.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["top_k_output"].eq(32)].copy()
    df["rho_bin"] = pd.qcut(df["rho"], q=8, duplicates="drop")
    centers = df.groupby("rho_bin", observed=True)["rho"].mean().rename("rho_center")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
    labels = {
        "accessible_ls": "accessible",
        "random_control": "random",
        "grad_orthogonal_control": "grad-orthogonal",
    }
    colors = {
        "accessible_ls": "#2563eb",
        "random_control": "#64748b",
        "grad_orthogonal_control": "#be123c",
    }
    for ax, value, title, ylabel in [
        (axes[0], "abs_delta_entropy", r"$\rho$ vs entropy movement", r"mean $|\Delta H|$"),
        (axes[1], "abs_delta_varentropy", r"$\rho$ vs varentropy movement", r"mean $|\Delta \mathrm{Var}|$"),
    ]:
        for direction, group in df.groupby("direction"):
            stats_df = mean_ci(group, ["rho_bin"], value)
            stats_df = stats_df.join(centers, on="rho_bin").sort_values("rho_center")
            ax.errorbar(
                stats_df["rho_center"],
                stats_df["mean"],
                yerr=stats_df["ci95"],
                marker="o",
                linewidth=1.8,
                capsize=2,
                label=labels.get(direction, direction),
                color=colors.get(direction, None),
            )
        ax.set_title(title)
        ax.set_xlabel(r"accessibility $\rho$ bin center")
        ax.set_ylabel(ylabel)
    axes[1].legend(frameon=False)
    save_figure(fig, ROOT / "experiments" / "02_local_perturbation_prediction", "fig04_rho_vs_delta_controls")


def steering_main() -> None:
    contrasts = read_csv("experiments/04_uncertainty_steering/outputs/equal_output_energy_contrasts.csv")
    specificity = read_csv("experiments/04_uncertainty_steering/outputs/specificity_summary.csv")
    records = pd.read_csv(
        ROOT / "experiments/04_uncertainty_steering/outputs/steering_records.csv",
        usecols=["mode", "direction", "sign", "epsilon", "delta_entropy", "abs_delta_entropy", "abs_delta_varentropy"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), constrained_layout=True)

    labels = {
        "accessible_ls": "accessible",
        "random_control": "random",
        "grad_orthogonal_control": "grad-orthogonal",
    }

    eq = contrasts.groupby(["epsilon", "control", "metric"], as_index=False).agg(
        accessible=("accessible", "mean"),
        control_value=("control_value", "mean"),
    )
    for metric, ax, title, ylabel in [
        ("abs_delta_entropy_mean", axes[0, 0], r"Equal Fisher-output energy: entropy", r"mean $|\Delta H|$"),
        (
            "abs_delta_varentropy_mean",
            axes[0, 1],
            r"Equal Fisher-output energy: varentropy",
            r"mean $|\Delta \mathrm{Var}|$",
        ),
    ]:
        metric_df = eq[eq["metric"].eq(metric)]
        acc = metric_df.groupby("epsilon", as_index=False)["accessible"].mean()
        ax.plot(acc["epsilon"], acc["accessible"], marker="o", linewidth=2.2, label="accessible")
        for control, group in metric_df.groupby("control"):
            group = group.sort_values("epsilon")
            ax.plot(group["epsilon"], group["control_value"], marker="o", label=labels.get(control, control))
        ax.set_title(title)
        ax.set_xlabel(r"output energy $\epsilon$")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    dose = (
        records[(records["mode"].eq("fisher_output_equal")) & (records["direction"].eq("accessible_ls"))]
        .groupby(["epsilon", "sign"], as_index=False)["delta_entropy"]
        .mean()
    )
    for sign, group in dose.groupby("sign"):
        group = group.sort_values("epsilon")
        axes[1, 0].plot(group["epsilon"], group["delta_entropy"], marker="o", linewidth=2.0, label=sign)
    axes[1, 0].axhline(0, color="#64748b", linewidth=1.0)
    axes[1, 0].set_title("Directional dose-response")
    axes[1, 0].set_xlabel(r"output energy $\epsilon$")
    axes[1, 0].set_ylabel(r"mean $\Delta H$")
    axes[1, 0].legend(frameon=False)

    spec = specificity.groupby("epsilon", as_index=False).agg(
        top1_change=("selected_top1_changed_rate", "mean"),
        target_change=("target_correct_changed_rate", "mean"),
        top5_jaccard=("top5_jaccard_mean", "mean"),
    )
    axes[1, 1].plot(spec["epsilon"], 1.0 - spec["top1_change"], marker="o", label="top-1 preserved")
    axes[1, 1].plot(spec["epsilon"], 1.0 - spec["target_change"], marker="o", label="target correctness preserved")
    axes[1, 1].plot(spec["epsilon"], spec["top5_jaccard"], marker="o", label="top-5 Jaccard")
    axes[1, 1].set_ylim(0.82, 1.01)
    axes[1, 1].set_title("Answer preservation")
    axes[1, 1].set_xlabel(r"output energy $\epsilon$")
    axes[1, 1].set_ylabel("preservation rate")
    axes[1, 1].legend(frameon=False)

    save_figure(fig, ROOT / "experiments" / "04_uncertainty_steering", "fig06_uncertainty_steering_main")


def steering_with_ci() -> None:
    path = ROOT / "experiments" / "controls" / "topk_robustness" / "outputs" / "topk_steering_records.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["top_k_output"].eq(32)].copy()
    labels = {
        "accessible_ls": "accessible",
        "random_control": "random",
        "grad_orthogonal_control": "grad-orthogonal",
    }
    order = ["accessible_ls", "random_control", "grad_orthogonal_control"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), constrained_layout=True)
    for ax, value, title, ylabel in [
        (axes[0], "abs_delta_entropy", r"Entropy steering", r"mean $|\Delta H|$"),
        (axes[1], "abs_delta_varentropy", r"Varentropy steering", r"mean $|\Delta \mathrm{Var}|$"),
        (axes[2], "full_vocab_top10_jaccard", r"Top-10 preservation", "mean Jaccard"),
    ]:
        summary = mean_ci(df, ["direction"], value).set_index("direction").loc[order].reset_index()
        x = np.arange(len(summary))
        ax.bar(x, summary["mean"], yerr=summary["ci95"], capsize=3, color=["#2563eb", "#64748b", "#be123c"])
        ax.set_xticks(x)
        ax.set_xticklabels([labels[d] for d in summary["direction"]], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    save_figure(fig, ROOT / "experiments" / "04_uncertainty_steering", "fig06_steering_vs_controls_ci")


def topk_robustness_figure() -> None:
    root = ROOT / "experiments" / "controls" / "topk_robustness" / "outputs"
    if not (root / "topk_rank_stability.csv").exists():
        return
    rank = pd.read_csv(root / "topk_rank_stability.csv")
    steering = pd.read_csv(root / "topk_steering_summary.csv")
    acc = steering[steering["direction"].eq("accessible_ls")].groupby("top_k_output", as_index=False).agg(
        abs_delta_entropy_mean=("abs_delta_entropy_mean", "mean"),
        abs_delta_varentropy_mean=("abs_delta_varentropy_mean", "mean"),
        top10=("full_vocab_top10_jaccard_mean", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), constrained_layout=True)
    axes[0].plot(rank["top_k_output"], rank["rho_spearman_vs_ref"], marker="o", color="#2563eb")
    axes[0].set_title(r"$\rho$ ranking stability")
    axes[0].set_xlabel("output top-k")
    axes[0].set_ylabel("Spearman vs k=32")
    axes[0].set_ylim(0.65, 1.02)
    axes[1].plot(acc["top_k_output"], acc["abs_delta_entropy_mean"], marker="o", label=r"$|\Delta H|$")
    axes[1].plot(acc["top_k_output"], acc["abs_delta_varentropy_mean"], marker="o", label=r"$|\Delta Var|$")
    axes[1].set_title("Accessible steering across top-k")
    axes[1].set_xlabel("output top-k")
    axes[1].set_ylabel("mean movement")
    axes[1].legend(frameon=False)
    axes[2].plot(acc["top_k_output"], acc["top10"], marker="o", color="#059669")
    axes[2].set_title("Top-10 preservation across top-k")
    axes[2].set_xlabel("output top-k")
    axes[2].set_ylabel("mean Jaccard")
    axes[2].set_ylim(0.90, 1.0)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(rank["top_k_output"].unique()))
        ax.set_xticklabels([str(int(v)) for v in sorted(rank["top_k_output"].unique())])
    save_figure(fig, ROOT / "experiments" / "controls" / "topk_robustness", "fig07_topk_robustness")


def gradient_baseline_figure() -> None:
    path = ROOT / "experiments" / "controls" / "gradient_baselines" / "outputs" / "gradient_baseline_correlations.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[(df["subset"].eq("accessible_only")) & (df["outcome"].isin(["abs_delta_entropy", "abs_delta_varentropy"]))].copy()
    names = {
        "rho": r"$\rho$",
        "grad_entropy_projection": r"$\|\Pi_B\nabla H\|$",
        "grad_varentropy_projection": r"$\|\Pi_B\nabla Var\|$",
        "fisher_output_norm": r"$\|F^{1/2}JB\|$",
        "jacobian_fro_norm": r"$\|JB\|$",
        "trace_fim": "trace FIM",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)
    for ax, outcome, title in [
        (axes[0], "abs_delta_entropy", r"Predicting $|\Delta H|$"),
        (axes[1], "abs_delta_varentropy", r"Predicting $|\Delta Var|$"),
    ]:
        group = df[df["outcome"].eq(outcome)].copy()
        group["label"] = group["predictor"].map(names)
        group = group.sort_values("spearman", ascending=False)
        ax.barh(group["label"], group["spearman"], color="#2563eb")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Spearman correlation")
        ax.set_xlim(-0.1, 1.0)
    save_figure(fig, ROOT / "experiments" / "controls" / "gradient_baselines", "fig08_gradient_baselines")


def main() -> None:
    setup_style()
    conceptual_figure()
    scalar_uncertainty_vs_rho()
    scalar_matched_pairs()
    accessibility_predicts_movement()
    rho_vs_delta_controls()
    steering_main()
    steering_with_ci()
    topk_robustness_figure()
    gradient_baseline_figure()


if __name__ == "__main__":
    main()
