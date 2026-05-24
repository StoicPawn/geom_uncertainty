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

    save_figure(fig, ROOT / "experiments" / "01_matched_scalar_uncertainty", "fig02_scalar_matched_pairs")


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


def layerwise_heatmap() -> None:
    df = read_csv("experiments/03_layerwise_k_structure/outputs/layerwise_k_summary.csv")
    view = (
        df[df["subspace"].eq("delta_pca")]
        .groupby(["layer", "k"], as_index=False)["rho_adjusted_mean"]
        .mean()
        .pivot(index="layer", columns="k", values="rho_adjusted_mean")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    im = ax.imshow(view.to_numpy(), aspect="auto", cmap="coolwarm", origin="lower")
    ax.set_title(r"Layerwise adjusted accessibility, delta PCA")
    ax.set_xlabel(r"subspace dimension $k$")
    ax.set_ylabel("layer")
    ax.set_xticks(np.arange(len(view.columns)))
    ax.set_xticklabels([str(v) for v in view.columns])
    ax.set_yticks(np.arange(len(view.index)))
    ax.set_yticklabels([str(v) for v in view.index])
    for i, layer in enumerate(view.index):
        for j, k in enumerate(view.columns):
            value = view.loc[layer, k]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=r"mean $\rho_{\mathrm{adj}}$")
    save_figure(fig, ROOT / "experiments" / "03_layerwise_k_structure", "fig04_layerwise_heatmap")


def compressibility_curves() -> None:
    df = read_csv("experiments/03_layerwise_k_structure/outputs/layerwise_k_summary.csv")
    view = (
        df[df["subspace"].isin(["delta_pca", "state_pca"])]
        .groupby(["subspace", "k"], as_index=False)
        .agg(rho_structured_mean=("rho_structured_mean", "mean"), rho_adjusted_mean=("rho_adjusted_mean", "mean"))
        .sort_values(["subspace", "k"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)
    for subspace, group in view.groupby("subspace"):
        axes[0].plot(group["k"], group["rho_structured_mean"], marker="o", label=subspace)
        axes[1].plot(group["k"], group["rho_adjusted_mean"], marker="o", label=subspace)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(view["k"].unique()))
        ax.set_xticklabels([str(int(v)) for v in sorted(view["k"].unique())])
        ax.set_xlabel(r"subspace dimension $k$")
    axes[0].set_title("Accumulated accessibility")
    axes[0].set_ylabel(r"mean $\rho$")
    axes[1].set_title("Adjusted over random baseline")
    axes[1].set_ylabel(r"mean $\rho_{\mathrm{adj}}$")
    axes[1].legend(frameon=False)
    save_figure(fig, ROOT / "experiments" / "03_layerwise_k_structure", "fig05_compressibility_curves")


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


def main() -> None:
    setup_style()
    conceptual_figure()
    scalar_matched_pairs()
    accessibility_predicts_movement()
    layerwise_heatmap()
    compressibility_curves()
    steering_main()


if __name__ == "__main__":
    main()
