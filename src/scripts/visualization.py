"""
publication-quality visualization for memory agent security research.

this module generates all figures and tables needed for a top-tier
conference submission (neurips / acm ccs). figures are designed to
match the visual language of recent security and ml research papers.

all figure data comes from BenchmarkResult / AttackMetrics / DefenseMetrics
dataclasses as defined in evaluation.benchmarking. all comments lowercase.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from evaluation.benchmarking import BenchmarkResult
from utils.logging import logger

# ---------------------------------------------------------------------------
# global figure style  (clean, academic, publication-ready)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "patch.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.constrained_layout.use": False,
    }
)

# attack / defense colour palettes — colour-blind safe
ATTACK_COLORS = {
    "agent_poison": "#E63946",
    "minja": "#F4A261",
    "injecmem": "#2A9D8F",
    "poisonedrag": "#9B5DE5",
}

DEFENSE_COLORS = {
    "watermark": "#264653",
    "validation": "#457B9D",
    "proactive": "#A8DADC",
    "composite": "#1D3557",
}

ATTACK_LABELS = {
    "agent_poison": "AgentPoison",
    "minja": "MINJA",
    "injecmem": "InjecMEM",
    "poisonedrag": "PoisonedRAG",
}

DEFENSE_LABELS = {
    "watermark": "Watermark",
    "validation": "Validation",
    "proactive": "Proactive",
    "composite": "Composite",
}


# ---------------------------------------------------------------------------
# helper utilities
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, path: str, tight: bool = True):
    """save figure as png and pdf, then close."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(str(p), dpi=300, bbox_inches="tight")
    # also save pdf for latex
    pdf_path = p.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    logger.log_visualization_save(str(p))


def _extract_attack_df(results: List[BenchmarkResult]) -> pd.DataFrame:
    """
    flatten attack metrics from benchmark results into a dataframe.

    columns: experiment_id, attack_type, asr_r, asr_a, asr_t,
             injection_success_rate, execution_time_avg, execution_time_std
    """
    rows = []
    for result in results:
        for attack_type, m in result.attack_metrics.items():
            rows.append(
                {
                    "experiment_id": result.experiment_id,
                    "attack_type": attack_type,
                    "attack_label": ATTACK_LABELS.get(attack_type, attack_type),
                    "asr_r": float(m.asr_r),
                    "asr_a": float(m.asr_a),
                    "asr_t": float(m.asr_t),
                    "isr": float(m.injection_success_rate),
                    "exec_time": float(m.execution_time_avg),
                    "exec_time_std": float(m.execution_time_std),
                    "benign_acc": float(m.benign_accuracy),
                }
            )
    return pd.DataFrame(rows)


def _extract_defense_df(results: List[BenchmarkResult]) -> pd.DataFrame:
    """
    flatten defense metrics from benchmark results into a dataframe.

    columns: experiment_id, defense_type, tpr, fpr, precision, recall,
             f1_score, execution_time_avg
    """
    rows = []
    for result in results:
        for defense_type, m in result.defense_metrics.items():
            rows.append(
                {
                    "experiment_id": result.experiment_id,
                    "defense_type": defense_type,
                    "defense_label": DEFENSE_LABELS.get(defense_type, defense_type),
                    "tpr": float(m.tpr),
                    "fpr": float(m.fpr),
                    "precision": float(m.precision),
                    "recall": float(m.recall),
                    "f1": float(m.f1_score),
                    "exec_time": float(m.execution_time_avg),
                    "exec_time_std": float(m.execution_time_std),
                    "tp": m.true_positives,
                    "fp": m.false_positives,
                    "tn": m.true_negatives,
                    "fn": m.false_negatives,
                    "total": m.total_tests,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figure 1: attack success rates — grouped bar chart (asr-r / asr-a / asr-t)
# ---------------------------------------------------------------------------


def plot_attack_success_rates(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    grouped bar chart of attack success rates (asr-r, asr-a, asr-t) per attack.

    reproduces the primary result table from agentpoison (chen et al., 2024)
    as a figure with error bars over multiple experimental runs.

    args:
        results: list of BenchmarkResult instances
        save_path: optional path to save the figure

    returns:
        matplotlib figure
    """
    df = _extract_attack_df(results)
    if df.empty:
        logger.log_visualization_error("no attack metrics data available")
        fig, _ = plt.subplots()
        return fig

    attack_types = [a for a in ATTACK_LABELS if a in df["attack_type"].unique()]
    metrics = ["asr_r", "asr_a", "asr_t"]
    metric_labels = ["ASR-R (Retrieval)", "ASR-A (Action)", "ASR-T (Task)"]
    metric_colors = ["#E63946", "#F4A261", "#2A9D8F"]

    # aggregate mean ± std over experiments per attack
    grouped = df.groupby("attack_type")[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = ["attack_type"] + [
        f"{m}_{s}" for m in metrics for s in ["mean", "std"]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    n_attacks = len(attack_types)
    n_metrics = len(metrics)
    bar_width = 0.22
    x = np.arange(n_attacks)

    for mi, (metric, label, color) in enumerate(
        zip(metrics, metric_labels, metric_colors)
    ):
        offsets = x + (mi - n_metrics / 2 + 0.5) * bar_width
        means = []
        stds = []
        for at in attack_types:
            row = grouped[grouped["attack_type"] == at]
            means.append(float(row[f"{metric}_mean"].values[0]) if not row.empty else 0)
            stds.append(float(row[f"{metric}_std"].values[0]) if not row.empty else 0)

        ax.bar(
            offsets,
            means,
            bar_width,
            label=label,
            color=color,
            alpha=0.88,
            capsize=4,
            yerr=stds,
            error_kw={"elinewidth": 1.2, "ecolor": "black", "capthick": 1.2},
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [ATTACK_LABELS.get(a, a) for a in attack_types], fontweight="bold"
    )
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Attack Success Rates (ASR-R / ASR-A / ASR-T)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    # reference lines from paper targets
    ax.axhline(0.80, color="#E63946", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(0.75, color="#F4A261", linestyle=":", linewidth=1, alpha=0.5)

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# figure 2: defense effectiveness — tpr / fpr / f1 comparison
# ---------------------------------------------------------------------------


def plot_defense_effectiveness(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    grouped bar chart comparing defense effectiveness across all defense types.

    shows tpr, 1-fpr (specificity), and f1-score side by side per defense,
    following the convention used in intrusion detection literature.

    args:
        results: list of BenchmarkResult instances
        save_path: optional path to save the figure

    returns:
        matplotlib figure
    """
    df = _extract_defense_df(results)
    if df.empty:
        logger.log_visualization_error("no defense metrics data available")
        fig, _ = plt.subplots()
        return fig

    defense_types = [d for d in DEFENSE_LABELS if d in df["defense_type"].unique()]
    metrics = ["tpr", "fpr", "f1"]
    metric_labels = ["TPR (Sensitivity)", "FPR (False Alarm)", "F1 Score"]
    metric_colors = ["#2A9D8F", "#E63946", "#264653"]

    grouped = df.groupby("defense_type")[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = ["defense_type"] + [
        f"{m}_{s}" for m in metrics for s in ["mean", "std"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    n_defenses = len(defense_types)
    n_metrics = len(metrics)
    bar_width = 0.22
    x = np.arange(n_defenses)

    for mi, (metric, label, color) in enumerate(
        zip(metrics, metric_labels, metric_colors)
    ):
        offsets = x + (mi - n_metrics / 2 + 0.5) * bar_width
        means = []
        stds = []
        for dt in defense_types:
            row = grouped[grouped["defense_type"] == dt]
            means.append(float(row[f"{metric}_mean"].values[0]) if not row.empty else 0)
            stds.append(float(row[f"{metric}_std"].values[0]) if not row.empty else 0)

        ax.bar(
            offsets,
            means,
            bar_width,
            label=label,
            color=color,
            alpha=0.88,
            capsize=4,
            yerr=stds,
            error_kw={"elinewidth": 1.2, "ecolor": "black", "capthick": 1.2},
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DEFENSE_LABELS.get(d, d) for d in defense_types], fontweight="bold"
    )
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Defense Effectiveness Metrics (TPR / FPR / F1)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# figure 3: roc curves for each defense type
# ---------------------------------------------------------------------------


def plot_roc_curves(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    roc-style scatter plot (tpr vs fpr) for each defense type.

    for each defense type, plots mean ± 1-std ellipse in tpr-fpr space,
    following the format used in anomaly detection benchmarks.

    args:
        results: list of BenchmarkResult instances
        save_path: optional path to save the figure

    returns:
        matplotlib figure
    """
    df = _extract_defense_df(results)
    if df.empty:
        logger.log_visualization_error("no defense metrics data")
        fig, _ = plt.subplots()
        return fig

    fig, ax = plt.subplots(figsize=(6, 6))

    # random classifier diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random classifier")

    defense_types = [d for d in DEFENSE_LABELS if d in df["defense_type"].unique()]

    for dt in defense_types:
        sub = df[df["defense_type"] == dt]
        fpr_vals = sub["fpr"].values
        tpr_vals = sub["tpr"].values
        color = DEFENSE_COLORS.get(dt, "#333333")
        label = DEFENSE_LABELS.get(dt, dt)

        ax.scatter(
            fpr_vals,
            tpr_vals,
            color=color,
            s=80,
            zorder=3,
            label=label,
            edgecolors="white",
            linewidths=0.8,
        )

        # mean marker
        if len(fpr_vals) > 0:
            fpr_mean = float(np.mean(fpr_vals))
            tpr_mean = float(np.mean(tpr_vals))
            ax.scatter(
                fpr_mean,
                tpr_mean,
                color=color,
                s=180,
                marker="*",
                zorder=4,
                edgecolors="black",
                linewidths=0.8,
            )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Space — Defense Mechanisms")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_aspect("equal")

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# figure 4: attack-defense interaction heatmap
# ---------------------------------------------------------------------------


def plot_attack_defense_heatmap(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    heatmap of defense effectiveness score (tpr - fpr) for each
    attack-defense pair, following the tabular style of adversarial ml papers.

    args:
        results: list of BenchmarkResult instances
        save_path: optional path to save the figure

    returns:
        matplotlib figure
    """
    attack_types = list(ATTACK_LABELS.keys())
    defense_types = list(DEFENSE_LABELS.keys())

    # build (defense x attack) effectiveness matrix
    matrix = np.zeros((len(defense_types), len(attack_types)))
    count_matrix = np.zeros_like(matrix)

    for result in results:
        for di, dt in enumerate(defense_types):
            if dt not in result.defense_metrics:
                continue
            dm = result.defense_metrics[dt]
            effectiveness = max(0.0, min(1.0, float(dm.tpr) - float(dm.fpr)))
            for ai, at in enumerate(attack_types):
                if at in result.attack_metrics:
                    matrix[di, ai] += effectiveness
                    count_matrix[di, ai] += 1

    # average over experiments
    with np.errstate(invalid="ignore"):
        avg_matrix = np.where(count_matrix > 0, matrix / count_matrix, 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(avg_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # annotations
    for di in range(len(defense_types)):
        for ai in range(len(attack_types)):
            val = avg_matrix[di, ai]
            text_color = "white" if val < 0.35 or val > 0.75 else "black"
            ax.text(
                ai,
                di,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                color=text_color,
                fontweight="bold",
            )

    ax.set_xticks(range(len(attack_types)))
    ax.set_yticks(range(len(defense_types)))
    ax.set_xticklabels([ATTACK_LABELS[a] for a in attack_types], fontweight="bold")
    ax.set_yticklabels([DEFENSE_LABELS[d] for d in defense_types], fontweight="bold")
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Defense Mechanism")
    ax.set_title("Defense Effectiveness (TPR − FPR) by Attack-Defense Pair")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Effectiveness Score (TPR − FPR)")

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# figure 5: watermark z-score distribution (unigram detection)
# ---------------------------------------------------------------------------


def plot_watermark_detection(
    z_scores_watermarked: List[float],
    z_scores_clean: List[float],
    z_threshold: float = 4.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    histogram of z-scores for watermarked vs clean content.

    visualises the statistical separation achieved by the unigram watermark
    (zhao et al., arXiv:2306.17439, iclr 2024).

    args:
        z_scores_watermarked: z-scores from watermarked content samples
        z_scores_clean: z-scores from clean/unwatermarked content samples
        z_threshold: detection threshold (default 4.0 from paper)
        save_path: optional save path

    returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(min(z_scores_clean, default=-5), min(z_scores_watermarked, default=-5)) - 1,
        max(max(z_scores_clean, default=15), max(z_scores_watermarked, default=15)) + 1,
        40,
    )

    ax.hist(
        z_scores_clean,
        bins=bins,
        alpha=0.65,
        color="#457B9D",
        label="Clean content",
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        z_scores_watermarked,
        bins=bins,
        alpha=0.65,
        color="#E63946",
        label="Watermarked content",
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )

    # detection threshold line
    ax.axvline(
        z_threshold,
        color="#264653",
        linestyle="--",
        linewidth=1.8,
        label=f"Detection threshold (z={z_threshold})",
    )

    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Density")
    ax.set_title(
        "Unigram-Watermark Z-Score Distribution\n"
        "(Zhao et al., arXiv:2306.17439, ICLR 2024)"
    )
    ax.legend(framealpha=0.9)

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# figure 6: memory integrity score across configurations
# ---------------------------------------------------------------------------


def plot_memory_integrity(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    horizontal bar chart of memory integrity score by experiment configuration.

    integrity = defense_tpr_avg - attack_asr_r_avg, clamped to [0, 1].

    args:
        results: list of BenchmarkResult instances
        save_path: optional save path

    returns:
        matplotlib figure
    """
    data = [
        {
            "experiment_id": r.experiment_id,
            "integrity_score": float(r.memory_integrity_score),
            "duration": float(r.test_duration),
        }
        for r in results
    ]
    df = pd.DataFrame(data).sort_values("integrity_score", ascending=True)

    if df.empty:
        logger.log_visualization_error("no experiment results for integrity plot")
        fig, _ = plt.subplots()
        return fig

    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.55)))

    colors = [
        "#E63946" if s < 0.4 else "#F4A261" if s < 0.7 else "#2A9D8F"
        for s in df["integrity_score"]
    ]

    bars = ax.barh(
        df["experiment_id"],
        df["integrity_score"],
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        height=0.55,
    )

    for bar, val in zip(bars, df["integrity_score"]):
        ax.text(
            min(val + 0.01, 0.99),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Memory Integrity Score")
    ax.set_title("Memory Integrity Score by Experiment Configuration")

    legend_patches = [
        mpatches.Patch(color="#E63946", label="Low (<0.40)"),
        mpatches.Patch(color="#F4A261", label="Medium (0.40–0.70)"),
        mpatches.Patch(color="#2A9D8F", label="High (>0.70)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9)

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# figure 7: execution latency comparison
# ---------------------------------------------------------------------------


def plot_latency_comparison(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    box plot comparing execution latency for attacks and defenses.

    important for demonstrating runtime overhead of defenses does not
    exceed the 10% overhead target from the research roadmap.

    args:
        results: list of BenchmarkResult instances
        save_path: optional save path

    returns:
        matplotlib figure
    """
    attack_df = _extract_attack_df(results)
    defense_df = _extract_defense_df(results)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # attack latency
    ax = axes[0]
    if not attack_df.empty:
        attack_types = [
            a for a in ATTACK_LABELS if a in attack_df["attack_type"].unique()
        ]
        data_to_plot = [
            attack_df[attack_df["attack_type"] == at]["exec_time"].values
            for at in attack_types
        ]
        labels = [ATTACK_LABELS[a] for a in attack_types]
        colors_list = [ATTACK_COLORS.get(a, "#333") for a in attack_types]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=labels,
            patch_artist=True,
            widths=0.45,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Attack Execution Latency")
    ax.yaxis.set_major_locator(MaxNLocator(6))

    # defense latency
    ax = axes[1]
    if not defense_df.empty:
        defense_types = [
            d for d in DEFENSE_LABELS if d in defense_df["defense_type"].unique()
        ]
        data_to_plot = [
            defense_df[defense_df["defense_type"] == dt]["exec_time"].values
            for dt in defense_types
        ]
        labels = [DEFENSE_LABELS[d] for d in defense_types]
        colors_list = [DEFENSE_COLORS.get(d, "#333") for d in defense_types]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=labels,
            patch_artist=True,
            widths=0.45,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Defense Execution Latency")
    ax.yaxis.set_major_locator(MaxNLocator(6))

    fig.suptitle("Execution Latency — Attacks vs Defenses", fontweight="bold")

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# figure 8: ablation — watermark detection vs z-score threshold
# ---------------------------------------------------------------------------


def plot_watermark_ablation(
    threshold_values: List[float],
    tpr_values: List[float],
    fpr_values: List[float],
    f1_values: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    ablation study: detection performance vs z-score threshold.

    shows how tpr, fpr, and f1 vary with the detection threshold,
    enabling threshold selection analysis.

    args:
        threshold_values: list of z-score threshold values tested
        tpr_values: tpr at each threshold
        fpr_values: fpr at each threshold
        f1_values: f1 at each threshold
        save_path: optional save path

    returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        threshold_values,
        tpr_values,
        "o-",
        color="#2A9D8F",
        label="TPR (Sensitivity)",
        linewidth=2,
        markersize=5,
    )
    ax.plot(
        threshold_values,
        fpr_values,
        "s-",
        color="#E63946",
        label="FPR (False Alarm)",
        linewidth=2,
        markersize=5,
    )
    ax.plot(
        threshold_values,
        f1_values,
        "^-",
        color="#264653",
        label="F1 Score",
        linewidth=2,
        markersize=5,
    )

    # optimal threshold annotation (max f1)
    best_idx = int(np.argmax(f1_values))
    best_thresh = threshold_values[best_idx]
    ax.axvline(
        best_thresh,
        color="#F4A261",
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal threshold (z={best_thresh:.1f})",
    )

    ax.set_xlabel("Z-Score Detection Threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "Ablation: Watermark Detection Performance vs Z-Score Threshold\n"
        "(Unigram-Watermark, Zhao et al., ICLR 2024)"
    )
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# figure 9: precision-recall curves per defense
# ---------------------------------------------------------------------------


def plot_precision_recall(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    precision vs recall scatter plot (with iso-f1 contours) per defense.

    follows the standard ap/pr-curve format used in object detection papers.

    args:
        results: list of BenchmarkResult instances
        save_path: optional save path

    returns:
        matplotlib figure
    """
    df = _extract_defense_df(results)
    if df.empty:
        logger.log_visualization_error("no defense metrics for precision-recall plot")
        fig, _ = plt.subplots()
        return fig

    fig, ax = plt.subplots(figsize=(6, 6))

    # iso-f1 contours
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        r_range = np.linspace(0.01, 1.0, 200)
        p_range = f1_val * r_range / (2 * r_range - f1_val + 1e-9)
        mask = (p_range >= 0) & (p_range <= 1)
        ax.plot(
            r_range[mask],
            p_range[mask],
            color="gray",
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
        )
        if mask.any():
            mid = mask.sum() // 2
            ax.annotate(
                f"F1={f1_val}",
                xy=(r_range[mask][mid], p_range[mask][mid]),
                fontsize=8,
                color="gray",
                ha="center",
            )

    defense_types = [d for d in DEFENSE_LABELS if d in df["defense_type"].unique()]
    for dt in defense_types:
        sub = df[df["defense_type"] == dt]
        color = DEFENSE_COLORS.get(dt, "#333")
        label = DEFENSE_LABELS.get(dt, dt)

        ax.scatter(
            sub["recall"].values,
            sub["precision"].values,
            color=color,
            s=80,
            label=label,
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
        )
        # mean point
        if len(sub) > 0:
            ax.scatter(
                sub["recall"].mean(),
                sub["precision"].mean(),
                color=color,
                s=200,
                marker="*",
                zorder=4,
                edgecolors="black",
                linewidths=0.8,
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — Defense Mechanisms")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_aspect("equal")

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# figure 10: radar chart — per-attack comparative profile
# ---------------------------------------------------------------------------


def plot_attack_radar(
    results: List[BenchmarkResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    radar (spider) chart comparing attack profiles across asr-r, asr-a, asr-t,
    isr, and benign accuracy preservation.

    args:
        results: list of BenchmarkResult instances
        save_path: optional save path

    returns:
        matplotlib figure
    """
    df = _extract_attack_df(results)
    if df.empty:
        logger.log_visualization_error("no attack data for radar chart")
        fig, _ = plt.subplots()
        return fig

    metrics = ["asr_r", "asr_a", "asr_t", "isr", "benign_acc"]
    metric_labels = ["ASR-R", "ASR-A", "ASR-T", "ISR", "Benign Acc."]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    attack_types = [a for a in ATTACK_LABELS if a in df["attack_type"].unique()]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for at in attack_types:
        sub = df[df["attack_type"] == at]
        values = [float(sub[m].mean()) for m in metrics]
        values += values[:1]
        color = ATTACK_COLORS.get(at, "#333")
        label = ATTACK_LABELS.get(at, at)

        ax.plot(angles, values, "o-", color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Attack Profile Comparison", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), framealpha=0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# phase 12: attack-defense matrix visualizations
# ---------------------------------------------------------------------------


def plot_matrix_asr_heatmap(
    matrix_result: Any,
    metric: str = "asr_r_under_defense",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    two-panel heatmap from an AttackDefenseEvaluator MatrixResult.

    left panel:  asr-r under defense per (attack, defense) pair.
                 lower = better for the defender (green-ish).
    right panel: defense_effectiveness per pair.
                 higher = better for the defender (green-ish).

    follows the attack-defense matrix format in adversarial ml papers
    (wang et al. neural cleanse, ieee s&p 2019; zhang et al. asb 2024).

    args:
        matrix_result: evaluation.attack_defense_matrix.MatrixResult instance
        metric: "asr_r_under_defense" or "asr_t_under_defense"
        save_path: optional path for saving the figure

    returns:
        matplotlib figure
    """
    attack_order = ["agent_poison", "minja", "injecmem", "poisonedrag"]
    attack_labels = ["AgentPoison", "MINJA", "InjecMEM", "PoisonedRAG"]
    defense_order = [
        "watermark",
        "validation",
        "proactive",
        "composite",
        "semantic_anomaly",
        "robust_rag",
    ]
    defense_labels = [
        "Watermark",
        "Validation",
        "Proactive",
        "Composite",
        "SAD (ours)",
        "RobustRAG",
    ]

    # filter to attacks/defenses present in matrix
    atk_present = [a for a in attack_order if a in matrix_result.results]
    def_present = [
        d
        for d in defense_order
        if any(d in matrix_result.results.get(a, {}) for a in atk_present)
    ]
    atk_labels_f = [attack_labels[attack_order.index(a)] for a in atk_present]
    def_labels_f = [defense_labels[defense_order.index(d)] for d in def_present]

    asr_matrix = np.full((len(atk_present), len(def_present)), np.nan)
    eff_matrix = np.full((len(atk_present), len(def_present)), np.nan)

    for ai, atk in enumerate(atk_present):
        for di, dfn in enumerate(def_present):
            pair = matrix_result.get(atk, dfn)
            if pair is not None:
                asr_matrix[ai, di] = getattr(pair, metric)
                eff_matrix[ai, di] = pair.defense_effectiveness

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3, len(atk_present) * 1.1 + 1)))

    for ax, data, title, cmap, vmin, vmax, fmt in [
        (
            axes[0],
            asr_matrix,
            f"Post-Defense {metric.replace('_', '-').upper()} (↓ better)",
            "RdYlGn_r",
            0.0,
            1.0,
            ".2f",
        ),
        (
            axes[1],
            eff_matrix,
            "Defense Effectiveness (↑ better)",
            "RdYlGn",
            0.0,
            1.0,
            ".2f",
        ),
    ]:
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for ai in range(data.shape[0]):
            for di in range(data.shape[1]):
                val = data[ai, di]
                if not np.isnan(val):
                    tc = "white" if val < 0.25 or val > 0.80 else "black"
                    ax.text(
                        di,
                        ai,
                        f"{val:{fmt}}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color=tc,
                        fontweight="bold",
                    )
        ax.set_xticks(range(len(def_labels_f)))
        ax.set_yticks(range(len(atk_labels_f)))
        ax.set_xticklabels(def_labels_f, rotation=30, ha="right", fontweight="bold")
        ax.set_yticklabels(atk_labels_f, fontweight="bold")
        ax.set_xlabel("Defense Mechanism")
        ax.set_ylabel("Attack Type")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Attack-Defense Interaction Matrix", fontweight="bold", fontsize=14)

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


def plot_retrieval_asr_bars(
    retrieval_metrics: Dict[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    grouped bar chart of asr-r, asr-a, asr-t per attack type from
    RetrievalSimulator or MultiTrialEvaluator results.

    args:
        retrieval_metrics: dict mapping attack_type → AttackMetrics (or summary dicts)
        save_path: optional save path

    returns:
        matplotlib figure
    """
    attack_order = ["agent_poison", "minja", "injecmem", "poisonedrag"]
    attack_labels = ["AgentPoison", "MINJA", "InjecMEM", "PoisonedRAG"]
    metric_keys = ["asr_r", "asr_a", "asr_t"]
    metric_labels = ["ASR-R", "ASR-A", "ASR-T"]
    metric_colors = ["#E63946", "#F4A261", "#2A9D8F"]

    attacks = [a for a in attack_order if a in retrieval_metrics]
    if not attacks:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "no retrieval data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    x = np.arange(len(attacks))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 5))

    for mi, (mk, ml, mc) in enumerate(zip(metric_keys, metric_labels, metric_colors)):
        vals = []
        errs = []
        for at in attacks:
            m = retrieval_metrics[at]
            if hasattr(m, mk):
                vals.append(float(getattr(m, mk)))
            elif isinstance(m, dict):
                vals.append(float(m.get(mk, 0.0)))
            else:
                vals.append(0.0)
            # error bars from multi-trial summaries if available
            ci_key = f"{mk}_ci"
            if isinstance(m, dict) and ci_key in m:
                ci = m[ci_key]
                errs.append((ci.get("upper", vals[-1]) - ci.get("lower", vals[-1])) / 2)
            else:
                errs.append(0.0)
        offset = (mi - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=ml,
            color=mc,
            alpha=0.85,
            yerr=errs if any(e > 0 for e in errs) else None,
            capsize=4,
            error_kw={"linewidth": 1.5},
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [attack_labels[attack_order.index(a)] for a in attacks], fontweight="bold"
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Attack Success Rate")
    ax.set_title(
        "Retrieval-Based Attack Success Rates (ASR-R / ASR-A / ASR-T)",
        fontweight="bold",
    )
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    if save_path:
        _save_figure(fig, save_path, tight=True)

    return fig


# ---------------------------------------------------------------------------
# comprehensive report generator
# ---------------------------------------------------------------------------


class BenchmarkVisualizer:
    """
    orchestrator for all publication-quality visualizations.

    generates the complete figure set for a research paper on
    memory agent security, saving both png and pdf outputs.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        """
        initialize visualizer.

        args:
            output_dir: directory where all figures will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.log_visualization_start("BenchmarkVisualizer", str(self.output_dir))

    def generate_all(
        self,
        results: List[BenchmarkResult],
        prefix: str = "fig",
    ) -> Dict[str, str]:
        """
        generate all figures for a benchmark result set.

        args:
            results: list of BenchmarkResult instances
            prefix: filename prefix (e.g., "fig" -> "fig_01_attack_asr.png")

        returns:
            dict mapping figure name to saved file path
        """
        saved: Dict[str, str] = {}
        p = self.output_dir

        plots: List[Tuple[str, Any]] = [
            (
                "01_attack_asr",
                lambda: plot_attack_success_rates(
                    results, str(p / f"{prefix}_01_attack_asr.png")
                ),
            ),
            (
                "02_defense_effectiveness",
                lambda: plot_defense_effectiveness(
                    results, str(p / f"{prefix}_02_defense_effectiveness.png")
                ),
            ),
            (
                "03_roc_curves",
                lambda: plot_roc_curves(
                    results, str(p / f"{prefix}_03_roc_curves.png")
                ),
            ),
            (
                "04_attack_defense_heatmap",
                lambda: plot_attack_defense_heatmap(
                    results, str(p / f"{prefix}_04_heatmap.png")
                ),
            ),
            (
                "05_memory_integrity",
                lambda: plot_memory_integrity(
                    results, str(p / f"{prefix}_05_integrity.png")
                ),
            ),
            (
                "06_latency",
                lambda: plot_latency_comparison(
                    results, str(p / f"{prefix}_06_latency.png")
                ),
            ),
            (
                "07_precision_recall",
                lambda: plot_precision_recall(
                    results, str(p / f"{prefix}_07_pr_curve.png")
                ),
            ),
            (
                "08_attack_radar",
                lambda: plot_attack_radar(results, str(p / f"{prefix}_08_radar.png")),
            ),
        ]

        for name, plot_fn in plots:
            try:
                plot_fn()
                saved[name] = str(p / f"{prefix}_{name.split('_', 1)[1]}.png")
                logger.log_visualization_complete(name, saved.get(name, ""))
            except Exception as exc:
                logger.log_visualization_error(f"failed to generate {name}: {exc}")

        return saved

    def generate_watermark_figures(
        self,
        z_watermarked: List[float],
        z_clean: List[float],
        threshold_vals: List[float],
        tpr_vals: List[float],
        fpr_vals: List[float],
        f1_vals: List[float],
        prefix: str = "wm",
    ) -> Dict[str, str]:
        """
        generate watermark-specific figures (z-score distribution + ablation).

        args:
            z_watermarked: z-scores from watermarked samples
            z_clean: z-scores from clean samples
            threshold_vals: thresholds for ablation
            tpr_vals: tpr at each threshold
            fpr_vals: fpr at each threshold
            f1_vals: f1 at each threshold
            prefix: filename prefix

        returns:
            dict mapping figure name to saved path
        """
        saved: Dict[str, str] = {}
        p = self.output_dir

        try:
            path = str(p / f"{prefix}_detection_distribution.png")
            plot_watermark_detection(z_watermarked, z_clean, save_path=path)
            saved["detection_distribution"] = path
            logger.log_visualization_complete("detection_distribution", path)
        except Exception as exc:
            logger.log_visualization_error(f"detection distribution failed: {exc}")

        try:
            path = str(p / f"{prefix}_ablation_threshold.png")
            plot_watermark_ablation(
                threshold_vals, tpr_vals, fpr_vals, f1_vals, save_path=path
            )
            saved["ablation_threshold"] = path
            logger.log_visualization_complete("ablation_threshold", path)
        except Exception as exc:
            logger.log_visualization_error(f"ablation plot failed: {exc}")

        return saved

    def generate_matrix_figures(
        self,
        matrix_result: Any,
        retrieval_metrics: Optional[Dict[str, Any]] = None,
        prefix: str = "m12",
    ) -> Dict[str, str]:
        """
        generate phase 12 attack-defense matrix and retrieval asr figures.

        args:
            matrix_result: MatrixResult from AttackDefenseEvaluator
            retrieval_metrics: dict of attack_type → AttackMetrics (optional)
            prefix: filename prefix

        returns:
            dict mapping figure name to saved path
        """
        saved: Dict[str, str] = {}
        p = self.output_dir

        try:
            path = str(p / f"{prefix}_attack_defense_matrix.png")
            plot_matrix_asr_heatmap(matrix_result, save_path=path)
            saved["attack_defense_matrix"] = path
            logger.log_visualization_complete("attack_defense_matrix", path)
        except Exception as exc:
            logger.log_visualization_error(f"matrix heatmap failed: {exc}")

        if retrieval_metrics:
            try:
                path = str(p / f"{prefix}_retrieval_asr_bars.png")
                plot_retrieval_asr_bars(retrieval_metrics, save_path=path)
                saved["retrieval_asr_bars"] = path
                logger.log_visualization_complete("retrieval_asr_bars", path)
            except Exception as exc:
                logger.log_visualization_error(f"retrieval asr bars failed: {exc}")

        return saved

    def generate_phase13_figures(
        self,
        attack_summaries: Optional[Dict[str, Any]] = None,
        adaptive_results: Optional[Dict[str, Any]] = None,
        evasion_results: Optional[Dict[str, Any]] = None,
        ablation_results: Optional[Dict[str, Any]] = None,
        prefix: str = "p13",
    ) -> Dict[str, str]:
        """
        generate all phase 13 publication figures.

        generates:
            {prefix}_comprehensive_summary.png  — 4-panel overview
            {prefix}_evasion_analysis.png       — evasion strategy comparison
            {prefix}_adaptive_tradeoff_*.png    — per-attack tradeoff plots
            {prefix}_ablation_corpus.png        — corpus size ablation
            {prefix}_ablation_topk.png          — top-k ablation
            {prefix}_ablation_sad_sigma.png     — sad threshold ablation (tpr/fpr)
            {prefix}_ablation_watermark_z.png   — watermark z-threshold ablation

        args:
            attack_summaries: from ComprehensiveEvaluator._run_attack_evaluation()
            adaptive_results: from ComprehensiveEvaluator._run_adaptive()
            evasion_results: from ComprehensiveEvaluator._run_evasion()
            ablation_results: from ComprehensiveEvaluator._run_ablations()
            prefix: filename prefix for all saved files

        returns:
            dict mapping figure name to saved path
        """
        saved: Dict[str, str] = {}
        p = self.output_dir

        # comprehensive 4-panel summary
        if attack_summaries and adaptive_results:
            try:
                path = str(p / f"{prefix}_comprehensive_summary.png")
                plot_comprehensive_summary(
                    attack_summaries, adaptive_results, save_path=path
                )
                saved["comprehensive_summary"] = path
                logger.log_visualization_complete("comprehensive_summary", path)
            except Exception as exc:
                logger.log_visualization_error(f"comprehensive summary failed: {exc}")

        # evasion analysis
        if evasion_results:
            try:
                path = str(p / f"{prefix}_evasion_analysis.png")
                plot_evasion_analysis(evasion_results, save_path=path)
                saved["evasion_analysis"] = path
                logger.log_visualization_complete("evasion_analysis", path)
            except Exception as exc:
                logger.log_visualization_error(f"evasion analysis failed: {exc}")

        # adaptive tradeoff per attack
        if adaptive_results:
            for at in ["agent_poison", "minja", "injecmem", "poisonedrag"]:
                r = adaptive_results.get(at)
                if r and "error" not in r:
                    try:
                        path = str(p / f"{prefix}_adaptive_tradeoff_{at}.png")
                        atk_label = {
                            "agent_poison": "AgentPoison",
                            "minja": "MINJA",
                            "injecmem": "InjecMEM",
                        }.get(at, at)
                        plot_adaptive_tradeoff(
                            r,
                            title=f"Evasion–Retrieval Tradeoff: {atk_label} vs. SAD",
                            save_path=path,
                        )
                        saved[f"adaptive_tradeoff_{at}"] = path
                        logger.log_visualization_complete(
                            f"adaptive_tradeoff_{at}", path
                        )
                    except Exception as exc:
                        logger.log_visualization_error(
                            f"tradeoff plot {at} failed: {exc}"
                        )

        # ablation curves
        if ablation_results:
            cs_pts = ablation_results.get("corpus_size", [])
            if cs_pts:
                try:
                    path = str(p / f"{prefix}_ablation_corpus.png")
                    plot_ablation_curve(
                        cs_pts,
                        "Corpus Size $N$",
                        metric="asr_r",
                        metric_label="ASR-R",
                        title="ASR-R vs. Corpus Size",
                        save_path=path,
                    )
                    saved["ablation_corpus"] = path
                    logger.log_visualization_complete("ablation_corpus", path)
                except Exception as exc:
                    logger.log_visualization_error(
                        f"corpus ablation plot failed: {exc}"
                    )

            # top-k ablation (use agent_poison)
            tk_pts = ablation_results.get("top_k_agent_poison", [])
            if tk_pts:
                try:
                    path = str(p / f"{prefix}_ablation_topk.png")
                    plot_ablation_curve(
                        tk_pts,
                        "Retrieval Top-$k$",
                        metric="asr_r",
                        metric_label="ASR-R",
                        title="ASR-R vs. Retrieval Top-$k$",
                        color="#d7191c",
                        save_path=path,
                    )
                    saved["ablation_topk"] = path
                    logger.log_visualization_complete("ablation_topk", path)
                except Exception as exc:
                    logger.log_visualization_error(f"topk ablation plot failed: {exc}")

            # sad sigma ablation
            sad_pts = ablation_results.get("sad_sigma_agent_poison", [])
            if sad_pts:
                try:
                    path = str(p / f"{prefix}_ablation_sad_sigma.png")
                    plot_ablation_tpr_fpr(
                        sad_pts,
                        "SAD Threshold $k$",
                        title="SAD Detection Rate vs. Threshold $k$",
                        save_path=path,
                    )
                    saved["ablation_sad_sigma"] = path
                    logger.log_visualization_complete("ablation_sad_sigma", path)
                except Exception as exc:
                    logger.log_visualization_error(
                        f"sad sigma ablation plot failed: {exc}"
                    )

            # watermark z-threshold ablation
            wm_pts = ablation_results.get("watermark_z_threshold", [])
            if wm_pts:
                try:
                    path = str(p / f"{prefix}_ablation_watermark_z.png")
                    plot_ablation_tpr_fpr(
                        wm_pts,
                        "Watermark $z$-Threshold",
                        title="Watermark Detection vs. $z$-Threshold",
                        save_path=path,
                    )
                    saved["ablation_watermark_z"] = path
                    logger.log_visualization_complete("ablation_watermark_z", path)
                except Exception as exc:
                    logger.log_visualization_error(
                        f"watermark ablation plot failed: {exc}"
                    )

        return saved


# ---------------------------------------------------------------------------
# phase 13: ablation curves, adaptive tradeoff, evasion analysis
# ---------------------------------------------------------------------------


def plot_ablation_curve(
    ablation_points: List[Any],
    param_label: str,
    metric: str = "asr_r",
    metric_label: str = "ASR-R",
    title: str = "",
    color: str = "#2c7bb6",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    line plot with bootstrap 95% ci shading for a single ablation study.

    each AblationPoint provides the x-value (param_value), the mean metric,
    and the ci bounds.  the plot follows the academic style of ablation figures
    in ml security papers (ci shown as shaded band, mean as solid line).

    args:
        ablation_points: list of AblationPoint (or dicts) from AblationStudy
        param_label: x-axis label (e.g. "Corpus Size $N$")
        metric: "asr_r", "tpr", "fpr", or "benign_acc"
        metric_label: y-axis label (e.g. "ASR-R")
        title: figure title
        color: line/band color
        save_path: optional save path (.png and .pdf generated)

    returns:
        matplotlib figure
    """

    def _get(pt, key, default=0.0):
        if isinstance(pt, dict):
            return pt.get(key, default)
        return getattr(pt, key, default)

    xs = [_get(pt, "param_value") for pt in ablation_points]
    if metric == "asr_r":
        means = [_get(pt, "asr_r_mean") for pt in ablation_points]
        lowers = [_get(pt, "asr_r_ci_lower") for pt in ablation_points]
        uppers = [_get(pt, "asr_r_ci_upper") for pt in ablation_points]
    elif metric == "tpr":
        means = [_get(pt, "tpr_mean") for pt in ablation_points]
        lowers = means  # no ci for tpr in ablation
        uppers = means
    elif metric == "fpr":
        means = [_get(pt, "fpr_mean") for pt in ablation_points]
        lowers = means
        uppers = means
    else:
        means = [_get(pt, "benign_acc_mean") for pt in ablation_points]
        lowers = means
        uppers = means

    fig, ax = plt.subplots(figsize=(6, 4))

    # shaded ci band
    has_ci = any(lo != m or hi != m for lo, m, hi in zip(lowers, means, uppers))
    if has_ci:
        ax.fill_between(xs, lowers, uppers, alpha=0.25, color=color, label="95% CI")

    ax.plot(
        xs,
        means,
        marker="o",
        color=color,
        linewidth=2,
        markersize=6,
        label=metric_label,
    )
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title or f"{metric_label} vs. {param_label}", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle="--")
    if has_ci:
        ax.legend(fontsize=10)

    # mark x-axis with integer labels if all values are integers
    if all(float(x) == int(float(x)) for x in xs):
        ax.set_xticks([int(x) for x in xs])

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_ablation_tpr_fpr(
    ablation_points: List[Any],
    param_label: str,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    two-panel line plot: tpr and fpr vs. a hyperparameter.

    used for sad threshold (sigma) and watermark z-threshold ablations
    to show the detection operating curve.

    args:
        ablation_points: list of AblationPoint or dicts with tpr_mean/fpr_mean
        param_label: x-axis label (e.g. "SAD Threshold $k$")
        title: figure title
        save_path: optional save path

    returns:
        matplotlib figure
    """

    def _get(pt, key, default=0.0):
        if isinstance(pt, dict):
            return pt.get(key, default)
        return getattr(pt, key, default)

    xs = [_get(pt, "param_value") for pt in ablation_points]
    tprs = [_get(pt, "tpr_mean") for pt in ablation_points]
    fprs = [_get(pt, "fpr_mean") for pt in ablation_points]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    axes[0].plot(xs, tprs, marker="o", color="#d7191c", linewidth=2, markersize=6)
    axes[0].set_xlabel(param_label, fontsize=12)
    axes[0].set_ylabel("TPR", fontsize=12)
    axes[0].set_title("True Positive Rate", fontsize=13)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    axes[1].plot(xs, fprs, marker="s", color="#1a9641", linewidth=2, markersize=6)
    axes[1].set_xlabel(param_label, fontsize=12)
    axes[1].set_ylabel("FPR", fontsize=12)
    axes[1].set_title("False Positive Rate", fontsize=13)
    axes[1].set_ylim(-0.02, max(max(fprs) * 1.2, 0.15))
    axes[1].grid(True, alpha=0.3, linestyle="--")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_adaptive_tradeoff(
    adaptive_result: Any,
    title: str = "Evasion–Retrieval Tradeoff (Adaptive vs. SAD)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    two-panel figure showing the fundamental adversarial tradeoff.

    left panel: sad tpr for standard vs. adaptive attack at each sigma.
    right panel: evasion rate vs. asr-r degradation at each sigma.

    demonstrates: as the attacker evades sad (evasion rate ↑), retrieval
    effectiveness (asr-r) drops — the fundamental tension exploited by sad.

    args:
        adaptive_result: AdaptiveSADResult (or dict) from AdaptiveSADEvaluator
        title: figure suptitle
        save_path: optional save path

    returns:
        matplotlib figure
    """

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    sigma_sweep = _get(adaptive_result, "sigma_sweep", [])

    if not sigma_sweep:
        # create empty placeholder figure
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No sigma sweep data", ha="center", va="center", fontsize=12)
        if save_path:
            _save_figure(fig, save_path)
        return fig

    sigmas = [s.get("sigma", i) for i, s in enumerate(sigma_sweep)]
    tpr_std = [s.get("tpr_standard", 0) for s in sigma_sweep]
    tpr_adv = [s.get("tpr_adaptive", 0) for s in sigma_sweep]
    evasion_rates = [s.get("evasion_rate", 0) for s in sigma_sweep]
    ret_degrad = [s.get("retrieval_degradation", 0) for s in sigma_sweep]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # panel 1: sad tpr vs sigma (standard vs adaptive)
    axes[0].plot(
        sigmas,
        tpr_std,
        marker="o",
        color="#d7191c",
        linewidth=2,
        markersize=6,
        label="Standard Attack",
    )
    axes[0].plot(
        sigmas,
        tpr_adv,
        marker="s",
        color="#2c7bb6",
        linewidth=2,
        linestyle="--",
        markersize=6,
        label="Adaptive Attack",
    )
    axes[0].set_xlabel("SAD Threshold $k$", fontsize=12)
    axes[0].set_ylabel("SAD TPR", fontsize=12)
    axes[0].set_title("Detection Rate vs. Threshold", fontsize=13)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    # panel 2: evasion rate vs retrieval degradation (tradeoff scatter)
    axes[1].scatter(
        evasion_rates,
        ret_degrad,
        c=sigmas,
        cmap="RdYlGn_r",
        s=80,
        zorder=3,
        edgecolors="k",
        linewidths=0.5,
    )
    for i, (x, y) in enumerate(zip(evasion_rates, ret_degrad)):
        axes[1].annotate(
            f"$k$={sigmas[i]:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    axes[1].set_xlabel("Evasion Rate", fontsize=12)
    axes[1].set_ylabel("ASR-R Degradation", fontsize=12)
    axes[1].set_title("Evasion–Retrieval Tradeoff", fontsize=13)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, max(max(ret_degrad) * 1.2, 0.1) if ret_degrad else 0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_evasion_analysis(
    evasion_results: Dict[str, Any],
    title: str = "Watermark Evasion Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    three-panel figure summarising the watermark evasion results.

    left panel:   tpr before/after for each evasion strategy (grouped bar).
    center panel: evasion success rate per strategy (bar).
    right panel:  mean z-score before/after per strategy (grouped bar).

    args:
        evasion_results: dict from ComprehensiveEvaluator._run_evasion()
        title: figure suptitle
        save_path: optional save path

    returns:
        matplotlib figure
    """
    strategies = ["paraphrase", "copy_paste_dilution", "adaptive_substitution"]
    labels = ["Paraphrase", "Dilution", "Adp. Subst."]
    valid = [
        s
        for s in strategies
        if s in evasion_results and "error" not in evasion_results[s]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    if not valid:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path:
            _save_figure(fig, save_path)
        return fig

    x = np.arange(len(valid))
    lbl = [labels[strategies.index(s)] for s in valid]

    # panel 1: tpr before / after
    tpr_before = [evasion_results[s].get("tpr_before", 0) for s in valid]
    tpr_after = [evasion_results[s].get("tpr_after", 0) for s in valid]
    w = 0.35
    axes[0].bar(x - w / 2, tpr_before, w, label="Before", color="#2c7bb6", alpha=0.85)
    axes[0].bar(x + w / 2, tpr_after, w, label="After", color="#d7191c", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lbl, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel("TPR", fontsize=12)
    axes[0].set_title("Watermark Detection TPR", fontsize=12)
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    # panel 2: evasion success rate
    evasion_rates = [evasion_results[s].get("evasion_success_rate", 0) for s in valid]
    bars = axes[1].bar(
        x, evasion_rates, color="#fdae61", edgecolor="k", linewidth=0.7, alpha=0.9
    )
    for bar, val in zip(bars, evasion_rates):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(lbl, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("Evasion Success Rate", fontsize=12)
    axes[1].set_title("Evasion Success Rate", fontsize=12)
    axes[1].set_ylim(0, 1.15)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    # panel 3: z-score before / after
    z_before = [evasion_results[s].get("mean_z_before", 0) for s in valid]
    z_after = [evasion_results[s].get("mean_z_after", 0) for s in valid]
    axes[2].bar(x - w / 2, z_before, w, label="Before", color="#2c7bb6", alpha=0.85)
    axes[2].bar(x + w / 2, z_after, w, label="After", color="#d7191c", alpha=0.85)
    axes[2].axhline(
        4.0, linestyle="--", color="gray", linewidth=1.2, label="$z$ threshold"
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(lbl, rotation=15, ha="right", fontsize=9)
    axes[2].set_ylabel("Mean $z$-Score", fontsize=12)
    axes[2].set_title("Mean Z-Score Before/After", fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_comprehensive_summary(
    attack_summaries: Dict[str, Any],
    adaptive_results: Dict[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel comprehensive summary figure for the paper.

    panel 1: attack asr-r bar chart with bootstrap ci error bars.
    panel 2: adaptive adversary evasion rate vs retrieval degradation (all attacks).
    panel 3: sad tpr standard vs adaptive (grouped bar by attack).
    panel 4: asr-r reduction under adaptive evasion (bar chart).

    args:
        attack_summaries: dict from ComprehensiveEvaluator._run_attack_evaluation()
        adaptive_results: dict from ComprehensiveEvaluator._run_adaptive()
        save_path: optional save path

    returns:
        matplotlib figure
    """
    attacks = ["agent_poison", "minja", "injecmem", "poisonedrag"]
    attack_labels = ["AgentPoison", "MINJA", "InjecMEM"]
    colors = ["#d7191c", "#fdae61", "#2c7bb6"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # panel 1: attack asr-r with ci
    asr_r_means, asr_r_errs = [], []
    for at in attacks:
        s = attack_summaries.get(at, {})
        r = s.get("asr_r", {})
        if isinstance(r, dict):
            mean_v = r.get("mean", 0)
            lower_v = r.get("lower", mean_v)
            upper_v = r.get("upper", mean_v)
        else:
            mean_v = lower_v = upper_v = r
        asr_r_means.append(mean_v)
        asr_r_errs.append([mean_v - lower_v, upper_v - mean_v])

    x = np.arange(len(attacks))
    err_lo = [e[0] for e in asr_r_errs]
    err_hi = [e[1] for e in asr_r_errs]
    bars = axes[0].bar(
        x, asr_r_means, color=colors, edgecolor="k", linewidth=0.7, alpha=0.9
    )
    axes[0].errorbar(
        x,
        asr_r_means,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.5,
        zorder=4,
    )
    for bar, val in zip(bars, asr_r_means):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(attack_labels, fontsize=11)
    axes[0].set_ylabel("ASR-R (95% CI)", fontsize=12)
    axes[0].set_title("Attack Success Rate — Retrieval", fontsize=12)
    axes[0].set_ylim(0, 1.15)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    # panel 2: evasion rate scatter
    evasion_rates, ret_degrad = [], []
    for at in attacks:
        r = adaptive_results.get(at, {})
        evasion_rates.append(r.get("evasion_rate", 0) if isinstance(r, dict) else 0)
        ret_degrad.append(
            r.get("retrieval_degradation", 0) if isinstance(r, dict) else 0
        )
    axes[1].scatter(
        evasion_rates,
        ret_degrad,
        c=colors,
        s=120,
        zorder=3,
        edgecolors="k",
        linewidths=0.8,
    )
    for i, (x_v, y_v) in enumerate(zip(evasion_rates, ret_degrad)):
        axes[1].annotate(
            attack_labels[i],
            (x_v, y_v),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=10,
        )
    axes[1].set_xlabel("Evasion Rate", fontsize=12)
    axes[1].set_ylabel("ASR-R Degradation", fontsize=12)
    axes[1].set_title(
        "Evasion–Retrieval Tradeoff\n(Adaptive Attack vs. SAD)", fontsize=12
    )
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].set_xlim(-0.05, 1.05)

    # panel 3: sad tpr standard vs adaptive
    tpr_std = [
        adaptive_results.get(at, {}).get("sad_tpr_standard", 0) for at in attacks
    ]
    tpr_adv = [
        adaptive_results.get(at, {}).get("sad_tpr_adaptive", 0) for at in attacks
    ]
    x_arr = np.arange(len(attacks))
    w = 0.35
    axes[2].bar(
        x_arr - w / 2, tpr_std, w, label="Standard", color="#d7191c", alpha=0.85
    )
    axes[2].bar(
        x_arr + w / 2, tpr_adv, w, label="Adaptive", color="#2c7bb6", alpha=0.85
    )
    axes[2].set_xticks(x_arr)
    axes[2].set_xticklabels(attack_labels, fontsize=11)
    axes[2].set_ylabel("SAD TPR", fontsize=12)
    axes[2].set_title("SAD Detection Rate:\nStandard vs. Adaptive Attack", fontsize=12)
    axes[2].set_ylim(0, 1.15)
    axes[2].legend(fontsize=10)
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")

    # panel 4: asr-r reduction from evasion
    asr_r_std_adv = [
        adaptive_results.get(at, {}).get("asr_r_standard", 0) for at in attacks
    ]
    asr_r_adv = [
        adaptive_results.get(at, {}).get("asr_r_adaptive", 0) for at in attacks
    ]
    x_arr2 = np.arange(len(attacks))
    axes[3].bar(
        x_arr2 - w / 2, asr_r_std_adv, w, label="Standard", color="#d7191c", alpha=0.85
    )
    axes[3].bar(
        x_arr2 + w / 2, asr_r_adv, w, label="Adaptive", color="#2c7bb6", alpha=0.85
    )
    axes[3].set_xticks(x_arr2)
    axes[3].set_xticklabels(attack_labels, fontsize=11)
    axes[3].set_ylabel("ASR-R", fontsize=12)
    axes[3].set_title(
        "ASR-R: Standard vs. Adaptive Attack\n(Evasion Cost)", fontsize=12
    )
    axes[3].set_ylim(0, 1.15)
    axes[3].legend(fontsize=10)
    axes[3].grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Phase 13: Comprehensive Evaluation Summary", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# statistical analysis helper
# ---------------------------------------------------------------------------


class StatisticalAnalyzer:
    """
    statistical analysis utilities for benchmark results.

    computes descriptive statistics, confidence intervals, and
    summaries for use in paper tables and figure captions.
    """

    def __init__(self):
        """initialize statistical analyzer."""
        pass

    def analyze_attack_patterns(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        compute per-attack descriptive statistics.

        args:
            results: list of BenchmarkResult instances

        returns:
            nested dict with per-attack mean/std/ci for each metric
        """
        df = _extract_attack_df(results)
        analysis: Dict[str, Any] = {}

        if df.empty:
            return analysis

        for at in df["attack_type"].unique():
            sub = df[df["attack_type"] == at]
            n = len(sub)
            analysis[at] = {}
            for metric in ["asr_r", "asr_a", "asr_t", "isr", "benign_acc"]:
                vals = sub[metric].values.astype(float)
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                ci_95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
                analysis[at][metric] = {
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "ci_95": round(ci_95, 4),
                    "n": n,
                }

        return analysis

    def analyze_defense_robustness(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """
        compute per-defense descriptive statistics.

        args:
            results: list of BenchmarkResult instances

        returns:
            nested dict with per-defense mean/std/ci for each metric
        """
        df = _extract_defense_df(results)
        analysis: Dict[str, Any] = {}

        if df.empty:
            return analysis

        for dt in df["defense_type"].unique():
            sub = df[df["defense_type"] == dt]
            n = len(sub)
            analysis[dt] = {}
            for metric in ["tpr", "fpr", "precision", "recall", "f1"]:
                vals = sub[metric].values.astype(float)
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                ci_95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
                analysis[dt][metric] = {
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "ci_95": round(ci_95, 4),
                    "n": n,
                }

        return analysis

    def generate_latex_table(
        self, results: List[BenchmarkResult], output_path: str
    ) -> str:
        """
        generate a latex table of attack/defense metrics for paper inclusion.

        args:
            results: list of BenchmarkResult instances
            output_path: path to save the .tex file

        returns:
            latex string
        """
        attack_stats = self.analyze_attack_patterns(results)
        defense_stats = self.analyze_defense_robustness(results)

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Attack and Defense Benchmark Results}",
            r"\label{tab:results}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"\textbf{Type} & \textbf{Method} & \textbf{ASR-R} & "
            r"\textbf{ASR-A} & \textbf{ASR-T} \\",
            r"\midrule",
        ]

        for at, stats in attack_stats.items():
            label = ATTACK_LABELS.get(at, at)
            asr_r = stats.get("asr_r", {})
            asr_a = stats.get("asr_a", {})
            asr_t = stats.get("asr_t", {})
            lines.append(
                f"Attack & {label} & "
                f"{asr_r.get('mean', 0):.3f}$\\pm${asr_r.get('std', 0):.3f} & "
                f"{asr_a.get('mean', 0):.3f}$\\pm${asr_a.get('std', 0):.3f} & "
                f"{asr_t.get('mean', 0):.3f}$\\pm${asr_t.get('std', 0):.3f} \\\\"
            )

        lines += [r"\midrule"]

        d_header = [
            r"\textbf{Type} & \textbf{Method} & \textbf{TPR} & "
            r"\textbf{FPR} & \textbf{F1} \\",
            r"\midrule",
        ]
        lines += d_header

        for dt, stats in defense_stats.items():
            label = DEFENSE_LABELS.get(dt, dt)
            tpr = stats.get("tpr", {})
            fpr = stats.get("fpr", {})
            f1 = stats.get("f1", {})
            lines.append(
                f"Defense & {label} & "
                f"{tpr.get('mean', 0):.3f}$\\pm${tpr.get('std', 0):.3f} & "
                f"{fpr.get('mean', 0):.3f}$\\pm${fpr.get('std', 0):.3f} & "
                f"{f1.get('mean', 0):.3f}$\\pm${f1.get('std', 0):.3f} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        latex_str = "\n".join(lines)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex_str)

        return latex_str

    def generate_statistical_report(
        self, results: List[BenchmarkResult], output_path: str
    ) -> str:
        """
        generate full statistical report as json.

        args:
            results: list of BenchmarkResult instances
            output_path: path to save json report

        returns:
            path to saved report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(results),
            "attack_analysis": self.analyze_attack_patterns(results),
            "defense_analysis": self.analyze_defense_robustness(results),
            "summary": {
                "most_effective_attack": None,
                "most_robust_defense": None,
            },
        }

        atk = report["attack_analysis"]
        if atk:
            best_atk = max(
                atk.items(),
                key=lambda x: x[1].get("asr_r", {}).get("mean", 0),
            )
            report["summary"]["most_effective_attack"] = best_atk[0]

        defn = report["defense_analysis"]
        if defn:
            best_def = max(
                defn.items(),
                key=lambda x: (
                    x[1].get("tpr", {}).get("mean", 0)
                    - x[1].get("fpr", {}).get("mean", 0)
                ),
            )
            report["summary"]["most_robust_defense"] = best_def[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path


# ---------------------------------------------------------------------------
# convenience wrapper
# ---------------------------------------------------------------------------


def create_experiment_dashboard(
    results: List[BenchmarkResult],
    output_dir: str = "reports/dashboard",
) -> str:
    """
    generate the full figure set + statistical report + html dashboard.

    args:
        results: list of BenchmarkResult instances
        output_dir: root directory for all outputs

    returns:
        path to the html dashboard index file
    """
    dashboard_dir = Path(output_dir)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    visualizer = BenchmarkVisualizer(str(dashboard_dir / "figures"))
    saved_plots = visualizer.generate_all(results, prefix="fig")

    analyzer = StatisticalAnalyzer()
    stats_path = str(dashboard_dir / "statistical_analysis.json")
    analyzer.generate_statistical_report(results, stats_path)
    latex_path = str(dashboard_dir / "results_table.tex")
    analyzer.generate_latex_table(results, latex_path)

    # html index
    figure_imgs = "\n".join(
        f"""
        <div class="figure">
            <h3>{name.replace("_", " ").title()}</h3>
            <img src="figures/{Path(path).name}" alt="{name}" style="max-width:100%;">
        </div>"""
        for name, path in saved_plots.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Memory Agent Security — Research Dashboard</title>
  <style>
    body {{ font-family: 'Helvetica Neue', sans-serif; margin: 40px; color: #222; }}
    h1 {{ border-bottom: 2px solid #264653; padding-bottom: 8px; }}
    h2 {{ color: #264653; }}
    h3 {{ color: #457B9D; margin-top: 30px; }}
    .figure {{ margin: 20px 0; }}
    .meta {{ background: #f8f9fa; padding: 12px 20px; border-radius: 4px; }}
    a {{ color: #457B9D; }}
  </style>
</head>
<body>
  <h1>Memory Agent Security Research Dashboard</h1>
  <div class="meta">
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Total Experiments:</strong> {len(results)}</p>
    <p><strong>Statistical Report:</strong>
       <a href="statistical_analysis.json">statistical_analysis.json</a></p>
    <p><strong>LaTeX Table:</strong>
       <a href="results_table.tex">results_table.tex</a></p>
  </div>
  <h2>Figures</h2>
  {figure_imgs}
</body>
</html>"""

    dashboard_path = dashboard_dir / "index.html"
    with open(dashboard_path, "w") as f:
        f.write(html)

    logger.log_visualization_complete("dashboard", str(dashboard_path))
    return str(dashboard_path)


if __name__ == "__main__":
    print("memory agent security — visualization module")
    print("import and call BenchmarkVisualizer or create_experiment_dashboard")
