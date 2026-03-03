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
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from evaluation.benchmarking import AttackMetrics, BenchmarkResult, DefenseMetrics
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
    grouped = (
        df.groupby("attack_type")[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
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

    grouped = (
        df.groupby("defense_type")[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
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
    ax.set_xticklabels(
        [ATTACK_LABELS[a] for a in attack_types], fontweight="bold"
    )
    ax.set_yticklabels(
        [DEFENSE_LABELS[d] for d in defense_types], fontweight="bold"
    )
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
        attack_types = [a for a in ATTACK_LABELS if a in attack_df["attack_type"].unique()]
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
        defense_types = [d for d in DEFENSE_LABELS if d in defense_df["defense_type"].unique()]
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

    ax.plot(threshold_values, tpr_values, "o-", color="#2A9D8F",
            label="TPR (Sensitivity)", linewidth=2, markersize=5)
    ax.plot(threshold_values, fpr_values, "s-", color="#E63946",
            label="FPR (False Alarm)", linewidth=2, markersize=5)
    ax.plot(threshold_values, f1_values, "^-", color="#264653",
            label="F1 Score", linewidth=2, markersize=5)

    # optimal threshold annotation (max f1)
    best_idx = int(np.argmax(f1_values))
    best_thresh = threshold_values[best_idx]
    ax.axvline(
        best_thresh, color="#F4A261", linestyle="--", linewidth=1.5,
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
                lambda: plot_attack_radar(
                    results, str(p / f"{prefix}_08_radar.png")
                ),
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
            plot_watermark_detection(
                z_watermarked, z_clean, save_path=path
            )
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

    def analyze_attack_patterns(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
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
