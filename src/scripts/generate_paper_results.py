"""
focused script for generating all paper results, tables, and figures.

this script runs the full evaluation suite with the exact parameters
stated in the paper (corpus=200, n_seeds=5, n_poison_base=5, top_k=5)
and saves all output to:
  - results/tables/  (latex tables 1-6)
  - docs/paper/figures/  (publication-quality png + pdf)

usage:
    cd src && python3 scripts/generate_paper_results.py
    cd src && python3 scripts/generate_paper_results.py --quick  # faster, reduced seeds
    cd src && python3 scripts/generate_paper_results.py --tables-only
    cd src && python3 scripts/generate_paper_results.py --figures-only

design:
  - loads the sentence-transformers model once per process (singleton)
  - runs all evaluation components with paper-exact parameters
  - generates all latex tables and all figures in a single pass
  - no html dashboard (paper-focused output only)

all comments are lowercase.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

# add src/ to python path
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

from evaluation.comprehensive_eval import ComprehensiveEvaluator
from scripts.visualization import BenchmarkVisualizer
from watermark.watermarking import create_watermark_encoder

# use stdlib logging directly — researchlogger has no .info() method
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("paper_results")


# ---------------------------------------------------------------------------
# output directories
# ---------------------------------------------------------------------------

_TABLES_DIR = _ROOT / "results" / "tables"
_FIGURES_DIR = _ROOT / "docs" / "paper" / "figures"


def _ensure_dirs() -> None:
    """create output directories if needed."""
    _TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    log.info("output dirs ready: %s | %s", _TABLES_DIR, _FIGURES_DIR)


# ---------------------------------------------------------------------------
# helper: save figure as both png and pdf
# ---------------------------------------------------------------------------


def _save_fig(fig: Any, stem: str) -> None:
    """save matplotlib figure as 300-dpi png and pdf."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend

    png_path = _FIGURES_DIR / f"{stem}.png"
    pdf_path = _FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    log.info("saved figure: %s (.png + .pdf)", stem)


# ---------------------------------------------------------------------------
# core evaluation runner
# ---------------------------------------------------------------------------


def run_comprehensive_evaluation(
    corpus_size: int,
    n_seeds: int,
    n_poison: int,
    top_k: int,
    seed_base: int,
    run_ablations: bool,
) -> dict[str, Any]:
    """run the full comprehensive evaluation and return results dict."""
    log.info(
        "starting comprehensive evaluation: corpus=%d n_seeds=%d n_poison=%d top_k=%d",
        corpus_size,
        n_seeds,
        n_poison,
        top_k,
    )
    t0 = time.time()

    evaluator = ComprehensiveEvaluator(
        corpus_size=corpus_size,
        n_poison=n_poison,
        top_k=top_k,
        n_seeds=n_seeds,
        seed_base=seed_base,
        run_matrix=True,
        run_evasion=True,
        run_adaptive=True,
        run_ablations=run_ablations,
    )
    result = evaluator.run()

    elapsed = time.time() - t0
    log.info("comprehensive evaluation complete in %.1fs", elapsed)
    return result, evaluator


# ---------------------------------------------------------------------------
# table generation
# ---------------------------------------------------------------------------


def generate_tables(result: Any, evaluator: Any) -> dict[str, str]:
    """generate all latex tables and save to results/tables/."""
    log.info("generating latex tables → %s", _TABLES_DIR)
    saved = evaluator.generate_paper_tables(result, str(_TABLES_DIR))
    for table_name, path in saved.items():
        log.info("  table %s → %s", table_name, Path(path).name)
    return saved


# ---------------------------------------------------------------------------
# figure generation
# ---------------------------------------------------------------------------


def _plot_attack_asr_bars(attack_summaries: dict[str, Any], save_stem: str) -> str:
    """
    generate publication-quality grouped bar chart of attack success rates.

    shows asr-r (with bootstrap ci), modelled asr-a, and asr-t for all
    four attacks.  matches the paper's table 1 in visual form.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    attack_labels = {
        "agent_poison": "AgentPoison",
        "minja": "MINJA",
        "injecmem": "InjecMEM",
        "poisonedrag": "PoisonedRAG",
    }

    attacks = [
        k
        for k in ["agent_poison", "minja", "injecmem", "poisonedrag"]
        if k in attack_summaries
    ]
    labels = [attack_labels.get(a, a) for a in attacks]

    asr_r_means = [
        attack_summaries[a].get("asr_r", {}).get("mean", 0.0) for a in attacks
    ]
    asr_r_lows = [
        attack_summaries[a].get("asr_r", {}).get("lower", 0.0) for a in attacks
    ]
    asr_r_highs = [
        attack_summaries[a].get("asr_r", {}).get("upper", 0.0) for a in attacks
    ]
    asr_a_means = [
        attack_summaries[a].get("asr_a", {}).get("mean", 0.0) for a in attacks
    ]
    asr_t_means = [
        attack_summaries[a].get("asr_t", {}).get("mean", 0.0) for a in attacks
    ]
    benign_means = [
        attack_summaries[a].get("benign_accuracy", {}).get("mean", 0.0) for a in attacks
    ]

    x = np.arange(len(attacks))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = ["#2166ac", "#4dac26", "#d01c8b", "#f1a340"]

    # asr-r with ci error bars
    yerr_low = [m - lo for m, lo in zip(asr_r_means, asr_r_lows)]
    yerr_high = [hi - m for m, hi in zip(asr_r_means, asr_r_highs)]
    ax.bar(
        x + offsets[0],
        asr_r_means,
        width,
        label="ASR-R (empirical)",
        color=colors[0],
        alpha=0.85,
        yerr=[yerr_low, yerr_high],
        capsize=4,
        error_kw={"linewidth": 1.2},
    )
    # asr-a
    ax.bar(
        x + offsets[1],
        asr_a_means,
        width,
        label="ASR-A (modelled)",
        color=colors[1],
        alpha=0.85,
    )
    # asr-t
    ax.bar(
        x + offsets[2],
        asr_t_means,
        width,
        label="ASR-T (end-to-end)",
        color=colors[2],
        alpha=0.85,
    )
    # benign accuracy
    ax.bar(
        x + offsets[3],
        benign_means,
        width,
        label="Benign Acc.",
        color=colors[3],
        alpha=0.85,
    )

    ax.set_xlabel("Attack", fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Attack Success Rates by Metric", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.0, 1.15)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    png_path = _FIGURES_DIR / f"{save_stem}.png"
    pdf_path = _FIGURES_DIR / f"{save_stem}.pdf"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure: %s (.png + .pdf)", save_stem)
    return str(pdf_path)


def generate_watermark_z_data() -> dict[str, Any]:
    """generate watermark z-score distributions for figure generation."""
    log.info("generating watermark z-score data")
    encoder = create_watermark_encoder("unigram")

    reference_texts = [
        (
            "the advanced memory agent system provides comprehensive capabilities for "
            "storing and retrieving information across multiple interaction contexts. "
            "this includes semantic indexing, temporal awareness, and contextual retrieval "
            "mechanisms that enable sophisticated information management workflows for "
            "enterprise applications and personal assistant deployments."
        ),
        (
            "adversarial attacks on memory systems exploit the fact that language models "
            "often retrieve relevant context from external stores without verification of "
            "the content's authenticity or integrity. injection attacks insert malicious "
            "instructions disguised as benign memories, causing the agent to execute "
            "unintended actions on behalf of an attacker across future sessions."
        ),
        (
            "watermarking provides a statistical guarantee of content provenance by "
            "embedding a secret key-derived signal into generated text. the unigram "
            "watermark partitions the vocabulary into green and red lists and biases "
            "token selection toward the green list during generation. detection uses "
            "a z-score test under the null hypothesis of random token selection."
        ),
        (
            "semantic memory systems are fundamental to autonomous agent operation "
            "in long-horizon tasks. by persisting relevant context across sessions, "
            "agents can maintain coherent task state, user preferences, and accumulated "
            "knowledge without requiring the entire conversation history to fit within "
            "a single context window."
        ),
        (
            "the threat model considered in this work covers three distinct attacker "
            "capabilities: full write access to the memory store, query-only access "
            "via the agent interface, and single-interaction injection via a broadly "
            "applicable anchor passage. each capability corresponds to a distinct "
            "real-world deployment scenario."
        ),
    ]

    z_watermarked = []
    z_clean = []

    for text in reference_texts:
        wm_text = encoder.embed(text, watermark="eval")
        stats_wm = encoder.get_detection_stats(wm_text)
        stats_clean = encoder.get_detection_stats(text)
        z_watermarked.append(stats_wm.get("z_score", 0.0))
        z_clean.append(stats_clean.get("z_score", 0.0))

    # generate larger sample by varying content length
    import random as rng_mod

    rng = rng_mod.Random(42)
    for i in range(45):
        base = rng.choice(reference_texts)
        # watermarked: full text
        wm_text = encoder.embed(base, watermark=f"eval_{i}")
        stats = encoder.get_detection_stats(wm_text)
        z_watermarked.append(stats.get("z_score", 0.0))
        # clean: first half of text (shorter, more variable)
        half = base[: len(base) // 2]
        stats_c = encoder.get_detection_stats(half)
        z_clean.append(stats_c.get("z_score", 0.0))

    # sweep z-threshold for roc-style ablation
    import numpy as np

    thresholds = np.linspace(0.0, 10.0, 50).tolist()
    tpr_vals, fpr_vals, f1_vals = [], [], []

    for thr in thresholds:
        tp = sum(1 for z in z_watermarked if z >= thr)
        fn = len(z_watermarked) - tp
        fp = sum(1 for z in z_clean if z >= thr)
        tn = len(z_clean) - fp

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tpr
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        tpr_vals.append(tpr)
        fpr_vals.append(fpr)
        f1_vals.append(f1)

    log.info(
        "z-score data: %d watermarked (mean=%.2f), %d clean (mean=%.2f)",
        len(z_watermarked),
        float(np.mean(z_watermarked)),
        len(z_clean),
        float(np.mean(z_clean)),
    )

    return {
        "z_watermarked": z_watermarked,
        "z_clean": z_clean,
        "threshold_vals": thresholds,
        "tpr_vals": tpr_vals,
        "fpr_vals": fpr_vals,
        "f1_vals": f1_vals,
    }


def _plot_matrix_heatmap_from_dict(matrix_dict: dict[str, Any], save_stem: str) -> str:
    """
    generate the attack-defense matrix heatmap from the serialized dict.

    produces a two-panel figure:
      left:  asr-r under defense (lower = better for defender)
      right: defense effectiveness = 1 - asr_r_under / asr_r_baseline
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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

    results_raw = matrix_dict.get("results", {})

    # build matrices (nan for missing pairs)
    n_atk = len(attack_order)
    n_def = len(defense_order)
    asr_mat = np.full((n_atk, n_def), np.nan)
    eff_mat = np.full((n_atk, n_def), np.nan)

    for ai, atk in enumerate(attack_order):
        for di, dfn in enumerate(defense_order):
            pair = results_raw.get(atk, {}).get(dfn, {})
            if pair:
                asr_mat[ai, di] = pair.get("asr_r_under_defense", np.nan)
                eff_mat[ai, di] = pair.get("defense_effectiveness", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    panels = [
        (axes[0], asr_mat, "ASR-R Under Defense\n(lower = safer)", "YlOrRd", 0.0, 1.0),
        (
            axes[1],
            eff_mat,
            "Defense Effectiveness\n(higher = better)",
            "RdYlGn",
            0.0,
            1.0,
        ),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_def))
        ax.set_xticklabels(defense_labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(n_atk))
        ax.set_yticklabels(attack_labels, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for ai2 in range(n_atk):
            for di2 in range(n_def):
                v = data[ai2, di2]
                if not np.isnan(v):
                    color = "white" if v > 0.65 else "black"
                    ax.text(
                        di2,
                        ai2,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                        fontweight="bold",
                    )

    fig.suptitle(
        "Attack-Defense Interaction Matrix (3 × 5)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    png_path = _FIGURES_DIR / f"{save_stem}.png"
    pdf_path = _FIGURES_DIR / f"{save_stem}.pdf"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    log.info("saved figure: %s (.png + .pdf)", save_stem)
    return str(pdf_path)


def generate_all_figures(
    result: Any,
    wm_data: dict[str, Any],
    run_ablations: bool,
) -> dict[str, str]:
    """generate all publication-quality figures for the paper."""
    import matplotlib

    matplotlib.use("Agg")

    log.info("generating all figures → %s", _FIGURES_DIR)
    vis = BenchmarkVisualizer(output_dir=str(_FIGURES_DIR))
    saved: dict[str, str] = {}

    attack_summaries = getattr(result, "attack_summaries", {})
    adaptive_results = getattr(result, "adaptive_sad_results", {})
    evasion_results = getattr(result, "evasion_results", {})
    ablation_results = getattr(result, "ablation_results", {})
    matrix_dict = getattr(result, "matrix_result_dict", None)

    # --- 1. custom attack asr bar chart (fig_01_attack_asr) ---
    try:
        if attack_summaries:
            path = _plot_attack_asr_bars(attack_summaries, "fig_01_attack_asr")
            saved["fig_01_attack_asr"] = path
    except Exception as exc:
        log.warning("attack asr figure failed: %s", exc)

    # --- 2. watermark z-score distribution + threshold ablation ---
    try:
        wm_saved = vis.generate_watermark_figures(
            z_watermarked=wm_data["z_watermarked"],
            z_clean=wm_data["z_clean"],
            threshold_vals=wm_data["threshold_vals"],
            tpr_vals=wm_data["tpr_vals"],
            fpr_vals=wm_data["fpr_vals"],
            f1_vals=wm_data["f1_vals"],
            prefix="wm",
        )
        saved.update(wm_saved)
        log.info("watermark figures: %d saved", len(wm_saved))
    except Exception as exc:
        log.warning("watermark figure generation error: %s", exc)

    # --- 3. attack-defense matrix heatmap (custom dict-based function) ---
    try:
        if matrix_dict:
            path = _plot_matrix_heatmap_from_dict(
                matrix_dict, "m12_attack_defense_matrix"
            )
            saved["m12_attack_defense_matrix"] = path
    except Exception as exc:
        log.warning("matrix heatmap failed: %s", exc)

    # --- 4. phase 13: adaptive tradeoff, evasion analysis, ablation curves ---
    try:
        p13_saved = vis.generate_phase13_figures(
            attack_summaries=attack_summaries,
            adaptive_results=adaptive_results,
            evasion_results=evasion_results,
            ablation_results=ablation_results if run_ablations else {},
            prefix="p13",
        )
        saved.update(p13_saved)
        log.info("phase 13 figures: %d saved", len(p13_saved))
    except Exception as exc:
        log.warning("phase 13 figure generation error: %s", exc)

    log.info("total figures saved: %d", len(saved))
    return saved


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """run full paper result generation."""
    parser = argparse.ArgumentParser(
        description="generate all paper results, tables, and figures"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="quick mode: corpus=100, n_seeds=3, no ablations (default: full)",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        default=False,
        help="generate only latex tables (skip figures)",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        default=False,
        help="generate only figures (skip table generation)",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=None,
        help="override corpus size (default: 200 full, 100 quick)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="override number of seeds (default: 5 full, 3 quick)",
    )
    args = parser.parse_args(argv)

    # resolve parameters
    if args.quick:
        corpus_size = args.corpus_size or 100
        n_seeds = args.n_seeds or 3
        run_ablations = False
    else:
        corpus_size = args.corpus_size or 200
        n_seeds = args.n_seeds or 5
        run_ablations = True

    n_poison = 5
    top_k = 5
    seed_base = 42

    log.info(
        "paper result generation: corpus=%d n_seeds=%d n_poison=%d top_k=%d"
        " run_ablations=%s",
        corpus_size,
        n_seeds,
        n_poison,
        top_k,
        run_ablations,
    )

    _ensure_dirs()
    t_start = time.time()

    # --- comprehensive evaluation ---
    result, evaluator = run_comprehensive_evaluation(
        corpus_size=corpus_size,
        n_seeds=n_seeds,
        n_poison=n_poison,
        top_k=top_k,
        seed_base=seed_base,
        run_ablations=run_ablations,
    )

    # --- tables ---
    if not args.figures_only:
        tables_saved = generate_tables(result, evaluator)
        log.info("tables saved: %d", len(tables_saved))

    # --- figures ---
    if not args.tables_only:
        wm_data = generate_watermark_z_data()
        figs_saved = generate_all_figures(result, wm_data, run_ablations)
        log.info("figures saved: %d", len(figs_saved))

    elapsed = time.time() - t_start
    log.info("all paper results generated in %.1fs (%.1fmin)", elapsed, elapsed / 60)
    print(
        f"\npaper results complete in {elapsed:.0f}s.\n"
        f"  tables  → {_TABLES_DIR}\n"
        f"  figures → {_FIGURES_DIR}\n"
    )


if __name__ == "__main__":
    main()
