"""
generate updated paper tables and figures from phase 28 results.

reads results/phase28/phase28_results.json and produces:
  - results/tables/table1_attack_results.tex (updated with 1k scale + measured asr-a)
  - results/tables/table5_adaptive_sad.tex (updated with 1k scale)
  - docs/neurips2026/figures/fig_corpus_scaling.pdf (200 vs 1000 comparison)
  - docs/neurips2026/figures/fig_measured_asr_a.pdf (gpt-2 vs gpt-4o-mini vs modelled)

all comments are lowercase.
"""

from __future__ import annotations

import json
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pathlib import Path


def load_results() -> dict:
    """load phase 28 results json."""
    path = Path("results/phase28/phase28_results.json")
    with open(path) as f:
        return dict(json.load(f))


def generate_table1(results: dict) -> None:
    """generate updated table 1 with 1k corpus + measured asr-a columns."""
    table_path = Path("results/tables/table1_attack_results.tex")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Attack evaluation results on synthetic memory corpus",
        r"  ($|\mathcal{M}|=1000$, $k=5$, 100 victim queries, seed-averaged over 5 trials).",
        r"  \asra{} columns: modelled (from prior work), GPT-2 lower bound, GPT-4o-mini.",
        r"  95\% CI via percentile bootstrap.}",
        r"\label{tab:attack_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"Attack & $\asrr$ & $\asra^{\text{mod}}$ & "
            r"$\asra^{\text{GPT-2}}$ & $\asra^{\text{4o-mini}}$ & "
            r"$\asrt$ & Ben.\ Acc. \\"
        ),
        r"\midrule",
    ]

    attack_names = {
        "agent_poison": r"\agentpoison{} \citep{chen2024agentpoison}",
        "minja": r"\minja{} \citep{dong2025minja}",
        "injecmem": r"\injecmem{} \citep{injecmem2026}",
    }

    for ar in results["attack_results"]:
        name = attack_names.get(ar["attack_type"], ar["attack_type"])
        ci = ar["asr_r_ci"]
        asr_r_str = f"${ar['asr_r']:.3f}_{{[{ci[0]:.3f},{ci[1]:.3f}]}}$"

        gpt2 = (
            f"${ar['asr_a_gpt2']:.2f}$" if ar.get("asr_a_gpt2") is not None else "---"
        )
        gpt4o = (
            f"${ar['asr_a_gpt4o_mini']:.2f}$"
            if ar.get("asr_a_gpt4o_mini") is not None
            else "---"
        )

        lines.append(
            f"{name} & {asr_r_str} & ${ar['asr_a_modelled']:.2f}$ & "
            f"{gpt2} & {gpt4o} & "
            f"${ar['asr_t_modelled']:.3f}$ & ${ar['benign_accuracy']:.3f}$ \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  table1 saved to {table_path}")


def generate_scaling_figure(results: dict) -> None:
    """generate corpus scaling comparison figure (200 vs 1000)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  skipping figures (matplotlib not available)")
        return

    fig_dir = Path("docs/neurips2026/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 200-entry baseline values (from phase 9 results)
    baseline_200 = {
        "agent_poison": 1.000,
        "minja": 0.650,
        "injecmem": 0.550,
    }

    # 1000-entry values from phase 28
    scale_1000 = {}
    for ar in results["attack_results"]:
        scale_1000[ar["attack_type"]] = ar["asr_r"]

    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    attacks = ["agent_poison", "minja", "injecmem"]
    labels = [
        r"\textsc{AgentPoison}",
        r"\textsc{Minja}",
        r"\textsc{InjecMem}",
    ]
    x = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bars_200 = ax.bar(
        x - width / 2,
        [baseline_200[a] for a in attacks],
        width,
        label=r"$|\mathcal{M}|=200$",
        color="#7ea6cf",
        edgecolor="white",
        linewidth=0.6,
    )
    bars_1000 = ax.bar(
        x + width / 2,
        [scale_1000.get(a, 0) for a in attacks],
        width,
        label=r"$|\mathcal{M}|=1{,}000$",
        color="#c47070",
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_ylabel(r"ASR-R")
    ax.set_title(
        r"\textbf{Corpus Scaling:} ASR-R at $|\mathcal{M}|=200$ vs.\ $1{,}000$",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

    # add value labels
    for bar in bars_200:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars_1000:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    neurips_fig_dir = Path("docs/neurips2026/figures")
    neurips_fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_corpus_scaling.{ext}", dpi=300)
        fig.savefig(neurips_fig_dir / f"fig_corpus_scaling.{ext}", dpi=300)
    plt.close(fig)
    print("  fig_corpus_scaling saved")


def generate_asr_a_figure(results: dict) -> None:
    """generate measured asr-a comparison figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  skipping figures (matplotlib not available)")
        return

    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    fig_dir = Path("docs/neurips2026/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    neurips_fig_dir = Path("docs/neurips2026/figures")
    neurips_fig_dir.mkdir(parents=True, exist_ok=True)

    attacks = ["agent_poison", "minja", "injecmem"]
    labels = [
        r"\textsc{AgentPoison}",
        r"\textsc{Minja}",
        r"\textsc{InjecMem}",
    ]

    modelled = []
    gpt2 = []
    gpt4o = []

    for at in attacks:
        for ar in results["attack_results"]:
            if ar["attack_type"] == at:
                modelled.append(ar["asr_a_modelled"])
                gpt2.append(ar.get("asr_a_gpt2") or 0.0)
                gpt4o.append(ar.get("asr_a_gpt4o_mini") or 0.0)

    x = np.arange(len(attacks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.bar(
        x - width,
        modelled,
        width,
        label=r"Modelled",
        color="#7ea6cf",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x,
        gpt2,
        width,
        label=r"GPT-2 (lower bound)",
        color="#e8b88a",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + width,
        gpt4o,
        width,
        label=r"GPT-4o-mini",
        color="#8fbf7f",
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_ylabel(r"ASR-A")
    ax.set_title(r"\textbf{Measured vs.\ Modelled ASR-A}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(fig_dir / f"fig_measured_asr_a.{ext}", dpi=300)
        fig.savefig(neurips_fig_dir / f"fig_measured_asr_a.{ext}", dpi=300)
    plt.close(fig)
    print("  fig_measured_asr_a saved")


def generate_encoder_figure(results: dict) -> None:
    """generate multi-encoder asr-r comparison figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  skipping figures (matplotlib not available)")
        return

    fig_dir = Path("docs/neurips2026/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    enc_results = results.get("encoder_results", {})
    attack_table = enc_results.get("attack_table", {})
    defense_table = enc_results.get("defense_table", {})

    if not attack_table:
        print("  skipping encoder figure (no encoder results)")
        return

    encoder_names = list(attack_table.keys())
    attack_types = ["agent_poison", "minja", "injecmem"]
    attack_labels = ["AgentPoison", "MINJA", "InjecMEM"]

    # defense auroc figure
    if defense_table:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # left: asr-r by encoder
        ax = axes[0]
        x = np.arange(len(encoder_names))
        width = 0.25
        for i, (at, label) in enumerate(zip(attack_types, attack_labels)):
            vals = [attack_table.get(enc, {}).get(at, 0) for enc in encoder_names]
            ax.bar(x + i * width, vals, width, label=label)
        ax.set_ylabel("ASR-R", fontsize=11)
        ax.set_title("Cross-Encoder ASR-R (Generic Templates)", fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(encoder_names, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # right: sad auroc by encoder
        ax2 = axes[1]
        for i, (at, label) in enumerate(zip(attack_types, attack_labels)):
            vals = [
                defense_table.get(enc, {}).get(at, {}).get("auroc", 0)
                for enc in encoder_names
            ]
            ax2.bar(x + i * width, vals, width, label=label)
        ax2.set_ylabel("SAD AUROC", fontsize=11)
        ax2.set_title("Cross-Encoder SAD AUROC", fontsize=12)
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(encoder_names, rotation=45, ha="right", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(fig_dir / f"fig_multi_encoder.{ext}", dpi=300)
        plt.close(fig)
        print("  fig_multi_encoder saved")


def generate_encoder_table(results: dict) -> None:
    """generate multi-encoder latex table."""
    enc_results = results.get("encoder_results", {})
    attack_table = enc_results.get("attack_table", {})
    defense_table = enc_results.get("defense_table", {})

    if not attack_table:
        print("  skipping encoder table (no results)")
        return

    table_path = Path("results/tables/table_multi_encoder_1k.tex")

    encoder_names = list(attack_table.keys())
    attacks = ["agent_poison", "minja", "injecmem"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Multi-encoder evaluation at $|\mathcal{M}|=1000$.",
        r"  ASR-R measures retrieval success of generic (non-optimized) poison templates.",
        r"  SAD AUROC measures separation quality using \memsad{} ($\sigma=2.0$, combined mode).}",
        r"\label{tab:multi_encoder}",
        r"\begin{tabular}{l ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{ASR-R} & \multicolumn{3}{c}{SAD AUROC} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Encoder & AP & MJ & IM & AP & MJ & IM \\",
        r"\midrule",
    ]

    for enc in encoder_names:
        atk = attack_table.get(enc, {})
        dfs = defense_table.get(enc, {})

        asr_vals = [f"${atk.get(a, 0):.3f}$" for a in attacks]
        auroc_vals = [f"${dfs.get(a, {}).get('auroc', 0):.3f}$" for a in attacks]

        lines.append(
            f"  {enc} & {' & '.join(asr_vals)} & {' & '.join(auroc_vals)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  encoder table saved to {table_path}")


def main() -> None:
    """generate all phase 28 tables and figures."""
    print("generating phase 28 paper tables and figures...")

    results = load_results()

    generate_table1(results)
    generate_scaling_figure(results)
    generate_asr_a_figure(results)
    generate_encoder_figure(results)
    generate_encoder_table(results)

    print("done.")


if __name__ == "__main__":
    main()
