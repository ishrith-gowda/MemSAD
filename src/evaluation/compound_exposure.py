"""
compound exposure analysis for persistent memory poisoning.

demonstrates that even low per-query asr-r compounds over multiple
sessions/queries, making "weak" attacks a serious persistent threat.
key insight: memory poisoning persists indefinitely, so the relevant
metric is not per-query asr-r but cumulative probability of at least
one successful retrieval over the agent's lifetime.

all comments are lowercase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ExposureResult:
    """result of compound exposure analysis for one attack."""

    attack_name: str
    asr_r: float
    # p(at least 1 success) at each session count
    session_counts: list[int] = field(default_factory=list)
    cumulative_probs: list[float] = field(default_factory=list)
    # expected sessions to first compromise
    expected_sessions_to_compromise: float = 0.0
    # sessions needed for 50%, 90%, 95% compromise probability
    sessions_for_50pct: int = 0
    sessions_for_90pct: int = 0
    sessions_for_95pct: int = 0


@dataclass
class CompoundExposureAnalysis:
    """full compound exposure analysis across attacks."""

    results: dict[str, ExposureResult] = field(default_factory=dict)
    # multi-query-per-session analysis
    queries_per_session: int = 1
    # with-defense analysis
    defense_results: dict[str, ExposureResult] = field(default_factory=dict)


def compute_cumulative_compromise(
    asr_r: float,
    max_sessions: int = 100,
    queries_per_session: int = 1,
) -> tuple[list[int], list[float]]:
    """
    compute p(at least 1 successful poisoned retrieval) over n sessions.

    for independent queries with per-query asr-r:
      p(compromise in n queries) = 1 - (1 - asr_r)^(n * queries_per_session)

    args:
        asr_r: per-query attack success rate (retrieval)
        max_sessions: maximum number of sessions to compute
        queries_per_session: number of queries per session

    returns:
        (session_counts, cumulative_probs)
    """
    sessions = list(range(1, max_sessions + 1))
    probs = []
    for n in sessions:
        total_queries = n * queries_per_session
        p_no_compromise = (1.0 - asr_r) ** total_queries
        probs.append(1.0 - p_no_compromise)
    return sessions, probs


def expected_sessions_to_compromise(
    asr_r: float,
    queries_per_session: int = 1,
) -> float:
    """
    expected number of sessions until first successful poisoned retrieval.

    for geometric distribution: E[N] = 1 / (1 - (1-asr_r)^q)
    where q = queries_per_session.

    args:
        asr_r: per-query attack success rate
        queries_per_session: queries per session

    returns:
        expected sessions (inf if asr_r == 0)
    """
    if asr_r <= 0.0:
        return float("inf")
    if asr_r >= 1.0:
        return 1.0
    p_session = 1.0 - (1.0 - asr_r) ** queries_per_session
    if p_session <= 0.0:
        return float("inf")
    return 1.0 / p_session


def sessions_for_threshold(
    asr_r: float,
    threshold: float,
    queries_per_session: int = 1,
) -> int:
    """
    minimum sessions needed to reach a given compromise probability.

    solves: 1 - (1 - asr_r)^(n*q) >= threshold
    => n >= log(1 - threshold) / (q * log(1 - asr_r))

    args:
        asr_r: per-query asr-r
        threshold: target cumulative probability (e.g. 0.95)
        queries_per_session: queries per session

    returns:
        minimum session count (int), or -1 if asr_r == 0
    """
    if asr_r <= 0.0:
        return -1
    if asr_r >= 1.0:
        return 1
    log_complement = np.log(1.0 - asr_r)
    if log_complement >= 0.0:
        return -1
    n_queries = np.log(1.0 - threshold) / log_complement
    n_sessions = n_queries / queries_per_session
    return int(np.ceil(n_sessions))


def run_compound_exposure_analysis(
    attack_asr_r: dict[str, float] | None = None,
    max_sessions: int = 100,
    queries_per_session: int = 5,
    defense_reduction: dict[str, float] | None = None,
) -> CompoundExposureAnalysis:
    """
    run full compound exposure analysis.

    args:
        attack_asr_r: mapping of attack name -> asr-r at n=1000.
                      defaults to measured values.
        max_sessions: maximum sessions to analyze
        queries_per_session: queries per session (realistic agent usage)
        defense_reduction: mapping of attack name -> residual asr-r after defense

    returns:
        CompoundExposureAnalysis with results for all attacks
    """
    if attack_asr_r is None:
        # measured values at |M|=1000, 5 seeds
        attack_asr_r = {
            "agentpoison": 1.00,
            "minja": 0.14,
            "injecmem": 0.07,
        }

    if defense_reduction is None:
        # residual asr-r after composite defense
        defense_reduction = {
            "agentpoison": 0.00,
            "minja": 0.00,
            "injecmem": 0.00,
        }

    analysis = CompoundExposureAnalysis(queries_per_session=queries_per_session)

    # undefended analysis
    for name, asr_r in attack_asr_r.items():
        sessions, probs = compute_cumulative_compromise(
            asr_r, max_sessions, queries_per_session
        )
        result = ExposureResult(
            attack_name=name,
            asr_r=asr_r,
            session_counts=sessions,
            cumulative_probs=probs,
            expected_sessions_to_compromise=expected_sessions_to_compromise(
                asr_r, queries_per_session
            ),
            sessions_for_50pct=sessions_for_threshold(asr_r, 0.50, queries_per_session),
            sessions_for_90pct=sessions_for_threshold(asr_r, 0.90, queries_per_session),
            sessions_for_95pct=sessions_for_threshold(asr_r, 0.95, queries_per_session),
        )
        analysis.results[name] = result

    # defended analysis
    for name, residual_asr_r in defense_reduction.items():
        sessions, probs = compute_cumulative_compromise(
            residual_asr_r, max_sessions, queries_per_session
        )
        result = ExposureResult(
            attack_name=f"{name}_defended",
            asr_r=residual_asr_r,
            session_counts=sessions,
            cumulative_probs=probs,
            expected_sessions_to_compromise=expected_sessions_to_compromise(
                residual_asr_r, queries_per_session
            ),
            sessions_for_50pct=sessions_for_threshold(
                residual_asr_r, 0.50, queries_per_session
            ),
            sessions_for_90pct=sessions_for_threshold(
                residual_asr_r, 0.90, queries_per_session
            ),
            sessions_for_95pct=sessions_for_threshold(
                residual_asr_r, 0.95, queries_per_session
            ),
        )
        analysis.defense_results[name] = result

    return analysis


def generate_compound_exposure_figure(
    analysis: CompoundExposureAnalysis,
    output_dir: str = "docs/neurips2026/figures",
) -> str:
    """
    generate compound exposure figure for the paper.

    shows cumulative compromise probability vs. session count for each attack,
    with and without defense.

    args:
        analysis: compound exposure analysis results
        output_dir: output directory for figure

    returns:
        path to saved figure
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # color scheme
    colors = {
        "agentpoison": "#d62728",
        "minja": "#ff7f0e",
        "injecmem": "#2ca02c",
    }
    labels = {
        "agentpoison": "AgentPoison (ASR-R=1.00)",
        "minja": "MINJA (ASR-R=0.14)",
        "injecmem": "InjecMEM (ASR-R=0.07)",
    }

    # left panel: undefended compound exposure
    for name, result in analysis.results.items():
        ax1.plot(
            result.session_counts,
            result.cumulative_probs,
            color=colors.get(name, "gray"),
            linewidth=2.0,
            label=labels.get(name, name),
        )

    # threshold lines
    for thresh, ls, lbl in [
        (0.50, "--", "50%"),
        (0.90, "-.", "90%"),
        (0.95, ":", "95%"),
    ]:
        ax1.axhline(y=thresh, color="gray", linestyle=ls, alpha=0.5, linewidth=0.8)
        ax1.text(
            len(analysis.results["minja"].session_counts) + 1,
            thresh - 0.02,
            lbl,
            fontsize=7,
            color="gray",
            va="top",
        )

    ax1.set_xlabel("Agent Sessions", fontsize=10)
    ax1.set_ylabel("P(At Least 1 Compromise)", fontsize=10)
    ax1.set_title("Compound Exposure (No Defense)", fontsize=11)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_xlim(1, len(analysis.results["minja"].session_counts))
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # right panel: with vs without defense
    for name, result in analysis.results.items():
        ax2.plot(
            result.session_counts,
            result.cumulative_probs,
            color=colors.get(name, "gray"),
            linewidth=2.0,
            alpha=0.4,
            linestyle="--",
        )

    for name, result in analysis.defense_results.items():
        base_name = name
        if result.asr_r > 0:
            ax2.plot(
                result.session_counts,
                result.cumulative_probs,
                color=colors.get(base_name, "gray"),
                linewidth=2.0,
                label=f"{base_name} + Composite",
            )
        else:
            # zero residual — flat line at 0
            ax2.axhline(
                y=0,
                color=colors.get(base_name, "gray"),
                linewidth=2.0,
                label=f"{base_name} + Composite",
            )

    ax2.set_xlabel("Agent Sessions", fontsize=10)
    ax2.set_ylabel("P(At Least 1 Compromise)", fontsize=10)
    ax2.set_title("Compound Exposure (With Composite Defense)", fontsize=11)
    ax2.legend(fontsize=8, loc="center right")
    ax2.set_xlim(1, len(analysis.results["minja"].session_counts))
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path / "fig_compound_exposure.pdf"
    png_path = out_path / "fig_compound_exposure.png"
    fig.savefig(str(pdf_path), bbox_inches="tight", dpi=300)
    fig.savefig(str(png_path), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"  saved: {pdf_path}")
    return str(pdf_path)


def generate_exposure_table(
    analysis: CompoundExposureAnalysis,
    output_dir: str = "results/tables",
) -> str:
    """generate latex table for compound exposure analysis."""
    attack_labels = {
        "agentpoison": r"\agentpoison{}",
        "minja": r"\minja{}",
        "injecmem": r"\injecmem{}",
    }

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Compound exposure: cumulative compromise probability over"
        r" agent sessions ($q = "
        + str(analysis.queries_per_session)
        + r"$ queries/session, $|\cM| = 1{,}000$).}",
        r"  \label{tab:compound_exposure}",
        r"  \vspace{2pt}",
        r"  \begin{tabular}{l c cccc c}",
        r"    \toprule",
        r"    Attack & $\asrr$ & $N_{50\%}$ & $N_{90\%}$"
        r" & $N_{95\%}$ & $\EE[N]$ & Defended \\",
        r"    \midrule",
    ]

    for name in ["agentpoison", "minja", "injecmem"]:
        r = analysis.results[name]
        label = attack_labels.get(name, name)
        n50 = str(r.sessions_for_50pct) if r.sessions_for_50pct > 0 else "1"
        n90 = str(r.sessions_for_90pct) if r.sessions_for_90pct > 0 else "1"
        n95 = str(r.sessions_for_95pct) if r.sessions_for_95pct > 0 else "1"
        en = f"{r.expected_sessions_to_compromise:.1f}"
        # defended status
        dr = analysis.defense_results.get(name)
        defended = r"$\asrr^* = 0$" if dr and dr.asr_r == 0.0 else "---"
        lines.append(
            f"    {label} & ${r.asr_r:.2f}$ & {n50} & {n90}"
            f" & {n95} & {en} & {defended} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \\[2pt]",
        r"  {\footnotesize $N_{p}$: sessions to reach $p$ compromise probability."
        r" $\EE[N]$: expected sessions to first compromise."
        r" Defended: residual $\asrr$ after composite portfolio.}",
        r"  \vspace{-6pt}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    table_path = out_path / "table_compound_exposure.tex"
    table_path.write_text(tex)
    print(f"  saved: {table_path}")
    return tex


def save_results(
    analysis: CompoundExposureAnalysis,
    output_dir: str = "results/compound_exposure",
) -> None:
    """save compound exposure results to json."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = {
        "queries_per_session": analysis.queries_per_session,
        "results": {k: asdict(v) for k, v in analysis.results.items()},
        "defense_results": {k: asdict(v) for k, v in analysis.defense_results.items()},
    }

    json_path = out_path / "compound_exposure_results.json"
    json_path.write_text(json.dumps(data, indent=2))
    print(f"  saved: {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="compound exposure analysis for persistent memory poisoning"
    )
    parser.add_argument("--max-sessions", type=int, default=100)
    parser.add_argument("--queries-per-session", type=int, default=5)
    args = parser.parse_args()

    print("running compound exposure analysis ...")
    analysis = run_compound_exposure_analysis(
        max_sessions=args.max_sessions,
        queries_per_session=args.queries_per_session,
    )

    print("\nresults (undefended):")
    for name, r in analysis.results.items():
        print(
            f"  {name}: asr_r={r.asr_r:.2f}, "
            f"E[N]={r.expected_sessions_to_compromise:.1f}, "
            f"N_50={r.sessions_for_50pct}, "
            f"N_90={r.sessions_for_90pct}, "
            f"N_95={r.sessions_for_95pct}"
        )

    print("\ngenerating figure ...")
    generate_compound_exposure_figure(analysis)

    print("\ngenerating table ...")
    generate_exposure_table(analysis)

    print("\nsaving results ...")
    save_results(analysis)

    print("done.")
