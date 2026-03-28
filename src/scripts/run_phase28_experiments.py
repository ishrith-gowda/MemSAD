"""
phase 28: experimental hardening for neurips 2026.

scales evaluation to 1000-entry corpus, 100 queries, measured asr-a,
and multi-encoder validation. produces all paper tables and figures.

usage:
    .venv/bin/python src/scripts/run_phase28_experiments.py [--quick] [--skip-openai] [--skip-gpt2]

all comments are lowercase.
"""

from __future__ import annotations

import json
import os
import sys
import time

# ensure src/ is on the python path (consistent with conftest.py convention)
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import dotenv
import numpy as np

# load .env for OPENAI_API_KEY
dotenv.load_dotenv()


# ---------------------------------------------------------------------------
# result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AttackResult:
    """single attack evaluation result at scale."""

    attack_type: str
    corpus_size: int
    n_queries: int
    n_poison: int
    top_k: int
    asr_r: float
    asr_r_ci: tuple[float, float]
    asr_a_modelled: float
    asr_a_gpt2: float | None = None
    asr_a_gpt4o_mini: float | None = None
    asr_t_modelled: float = 0.0
    asr_t_gpt2: float | None = None
    asr_t_gpt4o_mini: float | None = None
    benign_accuracy: float = 1.0
    isr: float = 0.0
    seed: int = 42


@dataclass
class DefenseResult:
    """defense evaluation result at scale."""

    defense: str
    attack_type: str
    tpr: float
    fpr: float
    auroc: float | None = None


@dataclass
class Phase28Results:
    """aggregate results from phase 28 experiments."""

    corpus_size: int
    n_victim_queries: int
    n_benign_queries: int
    n_seeds: int
    attack_results: list[AttackResult] = field(default_factory=list)
    defense_results: list[DefenseResult] = field(default_factory=list)
    encoder_results: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# experiment runner
# ---------------------------------------------------------------------------


class Phase28Runner:
    """
    runs all phase 28 experiments: corpus scaling, real asr-a, multi-encoder.

    args:
        corpus_size: number of benign entries (default 1000)
        n_victim_queries: number of victim queries (default 100)
        n_benign_queries: number of benign queries (default 100)
        n_seeds: number of random seeds for bootstrap ci (default 5)
        top_k: retrieval depth (default 5)
        skip_openai: skip openai api calls (default false)
        skip_gpt2: skip local gpt2 evaluation (default false)
        output_dir: directory for results
    """

    def __init__(
        self,
        corpus_size: int = 1000,
        n_victim_queries: int = 100,
        n_benign_queries: int = 100,
        n_seeds: int = 5,
        top_k: int = 5,
        skip_openai: bool = False,
        skip_gpt2: bool = False,
        output_dir: str = "results/phase28",
    ) -> None:
        self.corpus_size = corpus_size
        self.n_victim_queries = n_victim_queries
        self.n_benign_queries = n_benign_queries
        self.n_seeds = n_seeds
        self.top_k = top_k
        self.skip_openai = skip_openai
        self.skip_gpt2 = skip_gpt2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = Phase28Results(
            corpus_size=corpus_size,
            n_victim_queries=n_victim_queries,
            n_benign_queries=n_benign_queries,
            n_seeds=n_seeds,
        )

    def run_all(self) -> Phase28Results:
        """run all phase 28 experiments in sequence."""
        print("phase 28: experimental hardening")
        print(f"  corpus_size={self.corpus_size}")
        print(f"  n_victim_queries={self.n_victim_queries}")
        print(f"  n_benign_queries={self.n_benign_queries}")
        print(f"  n_seeds={self.n_seeds}")
        print(f"  skip_openai={self.skip_openai}")
        print(f"  skip_gpt2={self.skip_gpt2}")
        print()

        t0 = time.time()

        # stage 1: attack evaluation at scale
        self._run_attack_evaluation()

        # stage 2: measured asr-a (gpt-2 local)
        if not self.skip_gpt2:
            self._run_gpt2_asr_a()

        # stage 3: measured asr-a (gpt-4o-mini)
        if not self.skip_openai:
            self._run_openai_asr_a()

        # stage 4: defense evaluation at scale
        self._run_defense_evaluation()

        # stage 5: multi-encoder evaluation
        self._run_encoder_evaluation()

        self.results.timing["total_s"] = time.time() - t0
        print(f"\ntotal time: {self.results.timing['total_s']:.1f}s")

        # save results
        self._save_results()

        return self.results

    # -----------------------------------------------------------------------
    # stage 1: attack evaluation at scale
    # -----------------------------------------------------------------------

    def _run_attack_evaluation(self) -> None:
        """evaluate all attacks at scale using retrieval simulator."""
        from evaluation.retrieval_sim import RetrievalSimulator

        print("--- stage 1: attack evaluation at scale ---")
        t0 = time.time()

        attack_types = ["agent_poison", "minja", "injecmem"]
        all_asr_r: dict[str, list[float]] = {at: [] for at in attack_types}
        all_benign_acc: dict[str, list[float]] = {at: [] for at in attack_types}

        for seed_idx in range(self.n_seeds):
            sim = RetrievalSimulator(
                corpus_size=self.corpus_size,
                top_k=self.top_k,
                n_poison_per_attack=5,
                seed=42 + seed_idx,
                use_trigger_optimization=True,
            )

            for attack_type in attack_types:
                metrics = sim.evaluate_attack(attack_type)
                all_asr_r[attack_type].append(metrics.asr_r)
                all_benign_acc[attack_type].append(metrics.benign_accuracy)

            if seed_idx == 0:
                print(
                    f"  seed 0: ap={all_asr_r['agent_poison'][0]:.3f}, "
                    f"mj={all_asr_r['minja'][0]:.3f}, "
                    f"im={all_asr_r['injecmem'][0]:.3f}"
                )

        # compute bootstrap ci and store results
        modelled_asr_a = {
            "agent_poison": 0.68,
            "minja": 0.76,
            "injecmem": 0.57,
        }

        for attack_type in attack_types:
            values = np.array(all_asr_r[attack_type])
            mean_asr_r = float(np.mean(values))
            ci_lo = float(np.percentile(values, 2.5))
            ci_hi = float(np.percentile(values, 97.5))
            mean_ba = float(np.mean(all_benign_acc[attack_type]))
            ma = modelled_asr_a[attack_type]

            self.results.attack_results.append(
                AttackResult(
                    attack_type=attack_type,
                    corpus_size=self.corpus_size,
                    n_queries=self.n_victim_queries if self.corpus_size > 200 else 20,
                    n_poison=5,
                    top_k=self.top_k,
                    asr_r=mean_asr_r,
                    asr_r_ci=(ci_lo, ci_hi),
                    asr_a_modelled=ma,
                    asr_t_modelled=mean_asr_r * ma,
                    benign_accuracy=mean_ba,
                )
            )
            print(
                f"  {attack_type}: asr_r={mean_asr_r:.3f} "
                f"[{ci_lo:.3f}, {ci_hi:.3f}], "
                f"ben_acc={mean_ba:.3f}"
            )

        self.results.timing["attack_eval_s"] = time.time() - t0
        print(f"  time: {self.results.timing['attack_eval_s']:.1f}s\n")

    # -----------------------------------------------------------------------
    # stage 2: measured asr-a (gpt-2 local)
    # -----------------------------------------------------------------------

    def _run_gpt2_asr_a(self) -> None:
        """measure asr-a using local gpt-2 agent."""
        print("--- stage 2: measured asr-a (gpt-2 local) ---")
        t0 = time.time()

        try:
            from evaluation.agent_eval import LocalAgentEvaluator
            from evaluation.retrieval_sim import RetrievalSimulator

            evaluator = LocalAgentEvaluator(model_name="gpt2", seed=42)

            # build a simulator to generate poison entries and memory system
            sim = RetrievalSimulator(
                corpus_size=self.corpus_size,
                top_k=self.top_k,
                n_poison_per_attack=5,
                seed=42,
                use_trigger_optimization=True,
            )

            for attack_type in ["agent_poison", "minja", "injecmem"]:
                # generate poison entries via simulator internals
                poison_entries = sim._generate_poison_entries(
                    attack_type, sim._victim_queries
                )
                poison_texts = [p["content"] for p in poison_entries]

                # build vector memory with benign + poison
                mem, _poison_keys = sim._build_vector_memory(poison_entries)

                # for agent_poison, prepend trigger to queries (matches eval protocol)
                n_eval = min(20, len(sim._victim_queries))
                eval_queries = sim._victim_queries[:n_eval]
                if attack_type == "agent_poison" and sim._last_trigger_string:
                    eval_queries = [
                        f"{sim._last_trigger_string} {q}" for q in eval_queries
                    ]

                # evaluate with local agent
                result = evaluator.evaluate(
                    victim_queries=eval_queries,
                    memory_system=mem,
                    adversarial_passages=poison_texts,
                    attack_type=attack_type,
                    top_k=self.top_k,
                )

                # update attack results with measured asr-a
                for ar in self.results.attack_results:
                    if ar.attack_type == attack_type:
                        ar.asr_a_gpt2 = result.asr_a
                        ar.asr_t_gpt2 = ar.asr_r * result.asr_a

                print(
                    f"  {attack_type}: asr_a_gpt2={result.asr_a:.3f}, "
                    f"asr_r={result.asr_r:.3f}"
                )

        except ImportError as e:
            print(f"  skipped (missing dependency): {e}")
        except Exception as e:
            print(f"  error: {e}")
            import traceback

            traceback.print_exc()

        self.results.timing["gpt2_asr_a_s"] = time.time() - t0
        print(f"  time: {self.results.timing['gpt2_asr_a_s']:.1f}s\n")

    # -----------------------------------------------------------------------
    # stage 3: measured asr-a (gpt-4o-mini)
    # -----------------------------------------------------------------------

    def _run_openai_asr_a(self) -> None:
        """measure asr-a using gpt-4o-mini via openai api."""
        print("--- stage 3: measured asr-a (gpt-4o-mini) ---")
        t0 = time.time()

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("  skipped (no OPENAI_API_KEY)")
            self.results.timing["openai_asr_a_s"] = 0.0
            return

        try:
            from evaluation.agent_eval import OpenAIAgentEvaluator
            from evaluation.retrieval_sim import RetrievalSimulator

            evaluator = OpenAIAgentEvaluator(
                model="gpt-4o-mini",
                api_key=api_key,
                top_k=self.top_k,
            )

            sim = RetrievalSimulator(
                corpus_size=self.corpus_size,
                top_k=self.top_k,
                n_poison_per_attack=5,
                seed=42,
                use_trigger_optimization=True,
            )

            benign_texts = [e["content"] for e in sim._benign_entries]

            for attack_type in ["agent_poison", "minja", "injecmem"]:
                poison_entries = sim._generate_poison_entries(
                    attack_type, sim._victim_queries
                )
                poison_texts = [p["content"] for p in poison_entries]

                # use 20 queries for cost management
                n_eval = min(20, len(sim._victim_queries))
                eval_queries = sim._victim_queries[:n_eval]
                if attack_type == "agent_poison" and sim._last_trigger_string:
                    eval_queries = [
                        f"{sim._last_trigger_string} {q}" for q in eval_queries
                    ]

                result = evaluator.evaluate(
                    attack_type=attack_type,
                    victim_queries=eval_queries,
                    poison_entries=poison_texts,
                    benign_entries=benign_texts,
                    n_poison_injected=len(poison_entries),
                )

                for ar in self.results.attack_results:
                    if ar.attack_type == attack_type:
                        ar.asr_a_gpt4o_mini = result.asr_a
                        ar.asr_t_gpt4o_mini = ar.asr_r * result.asr_a

                print(
                    f"  {attack_type}: asr_a_gpt4o_mini={result.asr_a:.3f}, "
                    f"asr_r={result.asr_r:.3f}"
                )

        except ImportError as e:
            print(f"  skipped (missing dependency): {e}")
        except Exception as e:
            print(f"  error: {e}")
            import traceback

            traceback.print_exc()

        self.results.timing["openai_asr_a_s"] = time.time() - t0
        print(f"  time: {self.results.timing['openai_asr_a_s']:.1f}s\n")

    # -----------------------------------------------------------------------
    # stage 4: defense evaluation at scale
    # -----------------------------------------------------------------------

    def _run_defense_evaluation(self) -> None:
        """evaluate sad defense at scale corpus."""
        from defenses.semantic_anomaly import SemanticAnomalyDetector
        from evaluation.retrieval_sim import RetrievalSimulator

        print("--- stage 4: defense evaluation at scale ---")
        t0 = time.time()

        sim = RetrievalSimulator(
            corpus_size=self.corpus_size,
            top_k=self.top_k,
            n_poison_per_attack=5,
            seed=42,
            use_trigger_optimization=True,
        )

        benign_texts = [e["content"] for e in sim._benign_entries]

        # calibrate sad on benign entries
        sad = SemanticAnomalyDetector(threshold_sigma=2.0, scoring_mode="combined")
        sad.calibrate(benign_texts[:200], sim._victim_queries[:20])

        n_benign_test = min(100, len(benign_texts))

        for attack_type in ["agent_poison", "minja", "injecmem"]:
            poison_entries = sim._generate_poison_entries(
                attack_type, sim._victim_queries
            )
            poison_texts = [p["content"] for p in poison_entries]

            # use evaluate_on_corpus for tpr/fpr/auroc in one pass
            eval_result = sad.evaluate_on_corpus(
                poison_entries=poison_texts,
                benign_entries=benign_texts[:n_benign_test],
            )

            tpr = eval_result["tpr"]
            fpr = eval_result["fpr"]
            auroc = eval_result["auroc"]

            self.results.defense_results.append(
                DefenseResult(
                    defense="memsad",
                    attack_type=attack_type,
                    tpr=tpr,
                    fpr=fpr,
                    auroc=auroc,
                )
            )
            print(
                f"  sad vs {attack_type}: tpr={tpr:.3f}, "
                f"fpr={fpr:.3f}, auroc={auroc:.3f}"
            )

        self.results.timing["defense_eval_s"] = time.time() - t0
        print(f"  time: {self.results.timing['defense_eval_s']:.1f}s\n")

    # -----------------------------------------------------------------------
    # stage 5: multi-encoder evaluation
    # -----------------------------------------------------------------------

    def _run_encoder_evaluation(self) -> None:
        """evaluate across multiple encoders."""
        print("--- stage 5: multi-encoder evaluation ---")
        t0 = time.time()

        try:
            from evaluation.multi_encoder_eval import MultiEncoderEvaluator
            from evaluation.retrieval_sim import RetrievalSimulator

            # build corpus and queries from simulator
            sim = RetrievalSimulator(
                corpus_size=self.corpus_size,
                top_k=self.top_k,
                seed=42,
            )
            benign_texts = [e["content"] for e in sim._benign_entries]
            victim_queries = sim._victim_queries

            # local encoders (skip openai for speed unless explicitly enabled)
            encoder_names = [
                "minilm",
                "mpnet",
                "e5-base",
                "contriever",
            ]
            if not self.skip_openai:
                encoder_names.extend(["openai-small", "openai-large"])

            evaluator = MultiEncoderEvaluator(
                encoders=encoder_names,
                corpus_size=self.corpus_size,
                n_poison=5,
                top_k=self.top_k,
                sad_sigma=2.0,
            )

            result = evaluator.run(
                benign_texts=benign_texts,
                victim_queries=victim_queries,
                attack_types=["agent_poison", "minja", "injecmem"],
            )

            # store results
            self.results.encoder_results = {
                "attack_table": (
                    result.get_attack_table()
                    if hasattr(result, "get_attack_table")
                    else {}
                ),
                "defense_table": (
                    result.get_defense_table()
                    if hasattr(result, "get_defense_table")
                    else {}
                ),
                "raw": (
                    result.to_dict() if hasattr(result, "to_dict") else str(result)
                ),
            }

            print("  encoder evaluation complete")
            for enc_name in encoder_names:
                print(f"    {enc_name}: evaluated")

        except ImportError as e:
            print(f"  skipped (missing dependency): {e}")
        except Exception as e:
            print(f"  error: {e}")
            import traceback

            traceback.print_exc()

        self.results.timing["encoder_eval_s"] = time.time() - t0
        print(f"  time: {self.results.timing['encoder_eval_s']:.1f}s\n")

    # -----------------------------------------------------------------------
    # save results
    # -----------------------------------------------------------------------

    def _save_results(self) -> None:
        """save all results to json and generate latex tables."""
        # save json
        results_path = self.output_dir / "phase28_results.json"
        with open(results_path, "w") as f:
            json.dump(self._to_dict(), f, indent=2, default=str)
        print(f"results saved to {results_path}")

        # generate latex tables
        self._generate_attack_table()
        self._generate_defense_table()

    def _to_dict(self) -> dict[str, Any]:
        """convert results to serializable dict."""
        return {
            "corpus_size": self.results.corpus_size,
            "n_victim_queries": self.results.n_victim_queries,
            "n_benign_queries": self.results.n_benign_queries,
            "n_seeds": self.results.n_seeds,
            "attack_results": [asdict(ar) for ar in self.results.attack_results],
            "defense_results": [asdict(dr) for dr in self.results.defense_results],
            "encoder_results": self.results.encoder_results,
            "timing": self.results.timing,
        }

    def _generate_attack_table(self) -> None:
        """generate latex table for attack results."""
        table_path = self.output_dir / "table1_attack_results_1k.tex"

        attack_names = {
            "agent_poison": r"\agentpoison{} (triggered)",
            "minja": r"\minja{}",
            "injecmem": r"\injecmem{}",
        }

        lines = [
            r"\begin{tabular}{lccccccc}",
            r"  \toprule",
            (
                r"  Attack & $|\mathcal{M}|$ & $\asrr$ & "
                r"$\asra^{\text{mod}}$ & $\asra^{\text{GPT-2}}$ & "
                r"$\asra^{\text{4o-mini}}$ & $\asrt$ & Ben.\ Acc. \\"
            ),
            r"  \midrule",
        ]

        for ar in self.results.attack_results:
            name = attack_names.get(ar.attack_type, ar.attack_type)
            gpt2 = f"${ar.asr_a_gpt2:.2f}$" if ar.asr_a_gpt2 is not None else "---"
            gpt4o = (
                f"${ar.asr_a_gpt4o_mini:.2f}$"
                if ar.asr_a_gpt4o_mini is not None
                else "---"
            )
            asr_t = f"${ar.asr_t_modelled:.2f}$"
            lines.append(
                f"  {name} & ${ar.corpus_size}$ & "
                f"${ar.asr_r:.2f}_{{[{ar.asr_r_ci[0]:.2f},{ar.asr_r_ci[1]:.2f}]}}$ & "
                f"${ar.asr_a_modelled:.2f}$ & {gpt2} & {gpt4o} & "
                f"{asr_t} & ${ar.benign_accuracy:.2f}$ \\\\"
            )

        lines += [
            r"  \bottomrule",
            r"\end{tabular}",
        ]

        with open(table_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"attack table saved to {table_path}")

    def _generate_defense_table(self) -> None:
        """generate latex table for defense results."""
        table_path = self.output_dir / "table2_defense_results_1k.tex"

        lines = [
            r"\begin{tabular}{l ccc ccc c}",
            r"  \toprule",
            (
                r"  & \multicolumn{3}{c}{TPR ($\uparrow$)} & "
                r"\multicolumn{3}{c}{FPR ($\downarrow$)} & AUROC \\"
            ),
            r"  \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-8}",
            r"  Defense & AP & MJ & IM & AP & MJ & IM & (avg) \\",
            r"  \midrule",
        ]

        # group by defense
        defenses: dict[str, dict[str, DefenseResult]] = {}
        for dr in self.results.defense_results:
            if dr.defense not in defenses:
                defenses[dr.defense] = {}
            defenses[dr.defense][dr.attack_type] = dr

        for defense_name, attacks in defenses.items():
            ap = attacks.get("agent_poison")
            mj = attacks.get("minja")
            im = attacks.get("injecmem")

            aurocs = [
                d.auroc for d in [ap, mj, im] if d is not None and d.auroc is not None
            ]
            avg_auroc = f"${sum(aurocs) / len(aurocs):.3f}$" if aurocs else "---"

            tpr_ap = f"${ap.tpr:.2f}$" if ap else "---"
            tpr_mj = f"${mj.tpr:.2f}$" if mj else "---"
            tpr_im = f"${im.tpr:.2f}$" if im else "---"
            fpr_ap = f"${ap.fpr:.2f}$" if ap else "---"
            fpr_mj = f"${mj.fpr:.2f}$" if mj else "---"
            fpr_im = f"${im.fpr:.2f}$" if im else "---"

            display_name = r"\memsad{}" if defense_name == "memsad" else defense_name

            lines.append(
                f"  {display_name} & {tpr_ap} & {tpr_mj} & {tpr_im} & "
                f"{fpr_ap} & {fpr_mj} & {fpr_im} & {avg_auroc} \\\\"
            )

        lines += [
            r"  \bottomrule",
            r"\end{tabular}",
        ]

        with open(table_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"defense table saved to {table_path}")


# ---------------------------------------------------------------------------
# cli entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """run phase 28 experiments from command line."""
    quick = "--quick" in sys.argv
    skip_openai = "--skip-openai" in sys.argv
    skip_gpt2 = "--skip-gpt2" in sys.argv

    if quick:
        runner = Phase28Runner(
            corpus_size=200,
            n_victim_queries=20,
            n_benign_queries=20,
            n_seeds=3,
            skip_openai=skip_openai,
            skip_gpt2=skip_gpt2,
        )
    else:
        runner = Phase28Runner(
            corpus_size=1000,
            n_victim_queries=100,
            n_benign_queries=100,
            n_seeds=5,
            skip_openai=skip_openai,
            skip_gpt2=skip_gpt2,
        )

    runner.run_all()


if __name__ == "__main__":
    main()
