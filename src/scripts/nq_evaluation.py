"""
nq external validation: attack asr-r on a natural-questions-based corpus.

evaluates agentpoison, minja, and injecmem on a corpus combining nq knowledge
entries with synthetic entries to reach |M|=1,000, using nq questions as
victim queries. this validates that attack/defense results generalise beyond
the purely synthetic corpus used in main experiments.

references:
    - kwiatkowski et al. "natural questions: a benchmark for question answering
      research." tacl 2019.

all comments are lowercase.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# add src to path
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))

import numpy as np

from data.nq_subset import NQSubset
from data.synthetic_corpus import SyntheticCorpus
from defenses.semantic_anomaly import SemanticAnomalyDetector
from evaluation.retrieval_sim import RetrievalSimulator


def build_nq_corpus(
    target_size: int = 1000, seed: int = 42
) -> tuple[list[dict], list[str], list[str]]:
    """
    build a mixed corpus: nq knowledge entries + synthetic entries.

    returns:
        (benign_entries, victim_queries, benign_queries)
    """
    rng = np.random.default_rng(seed)

    nq = NQSubset()
    nq_entries = nq.get_corpus_entries()
    nq_victim_qs = [q["query"] for q in nq.get_victim_queries()]

    # fill up to target_size with synthetic entries from non-knowledge categories
    synthetic = SyntheticCorpus(seed=seed)
    synth_entries = synthetic.generate_benign_entries(target_size)
    # filter out knowledge category to keep nq entries as the knowledge base
    non_knowledge = [e for e in synth_entries if e.get("category") != "knowledge"]

    # combine: all nq entries + synthetic non-knowledge entries
    combined = nq_entries + non_knowledge
    # trim or pad to target_size
    if len(combined) > target_size:
        idx = rng.choice(len(combined), target_size, replace=False)
        combined = [combined[i] for i in idx]
    elif len(combined) < target_size:
        extra = synthetic.generate_benign_entries(target_size - len(combined))
        combined = combined + extra

    # benign queries: use nq benign questions (non-victim) from synthetic corpus
    synth_benign_qs = [q["query"] for q in synthetic.get_benign_queries_extended(100)]

    return combined, nq_victim_qs, synth_benign_qs


def evaluate_nq_attacks(
    corpus_size: int = 1000,
    n_seeds: int = 3,
    top_k: int = 5,
) -> dict:
    """
    run agentpoison, minja, injecmem on nq-based corpus.

    returns:
        dict with asr_r per attack (mean ± std across seeds)
    """
    attack_types = ["agent_poison", "minja", "injecmem"]
    results: dict[str, list[float]] = {a: [] for a in attack_types}

    print(f"running nq evaluation: corpus_size={corpus_size}, seeds={n_seeds}")

    for seed in range(n_seeds):
        benign_entries, victim_queries, benign_queries = build_nq_corpus(
            target_size=corpus_size, seed=seed
        )
        print(
            f"  seed {seed}: {len(benign_entries)} benign entries, "
            f"{len(victim_queries)} victim queries"
        )

        for attack_type in attack_types:
            n_poison = {"agent_poison": 5, "minja": 10, "injecmem": 15}[attack_type]
            use_trigger = attack_type == "agent_poison"

            sim = RetrievalSimulator(
                corpus_size=corpus_size,
                top_k=top_k,
                n_poison_per_attack=n_poison,
                seed=seed,
                use_trigger_optimization=use_trigger,
            )
            # override the simulator's benign corpus and queries with nq data
            sim._benign_entries = benign_entries
            sim._victim_queries = victim_queries[:50]  # 50 nq victim queries
            sim._benign_queries = benign_queries[:50]

            result = sim.evaluate_attack(attack_type)
            asr_r = result.asr_r
            results[attack_type].append(asr_r)
            print(f"    {attack_type}: asr_r={asr_r:.3f}")

    summary = {}
    for attack, vals in results.items():
        summary[attack] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals,
        }
    return summary


def evaluate_nq_defenses(
    corpus_size: int = 1000,
    seed: int = 42,
) -> dict:
    """
    run memsad on nq corpus with combined scoring.

    returns:
        dict with tpr/fpr per attack
    """
    benign_entries, victim_queries, _ = build_nq_corpus(
        target_size=corpus_size, seed=seed
    )
    benign_contents = [e["content"] for e in benign_entries[:50]]
    victim_qs_50 = victim_queries[:50]

    results = {}
    for attack_type in ["agent_poison", "minja", "injecmem"]:
        n_poison = {"agent_poison": 5, "minja": 10, "injecmem": 15}[attack_type]
        use_trigger = attack_type == "agent_poison"

        sim = RetrievalSimulator(
            corpus_size=corpus_size,
            top_k=5,
            n_poison_per_attack=n_poison,
            seed=seed,
            use_trigger_optimization=use_trigger,
        )
        sim._benign_entries = benign_entries
        sim._victim_queries = victim_qs_50
        sim._benign_queries = victim_qs_50

        poison_entries = sim._generate_poison_entries(attack_type, victim_qs_50)

        # calibrate memsad on nq benign entries
        detector = SemanticAnomalyDetector(threshold_sigma=2.0, scoring_mode="combined")
        detector.calibrate(benign_contents, victim_qs_50)

        tpr_hits = 0
        for p in poison_entries[:20]:
            score_result = detector.detect(p["content"])
            if score_result.is_anomalous:
                tpr_hits += 1
        tpr = tpr_hits / min(20, len(poison_entries))

        fpr_hits = 0
        for b in benign_entries[:100]:
            score_result = detector.detect(b["content"])
            if score_result.is_anomalous:
                fpr_hits += 1
        fpr = fpr_hits / 100
        results[attack_type] = {"tpr": tpr, "fpr": fpr}
        print(f"  {attack_type}: tpr={tpr:.3f}, fpr={fpr:.3f}")

    return results


if __name__ == "__main__":
    print("=== nq external validation ===")
    print("\n--- attack evaluation ---")
    attack_results = evaluate_nq_attacks(corpus_size=1000, n_seeds=3)
    for attack, stats in attack_results.items():
        print(f"  {attack}: asr_r = {stats['mean']:.3f} ± {stats['std']:.3f}")

    print("\n--- memsad defense evaluation ---")
    defense_results = evaluate_nq_defenses(corpus_size=1000)

    print("\n=== summary ===")
    print(
        json.dumps({"attacks": attack_results, "defenses": defense_results}, indent=2)
    )
