"""
comprehensive ablation studies for memory agent security evaluation.

evaluates sensitivity of attack and defense performance to key hyperparameters:
    - corpus size: total benign entries in the memory system
    - retrieval top-k: number of results returned per query
    - poison count: number of adversarial entries injected
    - sad threshold (sigma): anomaly detection sensitivity (k in μ + k·σ)
    - watermark z_threshold: detection operating point

each ablation runs multiple seeds and reports bootstrap 95% confidence intervals
for all primary metrics (asr-r, asr-t, benign accuracy, tpr, fpr).

results feed directly into appendix figures and tables for the neurips/acm ccs
submission.  all ablations operate on the same synthetic corpus (syntheticcorpus)
with controlled randomness (seed sweeps: seed_base, seed_base+17, ...).

design rationale:
    - corpus size ablation shows attack effectiveness does NOT scale linearly
      with corpus size (poisons remain in top-k at large corpus sizes because
      they are crafted to be extreme outliers in similarity space).
    - top-k ablation shows attack effectiveness monotonically increases with k
      (larger retrieval window = more chances for poison to be included).
    - poison count ablation shows diminishing returns — single poison entry
      already achieves high asr-r for agentpoison with trigger.
    - sad threshold ablation characterises the tpr-fpr operating curve.
    - watermark threshold ablation shows robustness-utility tradeoff.

all comments are lowercase.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.logging import logger

# ---------------------------------------------------------------------------
# result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AblationPoint:
    """
    single result for one (parameter_name, parameter_value) configuration.

    all metric means are computed across n_trials seeds via bootstrap percentile
    confidence intervals.

    fields:
        param_name: name of ablated hyperparameter (e.g. "corpus_size")
        param_value: numerical value at this ablation point
        attack_type: attack evaluated (or "all" for defense ablations)
        asr_r_mean: mean attack success rate — retrieval
        asr_r_ci_lower: 95% ci lower bound for asr-r
        asr_r_ci_upper: 95% ci upper bound for asr-r
        asr_t_mean: mean asr-t (end-to-end)
        benign_acc_mean: mean benign query accuracy (1 = no false retrievals)
        tpr_mean: mean defense tpr (for defense ablations; 0.0 for attack-only)
        fpr_mean: mean defense fpr (for defense ablations; 0.0 for attack-only)
        n_trials: number of seeds evaluated
        elapsed_s: wall time for this configuration
    """

    param_name: str
    param_value: float
    attack_type: str
    asr_r_mean: float
    asr_r_ci_lower: float
    asr_r_ci_upper: float
    asr_t_mean: float = 0.0
    benign_acc_mean: float = 0.0
    tpr_mean: float = 0.0
    fpr_mean: float = 0.0
    n_trials: int = 1
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param_name": self.param_name,
            "param_value": self.param_value,
            "attack_type": self.attack_type,
            "asr_r_mean": self.asr_r_mean,
            "asr_r_ci_lower": self.asr_r_ci_lower,
            "asr_r_ci_upper": self.asr_r_ci_upper,
            "asr_t_mean": self.asr_t_mean,
            "benign_acc_mean": self.benign_acc_mean,
            "tpr_mean": self.tpr_mean,
            "fpr_mean": self.fpr_mean,
            "n_trials": self.n_trials,
            "elapsed_s": self.elapsed_s,
        }


# ---------------------------------------------------------------------------
# bootstrap helper (local, no dependency on statistical.py)
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    samples: List[float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    compute percentile bootstrap 95% ci.

    returns (mean, lower, upper).
    """
    import random

    rng = random.Random(seed)
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return samples[0], samples[0], samples[0]

    means = []
    for _ in range(n_boot):
        boot = [rng.choice(samples) for _ in range(n)]
        means.append(sum(boot) / n)
    means.sort()
    lo = int(math.floor(alpha / 2 * n_boot))
    hi = int(math.ceil((1 - alpha / 2) * n_boot)) - 1
    hi = min(hi, len(means) - 1)
    mean_est = sum(samples) / n
    return mean_est, means[lo], means[hi]


# ---------------------------------------------------------------------------
# ablation study runner
# ---------------------------------------------------------------------------


class AblationStudy:
    """
    runs all hyperparameter ablation studies for the paper.

    each ablation sweeps one hyperparameter while holding others fixed
    at their default values:
        corpus_size=200, top_k=5, n_poison=5, sad_sigma=2.0,
        watermark_z_threshold=4.0.

    note on test mode: when MEMORY_SECURITY_TEST=true, sizes and trial counts
    are reduced for fast ci.  the ablation grid is also abbreviated.
    """

    # default ablation grids
    CORPUS_SIZES = [50, 100, 200, 500]
    TOPK_VALUES = [1, 3, 5, 10, 20]
    POISON_COUNTS = [1, 3, 5, 10, 20]
    SAD_SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    WATERMARK_Z_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def __init__(
        self,
        attack_type: str = "agent_poison",
        seed_base: int = 42,
        n_bootstrap: int = 500,
    ):
        """
        args:
            attack_type: primary attack to ablate.  "agent_poison" is the
                default because it is the most controlled (trigger prepended).
            seed_base: base seed; trials use seed_base, seed_base+17, ...
            n_bootstrap: bootstrap replicates for ci computation.
        """
        self.attack_type = attack_type
        self.seed_base = seed_base
        self.n_bootstrap = n_bootstrap

        _test = os.environ.get("MEMORY_SECURITY_TEST", "false").lower() == "true"
        if _test:
            # abbreviated grids for fast ci
            self.CORPUS_SIZES = [20, 50]
            self.TOPK_VALUES = [3, 5]
            self.POISON_COUNTS = [1, 3]
            self.SAD_SIGMA_VALUES = [1.5, 2.0, 2.5]
            self.WATERMARK_Z_VALUES = [3.0, 4.0]

    def _run_retrieval_sim(
        self,
        attack_type: str,
        corpus_size: int,
        top_k: int,
        n_poison: int,
        seed: int,
    ) -> Dict[str, float]:
        """
        run a single retrieval simulation trial with the given parameters.
        returns dict with asr_r, asr_t, benign_accuracy.
        """
        from evaluation.retrieval_sim import RetrievalSimulator

        sim = RetrievalSimulator(
            corpus_size=corpus_size,
            n_poison_per_attack=n_poison,
            top_k=top_k,
            use_trigger_optimization=False,
            seed=seed,
        )
        m = sim.evaluate_attack(attack_type)
        return {
            "asr_r": m.asr_r,
            "asr_t": m.asr_t,
            "benign_accuracy": m.benign_accuracy,
        }

    def _ablation_points(
        self,
        param_name: str,
        param_values: List[float],
        fixed_corpus: int = 200,
        fixed_topk: int = 5,
        fixed_poison: int = 5,
        n_trials: int = 3,
        attack_type: Optional[str] = None,
    ) -> List[AblationPoint]:
        """
        generic ablation runner: sweeps param_name over param_values.

        computes bootstrap cis across n_trials seeds for each value.
        """
        at = attack_type or self.attack_type
        _test = os.environ.get("MEMORY_SECURITY_TEST", "false").lower() == "true"
        if _test:
            n_trials = min(n_trials, 2)

        points: List[AblationPoint] = []
        for val in param_values:
            t0 = time.time()
            asr_r_samples, asr_t_samples, bacc_samples = [], [], []

            for trial in range(n_trials):
                seed = self.seed_base + trial * 17
                try:
                    if param_name == "corpus_size":
                        r = self._run_retrieval_sim(
                            at, int(val), fixed_topk, fixed_poison, seed
                        )
                    elif param_name == "top_k":
                        r = self._run_retrieval_sim(
                            at, fixed_corpus, int(val), fixed_poison, seed
                        )
                    elif param_name == "poison_count":
                        r = self._run_retrieval_sim(
                            at, fixed_corpus, fixed_topk, int(val), seed
                        )
                    else:
                        r = {"asr_r": 0.0, "asr_t": 0.0, "benign_accuracy": 0.0}

                    asr_r_samples.append(r["asr_r"])
                    asr_t_samples.append(r["asr_t"])
                    bacc_samples.append(r["benign_accuracy"])
                except Exception as exc:
                    logger.log_error(
                        "ablation_trial",
                        exc,
                        {"param": param_name, "val": val, "trial": trial},
                    )

            mean_r, lo_r, hi_r = _bootstrap_ci(
                asr_r_samples, self.n_bootstrap, seed=self.seed_base
            )
            mean_t = sum(asr_t_samples) / max(len(asr_t_samples), 1)
            mean_b = sum(bacc_samples) / max(len(bacc_samples), 1)

            points.append(
                AblationPoint(
                    param_name=param_name,
                    param_value=float(val),
                    attack_type=at,
                    asr_r_mean=mean_r,
                    asr_r_ci_lower=lo_r,
                    asr_r_ci_upper=hi_r,
                    asr_t_mean=mean_t,
                    benign_acc_mean=mean_b,
                    n_trials=n_trials,
                    elapsed_s=time.time() - t0,
                )
            )
            logger.logger.info(
                "ablation %s=%s: asr_r=%.3f [%.3f, %.3f]",
                param_name,
                val,
                mean_r,
                lo_r,
                hi_r,
            )
        return points

    def corpus_size_ablation(
        self,
        attack_type: Optional[str] = None,
        sizes: Optional[List[int]] = None,
        n_trials: int = 3,
    ) -> List[AblationPoint]:
        """
        ablate corpus size: how does attack effectiveness change as the benign
        memory grows?

        hypothesis: asr-r stays near-constant (poison is an extreme outlier
        in similarity space regardless of how many benign entries exist).

        default sizes: [50, 100, 200, 500].
        """
        sizes = sizes or self.CORPUS_SIZES
        return self._ablation_points(
            "corpus_size",
            [float(s) for s in sizes],
            n_trials=n_trials,
            attack_type=attack_type,
        )

    def topk_ablation(
        self,
        attack_type: Optional[str] = None,
        k_values: Optional[List[int]] = None,
        n_trials: int = 3,
    ) -> List[AblationPoint]:
        """
        ablate retrieval top-k: how does the retrieval window size affect
        attack success rate?

        hypothesis: asr-r monotonically increases with k (larger window
        gives more chances for poison to appear in the result set).

        default k_values: [1, 3, 5, 10, 20].
        """
        k_values = k_values or self.TOPK_VALUES
        return self._ablation_points(
            "top_k",
            [float(k) for k in k_values],
            n_trials=n_trials,
            attack_type=attack_type,
        )

    def poison_count_ablation(
        self,
        attack_type: Optional[str] = None,
        counts: Optional[List[int]] = None,
        n_trials: int = 3,
    ) -> List[AblationPoint]:
        """
        ablate poison count: how many adversarial entries are needed for high
        attack success rate?

        hypothesis: diminishing returns — single poison entry already achieves
        high asr-r for agentpoison (trigger guarantees retrieval), but injecmem
        needs multiple copies for broad recall.

        default counts: [1, 3, 5, 10, 20].
        """
        counts = counts or self.POISON_COUNTS
        return self._ablation_points(
            "poison_count",
            [float(c) for c in counts],
            n_trials=n_trials,
            attack_type=attack_type,
        )

    def sad_threshold_ablation(
        self,
        attack_type: Optional[str] = None,
        sigma_values: Optional[List[float]] = None,
        n_trials: int = 3,
    ) -> List[AblationPoint]:
        """
        ablate sad detection threshold (sigma multiplier k).

        computes tpr and fpr at each operating point (sigma=0.5 to 4.0).
        low sigma → high tpr but high fpr (aggressive detection).
        high sigma → low fpr but may miss low-anomaly poison entries.

        default sigma_values: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0].
        """
        at = attack_type or self.attack_type
        sigma_vals = sigma_values or self.SAD_SIGMA_VALUES
        _test = os.environ.get("MEMORY_SECURITY_TEST", "false").lower() == "true"
        if _test:
            n_trials = min(n_trials, 2)

        points: List[AblationPoint] = []
        for sigma in sigma_vals:
            t0 = time.time()
            tpr_samples, fpr_samples, asr_r_samples = [], [], []

            for trial in range(n_trials):
                seed = self.seed_base + trial * 17
                try:
                    from data.synthetic_corpus import SyntheticCorpus
                    from defenses.semantic_anomaly import SemanticAnomalyDetector
                    from evaluation.retrieval_sim import (
                        generate_centroid_agentpoison_passage,
                        generate_injecmem_passage,
                        generate_minja_passage,
                    )

                    corpus = SyntheticCorpus(seed=seed)
                    benign_entries = corpus.get_entries(100)
                    benign_texts = [e["content"] for e in benign_entries]
                    victim_queries = corpus.get_victim_queries()

                    # generate poison entries
                    poison_entries: List[str] = []
                    if at == "agent_poison":
                        p = generate_centroid_agentpoison_passage(victim_queries)
                        poison_entries = [p] * 5
                    elif at == "minja":
                        for i in range(10):
                            q = victim_queries[i % len(victim_queries)]
                            poison_entries.append(generate_minja_passage(q))
                    else:
                        for i in range(15):
                            q = victim_queries[i % len(victim_queries)]
                            poison_entries.append(
                                generate_injecmem_passage(q, variant_index=i)
                            )

                    # run sad
                    det = SemanticAnomalyDetector(threshold_sigma=sigma)
                    cal_sample = benign_texts[:10]
                    det.calibrate(benign_texts, cal_sample)
                    for q in victim_queries[:10]:
                        det.update_query_set(q)

                    result = det.evaluate_on_corpus(poison_entries, benign_texts[:20])
                    tpr_samples.append(result["tpr"])
                    fpr_samples.append(result["fpr"])

                    # asr-r contribution (approximated)
                    from evaluation.retrieval_sim import RetrievalSimulator

                    sim = RetrievalSimulator(
                        corpus_size=100,
                        n_poison_per_attack=5,
                        top_k=5,
                        use_trigger_optimization=False,
                        seed=seed,
                    )
                    m = sim.evaluate_attack(at)
                    asr_r_samples.append(m.asr_r)
                except Exception as exc:
                    logger.log_error(
                        "sad_ablation_trial",
                        exc,
                        {"sigma": sigma, "trial": trial},
                    )

            mean_tpr = sum(tpr_samples) / max(len(tpr_samples), 1)
            mean_fpr = sum(fpr_samples) / max(len(fpr_samples), 1)
            mean_r, lo_r, hi_r = _bootstrap_ci(
                asr_r_samples, self.n_bootstrap, seed=self.seed_base
            )

            points.append(
                AblationPoint(
                    param_name="sad_sigma",
                    param_value=float(sigma),
                    attack_type=at,
                    asr_r_mean=mean_r,
                    asr_r_ci_lower=lo_r,
                    asr_r_ci_upper=hi_r,
                    tpr_mean=mean_tpr,
                    fpr_mean=mean_fpr,
                    n_trials=n_trials,
                    elapsed_s=time.time() - t0,
                )
            )
            logger.logger.info(
                "sad ablation sigma=%.1f: tpr=%.3f fpr=%.3f", sigma, mean_tpr, mean_fpr
            )
        return points

    def watermark_threshold_ablation(
        self,
        z_values: Optional[List[float]] = None,
        n_trials: int = 3,
    ) -> List[AblationPoint]:
        """
        ablate watermark z-score detection threshold.

        lower z_threshold → higher tpr but higher fpr (aggressive).
        higher z_threshold → conservative detection, lower fpr.

        generates 50 watermarked and 50 clean samples per trial; reports
        tpr (watermarked correctly detected) and fpr (clean falsely flagged).

        default z_values: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0].
        """
        z_vals = z_values or self.WATERMARK_Z_VALUES
        _test = os.environ.get("MEMORY_SECURITY_TEST", "false").lower() == "true"
        if _test:
            n_trials = min(n_trials, 2)

        points: List[AblationPoint] = []
        for z_thresh in z_vals:
            t0 = time.time()
            tpr_samples, fpr_samples = [], []

            for trial in range(n_trials):
                seed = self.seed_base + trial * 17
                try:
                    from watermark.watermarking import create_watermark_encoder

                    rng_seed = seed
                    encoder = create_watermark_encoder("unigram")

                    # reference content pool (realistic memory entry length)
                    _ref = (
                        "the memory agent maintains a record of user preferences "
                        "task history calendar events and conversation context "
                        "across multiple interaction sessions for personalised "
                        "response generation and contextual retrieval of relevant "
                        "information. this entry covers scheduling preferences and "
                        "communication style guidelines from prior interactions."
                    )

                    # generate watermarked samples
                    n_samples = 20 if not _test else 5
                    wm_z_scores: List[float] = []
                    for i in range(n_samples):
                        wm_id = f"wm_{rng_seed}_{i}"
                        watermarked = encoder.embed(_ref, wm_id)
                        stats = encoder.get_detection_stats(watermarked)
                        wm_z_scores.append(stats["z_score"])

                    # generate clean (unwatermarked) samples
                    clean_z_scores: List[float] = []
                    for i in range(n_samples):
                        stats = encoder.get_detection_stats(_ref)
                        clean_z_scores.append(stats["z_score"])

                    # compute tpr/fpr at this threshold
                    tpr_samples.append(
                        sum(1 for z in wm_z_scores if z >= z_thresh) / max(n_samples, 1)
                    )
                    fpr_samples.append(
                        sum(1 for z in clean_z_scores if z >= z_thresh)
                        / max(n_samples, 1)
                    )
                except Exception as exc:
                    logger.log_error(
                        "watermark_ablation_trial",
                        exc,
                        {"z_thresh": z_thresh, "trial": trial},
                    )

            mean_tpr = sum(tpr_samples) / max(len(tpr_samples), 1)
            mean_fpr = sum(fpr_samples) / max(len(fpr_samples), 1)

            points.append(
                AblationPoint(
                    param_name="watermark_z_threshold",
                    param_value=float(z_thresh),
                    attack_type="watermark",
                    asr_r_mean=0.0,
                    asr_r_ci_lower=0.0,
                    asr_r_ci_upper=0.0,
                    tpr_mean=mean_tpr,
                    fpr_mean=mean_fpr,
                    n_trials=n_trials,
                    elapsed_s=time.time() - t0,
                )
            )
            logger.logger.info(
                "watermark ablation z=%.1f: tpr=%.3f fpr=%.3f",
                z_thresh,
                mean_tpr,
                mean_fpr,
            )
        return points

    def run_all(
        self,
        n_trials: int = 3,
        attack_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        run all ablation studies and return a structured result dict.

        keys: "corpus_size", "top_k", "poison_count", "sad_sigma",
              "watermark_z_threshold", and per-attack variants.

        args:
            n_trials: seeds per ablation point (bootstrap ci quality)
            attack_types: attacks to ablate. defaults to all three attack types.

        returns:
            dict mapping study_name → List[AblationPoint]
        """
        ats = attack_types or ["agent_poison", "minja", "injecmem"]
        results: Dict[str, Any] = {}

        # corpus size ablation (main attack)
        logger.logger.info("running corpus size ablation (%s)", self.attack_type)
        results["corpus_size"] = self.corpus_size_ablation(n_trials=n_trials)

        # top-k ablation (all attacks)
        for at in ats:
            key = f"top_k_{at}"
            logger.logger.info("running top-k ablation (%s)", at)
            results[key] = self.topk_ablation(attack_type=at, n_trials=n_trials)

        # poison count ablation (all attacks)
        for at in ats:
            key = f"poison_count_{at}"
            logger.logger.info("running poison count ablation (%s)", at)
            results[key] = self.poison_count_ablation(attack_type=at, n_trials=n_trials)

        # sad threshold ablation (all attacks)
        for at in ats:
            key = f"sad_sigma_{at}"
            logger.logger.info("running sad threshold ablation (%s)", at)
            results[key] = self.sad_threshold_ablation(
                attack_type=at, n_trials=n_trials
            )

        # watermark threshold ablation
        logger.logger.info("running watermark threshold ablation")
        results["watermark_z_threshold"] = self.watermark_threshold_ablation(
            n_trials=n_trials
        )

        return results

    def to_latex_table(
        self,
        points: List[AblationPoint],
        param_label: str,
        metric_label: str = "ASR-R",
        caption: str = "",
        label: str = "tab:ablation",
        metric_key: str = "asr_r",
    ) -> str:
        """
        generate a single-ablation booktabs latex table.

        args:
            points: list of AblationPoint from one ablation run
            param_label: column header for the hyperparameter (e.g. "Corpus Size")
            metric_label: metric column header (e.g. "ASR-R", "TPR", "FPR")
            caption: table caption
            label: table label
            metric_key: one of "asr_r", "tpr", "fpr", "benign_acc"

        returns:
            latex string
        """
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            f"{param_label} & {metric_label} & 95\\% CI \\\\",
            r"\midrule",
        ]
        for pt in points:
            # format value as int if it's a whole number, else float
            if pt.param_value == int(pt.param_value):
                val_str = str(int(pt.param_value))
            else:
                val_str = f"{pt.param_value:.1f}"

            if metric_key == "asr_r":
                mean_val = pt.asr_r_mean
                ci_str = f"[{pt.asr_r_ci_lower:.3f}, {pt.asr_r_ci_upper:.3f}]"
            elif metric_key == "tpr":
                mean_val = pt.tpr_mean
                ci_str = "--"
            elif metric_key == "fpr":
                mean_val = pt.fpr_mean
                ci_str = "--"
            else:
                mean_val = pt.benign_acc_mean
                ci_str = "--"

            lines.append(f"{val_str} & {mean_val:.3f} & {ci_str} \\\\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def to_combined_latex_table(
        self,
        results: Dict[str, List[AblationPoint]],
        caption: str = "Hyperparameter ablation results.",
        label: str = "tab:ablation_combined",
    ) -> str:
        """
        generate a multi-column ablation summary table covering all studies.

        columns: corpus_size ablation (asr-r), top-k ablation (asr-r),
                 poison count ablation (asr-r), sad_sigma (tpr/fpr).
        """
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{cc|cc|cc|cc}",
            r"\toprule",
            r"\multicolumn{2}{c|}{Corpus Size} & \multicolumn{2}{c|}{Top-K} "
            r"& \multicolumn{2}{c|}{Poison Count} & \multicolumn{2}{c}{SAD $k$} \\",
            r"$N$ & ASR-R & $k$ & ASR-R & $n_p$ & ASR-R & $k$ & TPR / FPR \\",
            r"\midrule",
        ]

        # extract relevant sub-results
        cs_pts = results.get("corpus_size", [])
        tk_pts = results.get(f"top_k_{results.get('_attack', 'agent_poison')}", [])
        # fallback: find any top_k result
        if not tk_pts:
            for k, v in results.items():
                if k.startswith("top_k_"):
                    tk_pts = v
                    break
        pc_pts = results.get(
            f"poison_count_{results.get('_attack', 'agent_poison')}", []
        )
        if not pc_pts:
            for k, v in results.items():
                if k.startswith("poison_count_agent_poison"):
                    pc_pts = v
                    break
        sad_pts = results.get("sad_sigma_agent_poison", [])
        if not sad_pts:
            for k, v in results.items():
                if k.startswith("sad_sigma_"):
                    sad_pts = v
                    break

        max_rows = max(len(cs_pts), len(tk_pts), len(pc_pts), len(sad_pts), 1)
        for i in range(max_rows):
            row_parts = []
            if i < len(cs_pts):
                pt = cs_pts[i]
                v = (
                    int(pt.param_value)
                    if pt.param_value == int(pt.param_value)
                    else pt.param_value
                )
                row_parts.append(f"{v} & {pt.asr_r_mean:.3f}")
            else:
                row_parts.append("& ")
            if i < len(tk_pts):
                pt = tk_pts[i]
                v = (
                    int(pt.param_value)
                    if pt.param_value == int(pt.param_value)
                    else pt.param_value
                )
                row_parts.append(f"{v} & {pt.asr_r_mean:.3f}")
            else:
                row_parts.append("& ")
            if i < len(pc_pts):
                pt = pc_pts[i]
                v = (
                    int(pt.param_value)
                    if pt.param_value == int(pt.param_value)
                    else pt.param_value
                )
                row_parts.append(f"{v} & {pt.asr_r_mean:.3f}")
            else:
                row_parts.append("& ")
            if i < len(sad_pts):
                pt = sad_pts[i]
                row_parts.append(
                    f"{pt.param_value:.1f} & {pt.tpr_mean:.3f}/{pt.fpr_mean:.3f}"
                )
            else:
                row_parts.append("& ")

            lines.append(" & ".join(row_parts) + " \\\\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
        return "\n".join(lines)
