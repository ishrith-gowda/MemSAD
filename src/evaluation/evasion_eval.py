"""
watermark evasion evaluation for the unigram provenance defense.

this module implements three classes of evasion attacks against the unigram
watermark (zhao et al., iclr 2024, arXiv:2306.17439) and measures their
empirical effect on detection performance (tpr, fpr, z-score distributions).

evasion classes implemented:

  1. paraphrasing attack — rewrites watermarked content using synonym
     substitution with semantic similarity preservation.  since the
     paraphraser samples from its own distribution, green token proportion
     drifts back toward the baseline gamma, reducing the z-score.

  2. copy-paste dilution attack — appends non-watermarked content to
     watermarked text, mathematically reducing z-score proportionally.
     the dilution ratio required to defeat detection follows:
         n_dilute > n_wm * ((z_wm / z_threshold)^2 - 1)

  3. adaptive token substitution attack — white-box attack with knowledge
     of the green/red partition.  iteratively replaces green-list tokens
     with red-list synonyms, minimising z-score while preserving meaning.

references:
    zhao et al. provable robust watermarking for ai-generated text.
    iclr 2024. arXiv:2306.17439.

    sadasivan et al. can ai-generated text be reliably detected?
    arXiv:2303.11156. (theoretical impossibility results)

    krishna et al. paraphrasing evades detectors of ai-generated text.
    arXiv:2303.13408.

    piet et al. mark my words: analyzing and evaluating language model
    watermarks. arXiv:2312.00273.

all comments are lowercase.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from utils.logging import logger

# ---------------------------------------------------------------------------
# evasion result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvasionResult:
    """
    result of a single evasion attack evaluation.

    tracks the watermark detection performance before and after the evasion
    attempt, providing the metrics needed for roc and z-score ablation plots.
    """

    attack_type: str  # "paraphrase", "copy_paste", "adaptive_substitution"
    n_samples: int

    # z-score statistics before evasion
    z_scores_before: list[float] = field(default_factory=list)
    # z-score statistics after evasion
    z_scores_after: list[float] = field(default_factory=list)

    # detection metrics before evasion
    tpr_before: float = 0.0
    fpr_before: float = 0.0

    # detection metrics after evasion
    tpr_after: float = 0.0
    fpr_after: float = 0.0

    # semantic preservation metrics
    mean_semantic_similarity: float = 0.0
    # fraction of samples where evasion succeeded (z_after < z_threshold)
    evasion_success_rate: float = 0.0

    # per-intensity results (for ablation plots)
    intensity_results: list[dict[str, Any]] = field(default_factory=list)
    execution_time_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        """compact summary for logging and tables."""
        return {
            "attack_type": self.attack_type,
            "tpr_before": self.tpr_before,
            "tpr_after": self.tpr_after,
            "tpr_delta": self.tpr_after - self.tpr_before,
            "evasion_success_rate": self.evasion_success_rate,
            "mean_z_before": (
                statistics.mean(self.z_scores_before) if self.z_scores_before else 0.0
            ),
            "mean_z_after": (
                statistics.mean(self.z_scores_after) if self.z_scores_after else 0.0
            ),
            "mean_semantic_similarity": self.mean_semantic_similarity,
        }


# ---------------------------------------------------------------------------
# synonym dictionary for paraphrase + adaptive attacks
# ---------------------------------------------------------------------------

# domain-relevant synonym mappings: preserves semantics while swapping tokens.
# used for both paraphrasing evasion (random selection) and adaptive
# substitution (pick the synonym that lands in the red list).
_SYNONYMS: dict[str, list[str]] = {
    # scheduling vocabulary
    "schedule": ["plan", "agenda", "timetable", "roster", "calendar"],
    "meeting": ["session", "gathering", "conference", "call", "sync"],
    "task": ["item", "action", "assignment", "work", "objective"],
    "deadline": ["due date", "target date", "cutoff", "timeline", "limit"],
    "reminder": ["alert", "notice", "notification", "prompt", "flag"],
    "priority": ["importance", "urgency", "ranking", "order", "precedence"],
    "pending": ["outstanding", "open", "unresolved", "waiting", "queued"],
    "completed": ["finished", "done", "resolved", "closed", "achieved"],
    # preference vocabulary
    "preference": ["choice", "option", "setting", "selection", "default"],
    "preferred": ["chosen", "selected", "desired", "recommended", "default"],
    "enabled": ["active", "on", "turned on", "activated", "running"],
    "configured": ["set up", "established", "defined", "arranged", "specified"],
    # authority vocabulary (adversarial framing words)
    "confirmed": ["verified", "acknowledged", "validated", "approved", "accepted"],
    "verified": ["confirmed", "checked", "validated", "authenticated", "proven"],
    "authorized": ["permitted", "allowed", "approved", "certified", "sanctioned"],
    "approved": ["confirmed", "accepted", "endorsed", "validated", "cleared"],
    "required": ["necessary", "mandatory", "needed", "essential", "obligatory"],
    "protocol": ["procedure", "process", "method", "approach", "policy"],
    "instruction": ["direction", "guideline", "order", "directive", "command"],
    # memory vocabulary
    "memory": ["record", "history", "entry", "note", "log"],
    "stored": ["saved", "recorded", "kept", "archived", "cached"],
    "retrieved": ["fetched", "loaded", "accessed", "obtained", "found"],
    "entry": ["record", "item", "note", "document", "data"],
    "history": ["record", "log", "past", "archive", "background"],
    # action vocabulary
    "execute": ["run", "perform", "carry out", "apply", "implement"],
    "apply": ["use", "implement", "employ", "put", "execute"],
    "activate": ["enable", "trigger", "start", "initiate", "turn on"],
    "override": ["bypass", "supersede", "replace", "overwrite", "cancel"],
    "update": ["modify", "change", "revise", "edit", "alter"],
    # infrastructure vocabulary
    "credentials": ["tokens", "keys", "secrets", "passwords", "access codes"],
    "authentication": ["login", "verification", "auth", "sign-in", "identity"],
    "access": ["permission", "entry", "right", "privilege", "clearance"],
    "service": ["system", "application", "component", "module", "endpoint"],
    "configuration": ["setup", "settings", "parameters", "options", "config"],
    # general vocabulary
    "information": ["data", "content", "details", "facts", "knowledge"],
    "system": ["platform", "application", "service", "framework", "environment"],
    "response": ["reply", "answer", "output", "result", "feedback"],
    "current": ["present", "existing", "active", "latest", "recent"],
    "standard": ["default", "typical", "normal", "regular", "common"],
}


def _paraphrase_text(
    text: str,
    substitution_rate: float,
    rng: random.Random,
) -> tuple[str, float]:
    """
    paraphrase text by random synonym substitution.

    iterates over tokens and randomly replaces them with synonyms from
    _SYNONYMS at the given substitution_rate.  tracks the substitution
    fraction as a proxy for semantic drift.

    args:
        text: input text to paraphrase
        substitution_rate: fraction of eligible tokens to substitute [0, 1]
        rng: random generator for reproducibility

    returns:
        (paraphrased_text, actual_substitution_rate)
    """
    tokens = text.split()
    modified = tokens.copy()
    n_substituted = 0
    n_eligible = 0

    for i, token in enumerate(tokens):
        clean = token.lower().strip(".,;:!?\"'()")
        if clean in _SYNONYMS:
            n_eligible += 1
            if rng.random() < substitution_rate:
                synonym = rng.choice(_SYNONYMS[clean])
                # preserve capitalisation of first letter
                if token[0].isupper():
                    synonym = synonym[0].upper() + synonym[1:]
                modified[i] = synonym
                n_substituted += 1

    actual_rate = n_substituted / max(n_eligible, 1)
    return " ".join(modified), actual_rate


def _adaptive_substitution(
    text: str,
    green_set: set,
    z_threshold: float,
    gamma: float,
    max_substitutions: int | None = None,
    rng: random.Random | None = None,
) -> tuple[str, int]:
    """
    white-box adaptive substitution: replace green-list tokens with red-list
    synonyms to drive z-score below detection threshold.

    algorithm:
      1. identify all green-list tokens in the text
      2. for each green-list token (in random order), find the synonym with
         the highest semantic similarity that falls in the red list
      3. substitute until z-score < z_threshold or no more options

    this approximates the optimal attack from sadasivan et al. (2023) under
    the constraint of synonym-restricted substitutions.

    args:
        text: watermarked text to attack
        green_set: the green-list character set (from UnigramWatermarkEncoder)
        z_threshold: detection threshold to defeat
        gamma: green-list proportion (default 0.25)
        max_substitutions: maximum number of substitutions (optional cap)
        rng: random generator

    returns:
        (evaded_text, n_substitutions_made)
    """
    if rng is None:
        rng = random.Random(42)

    tokens = text.split()
    modified = tokens.copy()
    n_substituted = 0

    # identify green tokens (tokens where the majority of alnum chars are green)
    def _token_is_green(t: str) -> bool:
        alnum = [c for c in t if c.isalnum()]
        if not alnum:
            return False
        green_count = sum(1 for c in alnum if c in green_set)
        return green_count / len(alnum) >= 0.5

    def _token_is_red(t: str) -> bool:
        return not _token_is_green(t)

    def _current_z(modified_tokens: list[str]) -> float:
        """compute z-score for current token sequence."""
        import math

        all_chars = " ".join(modified_tokens)
        alnum = [c for c in all_chars if c.isalnum()]
        n = len(alnum)
        if n < 10:
            return 0.0
        green = sum(1 for c in alnum if c in green_set)
        expected = gamma * n
        variance = gamma * (1 - gamma) * n
        std = math.sqrt(variance) if variance > 0 else 1.0
        return (green - expected) / std

    # identify positions with green tokens that have red synonyms
    green_positions = [
        i
        for i, t in enumerate(tokens)
        if _token_is_green(t.lower().strip(".,;:!?\"'()"))
        and t.lower().strip(".,;:!?\"'()") in _SYNONYMS
    ]
    rng.shuffle(green_positions)

    for pos in green_positions:
        if max_substitutions is not None and n_substituted >= max_substitutions:
            break

        # check if we've already defeated the watermark
        current_z = _current_z(modified)
        if current_z < z_threshold:
            break

        token = tokens[pos]
        clean = token.lower().strip(".,;:!?\"'()")
        synonyms = _SYNONYMS.get(clean, [])

        # find synonym that lands in red list (greedy: pick first red synonym)
        for syn in synonyms:
            if _token_is_red(syn):
                # prefer red-list synonyms that preserve semantics
                if token[0].isupper():
                    syn = syn[0].upper() + syn[1:]
                modified[pos] = syn
                n_substituted += 1
                break

    return " ".join(modified), n_substituted


# ---------------------------------------------------------------------------
# WatermarkEvasionEvaluator
# ---------------------------------------------------------------------------


class WatermarkEvasionEvaluator:
    """
    comprehensive evaluation of evasion strategies against the unigram watermark.

    evaluates three attack classes:
      - paraphrasing: synonym substitution at varying intensity rates
      - copy_paste: dilution with non-watermarked text at varying ratios
      - adaptive: white-box green→red token substitution

    for each attack, reports:
      - z-score distributions before/after (for ablation plots)
      - tpr/fpr before/after
      - evasion success rate vs. intensity
      - semantic preservation (approximated by substitution fraction)

    usage:
        evaluator = WatermarkEvasionEvaluator(encoder)
        results = evaluator.evaluate_all(watermarked_samples, clean_samples)
    """

    def __init__(
        self,
        encoder,
        n_samples: int = 50,
        seed: int = 42,
    ) -> None:
        """
        initialise evasion evaluator.

        args:
            encoder: UnigramWatermarkEncoder instance (provides z-score and
                     green-set access)
            n_samples: number of samples for each evaluation
            seed: random seed
        """
        self.encoder = encoder
        self.n_samples = n_samples
        self.seed = seed
        self._rng = random.Random(seed)
        self.logger = logger

    def _is_detected(self, text: str) -> tuple[bool, float]:
        """return (detected, z_score) for a text."""
        stats = self.encoder.get_detection_stats(text)
        return stats["detected"], stats["z_score"]

    def evaluate_paraphrasing(
        self,
        watermarked_samples: list[str],
        clean_samples: list[str],
        intensity_levels: list[float] | None = None,
    ) -> EvasionResult:
        """
        evaluate paraphrasing evasion across multiple substitution rate levels.

        for each intensity (substitution rate), computes:
          - tpr on watermarked samples after paraphrasing
          - fpr on clean samples after paraphrasing (should stay ~constant)
          - evasion success rate (fraction of watermarked samples that evade)
          - z-score distribution shift

        args:
            watermarked_samples: texts that have been watermarked
            clean_samples: texts that have NOT been watermarked
            intensity_levels: list of substitution rates to evaluate;
                              default [0.1, 0.2, 0.3, 0.4, 0.5]

        returns:
            EvasionResult with per-intensity breakdown
        """
        t_start = time.time()
        if intensity_levels is None:
            intensity_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

        wm = watermarked_samples[: self.n_samples]
        cl = clean_samples[: self.n_samples]

        # baseline z-scores and tpr/fpr
        z_before = [self._is_detected(t)[1] for t in wm]
        tpr_before = sum(1 for t in wm if self._is_detected(t)[0]) / max(len(wm), 1)
        fpr_before = sum(1 for t in cl if self._is_detected(t)[0]) / max(len(cl), 1)

        intensity_results = []
        z_after_final = []
        evasion_successes = []
        sims = []

        for rate in intensity_levels:
            z_at_rate = []
            evaded_at_rate = 0
            sim_at_rate = []

            for text in wm:
                paraphrased, actual_rate = _paraphrase_text(
                    text, rate, random.Random(self.seed + hash(text) % 10000)
                )
                detected, z = self._is_detected(paraphrased)
                z_at_rate.append(z)
                sim_at_rate.append(
                    1.0 - actual_rate
                )  # approximate semantic preservation
                if not detected:
                    evaded_at_rate += 1

            tpr_at_rate = (len(wm) - evaded_at_rate) / max(len(wm), 1)
            esr_at_rate = evaded_at_rate / max(len(wm), 1)

            intensity_results.append(
                {
                    "intensity": rate,
                    "tpr": tpr_at_rate,
                    "evasion_success_rate": esr_at_rate,
                    "mean_z": statistics.mean(z_at_rate) if z_at_rate else 0.0,
                    "mean_semantic_sim": (
                        statistics.mean(sim_at_rate) if sim_at_rate else 1.0
                    ),
                }
            )

            # record at max intensity for final metrics
            if rate == max(intensity_levels):
                z_after_final = z_at_rate
                evasion_successes = [evaded_at_rate / max(len(wm), 1)]
                sims = sim_at_rate

        # tpr/fpr at max intensity (most severe evasion)
        max_rate = max(intensity_levels)
        tpr_after = intensity_results[-1]["tpr"]
        fpr_after = sum(
            1
            for t in cl
            if self._is_detected(
                _paraphrase_text(t, max_rate, random.Random(self.seed))[0]
            )[0]
        ) / max(len(cl), 1)

        elapsed = time.time() - t_start
        self.logger.logger.info(
            f"paraphrase evasion: tpr {tpr_before:.3f} → {tpr_after:.3f} "
            f"at max rate={max_rate}"
        )

        return EvasionResult(
            attack_type="paraphrase",
            n_samples=len(wm),
            z_scores_before=z_before,
            z_scores_after=z_after_final,
            tpr_before=tpr_before,
            fpr_before=fpr_before,
            tpr_after=tpr_after,
            fpr_after=fpr_after,
            mean_semantic_similarity=(statistics.mean(sims) if sims else 1.0),
            evasion_success_rate=(evasion_successes[0] if evasion_successes else 0.0),
            intensity_results=intensity_results,
            execution_time_s=elapsed,
        )

    def evaluate_copy_paste_dilution(
        self,
        watermarked_samples: list[str],
        dilution_samples: list[str],
        dilution_ratios: list[float] | None = None,
    ) -> EvasionResult:
        """
        evaluate copy-paste dilution evasion across dilution ratios.

        mixes watermarked content with non-watermarked filler text.
        the theoretical required dilution to defeat detection is:
            n_dilute / n_wm > (z_wm / z_threshold)^2 - 1

        for z_wm ≈ 8 and z_threshold = 4: need ~3x dilution.
        for z_wm ≈ 6 and z_threshold = 4: need ~1.25x dilution.

        args:
            watermarked_samples: watermarked texts to dilute
            dilution_samples: non-watermarked texts to use as filler
            dilution_ratios: ratios of filler tokens to watermarked tokens;
                             default [0.5, 1.0, 1.5, 2.0, 3.0]

        returns:
            EvasionResult with per-dilution-ratio breakdown
        """
        import math

        t_start = time.time()
        if dilution_ratios is None:
            dilution_ratios = [0.5, 1.0, 1.5, 2.0, 3.0]

        wm = watermarked_samples[: self.n_samples]
        filler = dilution_samples[: self.n_samples]

        z_before = [self._is_detected(t)[1] for t in wm]
        tpr_before = sum(1 for t in wm if self._is_detected(t)[0]) / max(len(wm), 1)
        fpr_before = 0.0  # no clean samples needed here; fpr not meaningful

        intensity_results = []
        z_after_final = []
        final_tpr = tpr_before
        final_esr = 0.0

        for ratio in dilution_ratios:
            z_at_ratio = []
            evaded = 0

            for i, wm_text in enumerate(wm):
                filler_text = (
                    filler[i % len(filler)] if filler else "no additional content"
                )
                wm_tokens = wm_text.split()
                fill_tokens = filler_text.split()

                # dilute: append ratio * len(wm_tokens) filler tokens
                n_fill = int(ratio * len(wm_tokens))
                fill_subset = fill_tokens[:n_fill]
                # interleave 50/50 rather than appending (harder to detect by segment)
                combined_tokens = []
                wi, fi = 0, 0
                while wi < len(wm_tokens) or fi < len(fill_subset):
                    if wi < len(wm_tokens):
                        combined_tokens.append(wm_tokens[wi])
                        wi += 1
                    if fi < len(fill_subset):
                        combined_tokens.append(fill_subset[fi])
                        fi += 1
                diluted = " ".join(combined_tokens)

                detected, z = self._is_detected(diluted)
                z_at_ratio.append(z)
                if not detected:
                    evaded += 1

            tpr_at_ratio = (len(wm) - evaded) / max(len(wm), 1)
            esr_at_ratio = evaded / max(len(wm), 1)

            # theoretical prediction: z_diluted = z_original / sqrt(1 + ratio)
            mean_z_before = statistics.mean(z_before) if z_before else 0.0
            predicted_z = mean_z_before / math.sqrt(1 + ratio)

            intensity_results.append(
                {
                    "intensity": ratio,
                    "tpr": tpr_at_ratio,
                    "evasion_success_rate": esr_at_ratio,
                    "mean_z": statistics.mean(z_at_ratio) if z_at_ratio else 0.0,
                    "predicted_z": predicted_z,  # theoretical prediction
                    "mean_semantic_sim": 1
                    / (1 + ratio),  # approx: more dilution = less semantic
                }
            )

            if ratio == max(dilution_ratios):
                z_after_final = z_at_ratio
                final_tpr = tpr_at_ratio
                final_esr = esr_at_ratio

        elapsed = time.time() - t_start
        self.logger.logger.info(
            f"copy-paste dilution: tpr {tpr_before:.3f} → {final_tpr:.3f} "
            f"at max ratio={max(dilution_ratios)}"
        )

        return EvasionResult(
            attack_type="copy_paste",
            n_samples=len(wm),
            z_scores_before=z_before,
            z_scores_after=z_after_final,
            tpr_before=tpr_before,
            fpr_before=fpr_before,
            tpr_after=final_tpr,
            fpr_after=fpr_before,  # fpr unaffected by dilution of watermarked text
            mean_semantic_similarity=(1.0 / (1.0 + max(dilution_ratios))),
            evasion_success_rate=final_esr,
            intensity_results=intensity_results,
            execution_time_s=elapsed,
        )

    def evaluate_adaptive_substitution(
        self,
        watermarked_samples: list[str],
        clean_samples: list[str],
        substitution_budgets: list[int] | None = None,
    ) -> EvasionResult:
        """
        evaluate white-box adaptive token substitution evasion.

        the adversary knows the green/red partition (white-box setting) and
        greedily replaces green-list tokens with red-list synonyms.  this is
        the most powerful evasion attack under synonym-restricted substitutions.

        measures tpr degradation as a function of the substitution budget
        (maximum number of tokens the adversary is allowed to change).

        args:
            watermarked_samples: watermarked texts to attack
            clean_samples: clean texts (for fpr measurement)
            substitution_budgets: max substitution counts to evaluate;
                                  default [1, 3, 5, 10, 20]

        returns:
            EvasionResult with per-budget breakdown
        """
        t_start = time.time()
        if substitution_budgets is None:
            substitution_budgets = [1, 3, 5, 10, 20]

        wm = watermarked_samples[: self.n_samples]
        cl = clean_samples[: self.n_samples]

        # require green_set access from encoder
        green_set = getattr(self.encoder, "_green_set", set())
        z_threshold = getattr(self.encoder, "z_threshold", 4.0)
        gamma = getattr(self.encoder, "gamma", 0.25)

        z_before = [self._is_detected(t)[1] for t in wm]
        tpr_before = sum(1 for t in wm if self._is_detected(t)[0]) / max(len(wm), 1)
        fpr_before = sum(1 for t in cl if self._is_detected(t)[0]) / max(len(cl), 1)

        intensity_results = []
        z_after_final = []
        final_tpr = tpr_before
        final_esr = 0.0
        sims = []

        for budget in substitution_budgets:
            z_at_budget = []
            evaded = 0
            sims_at_budget = []

            for text in wm:
                attacked, n_subs = _adaptive_substitution(
                    text,
                    green_set,
                    z_threshold,
                    gamma,
                    max_substitutions=budget,
                    rng=random.Random(self.seed + hash(text) % 10000),
                )
                detected, z = self._is_detected(attacked)
                z_at_budget.append(z)
                sims_at_budget.append(1.0 - n_subs / max(len(text.split()), 1))
                if not detected:
                    evaded += 1

            tpr_at_budget = (len(wm) - evaded) / max(len(wm), 1)
            esr_at_budget = evaded / max(len(wm), 1)

            intensity_results.append(
                {
                    "intensity": budget,
                    "tpr": tpr_at_budget,
                    "evasion_success_rate": esr_at_budget,
                    "mean_z": statistics.mean(z_at_budget) if z_at_budget else 0.0,
                    "mean_semantic_sim": (
                        statistics.mean(sims_at_budget) if sims_at_budget else 1.0
                    ),
                }
            )

            if budget == max(substitution_budgets):
                z_after_final = z_at_budget
                final_tpr = tpr_at_budget
                final_esr = esr_at_budget
                sims = sims_at_budget

        elapsed = time.time() - t_start
        self.logger.logger.info(
            f"adaptive substitution: tpr {tpr_before:.3f} → {final_tpr:.3f} "
            f"at max budget={max(substitution_budgets)}"
        )

        return EvasionResult(
            attack_type="adaptive_substitution",
            n_samples=len(wm),
            z_scores_before=z_before,
            z_scores_after=z_after_final,
            tpr_before=tpr_before,
            fpr_before=fpr_before,
            tpr_after=final_tpr,
            # clean samples unaffected by attack on watermarked text
            fpr_after=fpr_before,
            mean_semantic_similarity=(statistics.mean(sims) if sims else 1.0),
            evasion_success_rate=final_esr,
            intensity_results=intensity_results,
            execution_time_s=elapsed,
        )

    def evaluate_all(
        self,
        watermarked_samples: list[str],
        clean_samples: list[str],
        dilution_samples: list[str] | None = None,
    ) -> dict[str, EvasionResult]:
        """
        run all three evasion evaluations and return a results dict.

        args:
            watermarked_samples: texts embedded with the unigram watermark
            clean_samples: texts without watermark (for fpr measurement)
            dilution_samples: non-watermarked texts for copy-paste dilution;
                              defaults to clean_samples if not provided

        returns:
            dict mapping attack_type → EvasionResult
        """
        if dilution_samples is None:
            dilution_samples = clean_samples

        self.logger.logger.info("starting comprehensive evasion evaluation")

        results: dict[str, EvasionResult] = {}

        results["paraphrase"] = self.evaluate_paraphrasing(
            watermarked_samples, clean_samples
        )
        results["copy_paste"] = self.evaluate_copy_paste_dilution(
            watermarked_samples, dilution_samples
        )
        results["adaptive_substitution"] = self.evaluate_adaptive_substitution(
            watermarked_samples, clean_samples
        )

        self.logger.logger.info(
            "evasion evaluation complete: "
            + ", ".join(
                f"{k}: tpr {v.tpr_before:.3f}→{v.tpr_after:.3f}"
                for k, v in results.items()
            )
        )

        return results

    def generate_evasion_report(
        self, results: dict[str, EvasionResult]
    ) -> dict[str, Any]:
        """
        generate a structured report suitable for notebook display and json export.

        args:
            results: output from evaluate_all()

        returns:
            nested dict with summary statistics and per-intensity breakdowns
        """
        report: dict[str, Any] = {
            "summary": {},
            "intensity_curves": {},
            "z_score_distributions": {},
            "detection_bounds": {
                "z_threshold": getattr(self.encoder, "z_threshold", 4.0),
                "gamma": getattr(self.encoder, "gamma", 0.25),
            },
        }

        for attack_type, result in results.items():
            report["summary"][attack_type] = result.summary()
            report["intensity_curves"][attack_type] = result.intensity_results
            report["z_score_distributions"][attack_type] = {
                "before": result.z_scores_before,
                "after": result.z_scores_after,
            }

        return report
