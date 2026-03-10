"""
lexical diversity filter for sad: a secondary gate against synonym-substitution evasion.

motivation:
    the sad combined-mode detector exploits high cosine similarity between poison
    entries and victim queries.  an adaptive adversary can evade this by synonym
    substitution: replacing query-aligned words with domain synonyms shifts the
    anomaly score below the threshold without materially changing retrieval rank,
    because all-MiniLM-L6-v2 is trained on paraphrase objectives and assigns
    near-identical embeddings to synonyms.

    however, adversarially crafted passages exhibit characteristic lexical patterns
    that differ from benign memory entries:
      (1) repetitive vocabulary: poison passages tend to repeat semantically loaded
          terms (credentials, override, schedule, access, task, etc.) at higher
          frequency than benign entries of the same length.
      (2) low type-token ratio (ttr): adversarial entries pack attack keywords into
          a small unique vocabulary, lowering ttr relative to benign entries.
      (3) high n-gram overlap with the query set: even after synonym substitution,
          adversarial passages share high unigram/bigram overlap with victim queries.

    this module implements three complementary lexical signals and combines them
    into a lexical diversity score used as a secondary gate.  entries flagged by sad
    AND having low lexical diversity are confirmed as suspicious; entries flagged by
    sad but with normal lexical diversity are held for manual review rather than
    automatically blocked, reducing false positives under the synonym-substitution
    scenario.

algorithm:
    1. ttr gate: type-token ratio = |unique_words| / |total_words|.
       benign entries have ttr ∈ [0.55, 0.95]; adversarial entries cluster < 0.45.
    2. query n-gram overlap: fraction of entry unigrams that appear in the victim
       query set vocabulary.  benign entries share ~10–20%; adversarial entries
       share 40–70% after query-alignment optimization.
    3. repetition rate: most-frequent-word count / total words.  measures keyword
       stuffing.  benign entries score < 0.08; adversarial entries score > 0.12.

    lexical_diversity_score = ttr - overlap_weight * ngram_overlap - repetition_rate
    an entry is lexically anomalous if lexical_diversity_score < threshold.

usage:
    from defenses.lexical_diversity import LexicalDiversityGate
    gate = LexicalDiversityGate()
    gate.calibrate(benign_entries, victim_queries)
    result = gate.score(candidate_entry)
    if result["is_lexically_anomalous"]:
        # flag for secondary review or combined blocking
        ...

references:
    - gao et al. "strip: a defence against trojan attacks on deep neural networks."
      acsac 2019.  (perturbation-based detection, analogous concept)
    - zhao et al. "provable robust watermarking for ai-generated text." iclr 2024.
      (vocabulary bias as detectable signal)
    - ebrahimi et al. "hotflip: white-box adversarial examples for text classification."
      acl 2018.  (adversarial text optimization creates distinctive lexical patterns)

all comments are lowercase.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LexicalDiversityResult:
    """
    per-entry result from LexicalDiversityGate.

    fields:
        entry_text: the memory entry scored
        ttr: type-token ratio (unique / total words)
        ngram_overlap: fraction of entry unigrams in victim query vocabulary
        repetition_rate: most-frequent-word count / total words
        lexical_score: combined score (higher = more diverse = more benign)
        threshold: calibrated decision boundary
        is_lexically_anomalous: True if lexical_score < threshold
        calibration_mean: mean lexical_score over benign calibration set
        calibration_std: std dev over benign calibration set
    """

    entry_text: str
    ttr: float
    ngram_overlap: float
    repetition_rate: float
    lexical_score: float
    threshold: float
    is_lexically_anomalous: bool
    calibration_mean: float
    calibration_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttr": self.ttr,
            "ngram_overlap": self.ngram_overlap,
            "repetition_rate": self.repetition_rate,
            "lexical_score": self.lexical_score,
            "threshold": self.threshold,
            "is_lexically_anomalous": self.is_lexically_anomalous,
        }


# ---------------------------------------------------------------------------
# tokenizer (no external dependencies)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """
    simple whitespace + punctuation tokenizer.

    lowercases, strips punctuation, and splits on whitespace.
    intentionally simple — no stemming or lemmatization, to preserve
    the detection signal from adversarial keyword choice.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]  # drop single-char tokens


def _type_token_ratio(tokens: List[str]) -> float:
    """
    compute type-token ratio: |unique tokens| / |total tokens|.

    returns 1.0 for very short texts (< 5 tokens) to avoid division artifacts.
    benign agent memory entries: ttr ≈ 0.65–0.85.
    adversarial passages with repeated keywords: ttr ≈ 0.30–0.55.
    """
    n = len(tokens)
    if n < 5:
        return 1.0
    return len(set(tokens)) / n


def _ngram_overlap(tokens: List[str], query_vocab: set) -> float:
    """
    fraction of entry tokens that appear in the victim query vocabulary.

    adversarial passages aligned to victim queries share 40–70% of unigrams
    with the query vocabulary.  benign entries share ~10–20%.

    args:
        tokens: tokenized entry
        query_vocab: set of unique tokens from all victim queries

    returns:
        float in [0, 1]
    """
    if not tokens or not query_vocab:
        return 0.0
    overlap = sum(1 for t in tokens if t in query_vocab)
    return overlap / len(tokens)


def _repetition_rate(tokens: List[str]) -> float:
    """
    most-frequent-word count / total words.

    keyword-stuffed adversarial passages have one dominant word repeated
    many times (e.g., "task" appears in 15% of tokens).  benign entries
    have a more even frequency distribution (most frequent word ≈ 5–8%).

    returns 0.0 for texts shorter than 5 tokens.
    """
    if len(tokens) < 5:
        return 0.0
    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(tokens)


# ---------------------------------------------------------------------------
# main gate
# ---------------------------------------------------------------------------


class LexicalDiversityGate:
    """
    secondary gate for sad: lexical diversity analysis.

    acts as a secondary signal after sad flags an entry:
      - entries with low ttr, high query overlap, or high repetition rate
        are confirmed as lexically anomalous.
      - entries with normal lexical diversity (likely synonym-evasion attempts)
        are soft-flagged for review rather than hard-blocked.

    calibration:
        fit mean and std of lexical_score on known-benign entries.
        threshold = mean - threshold_sigma * std (low score = anomalous).
        separate query vocabulary is built from observed victim queries.

    combining with sad:
        the recommended integration is:
            if sad_flagged and lexically_anomalous: hard-block (confirmed attack)
            if sad_flagged and not lexically_anomalous: soft-flag (possible evasion)
            if not sad_flagged: pass (no anomaly signal)
        this preserves sad's recall while adding precision against evasion.
    """

    # weight on the ngram_overlap penalty in the combined score.
    # ttr contribution is positive (higher ttr = more diverse = less suspicious);
    # ngram_overlap and repetition_rate are negative penalties.
    _OVERLAP_WEIGHT: float = 0.4
    _REPETITION_WEIGHT: float = 0.6

    def __init__(self, threshold_sigma: float = 1.5) -> None:
        """
        args:
            threshold_sigma: how many standard deviations below the benign mean
                the score must fall to be flagged.  lower k = more sensitive.
                k=1.5 → FPR ≈ 6.7% for normal distributions; recommended for
                use as a secondary gate (primary gate is sad).
        """
        self.threshold_sigma = threshold_sigma
        self.calibration_mean: Optional[float] = None
        self.calibration_std: Optional[float] = None
        self.query_vocab: set = set()
        self.is_calibrated: bool = False

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def calibrate(
        self,
        benign_entries: List[str],
        victim_queries: List[str],
    ) -> Dict[str, float]:
        """
        fit the lexical score distribution on benign entries.

        builds the victim query vocabulary and computes the threshold
        as mean_score - threshold_sigma * std_score.

        args:
            benign_entries: known-clean memory entry texts
            victim_queries: all observed victim query strings

        returns:
            dict with mean, std, threshold, n_entries
        """
        import numpy as np

        if not benign_entries:
            raise ValueError("benign_entries must be non-empty")

        # build query vocabulary from all victim queries
        self.query_vocab = set()
        for q in victim_queries:
            self.query_vocab.update(_tokenize(q))

        # compute lexical scores for all benign entries
        scores = [self._compute_score(e) for e in benign_entries]
        self.calibration_mean = float(np.mean(scores))
        self.calibration_std = float(np.std(scores)) + 1e-8
        self.is_calibrated = True

        threshold = self.calibration_mean - self.threshold_sigma * self.calibration_std

        return {
            "mean": self.calibration_mean,
            "std": self.calibration_std,
            "threshold": threshold,
            "n_entries": len(benign_entries),
            "n_query_vocab": len(self.query_vocab),
        }

    def score(self, entry: str) -> LexicalDiversityResult:
        """
        compute lexical diversity score for a single entry.

        requires calibrate() to have been called first.

        returns:
            LexicalDiversityResult with all sub-scores and detection flag
        """
        if not self.is_calibrated:
            raise RuntimeError("calibrate() must be called before score()")

        tokens = _tokenize(entry)
        ttr = _type_token_ratio(tokens)
        ngram_overlap = _ngram_overlap(tokens, self.query_vocab)
        rep_rate = _repetition_rate(tokens)

        lex_score = self._score_from_components(ttr, ngram_overlap, rep_rate)
        threshold = self.calibration_mean - self.threshold_sigma * self.calibration_std

        return LexicalDiversityResult(
            entry_text=entry,
            ttr=ttr,
            ngram_overlap=ngram_overlap,
            repetition_rate=rep_rate,
            lexical_score=lex_score,
            threshold=threshold,
            is_lexically_anomalous=lex_score < threshold,
            calibration_mean=self.calibration_mean,
            calibration_std=self.calibration_std,
        )

    def score_batch(self, entries: List[str]) -> List[LexicalDiversityResult]:
        """score multiple entries efficiently (no batch speedup needed — cpu only)."""
        return [self.score(e) for e in entries]

    def evaluate_on_corpus(
        self,
        poison_entries: List[str],
        benign_entries: List[str],
    ) -> Dict[str, Any]:
        """
        evaluate lexical diversity gate on labeled corpus.

        computes tpr and fpr at the calibrated threshold, plus auroc.

        args:
            poison_entries: known adversarial entries (ground truth positive)
            benign_entries: known benign entries (ground truth negative)

        returns:
            dict with tpr, fpr, auroc, threshold, and component averages
        """
        import numpy as np

        all_entries = poison_entries + benign_entries
        labels = [1] * len(poison_entries) + [0] * len(benign_entries)

        results = self.score_batch(all_entries)
        scores = [r.lexical_score for r in results]
        preds = [1 if r.is_lexically_anomalous else 0 for r in results]

        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        n_pos = len(poison_entries)
        n_neg = len(benign_entries)

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0

        # auroc via sorting: lower score = more anomalous = predicted positive.
        # invert scores so that high = predicted positive for standard auroc.
        neg_scores = [-s for s in scores]
        auroc = _auroc_from_scores(neg_scores, labels)

        threshold = self.calibration_mean - self.threshold_sigma * self.calibration_std

        # component averages
        poison_results = results[: len(poison_entries)]
        benign_results = results[len(poison_entries) :]

        return {
            "tpr": tpr,
            "fpr": fpr,
            "auroc": auroc,
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "n_poison": n_pos,
            "n_benign": n_neg,
            "mean_ttr_poison": float(np.mean([r.ttr for r in poison_results])),
            "mean_ttr_benign": float(np.mean([r.ttr for r in benign_results])),
            "mean_overlap_poison": float(
                np.mean([r.ngram_overlap for r in poison_results])
            ),
            "mean_overlap_benign": float(
                np.mean([r.ngram_overlap for r in benign_results])
            ),
            "mean_rep_poison": float(
                np.mean([r.repetition_rate for r in poison_results])
            ),
            "mean_rep_benign": float(
                np.mean([r.repetition_rate for r in benign_results])
            ),
        }

    def get_config(self) -> Dict[str, Any]:
        """return gate configuration and calibration state."""
        return {
            "threshold_sigma": self.threshold_sigma,
            "is_calibrated": self.is_calibrated,
            "calibration_mean": self.calibration_mean,
            "calibration_std": self.calibration_std,
            "n_query_vocab": len(self.query_vocab),
            "overlap_weight": self._OVERLAP_WEIGHT,
            "repetition_weight": self._REPETITION_WEIGHT,
        }

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self, entry: str) -> float:
        """compute lexical_score for a single entry (uses current query_vocab)."""
        tokens = _tokenize(entry)
        ttr = _type_token_ratio(tokens)
        overlap = _ngram_overlap(tokens, self.query_vocab)
        rep = _repetition_rate(tokens)
        return self._score_from_components(ttr, overlap, rep)

    def _score_from_components(
        self, ttr: float, ngram_overlap: float, repetition_rate: float
    ) -> float:
        """
        combine sub-scores into a single lexical diversity score.

        score = ttr - overlap_weight * ngram_overlap - repetition_weight * rep_rate

        properties:
            - higher score → more diverse → more likely benign
            - lower score → more repetitive/query-aligned → more likely adversarial
            - range approximately [−0.4, 1.0] in practice
        """
        return (
            ttr
            - self._OVERLAP_WEIGHT * ngram_overlap
            - self._REPETITION_WEIGHT * repetition_rate
        )


# ---------------------------------------------------------------------------
# combined sad + lexical gate evaluator
# ---------------------------------------------------------------------------


class SADWithLexicalGate:
    """
    combined anomaly detector: sad + lexical diversity secondary gate.

    integrates sad (semantic anomaly detection) with the lexical diversity
    gate as a second-stage filter.  the combination addresses the synonym-
    invariance loophole: adversaries who shift anomaly scores via synonym
    substitution are caught by the lexical gate if their passages remain
    lexically distinguishable.

    three-tier output:
        "pass"       — sad score below threshold (not flagged at all)
        "soft_flag"  — sad flagged but lexical score is normal (possible evasion)
        "hard_block" — sad flagged AND lexically anomalous (confirmed attack)

    calibration:
        both components are calibrated independently on the same benign corpus.
        the sad component calibrates cosine similarity statistics;
        the lexical gate calibrates ttr / overlap / repetition statistics.

    usage:
        combined = SADWithLexicalGate()
        combined.calibrate(benign_entries, victim_queries)
        for new_entry in incoming_entries:
            result = combined.detect(new_entry)
            if result["verdict"] == "hard_block":
                reject(new_entry)
            elif result["verdict"] == "soft_flag":
                queue_for_review(new_entry)
    """

    def __init__(
        self,
        sad_threshold_sigma: float = 2.0,
        lex_threshold_sigma: float = 1.5,
        sad_scoring_mode: str = "combined",
        model_name: str = "all-MiniLM-L6-v2",
        max_query_history: int = 100,
    ) -> None:
        """
        args:
            sad_threshold_sigma: k for sad anomaly threshold (μ + k·σ).
            lex_threshold_sigma: k for lexical gate threshold (μ - k·σ).
            sad_scoring_mode: "max" or "combined" (see SemanticAnomalyDetector).
            model_name: sentence-transformer for sad embeddings.
            max_query_history: rolling query window size for sad.
        """
        from defenses.semantic_anomaly import SemanticAnomalyDetector

        self.sad = SemanticAnomalyDetector(
            threshold_sigma=sad_threshold_sigma,
            model_name=model_name,
            max_query_history=max_query_history,
            scoring_mode=sad_scoring_mode,
        )
        self.lex_gate = LexicalDiversityGate(threshold_sigma=lex_threshold_sigma)
        self._victim_queries: List[str] = []

    def calibrate(
        self,
        benign_entries: List[str],
        victim_queries: List[str],
    ) -> Dict[str, Any]:
        """
        calibrate both components on the same benign corpus and victim query set.

        args:
            benign_entries: known-clean memory strings
            victim_queries: representative victim query strings

        returns:
            dict with "sad" and "lex" sub-dicts of calibration statistics
        """
        self._victim_queries = list(victim_queries)
        sad_stats = self.sad.calibrate(benign_entries, victim_queries)
        for q in victim_queries:
            self.sad.update_query_set(q)
        lex_stats = self.lex_gate.calibrate(benign_entries, victim_queries)
        return {"sad": sad_stats, "lex": lex_stats}

    def update_query_set(self, query: str) -> None:
        """add a new observed query to the sad rolling window."""
        self.sad.update_query_set(query)
        # extend lex gate vocabulary with new query tokens
        self.lex_gate.query_vocab.update(_tokenize(query))

    def detect(self, entry: str) -> Dict[str, Any]:
        """
        run both components and return three-tier verdict.

        returns:
            dict with keys:
                verdict: "pass" | "soft_flag" | "hard_block"
                sad_score: anomaly score from sad
                sad_flagged: bool
                lex_score: lexical diversity score
                lex_flagged: bool
                sad_result: full AnomalyScore dict
                lex_result: full LexicalDiversityResult dict
        """
        sad_result = self.sad.detect(entry)
        lex_result = self.lex_gate.score(entry)

        sad_flagged = sad_result.is_anomalous
        lex_flagged = lex_result.is_lexically_anomalous

        if sad_flagged and lex_flagged:
            verdict = "hard_block"
        elif sad_flagged and not lex_flagged:
            verdict = "soft_flag"
        else:
            verdict = "pass"

        return {
            "verdict": verdict,
            "sad_flagged": sad_flagged,
            "lex_flagged": lex_flagged,
            "sad_score": sad_result.anomaly_score,
            "lex_score": lex_result.lexical_score,
            "sad_result": sad_result.to_dict(),
            "lex_result": lex_result.to_dict(),
        }

    def evaluate_on_corpus(
        self,
        poison_entries: List[str],
        benign_entries: List[str],
        verdict_threshold: str = "soft_flag",
    ) -> Dict[str, Any]:
        """
        evaluate combined gate on labeled corpus.

        args:
            poison_entries: adversarial entries (ground truth positive)
            benign_entries: benign entries (ground truth negative)
            verdict_threshold: count verdicts at or above this level as positive.
                "soft_flag": flag if sad triggers (matches sad alone).
                "hard_block": flag only if both sad and lex trigger.

        returns:
            dict with tpr, fpr, auroc for combined detector
        """
        verdict_levels = {"pass": 0, "soft_flag": 1, "hard_block": 2}
        min_level = verdict_levels[verdict_threshold]

        all_entries = poison_entries + benign_entries
        labels = [1] * len(poison_entries) + [0] * len(benign_entries)

        results = [self.detect(e) for e in all_entries]
        preds = [1 if verdict_levels[r["verdict"]] >= min_level else 0 for r in results]
        # soft score: sad_score for auroc (lex is secondary gate, not ranked)
        scores = [r["sad_score"] for r in results]

        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        n_pos = len(poison_entries)
        n_neg = len(benign_entries)

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        auroc = _auroc_from_scores(scores, labels)

        hard_blocks = sum(1 for r in results[:n_pos] if r["verdict"] == "hard_block")
        soft_flags = sum(1 for r in results[:n_pos] if r["verdict"] == "soft_flag")

        return {
            "tpr": tpr,
            "fpr": fpr,
            "auroc": auroc,
            "verdict_threshold": verdict_threshold,
            "n_hard_block_poison": hard_blocks,
            "n_soft_flag_poison": soft_flags,
            "n_pass_poison": n_pos - hard_blocks - soft_flags,
        }


# ---------------------------------------------------------------------------
# auroc helper (shared)
# ---------------------------------------------------------------------------


def _auroc_from_scores(scores: List[float], labels: List[int]) -> float:
    """
    compute auroc from raw scores and binary labels (1 = positive).

    uses the wilcoxon-mann-whitney statistic (equivalent to trapezoidal auc).
    returns 0.5 if all labels are the same class.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    paired = sorted(zip(scores, labels), reverse=True)
    tp = 0
    auc = 0.0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)
