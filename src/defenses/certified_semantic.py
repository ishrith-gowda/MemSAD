"""certified robustness radius for the memsad semantic detector via randomized
smoothing (cohen, rosenfeld & kolter, 2019 -- already cited in the paper).

feature attribution (docs/research/phase38_findings.md) established that the
synonym loophole is a *semantic-feature* phenomenon: greedy word-level synonym
substitution drives the memsad max-cosine anomaly score below the detection
threshold, evading detection while (approximately) preserving retrieval. this
module certifies the semantic detector against that evasion.

two layers, with clearly separated rigor:

1. embedding-space l2 certificate (RIGOROUS). smooth the base detector with
   isotropic gaussian noise in embedding space:

       G(e) = 1[ P_{delta ~ N(0, sigma^2 I)}( score(e + delta) > tau ) >= 1/2 ].

   if the smoothed detector flags e0 with monte-carlo count k out of n and the
   one-sided clopper-pearson lower bound p_lower = LCB(k, n, alpha) exceeds 1/2,
   then by cohen et al. theorem 1 (binary case) G is constant on the l2 ball of
   radius

       R(e0) = sigma * Phi^{-1}(p_lower).

   this is an exact, distribution-free certificate on embedding perturbations.

2. synonym-count reduction (MODELED adversary). a synonym substitution moves the
   entry embedding e(c) -> e(c'); over r substitutions the total shift is at most
   r * delta_emb by the triangle inequality, where delta_emb is the maximum
   single-substitution embedding shift. measuring delta_emb over the adaptive
   synonym dictionary yields a certified synonym radius

       r*(c) = floor( R(e(c)) / delta_emb(c) ).

   this certifies robustness against a dictionary-bounded, bounded-shift
   adversary -- it is NOT a certificate against an unrestricted paraphrase
   adversary, because delta_emb is measured (first-order) rather than a global
   encoder lipschitz constant. this boundary is exactly the paper's proposition
   11 synonym-invariance loophole, now quantified.

the embedding-space certificate (layer 1) is the airtight contribution; layer 2
is an honest, explicitly-scoped reduction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from evaluation.statistical import clopper_pearson_ci

ScoreFn = Callable[[np.ndarray], float]


@dataclass
class EmbeddingCertificate:
    """rigorous embedding-space l2 certificate for one entry."""

    flagged: bool
    p_lower: float
    radius: float  # certified l2 radius in embedding space (0.0 if abstained)
    n_samples: int
    n_flagged: int
    sigma: float

    @property
    def abstained(self) -> bool:
        return self.radius == 0.0 and self.p_lower <= 0.5


@dataclass
class SynonymCertificate:
    """certified synonym-substitution radius (modeled bounded-shift adversary)."""

    embedding_radius: float
    delta_emb: float
    certified_radius: int
    n_applicable_substitutions: int


class SemanticSmoothingCertifier:
    """randomized-smoothing certifier for a scalar anomaly-score detector.

    args:
        score_fn: maps an embedding vector (d,) to a scalar anomaly score. the
            base detector flags iff score_fn(e) > tau.
        tau: detection threshold.
        sigma: smoothing noise standard deviation (embedding space).
        n_samples: monte-carlo samples for the smoothed detection probability.
        alpha: one-sided confidence level for the clopper-pearson lower bound.
        renormalize: if True, project noised embeddings back to the unit sphere
            (sentence embeddings are l2-normalized). the certificate radius is
            still reported in the ambient l2 metric, which upper-bounds geodesic
            movement, keeping the guarantee conservative.
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        tau: float,
        *,
        sigma: float = 0.25,
        n_samples: int = 1000,
        alpha: float = 1e-3,
        renormalize: bool = True,
        seed: int = 0,
    ) -> None:
        self.score_fn = score_fn
        self.tau = tau
        self.sigma = sigma
        self.n_samples = n_samples
        self.alpha = alpha
        self.renormalize = renormalize
        self._rng = np.random.default_rng(seed)

    def _count_flags(self, e0: np.ndarray, n: int) -> int:
        d = e0.shape[0]
        flags = 0
        # batch the noise for speed
        batch = 256
        remaining = n
        while remaining > 0:
            b = min(batch, remaining)
            noise = self._rng.normal(0.0, self.sigma, size=(b, d))
            samples = e0[None, :] + noise
            if self.renormalize:
                norms = np.linalg.norm(samples, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                samples = samples / norms
            for i in range(b):
                if self.score_fn(samples[i]) > self.tau:
                    flags += 1
            remaining -= b
        return flags

    def certify_embedding(self, e0: np.ndarray) -> EmbeddingCertificate:
        """certified embedding-space l2 radius for e0 (cohen binary case)."""
        k = self._count_flags(e0, self.n_samples)
        # one-sided lower confidence bound at level alpha == lower end of a
        # two-sided (1 - 2*alpha) clopper-pearson interval (cohen et al. trick).
        p_lower = clopper_pearson_ci(k, self.n_samples, alpha=2.0 * self.alpha).lower
        if p_lower > 0.5:
            radius = float(self.sigma * norm.ppf(p_lower))
        else:
            radius = 0.0
        return EmbeddingCertificate(
            flagged=k > self.n_samples / 2,
            p_lower=float(p_lower),
            radius=max(radius, 0.0),
            n_samples=self.n_samples,
            n_flagged=k,
            sigma=self.sigma,
        )

    def certify_synonyms(
        self, e0: np.ndarray, substitution_embeddings: list[np.ndarray]
    ) -> SynonymCertificate:
        """certified synonym radius = floor(embedding_radius / max shift).

        args:
            e0: embedding of the entry.
            substitution_embeddings: embeddings of the entry after each single
                applicable synonym substitution (measured with the same encoder).
        """
        emb = self.certify_embedding(e0)
        n_pairs = len(substitution_embeddings)
        if n_pairs == 0 or emb.radius == 0.0:
            return SynonymCertificate(
                embedding_radius=emb.radius,
                delta_emb=0.0,
                certified_radius=0,
                n_applicable_substitutions=n_pairs,
            )
        delta_emb = max(
            float(np.linalg.norm(e0 - e_sub)) for e_sub in substitution_embeddings
        )
        if delta_emb <= 0.0:
            radius = 0
        else:
            radius = int(np.floor(emb.radius / delta_emb))
        return SynonymCertificate(
            embedding_radius=emb.radius,
            delta_emb=delta_emb,
            certified_radius=max(radius, 0),
            n_applicable_substitutions=n_pairs,
        )


def make_max_cosine_score_fn(query_embeddings: np.ndarray) -> ScoreFn:
    """build a memsad max-cosine anomaly score function over a query set.

    query_embeddings: (n_q, d) l2-normalized victim-query embeddings. the score
    of an entry embedding e is max_q cos(e, q); for unit vectors this is
    max_q (e . q).
    """
    q = np.asarray(query_embeddings, dtype=np.float64)

    def score(e: np.ndarray) -> float:
        return float(np.max(q @ e))

    return score


@dataclass
class ScoreMarginCertificate:
    """direct score-margin certified synonym radius (modeled bounded-shift adversary).

    unlike the randomized-smoothing certificate, this does not perturb the
    embedding; it certifies directly on the anomaly score. it avoids the
    high-dimensional smoothing pathology (isotropic gaussian noise of norm
    ~sigma*sqrt(d) destroys the max-cosine score, collapsing the smoothing
    radius to zero) and is the practical certificate for memsad's semantic
    detector.
    """

    score: float
    tau: float
    delta_score: float
    certified_radius: int
    n_applicable_substitutions: int

    @property
    def margin(self) -> float:
        return self.score - self.tau

    @property
    def flagged(self) -> bool:
        return self.score > self.tau


# an entry no dictionary substitution can push toward the boundary is
# unconditionally robust within the modeled adversary; reported as this sentinel.
_UNCONDITIONAL = 10**9


def certified_score_margin_radius(
    entry_emb: np.ndarray,
    tau: float,
    substitution_embeddings: list[np.ndarray],
    score_fn: ScoreFn,
) -> ScoreMarginCertificate:
    """certify how many synonym substitutions keep the anomaly score above tau.

    with s0 = score_fn(entry_emb) and per-substitution max score reduction
    delta_score = max_j max(s0 - score_fn(c_j), 0) over single applicable
    substitutions c_j, any r <= floor((s0 - tau) / delta_score) substitutions
    provably keep the score above tau under a bounded-shift adversary (each step
    reduces the score by at most delta_score). this is the semantic analog of the
    empirical char-jsd certificate and inherits the same modeled-adversary caveat.
    """
    s0 = score_fn(entry_emb)
    sub_scores = [score_fn(e_sub) for e_sub in substitution_embeddings]
    return certified_radius_from_scores(s0, tau, sub_scores)


def certified_radius_from_scores(
    score: float, tau: float, substitution_scores: list[float]
) -> ScoreMarginCertificate:
    """score-level score-margin certificate (encoder- and ensemble-agnostic).

    args:
        score: anomaly score of the entry.
        tau: detection threshold.
        substitution_scores: anomaly score after each single applicable
            substitution. any aggregate score (single encoder max-cosine, or an
            ensemble mean over encoders) may be used, which is how the multi-
            encoder ensemble certificate reuses this helper.
    """
    n = len(substitution_scores)
    if score <= tau:
        return ScoreMarginCertificate(score, tau, 0.0, 0, n)
    if n == 0:
        return ScoreMarginCertificate(score, tau, 0.0, _UNCONDITIONAL, 0)
    delta_score = max(max(score - s, 0.0) for s in substitution_scores)
    if delta_score <= 0.0:
        radius = _UNCONDITIONAL
    else:
        radius = int((score - tau) // delta_score)
    return ScoreMarginCertificate(score, tau, delta_score, max(radius, 0), n)
