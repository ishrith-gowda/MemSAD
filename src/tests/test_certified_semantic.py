"""tests for the semantic randomized-smoothing certificate.

the anchor test is the linear-score case: for score(e) = w . e the smoothed
detector's certified radius equals the exact distance to the decision hyperplane,
d = (w . e0 - tau) / ||w||, up to clopper-pearson conservatism (R <= d). a bug in
the confidence lower bound or the Phi^{-1} mapping breaks this.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from defenses.certified_semantic import (
    SemanticSmoothingCertifier,
    certified_score_margin_radius,
    make_max_cosine_score_fn,
)


class TestLinearScoreExactRadius:
    def test_radius_matches_true_distance(self):
        rng = np.random.default_rng(0)
        d = 32
        w = rng.normal(size=d)
        w /= np.linalg.norm(w)
        tau = 0.0
        d0 = 0.25  # true signed distance to boundary
        e0 = d0 * w
        sigma = 0.25

        def score(e):
            return float(w @ e)

        cert = SemanticSmoothingCertifier(
            score,
            tau,
            sigma=sigma,
            n_samples=20000,
            alpha=1e-3,
            renormalize=False,
            seed=1,
        )
        res = cert.certify_embedding(e0)
        assert res.flagged
        # soundness: the certified radius never exceeds the true distance
        assert res.radius <= d0 + 1e-6
        # tightness: with many samples it recovers most of the true distance
        assert res.radius >= 0.15

    def test_abstains_below_threshold(self):
        rng = np.random.default_rng(2)
        d = 16
        w = rng.normal(size=d)
        w /= np.linalg.norm(w)

        def score(e):
            return float(w @ e)

        # e0 on the negative side of the boundary -> not flagged -> abstain
        e0 = -0.3 * w
        cert = SemanticSmoothingCertifier(
            score, 0.0, sigma=0.25, n_samples=4000, renormalize=False, seed=3
        )
        res = cert.certify_embedding(e0)
        assert res.radius == 0.0
        assert res.abstained

    def test_smoothed_detector_robust_within_radius(self):
        # empirically verify the guarantee: at displacement = certified radius in
        # the worst-case direction, the smoothed detection probability stays >= 1/2.
        rng = np.random.default_rng(4)
        d = 24
        w = rng.normal(size=d)
        w /= np.linalg.norm(w)
        d0 = 0.3
        e0 = d0 * w
        sigma = 0.25

        def score(e):
            return float(w @ e)

        cert = SemanticSmoothingCertifier(
            score,
            0.0,
            sigma=sigma,
            n_samples=20000,
            alpha=1e-3,
            renormalize=False,
            seed=5,
        )
        r = cert.certify_embedding(e0).radius
        assert r > 0
        # worst direction is -w (toward the boundary)
        e_shift = e0 - r * w
        # analytic smoothed prob at the shifted point: Phi((w.e_shift - tau)/sigma)
        p = float(norm.cdf((w @ e_shift) / sigma))
        assert p >= 0.5 - 1e-6


class TestMaxCosineScore:
    def test_max_cosine_matches_manual(self):
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        fn = make_max_cosine_score_fn(q)
        e = np.array([0.8, 0.6])
        assert fn(e) == pytest.approx(0.8)

    def test_certify_flags_high_similarity_entry(self):
        # an entry aligned with a query has high max-cosine -> flagged with a
        # positive certificate; tau set below its score.
        q = np.eye(8)
        fn = make_max_cosine_score_fn(q)
        e0 = np.zeros(8)
        e0[0] = 1.0  # perfectly aligned with query 0 -> score 1.0
        cert = SemanticSmoothingCertifier(
            fn, tau=0.5, sigma=0.25, n_samples=4000, renormalize=True, seed=6
        )
        res = cert.certify_embedding(e0)
        assert res.flagged
        assert res.radius > 0.0


class TestSynonymReduction:
    def test_synonym_radius_is_floor_ratio(self):
        rng = np.random.default_rng(7)
        d = 16
        w = rng.normal(size=d)
        w /= np.linalg.norm(w)
        e0 = 0.4 * w

        def score(e):
            return float(w @ e)

        cert = SemanticSmoothingCertifier(
            score,
            0.0,
            sigma=0.25,
            n_samples=8000,
            alpha=1e-3,
            renormalize=False,
            seed=8,
        )
        emb = cert.certify_embedding(e0)
        # three synthetic substitution embeddings at a known max shift
        shift = 0.05
        subs = [e0 + shift * rng.normal(size=d) for _ in range(3)]
        syn = cert.certify_synonyms(e0, subs)
        expected_delta = max(float(np.linalg.norm(e0 - s)) for s in subs)
        assert syn.delta_emb == pytest.approx(expected_delta)
        assert syn.certified_radius == int(np.floor(emb.radius / expected_delta))
        assert syn.n_applicable_substitutions == 3

    def test_no_substitutions_zero_radius(self):
        w = np.ones(8) / np.sqrt(8)
        e0 = 0.4 * w

        def score(e):
            return float(w @ e)

        cert = SemanticSmoothingCertifier(
            score, 0.0, sigma=0.25, n_samples=2000, renormalize=False, seed=9
        )
        syn = cert.certify_synonyms(e0, [])
        assert syn.certified_radius == 0
        assert syn.n_applicable_substitutions == 0


class TestScoreMarginCertificate:
    def _linear(self, w):
        def score(e):
            return float(w @ e)

        return score

    def test_abstains_below_threshold(self):
        w = np.ones(4) / 2.0
        score = self._linear(w)
        e0 = np.zeros(4)  # score 0 < tau
        cert = certified_score_margin_radius(e0, 0.5, [e0 + 0.1], score)
        assert not cert.flagged
        assert cert.certified_radius == 0

    def test_unconditional_when_no_substitution_reduces_score(self):
        w = np.array([1.0, 0.0])
        score = self._linear(w)
        e0 = np.array([1.0, 0.0])  # score 1.0 > tau
        # substitutions that only move orthogonally -> never reduce the score
        subs = [np.array([1.0, 0.3]), np.array([1.0, -0.2])]
        cert = certified_score_margin_radius(e0, 0.5, subs, score)
        assert cert.certified_radius >= 10**8  # unconditional sentinel

    def test_radius_is_floor_margin_over_delta(self):
        w = np.array([1.0, 0.0])
        score = self._linear(w)
        e0 = np.array([1.0, 0.0])  # score 1.0, tau 0.4 -> margin 0.6
        # worst substitution reduces score by 0.2 (x-component 0.8)
        subs = [np.array([0.8, 0.0]), np.array([0.9, 0.0])]
        cert = certified_score_margin_radius(e0, 0.4, subs, score)
        assert cert.delta_score == pytest.approx(0.2)
        assert cert.certified_radius == 3  # floor(0.6 / 0.2)

    def test_soundness_composed_substitutions_stay_flagged(self):
        # a bounded-shift adversary applying r* substitutions each reducing the
        # score by at most delta must not cross tau; r*+1 may.
        w = np.array([1.0, 0.0])
        score = self._linear(w)
        e0 = np.array([1.0, 0.0])
        tau = 0.4
        step = 0.15  # per-substitution reduction
        subs = [np.array([1.0 - step, 0.0])]
        cert = certified_score_margin_radius(e0, tau, subs, score)
        r, delta = cert.certified_radius, cert.delta_score
        # r worst-case reductions keep the score strictly above tau (soundness)
        assert score(e0) - r * delta > tau
        # r+1 worst-case reductions would reach or cross tau (tightness)
        assert score(e0) - (r + 1) * delta <= tau + 1e-12
