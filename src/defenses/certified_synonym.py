"""certified discrete synonym-robustness radius for memsad+.

the memsad gradient-coupling theorem certifies robustness against *continuous*
perturbations; proposition 11 (synonym-invariance loophole) shows that *discrete*
synonym substitution has epsilon_syn approx 0 and escapes that guarantee. memsad+
adds a character n-gram jensen-shannon divergence feature D_char that is provably
not synonym-invariant (paper eq. memsad_plus):

    D_char(c') >= D_char(c) - r * delta_char

for any adversary applying r single-word synonym substitutions, where delta_char
bounds the per-substitution shift in D_char.

this module turns that inequality into a *per-entry certificate*. if the benign
calibrated threshold is tau_char, then an entry c with margin
D_char(c) - tau_char > 0 is guaranteed to remain flagged against any adversary
using at most

    r*(c) = floor( (D_char(c) - tau_char) / delta_char )

single-word synonym substitutions. we provide two delta_char estimators:

  * analytical (mode="analytical"): a rigorous per-substitution upper bound via
    the fannes-audenaert entropy inequality applied to the jsd. this yields a
    *true certificate* -- no r-substitution adversary can evade -- at the cost of
    looseness.
  * empirical (mode="empirical"): the measured maximum single-substitution jsd
    shift over the entry vocabulary and the adaptive synonym dictionary. tighter,
    but a high-probability radius rather than a strict multi-step bound.

all quantities are computed in nats to match memsad_plus.jensen_shannon_divergence
(natural-log kl), whose range is [0, ln 2].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from attacks.adaptive_attack import _ADAPTIVE_SYNONYMS
from defenses.memsad_plus import (
    extract_char_ngrams,
    jensen_shannon_divergence,
    ngram_distribution,
)

_LN2 = math.log(2.0)


@dataclass
class CertifiedRadiusResult:
    """per-entry certified synonym-robustness result."""

    entry: str
    d_char: float
    tau_char: float
    delta_char: float
    certified_radius: int
    mode: str
    n_applicable_substitutions: int = 0
    # true when the char feature alone does not flag the entry (margin <= 0);
    # detection then falls back to the semantic component and no char certificate
    # is issued.
    char_flagged: bool = True

    @property
    def margin(self) -> float:
        return self.d_char - self.tau_char

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "d_char": self.d_char,
            "tau_char": self.tau_char,
            "delta_char": self.delta_char,
            "margin": self.margin,
            "certified_radius": self.certified_radius,
            "mode": self.mode,
            "n_applicable_substitutions": self.n_applicable_substitutions,
            "char_flagged": self.char_flagged,
        }


def _binary_entropy_nats(t: float) -> float:
    """binary entropy h(t) = -t ln t - (1-t) ln(1-t), in nats."""
    if t <= 0.0 or t >= 1.0:
        return 0.0
    return -t * math.log(t) - (1.0 - t) * math.log(1.0 - t)


def _fannes_audenaert_nats(l1: float, support_size: int) -> float:
    """upper bound on |H(p) - H(q)| in nats given ||p - q||_1 = l1.

    fannes-audenaert inequality: |H(p) - H(q)| <= T ln(d - 1) + h(T), where
    T = min(||p - q||_1 / 2, 1) is the total variation distance and d is the
    alphabet (support) size. valid for any d >= 2.
    """
    d = max(support_size, 2)
    t = min(0.5 * l1, 1.0)
    return t * math.log(d - 1) + _binary_entropy_nats(t)


def jsd_lipschitz_bound(l1: float, support_size: int) -> float:
    """rigorous upper bound on |JSD(p' || p0) - JSD(p || p0)| in nats.

    with JSD(x || p0) = H(m) - 0.5 H(x) - 0.5 H(p0), m = (x + p0) / 2, the p0
    term cancels under a perturbation p -> p'. bounding each entropy difference
    by fannes-audenaert and using ||m' - m||_1 = 0.5 ||p' - p||_1 gives

        |dJSD| <= FA(0.5 * l1) + 0.5 * FA(l1).

    the result is clamped to ln 2 (the jsd range).
    """
    bound = _fannes_audenaert_nats(
        0.5 * l1, support_size
    ) + 0.5 * _fannes_audenaert_nats(l1, support_size)
    return min(bound, _LN2)


def l1_ngram_perturbation_bound(
    word_out: str, word_in: str, n_before: int, n: int = 3
) -> float:
    """upper bound on ||p_{c'} - p_c||_1 for one word substitution word_out -> word_in.

    space-joined character n-grams: substituting a word changes only n-grams that
    overlap its span plus the n-1 char context on each side (and the two adjacent
    spaces). we conservatively over-count the affected n-grams, so the returned
    value is a strict upper bound. the total n-gram count shifts by exactly
    len(word_in) - len(word_out).
    """
    ctx = 2 * (n - 1) + 2  # n-1 context each side, plus two word-boundary spaces
    k_rem = len(word_out) + ctx
    k_add = len(word_in) + ctx
    n0 = max(n_before, 1)
    n1 = max(n0 + (len(word_in) - len(word_out)), 1)
    # renormalization drift on unchanged mass + added/removed mass
    drift = abs(n0 - n1) / max(n0, n1)
    bound = drift + k_add / n1 + k_rem / n0
    return min(bound, 2.0)  # l1 distance between two distributions is at most 2


def _applicable_substitutions(
    entry: str, synonym_dict: dict[str, list[str]]
) -> list[tuple[str, str]]:
    """(word_out, word_in) pairs the adaptive adversary could apply to this entry."""
    pairs: list[tuple[str, str]] = []
    words = {w.lower().strip(".,;:!?") for w in entry.split()}
    for w in words:
        for repl in synonym_dict.get(w, []):
            pairs.append((w, repl))
    return pairs


def _delta_char_analytical(
    entry: str, support_size: int, synonym_dict: dict[str, list[str]], n: int
) -> tuple[float, int]:
    """worst-case per-substitution jsd shift for this entry (rigorous upper bound).

    returns (delta_char, n_applicable). delta_char is the max over all single
    substitutions applicable to the entry of the analytical jsd lipschitz bound;
    this holds for any intermediate text state, so it is a valid multi-step
    per-substitution bound and yields a true certificate.
    """
    ngrams = extract_char_ngrams(entry, n)
    n_before = sum(ngrams.values())
    pairs = _applicable_substitutions(entry, synonym_dict)
    if not pairs:
        return 0.0, 0
    worst = 0.0
    for word_out, word_in in pairs:
        l1 = l1_ngram_perturbation_bound(word_out, word_in, n_before, n)
        worst = max(worst, jsd_lipschitz_bound(l1, support_size))
    return worst, len(pairs)


def _delta_char_empirical(
    entry: str,
    benign_char_dist: dict[str, float],
    synonym_dict: dict[str, list[str]],
    n: int,
) -> tuple[float, int]:
    """measured maximum single-substitution jsd shift for this entry.

    tighter than the analytical bound but a high-probability estimate: it measures
    the first-step shift and assumes subsequent steps do not exceed it.
    """
    base_dist = ngram_distribution(extract_char_ngrams(entry, n))
    base_jsd = jensen_shannon_divergence(base_dist, benign_char_dist)
    words = entry.split()
    pairs = 0
    worst = 0.0
    for idx, word in enumerate(words):
        key = word.lower().strip(".,;:!?")
        for repl in synonym_dict.get(key, []):
            pairs += 1
            new_words = words[:]
            new_words[idx] = repl
            new_dist = ngram_distribution(extract_char_ngrams(" ".join(new_words), n))
            new_jsd = jensen_shannon_divergence(new_dist, benign_char_dist)
            worst = max(worst, abs(base_jsd - new_jsd))
    return worst, pairs


def certified_synonym_radius(
    entry: str,
    benign_char_dist: dict[str, float],
    tau_char: float,
    *,
    n: int = 3,
    mode: str = "analytical",
    synonym_dict: dict[str, list[str]] | None = None,
) -> CertifiedRadiusResult:
    """certify how many single-word synonym substitutions memsad+ provably withstands.

    args:
        entry: candidate memory entry text.
        benign_char_dist: calibrated benign corpus n-gram distribution p0.
        tau_char: memsad+ character-jsd detection threshold (mu + kappa * sigma).
        n: character n-gram size (must match calibration).
        mode: "analytical" (true certificate) or "empirical" (tighter estimate).
        synonym_dict: adversary substitution dictionary; defaults to the adaptive
            adversary's curated domain dictionary.

    returns:
        CertifiedRadiusResult with the certified radius r*(c).
    """
    if synonym_dict is None:
        synonym_dict = _ADAPTIVE_SYNONYMS

    p_c = ngram_distribution(extract_char_ngrams(entry, n))
    d_char = jensen_shannon_divergence(p_c, benign_char_dist)

    if d_char <= tau_char:
        return CertifiedRadiusResult(
            entry=entry,
            d_char=d_char,
            tau_char=tau_char,
            delta_char=0.0,
            certified_radius=0,
            mode=mode,
            n_applicable_substitutions=0,
            char_flagged=False,
        )

    if mode == "analytical":
        support = len(set(p_c) | set(benign_char_dist))
        delta_char, n_pairs = _delta_char_analytical(entry, support, synonym_dict, n)
    elif mode == "empirical":
        delta_char, n_pairs = _delta_char_empirical(
            entry, benign_char_dist, synonym_dict, n
        )
    else:
        raise ValueError(f"unknown mode: {mode!r} (use 'analytical' or 'empirical')")

    margin = d_char - tau_char
    if delta_char <= 0.0:
        # no applicable substitution can move D_char at all -> unconditionally robust
        radius = math.inf if n_pairs == 0 else 10**9
        radius = int(min(radius, 10**9))
    else:
        radius = int(math.floor(margin / delta_char))

    return CertifiedRadiusResult(
        entry=entry,
        d_char=d_char,
        tau_char=tau_char,
        delta_char=delta_char,
        certified_radius=max(radius, 0),
        mode=mode,
        n_applicable_substitutions=n_pairs,
        char_flagged=True,
    )


@dataclass
class CorpusCertificate:
    """aggregate certificate over a set of poison entries."""

    mode: str
    n_entries: int
    n_char_flagged: int
    radii: list[int] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """fraction of entries with a strictly positive certificate."""
        if self.n_entries == 0:
            return 0.0
        return sum(1 for r in self.radii if r > 0) / self.n_entries

    @property
    def median_radius(self) -> float:
        flagged = sorted(r for r in self.radii if r > 0)
        if not flagged:
            return 0.0
        mid = len(flagged) // 2
        if len(flagged) % 2:
            return float(flagged[mid])
        return 0.5 * (flagged[mid - 1] + flagged[mid])


def certify_corpus(
    poison_entries: list[str],
    benign_char_dist: dict[str, float],
    tau_char: float,
    *,
    n: int = 3,
    mode: str = "analytical",
    synonym_dict: dict[str, list[str]] | None = None,
) -> CorpusCertificate:
    """certify a set of poison entries and aggregate the radius distribution."""
    radii: list[int] = []
    n_flagged = 0
    for entry in poison_entries:
        res = certified_synonym_radius(
            entry,
            benign_char_dist,
            tau_char,
            n=n,
            mode=mode,
            synonym_dict=synonym_dict,
        )
        radii.append(res.certified_radius)
        if res.char_flagged:
            n_flagged += 1
    return CorpusCertificate(
        mode=mode,
        n_entries=len(poison_entries),
        n_char_flagged=n_flagged,
        radii=radii,
    )
