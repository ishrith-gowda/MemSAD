<!-- DRAFT Author-AC confidential comment. Plain text+markdown, no links, anonymity. <=10k chars. -->

Corrections and new analyses addressing the meta-review.

**1. Correctness.** Three objections are valid; conceded and repaired:
(a) Corollary 17 (combined-score coupling) — the acute-angle step is a non-sequitur. The guarantee holds for s_max (Thm 7, unconditional). We deploy a disjunctive detector D=1[s_max>τ_max ∨ s_mean>τ_mean] whose s_max branch inherits Thm 7 for retrieval-effective poison and which empirically beats the linear score (MINJA 0.06→0.38, INJECMEM 0.00→0.50 at κ=2).
(b) Eq. 12 (Le Cam) — the first-order excess-error term is wrong (risk derivative 0 at the Bayes threshold). The Ω(1/ρ²) result is a threshold-estimation/calibration bound, correct and matched by Thm 11; we drop the flawed step, rate unchanged.
(c) Lemma 8 assumes the victim query is in the history support — deterministic in the cross-principal threat model, weakening only for same-principal self-poisoning (which u2M8 argues is not a threat). m2wH's q*∈H concern and u2M8's threat-model concern resolve each other.
Five further objections target already-revised text, which we quote per-review: discrete-token CVP (App B.7), √-scaling Wasserstein bound (App B.9), Laurent–Massart calibration (App B.4), composite=disjunction (App C), Le Cam calibration (not batch) formulation (App B.3). Two table fixes: Table 13 caption ("0.05" is excess-over-oracle, not absolute error) and Table 16 (its FPR mislabels BGE-Large as the primary detector; held-out FPR is 0.000, Table 12).

**2. Randomized-smoothing comparison (requested).** Cohen (2019) applied to the detector is vacuous in d=384: best R=0.008 vs a single substitution's shift 0.378; noise of norm σ√d destroys the score. MEMSAD's certificate is objective-conflict (any evasion reduces retrieval), not noise robustness — not a special case.

**3. Certified-radius informativeness (requested).** Gap g=s_adv−τ: AGENTPOISON +0.21, 100% detected, 38% with a full-σ margin (informative); MINJA boundary; INJECMEM-semantic and synonym vacuous. We will add the regime delineation.

**4. Fair comparison to related work.** We add the smoothing comparison and position gradient coupling relative to it; Le Cam is not claimed as a novel technique (only its application to the calibration floor).

**5. Citations.** The evaluated attacks are the published AGENTPOISON/MINJA/INJECMEM; the anonymized/public-ID mix is an error, fully corrected for camera-ready; no result depends on an anonymized source.

Anonymized repository (raw per-seed predictions, seeds, exact CI aggregation) available via Official Comment on request.
