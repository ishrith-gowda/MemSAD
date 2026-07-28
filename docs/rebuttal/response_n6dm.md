<!-- DRAFT rebuttal response to Reviewer n6dm (rating 4, conf 5). Plain text + markdown, no links, anonymity preserved. Target <=10,000 chars. Author reviews before posting. -->

We thank the reviewer for the exceptionally careful reading and the constructive, actionable questions. We ran the two analyses you requested; both are summarized below and will be folded into the final version. We also address each weakness directly.

**Q1 / W1 — When is the certified radius informative? (empirical gap analysis)**

We ran exactly the experiment you proposed: for each attack we measured, per poison entry, the gap g(c) = s_adv(c) − τ (τ = μ̂ + κσ̂, κ=2), and the fraction of entries that are detected (g>0) and *comfortably certified* (g > κσ̂, i.e. a full calibration-uncertainty above threshold, hence robust to a bounded perturbation). |M|=1000, all-MiniLM-L6-v2.

| attack | mean g | min g | detected | certified (g>κσ̂) |
|---|---|---|---|---|
| AGENTPOISON (triggered) | +0.213 | +0.197 | 100% | 38% |
| MINJA | −0.020 | −0.137 | 50% | 0% |
| INJECMEM (semantic score) | −0.181 | −0.270 | 0% | 0% |

This gives exactly the honest delineation you asked for:
- For AGENTPOISON — the continuous/WRITE-access attack the theory targets — the gap is large and positive for *every* entry (min +0.197), all detected, and 38% carry a full calibration-uncertainty robustness margin. Here the certificate is **empirically informative**, not vacuous.
- For MINJA the gap sits at the boundary (≈0), and for INJECMEM the *semantic* gap is negative — the semantic certificate is genuinely tight/vacuous here. INJECMEM is instead flagged by the lexical (character n-gram / domain-vocabulary) component of MEMSAD+, not the max-cosine score. We will state this explicitly.
- A single synonym substitution drives g negative for all attacks (evasion), consistent with Proposition 13.

We agree the gap is an attacker-influenced quantity and that the certificate is conditional; we will foreground this. The precise, honest statement the data supports is: the certified radius is informative for continuous perturbations of retrieval-effective (WRITE-access) attacks, and we do not claim protection in the small-gap (MINJA/synonym) regime. We are correcting the abstract's "regardless of adversary strategy" to "for adversaries whose retrieval-triggering queries are covered by the calibration distribution" (see also the q*∈H point below, which we now scope precisely).

**W5 — Comparison to randomized-smoothing certificates**

Excellent point; we implemented the comparison. We applied the Cohen–Rosenfeld–Kolter (2019) smoothing certificate to the MEMSAD max-cosine detector on a detected AGENTPOISON entry (score 0.783, gap +0.218) and swept the smoothing noise σ. One synonym substitution's embedding shift for this entry is δ_emb=0.378.

| σ | noise norm ≈ σ√d | p_lower | R_cohen |
|---|---|---|---|
| 0.05 | 0.98 | 0.561 | 0.008 |
| 0.10 | 1.96 | 0.000 | 0.000 |
| ≥0.25 | ≥4.9 | 0.000 | 0.000 |

At no σ does smoothing certify even one substitution's worth of embedding movement (best R=0.008 ≪ 0.378). The reason is dimensional: isotropic Gaussian noise of norm σ√384 destroys the max-cosine score for any σ large enough to yield a non-trivial radius, so p_lower collapses below ½. Randomized smoothing is therefore **vacuous for this detector**.

This clarifies the relationship, which we will add: smoothing certifies an L2 ball via *noise robustness* (R=σΦ⁻¹(p)); MEMSAD certifies via *detection–retrieval gradient coupling* — the evasion direction provably degrades the attacker's own retrieval objective (Thm 7). This is a constraint on the *attacker's utility*, not a noise-averaging argument, so gradient coupling is not a special case of smoothing; and, as shown, smoothing does not provide a usable certificate in this setting. We believe this is a favorable and clarifying distinction rather than an omission.

**W2 — Encoder regularity vs. black-box encoders**

We agree the theory should not silently assume more than the encoders provide. Two clarifications: (i) the coupling in Thm 7 is a *geometric* statement on the unit sphere (the max-cosine gradient and the retrieval-objective gradient are the same Riemannian gradient Π⊥ E(q*)); it does not require encoder invertibility, only differentiability of the score along the evasion path, which holds a.e. for the piecewise-smooth encoders used. (ii) We will add an empirical check of the local Lipschitz behavior of the six encoders used (finite-difference sensitivity of E along random and adversarial directions), reporting the constant that enters the certified-radius condition. We expect this to make the theory–experiment link explicit.

**W3 — Le Cam is a textbook technique**

We agree Le Cam's method itself is standard and do not claim it as a novel technique. The novelty we intend to claim is (a) the gradient-coupling theorem, and (b) the *reduction* establishing that calibrated memory-poisoning detection has an intrinsic Ω(1/ρ²) *calibration* sample-complexity floor. We are also revising the surrounding statement: the excess-error step (Eq. 12) is stated as first-order, but the risk derivative vanishes at the Bayes threshold, so the excess classification error is second-order in the threshold error. The correct and cleaner statement is a lower bound on *threshold-estimation* sample complexity (Ω(1/ρ²)), which is exactly what calibration performs and which the matching upper bound (Thm 11) already validates. The headline rate is unchanged; we will remove the flawed intermediate conversion. We thank the reviewer/AC for pushing on correctness here.

**W4 — Online regret bound feels tangential**

Fair. The regret bound's role is to guarantee that rolling calibration keeps τ̂ within O(σ√(log/m)) of τ*, which is precisely the condition the certified radius needs to remain valid under drift; we will make this connection explicit rather than presenting it as a standalone result, and are happy to move the derivation to the appendix.

**W6 — The practical attack (synonyms) is outside the theory's continuous scope**

We agree this is the central limitation and want to be maximally clear. Proposition 13 is not a hedge but the paper's main negative result: continuous-space coupling cannot cover discrete synonym substitution (ε_syn≈0 for paraphrase-trained encoders). We have made concrete progress on this frontier since submission that we will add: (i) the certified synonym radius is strongly encoder-dependent — under all-mpnet-base-v2 the margin exceeds the per-substitution shift, giving a positive certified radius (r*≥1) where all-MiniLM gives r*=0; and (ii) a max-over-encoders detector certifies r*≥1 for a majority of poison because a substitution must simultaneously drop the score across all encoders. This does not "close" the gap, but it converts it from an open hole into a characterized, partially-mitigable frontier with a concrete deployment recommendation (encoder choice). We will state the honest ceiling (r*=1) explicitly.

We are grateful for the review — the two experiments you requested materially strengthen the paper, and we have corrected the two overstatements (Eq. 12's leading term; the abstract's "regardless of adversary strategy"). We would welcome any further questions during the discussion period.
