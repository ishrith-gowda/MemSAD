<!-- DRAFT n6dm (4, conf 5). Plain text+markdown, no links, anonymity. <=10k chars. -->

We ran both requested analyses.

**Q1/W1. Gap distribution.** Per-entry gap g = s_adv - τ, τ = μ̂+κσ̂, κ=2, |M|=1000, all-MiniLM:
- AGENTPOISON (triggered): mean g = +0.213, min +0.197; 100% detected, 38% with g > κσ̂.
- MINJA: mean g = -0.02; 50% detected (boundary).
- INJECMEM (semantic score): mean g = -0.18; 0%, flagged by the lexical component, not max-cosine.
- A single synonym substitution drives g < 0 for all three.

So the certificate is informative for AGENTPOISON (its design regime, large positive gap), and honestly tight (MINJA) or vacuous (INJECMEM-semantic, synonym) elsewhere. We will add this, and change the abstract's "regardless of adversary strategy" to "for adversaries whose retrieval-triggering queries lie in the calibration support."

**W5. Randomized smoothing.** We applied Cohen et al. (2019) to the max-cosine detector on a detected AGENTPOISON entry (score 0.783, gap +0.218); one substitution's shift is δ_emb = 0.378. Certified radius R = σΦ⁻¹(p_lower): σ=0.05 gives R=0.008; σ≥0.10 gives R=0 (p_lower < ½, since isotropic noise of norm σ√384 destroys the score). Smoothing certifies nothing here. The distinction: smoothing certifies an L2 ball via noise robustness; MEMSAD's certificate is a constraint on the attacker's own objective (any evasion direction reduces retrieval, Thm 7), not a noise-averaging argument, so it is not a special case of smoothing. We will add the comparison.

**W2. Encoder regularity.** Thm 7 is geometric on the sphere (∇s_max = Π⊥ E(q*)); it needs differentiability of the score along the path a.e., not encoder invertibility. Empirical check, ‖E(c')-E(c)‖ over 377 single-token substitutions (unit vectors, diameter 2.0): mean 0.075 to 0.153, 95th pct ≤0.354, max ≤0.389 across all six encoders. Bounded; the per-encoder max is the Lipschitz constant that enters the radius. We will add the table.

**W3. Le Cam is textbook.** Agreed; we do not claim Le Cam as a novel technique. The contributions are (a) the gradient-coupling theorem and (b) the Ω(1/ρ²) calibration sample-complexity floor for this detection problem. Separately, Eq. 12's first-order excess-error term is wrong: the risk derivative vanishes at the Bayes threshold, so excess classification error is second-order in |τ̂-τ*|. We drop that step and state the bound on threshold estimation, |τ̂-τ*| = Ω(σρ) unless N = Ω(1/ρ²), matched by Thm 11's upper bound. The Ω(1/ρ²) rate is unchanged.

**W4. Regret bound tangential.** Its role is to keep |τ̂-τ*| = O(σ√(log/m)) under drift, which is the condition the certified radius requires. We will state this link explicitly and move the derivation to the appendix.

**W6. Synonyms outside the theory.** Agreed; Prop 13 is the paper's main negative result, not a hedge. New results on the frontier: the certified synonym radius is encoder-dependent (r* ≥ 1 under all-mpnet-base-v2 vs r* = 0 under all-MiniLM), and a max-over-encoders detector gives r* ≥ 1 for a majority of poison, since a substitution must drop the score across all encoders at once. The ceiling is r* = 1; we will state it as characterized and partially mitigable.
