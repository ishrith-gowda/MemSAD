# PASTE-READY rebuttal responses — Submission 30963 (post by AoE Aug 3)

**HOW TO POST (~5 min):** the per-review Rebuttal button is closed after Jul 27; use **Official Comment** (the PC confirmed Official Comment is treated as equivalent through Aug 3). Post ONE Official Comment per reviewer with the title shown; post the AC block as an **Author-AC Confidential Comment**. If a "readers" field appears, keep the default (reviewers + ACs). Ignore the PAT/LLM feedback (author-only, not used in review).

---

## ►► REVIEWER n6dm — Official Comment, title: "Response to Reviewer n6dm"

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

---

## ►► REVIEWER m2wH — Official Comment, title: "Response to Reviewer m2wH"

Correctness points are conceded and repaired below; already-revised items are quoted.

**W2. Corollary 17.** Correct: ⟨∇s_comb,∇s_max⟩>0 does not imply {δ:⟨δ,∇s_comb⟩<0}⊆{δ:⟨δ,∇s_max⟩<0} (this needs the gradients parallel); your δ=(1,-2) is a valid counterexample. Fix:
(a) Thm 7 holds unconditionally for s_max.
(b) We replace the linear s_comb with the disjunction D=1[s_max>τ_max ∨ s_mean>τ_mean]. A retrieval-effective poison has high s_max, so it triggers the s_max branch; evading that branch reduces s_max = retrieval, so the disjunction inherits Thm 7 for the entries that matter (the s_mean branch only adds low-max, low-retrieval detections). At κ=2 it beats the linear score: MINJA 0.06 to 0.38, INJECMEM 0.00 to 0.50, AGENTPOISON 1.00, at FPR 0.03 to 0.05 vs 0.01 (to 0 at κ=2.5).
(c) Reachability: on real poison ∠(∇s_max,∇s_mean)=40.5° [33.5,54.9] (not the ~90° the construction needs), and the retrieval-optimal direction +∇s_max evades s_comb for 0/8 entries; realizing any such direction in tokens is CVP-hard (Prop 15).

**W3. Eq. 12 / Le Cam / Table 13.** Correct: dR/dτ=0 at τ*=(μ₀+μ₁)/2, so excess error is second-order in |τ̂-τ*|; Eq. 12's linear coefficient is wrong. We drop that step. Le Cam gives |τ̂-τ*|=Ω(σρ) unless N=Ω(1/ρ²) (App B.3), matched by Thm 11's upper bound; the Ω(1/ρ²) calibration bound stands. Table 13: correct that FPR+(1-TPR)≤0.05 is unreachable absolutely at ρ=1.27 (oracle floor Φ(-0.635)≈0.26). The N_min column is ⌈4(1-ε)²/ρ²⌉ and the "0.05" is excess-over-oracle; the caption is wrong and we will rewrite it to "excess ≤0.05."

**W1. q*∈H.** Correct that Lemma 8 Step 1 uses it (Assumption 9). The certificate is deterministic exactly when the victim query lies in the history support: the cross-principal case (attacker writes; a different victim issues in-distribution queries). It weakens to probabilistic only when the attacker also controls the query to be atypical (same-principal self-poisoning, which u2M8 argues is not a threat). For AGENTPOISON the trigger makes q* atypical, so we use triggered calibration (Table 5, TPR=1.00). We correct "regardless of adversary strategy" to the calibration-support scope.

**W4. Weak numbers / inconsistencies.**
- AUROC (threshold-independent): AGENTPOISON 1.00, MINJA 0.951, INJECMEM 0.846. The κ=2 TPRs you cite are the FPR≈0 operating point; κ=1 gives MINJA 1.00. "MINJA TPR 0.40 to 1.00 across tables" is one ROC read at different κ, not a contradiction.
- Composite = disjunction π_wm∨π_SAD∨π_pro (App C; the weighted-sum form is deprecated there, not what Tables 3/11 report). Table 11: watermark rejects on AP/MJ, MEMSAD on MJ/IM. WRITE model: watermark carries MINJA; QUERY/SINGLE (auto-storage bypasses watermarking): MEMSAD is the operative detector. We will make the per-defense, per-threat-model attribution explicit.
- Table 12 (Clopper-Pearson, held-out, 20 trials): FPR=0.000 [0,0.004]; our reproduction ≈0.01 to 0.02 at κ=2. Table 16's 0.09/0.138 matches BGE-Large's FPR (Table 9), not the primary MiniLM detector's, a labeling error we will correct. Table 15 already calibrates all baselines on the identical regime (its caption).

**W5. Citations.** Correct: the bib mixes anonymized placeholders with public IDs (incl. INJECMEM). The evaluated attacks are the published AGENTPOISON, MINJA, INJECMEM; no result depends on an anonymized source. We will fix the full reference list for camera-ready.

**Q1** corrected detector: deploy the disjunction, whose s_max branch carries Thm 7 (unconditional for retrieval-effective poison). **Q2** Eq. 12's first-order term does not survive; the Ω(1/ρ²) bound is on threshold estimation, matched by Thm 11. **Q3** the reported composite is the disjunction; we will provide raw per-seed predictions, seeds, exact CI aggregation, and an anonymized repository to the AC via Official Comment.

---

## ►► REVIEWER u2M8 — Official Comment, title: "Response to Reviewer u2M8"

**W1. Threat model.** Agreed the meaningful threat is cross-principal; we will lead with it. Cases: (1) shared/multi-tenant memory, where one principal (or a compromised account) writes entries a different principal's session retrieves; (2) tool-output ingestion, where the agent writes untrusted web/email/document content into memory and the attacker controls the content, not the agent (the dominant vector, no write access needed); (3) MINJA (query-only) and INJECMEM (single-interaction) are different-principal by construction. WRITE (Def 2-3) is the worst-case adversary used to state the strongest guarantee; we will label it as such. The certificate is deterministic precisely in the cross-principal case (a legitimate victim issues in-distribution queries) and weakens only for same-principal self-poisoning, which, as you note, is not a meaningful threat. So the assumption holds where the threat is real.

**W2. Overclaiming.** Agreed. We correct "regardless of adversary strategy" to: holds for continuous perturbations under Assumption 9 (typical query). It does not cover a colluding user crafting an atypical query, nor discrete synonym substitution (Prop 13). We present the triggered-calibration operating point explicitly, not as universal coverage.

**W3. Hardness irrelevant if greedy synonyms win.** Agreed; we do not claim the hardness result defends against greedy substitution. Prop 15 (CVP) characterizes the worst-case optimal unconstrained-token evasion: gradient coupling (Thm 7) closes the continuous surface, the CVP reduction closes the unconstrained-token surface, and the low-perturbation synonym regime is left as the residual gap, exactly the frontier Prop 13 identifies and we report (80 to 100% evasion, ΔASR-R≈0). We will foreground this. New: the certified synonym radius is encoder-dependent (r* ≥ 1 under mpnet vs 0 under MiniLM; a max-over-encoders detector gives r* ≥ 1 for a majority), which partially mitigates it.

**Q. SINGLE access, |P|=1 vs 15 entries.** A wording ambiguity we will fix. "|P|=1" means one interaction (one write episode), not one entry; within that interaction INJECMEM writes 3×n_base=15 broad-anchor entries from 8 templates (lines 257-264). We will restate Def 2 as "a single adversarial interaction that writes |P|≥1 entries"; the threat-model results use the per-interaction definition.

---

## ►► AREA CHAIR d5Ze — Author-AC Confidential Comment, title: "Author response to the meta-review"

Corrections and new analyses addressing the meta-review.

**1. Correctness.** Three objections are valid; conceded and repaired:
(a) Corollary 17 (combined-score coupling): the acute-angle step is a non-sequitur. The guarantee holds for s_max (Thm 7, unconditional). We deploy a disjunctive detector D=1[s_max>τ_max ∨ s_mean>τ_mean] whose s_max branch inherits Thm 7 for retrieval-effective poison and which empirically beats the linear score (MINJA 0.06 to 0.38, INJECMEM 0.00 to 0.50 at κ=2).
(b) Eq. 12 (Le Cam): the first-order excess-error term is wrong (risk derivative 0 at the Bayes threshold). The Ω(1/ρ²) result is a threshold-estimation/calibration bound, correct and matched by Thm 11; we drop the flawed step, rate unchanged.
(c) Lemma 8 assumes the victim query is in the history support: deterministic in the cross-principal threat model, weakening only for same-principal self-poisoning (which u2M8 argues is not a threat). m2wH's q*∈H concern and u2M8's threat-model concern resolve each other.
Five further objections target already-revised text, which we quote per-review: discrete-token CVP (App B.7), √-scaling Wasserstein bound (App B.9), Laurent-Massart calibration (App B.4), composite=disjunction (App C), Le Cam calibration (not batch) formulation (App B.3). Two table fixes: Table 13 caption ("0.05" is excess-over-oracle, not absolute error) and Table 16 (its FPR mislabels BGE-Large as the primary detector; held-out FPR is 0.000, Table 12).

**2. Randomized-smoothing comparison (requested).** Cohen (2019) applied to the detector is vacuous in d=384: best R=0.008 vs a single substitution's shift 0.378; noise of norm σ√d destroys the score. MEMSAD's certificate is objective-conflict (any evasion reduces retrieval), not noise robustness, and not a special case.

**3. Certified-radius informativeness (requested).** Gap g=s_adv-τ: AGENTPOISON +0.21, 100% detected, 38% with a full-σ margin (informative); MINJA boundary; INJECMEM-semantic and synonym vacuous. We will add the regime delineation.

**4. Fair comparison to related work.** We add the smoothing comparison and position gradient coupling relative to it; Le Cam is not claimed as a novel technique (only its application to the calibration floor).

**5. Citations.** The evaluated attacks are the published AGENTPOISON/MINJA/INJECMEM; the anonymized/public-ID mix is an error, fully corrected for camera-ready; no result depends on an anonymized source.

Anonymized repository (raw per-seed predictions, seeds, exact CI aggregation) available via Official Comment on request.
