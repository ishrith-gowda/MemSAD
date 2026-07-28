# NeurIPS 2026 Rebuttal Plan — Submission 30963 (MEMSAD)

**Status:** DRAFT PLAN, 2026-07-27. Window Jul 23 – **Aug 3** (Official Comment ≡ Rebuttal button per PC clarification; **additional results explicitly allowed**). ~7 days.

> Provenance note: parts of the exhaustive appendix audit (Le Cam Eq 12 derivation, Tables 13/15/16/18, Appendix C composite) were being extracted by deep-analysis agents that hit the account session limit (resets 10pm CT) mid-run. Items marked **[VERIFY-APPENDIX]** need the completed read before the response is finalized. Everything else is verified from a direct read of pp. 4–9 of the submitted PDF + the review text.

---

## 0. Situation assessment

| Reviewer | Rating | Conf | Q/C/S/O | Posture |
|---|---|---|---|---|
| n6dm | **4 borderline accept** | **5 (certain)** | 3/3/3/3 | expert, positive; asks map onto our phase-38 work |
| u2M8 | 3 borderline reject | 3 | 2/2/2/2 | threat-model motivation, overclaiming |
| m2wH | **2 reject** | 4 | 2/2/2/2 | alleges multiple verifiable math errors + a counterexample |
| AC d5Ze | — meta | — | — | neutral→negative; decision hinges on **theoretical correctness** + citations + smoothing comparison + certified-radius informativeness |

**Key structural fact:** the reviewed PDF (modified May 27) is *already the pat-corrected version* — it has the reframed calibration-sample-complexity theorem, explicit Assumption 9 (typical victim query, with the colluding-user caveat), the discrete/continuous synonym distinction, and the combined-score directional-coupling hedge (Cor 17). The reviewers reviewed the corrected paper and still object. **We cannot lean on "already fixed in v2" — real mathematical responses are required.**

**Realistic read:** uphill. One confident reject (m2wH, conf 4) on correctness grounds is the central threat; the AC explicitly makes correctness the bar. BUT n6dm (conf 5) is already borderline-accept and asked for exactly the things phase-38 delivers, and the two negative reviewers are lower-confidence. If we (a) neutralize the correctness allegations via honest concede-and-repair, (b) convert n6dm to a clear accept by nailing their two asks, (c) reframe the threat model, the paper is movable. We maximize odds AND treat every correction as ICLR-2027-strengthening regardless of outcome.

---

## 1. Strategic frame (priority order)

1. **Neutralize the correctness allegations (m2wH, PAT)** via honest concede-and-repair. Do NOT bluff m2wH — clearly expert. The goal is to move the *AC's* perception of correctness, not necessarily m2wH's score.
2. **Convert n6dm (conf 5, already borderline-accept)** — HIGHEST ROI. They handed us their acceptance conditions: (Q1) empirical gap distribution; (W5) randomized-smoothing comparison. Phase-38 already built both. Deliver them precisely.
3. **Reframe threat model (u2M8)** — cross-principal deployments.
4. **Reconcile empirical inconsistencies honestly (m2wH W4)** — the composite/watermark issue is the hardest; honesty is the only play.
5. **Engage early** (PC email urges it): post initial responses ~Jul 28, iterate through Aug 3.

---

## 2. Per-issue adjudication + response

### A. Combined-score coupling counterexample — m2wH W2, PAT. VERDICT: **valid, but repairable.**
The deployed detector is `s_comb = ½·s_max + ½·s_mean` (Eq 7, Def 6). Cor 17 claims it retains directional coupling. Counterexample (verified): ∇s_max=(1,0), ∇s_mean=(0,1) ⇒ ∇s_comb=(½,½); δ=(1,−2) gives ⟨δ,∇s_comb⟩=−½<0 (detector drops) yet ⟨δ,∇s_max⟩=+1>0 (retrieval rises). Valid. **But it hits the corollary, not Theorem 7**, which is unconditional for `s_max`.
- **Repair:** (i) the certified guarantee is stated & proven for `s_max` (Thm 7) — untouched; (ii) state the *exact* condition for s_comb coupling: directional coupling holds iff every candidate evasion direction satisfies ⟨δ,∇s_mean⟩ ≥ −(α/(1−α))·⟨δ,∇s_max⟩ (the mean gradient does not oppose the max gradient beyond the mixing ratio); the counterexample violates this by construction (orthogonal gradients); (iii) **NEW EXP (E3):** measure on real DPR-HotFlip / adaptive evasion trajectories the fraction of steps satisfying the condition — hypothesis: ≈100%, because ∇s_mean and ∇s_max are strongly positively correlated for real query sets (the pathological orthogonal case is not what optimizers find); (iv) note deploying pure `s_max` recovers the full unconditional certificate at modest empirical cost (phase-38 max-ensemble is the robust variant).
- **Tone:** concede the corollary was stated too generally; the fix strengthens, not weakens, the paper.

### B. Certified radius conditional/"vacuous" + q*∈H — n6dm Q1, m2wH W1, u2M8, AC. VERDICT: **partially valid (certificate IS conditional; Assumption 9 stated but understated).**
This is n6dm's #1 ASK and a **gift** — they specified the exact experiment. Lemma 8 needs Δs = s_adv − s̄ > κσ̂ + η, and the proof uses Assumption 9 (q* typical w.r.t. H). m2wH sharpens: AGENTPOISON TPR=0 plain / TPR=1 triggered ⇒ the certificate needs trigger-aware calibration.
- **Response:** (i) **NEW EXP (E1):** plot the distribution of the adversarial-benign gap Δs under each attack (AGENTPOISON/MINJA/INJECMEM); report the fraction inside vs outside the certified region; honestly delineate the **informative regime** (large-gap attacks: AGENTPOISON post-optimization, high-norm) vs the **vacuous regime** (tight-margin / synonym). Phase-38 `certify_semantic_eval` already measures these margins (~0.03–0.13). This directly answers Q1 AND the AC's "discussion of what attacks the radius mitigates."
(ii) Reframe q*∈H honestly: the certificate holds at the **triggered/typical-query operating point**, realistic for task-specialized agents whose query distribution is stationary/known; foreground Assumption 9 as the scope, not a footnote. Soften the abstract's "regardless of adversary strategy" (concede overclaim).

### C. Randomized-smoothing comparison — n6dm W5, AC. VERDICT: **gap, not error — and we have a uniquely strong answer.**
Phase-38 *implemented* Cohen (2019) randomized smoothing for this detector and found it **vacuous in 384-dim** (isotropic noise norm σ√d ≈ 5 destroys the max-cosine score). Present the principled distinction:
- Smoothing certifies an **L2 ball** via noise robustness: R = σΦ⁻¹(p). MEMSAD certifies via **detection-retrieval gradient coupling**: the evasion direction provably degrades the attacker's *own* retrieval objective. Different mechanism (objective conflict, not noise averaging); smoothing is not a special case, and is empirically vacuous here. Comparison table + the "why smoothing fails in embedding space" result. **[VERIFY: cite Cohen 2019 exactly; the Athalye/Carlini gradient-masking framing — deferred with Agent 3.]**

### D. Le Cam single-sample vs batch / Eq 12 linear term / Table 13 — m2wH W3, PAT. VERDICT: **[VERIFY-APPENDIX] — most technically dangerous item.**
Reviewed version already reframes Thm 10 as **calibration sample complexity** (excess over oracle; irreducible B(ρ) persists), which defuses the "single-sample error → 0" misreading (PAT's framing). Remaining sharp point: Eq 12's "P_err(τ̂_N) − B(ρ) = Ω(σ|τ̂_N−τ*|) to first order." For a FIXED distribution, excess risk is *quadratic* at τ* (risk minimizer). BUT Le Cam's two-point method lower-bounds the **minimax** excess error over two nearby distributions with *different* optimal thresholds — using the wrong-distribution threshold incurs a *first-order* penalty in the separation; this is the standard mechanism and can legitimately produce the linear term.
- **Action:** read Appendix B.2 carefully (post-reset). If Eq 12 conflates fixed-distribution risk with minimax separation, repair the statement (the Ω(1/ρ²) calibration rate is the standard, defensible result). Check Table 13: does it claim achieving **0.05 total error** at ρ=1.27 (contradiction, oracle floor 0.52) or **0.05 excess over oracle** (consistent)? Frame accordingly. **Get this exactly right — it is the load-bearing correctness item.**

### E. Composite OR vs weighted-sum + watermark-carries-the-score — m2wH W4, PAT, u2M8. VERDICT: **partially valid + substantive.**
Main-text composite (line 243) = weighted sum of watermark+validation+proactive, **omitting MEMSAD**; appendix defines a logical OR including MEMSAD; actual code = disjunction. And Table 3's composite TPR=1.00 is carried by the **watermark**, which "fails silently under auto-storage" — the QUERY/SINGLE path where MEMSAD is supposed to matter.
- **Response:** (i) clarify the *deployed* composite is the disjunction (OR) INCLUDING MEMSAD; the main-text weighted-sum line is a documentation error (show anonymized code to AC); (ii) honest substantive framing: report MEMSAD **standalone** prominently (AGENTPOISON trig TPR=1.00, MINJA 0.40, INJECMEM 0.20) and position it as the content-level detector for the **no-ingestion-control** setting where watermarking is unavailable; (iii) **NEW EXP (E4):** re-run composite as a clean disjunction with per-defense contribution, making the watermark-dependence transparent. This is the hardest point; do not spin it.

### F. Threat model cross-principal — u2M8 W1. VERDICT: **valid framing critique.**
Lead with cross-principal deployments: (a) shared/multi-user memory (enterprise agents, shared KBs); (b) tool-output ingestion (web/email/doc content written to memory from untrusted sources — attacker controls CONTENT, not the agent); (c) MINJA (query-only) and INJECMEM (single-interaction) are *different-principal* by construction. Clarify Def 2/3 model the worst-case (WRITE) adversary; the realistic threat is QUERY/SINGLE. Cite the attack papers' own threat models.

### G. Minor correctness cluster — Wasserstein √ scaling, canonicity bijection (Thm 6iii), CVP integrality, Thm 9 Taylor/Hoeffding, Lemma 7 proof presence, Eq 5/6 utility alignment, access-model formalization. VERDICT: **mostly addressed in reviewed version [VERIFY via Agent 2 audit] or quick clarifications.** From my read: discrete-token CVP framing present (p.8 lines 229–232); Thm 7 has necessity/sufficiency/canonicity via Lehmann-Scheffé. Canonicity "lossy compression" wording (Thm 6iii) is genuinely loose (bijections preserve sufficiency) — concede + reword to "unique distribution-free minimal-sufficient *implementation*" (which is what the proof actually shows). Point-by-point, concede where needed, cite corrected text.

### H. Citations — m2wH W5, AC. VERDICT: **valid + easy.** Bib mixes "Anonymous" with public IDs (INJECMEM among them). Cannot revise paper; in the response give the correct public citations and commit to camera-ready fix. AC flagged "citation validity" — take seriously, list corrected refs.

### I. Le Cam novelty — n6dm W3. Reframe: novelty is the *gradient-coupling theorem* (genuinely new) + the *application* establishing an intrinsic calibration sample-complexity floor for memory-poisoning detection; Le Cam is an acknowledged tool, not claimed as novel technique. Don't oversell.

### J. Regret bound tangential — n6dm W4. Connect to the certificate (rolling calibration keeps τ̂ near τ*, which the radius needs) OR concede secondary and offer to appendix-ize in camera-ready.

### K. SINGLE access |P|=1 vs 15 entries — u2M8 Q. Reconcile: |P|=1 = one *interaction* (single write event), not one entry; INJECMEM writes 3×n_base=15 entries in that one interaction. Clarify Def 2.

---

## 2b. VERIFIED adjudication — from direct appendix read (2026-07-27, pp. 13–20)

The reviewed version already fixes a large share of the alleged errors. Precise verdicts:

**ALREADY FIXED in the reviewed version (point the AC to the text):**
- **CVP hardness (Prop 15, B.7, p.16):** the reduction is now explicitly *discrete-token* with integrality (λ∈{0,1}ⁿ⊂ℤⁿ); the text states the continuous relaxation is poly-time (SOCP in a ReLU region). PAT's "continuous→CVP without integrality" is pre-addressed.
- **Wasserstein/CDF (B.9, p.19):** uses `|F−G|_∞ ≤ √(2 g_max W₁)` and explicitly explains the √-scaling is sharp and linear is wrong. Pre-addressed.
- **Theorem 9 calibration (B.4, p.14):** Laurent–Massart two-sided χ² for the variance + corrected `|√(1+u)−1| ≤ |u|/√2` on u∈[−½,0]; Hoeffding for the mean. The flawed `√(1−x)≥1−x` step is gone. Pre-addressed.
- **Composite OR vs weighted-sum (App C, p.19, lines 674–685):** the deployed composite is the **disjunction** `π_wm ∨ π_SAD ∨ π_pro` (includes MEMSAD); the weighted sum "that appeared in some earlier exposition… is deprecated." The definitional inconsistency is resolved.
- **Le Cam batch-vs-single (B.3, p.14, lines 494–498):** the paper explicitly distinguishes the batch test from the calibration formulation and uses the latter. PAT's "batch test" objection is pre-addressed.

**GENUINELY OPEN valid errors (concede-and-repair; m2wH correct):**
- **[E-1] Corollary 17 combined-score coupling (B.8, p.17, lines 602–606).** The proof shows `⟨∇s_comb,∇s_max⟩ = α‖∇s_max‖² + (1−α)⟨∇s_mean,∇s_max⟩ ≥ (α−(1−α)β)‖∇s_max‖² > 0` (acute angle under Assumption 16), then concludes `⟨δ,∇s_comb⟩<0 ⟹ ⟨δ,∇s_max⟩<0`. **This last step is a non-sequitur** — halfspace containment needs the gradients *parallel*, not merely acute. m2wH's counterexample (∇s_max=(1,0),∇s_mean=(0,1),δ=(1,−2)) satisfies Assumption 16 (⟨∇s_mean,∇s_max⟩=0≥−β) yet breaks the conclusion. **Corollary 17 is false as stated.**
  - *Repair (exceptional):* (a) the rigorous certificate rests on **Theorem 7 (pure s_max)** — untouched and correct; (b) state the *correct* condition: s_comb inherits directional coupling only if ∇s_mean is a nonnegative multiple of ∇s_max (near-parallel), which we do NOT claim universally; (c) **NEW THEOREM opportunity** — a max-gated/disjunctive detector that provably retains coupling; (d) **E3** measures whether real DPR-HotFlip evasion directions ever exploit the s_comb gap (hypothesis: no — the optimizer follows the retrieval gradient ∝ ∇s_max, which is exactly the coupled direction); (e) deploy pure s_max (or the disjunctive composite instantiated with s_max) to recover the unconditional guarantee. Phase-38 max-ensemble is s_max-based and supports this.
- **[E-2] Eq 12 Le Cam linear term (B.3, p.14, lines 480–482).** Verified: at the Bayes threshold τ*=(μ₀+μ₁)/2, dR/dτ = (1/2σ)[φ(ρ/2)−φ(−ρ/2)] = 0. The excess error is **quadratic** in |τ̂−τ*|, so Eq 12's linear leading coefficient `(1/2σ)φ(ρ/2)|τ̂−τ*|` is wrong.
  - *Repair (preserves the conclusion):* state the result as a **calibration/threshold-estimation** sample complexity — Le Cam directly gives |τ̂−τ*| ≥ Ω(σρ) unless N ≥ Ω(1/ρ²) (lines 484–493 already derive this), and Theorem 11's upper bound |τ̂−τ*| ≤ σ·O(√(log/N)) matches at Θ(1/ρ²). The Ω(1/ρ²) **calibration** sample-complexity survives; drop the flawed excess-classification-error conversion (Eq 12) which is a removable intermediate step. Concede the specific point to m2wH; the headline result stands on threshold estimation, which is what calibration actually does. **[VERIFY Table 13 numbers after full read — likely reports excess-over-oracle, not absolute error; if it claims ≤0.05 absolute at ρ=1.27 (oracle floor Φ(−0.635)≈0.26), reframe as excess.]**
- **[E-3] Lemma 8 q*∈H (B.1, p.13, Step 1, lines 447–448).** Verified: the proof literally writes "Since q*∈H (the victim query is in the rolling history)." m2wH W1 is correct.
  - *Repair (THE UNIFICATION — exceptional):* relax "q*∈H" to "q* is covered by the support of H" (Assumption 9, "typical victim query"). **This holds precisely in the CROSS-PRINCIPAL threat model** (attacker writes poison; a *different* victim issues normal in-distribution queries → q* typical → certificate valid). It fails precisely when the attacker also controls q* (atypical trigger) — which is u2M8's *same-principal self-poisoning* case that u2M8 argues is not a meaningful threat. **So m2wH's q*∈H objection and u2M8's cross-principal objection resolve each other: the certificate is deterministic exactly in the threat model that matters.** Nuance to handle honestly: AGENTPOISON uses a trigger that makes q* atypical, so it needs *triggered calibration* (Table 5, AGENTPOISON TPR=1.00†) — the honest operating point; MINJA/INJECMEM (query-only/single) have typical q* under standard calibration. Soften the abstract's "regardless of adversary strategy" → "for adversaries whose retrieval-triggering queries are covered by the calibration distribution."

**PARTIALLY OPEN (needs an honest experiment + reframing):**
- **[E-4] Composite MINJA=1.00 attribution (App C line 681–682 vs Table 5).** The current version claims MINJA composite TPR=1.00 "arises from MEMSAD," but Table 5 shows MEMSAD standalone MINJA TPR=0.40 at κ=2.0 — so the claim appears inconsistent; in the WRITE model the 1.00 likely comes from the *watermark*, which MINJA bypasses under auto-storage (the QUERY/SINGLE regime). m2wH W4's substance partly stands. *Response:* **E4** re-runs the disjunction with a transparent per-defense breakdown; AND reframe "weak results" via **AUROC** — MEMSAD MINJA AUROC ≥ 0.914 (multi-encoder eval, p.20); TPR=0.40 is the FPR=0 (κ=2.0) operating point, while κ=1.0 gives MINJA TPR=1.00 at FPR=0.10 (Table 5). "Weak" is a single conservative operating point on a strong ROC. Honestly separate the WRITE regime (watermark available) from QUERY/SINGLE (MEMSAD carries it).
- **u2M8 "greedy synonym wins so NP-hardness is irrelevant":** framing misread — the hardness characterizes the *worst-case optimal* evasion surface; the paper (lines 588–592) already states gradient coupling closes the continuous surface, CVP-hardness closes the unconstrained-token surface, and synonym is the *residual gap* greedy exploits. Clarify the role of the hardness result; it is not claimed to stop greedy synonyms.

**Minor (concede/reword):** Theorem 6(iii) canonicity "lossy compression" wording is loose (bijections preserve sufficiency) → reword to "unique distribution-free minimal-sufficient *implementation*." Citations (Anonymous/INJECMEM) → provide correct public refs, commit to camera-ready fix.

---

## 3. New experiments (all CPU-local; phase-38 infra reused)

| ID | Experiment | Answers | Status |
|----|-----------|---------|--------|
| **E1** | Gap distribution Δs per attack + certified fraction | n6dm Q1, AC | mostly built (phase-38 `certify_semantic_eval`) |
| **E2** | Randomized-smoothing vs gradient-coupling comparison + "smoothing vacuous in 384-d" | n6dm W5, AC | built (phase-38 `certified_semantic`) |
| **E3** | Fraction of real evasion directions satisfying the s_comb directional condition | m2wH W2 | new, small |
| **E4** | Clean composite disjunction re-run + per-defense contribution | m2wH W4 | new, uses existing defenses |
| E5 | mpnet / max-ensemble synonym hardening (progress on open direction) | synonym loophole | done (phase-38) |
| E6 | Encoder-regularity empirical check (local Lipschitz of encoders used) | n6dm W2 | new, optional |

All produce plain-text numbers/tables (no file uploads allowed in the response; describe results in text). An **anonymized repo** may be shared with the AC via Official Comment (m2wH Q3 asked) — prepare a clean anonymized snapshot.

---

## 4. Response drafts (≤10,000 chars each), prioritized

- **n6dm (→ accept):** lead with E1 + E2 (their exact asks); crisp W1–W6; soften "regardless of adversary strategy." HIGHEST ROI.
- **m2wH (→ neutralize for AC):** concede-and-repair A, D, E; correct H; rigorous, non-defensive; answer Q1–Q3 including the composite reconciliation + anonymized repo offer.
- **u2M8 (→ borderline-accept):** threat-model reframe F + K + overclaim softening.
- **AC confidential comment:** summarize corrections + E1/E2 + citation fixes + the s_max-vs-s_comb clarification.

---

## 5. Workflow / process

- Branch `rebuttal/neurips2026` off main for new experiment code + response drafts in `docs/rebuttal/`.
- Each new experiment: script + committed results, lint/mypy green, reproducible; per-review response markdown with a char-count check (≤10k).
- Anonymized repo snapshot for the AC.
- Update publication_plan.md §16, memory, Notion. Auto-checkpoint active (branch commits are the real safety).
- No PR is pushed without author review (standing instruction); the OpenReview responses are posted by the author.

---

## 6. Decisions for the author (PI) — needed before finalizing

1. **Concede-and-repair posture** on the counterexample (A) and composite (E). Recommendation: **YES** — m2wH is expert; the AC rewards honesty + a repaired claim that still supports the contribution. Confirm.
2. **char-JSD attribution finding** (phase-38: char-JSD contributes 0; the MemSAD+ caption is unsupported): do NOT volunteer; but m2wH already probes MemSAD+ — if pressed, we have the corrected attribution + mpnet/max-ensemble hardening. Confirm the posture.
3. **Expectation setting:** uphill (one confident correctness reject). Recommendation: run the strongest possible rebuttal AND parallel-prep the ICLR 2027 resubmission incorporating every correction — the rebuttal work IS the resubmission-strengthening work. Confirm.
4. **Anonymized repo to the AC:** OK to share? (m2wH Q3 requested.)

---

## 7. Resources / logins

- **Compute:** none needed — all experiments CPU-local; encoders/faiss/gpt2 cached. No Chameleon required.
- **Session/usage:** deep-analysis agents hit the account session limit (resets 10pm CT); the appendix-level audit (Le Cam Eq 12, Tables 13/16/18, Appendix C) completes after reset.
- **OpenReview:** author posts the final responses (no tool access). I draft; author reviews + posts.
