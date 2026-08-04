<!-- DRAFT m2wH (2, conf 4). Plain text+markdown, no links, anonymity. <=10k chars. -->

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
