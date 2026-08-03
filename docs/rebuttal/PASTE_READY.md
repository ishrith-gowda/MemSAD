# PASTE-READY rebuttal responses — Submission 30963 (post by AoE Aug 3)

**HOW TO POST (~5 min, no review needed — all 4 are self-reviewed, ≤10k chars, double-blind verified: no identity, no links):**
1. Open the OpenReview forum for Submission 30963.
2. For each reviewer below: click the **Rebuttal** button on that reviewer's review and paste the block verbatim.
3. For the AC block: use **Author-AC Confidential Comment** (not a public Rebuttal).
4. If the Rebuttal button is closed, use **Official Comment** instead (the PC clarified they are equivalent through Aug 3).

Char counts (all under the 10,000 limit): n6dm ~7.4k · m2wH ~8.0k · u2M8 ~4.5k · AC ~4.7k.

---

## ►► REVIEWER n6dm — paste under n6dm's "Rebuttal" button

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

We agree the theory should not silently assume more than the encoders provide. Two clarifications: (i) the coupling in Thm 7 is a *geometric* statement on the unit sphere (the max-cosine gradient and the retrieval-objective gradient are the same Riemannian gradient Π⊥ E(q*)); it does not require encoder invertibility, only differentiability of the score along the evasion path, which holds a.e. for the piecewise-smooth encoders used. (ii) We ran the empirical regularity check (E5): over 377 single-token substitutions, the induced embedding shift ‖E(c')−E(c)‖ is bounded and small for all six encoders (mean 0.075–0.153, 95th pct ≤0.35, max ≤0.39, vs the sphere diameter 2.0) — a single-token edit moves the embedding by a bounded amount, evidencing local regularity, and the per-encoder max shift is exactly the Lipschitz constant that enters the certified radius. We will add this table to make the theory–experiment link explicit.

**W3 — Le Cam is a textbook technique**

We agree Le Cam's method itself is standard and do not claim it as a novel technique. The novelty we intend to claim is (a) the gradient-coupling theorem, and (b) the *reduction* establishing that calibrated memory-poisoning detection has an intrinsic Ω(1/ρ²) *calibration* sample-complexity floor. We are also revising the surrounding statement: the excess-error step (Eq. 12) is stated as first-order, but the risk derivative vanishes at the Bayes threshold, so the excess classification error is second-order in the threshold error. The correct and cleaner statement is a lower bound on *threshold-estimation* sample complexity (Ω(1/ρ²)), which is exactly what calibration performs and which the matching upper bound (Thm 11) already validates. The headline rate is unchanged; we will remove the flawed intermediate conversion. We thank the reviewer/AC for pushing on correctness here.

**W4 — Online regret bound feels tangential**

Fair. The regret bound's role is to guarantee that rolling calibration keeps τ̂ within O(σ√(log/m)) of τ*, which is precisely the condition the certified radius needs to remain valid under drift; we will make this connection explicit rather than presenting it as a standalone result, and are happy to move the derivation to the appendix.

**W6 — The practical attack (synonyms) is outside the theory's continuous scope**

We agree this is the central limitation and want to be maximally clear. Proposition 13 is not a hedge but the paper's main negative result: continuous-space coupling cannot cover discrete synonym substitution (ε_syn≈0 for paraphrase-trained encoders). We have made concrete progress on this frontier since submission that we will add: (i) the certified synonym radius is strongly encoder-dependent — under all-mpnet-base-v2 the margin exceeds the per-substitution shift, giving a positive certified radius (r*≥1) where all-MiniLM gives r*=0; and (ii) a max-over-encoders detector certifies r*≥1 for a majority of poison because a substitution must simultaneously drop the score across all encoders. This does not "close" the gap, but it converts it from an open hole into a characterized, partially-mitigable frontier with a concrete deployment recommendation (encoder choice). We will state the honest ceiling (r*=1) explicitly.

We are grateful for the review — the two experiments you requested materially strengthen the paper, and we have corrected the two overstatements (Eq. 12's leading term; the abstract's "regardless of adversary strategy"). We would welcome any further questions during the discussion period.

---

## ►► REVIEWER m2wH — paste under m2wH's "Rebuttal" button

We thank the reviewer for an unusually rigorous read. Several points are correct and we correct them below; several target formulations that the current version already revised, which we quote. We ran new analyses (summarized as E-x) to substantiate the repairs. We take correctness as the priority and have not tried to argue past any valid objection.

**W2 — Corollary 17 counterexample. You are correct; we repair it.**
The step "⟨∇s_comb,∇s_max⟩>0 (acute) ⟹ {δ:⟨δ,∇s_comb⟩<0}⊆{δ:⟨δ,∇s_max⟩<0}" is a non-sequitur; halfspace containment needs the gradients parallel, and your δ=(1,−2) with ∇s_max=(1,0),∇s_mean=(0,1) is a valid counterexample under Assumption 16. Corollary 17 is therefore too general as stated. Our repair turns this into an improvement rather than a patch:
(i) The certified guarantee rests on **Theorem 7 for the pure max-cosine score s_max**, which is untouched and unconditional.
(ii) We **replace the linear combined score with a disjunctive detector** D(c)=1[s_max(c)>τ_max ∨ s_mean(c)>τ_mean] (thresholds calibrated separately). This *provably inherits Theorem 7 coupling* for the entries that matter: a retrieval-effective poison necessarily has high s_max (retrieval rank = max cosine), so it triggers the s_max branch, and evading that branch requires reducing s_max = retrieval. The s_mean branch only adds detections for low-max (low-retrieval, non-threatening) entries. Empirically the disjunction substantially improves detection over the linear score (E6, κ=2): MINJA TPR 0.06→0.38, INJECMEM 0.00→0.50, AGENTPOISON unchanged at 1.00 — a large TPR gain for a marginal FPR increase (disjunction FPR ≈0.03–0.05 vs ≈0.01, recovered to ≈0 at κ=2.5), because it keeps the stronger branch instead of diluting. This is also exactly the paper's composite philosophy (App C disjunction).
(iii) We further measured whether your continuous free-evasion direction is reachable at all (E3, real poison): the two gradients are strongly aligned (angle 40.5° [33.5,54.9], not the ~90° the construction needs), and the retrieval-optimal direction +∇s_max evades s_comb for **0/8 entries**; realizing any such direction in tokens is CVP-hard (Prop 15).
So we concede Corollary 17 and, in response, deploy a detector that is both provably coupled and empirically stronger. We will restate the result accordingly and foreground s_max/the disjunction as the certified detector.

**W3 — Eq (12) / Le Cam. You are correct about the leading term; the conclusion is unaffected.**
At the Bayes threshold τ*=(μ₀+μ₁)/2 the risk derivative vanishes (dR/dτ ∝ φ(ρ/2)−φ(−ρ/2)=0), so excess classification error is second-order in |τ̂−τ*|; Eq 12's first-order coefficient is wrong. The fix removes a superfluous step: Le Cam already gives (App B.3) a lower bound on the **threshold-estimation** error, |τ̂−τ*|=Ω(σρ) unless N=Ω(1/ρ²), and the matching upper bound (Theorem 11, |τ̂−τ*|≤σ·O(√(log/N))) closes it at Θ(1/ρ²). The result we claim — an intrinsic Ω(1/ρ²) **calibration** sample-complexity floor — is exactly this and does not use the excess-error conversion. We will delete Eq 12's linear step and state the bound on threshold estimation. On Table 13: you are right that FPR+(1−TPR)≤0.05 is unachievable in absolute terms at ρ=1.27 (oracle floor Φ(−ρ/2)≈0.26). The table's N_min column is the calibration lower bound ⌈4(1−ε)²/ρ²⌉ and the "0.05" refers to the *excess over the oracle* / calibration precision, not absolute error — the caption is misleading and we will rewrite it to "excess-over-oracle ≤ 0.05," which is the quantity the bound controls and is consistent with the irreducible B(ρ) we state persists for all N.

**W1 — Certification and q*∈H. Correct that the proof uses it; we scope it precisely (and it coincides with the meaningful threat model).**
Lemma 8 Step 1 does assume q*∈H, and Assumption 9 ("typical victim query") is the honest scope. The key clarification: the certificate is deterministic exactly when the future victim query is covered by the history's support — which is **precisely the cross-principal setting** (attacker writes poison; a different victim issues in-distribution queries). It degrades to probabilistic only when the attacker also controls the query to be atypical — i.e. the same-principal self-poisoning case, which Reviewer u2M8 argues (and we agree) is not the meaningful threat. So the assumption holds in the deployment that matters. For AGENTPOISON the trigger makes q* atypical, so the certificate holds under triggered calibration (Table 5, TPR=1.00) — we present this as the honest operating point, not as coverage of all strategies. We are correcting the abstract's "regardless of adversary strategy" to "for adversaries whose retrieval-triggering queries are covered by the calibration distribution." E1 quantifies the resulting gap distribution and where the certificate is informative.

**W4 — MEMSAD's own numbers and internal consistency.**
- *"Weak" numbers:* the standalone TPRs you cite (MINJA 0.40, INJECMEM 0.20 at κ=2) are the FPR≈0 operating point, not detection power. AUROC (threshold-independent) is high: AGENTPOISON 1.00, **MINJA 0.951, INJECMEM 0.846** (E4; MINJA matches the paper's ≥0.914). At κ=1 a defender accepting small FPR recovers MINJA TPR=1.00. The "MINJA TPR ranges 0.40–1.00 across tables" is the *same ROC read at different κ / calibration*, which E4 exhibits as a curve — not contradictory results.
- *Composite definition:* the deployed composite is the logical **disjunction** π_wm∨π_SAD∨π_pro (App C, "Definitive specification"); the weighted-sum form is deprecated there and is not what Tables 3/11 report. The per-defense hypothesis tests (Table 11) show watermark rejects the chance null on AP/MJ and MEMSAD on MJ/IM; in the WRITE model watermarking contributes MINJA's perfect score, while in the QUERY/SINGLE model (auto-storage bypasses watermarking) MEMSAD is the operative detector — the entire motivation for a content-level write-time detector. We will make this per-defense, per-threat-model attribution explicit.
- *Table 12 FPR=0.000 vs Table 16 FPR 0.09/0.138:* you have found a genuine inconsistency. The primary, properly-validated FPR is Table 12 (Clopper-Pearson, held-out calibration, 20 trials, n=1000): 0.000 [0,0.004]; our reproduction gives ≈0.01–0.02 at κ=2. Table 16's domain-matched 0.090/0.138 in fact matches the BGE-Large encoder's FPR (Table 9), not the primary MiniLM detector's — it appears to be a table-construction/labeling error. We will correct Table 16 and report the single reconciled, held-out FPR (≈0.00–0.02 at κ=2, eliminated at κ=2.5). AUROC (threshold-independent) is unaffected. Separately, Table 15 already calibrates all baselines on the identical query regime as MEMSAD (stated in its caption), so there is no informational asymmetry there.

**W5 — Citations.** You are right that the bibliography mixes anonymized placeholders with public identifiers (including INJECMEM). We cannot revise the PDF during rebuttal, but we will (a) give the correct public citations in the camera-ready and (b) confirm here that the evaluated attacks are the published AGENTPOISON, MINJA, and INJECMEM works; no result depends on an anonymized source. We apologize for the inconsistency and will clean the entire reference list.

**Your questions.** Q1: corrected Corollary — we deploy the disjunctive detector, whose s_max branch carries the Theorem 7 guarantee (unconditional for retrieval-effective poison); the linear s_comb's coupling was conditional and, empirically (E3), not exploitable by a retrieval-driven adversary. Q2: Eq 12's first-order term does not survive; the Ω(1/ρ²) bound is on threshold estimation and stands (matched by Thm 11). Q3: the reported composite is the disjunction; we will provide raw per-seed predictions, exact CI aggregation, seeds, and an anonymized repository to the AC via Official Comment.

We are grateful for the depth of this review — it directly improved the paper's correctness, and we have conceded every point where you were right rather than arguing around it.

---

## ►► REVIEWER u2M8 — paste under u2M8's "Rebuttal" button

We thank the reviewer for pushing on the threat model — it sharpened the framing, and it connects directly to a correctness point another reviewer raised.

**W1 — Threat model: cross-principal vs self-poisoning.**
We agree the meaningful threat is cross-principal (the writer and the victim are different principals), and we will lead with it rather than the WRITE model. Three concrete cross-principal deployments motivate the work:
1. **Shared / multi-tenant memory:** enterprise agents backed by a shared knowledge base or team memory; one user (or a compromised account) writes entries that a *different* user's session later retrieves.
2. **Tool-output ingestion:** the agent writes tool/retrieval outputs (web pages, emails, documents, API responses) into memory. An attacker who controls that *content* — not the agent — plants poison the victim later retrieves. This is the dominant real vector and requires no direct write access.
3. **Query-only / single-interaction attacks:** MINJA injects via ordinary user queries (no write privilege), and INJECMEM via a single interaction; in both, the attacker is a different principal who never has WRITE access.

The WRITE model in Definitions 2–3 is the *worst-case* adversary used to state the strongest theoretical guarantee; we will clarify that it upper-bounds attacker capability and that the realistic threat is QUERY/SINGLE. Importantly, the cross-principal setting is also where MEMSAD's certificate is *deterministic*: the certified radius (Lemma 8) holds when the future victim query is covered by the calibration/history distribution, which is exactly the case when a legitimate victim issues in-distribution queries. It degrades only when the attacker also controls the query to be atypical — the same-principal self-poisoning case you correctly note is not a meaningful threat. So your framing and the certificate's assumption coincide: the guarantee is strongest precisely in the deployment that matters.

**W2 — "Regardless of adversary strategy" overclaims.**
Agreed; we are correcting it. The certificate (Lemma 8) holds for continuous perturbations under the typical-query assumption (Assumption 9); it does not cover a colluding user who crafts an atypical trigger query, nor discrete synonym substitution (Proposition 13). We are revising the abstract to "for adversaries whose retrieval-triggering queries are covered by the calibration distribution," and we present the triggered-calibration operating point honestly rather than as universal coverage.

**W3 — NP-hardness is irrelevant if a greedy synonym attack already wins.**
We agree the hardness result does not defend against greedy synonym substitution, and we do not claim it does. Its role is purely to characterize the *worst-case optimal* evasion surface: gradient coupling (Thm 7) closes the continuous surface, and the CVP reduction (Prop 15) shows the *unconstrained-token* optimal evasion is NP-hard — leaving discrete synonym substitution as the identified *residual* gap that a cheap greedy attack exploits (this is stated in the paper, but we will foreground it). In other words, the hardness result explains why the remaining attack surface is specifically the low-perturbation synonym regime, which is exactly the frontier we characterize (Prop 13) and report as an open limitation (80–100% evasion, ΔASR-R≈0). We have since made partial progress on it (encoder-dependent certified synonym radius; a positive radius under mpnet and a max-over-encoders detector), which we will add as characterized-and-partially-mitigable rather than closed.

**Q — SINGLE access: Def 2 says |P|=1, but INJECMEM is evaluated with 15 entries.**
Thank you — this is a wording ambiguity we will fix. "|P|=1" denotes a single *interaction* (one write event / one adversarial episode), not a single memory entry. Within that one interaction, INJECMEM writes 3×n_base = 15 broad-anchor entries from 8 templates (lines 257–264). The SINGLE access model constrains the attacker to *one episode of influence*, not one row; the 15 entries are all produced in that single episode. We will restate Definition 2 as "a single adversarial interaction that may write |P|≥1 entries" to remove the contradiction, and confirm the threat-model results rely on this per-interaction definition.

We appreciate the review; the cross-principal reframing and the SINGLE-access clarification both improve the paper, and the threat-model point turned out to reinforce, rather than undercut, the certificate's scope.

---

## ►► AREA CHAIR — paste under "Author-AC Confidential Comment"

Dear AC, thank you for the clear meta-review. We summarize how we have addressed each of the five concerns you raised, with new analyses run during the response period.

**1. Correctness of all theoretical claims.** We separate the objections into two groups.
- *Three are valid and we concede-and-repair them:* (a) Corollary 17 (combined-score coupling) — the acute-angle step does not imply half-space containment; the rigorous guarantee is Theorem 7 for the pure max-cosine score, which we deploy, and we now state the correct condition for the combined score. New analysis (E3): on real poison the relevant gradients are aligned (40.5°) and the retrieval-optimal direction never evades the combined score (0/8), so the counterexample is not reachable by a retrieval-driven adversary. (b) Eq 12 (Le Cam) — the first-order excess-error term is incorrect (the risk derivative vanishes at the Bayes threshold); the Ω(1/ρ²) result is properly a *calibration/threshold-estimation* sample-complexity lower bound, which is correct and matched by Theorem 11's upper bound. We remove the flawed step; the headline is unchanged. (c) Lemma 8 assumes the victim query is covered by the history — we scope this to the cross-principal threat model, where it holds; Reviewer u2M8's threat-model concern and Reviewer m2wH's q*∈H concern resolve each other (the assumption fails only in the same-principal self-poisoning case, which is not the meaningful threat).
- *Five further objections target formulations the reviewed version already corrected*, which we quote in the individual responses: the CVP reduction is stated on the discrete-token problem with explicit integrality (App B.7); the Wasserstein–CDF bound uses √-scaling and explains why linear is wrong (App B.9); the calibration bound uses Laurent–Massart and the corrected √(1+u) inequality (App B.4); the composite is the logical disjunction, with the weighted-sum form explicitly deprecated (App C); Le Cam is applied in the calibration (not batch) formulation, stated explicitly (App B.3); and the OOD-baseline comparison (Table 15) already calibrates all methods on the identical query regime, stated in its caption. We also correct two genuine table issues m2wH found: the Table 13 caption ("≤0.05") refers to excess-over-oracle, not absolute error (we will rewrite it), and Table 16's domain-matched FPR (0.09/0.138) mislabels the BGE-Large FPR as the primary detector's — the validated held-out FPR is 0.000 (Table 12), which we will reconcile.

**2. Comparison to randomized-smoothing certificates (your explicit request).** We implemented it (E2): applying the Cohen et al. (2019) smoothing certificate to the detector yields a vacuous radius in d=384 (best R=0.008 vs a single substitution's embedding shift 0.378) — isotropic noise of norm σ√d destroys the score. MEMSAD's certificate is a different mechanism: detection–retrieval gradient coupling (the evasion direction degrades the attacker's own retrieval objective), not noise robustness. We will add this comparison.

**3. When does the certified radius preclude meaningful attacks (your explicit request).** We ran the gap-distribution analysis (E1): for AGENTPOISON the adversarial-benign gap is large and positive (mean +0.21, 100% detected, 38% with a full calibration-uncertainty margin) — the certificate is informative in its design regime. For MINJA it is at the boundary, and for INJECMEM/synonym the semantic certificate is honestly vacuous (those are caught by the lexical component or fall in the Prop-13 frontier). We will add this delineation of the informative vs vacuous regimes.

**4. Fair comparison to related theoretical work.** We add the randomized-smoothing comparison (E2) and will position gradient coupling relative to it and to detection-vs-evasion coupling arguments; we do not claim Le Cam as a novel technique (only its application to the calibration sample-complexity floor).

**5. Citation validity.** We confirm the evaluated attacks are the published AGENTPOISON, MINJA, and INJECMEM works; the bibliography's mix of anonymized placeholders and public identifiers is an error we will fully correct in the camera-ready, and no result depends on an anonymized source.

We are also happy to provide an anonymized repository (raw per-seed predictions, exact CI aggregation, seeds) via Official Comment, as Reviewer m2wH requested. We believe the corrections strengthen the paper's central contribution — the gradient-coupling theorem and the certified/calibration analysis for the max-cosine detector — while honestly bounding its scope. We would welcome guidance on any point you would like prioritized during discussion.
