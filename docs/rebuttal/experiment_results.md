# Rebuttal experiment results — Submission 30963

Reproducible via the `src/scripts/rebuttal_e*.py` scripts (CPU-local, `all-MiniLM-L6-v2`, |M|=1000, κ=2, n=8 poison/attack). Numbers here are transcribed into the OpenReview responses (no file uploads permitted).

## E1 — certified-region gap distribution (reviewer n6dm, Q1)

Gap `g(c) = s_adv(c) − τ`, τ = μ̂ + κσ̂. `det%` = fraction with g>0 (detected); `cert%` = fraction with g > κσ̂ (a full calibration-uncertainty above threshold — robust to a bounded perturbation). AGENTPOISON at its triggered operating point (Table 5, TPR=1.00); MINJA/INJECMEM under standard calibration.

| attack | variant | mean g | p50 | min | max | det% | cert% |
|---|---|---|---|---|---|---|---|
| agentpoison | original | **+0.213** | +0.210 | +0.197 | +0.244 | **100** | 38 |
| agentpoison | synonym | −0.047 | −0.050 | −0.077 | −0.009 | 0 | 0 |
| minja | original | −0.020 | +0.002 | −0.137 | +0.084 | 50 | 0 |
| minja | synonym | −0.092 | −0.089 | −0.254 | +0.080 | 12 | 0 |
| injecmem | original | −0.181 | −0.204 | −0.270 | −0.045 | 0 | 0 |
| injecmem | synonym | −0.214 | −0.231 | −0.297 | −0.090 | 0 | 0 |

**Reading (the honest informative-vs-vacuous delineation n6dm requested):**
- **AGENTPOISON (WRITE-access, the attack MEMSAD is designed for):** the gap is large and positive (mean +0.21, min +0.20 — every entry above threshold), **100% detected, 38% comfortably certified** (g > κσ̂, a full calibration-uncertainty robustness margin). The certificate is **empirically informative** in its design regime.
- **MINJA:** gap ≈ 0 — the boundary/tight regime (semantic detection ~50%, matching Table 5's TPR=0.40 at κ=2). Certificate tight, not vacuous.
- **INJECMEM:** semantic gap is negative — the **semantic** certificate is vacuous here; INJECMEM is caught by the lexical/domain component (MemSAD+), not the semantic score. Consistent with the per-component attribution.
- **Synonym substitution:** drives the gap negative across all attacks (evasion) — the stated Proposition-13 frontier.

**Response framing:** the certified radius is informative precisely for the continuous/WRITE-access attacks it targets, and we honestly acknowledge it is tight (MINJA) or vacuous (INJECMEM-semantic, synonym) elsewhere — which coincides with the paper's stated scope. This directly answers "when is the certificate informative."

## E2 — randomized-smoothing comparison (reviewer n6dm W5 + AC)

Cohen (2019) randomized smoothing applied to the MEMSAD max-cosine detector on a *detected* AGENTPOISON entry (base score 0.783, gap +0.218). One synonym substitution's embedding shift for this entry: **δ_emb = 0.378**.

| σ | noise norm ≈ σ√d | p_lower | R_cohen | R ≥ δ_emb? |
|---|---|---|---|---|
| 0.05 | 0.98 | 0.561 | **0.008** | no |
| 0.10 | 1.96 | 0.000 | 0.000 | no |
| 0.25 | 4.90 | 0.000 | 0.000 | no |
| 0.50 | 9.80 | 0.000 | 0.000 | no |
| 1.00 | 19.60 | 0.000 | 0.000 | no |

**Reading:** at *no* σ does smoothing certify even one substitution's worth of embedding movement (best R=0.008 ≪ δ_emb=0.378). Small σ → tiny R; σ≥0.10 → isotropic noise of norm σ√384 destroys the max-cosine score, p_lower<½, R=0. **Randomized smoothing is vacuous for this detector in d=384.**

**Mechanistic distinction (the answer to the AC):** smoothing certifies an L2 ball via *noise robustness* (R=σΦ⁻¹(p)); MEMSAD certifies via *detection–retrieval gradient coupling* — the evasion direction provably degrades the attacker's own retrieval objective (Thm 7), a constraint on the *attacker's utility*, not a noise-averaging argument. Gradient coupling is not a special case of smoothing, and (E2) smoothing does not yield a usable certificate here. This is a genuine, favorable distinction — not a weakness.

## E3 — combined-score coupling: is the counterexample reachable? (reviewer m2wH W2)

We **concede Corollary 17 is stated too generally** (the acute-angle Assumption 16 does not imply halfspace containment — m2wH is correct). E3 quantifies whether a retrieval-driven attacker can reach the free-evasion wedge on real detected AGENTPOISON poison. On the unit sphere ∇s_max = Π_e(q\*) (argmax query), ∇s_mean = Π_e(q̄) (mean query).

| quantity | value | meaning |
|---|---|---|
| angle(∇s_max, ∇s_mean) | **40.5°** [33.5, 54.9] | counterexample needs ~90° (orthogonal); real gradients are acute |
| free-evasion wedge half-angle = angle(∇s_max, ∇s_comb) | 17.5° [14.5, 23.6] | narrow wedge |
| retrieval-optimal dir (+∇s_max) evades s_comb | **0/8** | the attacker's own objective never evades |

**Reading:** the counterexample is a valid *continuous-space* observation, but on real poison the two gradients are strongly positively aligned (40.5°, far from 90°), the free-evasion wedge is narrow, and the retrieval-optimal direction never evades s_comb (0/8). Repair: (i) the rigorous certificate is Theorem 7 (pure **s_max**, zero wedge, unconditional); (ii) s_comb is an empirical enhancement whose coupling we now correctly characterize as conditional; (iii) realizing any free-evasion direction in token space is CVP-hard (Prop 15). We deploy the disjunctive composite with the s_max-coupled component to retain the guarantee.
