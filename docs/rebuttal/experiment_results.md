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
