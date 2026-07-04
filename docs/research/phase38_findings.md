# phase 38 findings — feature attribution of memsad+ and the semantic synonym gap

**date:** 2026-07-04
**branch:** `feat/phase-38-synonym-hardening`
**reproduce:** `PYTHONPATH=src python3 src/scripts/feature_attribution.py`

## summary

While designing a certified synonym-robustness radius for MemSAD+, a per-component
attribution of the detector (semantic / character n-gram JSD / domain-vocabulary
concentration) on the paper's own poison generators and calibration procedure
produced a result that changes the extension's direction and flags a caption-level
inaccuracy in the submitted paper (arXiv 2605.03482, appendix `app:memsad_plus`).

## attribution result (paper calibration: triggered for agentpoison, plain otherwise)

| attack / variant        | semantic | char-JSD | domain-vocab | any-of | paper table |
|-------------------------|:--------:|:--------:|:------------:|:------:|:-----------:|
| agentpoison (triggered) |   5/5    |  **0/5** |     0/5      |  5/5   | 1.00        |
| injecmem original       |   1/5    |  **0/5** |     2/5      |  3/5   | 0.60        |
| injecmem **synonym**    |   0/5    |  **0/5** |     2/5      |  2/5   | 0.40        |

n = 5 poison passages per cell; `all-MiniLM-L6-v2`; `|M|=1000`, kappa=2.0,
combined scoring, any-of detection; matches `run_memsad_plus_evaluation`.

## confirmed conclusions

1. **char-JSD contributes exactly zero detections** across every attack and
   variant. The `tab:memsad_plus` caption ("Character n-gram JSD features improve
   InjecMEM detection by 3x") is **not supported by the implementation**. The
   entire MemSAD+ lift (InjecMEM 0.20 -> 0.60 original, -> 0.40 synonym) is
   produced by the **domain-vocabulary concentration** component. On this corpus
   the poison is *less* char-divergent from the benign aggregate than the (short)
   benign entries themselves, so `D_char > tau_char` never fires on poison
   (0/15 poison flagged; 5/60 benign flagged — worse than chance).

2. **the synonym gap is a semantic-feature phenomenon.** InjecMEM semantic
   detection drops 1/5 -> 0/5 under synonym substitution; the residual 0.40 is
   held entirely by domain-vocab. agentpoison under triggered calibration is
   robust (5/5 -> 5/5).

## implications

- **for the certificate (this extension):** the certified radius must target the
  **semantic** feature (where the gap lives), not char-JSD. Plan: randomized
  smoothing (Cohen et al. 2019, already cited in the paper) in embedding space to
  obtain a certified L2 radius R(c), divided by the measured maximum
  per-substitution embedding shift delta_emb(c), yielding a certified synonym
  substitution radius r*(c) = floor(R(c) / delta_emb(c)). Clopper-Pearson
  (already in `evaluation/statistical.py`) supplies the confidence lower bound on
  the smoothed detection probability.
- **for the submitted paper:** the char-JSD attribution in `app:memsad_plus` is
  factually off. STATUS: awaiting author decision (rebuttal note vs arXiv v3
  correction). The char-JSD certificate framework built in
  `src/defenses/certified_synonym.py` remains a valid general-case contribution
  (it certifies char-level / unicode attacks where D_char *does* separate); it is
  simply not the driver for the synonym attacks studied here.

## caveats

- minja detection did not reproduce the paper's MemSAD TPR=1.00 under
  `MemSADPlusDetector` plain calibration (0/5 here). This is a
  `MemSADPlusDetector` vs `SemanticAnomalyDetector` query-handling difference
  (the latter uses a rolling query set), not central to the char/domain
  attribution. A full minja reconciliation is tracked as a follow-up.
- attribution is on n=5 per cell (matching the paper's appendix scale); the
  certified-radius experiments will use the larger bootstrap/Clopper-Pearson
  protocol (20 trials, n=1000).
