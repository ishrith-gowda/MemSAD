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

## semantic certificate results (2026-07-05)

two certificates for the memsad semantic (max-cosine) detector were implemented
(`src/defenses/certified_semantic.py`, 11 tests) and evaluated on real
`all-MiniLM-L6-v2` embeddings of triggered-agentpoison poison
(`src/scripts/certify_semantic_eval.py`):

1. **randomized smoothing (cohen et al. 2019), embedding space.** RIGOROUS but
   **vacuous here**: in d=384, isotropic gaussian noise of any useful sigma has
   l2 norm ~sigma*sqrt(d) approx 5, which destroys the max-cosine score, so the
   smoothed detection probability collapses below 1/2 and R = 0 for every entry.
   this is a genuine negative result about naive RS in high-dimensional embedding
   space, not a bug (the certifier passes an exact linear-score anchor test).

2. **direct score-margin certificate.** r*(c) = floor((s(c) - tau) / delta_score),
   delta_score = max single-substitution score reduction. measured values:

   | entry | s(c)  | tau   | margin | delta_score | r* |
   |-------|-------|-------|--------|-------------|----|
   | p0    | 0.597 | 0.565 | 0.032  | 0.081       | 0  |
   | p2    | 0.572 | 0.565 | 0.007  | 0.094       | 0  |
   | p3    | 0.612 | 0.565 | 0.047  | 0.098       | 0  |

**central finding:** memsad's semantic detector operates within *one
substitution's worth of score shift* of its own threshold (margin ~0.03 vs
per-substitution shift ~0.09). the certified synonym radius is therefore 0: a
single synonym substitution suffices to evade. this rigorously quantifies the
paper's proposition 11 synonym loophole as a **margin-geometry** phenomenon
rather than an incidental gap -- and shows it cannot be closed by certification
alone without widening the operating margin (which trades off FPR).

**preliminary (needs a full adaptive-attack study to confirm):** aligning the
adaptive crafter to the *triggered* query set (rather than the untriggered
queries used in the paper's `run_memsad_plus_evaluation`) suggests a
trigger-aware adversary evades triggered-agentpoison with ~1 substitution,
whereas the paper reports triggered-agentpoison as synonym-robust (5/5). this
must be verified by re-scoring crafted passages against the full 100-query
detector before any claim is made.

**framing decision (resolved 2026-07-05): both -- characterize + attempt
widening.** paper-caption fix handled as a rebuttal / iclr-revision note (no
arxiv v2 change).

## multi-encoder + ensemble certified margin (2026-07-05)

`src/scripts/certified_ensemble_eval.py`; triggered-agentpoison poison, kappa=2,
score-margin certificate reused via `certified_radius_from_scores`.

| encoder                | flagged | mean margin | mean delta_score | median r* |
|------------------------|:-------:|:-----------:|:----------------:|:---------:|
| all-MiniLM-L6-v2 (paper default) | 5/5 | 0.078 | 0.100 | **0** |
| **all-mpnet-base-v2**  |   5/5   |    0.125    |      0.081        |  **1**    |
| paraphrase-MiniLM-L6-v2|   5/5   |    0.033    |      0.073        |   0       |
| bge-base-en-v1.5       |   0/5   |     --      |       --         |   --      |
| e5-base-v2             |   5/5   |    0.018    |      0.017        |  **1**    |
| contriever             |   0/5   |     --      |       --         |   --      |
| **mean-ensemble (6)**  |   5/5   |    0.050    |      0.048       |   0 (max 1) |

**positive result (defense recommendation):** certified synonym robustness is
strongly encoder-dependent. **all-mpnet-base-v2 achieves a positive certified
radius (median r* = 1) where the paper's default all-MiniLM-L6-v2 yields r* = 0**
-- its larger margin (0.125 vs 0.078) exceeds its per-substitution shift. swapping
the default encoder to mpnet is a concrete, certificate-backed hardening.

**ensemble aggregation comparison (fixed aggregations -> certificate carries over):**

| aggregation | flagged | mean margin | mean delta_score | median r* | max r* |
|-------------|:-------:|:-----------:|:----------------:|:---------:|:------:|
| mean        |   5/5   |    0.052    |      0.049       |    0      |   1    |
| **max**     |   5/5   |    0.018    |      0.017       |  **1**    |   1    |
| weighted (1/std) | 5/5 |   0.035    |      0.035       |    0      |   1    |

**widening result (principled):** the **max aggregation achieves median r* = 1
with full 5/5 coverage** -- it matches the best single encoder (mpnet) *without
needing to know a priori which encoder is best*, because a synonym substitution
must drop the maximum across ALL encoders simultaneously, collapsing the
per-substitution shift to delta_score = 0.017. this is the correct widener: the
mean dilutes (r* = 0) and weighting by inverse-variance does not help, but the max
recovers the best-encoder radius robustly. honest ceiling: r* = 1 (certifiably
robust to one substitution) -- the loophole is fundamentally tight; reaching
r* >= 2 needs a different detector or wider margins, not a different aggregation.

## ANN-scaling deployment study (2026-07-05)

`src/scripts/ann_scaling_eval.py`; faiss exact (IndexFlatIP) vs HNSW (M=32,
efSearch=64) vs IVF (nprobe<=16); poison inserted into the store; recall@5 =
fraction of victim queries whose top-k contains poison.

| \|M\| | index | poison recall@5 | cert margin |
|------|-------|:---------------:|:-----------:|
| 1000 | flat (exact) | 0.940 | 0.0802 |
| 1000 | hnsw | 0.940 | 0.0802 |
| 1000 | ivf  | 0.940 | 0.0802 |
| 10000| flat (exact) | 0.780 | 0.1099 |
| 10000| hnsw | 0.740 | 0.1099 |
| 10000| ivf  | 0.720 | 0.1099 |

**deployment findings:**
1. **the certified margin is exactly index-independent** (identical across
   flat/hnsw/ivf at each scale). the certificate is a *write-time* calibration
   property scored against the fixed victim-query set, so read-time ANN
   approximation cannot degrade it -- a clean deployment-robustness result.
2. **ANN does not incidentally mitigate the threat.** poison recall under HNSW
   (0.74) and IVF (0.72) tracks exact (0.78) at |M|=10k; poison crafted for
   maximal query similarity survives approximate retrieval. the small recall drop
   is the generic ANN recall gap (affecting benign and poison alike), not a
   poison-specific defense.

**caveat:** the 10k corpus is synthetically scaled from the 1k base by
jitter-resampling (the synthetic generator emits ~1k distinct entries), so the
*absolute* margin at 10k (0.110) reflects the resampled benign distribution and
should be read as illustrative; the *index-independence* and the *poison-recall*
comparisons are the robust conclusions. a genuinely diverse large corpus (e.g.
NQ / MS-MARCO passages) is the follow-up for camera-ready.

## remaining

scale to the 20-trial n=1000 protocol for Clopper-Pearson CIs on the fraction of
poison with r* >= 1 (certified-robust rate).
