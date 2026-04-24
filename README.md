# MemSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents

Research code and evaluation framework for the NeurIPS 2026 submission

> **MemSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents**
> *Anonymous Author(s); under double-blind review.*

[![CI](https://github.com/ishrith-gowda/MemSAD/actions/workflows/ci.yml/badge.svg)](https://github.com/ishrith-gowda/MemSAD/actions/workflows/ci.yml)
[![Interactive demo](https://img.shields.io/badge/demo-live-brightgreen)](https://ishrith-gowda-memsad-demo.hf.space/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

LLM agents persist context across sessions through external memory systems (Mem0, A-MEM, MemGPT). This creates an attack surface: an adversary who can write to that memory can covertly influence future agent behavior — redirecting actions, exfiltrating data, or bypassing safety controls.

**MemSAD** is the first memory-poisoning defense with formal guarantees. It is a calibration-based, write-time anomaly detector whose threshold $\mu + \kappa\sigma$ is derived from a benign reference corpus. We prove four theorems about its behavior:

1. **Gradient coupling** (Theorem 1) — any monotone detector inherits the retrieval-loss gradient, so continuous evasion is impossible.
2. **Certified detection radius** (Lemma 2) — checkable, per-entry guarantee whenever the adversarial-benign similarity gap exceeds calibration uncertainty.
3. **Minimax optimality via Le Cam's method** (Theorem 3) — any threshold detector requires $\Omega(1/\rho^2)$ calibration samples; MemSAD achieves this up to $\log(1/\delta)$.
4. **Synonym-invariance loophole** (Proposition 11) — formal boundary characterizing where the continuous guarantees stop, motivating the hybrid MemSAD+ (character n-gram) extension.

Empirically, on a $3 \times 5$ attack-defense matrix with bootstrap CIs, Bonferroni-corrected hypothesis tests, and Clopper-Pearson validation (20 trials, $n=1{,}000$), the composite defense reaches $\text{TPR}=1.00$, $\text{FPR}=0.00$ across every attack. Synonym substitution, as Proposition 11 predicts, remains the only construction that evades: 80–100% evasion at $\Delta \text{ASR-R} \approx 0$.

---

## Key contributions

1. **Stackelberg formalization** of the memory-poisoning game with explicit leader–follower dynamics between the detector and a white-box adversary.
2. **Four formal guarantees**: gradient coupling, certified radius, Le Cam minimax lower bound, and the formal synonym-invariance loophole.
3. **MemSAD**, a write-time detector with $O(md)$ per-entry cost — $\sim 2$ ms/entry on commodity hardware.
4. **MemSAD+**, a hybrid detector adding character n-gram JSD features that partially close the synonym gap (InjecMEM TPR $0.00 \to 0.40$).
5. **Comprehensive empirical validation** across three attacks (AgentPoison, MINJA, InjecMEM) × five defenses (watermark, validation, proactive, composite, MemSAD), six sentence encoders, and a cross-corpus Natural Questions replication.
6. **Tool-use evaluation** with GPT-4o-mini confirming the threat is practical: ASR-A of 0.48 on triggered queries.
7. **Multi-agent SIR propagation** showing composite defense prevents cross-agent epidemic spread entirely.
8. **Production-quality artifact**: 593 tests passing; deterministic pipeline reproducing every table and figure in under 5 minutes on CPU.

---

## Interactive demo

A live SPA walkthrough of the framework — single-run retrieval, threshold sweep, attack×defense matrix, and artifact reproducibility details — is hosted at:

**https://ishrith-gowda-memsad-demo.hf.space/**

The frontend source lives under `src/frontend/` (React + Vite).

---

## Installation

Python 3.10–3.13 are supported. All experiments run on CPU; GPU optional for DPR-HotFlip and token-level watermark.

```bash
git clone https://github.com/ishrith-gowda/MemSAD
cd MemSAD
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Model downloads (automatic on first run):** `sentence-transformers/all-MiniLM-L6-v2` (90 MB) for the primary encoder; `facebook/dpr-ctx_encoder-single-nq-base` (440 MB) for DPR-HotFlip trigger optimization.

---

## Quick start

### Smoke test (~10 seconds)

```bash
python3 smoke_test.py
```

### Fast test mode (~3 minutes, corpus=15, reduced parameters)

```bash
MEMORY_SECURITY_TEST=true python3 -m pytest src/tests/ tests/ -q
```

### Full test suite (~30 minutes)

```bash
python3 -m pytest src/tests/ tests/ -q
```

---

## Reproducing paper results

### Full pipeline (all tables + figures)

```bash
python3 -m src.scripts.run_pipeline --mode quick   # ~5 minutes, reduced params
python3 -m src.scripts.run_pipeline --mode full    # ~60 minutes, paper params
```

Output is written to `pipeline_output/`.

### Individual figures

```bash
# sad roc curve + plain-vs-triggered calibration comparison
python3 -m src.scripts.generate_sad_figures

# phase 17 figures: dpr convergence, token watermark comparison, measured asr-a
python3 -m src.scripts.generate_phase17_figures

# phase 22 figures: multi-encoder cka heatmap, clopper-pearson fpr validation
python3 -m src.scripts.generate_phase22_figures

# phase 35 mechanism figures: gradient coupling, empirical trajectory, synonym loophole
python3 -m src.scripts.generate_phase35_figures
```

### Compile the paper

```bash
cd docs/neurips2026
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Pre-generated figures are checked into `docs/neurips2026/figures/` so the paper compiles without re-running experiments.

---

## Repository structure

```
src/
  attacks/
    implementations.py             - agentpoison, minja, injecmem
    adaptive_attack.py             - white-box adaptive adversary (synonym substitution)
    trigger_optimization/
      optimizer.py                 - coordinate-descent trigger optimizer
      dpr_optimizer.py             - dpr-hotflip trigger optimizer (phase 17)
  defenses/
    implementations.py             - watermark, validation, proactive, composite
    semantic_anomaly.py            - memsad detector (calibrate, detect, threshold_sweep,
                                     calibrate_triggered, combined scoring mode)
    lexical_diversity.py           - lexical-diversity gate + sad+gate wrapper
  evaluation/
    retrieval_sim.py               - faiss-backed retrieval simulation
    attack_defense_matrix.py       - 3x5 attack-defense cross-product evaluator
    ablation_study.py              - corpus/topk/poison/sigma sweeps
    statistical.py                 - bootstrap ci, clopper-pearson, multi-trial fpr
    comprehensive_eval.py          - end-to-end paper result generation
    multi_encoder_eval.py          - 6-encoder generalization + cka transferability
    multi_agent_propagation.py     - sir epidemic model for memory propagation
    hypothesis_testing.py          - bonferroni-corrected tests, power analysis
    evasion_eval.py                - watermark evasion evaluator
    agent_eval.py                  - gpt-2 local + gpt-4o-mini openai agent evaluators
  watermark/
    watermarking.py                - unigram, lsb, semantic, crypto, composite encoders
    token_watermark.py             - token-level watermark (zhao et al. iclr 2024)
  memory_systems/
    vector_store.py                - faiss indexflatip + all-minilm-l6-v2
    graph_memory.py                - entity-relation graph memory + structural attacks
    wrappers.py                    - mem0, a-mem, memgpt wrappers
  data/
    synthetic_corpus.py            - 200-entry seed corpus (also extends to 1,000)
    corpus_extended.py             - 1,000-entry corpus used in paper tables
    nq_subset.py                   - natural questions cross-corpus replication set
  scripts/
    run_pipeline.py                - 6-phase end-to-end pipeline (quick/full mode)
    generate_sad_figures.py        - phase 11 sad figures
    generate_phase17_figures.py    - phase 17 figures (dpr, token-wm, measured asr-a)
    generate_phase22_figures.py    - phase 22 figures (multi-encoder, cka, clopper-pearson)
    generate_phase35_figures.py    - phase 35 figures (mechanism, coupling, synonym)
    generate_sir_simulation.py     - phase 32 sir multi-agent propagation figures
    encoder_generalization.py      - 6-encoder sad evaluation

docs/
  neurips2026/
    main.tex                       - paper root (pdflatex + bibtex cycle)
    figures/                       - 30+ pdf/png figure pairs
    references.bib                 - bibliography
    checklist.tex                  - neurips 2026 reproducibility checklist
  research/
    progress_report.md             - phase-by-phase implementation log
    memory-agent-research-document.md  - historical research scoping doc
  api/API_REFERENCE.md             - module-level api surface
  guides/USAGE_GUIDE.md            - end-to-end usage patterns
  publication_strategy.md          - venue strategy notes

results/
  tables/                          - generated latex tables
  phase28/                         - phase 28 defense-results snapshots

src/app/                           - gradio backend (legacy; frontend now react spa)
src/frontend/                      - react + vite demo (deployed to hf space)
tests/, src/tests/                 - 593 tests total (593 pass)
```

---

## Key results

### Attack success rates (|M| = 1,000, 100 queries, 5 seeds, bootstrap 95% CI)

| Attack | ASR-R | ASR-A (GPT-2) | ASR-A (GPT-4o-mini) |
|---|---|---|---|
| AgentPoison (triggered) | 1.000 | 0.42 | 0.48 |
| MINJA | 1.000 | 0.51 | 0.20* |
| InjecMEM | 0.852 | 0.29 | 0.00* |

*GPT-4o-mini's safety alignment suppresses downstream ASR-A for the two attacks whose payloads rely on producing explicit tool-use text; the retrieval-level attack (ASR-R) still succeeds.

### Attack × defense matrix (ASR-R under defense; lower is better)

| Attack | No defense | Perplexity | Similarity cap | LLM sanitizer | **MemSAD** | Composite |
|---|---|---|---|---|---|---|
| AgentPoison | 0.842 | 0.781 | 0.594 | 0.402 | **0.073** | 0.000 |
| MINJA | 0.751 | 0.692 | 0.503 | 0.396 | **0.091** | 0.000 |
| InjecMEM | 0.488 | 0.412 | 0.322 | 0.275 | **0.264** | 0.000 |

MemSAD row reports 95% Clopper–Pearson CIs; composite defense achieves ASR-R = 0 on every attack at TPR = 1.00, FPR = 0.00.

### MemSAD threshold sweep (σ)

At σ = 2.0 with combined scoring (default operating point): AgentPoison triggered-calibration TPR = 1.000, MINJA TPR = 1.000, InjecMEM TPR = 0.433; all three at FPR = 0.000.

---

## Development

Pre-commit hooks enforce `black`, `isort`, `ruff`, `flake8`, and `pyupgrade`. CI (`.github/workflows/ci.yml`) runs lint, typecheck (mypy), security (Bandit), and the full test matrix on Python 3.10–3.13 for every PR.

```bash
pre-commit install
pre-commit run --all-files
```

`setup.cfg` pins `flake8 max-line-length=88` and isort profile `black`.

---

## Citation

```bibtex
@inproceedings{memsad2026,
  title     = {{MemSAD}: Gradient-Coupled Anomaly Detection for
               Memory Poisoning in Retrieval-Augmented Agents},
  author    = {Anonymous Author(s)},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under review}
}
```

Full author list, affiliations, and acknowledgements will be disclosed after double-blind review.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
