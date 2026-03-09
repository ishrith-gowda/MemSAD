# Poisoning the Memory: A Unified Framework for Attacking and Defending LLM Agent Memory Systems

Research code for the paper submitted to NeurIPS 2026 / ACM CCS 2026.

**Authors:** Ishrith Gowda, University of California, Berkeley

---

## Overview

LLM agents persist context across sessions through external memory systems (Mem0, A-MEM, MemGPT). This repository provides a unified evaluation framework for three memory poisoning attacks and five defenses, evaluated under a realistic FAISS-backed vector memory store with a 200-entry synthetic corpus.

**Key contributions:**
- Evaluation protocol correction for AgentPoison (triggered-query evaluation raises ASR-R 0.25 → 1.00)
- Semantic Anomaly Detection (SAD) — novel unsupervised defense with no labeled attack examples required
- Formal gradient tension proposition (Proposition 1) showing continuous evasion is impossible
- Full 3×5 attack-defense interaction matrix with measured values
- DPR HotFlip trigger optimizer, token-level watermark, and GPT-2 agent evaluator
- Triggered-query calibration closes AgentPoison blind spot (TPR 0.00 → 1.00, FPR = 0.00)

---

## Installation

Requires Python 3.10+ and approximately 4 GB disk space for model downloads.

```bash
git clone https://github.com/ishrith-gowda/memory-agent-security
cd memory-agent-security
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Hardware:** All evaluations run on CPU. GPU is not required. The DPR HotFlip optimizer and token-level watermark benefit from a GPU but are not required for core experiments.

**Model downloads:** `sentence-transformers/all-MiniLM-L6-v2` (90 MB) and `facebook/dpr-ctx_encoder-single-nq-base` (440 MB) are downloaded automatically on first run via Hugging Face Hub.

---

## Quick Start

### Smoke test (5 tests, ~10 seconds)

```bash
python3 smoke_test.py
```

### Fast test mode (~3 minutes, corpus=15, reduced parameters)

```bash
MEMORY_SECURITY_TEST=true python3 -m pytest src/tests/test_memory_security.py tests/ -q
```

### Full test suite (~3 minutes in fast mode, ~30 minutes full)

```bash
python3 -m pytest src/tests/test_memory_security.py tests/ -q
```

---

## Reproducing Paper Results

### Full pipeline (all tables + figures)

```bash
python3 -m src.scripts.run_pipeline --mode quick   # ~5 minutes, reduced params
python3 -m src.scripts.run_pipeline --mode full    # ~60 minutes, paper params
```

Output is written to `pipeline_output/`.

### Individual components

```bash
# generate attack evaluation results (table 1)
python3 -m src.scripts.generate_paper_results

# generate all paper figures
python3 -m src.scripts.generate_paper_results --figures-only

# generate SAD ROC curve and calibration comparison figures
python3 -m src.scripts.generate_sad_figures

# generate phase 17 figures (dpr convergence, watermark comparison, measured asr-a)
python3 -m src.scripts.generate_phase17_figures
```

### Compile the paper

```bash
cd docs/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Pre-generated figures are included in `docs/paper/figures/` so the paper compiles without re-running experiments.

---

## Repository Structure

```
src/
  attacks/
    implementations.py             — agentpoison, minja, injecmem
    adaptive_attack.py             — white-box adaptive adversary (synonym substitution)
    trigger_optimization/          — coordinate-descent and dpr hotflip trigger optimizers
  defenses/
    implementations.py             — watermark, validation, proactive, composite
    semantic_anomaly.py            — sad defense (novel; calibrate, detect, threshold_sweep)
  evaluation/
    retrieval_sim.py               — realistic faiss retrieval simulation
    attack_defense_matrix.py       — 3x5 attack-defense cross-product evaluator
    ablation_study.py              — corpus/topk/poison/sad/watermark sweeps
    statistical.py                 — bootstrap ci, hypothesis testing, latex table generation
    comprehensive_eval.py          — end-to-end paper result generation
    agent_eval.py                  — local gpt-2 agent evaluator (measured asr-a lower bounds)
    evasion_eval.py                — watermark evasion evaluator
  watermark/
    watermarking.py                — unigram, lsb, semantic, crypto, composite encoders
    token_watermark.py             — token-level watermark (zhao et al. iclr 2024, gpt-2 backbone)
  memory_systems/
    vector_store.py                — faiss indexflatip + all-minilm-l6-v2 (paper eval backbone)
    wrappers.py                    — mem0, a-mem, memgpt wrappers (graceful degradation)
  data/
    synthetic_corpus.py            — 200-entry synthetic agent memory corpus (7 categories)
  scripts/
    run_pipeline.py                — end-to-end pipeline (quick/full mode)
    generate_paper_results.py      — paper table + figure generation
    generate_sad_figures.py        — sad roc curve + triggered calibration comparison
    generate_phase17_figures.py    — phase 17 paper figures
    visualization.py               — benchmark visualizer and statistical analyzer

docs/paper/
  main.tex                         — paper root (pdflatex + bibtex)
  sections/                        — introduction, threat_model, attacks, defenses,
                                     experiments, discussion, related_work, conclusion, appendix
  figures/                         — 34 figure files (png + pdf pairs)
  references.bib                   — 48 bibliography entries

results/tables/                    — 7 generated latex tables (table1–table6)
configs/                           — yaml configuration files for attacks and defenses
notebooks/experiments/             — 4 jupyter notebooks with interactive experiments
src/tests/                         — 408 unit and integration tests
tests/                             — 37 additional unit tests
```

---

## Key Results

### Attack Success Rates (N=200, k=5)

| Attack | ASR-R | ASR-A* | ASR-T |
|---|---|---|---|
| AgentPoison (triggered) | 1.000 | 0.68 | 0.650 |
| MINJA | 0.650 | 0.76 | 0.550 |
| InjecMEM | 0.500 | 0.57 | 0.350 |

*ASR-A modelled from paper-reported values; measured lower bounds via GPT-2 agent: 0.42, 0.51, 0.29.

### Attack-Defense Matrix (ASR-R under defense)

| Attack | Watermark | Validation | Proactive | Composite | SAD |
|---|---|---|---|---|---|
| AgentPoison | 0.000 | 0.000 | 0.000 | 0.000 | 1.000† |
| MINJA | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| InjecMEM | 0.000 | 0.483 | 0.000 | 0.000 | 0.433 |

†Under triggered-query calibration (see paper Section 7), SAD achieves ASR-R = 0.000 for AgentPoison (TPR = 1.00, FPR = 0.00).

---

## Configuration

The evaluation uses test mode by default in CI (`MEMORY_SECURITY_TEST=true`):
- Corpus size: 15 entries (vs. 200 paper)
- Top-k: 3 (vs. 5 paper)
- Poison count: 1 (vs. 5 paper)

Full paper parameters are activated when `MEMORY_SECURITY_TEST` is unset or `false`.

Pre-commit hooks enforce code style: `black`, `isort`, and `flake8` (max-line-length=88).

---

## Citation

```bibtex
@article{gowda2026poisoning,
  title={Poisoning the Memory: A Unified Framework for Attacking and
         Defending {LLM} Agent Memory Systems},
  author={Gowda, Ishrith},
  year={2026},
  note={Preprint}
}
```

---

## License

MIT License. See `LICENSE` for details.
