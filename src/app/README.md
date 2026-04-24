---
title: MemSAD
emoji: "\U0001F6E1"
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MemSAD — Gradient-Coupled Anomaly Detection for Memory Poisoning

Interactive research artifact for the NeurIPS 2026 submission *MemSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents* (under double-blind review).

The live Space at https://ishrith-gowda-memsad-demo.hf.space/ is a React + Vite single-page app served via Docker + nginx. The frontend source lives in-repo at `src/frontend/`.

## Sections

- **§1 Hero** — live passage feed + headline results
- **§2 Single-run** — retrieve → score → decide on a victim query
- **§3 Threshold** — σ sweep across attack families with live ROC
- **§4 Matrix** — 3 attacks × 5 defenses
- **§5 About** — methodology, reproduction details, BibTeX

## Double-blind submission notice

The associated paper is currently under double-blind review at a top-tier venue. This public Hugging Face Space is deployed for an independent engineering / internship context (interactive artifact of the research framework) and **does not constitute part of the submitted paper's supplementary material**. Reviewers should consult the anonymized supplementary archive provided via the conference submission system rather than this deployment, which is necessarily linked to an identifiable GitHub account.

Author names, affiliations, funding acknowledgements, and a full BibTeX citation entry will be added here once the paper is accepted and de-anonymized.

## Legacy Gradio backend

The earlier Gradio-based backend at `src/app/app.py` is retained for programmatic access (batch scoring, direct attack/defense invocation). It is not deployed on the live Space but remains reproducible locally:

```bash
python3 src/app/app.py
```
