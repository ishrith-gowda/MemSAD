---
title: MemSAD Memory Agent Security Demo
emoji: "\U0001F6E1"
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
---

# MemSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning

Interactive demonstration of memory poisoning attacks (AgentPoison, MINJA, InjecMEM) against LLM agent memory systems and the MemSAD defense.

## Features

- Select an attack type and observe poison passages retrieved by FAISS vector search
- Toggle MemSAD defense to watch anomaly detection scores
- Threshold sweep across sigma values
- Comparative evaluation across all attacks

## Paper

*MemSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents* (under review)

## Double-Blind Submission Notice

the associated paper is currently under double-blind review at a top-tier venue.
this public Hugging Face Space is deployed for an independent engineering /
internship context (interactive artifact of the research framework) and
**does not constitute part of the submitted paper's supplementary material**.
reviewers are asked to consult the anonymized supplementary archive provided
via the conference submission system rather than this deployment, which is
necessarily linked to an identifiable GitHub account.

author names, affiliations, funding acknowledgements, and a full bibtex
citation entry will be added here once the paper is accepted and
de-anonymized.
