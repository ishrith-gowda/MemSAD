---
title: MemSAD Memory Agent Security Demo
emoji: "\U0001F6E1"
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# MemSAD: Semantic Anomaly Detection for Memory Poisoning Attacks

Interactive demonstration of memory poisoning attacks (AgentPoison, MINJA, InjecMEM) against LLM agent memory systems and the MemSAD defense.

## Features

- Select an attack type and observe poison passages retrieved by FAISS vector search
- Toggle MemSAD defense to watch anomaly detection scores
- Threshold sweep across sigma values
- Comparative evaluation across all attacks

## Paper

*Evaluating and Defending Against Memory Poisoning Attacks on LLM Agents* (under review)
