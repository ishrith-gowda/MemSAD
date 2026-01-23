# Research Methodology and Principles

## Document Information

| field | value |
|-------|-------|
| project | memory agent security research |
| target venues | NeurIPS 2026, ACM CCS 2026 |
| research group | UC Berkeley AI Research (BAIR) |
| advisor | Dr. Xuandong Zhao |

---

## 1. Research Problem Statement

### 1.1 Problem Definition

memory-augmented large language model (LLM) agents represent a critical advancement in AI systems, enabling persistent context across conversations through external memory systems. however, these memory systems introduce novel attack surfaces that have not been systematically characterized or defended against.

this research addresses two fundamental questions:

1. **attack characterization**: how can adversaries exploit memory systems in LLM agents, and what are the success rates, persistence characteristics, and benign utility impacts of different attack methodologies?

2. **defense development**: can provenance-aware watermarking techniques effectively detect and mitigate memory poisoning attacks while maintaining acceptable false positive rates and computational overhead?

### 1.2 Research Hypothesis

we hypothesize that:

1. memory poisoning attacks (agentpoison, minja, injecmem) achieve high success rates (>70% asr) against current memory systems due to lack of provenance verification.

2. watermarking-based defenses adapted from dr. zhao's text watermarking methodology can detect >90% of poisoning attacks with <5% false positive rate.

3. the defense-attack tradeoff can be optimized through composite watermarking techniques that balance robustness, detectability, and computational efficiency.

---

## 2. Theoretical Foundations

### 2.1 Memory Agent Architecture

memory-augmented agents follow a retrieve-augmented generation (RAG) paradigm:

```
user query → retrieval from memory → augmented context → LLM generation → response
                    ↑
            attack surface
```

the memory retrieval step creates vulnerability because:
- retrieved content directly influences agent behavior
- no provenance verification exists in current systems
- adversaries can inject content that appears semantically relevant

### 2.2 Attack Taxonomy

based on published research, memory attacks are categorized by:

| dimension | categories |
|-----------|------------|
| **access level** | direct write, query-only, observation-only |
| **attack goal** | retrieval hijacking, action manipulation, information extraction |
| **persistence** | ephemeral, session-persistent, cross-session persistent |
| **trigger mechanism** | always-on, keyword-triggered, context-triggered |

### 2.3 Watermarking Theory

dr. zhao's watermarking methodology provides theoretical guarantees:

**unigram watermark (iclr 2024):**
- vocabulary partitioned into green (γ) and red (1-γ) lists using prf
- generation bias δ increases green token probability
- detection via z-score: z = (|green tokens| - γn) / √(γ(1-γ)n)
- provable robustness to text editing (2x better than kgw)

**permute-and-flip decoder (iclr 2025):**
- distortion-free watermarking (preserves distribution)
- optimal quality-stability tradeoff
- uses exponential noise for token selection

### 2.4 Defense-in-Depth Principle

the composite defense architecture follows defense-in-depth:

```
content → watermark verification → pattern analysis → anomaly detection → decision
              (cryptographic)         (syntactic)         (statistical)
```

multiple independent detection mechanisms increase overall robustness against adaptive adversaries.

---

## 3. Experimental Methodology

### 3.1 Attack Evaluation Protocol

**independent variables:**
- attack type (agentpoison, minja, injecmem)
- memory system (mem0, a-mem, memgpt)
- poison rate (0.01%, 0.1%, 1.0%)
- attack parameters (strength, depth, persistence)

**dependent variables:**
- asr-r: proportion of queries that retrieve poisoned content
- asr-a: proportion of retrievals that result in target action
- asr-t: end-to-end task hijacking rate
- benign accuracy: utility on non-adversarial queries

**control conditions:**
- baseline without defense
- clean memory system without poisoning
- random content injection (non-adversarial)

### 3.2 Defense Evaluation Protocol

**independent variables:**
- defense type (watermark, validation, proactive, composite)
- watermark algorithm (lsb, semantic, cryptographic, composite)
- detection threshold (0.5, 0.7, 0.9)

**dependent variables:**
- true positive rate (tpr): poisoned content correctly detected
- false positive rate (fpr): clean content incorrectly flagged
- precision, recall, f1-score
- detection latency (ms)

**evaluation datasets:**
- clean content: legitimate memory entries from longmemeval/locomo
- poisoned content: generated using attack implementations

### 3.3 Statistical Analysis

all experiments use:
- minimum 10 trials per condition
- mean and standard deviation reporting
- statistical significance testing (t-test, p<0.05)
- confidence intervals (95%)

benchmark results include:
- sample size and trial count
- execution time statistics
- memory integrity score (defense effectiveness - attack success)

---

## 4. Evaluation Metrics

### 4.1 Attack Success Metrics

| metric | formula | interpretation |
|--------|---------|----------------|
| asr-r | retrieved_poisoned / total_queries | retrieval hijacking effectiveness |
| asr-a | target_action / retrieved_poisoned | action manipulation given retrieval |
| asr-t | successful_hijacks / total_queries | end-to-end attack success |
| isr | successful_injections / injection_attempts | injection success rate |

### 4.2 Defense Performance Metrics

| metric | formula | interpretation |
|--------|---------|----------------|
| tpr | tp / (tp + fn) | sensitivity to attacks |
| fpr | fp / (fp + tn) | false alarm rate |
| precision | tp / (tp + fp) | detection reliability |
| recall | tp / (tp + fn) | attack coverage |
| f1 | 2 × (precision × recall) / (precision + recall) | harmonic mean |
| auroc | area under roc curve | threshold-independent performance |

### 4.3 Target Performance Thresholds

based on published baselines and practical requirements:

| metric | target | rationale |
|--------|--------|-----------|
| attack asr | >70% | demonstrates significant threat |
| defense tpr | >95% | acceptable miss rate <5% |
| defense fpr | <5% | usability requirement |
| detection latency | <100ms | real-time requirement |
| memory integrity | >0.8 | net positive defense effect |

---

## 5. Research Principles

### 5.1 Reproducibility

all experiments must be reproducible:
- fixed random seeds for stochastic components
- explicit configuration files (yaml/json)
- version-controlled dependencies (requirements.txt)
- detailed logging of all parameters and results

### 5.2 Transparency

full disclosure of:
- attack success rates (including failures)
- defense false positive rates
- computational requirements
- limitations and boundary conditions

### 5.3 Ethical Considerations

this research follows responsible disclosure:
- attacks are characterized for defense development
- no real-world deployment of attacks
- findings shared with memory system developers
- defense tools released publicly

### 5.4 Scientific Rigor

- claims supported by statistical evidence
- baseline comparisons for all results
- ablation studies for component contributions
- discussion of threats to validity

---

## 6. Implementation Standards

### 6.1 Code Quality

- all comments, docstrings, print statements lowercase
- comprehensive type hints
- abstract base classes for extensibility
- factory patterns for instantiation

### 6.2 Documentation

- api reference for all public interfaces
- usage examples for common workflows
- research documentation with citations
- inline comments for complex algorithms

### 6.3 Testing

- unit tests for individual components
- integration tests for workflows
- smoke tests for quick verification
- benchmark tests for performance

---

## 7. Timeline and Milestones

### 7.1 phase 1: foundation (weeks 1-4)

- [x] repository structure and dependencies
- [x] base interfaces and protocols
- [x] initial implementations

### 7.2 phase 2: audit and inventory (weeks 5-6)

- [x] codebase audit documentation
- [x] research methodology documentation
- [ ] gap analysis and remediation

### 7.3 phase 3: attack characterization (weeks 7-12)

- [ ] full attack reproduction
- [ ] cross-system evaluation
- [ ] statistical analysis

### 7.4 phase 4: defense development (weeks 13-18)

- [ ] advanced watermarking integration
- [ ] defense optimization
- [ ] comparative evaluation

### 7.5 phase 5: publication (weeks 19-24)

- [ ] experimental validation
- [ ] paper writing
- [ ] submission to neurips 2026 / acm ccs 2026

---

## 8. Key References

### 8.1 Attack Papers

1. chen et al. "agentpoison: red-teaming llm agents via poisoning memory or knowledge bases." neurips 2024. arxiv:2407.12784

2. dong et al. "memory injection attacks on llm agents via query-only interaction." neurips 2025. arxiv:2503.03704

3. "injecmem: targeted memory injection with single interaction." iclr 2026 submission. openreview:QVX6hcJ2um

4. zhang et al. "agent security bench (asb): formalizing and benchmarking attacks and defenses in llm-based agents." iclr 2025. arxiv:2410.02644

### 8.2 Defense and Watermarking Papers

1. zhao et al. "provable robust watermarking for ai-generated text." iclr 2024. arxiv:2306.17439

2. zhao et al. "permute-and-flip: an optimally stable and watermarkable decoder for llms." iclr 2025. arxiv:2402.05864

3. zhao et al. "sok: watermarking for ai-generated content." ieee s&p 2025. arxiv:2411.18479

### 8.3 Memory System Papers

1. "mem0: building production-ready ai agents with scalable long-term memory." arxiv:2504.19413

2. "a-mem: agentic memory for llm agents." neurips 2025. arxiv:2502.12110

3. packer et al. "memgpt: towards llms as operating systems." arxiv:2310.08560

### 8.4 Benchmark Papers

1. wu et al. "longmemeval: benchmarking chat assistants on long-term interactive memory." iclr 2025. arxiv:2410.10813

2. "locomo: evaluating very long-term conversational memory of llm agents." acl 2024. arxiv:2402.17753

---

*methodology document version: 1.0*
*last updated: january 2026*
