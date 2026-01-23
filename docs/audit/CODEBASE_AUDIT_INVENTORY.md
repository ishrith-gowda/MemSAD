# Memory Agent Security Research - Codebase Audit Inventory

## Document Information

| Field | Value |
|-------|-------|
| audit date | january 2026 |
| auditor | research assistant |
| project | memory agent security research |
| target venue | NeurIPS 2026 / ACM CCS 2026 |
| advisor | Dr. Xuandong Zhao (UC Berkeley BAIR) |

---

## 1. Project Overview

### 1.1 Research Objectives

this research framework characterizes memory poisoning attacks on memory-augmented LLM agents (Mem0, A-MEM, MemGPT) and develops provenance-aware watermarking defenses based on Dr. Xuandong Zhao's established watermarking methodology.

### 1.2 Repository Structure

```
memory-agent-security/
├── src/                          # core source code
│   ├── attacks/                  # attack implementations
│   │   ├── base.py              # abstract attack interface
│   │   └── implementations.py    # agentpoison, minja, injecmem
│   ├── defenses/                 # defense implementations
│   │   ├── base.py              # abstract defense interface
│   │   └── implementations.py    # watermark, validation, proactive, composite
│   ├── memory_systems/           # memory system integrations
│   │   ├── base.py              # memorysystem protocol
│   │   └── wrappers.py          # mem0, amem, memgpt wrappers
│   ├── watermark/                # watermarking algorithms
│   │   └── watermarking.py      # lsb, semantic, cryptographic, composite
│   ├── evaluation/               # benchmarking framework
│   │   └── benchmarking.py      # evaluators and benchmark runner
│   ├── scripts/                  # automation scripts
│   │   ├── experiment_runner.py # experiment automation
│   │   └── visualization.py     # plotting and analysis
│   ├── tests/                    # test suite
│   │   ├── conftest.py          # pytest fixtures
│   │   └── test_memory_security.py
│   └── utils/                    # shared utilities
│       ├── config.py            # configuration management
│       └── logging.py           # research logger
├── configs/                      # configuration files
│   └── memory/                   # memory system configs
├── docs/                         # documentation
│   ├── api/                      # api reference
│   ├── research/                 # research documentation
│   └── guides/                   # usage guides
├── external/                     # git submodules
│   ├── mem0/                     # mem0 library
│   ├── amem/                     # a-mem library
│   └── memgpt/                   # memgpt/letta library
├── setup.py                      # project setup
├── smoke_test.py                 # functionality verification
└── requirements.txt              # dependencies
```

---

## 2. Attack Implementations Inventory

### 2.1 Attack Base Class (`src/attacks/base.py`)

| component | description | lines |
|-----------|-------------|-------|
| `MemorySystem` | protocol defining memory system interface | 22-45 |
| `Attack` | abstract base class for all attacks | 47-140 |
| `AttackDefensePair` | pairing mechanism for attack-defense testing | 142-209 |

**abstract methods required by attack subclasses:**
- `attack_type` (property): returns attack type identifier
- `target_systems` (property): returns list of compatible memory systems
- `description` (property): returns human-readable description
- `execute(memory_system)`: executes attack and returns results dict

### 2.2 AgentPoison Attack (`src/attacks/implementations.py:21-235`)

| attribute | value |
|-----------|-------|
| **research paper** | "AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases" (NeurIPS 2024) |
| **arxiv** | arXiv:2407.12784 |
| **github** | https://github.com/AI-secure/AgentPoison |
| **attack type** | `agent_poison` |
| **target systems** | mem0, amem, memgpt |

**implemented techniques:**
1. `content_corruption`: character-level corruption using visual similarity (a→@, e→3, i→1, o→0, s→$)
2. `false_memories`: injection of fabricated memory entries with misleading information
3. `context_manipulation`: modifying context to alter interpretation of legitimate content

**configurable parameters:**
- `poison_types`: list of enabled corruption techniques
- `poison_strength`: intensity of corruption (0.0-1.0)
- `target_memory_system`: default memory system target

### 2.3 MINJA Attack (`src/attacks/implementations.py:237-438`)

| attribute | value |
|-----------|-------|
| **research paper** | "Memory Injection Attacks on LLM Agents via Query-Only Interaction" (NeurIPS 2025) |
| **arxiv** | arXiv:2503.03704 |
| **openreview** | https://openreview.net/forum?id=QINnsnppv8 |
| **attack type** | `minja` |
| **target systems** | mem0, amem, memgpt |

**implemented techniques:**
1. `direct_injection`: explicit malicious payload insertion
2. `stealth_injection`: payload disguised as legitimate metadata
3. `cascade_injection`: linked malicious content creating reference chains

**key distinction from agentpoison:**
- minja requires only query interaction (no direct memory write access)
- achieves 98.2% injection success rate in original paper

### 2.4 InjecMEM Attack (`src/attacks/implementations.py:441-665`)

| attribute | value |
|-----------|-------|
| **research paper** | "InjecMEM: Targeted Memory Injection with Single Interaction" (ICLR 2026 submission) |
| **openreview** | https://openreview.net/forum?id=QVX6hcJ2um |
| **attack type** | `injecmem` |
| **target systems** | mem0, amem, memgpt |

**implemented techniques:**
1. `content_replacement`: semantic word substitution (trust→distrust, secure→insecure)
2. `backdoor_insertion`: hidden triggers for future activation
3. `trigger_activation`: immediate execution of malicious payloads

**configurable parameters:**
- `manipulation_types`: enabled manipulation techniques
- `persistence_level`: determines number of persistent memory entries created

### 2.5 Attack Suite (`src/attacks/implementations.py:694-767`)

utility class for batch attack execution across all implemented attack types.

---

## 3. Defense Implementations Inventory

### 3.1 Defense Base Class (`src/defenses/base.py`)

| component | description | lines |
|-----------|-------------|-------|
| `Defense` | abstract base class for all defenses | 21-137 |

**abstract methods required by defense subclasses:**
- `defense_type` (property): returns defense type identifier
- `protected_attacks` (property): returns list of defended attack types
- `description` (property): returns human-readable description
- `activate(memory_system)`: activates defense on target system
- `deactivate()`: deactivates defense
- `detect_attack(operation, **kwargs)`: detects ongoing attacks

### 3.2 Watermark Defense (`src/defenses/implementations.py:23-181`)

| attribute | value |
|-----------|-------|
| **theoretical foundation** | dr. xuandong zhao's watermarking methodology |
| **primary papers** | "Provable Robust Watermarking for AI-Generated Text" (ICLR 2024), "Permute-and-Flip" (ICLR 2025) |
| **defense type** | `watermark` |
| **protected attacks** | agent_poison, minja, injecmem |

**detection mechanism:**
1. checks for watermark presence in content
2. verifies watermark integrity against registry
3. flags missing or tampered watermarks as potential attacks

**configurable parameters:**
- `encoder_type`: watermark algorithm (lsb, semantic, crypto, composite)
- `detection_threshold`: minimum confidence for valid watermark (default: 0.7)

### 3.3 Content Validation Defense (`src/defenses/implementations.py:183-455`)

| attribute | value |
|-----------|-------|
| **theoretical foundation** | pattern-based anomaly detection |
| **defense type** | `content_validation` |
| **protected attacks** | agent_poison, minja, injecmem |

**validation methods:**
1. `checksum_verification`: sha256/md5 integrity checking
2. `pattern_analysis`: detection of known malicious patterns (MALICIOUS_INJECTION, BACKDOOR:, TRIGGER:, etc.)
3. `anomaly_detection`: statistical analysis of character distribution and word repetition

### 3.4 Proactive Defense (`src/defenses/implementations.py:457-617`)

| attribute | value |
|-----------|-------|
| **theoretical foundation** | attack simulation and prevention |
| **defense type** | `proactive` |
| **protected attacks** | agent_poison, minja, injecmem |

**mechanism:**
- runs attack simulations on content to detect vulnerabilities
- monitors memory operations in real-time
- blocks suspicious activities before damage occurs

### 3.5 Composite Defense (`src/defenses/implementations.py:619-796`)

| attribute | value |
|-----------|-------|
| **theoretical foundation** | multi-layered defense-in-depth |
| **defense type** | `composite` |
| **protected attacks** | agent_poison, minja, injecmem |

**architecture:**
- combines watermark, validation, and proactive defenses
- configurable weights for each component (default: watermark=0.4, validation=0.4, proactive=0.2)
- weighted confidence aggregation for final detection decision

### 3.6 Defense Suite (`src/defenses/implementations.py:828-922`)

utility class for coordinated defense activation and detection across all defense types.

---

## 4. Watermarking Algorithms Inventory

### 4.1 Base Encoder (`src/watermark/watermarking.py:22-98`)

abstract base class defining watermark encoder interface with methods:
- `embed(content, watermark)`: embeds watermark into content
- `extract(content)`: extracts watermark from content
- `detect(content, watermark)`: detects specific watermark with confidence score

### 4.2 LSB Watermark Encoder (`src/watermark/watermarking.py:100-189`)

| attribute | value |
|-----------|-------|
| **algorithm** | least significant bit steganography |
| **robustness** | low (vulnerable to character modifications) |
| **capacity** | 1 bit per character |
| **detectability** | low |

**mechanism:**
- converts watermark to binary representation
- embeds bits in lsb of character codes
- extraction reverses the process

### 4.3 Semantic Watermark Encoder (`src/watermark/watermarking.py:192-277`)

| attribute | value |
|-----------|-------|
| **algorithm** | natural language pattern embedding |
| **robustness** | medium (survives simple edits) |
| **capacity** | limited |
| **detectability** | very low |

**mechanism:**
- uses hash of watermark for deterministic transformations
- applies synonym substitution and punctuation patterns
- detection looks for semantic markers

### 4.4 Cryptographic Watermark Encoder (`src/watermark/watermarking.py:279-368`)

| attribute | value |
|-----------|-------|
| **algorithm** | RSA-PSS digital signatures |
| **robustness** | high (cryptographically secure) |
| **capacity** | unlimited (appended as metadata) |
| **detectability** | high (explicit marker) |

**mechanism:**
- generates RSA 2048-bit key pair
- signs watermark data with private key
- appends base64-encoded signature as html comment
- verification uses public key

### 4.5 Composite Watermark Encoder (`src/watermark/watermarking.py:371-477`)

| attribute | value |
|-----------|-------|
| **algorithm** | multi-technique combination |
| **robustness** | highest (redundant encoding) |
| **capacity** | varied |
| **detectability** | medium |

**mechanism:**
- applies all three encoders sequentially
- extraction uses consensus voting (requires 2+ encoders to agree)
- weighted confidence scoring

### 4.6 Provenance Tracker (`src/watermark/watermarking.py:509-639`)

| component | description |
|-----------|-------------|
| content registry | stores watermark mappings for all registered content |
| watermark generation | creates unique identifiers from content metadata |
| provenance verification | validates content origin using watermark lookup |
| anomaly detection | identifies missing or tampered watermarks |

---

## 5. Evaluation Framework Inventory

### 5.1 Attack Metrics (`src/evaluation/benchmarking.py:27-58`)

| metric | description | target |
|--------|-------------|--------|
| `asr_r` | attack success rate - retrieval | ≥80% |
| `asr_a` | attack success rate - availability | ≥75% |
| `asr_t` | attack success rate - tampering | ≥70% |
| `execution_time_avg` | mean execution time | minimize |
| `execution_time_std` | execution time variance | minimize |
| `error_rate` | proportion of failed attempts | <5% |

### 5.2 Defense Metrics (`src/evaluation/benchmarking.py:60-113`)

| metric | description | target |
|--------|-------------|--------|
| `tpr` | true positive rate (recall) | >95% |
| `fpr` | false positive rate | <5% |
| `precision` | positive predictive value | >95% |
| `recall` | same as tpr | >95% |
| `f1_score` | harmonic mean of precision and recall | >0.95 |
| `execution_time_avg` | mean detection time | minimize |

### 5.3 Benchmark Result (`src/evaluation/benchmarking.py:115-146`)

comprehensive result dataclass containing:
- experiment metadata (id, timestamp, duration)
- attack metrics for all attack types
- defense metrics for all defense types
- system configuration snapshot
- memory integrity score (defense effectiveness - attack success)

### 5.4 Attack Evaluator (`src/evaluation/benchmarking.py:148-272`)

systematically evaluates attack performance:
- initializes memory systems for testing
- runs configurable number of trials per attack type
- calculates aggregate metrics with statistical measures

### 5.5 Defense Evaluator (`src/evaluation/benchmarking.py:274-405`)

evaluates defense effectiveness:
- tests against clean content (measures false positives)
- tests against poisoned content (measures true positives)
- calculates tpr, fpr, precision, recall, f1

### 5.6 Benchmark Runner (`src/evaluation/benchmarking.py:407-601`)

orchestrates comprehensive evaluation experiments:
- generates poisoned test datasets using attack suite
- evaluates all attacks and defenses
- calculates memory integrity score
- supports batch experiment execution
- provides result persistence (json save/load)

### 5.7 Evaluation Report Generator (`src/evaluation/benchmarking.py:603-785`)

generates comprehensive evaluation reports:
- summary statistics across experiments
- attack performance analysis by type
- defense effectiveness analysis
- automated recommendations based on results

---

## 6. Memory System Integrations

### 6.1 Memory System Protocol (`src/memory_systems/base.py`)

defines required interface for memory system compatibility:

| method | description |
|--------|-------------|
| `store(key, value)` | stores key-value pair |
| `retrieve(key)` | retrieves value by key |
| `search(query)` | semantic search |
| `get_all_keys()` | lists all stored keys |

### 6.2 Mem0 Wrapper (`src/memory_systems/wrappers.py`)

| attribute | value |
|-----------|-------|
| **external library** | mem0ai |
| **architecture** | hybrid dual-store (vector + graph) |
| **vector backends** | qdrant, chroma, pinecone, faiss, pgvector |
| **paper** | arXiv:2504.19413 |

### 6.3 A-MEM Wrapper (`src/memory_systems/wrappers.py`)

| attribute | value |
|-----------|-------|
| **external library** | agentic_memory |
| **architecture** | zettelkasten-style linked memories |
| **vector backend** | chromadb |
| **paper** | arXiv:2502.12110 (NeurIPS 2025) |

### 6.4 MemGPT Wrapper (`src/memory_systems/wrappers.py`)

| attribute | value |
|-----------|-------|
| **external library** | letta (formerly memgpt) |
| **architecture** | os-inspired hierarchical memory |
| **memory tiers** | main context (ram) + external context (disk) |
| **paper** | arXiv:2310.08560 |

---

## 7. Utility Components

### 7.1 Configuration Manager (`src/utils/config.py`)

- yaml configuration loading and validation
- centralized experiment parameter management
- memory system configuration helpers

### 7.2 Research Logger (`src/utils/logging.py`)

- file and console output with configurable levels
- specialized logging methods for attacks, defenses, experiments
- structured error logging with context

---

## 8. Test Infrastructure

### 8.1 Test Suite (`src/tests/`)

| file | purpose |
|------|---------|
| `conftest.py` | pytest fixtures and shared test configuration |
| `test_memory_security.py` | comprehensive unit and integration tests |

### 8.2 Smoke Test (`smoke_test.py`)

quick verification of core functionality:
- memory system initialization
- attack execution
- defense activation and detection
- watermarking encode/decode
- evaluation framework

---

## 9. Documentation

### 9.1 Research Documentation (`docs/research/`)

| document | content |
|----------|---------|
| `memory-agent-research-document.md` | comprehensive research guide including venue strategy, technical details, attack reproduction requirements, timeline |

### 9.2 API Reference (`docs/api/`)

| document | content |
|----------|---------|
| `API_REFERENCE.md` | complete api documentation with examples |

### 9.3 Context Documentation (`docs/context/`)

claude code context files for various framework components.

---

## 10. Identified Gaps and Recommendations

### 10.1 Critical Gaps

| gap | severity | recommendation |
|-----|----------|----------------|
| external memory systems not pip-installable | high | install mem0ai, letta packages via pip or configure api keys |
| attack implementations are simplified simulations | medium | implement full trigger optimization from agentpoison paper |
| minja code not publicly released | medium | implement based on paper methodology when code becomes available |
| no real benchmark datasets integrated | medium | integrate longmemeval and locomo datasets |

### 10.2 Enhancement Opportunities

| area | current state | recommended improvement |
|------|---------------|-------------------------|
| watermarking | basic implementations | integrate dr. zhao's unigram-watermark and pf-decoder |
| evaluation | simulated metrics | run against real memory systems with api keys |
| reproducibility | manual setup | add makefile targets and docker configuration |
| ci/cd | none | add github actions for automated testing |

### 10.3 Research Alignment

| requirement | status | notes |
|-------------|--------|-------|
| agentpoison reproduction | partial | needs trigger optimization algorithm |
| minja reproduction | partial | awaiting public code release |
| injecmem reproduction | partial | based on openreview submission |
| watermark defense | implemented | using simplified encoders |
| evaluation metrics | implemented | asr-r/a/t, tpr/fpr/precision/recall |

---

## 11. Quality Assurance Checklist

### 11.1 Code Style

- [x] all docstrings lowercase
- [x] all comments lowercase
- [x] all print statements lowercase
- [x] no emojis in code
- [x] absolute imports (converted from relative)
- [x] abstract methods implemented in subclasses

### 11.2 Documentation

- [x] api reference complete
- [x] research documentation comprehensive
- [x] usage guide available
- [x] configuration examples provided

### 11.3 Testing

- [x] smoke test available
- [x] unit test structure in place
- [ ] integration tests with real memory systems (requires api keys)
- [ ] benchmark tests against published datasets

---

## 12. Summary

this audit inventory documents the complete memory agent security research framework codebase. the framework provides:

- **3 attack implementations** based on published research (agentpoison, minja, injecmem)
- **4 defense mechanisms** including provenance-aware watermarking
- **4 watermarking algorithms** (lsb, semantic, cryptographic, composite)
- **comprehensive evaluation framework** with standard research metrics
- **3 memory system integrations** (mem0, a-mem, memgpt)

the codebase is structurally complete for phase 1 (setup) requirements. phase 2 (audit/inventory) is documented in this file. the framework is ready for experimental validation once external memory system api keys are configured.

---

*audit completed: january 2026*
*next phase: experimental validation and benchmark execution*
