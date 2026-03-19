# Comprehensive Research Documentation: Adversarial Robustness of Memory-Augmented LLM Agents

**For UC Berkeley AI Research (BAIR) – Dr. Xuandong Zhao's Research Group**

This documentation provides exhaustive technical guidance for characterizing memory poisoning attacks on Mem0/A-MEM/MemGPT and developing a provenance-aware defense framework using watermarking techniques. The research targets a **6-month timeline starting January 2026** with **15-20 hours/week** commitment.

---

## 1. Publication venue strategy for 2026

For a project starting January 2026 with results expected by June 2026, **NeurIPS 2026** (deadline May 15) and **ACM CCS 2026 Cycle 2** (deadline April 29) represent the optimal primary targets. IEEE S&P 2026 and USENIX Security 2026 Cycle 2 deadlines have already passed as of January 2026.

### Feasible venue timeline

| Venue | Deadline | Notification | Acceptance Rate | Fit Score |
|-------|----------|--------------|-----------------|-----------|
| **ICML 2026** | Jan 28, 2026 | April 30, 2026 | 25-28% | ⚠️ Very tight |
| **ACM CCS 2026 Cycle 2** | April 29, 2026 | July 17, 2026 | 18-20% | ✅ Excellent |
| **NeurIPS 2026** | May 15, 2026 | Sept 18, 2026 | 25-26% | ✅ Primary target |
| **ICLR 2027** | ~Oct 2026 | ~Jan 2027 | ~31% | ✅ Backup option |
| **NeurIPS Workshops** | Sept-Oct 2026 | Oct-Nov 2026 | 30-50% | ✅ Workshop track |

### Format requirements by venue

**NeurIPS 2026**: 9 pages content + unlimited references/appendix, NeurIPS LaTeX style, double-blind review. **ACM CCS 2026**: 12 pages double-column ACM sigconf format excluding bibliography/appendices. **ICLR 2027**: 10 pages (increased from 9 in 2026) + unlimited references/appendix.

### Workshop-first strategy is recommended

For an undergraduate researcher, submitting to workshops first provides valuable community feedback, faster publication cycles (**3-4 months** vs 6-8 months), and builds publication record for graduate applications. Most workshops are non-archival, allowing submission of expanded versions to main conferences. Target **NeurIPS ML Safety Workshop**, **Adversarial ML Frontiers (AdvML-Frontiers)**, or **Multi-Agent Security (MASEC)** workshops with deadlines typically in September-October 2026.

### Optimal submission sequence

**January-April**: Develop core results while aiming for CCS Cycle 2 (April 29). **May**: Submit to NeurIPS 2026 (May 15). **September-October**: Submit workshop papers and prepare ICLR 2027 submission. This parallel strategy maximizes publication opportunities while maintaining research momentum.

---

## 2. Technical implementation details

### Mem0: Universal memory layer

**Repository**: https://github.com/mem0ai/mem0 (Apache 2.0 license)  
**Paper**: arXiv:2504.19413 – "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"  
**Documentation**: https://docs.mem0.ai

Mem0 implements a **hybrid dual-store architecture** combining vector storage for semantic similarity search with optional graph storage for relationship modeling. The vector store supports **24+ providers** including Qdrant, Chroma, Pinecone, FAISS, pgvector, and Weaviate. Graph storage options include Neo4j, Memgraph, and Neptune.

**Core memory pipeline** flows through five stages: (1) LLM extraction of atomic facts from conversations, (2) vector embedding creation using configurable models (default: `text-embedding-3-small`), (3) memory decision by LLM (ADD/UPDATE/DELETE/NOOP), (4) parallel storage to vector and graph stores, and (5) retrieval via vector similarity plus optional graph traversal.

```python
from mem0 import Memory

config = {
    "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "port": 6333}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}}
}
memory = Memory(config)
memory.add(messages, user_id="user123")  # Store memories
memory.search(query, user_id="user123", limit=5)  # Retrieve
```

**Watermarking integration points** exist at five locations: (1) extraction prompt modification in `mem0/configs/prompts.py`, (2) embedding generation in `mem0/embeddings/`, (3) memory creation in `_create_memory()` method, (4) metadata fields during storage, and (5) verification during retrieval in `mem0/memory/main.py`.

### A-MEM: Agentic memory with Zettelkasten linking

**Repository**: https://github.com/agiresearch/A-mem (MIT license)  
**Paper**: arXiv:2502.12110 – "A-MEM: Agentic Memory for LLM Agents" (NeurIPS 2025)  
**Additional repos**: https://github.com/WujiangXu/A-mem-sys (production), https://github.com/WujiangXu/AgenticMemory (benchmarks)

A-MEM implements the Zettelkasten method with each memory note containing: original content (c_i), timestamp (t_i), LLM-generated keywords (K_i), tags (G_i), contextual description (X_i), dense embedding (e_i), and linked memories (L_i). The **memory formation process** generates keywords and context via LLM, computes embeddings combining all textual components, then identifies and creates links to semantically similar existing memories.

```python
from agentic_memory.memory_system import AgenticMemorySystem

memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend="openai",
    llm_model="gpt-4o-mini"
)
memory_system.search_agentic("neural networks", k=5)
```

**Key differences from Mem0**: A-MEM uses ChromaDB vector storage only (no graph store), implements LLM-generated semantic links rather than explicit graph edges, and features **dynamic attribute updates** through memory evolution—new memories trigger updates to existing memory attributes over time.

### MemGPT/Letta: Hierarchical context management

**Repository**: https://github.com/letta-ai/letta (Apache 2.0, renamed from MemGPT)  
**Paper**: arXiv:2310.08560 – "MemGPT: Towards LLMs as Operating Systems"  
**Documentation**: https://docs.letta.com

MemGPT implements an **OS-inspired virtual memory system** with two tiers: main context (in-context memory like RAM) containing system prompt, core memory blocks, and conversation history; and external context (out-of-context like disk) containing recall memory (conversation search DB) and archival memory (long-term vector DB). The LLM manages its own memory through **self-editing memory tools**:

```python
from letta_client import Letta

client = Letta(api_key=os.getenv("LETTA_API_KEY"))
agent = client.agents.create(
    model="openai/gpt-4.1",
    embedding="openai/text-embedding-3-small",
    memory_blocks=[
        {"label": "human", "value": "Name: Alice. Preferences: ..."},
        {"label": "persona", "value": "I am a helpful assistant..."}
    ]
)
```

The **heartbeat mechanism** enables multi-step reasoning by setting `request_heartbeat=true` in tool calls, allowing the LLM to chain multiple memory operations. Watermarking integration should target the memory tool implementations (`core_memory_append`, `core_memory_replace`, `archival_memory_insert`) since the self-editing nature requires watermarks to persist through agent-initiated modifications.

---

## 3. Dr. Xuandong Zhao's watermarking methodology

Dr. Zhao is a postdoctoral researcher at UC Berkeley BAIR (since June 2024), working with Prof. Dawn Song. His PhD research at UC Santa Barbara with Prof. Yu-Xiang Wang and Prof. Lei Li established foundational work in LLM watermarking.

### Unigram-Watermark (ICLR 2024)

**Paper**: "Provable Robust Watermarking for AI-Generated Text" – arXiv:2306.17439  
**GitHub**: https://github.com/XuandongZhao/Unigram-Watermark  
**Demo**: https://huggingface.co/spaces/Xuandong/Unigram-Watermark

The core mechanism extends Kirchenbauer et al.'s KGW watermark with a **fixed green-red list split** that remains constant across all tokens (K=1, hence "Unigram"). The vocabulary is partitioned into green (γ portion) and red lists using a PRG seeded only by a secret key. During generation, the LLM increases probability of green tokens via bias parameter δ. Detection counts green tokens and computes z-score.

**Key parameters for memory systems**: γ (gamma) = 0.25-0.5 controls green list proportion (lower = stronger but more detectable), δ (delta) = 1.0-3.0 controls bias strength, z-threshold = 4.0 for detection, minimum 50-100 tokens for reliable detection. The Unigram approach is **2x more robust** to text editing than KGW due to its n-gram independence.

### Permute-and-Flip decoder (ICLR 2025)

**Paper**: "Permute-and-Flip: An optimally stable and watermarkable decoder for LLMs" – arXiv:2402.05864  
**GitHub**: https://github.com/XuandongZhao/pf-decoding  
**Also in MarkLLM**: https://github.com/THU-BPM/MarkLLM at `watermark/pf/pf.py`

PF decoder is **up to 2x better** in quality-stability tradeoff and **never worse** than any other decoder. Unlike Unigram-Watermark, PF is **distortion-free**—it does not change the sampling distribution. The algorithm computes logits, generates random permutation of vocabulary using cryptographic key, samples Exp(1) random variables, and selects tokens based on noise-adjusted scores with probability threshold flipping.

```bash
python run.py --model_name 'NousResearch/Llama-2-7b-hf' \
  --prompt_path 'data/c4.jsonl' --temperature 0.9 \
  --top_p 1.0 --ngram 8 --max_gen_len 256
```

### SoK: Watermarking for AI-Generated Content (IEEE S&P 2025)

**Paper**: arXiv:2411.18479, IEEE S&P 2025 pages 2621-2639

This comprehensive systematization covers 100+ papers on watermarking for text, image, audio, and video. Key findings include taxonomy of text watermarking methods (green-red, Gumbel, undetectable, PRC), evaluation framework dimensions (detection accuracy, robustness, quality impact, overhead), and threat models (removal, spoofing, stealing attacks).

### Adaptation strategy for memory entry verification

For the provenance-aware defense framework, use **Unigram-Watermark** for memory entries that may undergo editing (more robust) and **PF decoder** for entries requiring quality preservation (distortion-free). Implement entry-level keys for attribution, hierarchical detection using geometry cover method (arXiv:2410.03600) for partial watermark detection, and multi-bit encoding for embedding metadata (timestamp, source) in watermark patterns.

---

## 4. Attack reproduction requirements

### AgentPoison (NeurIPS 2024)

**Paper**: arXiv:2407.12784 – "AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases"  
**GitHub**: https://github.com/AI-secure/AgentPoison (MIT license)  
**Project Page**: https://billchan226.github.io/AgentPoison.html

AgentPoison is the **first backdoor attack targeting RAG-based LLM agents** through memory/knowledge base poisoning. The attack generates triggers via constrained optimization that maps triggered instances to a unique region in embedding space. No model training is required—the attack works without fine-tuning.

**Codebase structure**: `/algo/` contains trigger optimization algorithms, `/agentdriver/` implements autonomous driving agent, `/ReAct/` implements knowledge QA agent, `/EhrAgent/` implements healthcare agent. Three success metrics: **ASR-r** (retrieval), **ASR-a** (action), **ASR-t** (end-to-end task hijacking). Performance: ≥80% average ASR with <1% benign performance drop at <0.1% poison rate.

```bash
conda env create -f environment.yml && conda activate agentpoison
python algo/trigger_optimization.py --agent ad --algo ap \
  --model dpr-ctx_encoder-single-nq-base --save_dir ./results \
  --ppl_filter --target_gradient_guidance --asr_threshold 0.5 \
  --num_adv_passage_tokens 10 --golden_trigger -w -p
```

**Compute requirements**: GPU recommended for trigger optimization; key parameters include `--num_iter 1000`, `--per_gpu_eval_batch_size 64`, `--num_cand 100`. LLaMA-3 fine-tuned planner available for offline inference.

### MINJA (NeurIPS 2025)

**Paper**: arXiv:2503.03704 – "Memory Injection Attacks on LLM Agents via Query-Only Interaction"  
**OpenReview**: https://openreview.net/forum?id=QINnsnppv8  
**Code**: Announced but not yet publicly released as of January 2026

MINJA is a **query-only attack requiring NO direct memory access**—any regular user could become an attacker. The methodology uses bridging steps (reasoning linking victim query to malicious reasoning), indication prompts (guiding agent to generate bridging steps autonomously), and progressive shortening strategy (gradually removes indication prompt).

**Performance**: **98.2%** injection success rate (ISR), **>70%** attack success rate (ASR) on most datasets. Key difference from AgentPoison: requires only query interaction vs. direct memory write access.

### InjecMEM (ICLR 2026 Submission)

**OpenReview**: https://openreview.net/forum?id=QVX6hcJ2um (submission #18062, under review)

InjecMEM achieves targeted memory injection with **single interaction**. The two-part design includes retriever-agnostic anchor (concise on-topic passage with high-recall cues) and adversarial command (optimized via gradient-based coordinate search). Evaluates on MemoryOS system; claims higher RSR/ASR than baselines with persistence after benign drift.

### Agent Security Bench – ASB (ICLR 2025)

**Paper**: arXiv:2410.02644  
**GitHub**: https://github.com/agiresearch/ASB (MIT license)  
**Website**: https://luckfort.github.io/ASBench/

ASB provides comprehensive benchmarking with **10 scenarios, 10 agents, 400+ tools, 27 attack/defense types, 7 metrics**. Attack types include direct prompt injection (DPI), observation prompt injection (OPI), memory poisoning, and PoT backdoor. Key finding: highest average ASR of **84.30%** for mixed attacks; current defenses largely ineffective.

```bash
python scripts/agent_attack.py --cfg_path config/MP.yml  # Memory poisoning
```

Configuration for custom attacks/defenses via YAML files in `/config/`; supports 13 LLM backbones including GPT-4o, Claude-3.5, LLaMA-3 variants.

---

## 5. Research project structure

### Recommended repository organization

Use the **Cookiecutter Data Science V2** template adapted for ML security research:

```
memory-agent-security/
├── README.md
├── Makefile                 # Commands: make attack, make defend, make eval
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Prepared evaluation data
│   └── external/            # Third-party benchmarks
├── configs/                 # YAML experiment configurations
├── src/
│   ├── attacks/             # Attack implementations
│   │   ├── agentpoison/
│   │   ├── minja/
│   │   └── custom/
│   ├── defenses/            # Defense implementations
│   │   ├── watermark/
│   │   └── detection/
│   ├── memory_systems/      # Mem0, A-MEM, MemGPT wrappers
│   └── evaluation/          # Benchmark runners
├── notebooks/               # Analysis notebooks (1.0-initials-description.ipynb)
├── tests/                   # Unit and integration tests
├── scripts/                 # CLI experiment scripts
└── reports/figures/         # Generated figures for papers
```

For a 6-month undergraduate project, **monorepo structure is recommended** for simpler dependency management and easier advisor review. Maintain clear directory separation between attacks and defenses with shared utilities in common modules.

### Experiment tracking recommendation

**Use Weights & Biases (wandb)** for the project—free academic tier provides 100GB storage, superior visualization, automatic code/git tracking, and collaborative features for sharing with advisor. Supplement with **DVC** for large dataset versioning.

```python
import wandb
wandb.init(project="memory-agent-security", name="agentpoison-reproduction")
wandb.config.update({"attack": "agentpoison", "poison_rate": 0.001})
wandb.log({"ASR-r": 0.82, "ASR-a": 0.75, "benign_ACC": 0.94})
```

### Documentation standards

Every attack/defense implementation should include threat model specification (access level, knowledge assumptions), parameter documentation table, usage examples with expected outputs, and reference to source papers. Weekly advisor updates should follow structured format: progress, key results, blockers/questions, and next week's plan.

---

## 6. Evaluation metrics and datasets

### LongMemEval benchmark

**Paper**: arXiv:2410.10813 – "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)  
**GitHub**: https://github.com/xiaowu0162/LongMemEval  
**Dataset**: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Tests five core memory abilities: information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. Dataset sizes: LongMemEval_S (~115k tokens, ~40 sessions), LongMemEval_M (~500 sessions). **500 curated questions** across 7 question types.

```bash
pip install -r requirements-lite.txt
python3 src/evaluation/evaluate_qa.py gpt-4o hypothesis_file data/longmemeval_oracle.json
```

### LoCoMo benchmark

**Paper**: arXiv:2402.17753 – "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024)  
**GitHub**: https://github.com/snap-research/locomo

Contains 10 high-quality conversations with ~300 turns and ~9K tokens each. Evaluates QA (five reasoning types), event graph summarization, and multi-modal dialog generation. Metrics include partial-match F1, LLM-as-judge accuracy, ROUGE scores, and FactScore.

### Standard attack metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **ASR-r** | % queries retrieving poisoned memories | ≥80% |
| **ASR-a** | % executing target action given retrieval | ≥75% |
| **ASR-t** | End-to-end adversarial impact | ≥70% |
| **ISR** | Injection success rate | ≥95% |
| **ACC** | Benign utility preservation | ≥99% |
| **Poison Rate** | Fraction of KB poisoned | <0.1% |

### Standard defense metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **TPR** | True positive rate (poisoned detected) | >95% |
| **FPR** | False positive rate | <5% |
| **DACC** | Detection accuracy | >97% |
| **ASR-d** | Post-defense ASR | <20% |
| **NRP** | Net resilient performance | Maximize |

For watermark detection: **TPR at 1% FPR** >94%, **AUROC** >97%, minimum 100+ tokens for reliable detection.

---

## 7. Compute and infrastructure

### Berkeley Savio HPC resources

**Best GPU options for memory agent research**:
- **A40 nodes (48GB VRAM)**: Best for Llama 70B with quantization
- **L40 nodes (46GB VRAM)**: Newer generation, good for inference  
- **RTX A5000 (24GB)**: Suitable for 7-13B models

**Access via Faculty Computing Allowance (FCA)**: 300,000 Service Units/year per faculty member, free for Berkeley researchers. Contact research-it.berkeley.edu for Savio access, FCA requests through PI's BRC portal.

```bash
#SBATCH --partition=savio3_gpu
#SBATCH --gres=gpu:A40:2
#SBATCH --account=fc_yourproject
#SBATCH --time=04:00:00
```

### Cloud alternatives

Memory agent research primarily uses **API-based inference** (GPT-4, Claude), making GPU requirements different from training. Estimated API costs: $50-$200/month for experiments.

For local model experiments requiring overflow from Savio:
- **RunPod**: A100 40GB at $1.29/hr
- **Lambda Labs**: H100 at $2.49/hr
- **Google Colab Pro**: $10/month for development

### Estimated compute for full project

| Phase | GPU Hours | Estimated Cost |
|-------|-----------|----------------|
| Attack characterization | 100-200 hrs | $100-$500 |
| Defense development | 150-300 hrs | $200-$800 |
| Benchmark evaluation | 100-200 hrs | $150-$500 |
| **Total** | **350-700 hrs** | **$450-$1,800** |

Most costs will be API usage for memory agent testing rather than GPU compute.

---

## 8. Timeline optimization

### Realistic 6-month timeline at 15-20 hours/week

**Phase 1: Foundation (Weeks 1-4, ~80 hours)**
- Week 1-2: Literature review, repository setup with Cookiecutter structure
- Week 3-4: Environment configuration, baseline agent deployment (Mem0, MemGPT)
- **Deliverables**: Repository initialized, literature review doc (15-20 papers), working memory agent instances

**Phase 2: Attack characterization (Weeks 5-10, ~120 hours)**
- Week 5-6: Reproduce AgentPoison, implement attack pipeline
- Week 7-8: Adapt MINJA methodology (query-only), run evaluations
- Week 9-10: Characterize attacks on all three memory systems, analyze patterns
- **Deliverables**: Working attack implementations, evaluation results on LongMemEval/LoCoMo, characterization analysis document

**Phase 3: Defense development (Weeks 11-18, ~160 hours)**
- Week 11-13: Implement Unigram-Watermark integration for memory entries
- Week 14-15: Implement PF decoder variant for quality-critical entries
- Week 16-18: Evaluate defense against characterized attacks, iterate design
- **Deliverables**: Watermarking defense implementation, comparative evaluation, results tables/figures

**Phase 4: Analysis and writing (Weeks 19-24, ~120 hours)**
- Week 19-20: Fill experimental gaps, ablation studies
- Week 21-22: Draft paper for NeurIPS 2026 (9 pages + appendix)
- Week 23-24: Advisor revisions, polish figures
- **Deliverables**: Complete paper draft, reproducibility package

**Phase 5: Buffer and submission (Weeks 25-26, ~40 hours)**
- Final revisions, code cleanup for public release
- Submit to NeurIPS 2026 (May 15) and/or prepare CCS Cycle 2 submission

### Weekly advisor communication structure

```markdown
# Weekly Update - [Date]

## Progress This Week
- [Completed task with specific outcome]

## Key Results
- [Quantitative result: e.g., "AgentPoison achieves 78% ASR-r on Mem0"]

## Blockers/Questions
1. [Specific technical question]
2. [Decision needed]

## Plan for Next Week
- [ ] Specific task 1
- [ ] Specific task 2

## Hours Logged: X
```

Schedule 30-minute weekly check-ins, follow up with written summary of decisions, maintain shared progress document in Google Docs or Notion.

---

## Key resource summary

### GitHub repositories

| System | URL |
|--------|-----|
| Mem0 | https://github.com/mem0ai/mem0 |
| A-MEM | https://github.com/agiresearch/A-mem |
| MemGPT/Letta | https://github.com/letta-ai/letta |
| AgentPoison | https://github.com/AI-secure/AgentPoison |
| Agent Security Bench | https://github.com/agiresearch/ASB |
| Unigram-Watermark | https://github.com/XuandongZhao/Unigram-Watermark |
| PF-Decoding | https://github.com/XuandongZhao/pf-decoding |
| MarkLLM | https://github.com/THU-BPM/MarkLLM |
| LongMemEval | https://github.com/xiaowu0162/LongMemEval |
| LoCoMo | https://github.com/snap-research/locomo |

### Key paper citations

```bibtex
@inproceedings{zhao2024provable,
  title={Provable Robust Watermarking for AI-Generated Text},
  author={Zhao, Xuandong and Ananth, Prabhanjan and Li, Lei and Wang, Yu-Xiang},
  booktitle={ICLR 2024}
}

@inproceedings{chen2024agentpoison,
  title={AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases},
  author={Chen, Zhaorun and Xiang, Zhen and Xiao, Chaowei and Song, Dawn and Li, Bo},
  booktitle={NeurIPS 2024}
}

@article{dong2025minja,
  title={Memory Injection Attacks on LLM Agents via Query-Only Interaction},
  author={Dong, Shen and others},
  booktitle={NeurIPS 2025}
}

@inproceedings{zhang2025asb,
  title={Agent Security Bench: Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents},
  author={Zhang, Hanrong and others},
  booktitle={ICLR 2025}
}

@article{wu2024longmemeval,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author={Wu, Di and others},
  booktitle={ICLR 2025}
}
```

---

## Conclusion

This research project is well-positioned for high-impact publication at NeurIPS 2026 or ACM CCS 2026. The combination of Dr. Zhao's established watermarking methodology with the emerging threat landscape of memory-augmented LLM agents creates a timely and significant contribution. The provenance-aware defense framework addresses a critical gap—current defenses show **>60% false negative rates** according to ASB benchmarks.

**Critical success factors**: (1) Start attack characterization immediately to meet CCS Cycle 2 deadline (April 29), (2) leverage Savio HPC for cost-effective experimentation, (3) use watermarking code from Dr. Zhao's existing repositories to accelerate defense development, and (4) evaluate on both LongMemEval and LoCoMo to demonstrate generalization. The workshop-first strategy provides backup publication paths while the main conference submissions mature.
