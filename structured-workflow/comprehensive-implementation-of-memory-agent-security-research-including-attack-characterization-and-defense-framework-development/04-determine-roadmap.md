# 04-determine-roadmap: implementation roadmap and sequence

## implementation sequence overview

```
foundation (weeks 1-2) → core_interfaces (weeks 3-4) → memory_systems (weeks 5-6)
    ↓
watermarking (weeks 7-8) → attacks (weeks 9-12) → defenses (weeks 13-16)
    ↓
evaluation (weeks 17-20) → infrastructure (weeks 21-24)
```

## detailed phase breakdown

### phase 1: foundation (steps 1-2)

**duration:** 2 weeks  
**focus:** establish development infrastructure  
**risk level:** low  
**dependencies:** none

- [ ] step_01: configuration management
- [ ] step_02: logging infrastructure

**validation checkpoint:** basic project setup functional

### phase 2: core interfaces (step 3)

**duration:** 2 weeks  
**focus:** define system contracts  
**risk level:** low  
**dependencies:** logging

- [x] step_03: attack and defense base interfaces

**validation checkpoint:** abstract classes instantiable

### phase 3: memory systems (step 4)

**duration:** 2 weeks  
**focus:** external system integration  
**risk level:** medium  
**dependencies:** config, logging

- [x] step_04: memory system wrappers

**validation checkpoint:** wrappers connect to external systems

### phase 4: watermarking (step 5)

**duration:** 2 weeks  
**focus:** core defense algorithms  
**risk level:** high  
**dependencies:** logging

- [x] step_05: watermarking algorithms

**validation checkpoint:** watermark embedding/detection works

### phase 5: attacks (step 6)

**duration:** 4 weeks  
**focus:** implement attack methodologies  
**risk level:** high  
**dependencies:** interfaces, wrappers

- [x] step_06: agentpoison, minja, injecmem implementations

**validation checkpoint:** attacks execute and poison memory

### phase 6: defenses (step 7)

**duration:** 4 weeks  
**focus:** implement defense mechanisms  
**risk level:** high  
**dependencies:** interfaces, watermarking, wrappers

- [x] step_07: provenance tracking and detection

**validation checkpoint:** defenses detect and mitigate attacks

### phase 7: evaluation (step 8)

**duration:** 4 weeks  
**focus:** benchmarking and metrics  
**risk level:** medium  
**dependencies:** attacks, defenses, wrappers

- [x] step_08: evaluation framework and benchmarks

**validation checkpoint:** full evaluation pipeline produces results

### phase 8: infrastructure (step 9)

**duration:** 4 weeks
**focus:** testing and automation
**risk level:** low
**dependencies:** evaluation

- [x] step_09: testing, scripts, visualization, documentation

**validation checkpoint:** project ready for research experimentation

### phase 9: realistic retrieval simulation and experimental validation (step 10)

**duration:** 2 weeks
**focus:** replace trivially 100% asr evaluation with faiss-backed semantic retrieval simulation; comprehensive experiment notebooks
**risk level:** medium
**dependencies:** evaluation, memory systems, attacks

- [x] step_10a: vectormemorysystem — faiss indexflatip + sentence-transformers (all-MiniLM-L6-v2, 384-dim cosine)
- [x] step_10b: syntheticcorpus — 200 realistic agent memory entries across 7 categories, 20 victim queries, 20 benign queries
- [x] step_10c: retrievalsimulator — paper-faithful asr-r, asr-a, asr-t with modelled per-retrieval action rates
- [x] step_10d: attack poison generators — agentpoison (trigger echo), minja (bridging steps), injecmem (factual anchor templates)
- [x] step_10e: benchmarking update — lazy-import retrieval sim, test mode (corpus=15 fast) vs research mode (corpus=200)
- [x] step_10f: notebook 01_attack_characterization — asr bar chart, per-query heatmap, cosine distributions, stealthiness trade-off, poison count ablation
- [x] step_10g: notebook 02_defense_evaluation — defense metrics, roc space, attack-defense interaction matrix, asr-r reduction
- [x] step_10h: notebook 03_ablation_study — watermark z-score, threshold ablation, corpus size ablation, top-k ablation, query-category similarity heatmap
- [x] step_10i: notebook 04_results_visualization — radar charts, threat model diagram, 3d scatter, pareto frontier, normalized effectiveness heatmap

**validated results (corpus=200, n_poison_base=5, top_k=5):**
- agent_poison: asr-r=0.250, asr-a=0.600, asr-t=0.150, benign_acc=0.850
- minja: asr-r=0.700, asr-a=0.786, asr-t=0.550, benign_acc=0.900
- injecmem: asr-r=0.500, asr-a=0.400, asr-t=0.200, benign_acc=0.800

**validation checkpoint:** end-to-end pipeline produces non-trivial, differentiated asr values; 4 experiment notebooks with 20+ publication-quality figures

## dependency graph

```
step_01 (config) → step_02 (logging) → step_03 (interfaces)
    ↓                    ↓                    ↓
step_04 (wrappers) ← step_05 (watermark) ← step_06 (attacks)
    ↓                    ↓                    ↓
step_07 (defenses) → step_08 (evaluation) → step_09 (infrastructure)
```

## risk assessment

### high risk phases

- **watermarking:** algorithm correctness critical for defense effectiveness
- **attacks:** implementation must match paper methodologies
- **defenses:** integration with watermarking must be robust

### mitigation strategies

- **incremental testing:** validate each algorithm independently
- **literature review:** consult original papers for implementation details
- **modular design:** isolate high-risk components for easier rollback

## validation checkpoints

### checkpoint 1: foundation complete

- configuration loads
- logging works
- basic project structure functional

### checkpoint 2: interfaces defined

- abstract classes implemented
- inheritance patterns established
- type safety enforced

### checkpoint 3: systems integrated

- memory wrappers functional
- external apis accessible
- error handling robust

### checkpoint 4: algorithms working

- watermarking embeds/detects
- attacks poison memory
- defenses mitigate threats

### checkpoint 5: evaluation ready

- metrics calculate correctly
- benchmarks run
- results analyzable

### checkpoint 6: research operational

- experiments automated
- visualizations generated
- documentation complete

## contingency planning

### timeline slippage

- **buffer weeks:** weeks 23-24 allocated for catch-up
- **parallel work:** independent components can be developed simultaneously
- **scope adjustment:** nice-to-have features can be deferred

### technical challenges

- **api issues:** fallback to mock implementations
- **performance problems:** optimize incrementally
- **compatibility issues:** version pinning and virtual environments

### rollback points

- **daily commits:** git history allows granular rollback
- **feature branches:** isolate experimental changes
- **backup configs:** preserve working configurations

## success metrics

### quantitative

- **code coverage:** >80% for core modules
- **test pass rate:** >95% for implemented features
- **performance:** <10% overhead for defenses
- **reproducibility:** experiments run consistently

### qualitative

- **modularity:** components easily swappable
- **documentation:** all code well-documented
- **maintainability:** clear code structure
- **research value:** results support publication goals

## next steps

proceed to write_or_refactor phase with step_01 (configuration management) as the first implementation target. focus on establishing solid foundations before moving to higher-risk algorithm implementations.
