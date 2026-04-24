# publication strategy: memory agent security

**author**: ishrith gowda, uc berkeley
**last updated**: 2026-04-24
**paper**: *memsad: gradient-coupled anomaly detection for memory poisoning in retrieval-augmented agents* (double-blind, neurips 2026)

---

## 0. current status

- paper locked at 9-page main body + 20-page appendix (31 pages total, `docs/neurips2026/main.tex`)
- phase 35 (mechanism + process figures per dr. xuandong zhao's review) merged as pr #72, closing issue #71
- abstract deadline **may 4, 2026**; full paper deadline **may 6, 2026**
- primary submission target: **neurips 2026**
- backup plan if rejected: iclr 2027 (~oct 2026 deadline) with revision window for reviewer feedback
- interactive demo live at https://ishrith-gowda-memsad-demo.hf.space/

---

## 1. venue ranking by prestige (h5-index)

the optimization criterion is **prestige, impact factor, and resume impressiveness** — not research fit. the h5-index is the primary ranking metric (google scholar, covering 2020-2024 publications).

| rank | venue | h5-index | type | fit |
|------|-------|----------|------|-----|
| 1 | **neurips** | 371 | ml conference | strong (safety/robustness track) |
| 2 | **iclr** | 362 | ml conference | strong (trustworthy ml) |
| 3 | **icml** | 272 | ml conference | moderate |
| 4 | **aaai** | ~180 | ai conference | moderate |
| 5 | **acl** | 236 | nlp conference | moderate (rag/llm angle) |
| 6 | **emnlp** | ~160 | nlp conference | moderate |
| 7 | **ieee s&p** | 98 | security conference | very strong |
| 8 | **acm ccs** | 93 | security conference | very strong |
| 9 | **usenix security** | 92 | security conference | very strong |
| 10 | **ndss** | ~70 | security conference | strong |
| 11 | **ieee satml** | ~30 | ml+security | perfect fit |

**key insight**: neurips/iclr/icml are 3-4x more prestigious by h5-index than the top security venues. a neurips publication carries significantly more weight on a resume for any ml/ai role, recruiting pipeline, or academic profile than ccs or s&p.

---

## 2. deadline calendar (actionable, as of march 23, 2026)

### near-term (next 6 months)

| venue | abstract deadline | paper deadline | notification | conference |
|-------|-------------------|---------------|--------------|------------|
| **ndss 2027 c1** | apr 16, 2026 | apr 23, 2026 | jul 2, 2026 | feb 23-27, 2027 |
| **acm ccs 2026 c2** | apr 22, 2026 | apr 29, 2026 | ~jul 2026 | oct/nov 2026 |
| **neurips 2026** | may 4, 2026 | may 6, 2026 | ~sep 2026 | dec 2026 |
| **ieee s&p 2027 c1** | may 29, 2026 | jun 5, 2026 | ~oct 2026 | may 18-21, 2027 |
| **usenix sec 2027 c1** | aug 18, 2026 | aug 25, 2026 | ~dec 2026 | aug 11-13, 2027 |

### later (6-12 months out)

| venue | expected deadline | conference |
|-------|-------------------|------------|
| **iclr 2027** | ~oct 2026 | apr/may 2027 |
| **ndss 2027 c2** | aug 6, 2026 | feb 23-27, 2027 |
| **ieee s&p 2027 c2** | nov 13, 2026 | may 18-21, 2027 |
| **icml 2027** | ~jan 2027 | jul 2027 |
| **usenix sec 2027 c2** | jan 26, 2027 | aug 11-13, 2027 |
| **neurips 2027** | ~may 2027 | dec 2027 |

---

## 3. dual submission rules (critical constraints)

- **cannot** submit to two archival conferences simultaneously
- **can** post to arxiv at any time (all top venues allow this)
- **can** submit a short 4-page version to a non-archival workshop while a full paper is under review at a conference
- **can** submit an extended journal version AFTER conference acceptance (25-30% new material required)
- workshop papers at neurips/icml/iclr do not count as prior publication if the workshop has no formal proceedings

---

## 4. optimal publication strategy

### primary target: neurips 2026

**rationale**: highest achievable prestige (h5: 371). deadline is may 6, 2026 — 6 weeks away. the paper content (28 pages, 24 figures, 8 tables, 9 sections) is fully written and needs reformatting to neurips style (9 pages main + unlimited appendix). the safety/robustness track is a natural fit for our attack-defense evaluation framework.

**risk assessment**: neurips acceptance rate is ~25-26%. single-author papers face implicit bias. however, the work is comprehensive (4 attacks, 11 defenses, novel sad defense, adaptive adversary analysis, statistical rigor) and includes a clear novel contribution (gradient tension theorem, triggered calibration).

### backup cascade (if neurips rejects)

neurips notification is ~september 2026. the following cascade ensures no dead time:

1. **neurips 2026** (may 6) -> notification ~sep 2026
2. if rejected -> **iclr 2027** (~oct 2026) -> notification ~jan 2027
3. if rejected -> **icml 2027** (~jan 2027) -> notification ~may 2027
4. if rejected -> **neurips 2027** (~may 2027)

at each stage, incorporate reviewer feedback to strengthen the paper. the paper only improves with each cycle.

### parallel non-archival submissions (allowed)

while the full paper is under review at any conference above:

- **neurips 2026 workshop** on ml safety/trustworthy ai/adversarial ml (~sep 2026 deadline): submit a 4-page extended abstract focusing on the sad defense contribution alone. this is explicitly allowed by neurips dual-submission policy for non-archival workshops.
- **arxiv preprint**: post immediately (april 2026) to establish priority and accumulate citations.

### journal track (after conference acceptance)

once the conference paper is accepted:

- **tmlr** (rolling submissions, ~76-day decisions, 62% acceptance): submit extended version with 25-30% new material. the graph memory attacks, multi-agent propagation, and openai agent evaluation sections from the appendix provide sufficient extension material.
- alternative: **ieee tifs** (h5: 98, top security journal) for the security angle.
- tmlr papers with j2c certification can be presented at neurips/icml/iclr via the journal-to-conference track (additional presentation venue).

---

## 5. maximum legitimate publication output

| # | publication | venue | type | timing |
|---|-------------|-------|------|--------|
| 1 | arxiv preprint | arxiv | preprint | april 2026 |
| 2 | conference paper | neurips 2026 (or iclr/icml 2027) | top-tier archival | dec 2026 (or later) |
| 3 | workshop paper | neurips 2026 workshop | non-archival | dec 2026 |
| 4 | journal extension | tmlr or ieee tifs | journal | 2027 |

**total: 1 arxiv + 1 top-tier conference + 1 workshop + 1 journal = 4 distinct publications**

all from a single research project, all following standard academic ethics and dual-submission rules.

---

## 6. manuscript preparation plan

### phase 30a: neurips 2026 submission (primary)

- **format**: neurips 2026 style (neurips_2026.sty), 9 pages main + references + appendix
- **page budget**: ~9 pages main body, unlimited appendix
- **key sections to fit in 9 pages**:
  - introduction (1.5 pages)
  - related work (1 page)
  - threat model + attack formalization (1 page)
  - sad defense + gradient tension proposition (1.5 pages)
  - experimental setup + results (2.5 pages)
  - discussion + conclusion (1.5 pages)
- **appendix**: ablation studies, full attack-defense matrix, encoder generalization, adaptive adversary details, proofs, additional figures
- **what to emphasize for neurips**: novel defense method (sad), theoretical contribution (gradient tension), comprehensive empirical evaluation, practical implications for llm safety

### phase 30b: arxiv preprint

- **format**: arxiv-compatible (current full 28-page version is suitable)
- **timing**: submit to arxiv 1-2 weeks before neurips deadline
- **category**: cs.CR (cryptography and security) + cs.LG (machine learning) + cs.AI (artificial intelligence)

### phase 30c: workshop paper

- **format**: 4-page extended abstract
- **focus**: sad defense contribution only (novel method + key results)
- **timing**: submit after workshop cfps are announced (~aug-sep 2026)

---

## 7. action items (immediate)

1. download neurips 2026 style files from neurips.cc
2. restructure paper into 9-page neurips format
3. select and condense figures (pick 4-5 most impactful)
4. write neurips-specific abstract (150 words max)
5. post arxiv preprint (~late april 2026)
6. submit to neurips by may 4 (abstract) / may 6 (full paper)

---

## sources

- [neurips 2026 dates and deadlines](https://neurips.cc/Conferences/2026/Dates)
- [neurips 2026 call for papers](https://neurips.cc/Conferences/2026/CallForPapers)
- [acm ccs 2026 call for papers](https://www.sigsac.org/ccs/CCS2026/call-for/call-for-papers.html)
- [ieee s&p 2027](https://sec-deadlines.github.io/)
- [usenix security 2027](https://www.usenix.org/conference/usenixsecurity27)
- [ndss 2027](https://www.ndss-symposium.org/)
- [icml 2026 dates](https://icml.cc/Conferences/2026/Dates)
- [iclr 2026 dates](https://iclr.cc/Conferences/2026/Dates)
- [tmlr editorial policies](https://jmlr.org/tmlr/editorial-policies.html)
- [neurips/iclr/icml journal-to-conference track](https://neurips.cc/public/JournalToConference)
- [google scholar metrics - ai](https://scholar.google.es/citations?view_op=top_venues&hl=en&vq=eng_artificialintelligence)
- [google scholar metrics - security](https://scholar.google.es/citations?view_op=top_venues&hl=en&vq=eng_computersecuritycryptography)
- [security conference deadlines](https://sec-deadlines.github.io/)
