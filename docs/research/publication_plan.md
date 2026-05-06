# MEMSAD Publication Plan

**Paper:** MEMSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents
**Author:** Ishrith Gowda (sole author)
**Affiliation footnote:** Song Lab, Berkeley AI Research, UC Berkeley (approved by X. Zhao, 2026-04-27)
**Format:** 28 pages, NeurIPS 2026 style, currently in `[preprint]` (de-anonymized) mode
**Status as of 2026-05-01:** locked for submission; arxiv-ready

---

## 0. Reading guide

This is the operational publication plan. It is structured as a claim ladder (preprint → top venue → backup → workshop visibility) with explicit rejection contingencies, deadline tracking, and a promotion strategy. Every date below is annotated with one of three tags:

- **CONFIRMED** — verified against an official venue source
- **VERIFY** — based on Jan-2026 knowledge of typical timelines; **MUST be re-checked on the venue website before any action is taken**
- **ESTIMATE** — pattern-based projection only; do not rely on without verification

No deadline in this document is currently CONFIRMED. **Before submitting anywhere, run the deadline verification checklist in §8.**

---

## 1. Executive summary

**Strategy in one paragraph.** Establish public priority via arXiv immediately. Submit to NeurIPS 2026 main track as the primary venue (best fit: theory-forward ML-security paper, exactly the work NeurIPS rewards, and the paper is already formatted for it). Run a parallel workshop submission to a NeurIPS 2026 satellite (SafeGenAI / AdvML / ML Safety) for community visibility regardless of main-track outcome. If NeurIPS main rejects, route to ICLR 2027 (next top-tier ML venue) — do *not* attempt USENIX Security or IEEE S&P unless the paper is substantially restructured for a systems-security audience, because (a) those venues prefer empirical systems work over theory, and (b) the existing arXiv preprint will run afoul of their stricter anonymity / no-public-disclosure policies. Treat TMLR as a last-resort path if both ML conferences reject — it is rolling, has good acceptance rates for rigorous theoretical work, and is openly favoured for thorough papers that don't fit a 9-page conference format.

**Why not security venues as primary.** This paper has 6 theorems, a minimax lower bound, a hardness reduction, an online regret bound, and a Fisher-Rao geometric argument. That is an ML-theory paper with an empirical security application. Top security conferences (USENIX, S&P, CCS, NDSS) prefer 13-page system-and-empirical work; theorems are tolerated, not rewarded. Reviewers there will under-credit the formal contributions. NeurIPS reviewers will engage with them. Single-author papers are also more common at NeurIPS than at top security venues, where institutional multi-author papers dominate.

**Time horizon.** Target arXiv this week, NeurIPS submission within the May 2026 window if the deadline holds, NeurIPS reviews and decision by end-Sept 2026, NeurIPS conference December 2026. If rejected, ICLR 2027 deadline late-September → April-2027 conference. Workshop track runs August-October 2026 in parallel.

---

## 2. Paper positioning and community fit

### 2.1 What this paper actually is

A theory-led ML-security paper. Specifically:
- **Formal contributions:** 6 theorems including gradient coupling, certified radius, minimax lower bound (Le Cam), online regret bound, Fisher-Rao detection-evasion metric, hardness reduction from CVP.
- **Empirical contributions:** 3×5 attack-defense matrix, bootstrap 95% CIs, Bonferroni-corrected hypothesis tests, Clopper-Pearson FPR validation (20 trials, n=1000), multi-encoder generalization (6 encoders), tool-use eval (GPT-4o-mini), SIR multi-agent propagation simulation, NQ cross-corpus generalization, Mem0 production validation.
- **Community placement:** sits at the intersection of (a) trustworthy ML, (b) adversarial ML / robust learning, (c) RAG security, (d) LLM agent safety. The first two are NeurIPS/ICLR-native; the last two are increasingly visible at NeurIPS/ICLR but originated in security venues.

### 2.2 Single-author paper consideration

Sole-authored papers signal independence and ownership but raise visibility risk: no co-author network to share the paper, no institutional grouping, and no senior name to catch reviewer attention. Mitigate by: aggressive arXiv promotion, posting on Twitter/X (@research community), submitting to workshops with public talks, and emailing the paper directly to 5–10 known researchers in the area for feedback after arXiv release (see §7).

### 2.3 Affiliation considerations

The Song-Lab footnote is the only institutional anchor. Do not overstate it elsewhere (cover letter, talks, social posts) — keep wording aligned with what was approved (see [project_authorship.md](../../../Users/IshrithG/.claude/projects/-Volumes-usb-drive-memory-agent-security/memory/project_authorship.md)).

---

## 3. Venue tier analysis

### Tier 1 — Primary target

| Venue | Fit | Prestige (ML-Sec) | Deadline | Conference dates | Status |
|---|---|---|---|---|---|
| **NeurIPS 2026** main track | Excellent — theory-heavy ML-security paper, format already matches | A+ | Abstract ~mid-May 2026, paper ~May 22 2026 (**VERIFY** at neurips.cc) | December 2026, Sydney | **PRIMARY TARGET** |

### Tier 2 — Backup ML conferences (sequential, not parallel — no dual submission)

| Venue | Fit | Prestige | Deadline | Status |
|---|---|---|---|---|
| **ICLR 2027** | Excellent — same audience as NeurIPS, slightly more theory-friendly | A+ | Late Sept 2026 (**ESTIMATE**) | First fallback if NeurIPS rejects |
| **ICML 2027** | Excellent | A+ | Late Jan 2027 (**ESTIMATE**) | Second fallback |
| **AAAI 2027** | Good but lower prestige | A | Aug 2026 (**ESTIMATE**) | Third fallback |
| **COLM 2026** (Conference on Language Modeling) | Moderate — language-modeling-centric, but security/safety tracks accept this work | A- | March 2026 (**LIKELY PASSED**) | Probably out of cycle |

### Tier 3 — Security conferences (only if substantially restructured)

| Venue | Fit | Prestige | Deadline | Status |
|---|---|---|---|---|
| **IEEE S&P 2027** | Moderate — would need rewrite emphasizing systems+empirical | A+ | Summer + fall cycles, ~June 2026 (**VERIFY**) | Skip for v1; consider for extended/v2 |
| **USENIX Security 2027** | Moderate — same caveat | A+ | Multiple cycles (Feb/June/Oct) (**VERIFY**) | Skip — arXiv preprint may conflict with anonymity rules |
| **ACM CCS 2026** | Moderate | A+ | Cycle 2 ~spring 2026 (**LIKELY PASSED OR IMMINENT — VERIFY**) | Skip for v1 |
| **NDSS 2027** | Moderate | A | ~June 2026 (**VERIFY**) | Skip for v1 |

### Tier 4 — Specialty venues

| Venue | Fit | Prestige | Notes |
|---|---|---|---|
| **SaTML 2027** (Conf. on Secure and Trustworthy ML) | Strong — IEEE-sponsored ML-security conference; perfect topical match | A- (rising) | Strong backup if NeurIPS+ICLR fail; emerging venue but high-quality |
| **TMLR** | Strong — rolling submission, accepts thorough theory+empirical work, no length limit | A- (open access; high quality but no conference acceptance rate prestige) | Last resort or supplement |
| **ALT 2027 / COLT 2026** | Pure-theory only — paper is too applied | A | Not a fit |

### Tier 5 — Workshops (parallel to main submissions; non-archival)

Workshops should run **in parallel** with the NeurIPS submission to maximize community visibility regardless of main-track outcome. Most NeurIPS workshops are non-archival, so they don't conflict with main-track or with later resubmission elsewhere.

| Workshop | Co-located with | Deadline | Fit |
|---|---|---|---|
| **AdvML-Frontiers @ NeurIPS 2026** | NeurIPS 2026 | ~Sept 2026 (**ESTIMATE**) | Excellent |
| **SafeGenAI @ NeurIPS 2026** | NeurIPS 2026 | ~Sept 2026 (**ESTIMATE**) | Excellent |
| **ML Safety / Trustworthy ML @ NeurIPS 2026** | NeurIPS 2026 | ~Sept 2026 (**ESTIMATE**) | Excellent |
| **AISec @ CCS 2026** | CCS 2026 | ~July 2026 (**ESTIMATE**) | Excellent — security workshop, accepts theory-applied papers |
| **SatML / TrojAI** | Various | Varies | Specialty, good fit |

---

## 4. Primary plan (target sequence)

### Phase A — arXiv preprint (this week)

**Goal:** establish public priority, enable downstream venue submissions, become discoverable.

1. Final compile check: `\usepackage[preprint]{neurips_2026}` in [docs/neurips2026/main.tex:4](../../docs/neurips2026/main.tex#L4); affiliation footnote rendered correctly; 28 pages; no anonymous strings remaining anywhere; bibliography clean.
2. arXiv submission, primary category **cs.CR** (Cryptography and Security), cross-list **cs.LG** (Machine Learning) and **cs.AI** (Artificial Intelligence). The cs.CR primary placement signals security audience; cross-listing reaches ML researchers.
3. Title abstract on arXiv must match paper exactly. Do **not** edit the abstract for arXiv to be more flashy — divergence between arXiv and the camera-ready PDF causes problems.
4. Once arXiv ID is assigned, post a single Twitter/X thread (template in §7) and email 5–10 named researchers (list in §7).

**Action items:**
- [ ] Verify no anonymous strings: `grep -i "anonymous" docs/neurips2026/main.tex`
- [ ] Verify references compile clean and no `??` remain in PDF
- [ ] Prepare arXiv abstract (~250 words from current abstract)
- [ ] Submit to arXiv; wait for ID
- [ ] Post Twitter/X thread; email researchers

### Phase B — NeurIPS 2026 main-track submission (May 2026 if deadline holds)

**Goal:** primary publication target.

1. **VERIFY DEADLINE:** check https://neurips.cc/Conferences/2026 for the exact abstract and full-paper deadlines and the OpenReview portal.
2. Switch back to anonymous mode for the submission copy: `\usepackage{neurips_2026}` (no options) and replace author block with `\author{Anonymous Author(s)}`. Keep a separate de-anonymized branch / copy for the arXiv preprint. **NeurIPS allows arXiv preprints and does not require them to be removed during review** — but the submission copy itself must be anonymous.
3. Verify all self-references in the paper are anonymous (no "in our prior work [Gowda 2026]", no leakage via figure captions or filename strings).
4. Submit. Track reviewer feedback through OpenReview.

**Action items:**
- [ ] Re-verify NeurIPS 2026 deadline on the venue website
- [ ] Branch the repo for the anonymous NeurIPS submission copy: `git checkout -b submit/neurips2026`
- [ ] Switch package option and author block; verify by `grep` that no identifying string remains
- [ ] Compile and submit via OpenReview

### Phase C — Workshop submissions (Aug–Oct 2026, parallel to NeurIPS review)

**Goal:** community visibility regardless of NeurIPS outcome; talk slots; reviewer feedback before potential resubmission.

1. Pick 2 workshops max (more is diminishing returns and presentation overhead). Recommended: **AdvML-Frontiers** + **SafeGenAI**, both at NeurIPS 2026.
2. Workshop versions are typically 4–8 pages — produce a short version that focuses on the gradient coupling theorem + empirical headline result. Keep the full-paper arXiv as the reference.
3. Workshops are usually non-archival; verify each workshop's specific policy before submitting.

### Phase D — Decision branch

**If NeurIPS accepts:** prepare camera-ready (de-anonymize), prepare conference talk + poster, plan extended TMLR / journal version (§6).

**If NeurIPS rejects:** triage feedback honestly. Substantial revision based on reviews → ICLR 2027 (deadline ~late September 2026 — **VERIFY**). If reviews are mixed, also send to AAAI 2027 (~Aug 2026) as a parallel option, since AAAI moves on a faster cycle.

**If both reject:** route to TMLR (rolling; no conference deadline pressure). TMLR allows longer papers, which fits the 28-page format without truncation.

---

## 5. Rejection contingency tree

```
arXiv (always done first; ~no risk)
  │
  ▼
NeurIPS 2026 main
  │
  ├── ACCEPT → camera-ready + talk + poster + extended journal version (§6)
  │
  └── REJECT
        │
        ├── if reviews substantive (e.g., revise theorem proofs, add experiment)
        │     → ICLR 2027 (~Sept 2026)
        │
        └── if reviews dismissive (community fit issue, not technical)
              → SaTML 2027 (better topical fit) OR TMLR (rolling, no deadline)

ICLR 2027 (if reached)
  │
  ├── ACCEPT → camera-ready, May-2027 conference
  │
  └── REJECT → TMLR

```

Workshop track (NeurIPS 2026 satellites) runs in parallel and is independent of the main outcome.

---

## 6. Camera-ready and extended versions

### 6.1 If accepted at a conference

- Conference camera-ready usually requires 9-page main + appendix; current paper is already 28 pages with a clean main/appendix split. The main-body length is locked at 9 pages; appendix can grow.
- Update `\usepackage{neurips_2026}` to `\usepackage[final,main]{neurips_2026}` for camera-ready (verify exact option in the year's style file).
- De-anonymize fully; verify the affiliation footnote remains exactly as approved.

### 6.2 Extended version (TMLR or journal)

After conference acceptance, prepare an extended TMLR-style version that:
- Restores any content cut for page limits
- Adds the multi-encoder full table that was abbreviated
- Includes a "deployment study" section with longer-running production validation results
- Adds reviewer-requested experiments from conference reviews

Target: TMLR submission ~3–6 months after conference acceptance.

### 6.3 Software / artifact release

NeurIPS, ICLR, ICML increasingly weight reproducibility. Plan:
- Prepare a clean public GitHub repo (separate from the working repo) with: README, install instructions, paper-replication scripts, dataset generators, evaluation pipeline, pretrained encoder configs.
- Use a permissive license (MIT or Apache-2.0).
- Include a `REPRODUCE.md` that lists exact commands to regenerate every table and figure in the paper.
- Apply for the NeurIPS / ICLR Reproducibility / Artifact badge.

Actions are deferred until conference acceptance, but the public-repo structure should be prepared in parallel during the review window so a public release can happen within 1 week of acceptance announcement.

---

## 7. Promotion and visibility strategy

### 7.1 arXiv announcement (week of arXiv submission)

**Twitter/X thread (template, 6–8 tweets):**
1. Title + arXiv link + 1-sentence hook ("LLM agents store memory. Adversaries can poison it. We give the first formally-guaranteed defense.")
2. The threat: 1 tweet on AgentPoison/MINJA/InjecMEM and what compound exposure means
3. The contribution: gradient coupling theorem, plain-English version
4. The empirical headline: composite defense reaches TPR=1.00 / FPR=0.00; synonym evasion is the formal frontier
5. Why this matters for deployed agents (Mem0, A-MEM, etc.)
6. Open question: closing the synonym gap
7. Code/data note (release upon publication)
8. Tag relevant researchers (with restraint — 3–4 max)

### 7.2 Direct researcher outreach (post-arXiv, within 2 weeks)

Email 5–10 researchers actively working in adversarial ML, LLM security, RAG, or agent safety. Format: 100–150 words, "I just posted a paper that may be relevant to your work on X; here's the arXiv link; happy to discuss." Include the **specific** connection to their work — generic emails get ignored.

Candidate target list (verify each is still active in the area before emailing):
- Senior researchers cited heavily in the paper (Chen, Dong, Zou, Zhao at top venues)
- Lab heads at Berkeley, Stanford, MIT, CMU active in adversarial ML / agent safety
- Industry research leads at Anthropic, OpenAI, Google DeepMind, Meta AI working on agent safety

Tone: respectful, brief, asking for feedback rather than endorsement. Do **not** ask for citations or downstream favors.

### 7.3 Talks and presentations

- Submit to AdvML-Frontiers / SafeGenAI workshops (§4 Phase C) — workshop talks are higher-impact than posters.
- If accepted at a NeurIPS workshop, prepare a 10-min talk and a poster.
- After the conference, propose a guest talk at: Berkeley reading groups, Stanford security lunch, Anthropic/OpenAI internal security colloquia (via warm intros).

### 7.4 Blog post

Write a 1500-word blog post explaining the gradient coupling intuition without LaTeX, after arXiv is up. Host on a personal site or Medium. Cross-link from Twitter and the GitHub README. Blog posts frequently outperform paper PDFs in citation funnel for early-career researchers.

---

## 8. Deadline verification checklist (do this before any submission)

This is the most important section. Every date in §3 and §4 is unverified. **Before any submission action, verify the exact current deadline on the venue's official site.**

- [ ] **NeurIPS 2026** — visit https://neurips.cc/Conferences/2026, locate "Important Dates"; record abstract deadline, full-paper deadline, supplementary deadline, anonymity policy, dual-submission policy
- [ ] **ICLR 2027** — visit https://iclr.cc, locate next-year deadlines (may not be posted yet as of May 2026; check https://openreview.net for the venue page)
- [ ] **ICML 2027** — visit https://icml.cc
- [ ] **AAAI 2027** — visit https://aaai.org/conference/aaai/
- [ ] **SaTML 2027** — visit https://satml.org
- [ ] **TMLR** — confirm submission portal at https://jmlr.org/tmlr (rolling)
- [ ] **NeurIPS 2026 workshop list** — published ~July/August 2026 typically; check NeurIPS site
- [ ] **arXiv** — confirm category policy at https://arxiv.org/help/cross_listing for cs.CR + cs.LG + cs.AI cross-listing

For each verified deadline, update §3 and §4 inline (replace **VERIFY** / **ESTIMATE** with **CONFIRMED — [date]**).

---

## 9. Constraints and red flags

- **Dual submission is forbidden** at every Tier-1 venue. Do not submit to two main-track venues simultaneously. Workshops (non-archival) do not count.
- **arXiv ↔ anonymity policies vary.** NeurIPS, ICLR, ICML, TMLR all allow concurrent arXiv preprints. USENIX Security, IEEE S&P, ACM CCS, NDSS have stricter rules — most allow preprints but require the submission copy to be anonymous and may forbid public version updates during review. **VERIFY** before submitting to any security venue.
- **The Song-Lab affiliation footnote** is the only institutional claim and was approved verbatim. Do not modify it without re-approval from X. Zhao.
- **The paper is sole-authored.** Do not add reviewers, mentors, or "thanks" lines that imply co-contribution beyond the approved footnote and a generic acknowledgments paragraph.
- **The paper is currently 28 pages.** NeurIPS main-paper limit is 9 pages excluding references; the current main body is 9 pages (verified at lock time) and the rest is appendix. Re-verify the page split before the NeurIPS submission compile.
- **Reviewer leakage risk.** When emailing researchers (§7.2) before NeurIPS reviews start, avoid emailing anyone likely to be assigned as a reviewer (i.e., any senior person at a major lab who is actively in this exact subfield). Wait until after submission.

---

## 10. Status tracker

| Milestone | Target date | Status | Owner |
|---|---|---|---|
| Affiliation footnote approved | 2026-04-27 | ✓ DONE | — |
| Paper locked (28 pages) | 2026-04-25 | ✓ DONE | — |
| Switch to preprint mode | 2026-04-30 | ✓ DONE | — |
| Supplementary zip prepared (anonymized, 444 KB) | 2026-05-04 | ✓ DONE | — |
| NeurIPS 2026 submission | 2026-05-05 | ✓ DONE | — |
| arXiv submission | within 72 hours | TODO — NEXT | self |
| arXiv announcement (Twitter + emails) | within 1 week of arXiv ID | TODO | self |
| NeurIPS workshop submissions | Aug–Oct 2026 | TODO | self |
| Public GitHub repo prepared | during review window | TODO | self |
| NeurIPS 2026 decision | end-Sept 2026 | TBD | — |

---

## 11. References to other project documents

- [docs/research/progress_report.md](progress_report.md) — phase-by-phase research history through phase 34
- [docs/neurips2026/main.tex](../neurips2026/main.tex) — the paper source
- [memory/project_authorship.md](../../../Users/IshrithG/.claude/projects/-Volumes-usb-drive-memory-agent-security/memory/project_authorship.md) — authorship + footnote approval record

---

*Last updated: 2026-05-01. Update this file as deadlines verify and milestones complete. When NeurIPS deadline is confirmed, replace all **VERIFY** tags in §3 and §4 with **CONFIRMED — [date]**.*
