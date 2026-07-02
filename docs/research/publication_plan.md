# MEMSAD Publication Plan

**Paper:** MEMSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents
**Author:** Ishrith Gowda (sole author)
**Affiliation footnote:** Song Lab, Berkeley AI Research, UC Berkeley (approved by X. Zhao, 2026-04-27)
**Format:** 28 pages, NeurIPS 2026 style, currently in `[preprint]` (de-anonymized) mode
**Status as of 2026-07-02:** NeurIPS 2026 submitted (May 5, confirmed via `f9941d5` baseline commit); **arXiv v1 live at https://arxiv.org/abs/2605.03482**; in NeurIPS review silence period (reviews est. late July, notification Sep 24 CONFIRMED); PR #77 (pat-feedback) merged and PR #76 closed as superseded 2026-07-02; arXiv v2 upload is the top pending user action (replace against 2605.03482); **AAAI 2027 fallback ruled out 2026-07-02 — dual-submission conflict (see §14)**; workshop list watch begins Jul 11; announcement/blog/reproduction tracks re-scheduled in §14

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
| **NeurIPS 2026** main track | Excellent — theory-heavy ML-security paper, format already matches | A+ | Abstract May 4 2026, paper May 6 2026 (**CONFIRMED — 2026-06-08** via neurips.cc/Conferences/2026/CallForPapers); notification Sep 24 2026 | Dec 6-12 2026, Sydney + Atlanta + Paris (multi-site, **CONFIRMED — 2026-06-08**) | **PRIMARY TARGET — SUBMITTED** |

### Tier 2 — Backup ML conferences (sequential, not parallel — no dual submission)

| Venue | Fit | Prestige | Deadline | Status |
|---|---|---|---|---|
| **ICLR 2027** | Excellent — same audience as NeurIPS, slightly more theory-friendly | A+ | Late Sept / Oct 2026 (**ESTIMATE — venue page 404 as of 2026-06-08**) | First fallback if NeurIPS rejects |
| **ICML 2027** | Excellent | A+ | Late Jan 2027 (**ESTIMATE — venue page 404 as of 2026-06-08**) | Second fallback |
| **AAAI 2027** | Good but lower prestige (h5 ~232) | A | Abstract Jul 21 2026 / paper **Jul 28 2026** (**CONFIRMED — 2026-06-08** via aaai.org/conference/aaai/aaai-27/); conf Feb 16-23 2027 Montréal | Third fallback — **earlier than originally estimated, ~7 weeks out** |
| **COLM 2026** (Conference on Language Modeling) | Moderate — language-modeling-centric, but security/safety tracks accept this work | A- | Paper Mar 31 2026 (**CONFIRMED PASSED — 2026-06-08**); conf Oct 6-9 2026 SF | Out of cycle |

### Tier 3 — Security conferences (only if substantially restructured)

| Venue | Fit | Prestige | Deadline | Status |
|---|---|---|---|---|
| **IEEE S&P 2027** | Moderate — would need rewrite emphasizing systems+empirical | A+ | C1 abstract **Jun 4 2026** (passed), C1 paper **Jun 11 2026** (3 days out as of 2026-06-08); C2 paper **Nov 17 2026** (**CONFIRMED — 2026-06-08** via sp2027.ieee-security.org/cfpapers.html); conf Montréal TBA | Skip for v1; consider for extended/v2 — note arXiv during review permitted with "no widely advertising" caveat |
| **USENIX Security 2027** | Moderate — same caveat | A+ | C1 abstract **Aug 18 2026** / paper **Aug 25 2026** (**CONFIRMED — 2026-06-08** via usenix.org/conference/usenixsecurity27); conf Aug 11-13 2027 Denver | Skip — anonymity/preprint policy not yet officially confirmed for arXiv'd manuscripts |
| **ACM CCS 2026** | Moderate | A+ | Cycle 2 paper **Apr 29 2026** (**CONFIRMED PASSED — 2026-06-08**); conf Nov 15-19 2026 The Hague | Skip for v1 |
| **NDSS 2027** | Moderate | A | C1 paper **May 6 2026** (passed); C2 paper **Aug 19 2026** (**CONFIRMED — 2026-06-08** via ndss-symposium.org/ndss2027); conf Mar 22-26 2027 Seoul; preprint "not encouraged" but permitted | Skip for v1 |

### Tier 4 — Specialty venues

| Venue | Fit | Prestige | Notes |
|---|---|---|---|
| **SaTML 2027** (Conf. on Secure and Trustworthy ML) | Strong topical match — but **DOES NOT YET EXIST as of 2026-06-08** | A- (rising) | satml.org says "looking for a host for SaTML 2027" — cannot rely on this as a backup until a host is announced and deadlines posted |
| **TMLR** | Strong — rolling submission, accepts thorough theory+empirical work, no length limit | A- (open access; high quality but no conference acceptance rate prestige) | Last resort or supplement. "76-day decision / 62% acceptance" community lore — official stats not published. |
| **ALT 2027 / COLT 2026** | Pure-theory only — paper is too applied | A | Not a fit |

### Tier 5 — Workshops (parallel to main submissions; non-archival)

Workshops should run **in parallel** with the NeurIPS submission to maximize community visibility regardless of main-track outcome. Most NeurIPS workshops are non-archival, so they don't conflict with main-track or with later resubmission elsewhere.

**NeurIPS 2026 workshop track (CONFIRMED — 2026-06-08 via neurips.cc/Conferences/2026/CallForWorkshops):**
- Workshop proposals due Jun 6 2026 (passed); proposal acceptance Jul 11 2026; **workshop paper-contribution deadline Aug 29 2026** (this is the date you care about).
- **The list of accepted NeurIPS 2026 workshops is NOT yet posted** — it appears after Jul 11. Cannot pre-commit to specific workshops.

| Workshop | Co-located with | Deadline | Fit | 2026 status |
|---|---|---|---|---|
| **AdvML-Frontiers** | Historically NeurIPS; AdvML-Frontiers 2026 is at **COLM 2026** | TBD | Excellent | NOT at NeurIPS 2026 — may resurface at NeurIPS 2027 |
| **SafeGenAI** | Ran at NeurIPS 2024 | TBD | Excellent | **NeurIPS 2026 existence UNVERIFIED** — wait for workshop list |
| **SoLaR** (Socially Responsible Language Modeling Research) | Historically NeurIPS / ICLR | TBD | Strong | **NeurIPS 2026 existence UNVERIFIED** |
| **Generic "ML Safety / Trustworthy ML / Adversarial ML"** | NeurIPS 2026 | Aug 29 2026 once workshop is accepted | Excellent | Watch workshop list after Jul 11 |
| **AISec @ CCS 2026** | CCS 2026 (Nov 15-19 The Hague) | ~July 2026 (**ESTIMATE**) | Excellent — security workshop, accepts theory-applied papers | Verify when CFP posts |
| **TrojAI** | Various | Varies | Specialty, good fit | Verify current year |

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
| arXiv v1 submission | within 72 hours | ✓ DONE — live at https://arxiv.org/abs/2605.03482 (confirmed by author 2026-06-08) | self |
| pat-feedback proof corrections | 2026-05-07 onward | ✓ DONE locally — **PR #77 open since 2026-05-06, NOT MERGED**. 20+ fix commits on `feat/phase-37-pat-feedback` + 4 uncommitted modified files | self |
| arXiv v2 source bundle rebuilt | 2026-05-?? | ✓ DONE locally (commits a8ad623, 9ed8684) | self |
| arXiv v2 UPLOAD as revision against 2605.03482 | **re-targeted 2026-07-03** | ⏳ TOP PENDING USER ACTION — upload via arxiv.org "Replace" against 2605.03482 (source bundle ready after PR #77 merge) | self |
| Merge PR #77 (pat-feedback) to main + close | by 2026-05-15 | ✓ DONE 2026-07-02 (7.5 weeks late; all CI green at merge) | self |
| Merge PR #76 (fig 2 vertical) or close | by 2026-04-30 | ✓ CLOSED 2026-07-02 as superseded — both fig-2 commits (dcd67b1, 9ff4177) are contained in PR #77's branch | self |
| arXiv announcement (Twitter + emails) | **re-targeted: within 72h of v2 upload** | ⏳ RESCHEDULED — v1 launch window burned; run smaller "v2 with corrected proofs" push; researcher emails don't expire | self |
| Blog post (gradient coupling intuition) | **re-targeted 2026-07-15** | ⏳ RESCHEDULED | self |
| Public reproduction repo (clean release + replication guide) | staged by 2026-09-01 | ⏳ IN PROGRESS — README "Reproducing paper results" section serves as the in-repo guide; public-repo split remains | self |
| NeurIPS 2026 workshop list published | ~2026-07-11 (workshop proposal acceptance date, **CONFIRMED**) | ⏳ UPCOMING — re-verified not yet posted as of 2026-07-02 | — |
| Workshop choice (whichever NeurIPS 2026 satellites are accepted) | by 2026-07-20 | ⏳ UPCOMING — wait for list | self |
| NeurIPS 2026 reviews released | NOT YET PUBLISHED on neurips.cc — estimate late July / early Aug from prior years | ⏳ UPCOMING — not yet released as of 2026-07-02 | — |
| NeurIPS 2026 rebuttal | NOT YET PUBLISHED — estimate early-mid Aug | ⏳ UPCOMING | self |
| AAAI 2027 fallback decision (submit ~Jul 28 or skip) | by 2026-07-15 | ✓ DECIDED 2026-07-02 — **SKIP** (dual-submission conflict, see §14; AAAI paper deadline Jul 28 precedes NeurIPS notification Sep 24) | self |
| NeurIPS workshop paper deadline | **2026-08-29 CONFIRMED** | ⏳ UPCOMING | self |
| NeurIPS 2026 notification | **2026-09-24 CONFIRMED** | ⏳ UPCOMING | — |
| ICLR 2027 backup paper prep (incorporate NeurIPS reviewer feedback) | by 2026-09-30 | ⏳ NOT STARTED — must pre-stage before reviews land | self |
| ICLR 2027 submission | ICLR 2027 venue page returns 404 as of 2026-06-08; expect ~Sept-Oct historically | ⏳ CONDITIONAL on NeurIPS outcome | self |
| IEEE S&P 2027 C2 submission (if pivoting to security venue) | 2026-11-17 paper (**CONFIRMED**) | ⏳ CONDITIONAL — only if restructuring for systems-security audience | self |

---

## 11. References to other project documents

- [docs/research/progress_report.md](progress_report.md) — phase-by-phase research history through phase 34
- [docs/neurips2026/main.tex](../neurips2026/main.tex) — the paper source
- [memory/project_authorship.md](../../../Users/IshrithG/.claude/projects/-Volumes-usb-drive-memory-agent-security/memory/project_authorship.md) — authorship + footnote approval record

---

*Last updated: 2026-06-08 (deadline re-verification pass via web research; corrections logged in §13). Update this file as deadlines verify and milestones complete.*

---

## 13. Deadline re-verification — 2026-06-08

Full web-source pass against neurips.cc, aaai.org, sp2027.ieee-security.org, usenix.org, ndss-symposium.org, sigsac.org, satml.org, colmweb.org, jmlr.org, sec-deadlines.github.io, and Google Scholar Metrics.

**Confirmed correct (no change needed):**
- NeurIPS 2026: abstract May 4, paper May 6, conference Dec 6-12 — but note **multi-site (Sydney + Atlanta + Paris)**, not Sydney only. arXiv preprints permitted ("non-anonymous preprints will not result in rejection" per MainTrackHandbook).
- CCS 2026 C2 (Apr 29) passed; conference Nov 15-19 The Hague.
- USENIX Sec 2027 C1: abstract Aug 18 / paper Aug 25; conference Aug 11-13 Denver.
- COLM 2026: paper Mar 31 passed.
- TMLR: rolling, no length limit, double-blind. The "76-day / 62%" stats are community lore and not officially published — treat as estimates.
- Top-3 ML h5-index: NeurIPS=371, ICLR=362, ICML=272.

**Corrections from prior plan version:**
1. NeurIPS 2026 **notification date is Sep 24 2026** (plan had "end-Sept" — pin it).
2. NeurIPS 2026 conference is **multi-site** (Sydney + Atlanta + Paris).
3. AAAI 2027 deadlines: **abstract Jul 21 / paper Jul 28 2026** (plan said "Aug 2026" — earlier than estimated). Conference Feb 16-23 2027 Montréal.
4. AAAI h5-index is **~232** (plan said ~180 — stale by ~3 years).
5. IEEE S&P 2027 C1: **abstract Jun 4 / paper Jun 11 2026** (plan said May 29 / Jun 5; off by ~6 days). As of 2026-06-08, **C1 paper deadline has NOT yet passed — three days out.**
6. IEEE S&P 2027 C2: **paper Nov 17 2026** (plan said Nov 13).
7. NDSS 2027 C1 paper was **May 6 2026** (plan said Apr 23). NDSS 2027 C2 paper is **Aug 19 2026** (plan said Aug 13). Conference is **Mar 22-26 2027 Seoul** (plan said Feb 23-27).
8. **SaTML 2027 does not yet exist** — satml.org explicitly: "we are now looking for a host for SaTML 2027". Cannot rely on as backup.
9. **NeurIPS 2026 workshop list is NOT yet published** — proposal acceptance is Jul 11; workshop paper deadline is Aug 29. AdvML-Frontiers 2026 is at **COLM 2026, not NeurIPS**. SafeGenAI / SoLaR at NeurIPS 2026 existence is **UNVERIFIED**.

**Unverifiable from official sources as of 2026-06-08:**
- ICLR 2027 venue page returns HTTP 404. "Late Sept / Oct 2026" deadline is a historical-pattern estimate only.
- ICML 2027 venue page returns HTTP 404. "Late January 2027" is historical estimate.
- NeurIPS 2026 review/rebuttal calendar — not published on the official site. "Late July / early Aug" is a historical-pattern guess.
- AISec @ CCS 2026 deadline — workshop CFP not yet posted (verify when CFP is up).

**arXiv v1 resolution (2026-06-08):**
- arXiv v1 is confirmed live at **https://arxiv.org/abs/2605.03482**. The repo never recorded the ID — recommend adding `arxiv_id: 2605.03482` to a project-level README or to MEMORY.md so it isn't lost again.
- Recovery path: upload v2 as a "Replace" against 2605.03482 once PR #77 is merged; do NOT submit as a new arXiv entry.

**Decision implications:**
- AAAI 2027 deadline (Jul 28) is much sooner than I previously stated — if you want a parallel fallback submission, the decision-to-submit window is ~7 weeks not ~10.
- IEEE S&P 2027 C1 paper deadline is Jun 11 — three days from now. Plan still says skip Tier-3 security venues without restructuring, so this remains a skip absent a strategy pivot.
- SaTML 2027 cannot be relied on as a backup; the contingency cascade in §5 effectively becomes NeurIPS → ICLR 2027 → ICML 2027 → AAAI 2027 → TMLR.

---

## 12. Slip review — 2026-06-08

Five-week gap between paper submission (May 5) and this status update. Plan-Phase A (arXiv) launched on time; Phase B (NeurIPS) shipped on time; but everything downstream of submission silently slipped while the pat-feedback proof-correction cycle consumed cycles.

**What slipped:**

1. **arXiv v2 not uploaded.** Source bundle has been rebuilt locally (commits a8ad623, 9ed8684) with pat-feedback proof corrections — theorem 8 reframed as calibration sample complexity, theorem 9 Hoeffding interval fixed, prop 13 made discrete-token CVP, full lemma 7 proof added, Stackelberg eqs 5–6 aligned, kappa notation standardized, Maurer empirical-Bernstein + DKW tight constants added to bib. None of this is on arxiv.org. **The arXiv preprint is currently strictly weaker than the local v2.**
2. **`feat/phase-37-pat-feedback` branch not merged.** main is at `acda258` (phase 36 doc sync). Uncommitted on this branch: `docs/arxiv/neurips_2026.tex`, `figures/fig_corpus_scaling.{pdf,png}`, `figures/fig_multi_encoder.pdf`, `results/tables/table1_attack_results.tex`. Untracked: `src/frontend/`, `external/mem0`, `claude_design/`.
3. **arXiv announcement (Twitter thread + 5–10 researcher emails + blog post)** — Plan §7 said within 1 week of arXiv ID. Five weeks overdue. Has direct cost: paper visibility during the NeurIPS silence period was the entire point of the arXiv-first sequence, and that window is mostly burned.
4. **Public reproduction repo not started.** Plan §6.3 said stage in parallel during review window. No REPRODUCE.md, no clean public release skeleton.
5. **Workshop track triage not started.** CFPs typically drop ~July; nothing pre-staged.

**Cost assessment:**

| Slip | Cost | Recoverable? |
|---|---|---|
| arXiv v2 upload | Low — but cited version is buggier than your locked version | Yes, upload tonight |
| Branch merge | Low — internal hygiene | Yes, merge + close PRs |
| Announcement | Medium-high — visibility window mostly closed; can still do a smaller push tied to v2 release | Partially |
| Reproduction repo | Low now, high if NeurIPS accepts and you have a week to ship a public release | Yes if started this month |
| Workshop track | Low — CFPs haven't dropped yet | Fully |

**What was NOT in the plan and consumed the time:**
- pat-feedback proof-correction cycle: ~20 commits of substantive theorem-level corrections (theorem 8/9 calibration, prop 13 CVP discrete-token, lemma 7 full proof, Stackelberg formalization). This was reviewer-grade hardening, not cosmetics. **Net positive for the paper, but the plan didn't budget for it and downstream tracks slipped.**

**Recovery sequence (next 7 days):**

1. Decide: arXiv v2 upload now, or hold for a single bigger upload that bundles pat-feedback + any NeurIPS reviewer-informed edits in August. **Recommendation: upload v2 now** — five weeks of citers pointing at a buggier v1 is a direct cost, and waiting on NeurIPS reviews to gate the upload means another 8+ weeks of stale public version.
2. Merge `feat/phase-37-pat-feedback` to main, commit the four modified files, decide what to do with the three untracked dirs (`src/frontend/`, `claude_design/`, `external/mem0` — figure out if any of this is the reproduction repo skeleton or a fork-and-modify experiment).
3. Once v2 is on arxiv, do the smaller announcement push: a tight 4-tweet thread keyed to "v2 with corrected proofs is up" (lower-stakes than a launch thread, but recovers some visibility), plus the 5–10 researcher emails (these are still high-value — they don't expire).
4. Start REPRODUCE.md and the clean public repo skeleton this week. Do not wait for NeurIPS notification.
5. Workshop triage: pick AdvML-Frontiers OR SafeGenAI (not both — diminishing returns). Watch the NeurIPS site for the workshop list (expected mid-July).
6. Pre-stage ICLR 2027 backup: identify which sections of the paper are most likely to need expansion vs which are most NeurIPS-reviewer-bait and prep two parallel revision branches before NeurIPS reviews land.

---

## 14. Status pass — 2026-07-02

Three-and-a-half-week gap since the 2026-06-08 pass. Verification re-run against neurips.cc, aaai.org, iclr.cc, and the GitHub repo. Triage below follows the rule: dead deadlines get closed with rationale; recoverable ones get re-scheduled; upcoming ones get confirmed.

**Verified as of 2026-07-02:**
- NeurIPS 2026 reviews are NOT yet released; notification remains **Sep 24 2026 (CONFIRMED)**. No action was missed in the NeurIPS main-track pipeline.
- NeurIPS 2026 accepted-workshop list is NOT yet posted (proposal acceptance Jul 11). The Jul 11–20 workshop-triage window is intact.
- AAAI 2027: abstract **Jul 21** / paper **Jul 28 2026** re-confirmed via aaai.org; OpenReview submission site opened Jun 30 2026.
- ICLR 2027: still unannounced (no venue page). Historical pattern says ~late-Sept 2026 deadline; keep the Oct 15 placeholder and verify monthly.

**Decision: AAAI 2027 fallback is structurally impossible — SKIP (closed 2026-07-02).**
The paper is under review at NeurIPS until Sep 24. AAAI-27's paper deadline (Jul 28) falls inside the NeurIPS review period, and both venues forbid concurrent submission of the same work. Submitting to AAAI would require withdrawing from NeurIPS, which is strictly dominated (NeurIPS is the stronger venue and the paper is already through the submission gate). The §5 contingency cascade therefore collapses to: **NeurIPS 2026 → ICLR 2027 → ICML 2027 → TMLR**, with the non-archival NeurIPS workshop track running in parallel. This was latent in §9's dual-submission constraint but never propagated to the §10 tracker — now resolved.

**Repo hygiene completed 2026-07-02:**
- PR #77 (pat-feedback, 22 commits + final pending fixes) merged to main after green CI; PR #76 closed as superseded (its two commits are ancestors of #77's branch); issue #75 closed.
- Pending working-tree changes committed: table 1 caption correction (GPT-2 **upper** bound, not lower), regenerated fig_corpus_scaling + fig_multi_encoder, arXiv build artifacts, this plan document.
- Infra hardening PR: CodeQL scanning, frontend demo + design assets tracked (design mockups relocated from repo root to docs/design), auto-checkpoint script, npm dependabot ecosystem. Follow-up flagged: external/amem and external/mem0 are bare gitlinks with no .gitmodules entries (fresh clones cannot resolve them); recording their upstream URLs needs a manual pass.
- Branch protection enabled on main (required status checks; no merges on red).

**Re-scheduled tracks (recoverable slips):**
1. **arXiv v2 upload — 2026-07-03 (user action).** Everything needed is on main after the #77 merge; upload via arxiv.org "Replace" against 2605.03482. This is the single highest-leverage pending action: every current citer reads the buggier v1.
2. **Announcement — within 72h of v2.** Reframed as a "v2 with corrected proofs" push (4-tweet thread + the 5–10 researcher emails, which do not expire in value).
3. **Blog post — 2026-07-15.**
4. **Reproduction repo — the README's "Reproducing paper results" section already serves as the in-repo guide; public split staged by 2026-09-01** so a release can ship within a week of a Sep 24 accept.
5. **Workshop triage — check list Jul 11–14, lock choice by Jul 20, short paper (4–6 pp) drafted by Aug 15 for the Aug 29 deadline.**
6. **ICLR 2027 pre-stage — by Sep 15** (before NeurIPS reviews land), since the ICLR deadline has historically fallen within days of the NeurIPS notification date.

**Automation installed 2026-07-02** (to prevent a third silent slip): weekly scheduled deadline-watch (venue pages + review-release check + Notion sync), and a local git auto-checkpoint that snapshots uncommitted work to `refs/checkpoints/*` so nothing on the working tree can be lost between sessions.
