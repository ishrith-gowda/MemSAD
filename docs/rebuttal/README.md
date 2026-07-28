# NeurIPS 2026 Rebuttal — review index (Submission 30963, MEMSAD)

Everything is on branch `rebuttal/neurips2026` (stacked on phase-38), **nothing pushed**. This is the entry point for the author's final review. Window: Jul 23 – **Aug 3** (Official Comment ≡ Rebuttal button; additional results allowed).

## Read in this order
1. **The four response drafts** (this is the gate — you post them, I can't):
   - [response_n6dm.md](response_n6dm.md) — reviewer n6dm (4, conf 5, borderline accept). ~7.4k chars. Convert to accept.
   - [response_m2wH.md](response_m2wH.md) — reviewer m2wH (2, conf 4, reject). ~8.0k. Concede-and-repair.
   - [response_u2M8.md](response_u2M8.md) — reviewer u2M8 (3, conf 3). ~4.5k. Threat-model reframe.
   - [response_AC.md](response_AC.md) — Author-AC confidential comment. ~4.7k. Correctness + the two asks.
2. [experiment_results.md](experiment_results.md) — the six experiments (E1–E6) with tables; every number in the responses is transcribed from here.
3. [neurips2026_rebuttal_plan.md](neurips2026_rebuttal_plan.md) — full per-issue adjudication (§2b verified from the appendix; §17 execution status) if you want the reasoning behind each response.
4. [anonymized_repo_plan.md](anonymized_repo_plan.md) — the AC-repo prep + anonymity checklist (Phase-2 asset; gated on your sign-off).

## The six experiments (reproduce: `PYTHONPATH=src python3 src/scripts/rebuttal_e{1..6}_*.py`)
| Exp | Answers | Headline |
|---|---|---|
| E1 gap distribution | n6dm Q1 | AGENTPOISON gap +0.21, 100% detected, 38% certified → certificate informative in-regime |
| E2 smoothing comparison | n6dm W5 / AC | Cohen smoothing vacuous in d=384 (R=0.008 vs δ=0.378); distinct mechanism |
| E3 coupling condition | m2wH W2 | gradients acute (40.5°); retrieval-optimal dir evades s_comb 0/8 |
| E4 ROC/AUROC | m2wH W4 | AUROC 1.00/0.951/0.846 → low TPR is the FPR=0 operating point |
| E5 encoder regularity | n6dm W2 | per-substitution embedding shift bounded (max ≤0.39) across 6 encoders |
| E6 disjunctive detector | m2wH W2 | coupling-preserving AND improves detection (MINJA 0.06→0.38, INJECMEM 0.00→0.50) |

## Strategy in one paragraph
Of m2wH's + PAT's ~13 objections, **7 target things the reviewed version already fixed** (CVP discrete-token, Wasserstein √, Theorem 9, composite=disjunction, Le Cam calibration-framing, Table 15 fair calibration, batch-vs-single) — answered by quoting the text. **3 are genuinely valid** (Corollary 17, Eq 12, Lemma 8 q\*∈H) — conceded and *converted*: E6's disjunctive detector (a coupling-preserving improvement), a cleaner threshold-estimation theorem, and the **cross-principal unification** (m2wH's q\*∈H concern and u2M8's threat-model concern resolve each other). **2 more table concessions** (Table 13 caption, Table 16 FPR mislabel) conceded cleanly. Both of the AC's explicit asks (smoothing comparison, certified-radius informativeness) are delivered as new experiments. Every valid criticism is conceded, not argued past.

## Posting playbook (author actions)
1. **Post the initial responses now / ASAP** (PC email urges engaging early; the discussion runs to Aug 3). Use the per-review **Rebuttal** button for n6dm/m2wH/u2M8; use **Author-AC Confidential Comment** for the AC note.
2. Rules honored in the drafts: ≤10k chars each (verified), plain-text+markdown, **no links, no file uploads, anonymity preserved** (transcribe the tables as text).
3. **Watch for Phase-2 follow-ups** (Jul 27–Aug 3) and respond via Official Comment. If m2wH asks for the repo, execute [anonymized_repo_plan.md](anonymized_repo_plan.md) (needs your sign-off + your GitHub for the anonymous.4open.science link).
4. After the cycle: fold every correction into the ICLR-2027 resubmission regardless of outcome (the rebuttal work is the resubmission-strengthening work).

## Status
All drafts finalized + self-reviewed (accuracy, cross-consistency, char limits, FPR-tradeoff honesty). All six experiments reproduce; scripts lint+mypy clean; phase-38 tests (31) green. The only remaining step is the author's read + post.
