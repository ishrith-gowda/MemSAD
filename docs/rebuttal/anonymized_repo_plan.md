# Anonymized repository for the AC — preparation plan (Submission 30963)

m2wH (Q3) requested "a working anonymized repository" with raw per-seed predictions, independent experimental units, exact CI aggregation, seeds, and data. The AC response offers to provide it via Official Comment. **This is outward-facing and double-blind-critical — execute only after author sign-off; a single leaked identifier breaks anonymity.**

## Delivery mechanism
- OpenReview forbids file uploads and links in the *rebuttal*, but explicitly permits an **anonymized code link sent to the AC via Official Comment** (per the handbook). Use **https://anonymous.4open.science** (standard for NeurIPS double-blind) — it proxies a GitHub repo and strips identity. The author creates the link (requires the author's GitHub OAuth); the agent cannot and should not create external links.

## What to INCLUDE (what m2wH asked for)
- `src/` (attacks, defenses, evaluation, scripts) — the full pipeline.
- The rebuttal experiment scripts `src/scripts/rebuttal_e{1..6}_*.py` (they reproduce every number in the responses).
- `src/tests/` — the test suite (593 tests) as evidence of rigor.
- A top-level `REPRODUCE.md`: exact commands + seeds (seed=42 throughout) to regenerate each table/figure and each rebuttal experiment; per-seed raw predictions dump; Clopper-Pearson aggregation code path (`src/evaluation/statistical.py`).
- `requirements.txt`, minimal README (anonymized).

## What to STRIP / SCRUB (anonymity checklist — verify each before upload)
1. **Author name** "Ishrith Gowda" / "ishrith-gowda" / "ishrithgowda@berkeley.edu" — grep the entire tree (code, comments, docstrings, README, configs).
2. **Affiliation** — the Song Lab / BAIR / UC Berkeley footnote and any acknowledgments.
3. **Repo identity** — GitHub URLs (`github.com/ishrith-gowda/MemSAD`, old `memory-agent-security`), badge links, the HF Space demo URL (`ishrith-gowda-memsad-demo`).
4. **Git metadata** — ship a *fresh* repo (no `.git` history; author name/email live in every commit). Copy the working tree only.
5. **arXiv id** `2605.03482` (de-anonymizes via the arXiv author page).
6. **`.env` / secrets** — ensure no `OPENAI_API_KEY` or any key is present (it is gitignored; verify it is absent from the snapshot).
7. **`docs/rebuttal/`, `docs/research/` (publication_plan, phase38_findings), memory files, `external/` submodules, `src/frontend/`, `docs/design/`** — exclude; they contain planning/identity/irrelevant material.
8. **CLAUDE.md, .claude/** — exclude (tooling/identity).
9. Author-identifying phrasing in comments (e.g., "phase 37/38", internal milestone names) — optional scrub; not identity-critical but tidy.

## Suggested one-time preparation (author runs, then reviews the diff before upload)
1. `rsync` the working tree into a clean dir excluding the STRIP list (no `.git`, no `docs/rebuttal|research|design`, no `external`, no `src/frontend`, no `.claude`, no memory).
2. Run a scrub pass: replace the name/email/affiliation/URLs/arXiv-id strings with neutral placeholders ("Anonymous", "anon-org", "https://anonymous.4open.science/r/MemSAD").
3. **Verify:** `grep -rIiE 'gowda|berkeley|song lab|bair|2605\.03482|ishrith|OPENAI_API_KEY'` returns nothing.
4. Add `REPRODUCE.md`; smoke-test `PYTHONPATH=src python3 src/scripts/rebuttal_e1_gap_distribution.py` in the clean dir.
5. Push the clean dir to a throwaway GitHub repo; generate the anonymous.4open.science link; paste it to the AC via Author-AC Confidential Comment.

## Status
Plan ready. The agent can generate the scrub/rsync script on request, but will NOT create the snapshot or any external link without explicit author sign-off (outward-facing + anonymity-critical). Recommended timing: after the initial responses are posted (the repo is a Phase-2 discussion asset, not needed for the initial rebuttal).
