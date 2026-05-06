# arXiv preprint — MEMSAD

Self-contained build directory for the **arXiv preprint version** of the MEMSAD paper.

## What's in this folder

- `main.tex` — paper source, configured for arXiv (`\usepackage[preprint]{neurips_2026}`, de-anonymized author block, no `\input{checklist}`)
- `main.pdf` — current compiled output (28 pages)
- `references.bib` — bibliography
- `neurips_2026.sty` / `neurips_2026.tex` — NeurIPS style files
- `figures/` — every figure used in the paper

## How this differs from `docs/neurips2026/`

| | `docs/neurips2026/` | `docs/arxiv/` (this folder) |
|---|---|---|
| Package option | `\usepackage{neurips_2026}` (anonymous mode for submission) | `\usepackage[preprint]{neurips_2026}` (de-anonymized) |
| Author block | `\author{Anonymous Author(s)}` | Real name, affiliation, email + Song Lab footnote |
| Footer | "Submitted to NeurIPS 2026. Do not distribute." | "Preprint." |
| `\input{checklist}` | Included | Commented out |
| Acknowledgments section | (already removed from both) | (already removed from both) |
| Purpose | Conference submission via OpenReview | arXiv preprint upload |

**Do not cross-edit between folders.** When the paper changes:
1. Edit one canonical version (recommended: `docs/arxiv/`).
2. Copy across to `docs/neurips2026/` and re-apply the anonymous-mode toggles when needed (camera-ready, resubmission).

## Build

From inside `docs/arxiv/`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or use the watch script (`watch_paper.sh` in repo root) pointed at this directory.

## arXiv submission checklist

- [ ] PDF compiles cleanly (no `??` warnings, no missing references)
- [ ] Author block shows real name + affiliation footnote
- [ ] Footer reads "Preprint."
- [ ] All figures resolve (`figures/` populated, no missing images)
- [ ] Bibliography present (page 10 of PDF)
- [ ] License selected on arXiv: **CC BY**
- [ ] Primary subject class: **cs.CR** (Cryptography and Security)
- [ ] Cross-list: **cs.LG, cs.AI**

## Source bundle for arXiv upload

arXiv prefers source uploads (it re-compiles for consistency). When uploading:

```bash
# from this directory
zip -r main_source.zip main.tex references.bib neurips_2026.sty neurips_2026.tex figures/ \
    -x '*.aux' '*.log' '*.out' '*.bbl' '*.blg'
```

Upload `main_source.zip` to the arXiv submission portal.
