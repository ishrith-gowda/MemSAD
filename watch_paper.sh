#!/usr/bin/env bash
# auto-recompile docs/neurips2026/main.tex whenever any .tex or .bib file changes.
# uses fswatch (brew install fswatch) to monitor for file-system events.
#
# usage:
#   ./watch_paper.sh          # compile + watch for changes
#   ./watch_paper.sh --once   # compile once and exit
#
# the compiled pdf is at: docs/neurips2026/main.pdf
# open it in any pdf viewer that supports auto-refresh (e.g. preview, skim).

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")/docs/neurips2026" && pwd)"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

compile() {
    local start_time
    start_time=$(date +%s)
    echo ""
    echo "=== compiling: $(date '+%H:%M:%S') ==="
    cd "$PAPER_DIR"
    pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/latex_watch.log 2>&1 \
        && bibtex main >> /tmp/latex_watch.log 2>&1 \
        && pdflatex -interaction=nonstopmode main.tex >> /tmp/latex_watch.log 2>&1 \
        && pdflatex -interaction=nonstopmode main.tex >> /tmp/latex_watch.log 2>&1
    local exit_code=$?
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    if [ $exit_code -eq 0 ]; then
        local pages
        pages=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "?")
        echo "--- ok: ${pages} pages in ${elapsed}s  [docs/neurips2026/main.pdf] ---"
    else
        echo "--- COMPILATION FAILED (${elapsed}s) ---"
        echo "last 20 lines of log:"
        tail -20 /tmp/latex_watch.log
    fi
    cd "$REPO_ROOT"
    return $exit_code
}

# initial compile
compile || true

if [ "${1:-}" = "--once" ]; then
    exit 0
fi

echo ""
echo "watching docs/neurips2026/ for changes (.tex / .bib / figures) ..."
echo "open docs/neurips2026/main.pdf in Skim or Preview (auto-refresh on save)."
echo "press ctrl+c to stop."
echo ""

# watch .tex files, .bib files, and figures directory
fswatch -o \
    --include='\.tex$' \
    --include='\.bib$' \
    --include='figures/' \
    --exclude='.*\.aux$' \
    --exclude='.*\.log$' \
    --exclude='.*\.out$' \
    --exclude='.*\.toc$' \
    --exclude='.*\.synctex' \
    "$PAPER_DIR" \
    | while read -r _; do
        compile || true
    done
