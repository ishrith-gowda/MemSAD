#!/usr/bin/env bash
# auto-checkpoint: snapshot the working tree (tracked + untracked, not ignored)
# into refs/checkpoints/<utc-timestamp> without touching the index, worktree,
# or current branch. safe to run unattended from launchd/cron; exits quietly
# when the repo is clean, unmounted, or unavailable.
#
# usage: auto_checkpoint.sh [repo_dir]
# recovery: git stash apply <checkpoint-sha>  (or)  git checkout <ref> -- <path>

set -euo pipefail

repo_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$repo_dir" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

# nothing to snapshot when the tree is clean
if [ -z "$(git status --porcelain)" ]; then
  exit 0
fi

git_dir="$(git rev-parse --git-dir)"
tmp_index="$(mktemp)"
trap 'rm -f "$tmp_index"' EXIT

# start from the real index so staged-but-uncommitted state is preserved
if [ -f "$git_dir/index" ]; then
  cp "$git_dir/index" "$tmp_index"
fi

GIT_INDEX_FILE="$tmp_index" git add -A
tree="$(GIT_INDEX_FILE="$tmp_index" git write-tree)"

parent="$(git rev-parse HEAD)"
branch="$(git rev-parse --abbrev-ref HEAD)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

# skip if this exact tree is already the latest checkpoint
last_ref="$(git for-each-ref --sort=-creatordate --count=1 --format='%(refname)' refs/checkpoints)"
if [ -n "$last_ref" ] && [ "$(git rev-parse "$last_ref^{tree}")" = "$tree" ]; then
  exit 0
fi

commit="$(git commit-tree "$tree" -p "$parent" -m "checkpoint: ${branch} at ${stamp}")"
git update-ref "refs/checkpoints/${stamp}" "$commit"

# best-effort offsite copy; ignore failures (offline, auth, drive ejected)
git push origin "refs/checkpoints/${stamp}:refs/checkpoints/${stamp}" >/dev/null 2>&1 || true

# prune local checkpoints older than 30 days
cutoff="$(( $(date +%s) - 30 * 24 * 3600 ))"
git for-each-ref --format='%(refname) %(creatordate:unix)' refs/checkpoints |
  while read -r ref ts; do
    if [ -n "$ts" ] && [ "$ts" -lt "$cutoff" ]; then
      git update-ref -d "$ref"
    fi
  done

echo "checkpoint ${stamp} created for ${branch} (${commit})"
