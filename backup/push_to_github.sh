#!/usr/bin/env bash
# Push local main branch to GitHub private-backup.
#
# Prerequisites (pick one):
#   A) GitHub PAT:
#        export GITHUB_TOKEN='ghp_...'
#        git remote set-url origin "https://${GITHUB_TOKEN}@github.com/zhr-ar/ewa-online-dt.git"
#   B) SSH key added to GitHub:
#        git remote set-url origin git@github.com:zhr-ar/ewa-online-dt.git
#
# Usage:
#   bash backup/push_to_github.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Git status ==="
git status -sb
echo ""
echo "=== Commits to push ==="
git log --oneline origin/private-backup..HEAD 2>/dev/null || git log --oneline -3

echo ""
echo "=== Pushing main -> origin/private-backup ==="
git push origin main:private-backup

echo ""
echo "Done. Verify at: https://github.com/zhr-ar/ewa-online-dt/tree/private-backup"
