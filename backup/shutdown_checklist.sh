#!/usr/bin/env bash
# Pre-shutdown checklist for the AWS GPU instance.
#
# Usage: bash backup/shutdown_checklist.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

pass() { echo "  [OK]   $1"; }
fail() { echo "  [TODO] $1"; }

echo "=== EWA Instance Shutdown Checklist ==="
echo ""

# 1. Git push
if git rev-parse origin/private-backup >/dev/null 2>&1; then
  UNPUSHED=$(git rev-list --count origin/private-backup..HEAD 2>/dev/null || echo "?")
  if [[ "$UNPUSHED" == "0" ]]; then
    pass "Git pushed to private-backup"
  else
    fail "Git: $UNPUSHED commit(s) not on origin/private-backup — run: bash backup/push_to_github.sh"
  fi
else
  fail "Git: cannot compare with origin/private-backup — run: bash backup/push_to_github.sh"
fi

# 2. AWS credentials
if aws sts get-caller-identity >/dev/null 2>&1; then
  pass "AWS credentials configured"
else
  fail "AWS: run 'aws configure' then 'bash backup/s3_sync.sh'"
fi

# 3. Local data sizes
echo ""
echo "=== Local data to back up (~70 GB total) ==="
for d in exp data exp_results figures_comparison figures_ewa; do
  if [[ -d "$REPO_ROOT/$d" ]]; then
    printf "  %-22s %s\n" "$d/" "$(du -sh "$REPO_ROOT/$d" | cut -f1)"
  fi
done
printf "  %-22s %s\n" "~/.d4rl/" "$(du -sh "$HOME/.d4rl" 2>/dev/null | cut -f1 || echo 'missing')"
printf "  %-22s %s\n" "~/.mujoco/" "$(du -sh "$HOME/.mujoco" 2>/dev/null | cut -f1 || echo 'missing')"

# 4. Old sibling project
echo ""
OLD_PROJECT="/home/ubuntu/online_decision_transformer/online-dt-main"
if [[ -d "$OLD_PROJECT" ]]; then
  echo "=== Optional: old project online-dt-main/ ($(du -sh "$OLD_PROJECT" | cut -f1)) ==="
  echo "  ewa-online-dt-main is a superset — safe to delete after S3 backup:"
  echo "  rm -rf $OLD_PROJECT"
fi

echo ""
echo "=== After backup, in AWS Console ==="
echo "  1. Terminate instance (not just Stop)"
echo "  2. Delete EBS volume if not auto-deleted"
echo "  3. Release Elastic IP if attached"
