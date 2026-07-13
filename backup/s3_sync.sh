#!/usr/bin/env bash
# Sync experiment data and datasets to S3 before terminating the GPU instance.
#
# Prerequisites:
#   aws configure
#   # or: export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-1
#
# Usage:
#   export S3_BUCKET='your-unique-ewa-backup-bucket'
#   bash backup/s3_sync.sh
#
# Estimated upload size: ~70 GB (exp + data + exp_results + d4rl + figures)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-ewa-backup}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$S3_BUCKET" ]]; then
  echo "ERROR: Set S3_BUCKET first, e.g.:"
  echo "  export S3_BUCKET='your-unique-ewa-backup-bucket'"
  exit 1
fi

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "ERROR: AWS credentials not configured. Run: aws configure"
  exit 1
fi

AWS_FLAGS=(--only-show-errors)
if [[ "$DRY_RUN" == "1" ]]; then
  AWS_FLAGS+=(--dryrun)
  echo "DRY RUN mode (no uploads)"
fi

sync_dir() {
  local src="$1"
  local dest="$2"
  if [[ -d "$src" ]]; then
    echo ""
    echo ">>> Syncing $(du -sh "$src" | cut -f1)  $src  ->  s3://${S3_BUCKET}/${dest}/"
    aws s3 sync "$src" "s3://${S3_BUCKET}/${dest}/" "${AWS_FLAGS[@]}"
  else
    echo "Skipping missing directory: $src"
  fi
}

echo "=== EWA backup to s3://${S3_BUCKET}/${S3_PREFIX}/ ==="
aws s3 mb "s3://${S3_BUCKET}" 2>/dev/null || true

sync_dir "$REPO_ROOT/exp"           "${S3_PREFIX}/exp"
sync_dir "$REPO_ROOT/data"          "${S3_PREFIX}/data"
sync_dir "$REPO_ROOT/exp_results"   "${S3_PREFIX}/exp_results"
sync_dir "$REPO_ROOT/figures_comparison" "${S3_PREFIX}/figures_comparison"
sync_dir "$REPO_ROOT/figures_ewa"   "${S3_PREFIX}/figures_ewa"
sync_dir "$REPO_ROOT/backup"        "${S3_PREFIX}/env"
sync_dir "$HOME/.d4rl"              "${S3_PREFIX}/d4rl"
sync_dir "$HOME/.mujoco"            "${S3_PREFIX}/mujoco"

# Legacy online-dt-main experiments (sibling project)
ODT_ROOT="$(dirname "$REPO_ROOT")/online-dt-main"
sync_dir "$ODT_ROOT/exp"            "${S3_PREFIX}/online-dt-main/exp"
sync_dir "$ODT_ROOT/exp_results"    "${S3_PREFIX}/online-dt-main/exp_results"
sync_dir "$ODT_ROOT/figures_odt"    "${S3_PREFIX}/online-dt-main/figures_odt"

echo ""
echo "=== Done ==="
echo "Restore later with:"
echo "  aws s3 sync s3://${S3_BUCKET}/${S3_PREFIX}/exp/ ./exp/"
echo "  aws s3 sync s3://${S3_BUCKET}/${S3_PREFIX}/data/ ./data/"
