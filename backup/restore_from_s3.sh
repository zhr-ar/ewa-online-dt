#!/usr/bin/env bash
# Restore experiment data from S3 on a new instance.
#
# Usage:
#   export S3_BUCKET='your-unique-ewa-backup-bucket'
#   bash backup/restore_from_s3.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-ewa-backup}"

if [[ -z "$S3_BUCKET" ]]; then
  echo "ERROR: Set S3_BUCKET first"
  exit 1
fi

restore_dir() {
  local dest="$1"
  local src="$2"
  mkdir -p "$dest"
  echo ">>> Restoring s3://${S3_BUCKET}/${src}/ -> $dest"
  aws s3 sync "s3://${S3_BUCKET}/${src}/" "$dest" --only-show-errors
}

cd "$REPO_ROOT"
restore_dir "$REPO_ROOT/exp"                "${S3_PREFIX}/exp"
restore_dir "$REPO_ROOT/data"               "${S3_PREFIX}/data"
restore_dir "$REPO_ROOT/exp_results"        "${S3_PREFIX}/exp_results"
restore_dir "$REPO_ROOT/figures_comparison" "${S3_PREFIX}/figures_comparison"
restore_dir "$REPO_ROOT/figures_ewa"        "${S3_PREFIX}/figures_ewa"
restore_dir "$HOME/.d4rl"                   "${S3_PREFIX}/d4rl"
restore_dir "$HOME/.mujoco"                 "${S3_PREFIX}/mujoco"

echo ""
echo "Recreate conda env:"
echo "  conda env create -f backup/conda-env-odt.yaml"
