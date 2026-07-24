#!/usr/bin/env bash
# Clean up qfm_text_runs/ checkpoints (*.qfm files are gitignored but consume ~3.9 GB).
# Keeps metrics.ndjson, train.log, train.toml (tracked in git).
#
# Usage:
#   tools/clean_qfm_text_runs.sh          # dry-run (shows what would be deleted)
#   tools/clean_qfm_text_runs.sh --force  # actually delete

set -euo pipefail

RUNS_DIR="$(cd "$(dirname "$0")/.." && pwd)/qfm_text_runs"

if [ ! -d "$RUNS_DIR" ]; then
  echo "qfm_text_runs/ not found; nothing to clean."
  exit 0
fi

TOTAL=$(find "$RUNS_DIR" -name '*.qfm' -type f | wc -l)
SIZE=$(find "$RUNS_DIR" -name '*.qfm' -type f -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)

echo "Found $TOTAL checkpoint files (*.qfm), total size: ${SIZE:-0}"

if [ "${1:-}" = "--force" ]; then
  find "$RUNS_DIR" -name '*.qfm' -type f -delete
  echo "Deleted $TOTAL checkpoint files."
else
  echo "Dry-run. Pass --force to delete."
  find "$RUNS_DIR" -name '*.qfm' -type f -printf '  %p (%s bytes)\n' | head -20
  [ "$TOTAL" -gt 20 ] && echo "  ... and $((TOTAL - 20)) more"
fi
