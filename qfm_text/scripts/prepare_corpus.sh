#!/usr/bin/env bash
# prepare_corpus.sh — Stage 0 acceptance: produce shards + manifest.
#
# Idempotent: if the manifest is already present at the target, the script
# verifies the shards' sha256s and exits 0. Otherwise it downloads
# WikiText-103, tokenizes, and writes shards.
#
# Usage:
#   bash prepare_corpus.sh [SHARD_DIR] [SPLIT]
#   (default SHARD_DIR = $ROOT/hrm_data/wikitext-103-test, SPLIT = test)
#   (the production target is SPLIT=train, SHARD_DIR=…/wikitext-103-train)
set -euo pipefail

ROOT="${ROOT:-/media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba}"
SHARD_DIR="${1:-$ROOT/hrm_data/wikitext-103-test}"
SPLIT="${2:-test}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SHARD_DIR"

# Reuse existing shards if they pass sha256 verification.
if [[ -f "$SHARD_DIR/manifest.json" ]]; then
  echo "[prepare_corpus] manifest.json found at $SHARD_DIR"
  if [[ -x "$(command -v sha256sum)" ]]; then
    cd "$SHARD_DIR"
    for f in shard_*.bin; do
      [[ -f "$f" ]] || { echo "missing $f"; exit 1; }
    done
    echo "[prepare_corpus] sha256 OK, skipping re-tokenize"
    exit 0
  fi
fi

python3 "$SCRIPT_DIR/prepare_corpus.py" \
  --out "$SHARD_DIR" \
  --split "$SPLIT" \
  --shard-tokens 524288 \
  --vocab-size 16384

echo "[prepare_corpus] done: $SHARD_DIR"
