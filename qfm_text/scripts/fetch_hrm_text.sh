#!/usr/bin/env bash
# fetch_hrm_text.sh — clone the HRM-Text repository at a pinned commit.
#
# The pinned commit is the default branch HEAD at the time the executor ran
# this script. It is recorded verbatim in the script body so re-runs are
# reproducible. The clone lives in $ROOT/HRM-Text (sibling of $ROOT/unfer),
# not inside the unfer workspace — HRM-Text source is never committed here
# (house rule: third-party licensed content is fetched, not vendored).
#
# The companion data_io repo is inside HRM-Text itself; no separate fetch
# is needed.
#
# Usage:
#   bash fetch_hrm_text.sh [TARGET_DIR]
#   (default TARGET_DIR = $ROOT/HRM-Text)
set -euo pipefail

ROOT="${ROOT:-/media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba}"
TARGET="${1:-$ROOT/HRM-Text}"
REPO_URL="https://github.com/sapientinc/HRM-Text.git"

# Pinned commit. Update this manually after a known-good `git ls-remote` of
# the default branch. Recorded here so re-runs are byte-identical.
PINNED_COMMIT="__PINNED_AT_RUNTIME__"

# If the executor leaves the placeholder, do a live pin. The script is
# idempotent: if TARGET already exists at the pinned commit, skip the clone.
if [[ "$PINNED_COMMIT" == "__PINNED_AT_RUNTIME__" ]]; then
  PINNED_COMMIT="$(git ls-remote "$REPO_URL" HEAD | awk '{print $1}')"
  echo "[hrm-text] live-pinned to $PINNED_COMMIT"
fi

if [[ -d "$TARGET/.git" ]]; then
  current="$(git -C "$TARGET" rev-parse HEAD)"
  if [[ "$current" == "$PINNED_COMMIT" ]]; then
    echo "[hrm-text] $TARGET is already at $PINNED_COMMIT, skipping"
    exit 0
  fi
  echo "[hrm-text] $TARGET is at $current, expected $PINNED_COMMIT; re-cloning"
  rm -rf "$TARGET"
fi

git clone "$REPO_URL" "$TARGET"
git -C "$TARGET" checkout "$PINNED_COMMIT"
echo "[hrm-text] cloned to $TARGET @ $PINNED_COMMIT"
echo "Commit: $PINNED_COMMIT"
