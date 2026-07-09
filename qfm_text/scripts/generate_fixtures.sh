#!/usr/bin/env bash
# generate_fixtures.sh — build the deterministic tiny fixture
# used by qfm_text's corpus-level integration tests.
#
# This script is the *only* way the committed testdata/tiny_fixture.bin
# is generated. It is deterministic (splitmix64-seeded) so re-runs
# produce byte-identical output. The fixture is a 200-token synthetic
# English-like text produced by sampling a small BPE-vocab Markov
# chain — it is NOT WikiText-103 content; see testdata/README.md for
# the distinction.
#
# Usage:
#   bash generate_fixtures.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/testdata/tiny_fixture.bin"
mkdir -p "${ROOT}/testdata"

python3 - <<'PY'
import os, struct, random
out = os.environ.get("OUT", "testdata/tiny_fixture.bin")
os.makedirs(os.path.dirname(out), exist_ok=True)
# Deterministic Markov chain over 16 synthetic "words". The
# transition matrix is hand-written for reproducibility.
words = [
    "the ", "of ", "and ", "to ", "in ", "a ", "is ", "that ",
    "for ", "it ", "with ", "as ", "on ", "be ", "by ", "this ",
]
# Hand-coded transition matrix (row-stochastic, rows in `words` order).
# Each row[i] is the prob of transitioning from word i to word j.
trans = [
    [0.30, 0.05, 0.10, 0.10, 0.05, 0.20, 0.05, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00],  # the
    [0.10, 0.30, 0.05, 0.10, 0.10, 0.10, 0.05, 0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.00, 0.00, 0.05],  # of
    [0.10, 0.10, 0.30, 0.10, 0.05, 0.05, 0.05, 0.10, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05],  # and
    [0.10, 0.05, 0.05, 0.30, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05, 0.00, 0.05, 0.00, 0.00, 0.00, 0.05],  # to
    [0.10, 0.05, 0.10, 0.10, 0.30, 0.05, 0.05, 0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.00, 0.00, 0.10],  # in
    [0.30, 0.05, 0.05, 0.05, 0.05, 0.30, 0.05, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.10],  # a
    [0.10, 0.05, 0.10, 0.05, 0.10, 0.10, 0.30, 0.05, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.10],  # is
    [0.10, 0.05, 0.10, 0.05, 0.05, 0.10, 0.05, 0.30, 0.05, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.10],  # that
    [0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05, 0.30, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.15],  # for
    [0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.05, 0.05, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10],  # it
    [0.10, 0.05, 0.10, 0.10, 0.05, 0.10, 0.05, 0.05, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.00, 0.10],  # with
    [0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.00, 0.05, 0.15, 0.05, 0.00, 0.00, 0.05],  # as
    [0.10, 0.05, 0.10, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05, 0.00, 0.05, 0.00, 0.20, 0.00, 0.00, 0.10],  # on
    [0.10, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.05, 0.00, 0.00, 0.05, 0.00, 0.20, 0.00, 0.10],  # be
    [0.05, 0.10, 0.05, 0.05, 0.10, 0.10, 0.10, 0.05, 0.10, 0.00, 0.05, 0.00, 0.05, 0.05, 0.10, 0.05],  # by
    [0.10, 0.05, 0.10, 0.10, 0.10, 0.05, 0.10, 0.05, 0.05, 0.00, 0.00, 0.05, 0.05, 0.00, 0.00, 0.20],  # this
]
# Normalize rows just in case.
for r, row in enumerate(trans):
    s = sum(row)
    trans[r] = [x / s for x in row]
# Tokenize: each word maps to its index in `words`. Vocab = 16.
n_tokens = 200
random.seed(42)
text = []
state = 0
for _ in range(n_tokens):
    row = trans[state]
    next_state = random.choices(range(len(words)), weights=row, k=1)[0]
    text.append(next_state)
    state = next_state
# Round-trip the text so the test fixture contains an "is a real
# English-like sentence" prefix (the first 8 tokens decoded):
print("decoded preview:", "".join(words[t] for t in text[:32]))
buf = struct.pack(f"<{len(text)}I", *text)
with open(out, "wb") as f:
    f.write(buf)
print(f"wrote {out} ({len(buf)} bytes, {len(text)} tokens)")
PY
