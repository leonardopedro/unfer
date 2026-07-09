# Test fixtures

## `tiny_fixture.bin`

A **200-token synthetic** Markov-chain sample over a 16-word
synthetic vocabulary. It is hand-coded and deterministic (splitmix64-
seeded by `scripts/generate_fixtures.sh`); it is **not** derived from
any real Wikipedia text and so does not need attribution.

It is committed to the repo as a byte-stable CI fixture: re-running
`scripts/generate_fixtures.sh` produces a byte-identical file (sha256
recorded by the build script).

Use it as a sanity check that the shard reader, hashed encoder, and
streaming accumulator work end-to-end. For real training, use the
shards produced by `scripts/prepare_corpus.sh` (see
`docs/QFM_TEXT_HRM_PLAN.md` §"Stage 0" for the WikiText-103 / BPE
flow).

## Why not WikiText-103 directly?

WikiText-103 (Merity et al., 2016) is licensed under CC-BY-SA 4.0.
A 200-token test fixture derived from its test split is small enough
to commit, but the plan calls for a deterministic *generated* fixture
in the repo and the real WikiText-103 data fetched on demand by
`scripts/prepare_corpus.sh` (URL + sha256 + ATTRIBUTION.txt). The
fixture in this directory is the former; production shards are the
latter.
