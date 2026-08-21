# Seed note: H11 test discipline

**Category**: testing
**Date**: H11 stage
**Status**: implemented

## What
Added the three dsh-style test tiers over the existing unit harness:
keyless NDJSON snapshot replay for `unfer_agent`, a per-file coverage gate on
`prob_kernel/src` + `unfer_ffi/src`, and real-entry-path smokes.

## Why
The ~350-test suite plus a single golden gate missed the "green unit tests,
broken product" class (stale artifacts, masked settle, wrong entry wiring). The
coverage gate immediately surfaced `prob_kernel/src/event.rs` at ~0% in the
kernel's own suite — the Born-rule matcher only ran through the FFI path — which
got a direct unit suite.

## How verified
- `scripts/verify-invariants` (32 passed) now includes the coverage + smoke
  gates.
- `bash scripts/coverage_gate`: 11 files >= 40%, `build.rs` (CUDA) exempted.
- `bash scripts/smoke_gate`: cdylib builds, unfer_agent version round-trip,
  modhost fails closed on a missing module dir.

## Frozen
This note is archived and frozen (dsh notes policy).