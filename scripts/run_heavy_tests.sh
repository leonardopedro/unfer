#!/usr/bin/env bash
# Run the heavy (#[ignore]d) physics/SIRK tests that are too slow for the
# default suite and produce a timestamped log for later inspection.
#
#   scripts/run_heavy_tests.sh [--quick]
#
# Covers:
#   - fock_sirk/tests/cdb_hamiltonian_match.rs        (heavy CAS-photon SIRK solve)
#   - fock_sirk/tests/latex_cas_hamiltonian_match.rs  (heavy CAS + LaTeX B²+kinetic compiles)
#   - qfm_text/tests/{oxieml_fit_real_w,print_c0}.rs  (need the trained checkpoint;
#                                                      skipped with a note if the
#                                                      external drive is not mounted)
#
# The log is written to logs/heavy_tests_<timestamp>.log (created if missing)
# and echoed to stdout. Exit 0 = all heavy tests green (or skipped for a
# documented reason); 1 = a heavy test failed.

set -u

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
QUICK="${1:-}"

ts="$(date +%Y%m%d_%H%M%S)"
log_dir="$repo_root/logs"
mkdir -p "$log_dir"
LOG="$log_dir/heavy_tests_$ts.log"

echo "Heavy SIRK/physics test run — $(date)"
echo "Log: $LOG"
echo

# Keep a summary line per suite; full output goes to the log via tee.
summary() { tee -a "$LOG" | sed 's/^/  /'; }

run_suite() {
    local name="$1"; shift
    local cmd=("$@")
    echo "=== $name ===" | tee -a "$LOG"
    if "${cmd[@]}" > >(summary) 2>&1; then
        echo "  [pass] $name" | tee -a "$LOG"
        return 0
    else
        echo "  [FAIL] $name" | tee -a "$LOG"
        return 1
    fi
}

FAILED=0

# 1. fock_sirk heavy SIRK solve + compiler-route compiles.
# Run in RELEASE mode: the CAS/LaTeX expansion is pathologically slow
# unoptimized (~870 s per LaTeX-dagger compile in debug vs ~0.1 s in release),
# and the 900 s debug budget killed the suite mid-run (log 20260823_190214).
# Release is the numerically optimized profile (opt-level 3); the timeout stays
# at 1800 s only as cold-build headroom.
run_suite "fock_sirk cdb_hamiltonian_match --release --ignored" \
    bash -c "cd '$repo_root/fock_sirk' && timeout 1800 cargo test --release --test cdb_hamiltonian_match -- --ignored" \
    || FAILED=$((FAILED + 1))

run_suite "fock_sirk latex_cas_hamiltonian_match --release --ignored" \
    bash -c "cd '$repo_root/fock_sirk' && timeout 1800 cargo test --release --test latex_cas_hamiltonian_match -- --ignored" \
    || FAILED=$((FAILED + 1))

# 2. qfm_text heavy tests: need the trained checkpoint on the external drive.
CKPT="/media/leo/e7ed9d6f-5f0a-4e19-a74e-83424bc154ba/unfer/qfm_text_runs/m8_nohash_shard0_v3/checkpoint_epoch0.qfm"
if [ -f "$CKPT" ]; then
    run_suite "qfm_text oxieml_fit_real_w --ignored" \
        bash -c "cd '$repo_root/qfm_text' && timeout 900 cargo test --release --test oxieml_fit_real_w -- --nocapture --ignored" \
        || FAILED=$((FAILED + 1))
    run_suite "qfm_text print_c0 --ignored" \
        bash -c "cd '$repo_root/qfm_text' && timeout 900 cargo test --release --test print_c0 -- --nocapture --ignored" \
        || FAILED=$((FAILED + 1))
else
    echo "=== qfm_text heavy tests (skipped) ===" | tee -a "$LOG"
    echo "  checkpoint not mounted at $CKPT — skipping oxieml_fit_real_w / print_c0" | tee -a "$LOG"
    echo "  (mount the drive and re-run the script to include them)" | tee -a "$LOG"
fi

echo
if [ "$FAILED" -eq 0 ]; then
    echo "ALL HEAVY TESTS GREEN — see $LOG" | tee -a "$LOG"
else
    echo "$FAILED heavy suite(s) FAILED — see $LOG" | tee -a "$LOG"
fi
exit "$FAILED"
