#!/usr/bin/env bash
# Cadabra module verification gate — runs every docs/*.cdb module and CERTIFIES
# that each labeled zero-check actually reduces to zero.
#
# ## Why this exists (the gap it closes)
#
# `cadabra2-cli -q -n <file>` exits 0 whenever the Python interpreter finishes
# without an uncaught exception. It does NOT exit non-zero when a printed CHECK
# handle comes out non-zero — a wrong coefficient, a residual e^{-1}, or a
# broken substitution all still exit 0. So "the module ran" and "the module's
# checks are zero" are different claims, and only the first is signalled by the
# exit status.
#
# That gap is not hypothetical: `docs/qg_densitized_hamiltonian.cdb` carried a
# `diff` handle that printed a non-zero remainder (a power-vs-product rendering
# mismatch) while the module still exited 0 and the Rust substring probes
# passed. Nothing structural would have caught it. This gate parses the modules'
# own "(0 = ok)" labeled lines and fails if any of them is non-zero, so a
# regression is loud instead of silent.
#
# ## Honesty conventions (matching scripts/regen_mass_gap.sh)
#
#   PASS         — module ran, and every "0 = ok" check printed 0.
#   FAIL         — module crashed (non-zero exit), or a "0 = ok" check printed
#                  a non-zero value. A real regression.
#   UNAVAILABLE  — cadabra2-cli is not on PATH / no CADABRA_CLI override. The
#                  engine is absent on this host; nothing is known to be broken.
#                  Reported separately from FAIL and never counted as PASS.
#                  --strict turns UNAVAILABLE into a failure (CI on an engine host).
#   NOTE         — informational (e.g. a module with no zero-checks to certify).
#
# This gate deliberately does NOT re-derive the physics or check >= 0 positivity
# statements; it certifies exactly one claim — every zero-intent handle is zero —
# and is silent about everything else.
#
# ## Two kinds of zero-intent handle, both certified
#
#   1. Lines the module PRINTS in the "(0 = ok)" convention (parsed from output).
#   2. Named handles the module DEFINES but does not print — auto-discovered from
#      `^<name> :=` definitions matching the ZERO_INTENT pattern below (diff,
#      Hkern, *_check, ...).  For these the harness appends a direct Python
#      `print` of the variable to a temp copy and certifies it is 0.  (Direct
#      variable reference, NOT the @() insertion macro — @() mis-resolves
#      underscored/short-stem names, see docs/CADABRA2_GOTCHAS.md.)  This is
#      what makes the 9 modules that only "exit 0" machine-certifiable instead of
#      silently unchecked.
#
# A handle that is DOCUMENTED to be a non-zero VALUE check is excepted below, so
# it is not falsely flagged.
#
# Usage:
#   bash scripts/verify_cdb_modules.sh            # run all docs/*.cdb
#   bash scripts/verify_cdb_modules.sh --strict   # UNAVAILABLE counts as failure
#   CADABRA_CLI=/path/to/cadabra2-cli bash scripts/verify_cdb_modules.sh
set -u

cd "$(dirname "$0")/.." || exit 2

STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

# Zero-intent handle names the harness auto-discovers and certifies == 0.
# Kept narrow on purpose: only names that are unambiguously verification
# kernels.  Extend this regex when a module adds a new kernel suffix.
ZERO_INTENT='^(diff[0-9]*|Hkern|Jcheck|Unorm|[A-Za-z0-9]+_check)$'

# Documented NON-zero value checks (each is referenced by its module's own
# header comment).  These are certified as *present* but not as zero.
#   Vplat_check  qg_starobinsky_einstein_frame.cdb — reduces to the plateau
#                M^4/(16 alpha) it tests (see that module's lines ~49-50,141).
VALUE_CHECKS='^(Vplat_check)$'

# Explicit per-module overrides for kernels whose names do NOT fit ZERO_INTENT.
# Format:  "<module-base> <zero-handle ...> [-- <value-handle ...>]"
#   mass_gap_norm_residual  G1..G4 are the identity checks (reduce to 0); G5 is
#                           the non-negative residual q^2 the module is ABOUT
#                           (a value, not a check — its CHECK 5 comment).
# Add a line here when a module names its kernels outside the ZERO_INTENT
# pattern, so those kernels are still certified instead of silently skipped.
EXPLICIT_OVERRIDES='mass_gap_norm_residual G1 G2 G3 G4 -- G5'

# Resolve the engine: an explicit CADABRA_CLI wins, else a PATH search.
CDB="${CADABRA_CLI:-cadabra2-cli}"

PASS=0; FAIL=0; UNAVAIL=0; NOTE=0
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()   { printf '  [PASS]       %s\n' "$1"; PASS=$((PASS+1)); }
no()   { printf '  [FAIL]       %s\n' "$1"; FAIL=$((FAIL+1)); }
na()   { printf '  [UNAVAILABLE] %s\n' "$1"; UNAVAIL=$((UNAVAIL+1)); }
note() { printf '  [NOTE]       %s\n' "$1"; NOTE=$((NOTE+1)); }

echo "verify_cdb_modules — strict=$STRICT"
echo "  engine: $CDB"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── 0. Engine present? ─────────────────────────────────────────────────────
step "0. cadabra2 engine"
if ! command -v "$CDB" >/dev/null 2>&1 && [ ! -x "$CDB" ]; then
    na "cadabra2-cli not found (set CADABRA_CLI or enter the nix devShell: nix develop ./nix/aarch64)"
    echo
    echo "VERDICT: UNAVAILABLE — the engine is absent on this host."
    echo "         No module ran, so nothing is certified and nothing is broken."
    [ "$STRICT" -eq 1 ] && { echo "         --strict: treating UNAVAILABLE as failure."; exit 1; }
    exit 0
fi
ok "engine: $("$CDB" --version 2>/dev/null | head -1 || echo present)"

# ── 1. Run every module, and certify its zero-checks ───────────────────────
step "1. run docs/*.cdb and certify every '(0 = ok)' check reaches 0"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for f in docs/*.cdb; do
    [ -f "$f" ] || continue
    base="$(basename "$f" .cdb)"
    out="$TMP/$base.out"

    # Auto-discover zero-intent handles the module DEFINES but may not print.
    # Two sources, merged in Python: (1) names matching ZERO_INTENT, (2) the
    # EXPLICIT_OVERRIDES table (which also names documented NON-zero value
    # results so they are never falsely flagged).  We append a direct Python
    # print of each discovered variable to a TEMP COPY and run that copy, so the
    # module is still executed exactly once.  (Direct variable reference, not
    # @() — @() mis-resolves underscored/short-stem names; CADABRA2_GOTCHAS.md.)
    disc_out="$(DISC_BASE="$base" ZERO_INTENT="$ZERO_INTENT" VALUE_CHECKS="$VALUE_CHECKS" \
        EXPLICIT_OVERRIDES="$EXPLICIT_OVERRIDES" python3 - "$f" <<'PY'
import os, re, sys
path = sys.argv[1]
base = os.environ["DISC_BASE"]
zi = re.compile(os.environ["ZERO_INTENT"])
vc = re.compile(os.environ["VALUE_CHECKS"])
ov_zero, ov_value = [], []
for line in os.environ.get("EXPLICIT_OVERRIDES", "").splitlines():
    parts = line.split()
    if not parts or parts[0] != base:
        continue
    rest = parts[1:]
    if "--" in rest:
        i = rest.index("--")
        ov_zero += rest[:i]
        ov_value += rest[i+1:]
    else:
        ov_zero += rest
text = open(path, errors="replace").read()
defined = set(re.findall(r'^([A-Za-z][A-Za-z0-9_]*)\s*:=', text, re.M))
value = {n for n in defined if vc.fullmatch(n)} | set(ov_value)
disc = sorted(({n for n in defined if zi.fullmatch(n)} | set(ov_zero)) - value)
print("DISCOVERED:" + ",".join(disc))
print("VALUE:" + ",".join(sorted(value)))
PY
)"
    discovered="$(printf '%s\n' "$disc_out" | sed -n 's/^DISCOVERED://p' | tr ',' '\n' | grep . || true)"
    value_presents="$(printf '%s\n' "$disc_out" | sed -n 's/^VALUE://p' | tr ',' '\n' | grep . || true)"

    runfile="$f"
    if [ -n "$discovered" ]; then
        runfile="$TMP/$base.probe.cdb"
        cp "$f" "$runfile"
        # Unique marker per handle so the parser cannot confuse it with module
        # prose.  Direct variable reference (not @()) sidesteps the underscore
        # gotcha.  Non-zero-intent / value checks are NOT probed as zero.
        {
            printf '\n'
            for h in $discovered; do
                printf 'print("ZCHK %s<<", %s, ">>");\n' "$h" "$h"
            done
        } >> "$runfile"
    fi

    # Run the module (probe copy if any). Capture combined output; preserve exit.
    if ! timeout 240 "$CDB" -q -n "$runfile" >"$out" 2>&1; then
        no "$base — module exited non-zero (crash). First error line:"
        grep -iE 'error|traceback|exception|does not exist' "$out" | head -1 | sed 's/^/             /'
        continue
    fi

    # Certify BOTH kinds of zero-intent handle in one pass:
    #   (a) printed lines labeled "(0 = ok)" — value is after the LAST ':';
    #   (b) auto-discovered handles, read from the ZCHK<name><<val>> markers.
    # A bare '0' (ignoring spaces / a trailing parenthetical) is the pass.
    report="$(python3 - "$out" <<PY
import re, sys
path = sys.argv[1]
conv = re.compile(r'\(\s*0\s*=\s*ok\s*\)')
zchk = re.compile(r'ZCHK ([A-Za-z0-9_]+)<< (.*?) >>')
total = 0
bad = []
for i, raw in enumerate(open(path, errors='replace'), 1):
    line = raw.rstrip('\n')
    # (a) the module's own "(0 = ok)" printed checks
    if conv.search(line):
        total += 1
        if ':' not in line:
            bad.append((i, 'NO-COLON', line.strip()))
        else:
            val = line.rsplit(':', 1)[1].strip()
            v = val.split('(')[0].strip().replace(' ', '')
            if v != '0':
                bad.append((i, v, line.strip()))
    # (b) the auto-discovered ZCHK probes
    m = zchk.search(line)
    if m:
        total += 1
        name, val = m.group(1), m.group(2).strip()
        v = val.split('(')[0].strip().replace(' ', '')
        if v != '0':
            bad.append((i, v, f'{name} (discovered handle) = {val}'))
print(total)
for i, v, l in bad:
    sys.stderr.write(f'{i}|{v}|{l}\n')
PY
)"
    n_total="$(printf '%s' "$report" | head -1)"
    n_bad="$(printf '%s' "$report" | tail -n +2 | grep -c . || true)"
    n_disc="$(printf '%s' "$discovered" | grep -c . || true)"
    n_val="$(printf '%s' "$value_presents" | grep -c . || true)"

    if [ "${n_bad:-0}" -gt 0 ]; then
        no "$base — $n_bad of $n_total zero-intent check(s) printed NON-ZERO:"
        printf '%s' "$report" | tail -n +2 | grep . | head -6 | while IFS='|' read -r ln v l; do
            printf '             line %s -> [%s]\n               %s\n' "$ln" "$v" "$l"
        done
    elif [ "${n_total:-0}" -gt 0 ]; then
        extra=""
        [ "${n_disc:-0}" -gt 0 ] && extra="$extra; $n_disc discovered handle(s) certified"
        [ "${n_val:-0}" -gt 0 ] && extra="$extra; $n_val documented value-check(s) present (not zero)"
        ok "$base — ran; $n_total zero-intent check(s) all reached 0$extra"
    else
        note "$base — ran; no zero-intent handles to certify (exit 0 only)"
    fi
done

# ── verdict ─────────────────────────────────────────────────────────────────
printf '\n────────────────────────────────────────────────────────\n'
printf 'PASS %d   FAIL %d   UNAVAILABLE %d   NOTE %d\n' "$PASS" "$FAIL" "$UNAVAIL" "$NOTE"
if [ "$FAIL" -gt 0 ]; then
    echo "VERDICT: FAILED — at least one module crashed or a zero-check is non-zero."
    exit 1
fi
if [ "$UNAVAIL" -gt 0 ]; then
    echo "VERDICT: INCOMPLETE — some modules could not run. This is NOT a green gate."
    [ "$STRICT" -eq 1 ] && { echo "         --strict: treating UNAVAILABLE as failure."; exit 1; }
    exit 0
fi
echo "VERDICT: all modules ran; every labeled zero-check certified 0."
exit 0
