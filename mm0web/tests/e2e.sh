#!/usr/bin/env bash
# End-to-end driver for mm0web.
#
#   ./tests/e2e.sh            all scenarios
#   ./tests/e2e.sh 1 3        only scenarios 1 and 3
#
# Everything runs against a mock Zenodo (`tests/mock_zenodo.py`) so the real
# deposit path — create draft, upload, publish, new version, resume — is
# exercised without a network or an account. Scenarios:
#
#   1  happy path: verify, publish, status → published, DOI, events, idempotence
#   2  retry: an injected publish failure is retried, then lands
#   3  crash-resume: killed mid-upload, the restart reuses the same deposit
#   4  limits: per-minute rate limit and the per-client daily cap
#   5  no API key: verify and store work, publish says so plainly
#   6  search: the semantic arm over a fake OpenAI-schema embedder (rrf
#      fusion, JSON + HTML), the `_bulk` NDJSON export, and a dead embed
#      endpoint degrading to substring-only search

set -u

SELECTED=("$@")
should_run() { # should_run <n> — true unless the user picked specific scenarios
  [ ${#SELECTED[@]} -eq 0 ] && return 0
  local n
  for n in "${SELECTED[@]}"; do [ "$n" = "$1" ] && return 0; done
  return 1
}

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WEB_BIN="$ROOT/target/debug/mm0web"
MOCK="$ROOT/mm0web/tests/mock_zenodo.py"
MOCK_PORT="${MOCK_PORT:-8919}"
WEB_PORT="${WEB_PORT:-18919}"
EMBED_PORT="${EMBED_PORT:-8929}"
BASE="http://127.0.0.1:$WEB_PORT"
ZBASE="http://127.0.0.1:$MOCK_PORT/api"
WORK="$(mktemp -d /tmp/mm0web-e2e.XXXXXX)"
PASS=0; FAIL=0
MOCK_PID=""; WEB_PID=""; EMBED_PID=""

say()  { printf '%s\n' "$*"; }
ok()   { PASS=$((PASS+1)); say "  ok   $*"; }
bad()  { FAIL=$((FAIL+1)); say "  FAIL $*"; }
check() { # check <name> <expected> <actual>
  if [ "$2" = "$3" ]; then ok "$1 ($3)"; else bad "$1: expected $2, got $3"; fi
}

jget() { # jget <json-file> <dotted.path>
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    v = json.load(open(sys.argv[1]))
except Exception:
    print(""); raise SystemExit
for k in sys.argv[2].split("."):
    if isinstance(v, list):
        try: v = v[int(k)]
        except Exception: v = None
    elif isinstance(v, dict):
        v = v.get(k)
    else:
        v = None
    if v is None: break
print("" if v is None else v)
PY
}

stats_field() { # stats_field <key>
  curl -sS -m 5 "http://127.0.0.1:$MOCK_PORT/__stats" -o "$WORK/stats.json" 2>/dev/null
  jget "$WORK/stats.json" "$1"
}

run_mock() { # run_mock [VAR=val ...]
  env MOCK_PORT="$MOCK_PORT" "$@" python3 "$MOCK" >"$WORK/mock.log" 2>&1 &
  MOCK_PID=$!
  wait_url "http://127.0.0.1:$MOCK_PORT/__stats" 20 || return 1
}

run_web() { # run_web [VAR=val ...]  — overrides come after the defaults
  env \
    MM0WEB_ZENODO_BASE="$ZBASE" \
    ZENODO_API_KEY="${E2E_KEY-mock-key}" \
    MM0WEB_CREATOR="Ada Lovelace" \
    MM0WEB_CREATOR_ORCID="0000-0002-1825-0097" \
    MM0WEB_PUBLIC_URL="$BASE" \
    MM0WEB_RATE="${E2E_RATE:-200}" \
    MM0WEB_DAILY_CAP="${E2E_DAILY_CAP:-50}" \
    MM0WEB_ZENODO_SQUASH_AFTER=3 \
    "$@" \
    "$WEB_BIN" --bind "127.0.0.1:$WEB_PORT" --state "$WORK/state" >"$WORK/web.log" 2>&1 &
  WEB_PID=$!
  wait_url "$BASE/healthz" 20 || return 1
}

stop_web()  { [ -n "$WEB_PID" ] && kill -9 "$WEB_PID" 2>/dev/null; wait "$WEB_PID" 2>/dev/null; WEB_PID=""; }
stop_mock() { [ -n "$MOCK_PID" ] && kill "$MOCK_PID" 2>/dev/null; wait "$MOCK_PID" 2>/dev/null; MOCK_PID=""; }
stop_embed() { [ -n "$EMBED_PID" ] && kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null; EMBED_PID=""; }
cleanup() { stop_web; stop_mock; stop_embed; rm -rf "$WORK"; }
trap cleanup EXIT

wait_url() { # wait_url <url> <seconds>
  local i=0
  while [ "$i" -lt "$(( $2 * 10 ))" ]; do
    if curl -sS -m 2 -o /dev/null "$1" 2>/dev/null; then return 0; fi
    sleep 0.1; i=$((i+1))
  done
  say "  FAIL $1 never came up"; tail -5 "$WORK/web.log" 2>/dev/null; return 1
}

# Fast fixture: a spec and the same text as its proof file (1 sort), which the
# real toolchain accepts in ~50ms. `peano` is used once, as the big case.
FIX="$WORK/fix"; mkdir -p "$FIX"
printf 'sort Prop;\n' > "$FIX/fast.mm0"
cp "$FIX/fast.mm0" "$FIX/fast.mm1"
# a second, distinct declaration so the search corpus has two rows
printf 'sort Nat;\n' > "$FIX/nat.mm0"
cp "$FIX/nat.mm0" "$FIX/nat.mm1"
# spec is fine; the proof file names a sort the spec never declared
printf 'sort Prop;\n' > "$FIX/bad.mm0"
printf 'sort Prop;\nsort Nope;\n' > "$FIX/bad.mm1"

submit() { # submit <verify|publish> <title> <mm0> <mm1> [extra curl args]
  local action="$1" title="$2" mm0="$3" mm1="$4"; shift 4
  curl -sS -m "${CURL_TIMEOUT:-300}" -o "$WORK/out" -w '%{http_code}' \
    -X POST "$BASE/$action" -H 'Accept: application/json' \
    --data-urlencode "project=unfer" \
    --data-urlencode "title=$title" \
    --data-urlencode "mm0@$mm0" \
    --data-urlencode "mm1@$mm1" "$@" 2>/dev/null
}

wait_published() { # wait_published <proof-id> <seconds>
  local id="$1" i=0 state=""
  while [ "$i" -lt "$2" ]; do
    curl -sS -m 5 "$BASE/proof?project=unfer&id=$id" -o "$WORK/proof.json" 2>/dev/null
    state="$(jget "$WORK/proof.json" state)"
    [ "$state" = "published" ] && return 0
    sleep 0.5; i=$((i+1))
  done
  return 1
}

# ---------------------------------------------------------------------------
if should_run 1; then
say "== scenario 1: happy path =="
rm -rf "$WORK/state"   # every scenario pairs a fresh mock with fresh local state
run_mock
run_web

check "healthz" 200 "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' "$BASE/healthz")"
say "  $(curl -sS -m 5 "$BASE/healthz" | head -1)"
check "index page" 200 "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' "$BASE/")"

check "verify valid proof" 200 "$(submit verify fast "$FIX/fast.mm0" "$FIX/fast.mm1")"
[ "$(jget "$WORK/out" accepted)" = "True" ] && ok "verifier says accepted" || bad "body: $(head -c 200 "$WORK/out")"

check "verify rejects a bad proof" 400 "$(submit verify bad "$FIX/bad.mm0" "$FIX/bad.mm1")"
[ -n "$(jget "$WORK/out" stage)" ] && ok "rejection names the stage ($(jget "$WORK/out" stage))" || bad "body: $(head -c 200 "$WORK/out")"

# the browser (HTML) path as well as the JSON one
HTML="$(curl -sS -m 60 -X POST "$BASE/verify" \
  --data-urlencode "project=unfer" --data-urlencode "title=fast" \
  --data-urlencode "mm0@$FIX/fast.mm0" --data-urlencode "mm1@$FIX/fast.mm1")"
echo "$HTML" | grep -q "<h1>Accepted</h1>" && ok "HTML verify renders" || bad "html: $(echo "$HTML" | head -c 200)"

check "publish queues with 202" 202 "$(submit publish peano-fast "$FIX/fast.mm0" "$FIX/fast.mm1")"
FP="$(jget "$WORK/out" proof)"
say "  queued proof id: $FP"
[ "$(jget "$WORK/out" queued)" = "True" ] && ok "response says queued" || bad "not queued: $(head -c 300 "$WORK/out")"

if wait_published "$FP" 40; then ok "status reached published"; else bad "never published: $(cat "$WORK/proof.json" 2>/dev/null)"; fi
DOI="$(jget "$WORK/proof.json" doi)"
REC="$(jget "$WORK/proof.json" record)"
say "  doi=$DOI record=$REC"
[ -n "$DOI" ] && ok "DOI surfaced on /proof" || bad "no DOI on /proof"
[ -n "$REC" ] && ok "record id surfaced on /proof" || bad "no record id"

ST="$(curl -sS -m 5 "$BASE/status?project=unfer&id=$FP")"
echo "$ST" | grep -q "doi.org/$DOI" && ok "status page links the DOI" || bad "status page has no DOI link"
echo "$ST" | grep -q "records/$REC" && ok "status page links the record" || bad "status page has no record link"
echo "$ST" | grep -q 'http-equiv="refresh"' && bad "status still refreshing after publish" || ok "status stopped refreshing"

curl -sS -m 5 "$BASE/events?project=unfer" -o "$WORK/events.json"
python3 -c 'import json,sys; print(",".join(e["action"] for e in json.load(open(sys.argv[1]))["events"]))' "$WORK/events.json" > "$WORK/actions.txt"
say "  audit actions: $(cat "$WORK/actions.txt")"
grep -q "verified" "$WORK/actions.txt" && ok "audit records verified" || bad "no verified event"
grep -q "queued" "$WORK/actions.txt" && ok "audit records queued" || bad "no queued event"
grep -q "published" "$WORK/actions.txt" && ok "audit records published" || bad "no published event"

# idempotence: identical content must not be pushed again
check "re-publish is idempotent" 200 "$(submit publish peano-fast "$FIX/fast.mm0" "$FIX/fast.mm1")"
[ "$(jget "$WORK/out" already)" = "True" ] && ok "reports already published" || bad "not flagged already: $(head -c 200 "$WORK/out")"
[ "$(stats_field deposits)" = "1" ] && ok "still one deposit" || bad "deposits=$(stats_field deposits)"

# a second, different proof must arrive in the same deposit
check "second proof queues" 202 "$(submit publish second "$FIX/fast.mm0" "$FIX/fast.mm1")"
FP2="$(jget "$WORK/out" proof)"
wait_published "$FP2" 40 && ok "second proof published" || bad "second proof stuck: $(cat "$WORK/proof.json" 2>/dev/null)"
sleep 0.5
stats_field deposits >/dev/null   # refresh before quoting the file
say "  mock stats: $(cat "$WORK/stats.json")"
[ "$(stats_field deposits)" = "1" ] && ok "both revisions share one deposit" || bad "deposits=$(stats_field deposits) (expected 1)"

curl -sS -m 20 "$BASE/manifest?project=unfer" -o "$WORK/manifest.json"
[ "$(jget "$WORK/manifest.json" project)" = "unfer" ] && ok "manifest endpoint serves the project" || bad "manifest: $(head -c 200 "$WORK/manifest.json")"

# the real toolchain, once, with the full peano development
check "peano verifies" 200 "$(CURL_TIMEOUT=400 submit verify peano /tmp/mm0/examples/peano.mm0 /tmp/mm0/examples/peano.mm1)"
grep -q "accepted" "$WORK/out" && ok "peano accepted by mm0-rs + mm0-c" || bad "peano: $(head -c 200 "$WORK/out")"

stop_web; stop_mock
fi

# ---------------------------------------------------------------------------
if should_run 2; then
say "== scenario 2: an injected publish failure is retried =="
rm -rf "$WORK/state"
run_mock MOCK_FAIL_PUBLISH=1
run_web
check "publish still answers 202" 202 "$(submit publish retry "$FIX/fast.mm0" "$FIX/fast.mm1")"
FP="$(jget "$WORK/out" proof)"
if wait_published "$FP" 40; then ok "retried and landed"; else bad "gave up: $(cat "$WORK/proof.json" 2>/dev/null)"; fi
curl -sS -m 5 "$BASE/events?project=unfer" -o "$WORK/events.json"
grep -q '"publish-failed"' "$WORK/events.json" && ok "the failed attempt was audited" || bad "no publish-failed event"
grep -q '"published"' "$WORK/events.json" && ok "the retry was audited" || bad "no published event"
[ "$(stats_field publishes)" = "1" ] && ok "published exactly once" || bad "publishes=$(stats_field publishes)"
stop_web; stop_mock
fi

# ---------------------------------------------------------------------------
if should_run 3; then
say "== scenario 3: killed mid-upload, the restart resumes the same deposit =="
rm -rf "$WORK/state"
run_mock MOCK_DELAY_UPLOAD=3
run_web
submit publish crashy "$FIX/fast.mm0" "$FIX/fast.mm1" >/dev/null 2>&1 &
SUBMIT_PID=$!
sleep 1.2                 # mid-payload-upload
kill -9 "$WEB_PID" 2>/dev/null
wait "$WEB_PID" 2>/dev/null; WEB_PID=""
wait "$SUBMIT_PID" 2>/dev/null
say "  service killed mid-push"
run_web                  # same state dir, same mock
check "resubmitting the queued proof is accepted" 202 "$(submit publish crashy "$FIX/fast.mm0" "$FIX/fast.mm1")"
FP="$(jget "$WORK/out" proof)"
if wait_published "$FP" 90; then ok "resumed and published"; else bad "did not resume: $(cat "$WORK/proof.json" 2>/dev/null)"; fi
sleep 0.5
stats_field deposits >/dev/null   # refresh before quoting the file
say "  mock stats: $(cat "$WORK/stats.json")"
[ "$(stats_field deposits)" = "1" ] && ok "no orphaned deposit was created" || bad "orphaned deposit: $(cat "$WORK/stats.json")"
stop_web; stop_mock
fi

# ---------------------------------------------------------------------------
if should_run 4; then
say "== scenario 4: rate limit and daily cap =="
rm -rf "$WORK/state"
run_mock
E2E_RATE=3 run_web
for i in 1 2 3; do submit verify "r$i" "$FIX/fast.mm0" "$FIX/fast.mm1" >/dev/null; done
check "fourth request inside the window is 429" 429 "$(submit verify r4 "$FIX/fast.mm0" "$FIX/fast.mm1")"
stop_web; rm -rf "$WORK/state"
E2E_DAILY_CAP=2 run_web
check "first new proof passes the daily cap" 202 "$(submit publish d1 "$FIX/fast.mm0" "$FIX/fast.mm1")"
FP="$(jget "$WORK/out" proof)"; wait_published "$FP" 40 || bad "d1 never published"
check "second new proof passes" 202 "$(submit publish d2 "$FIX/fast.mm0" "$FIX/fast.mm1")"
FP="$(jget "$WORK/out" proof)"; wait_published "$FP" 40 || bad "d2 never published"
# re-submitting an already-published proof is free: the content-addressed id
# dedup's it (200/already) instead of burning a daily slot or a queue slot
check "re-submitting a published proof is not capped" 200 "$(submit publish d1 "$FIX/fast.mm0" "$FIX/fast.mm1")"
[ "$(jget "$WORK/out" already)" = "True" ] && ok "and is deduplicated, not re-queued" || bad "body: $(head -c 200 "$WORK/out")"
check "third new proof is capped" 429 "$(submit publish d3 "$FIX/fast.mm0" "$FIX/fast.mm1")"
grep -q "daily cap" "$WORK/out" && ok "cap message is explicit" || bad "cap body: $(head -c 200 "$WORK/out")"
stop_web; stop_mock
fi

# ---------------------------------------------------------------------------
if should_run 5; then
say "== scenario 5: no API key =="
rm -rf "$WORK/state"
run_mock
run_web ZENODO_API_KEY=
check "publish without a key is 502" 502 "$(submit publish nokey "$FIX/fast.mm0" "$FIX/fast.mm1")"
grep -q "ZENODO_API_KEY" "$WORK/out" && ok "message names the missing key" || bad "body: $(head -c 200 "$WORK/out")"
NOKEY_ID="$(curl -sS -m 5 "$BASE/proofs?project=unfer" | grep -o 'id=nokey-[a-f0-9]*' | head -1 | cut -d= -f2)"
[ -n "$NOKEY_ID" ] && ok "the proof was still stored locally ($NOKEY_ID)" || bad "proof not stored locally"
curl -sS -m 5 "$BASE/proof?project=unfer&id=$NOKEY_ID" -o "$WORK/proof.json"
[ "$(jget "$WORK/proof.json" state)" = "verified" ] && ok "state stays verified, not published" || bad "state=$(jget "$WORK/proof.json" state)"
check "verify still works without a key" 200 "$(submit verify nokey2 "$FIX/fast.mm0" "$FIX/fast.mm1")"
check "healthz still ready" 200 "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' "$BASE/healthz")"
stop_web; stop_mock
fi

# ---------------------------------------------------------------------------
if should_run 6; then
say "== scenario 6: semantic search, rrf fusion, and the _bulk export =="
rm -rf "$WORK/state"

# A stand-in embedding server speaking the OpenAI /embeddings schema:
# bag-of-words vectors over 4096 md5 bins, so a query sharing a word with
# a declaration scores cosine > 0 while unrelated rows do not. The
# transport and the wiring are the point here — curl -> socket -> reply ->
# index -> cosine -> rrf — not the geometry; the ranking semantics have
# their own unit tests in Rust.
cat > "$WORK/fake_embed.py" <<'PY'
import hashlib, json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

def vec(text):
    v = [0.0] * 4096
    for w in text.lower().split():
        w = w.strip("(),.;:{}[]\\\"'")
        if w:
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 4096] += 1.0
    return v

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/embeddings":
            self.send_error(404)
            return
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n))
        body = json.dumps({"data": [
            {"index": i, "embedding": vec(t)}
            for i, t in enumerate(req["input"])
        ]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
PY
python3 "$WORK/fake_embed.py" "$EMBED_PORT" >"$WORK/embed.log" 2>&1 &
EMBED_PID=$!
# wait_url only needs an HTTP answer; the handler has no GET, and a 501
# is still proof the socket is up.
wait_url "http://127.0.0.1:$EMBED_PORT/embeddings" 10 || exit 1

# no API key: publishing returns 502 but stores locally (scenario 5),
# which is all the corpus needs.
run_web ZENODO_API_KEY= MM0WEB_EMBED_URL="http://127.0.0.1:$EMBED_PORT"

check "publish stores without a key (prop)" 502 "$(submit publish sem-prop "$FIX/fast.mm0" "$FIX/fast.mm1")"
check "publish stores without a key (nat)" 502 "$(submit publish sem-nat "$FIX/nat.mm0" "$FIX/nat.mm1")"

i=0
while [ "$i" -lt 20 ]; do
  curl -sS -m 5 "$BASE/api/v1/search?q=zzzz" -o "$WORK/s.json" 2>/dev/null
  [ "$(jget "$WORK/s.json" corpus)" = "2" ] && break
  sleep 0.5; i=$((i+1))
done
check "corpus indexed" 2 "$(jget "$WORK/s.json" corpus)"

curl -sS -m 30 "$BASE/api/v1/search?q=Prop" -o "$WORK/s.json"
check "search: corpus" 2 "$(jget "$WORK/s.json" corpus)"
check "search: embedder reported" "embeddinggemma-2@0" "$(jget "$WORK/s.json" embedder)"
check "search: substring arm answered" 1 "$(jget "$WORK/s.json" arms.substring)"
SEM="$(jget "$WORK/s.json" arms.semantic)"
[ -n "$SEM" ] && [ "$SEM" -ge 1 ] 2>/dev/null && ok "semantic arm answered ($SEM row(s))" || bad "semantic arm silent: $(head -c 300 "$WORK/s.json")"
check "search: fusion is rrf" rrf "$(jget "$WORK/s.json" fusion)"
check "search: no arm unavailable" "" "$(jget "$WORK/s.json" unavailable.semantic)"
check "search: top hit is Prop" Prop "$(jget "$WORK/s.json" results.0.declaration)"

curl -sS -m 30 "$BASE/api/v1/search?q=Nat" -o "$WORK/s.json"
check "search (nat): top hit is Nat" Nat "$(jget "$WORK/s.json" results.0.declaration)"
check "search (nat): fusion is rrf" rrf "$(jget "$WORK/s.json" fusion)"

HTML="$(curl -sS -m 30 "$BASE/search?q=Prop")"
echo "$HTML" | grep -q "semantic via embeddinggemma-2@0" && ok "HTML page names the model" || bad "html: $(echo "$HTML" | head -c 300)"
echo "$HTML" | grep -q "fused by rrf" && ok "HTML page says fused by rrf" || bad "html has no fusion note: $(echo "$HTML" | head -c 300)"

# ---- the _bulk export: the load side of the xerj arm --------------------
curl -sS -m 30 -D "$WORK/bulk.hdr" "$BASE/api/v1/search-bulk" -o "$WORK/bulk.ndjson"
grep -qi '^content-type: application/x-ndjson' "$WORK/bulk.hdr" && ok "bulk content-type is x-ndjson" || bad "headers: $(head -2 "$WORK/bulk.hdr" | tr -d '\\r')"
BULK_DOCS="$(python3 - "$WORK/bulk.ndjson" <<'PY'
import json, sys
lines = [l for l in open(sys.argv[1]).read().splitlines() if l]
if len(lines) % 2:
    print("odd line count %d" % len(lines))
    raise SystemExit
n = 0
for i, line in enumerate(lines):
    obj = json.loads(line)          # every line must stand alone
    if i % 2 == 0:
        assert "index" in obj and "_id" in obj["index"], obj
    else:
        for key in ("project", "proof", "title", "name", "kind",
                    "line", "statement", "text"):
            assert key in obj, key
        n += 1
print(n)
PY
)"
check "bulk exports every declaration as a doc" 2 "$BULK_DOCS"
grep -q '/Prop@' "$WORK/bulk.ndjson" && ok "bulk carries the Prop _id" || bad "no /Prop@ id in the export"
grep -q '/Nat@' "$WORK/bulk.ndjson" && ok "bulk carries the Nat _id" || bad "no /Nat@ id in the export"

# ---- a dead embed endpoint degrades, it does not take search down -------
stop_web
stop_embed                 # the same port, now refusing connections
run_web ZENODO_API_KEY= MM0WEB_EMBED_URL="http://127.0.0.1:$EMBED_PORT"
curl -sS -m 30 "$BASE/api/v1/search?q=Prop" -o "$WORK/s.json"
check "dead embedder: substring still answers" 1 "$(jget "$WORK/s.json" count)"
check "dead embedder: fusion falls back to single" single "$(jget "$WORK/s.json" fusion)"
DEAD="$(jget "$WORK/s.json" unavailable.semantic)"
[ -n "$DEAD" ] && ok "semantic arm says why it is down ($DEAD)" || bad "no semantic unavailability: $(head -c 300 "$WORK/s.json")"
check "search still reports the configured model" "embeddinggemma-2@0" "$(jget "$WORK/s.json" embedder)"
stop_web
fi

# ---------------------------------------------------------------------------
say ""
say "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
