#!/usr/bin/env bash
# =============================================================================
# voice.sh — a live dialogue loop with the local Gemma4 model, like Alexa.
#
# Four interactive modes, one per input/output combination:
#
#   ./voice.sh full     microphone → you     (speak, it answers aloud)
#   ./voice.sh speak    keyboard  → you     (type, it answers aloud)
#   ./voice.sh listen   microphone → text   (speak, it answers in text)
#   ./voice.sh type     keyboard  → text    (plain chat, still dialogue-aware)
#
# One-shot modes for embedders (an emthin `\voice` block drives these):
#
#   ./voice.sh turn    [--say] [--history FILE] "TEXT"   text  → text
#   ./voice.sh dictate [--say] [--history FILE]          voice → text
#
#   The one-shot contract: exactly one JSON object on stdout —
#   {"utterance": "...", "reply": "..."} — progress on stderr, exit 0.
#   A failure is exit != 0 with `FAIL <reason>` on stderr and nothing on
#   stdout. `--history FILE` reads the rolling conversation from FILE
#   (OpenAI-style [{"role": ..., "content": ...}]) instead of the session
#   file and does not persist the new turns: the caller owns the
#   transcript, so an editor can keep it in its own document. `--say`
#   speaks the reply with piper (interactive full/speak always do).
#
# ASR: parakeet-tdt-0.6b-v2 int8 via sherpa-onnx (nix develop .#voice).
# TTS: piper en_US-lessac-medium.
# LLM: the FastFlowLM server already configured on this machine
#      (default http://127.0.0.1:8081, model gemma4-it:12b) — start it with
#      /usr/local/bin/gemma4-toggle if it is not up.
#
# The model is prompted as a *live spoken dialogue*, not a one-shot Q&A:
# dialogue_prompt.md is the system prompt, and the rolling transcript of
# the current session is sent every turn, so the model knows this is a
# continuous conversation. History survives across turns within a session
# (in-memory file) — start a session with --fresh or `voice.sh reset`.
#
# Environment overrides:
#   LLM_URL     (default http://127.0.0.1:8081/v1)
#   LLM_MODEL   (default gemma4-it:12b)
#   VOICE_MODEL (default en_US-lessac-medium.onnx; must sit next to its .json)
#   VOICE_SECS  recording length in seconds for full/listen/dictate (default 6)
#   PARAKEET    path to the parakeet wrapper (set by `nix develop .#voice`)
#   PARAKEET_THREADS  decode threads (default 4)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_URL="${LLM_URL:-http://127.0.0.1:8081/v1}"
LLM_MODEL="${LLM_MODEL:-gemma4-it:12b}"
VOICE_MODEL="${VOICE_MODEL:-${PIPER_VOICE:-/usr/share/piper5/voices/en_US-lessac-medium.onnx}}"
VOICE_SECS="${VOICE_SECS:-6}"
PARAKEET="${PARAKEET:-parakeet}"
PROMPT_FILE="$HERE/dialogue_prompt.md"
STATE_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/unfer-voice-session.json"
TMPD="$(mktemp -d /tmp/voice.XXXXXX)"
trap 'rm -rf "$TMPD"' EXIT

die() { printf 'FAIL %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing tool: $1 (run inside: nix develop .#voice)"; }

# ── argument parsing ─────────────────────────────────────────────────────────
# Flags are taken wherever they appear, so `turn --say "hi"` and
# `turn "hi" --say` both work; everything else is positional (mode word,
# then the utterance words for `turn`).
SAY=0
HISTORY=""
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --say) SAY=1; shift ;;
    --history)
      [ $# -ge 2 ] || die "--history needs a file path"
      HISTORY="$2"; shift 2 ;;
    --history=*) HISTORY="${1#--history=}"; shift ;;
    *) POS+=("$1"); shift ;;
  esac
done
set -- ${POS[@]+"${POS[@]}"}

MODE="${1:-full}"
if [ $# -gt 0 ]; then shift; fi
REST="$*"

case "$MODE" in
  turn|dictate|full|speak|listen|type) ;;
  reset) rm -f "$STATE_FILE"; echo "session cleared."; exit 0 ;;
  -h|--help|help) sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) die "unknown mode '$MODE' (want: turn | dictate | full | speak | listen | type | reset)" ;;
esac

# `--say` is a one-shot flag; the interactive modes decide speaking by mode.
case "$MODE" in turn|dictate) ;; *) SAY=0 ;; esac

need jq
need curl
# Recording is needed by the mic modes: interactive full/listen and the
# one-shot dictate.
case "$MODE" in
  dictate|listen|full)
    need pw-record
    [ -x "$PARAKEET" ] || need "$PARAKEET" ;;
esac
# Speaking: interactive full/speak always, one-shot only with --say.
if [ "$MODE" = "full" ] || [ "$MODE" = "speak" ] || [ "$SAY" = 1 ]; then
  need pw-play
  [ -f "$VOICE_MODEL" ] || die "piper voice not found: $VOICE_MODEL (nix develop .#voice sets it)"
fi

# ── session transcript ───────────────────────────────────────────────────────
# A JSON array of {role, content}. Kept on disk so an accidental exit does not
# lose the conversation; `--fresh`/`reset` starts over. With `--history` the
# caller owns the transcript, so no session file is touched at all.
if [ -z "$HISTORY" ]; then
  [ -f "$STATE_FILE" ] || echo '[]' > "$STATE_FILE"
fi

history() { cat "$STATE_FILE"; }

push_turn() { # $1=role $2=content
  local tmp="$STATE_FILE.tmp"
  jq --arg r "$1" --arg c "$2" '. + [{role: $r, content: $c}]' "$STATE_FILE" > "$tmp"
  mv "$tmp" "$STATE_FILE"
}

# ── the dialogue-aware turn ──────────────────────────────────────────────────
# System prompt + rolling history + this turn, posted OpenAI-style. The system
# prompt is what makes the model treat this as live dialogue rather than a
# document-answering session; the history is what makes it *continuous*.
ask_llm() { # $1=user utterance → prints the reply
  local utterance="$1"
  local body="$TMPD/req.json"
  local hist_json
  if [ -n "$HISTORY" ]; then
    [ -f "$HISTORY" ] || die "history file not found: $HISTORY"
    hist_json="$(cat "$HISTORY")"
    printf '%s' "$hist_json" | jq -e 'type == "array"' >/dev/null \
      || die "history file is not a JSON array: $HISTORY"
  else
    hist_json="$(cat "$STATE_FILE")"
  fi
  jq -n \
    --arg model "$LLM_MODEL" \
    --rawfile sys "$PROMPT_FILE" \
    --arg user "$utterance" \
    --argjson hist "$hist_json" \
    '{model: $model,
      temperature: 0.7,
      max_tokens: 480,
      messages: ([{role: "system", content: $sys}] + $hist + [{role: "user", content: $user}])}' \
    > "$body"
  local resp="$TMPD/resp.json"
  curl -sS --max-time 120 "$LLM_URL/chat/completions" \
    -H 'Content-Type: application/json' --data @"$body" > "$resp"
  local reply
  reply="$(jq -r '.choices[0].message.content // empty' "$resp")"
  [ -n "$reply" ] || die "no reply from $LLM_URL (model up? /usr/local/bin/gemma4-toggle): $(head -c 300 "$resp")"
  if [ -z "$HISTORY" ]; then
    push_turn user "$utterance"
    push_turn assistant "$reply"
  fi
  printf '%s' "$reply"
}

# ── audio I/O ────────────────────────────────────────────────────────────────
record() { # $1=outfile; fixed-length capture, 16 kHz mono (parakeet's rate)
  echo "● listening (${VOICE_SECS}s)…" >&2
  pw-record --rate 16000 --channels 1 --format s16 "$1" &
  local pid=$!
  sleep "$VOICE_SECS"
  kill -INT "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

transcribe() { # $1=wav → prints text (sherpa-onnx-offline emits one JSON line by default)
  "$PARAKEET" "$1" 2>/dev/null | grep '^{' | tail -n1 | jq -r '.text // empty'
}

speak() { # $1=text
  printf '%s' "$1" | piper --model "$VOICE_MODEL" --output_file "$TMPD/out.wav" 2>/dev/null
  pw-play "$TMPD/out.wav"
}

get_utterance() { # → prints the user's turn (text or audio depending on mode)
  case "$MODE" in
    type)   printf 'you> ' >&2; IFS= read -r line || exit 0; printf '%s' "$line" ;;
    speak)  printf 'you> ' >&2; IFS= read -r line || exit 0; printf '%s' "$line" ;;
    full|listen)
      record "$TMPD/in.wav"
      local t
      t="$(transcribe "$TMPD/in.wav")"
      [ -n "$t" ] || { echo "(heard nothing — try again)" >&2; return 1; }
      printf 'you> %s\n' "$t" >&2
      printf '%s' "$t" ;;
  esac
}

# ── one-shot execution (turn / dictate) ──────────────────────────────────────
# The embedder contract: one JSON object on stdout, everything else on stderr.
oneshot() {
  local utt reply
  case "$MODE" in
    turn)
      [ -n "$REST" ] || die "turn needs an utterance: voice.sh turn \"hello there\""
      utt="$REST" ;;
    dictate)
      record "$TMPD/in.wav"
      utt="$(transcribe "$TMPD/in.wav")"
      [ -n "$utt" ] || die "heard nothing (microphone up? VOICE_SECS=${VOICE_SECS}s)" ;;
  esac
  printf 'you> %s\n' "$utt" >&2
  reply="$(ask_llm "$utt")"
  if [ "$SAY" = 1 ]; then
    speak "$reply"
  fi
  jq -n --arg u "$utt" --arg r "$reply" '{utterance: $u, reply: $r}'
}

case "$MODE" in
  turn|dictate) oneshot; exit 0 ;;
esac

# ── the loop ─────────────────────────────────────────────────────────────────
echo "[voice:$MODE] talking to $LLM_MODEL at $LLM_URL — ctrl-c to stop, 'bye' to end session"
while true; do
  utt="$(get_utterance || true)"
  [ -n "${utt:-}" ] || continue
  case "$utt" in bye|exit|quit) echo "bye."; exit 0 ;; esac
  reply="$(ask_llm "$utt")"
  case "$MODE" in
    full|speak) printf 'gemma> %s\n' "$reply"; speak "$reply" ;;
    listen|type) printf 'gemma> %s\n' "$reply" ;;
  esac
done
