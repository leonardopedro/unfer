# voice/ — a live dialogue with the local Gemma4 model (Alexa-style)

Four modes, one per input/output combination. Speak or type in; get the
answer aloud or in text. The model is prompted as a **live spoken
dialogue**, not a one-shot Q&A — that distinction is the whole point:
Gemma answers differently when it knows it is mid-conversation with a
person who is speaking (short spoken turns, no markdown, ASR-error
charity, and the rolling transcript of the session is sent every turn so
answers stay in context).

| mode | input | output | command |
|---|---|---|---|
| full | microphone | spoken reply | `./voice.sh full` |
| speak | keyboard | spoken reply | `./voice.sh speak` |
| listen | microphone | text reply | `./voice.sh listen` |
| type | keyboard | text reply | `./voice.sh type` |

`./voice.sh reset` clears the session transcript.

## Running it

```sh
# 1. the speech stack (parakeet ASR + piper TTS), via the workspace flake:
nix develop .#voice

# 2. the Gemma server (FastFlowLM on the NPU, port 8081) — if it is not
#    already up:
/usr/local/bin/gemma4-toggle

# 3. talk:
cd voice && ./voice.sh full
```

No `nix develop` needed if `PARAKEET`, `PIPER_VOICE`, `piper`, `pw-record`
and `pw-play` are already on PATH (the script only checks the tools its
mode actually uses).

## The pieces

- **ASR — parakeet.** NVIDIA parakeet-tdt-0.6b-v2, int8 ONNX export
  (~661 MB), run by `sherpa-onnx-offline` through the flake's
  `parakeet-compact` wrapper. Measured RTF 0.029 on this host's CPU:
  a 6-second utterance transcribes in ~0.2 s. Pinned by sha256 in
  `../flake.nix`; `voice.sh` records at 16 kHz mono, the model's native
  rate, so no resampling sits in the path.
- **TTS — piper.** `en_US-lessac-medium` (~63 MB), also pinned in the
  flake. `speak`/`full` modes synthesize the reply to a temp wav and
  `pw-play` it.
- **LLM — Gemma4 IT 12B** via the already-configured FastFlowLM server
  (`http://127.0.0.1:8081/v1`, OpenAI-compatible `/chat/completions`).
  Overrides: `LLM_URL`, `LLM_MODEL`.
- **Dialogue awareness — `dialogue_prompt.md`.** The system prompt that
  reframes the model from "answer this document question" to "you are in
  a live, spoken, continuous dialogue": plain spoken prose (TTS reads it
  aloud, the human never sees it), 1–3 sentence turns, charity toward
  ASR mishearings, and continuity with the earlier turns in the session
  transcript (kept in `~/.local/state/unfer-voice-session.json`).

## Notes

- Recording in `full`/`listen` is fixed-length (`VOICE_SECS`, default 6
  s) — no VAD yet; say something or the transcript is empty and it
  re-prompts.
- The session file is per-user state, not repo state; `reset` or
  `--fresh`-style restarts are just deleting it.
