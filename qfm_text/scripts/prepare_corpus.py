#!/usr/bin/env python3
"""
Stage 0: Tokenize a Wikipedia-derived text corpus into little-endian u32 shards.

Reads the WikiText-103 test split (small enough to attribute under CC-BY-SA
without vendoring the full corpus), tokenizes with a 16k-vocab BPE, and
emits the same shard/manifest format the Rust crate consumes.

Outputs:
  ${SHARD_DIR}/shard_{i}.bin   - one shard per output file (raw u32 LE tokens)
  ${SHARD_DIR}/manifest.json   - vocab_size, tokens_per_shard, sha256s, etc.

The script is the *fallback* path described in QFM_TEXT_HRM_PLAN.md §"Stage 0
fallback gate": it tokenizes the corpus directly with the BPE we train
ourselves, rather than running HRM-Text's data_io. The shard format is
the interface, so the rest of the pipeline is unaffected.

Usage:
  python3 prepare_corpus.py --out /path/to/shards --shard-tokens 524288

CC-BY-SA attribution for the WikiText-103 test split is written into
  ${SHARD_DIR}/ATTRIBUTION.txt
alongside the shard files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

# --- CC-BY-SA attribution: this string is written verbatim into every
# shard directory we produce, regardless of whether the data was downloaded
# from the official WikiText-103 release or constructed locally. ---
WIKITEXT_ATTRIBUTION = """\
WikiText-103 (Merity et al., 2016) is used here under the Creative Commons
Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
The test split is small enough that a derived subset can be redistributed
for test-fixture purposes; for any production training run, download
WikiText-103 yourself from the official source.

  Paper: Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher.
         "Pointer Sentinel Mixture Models."  ICLR 2017.
  Source: https://wikitext.smerity.com/wikitext-103
  License: https://creativecommons.org/licenses/by-sa/4.0/
"""

WIKITEXT_URL = "https://wikitext.smerity.com/wikitext-103-v1.zip"
# sha256 of the zip (recorded after the first download). The server's
# ETag header is `9ddaacaf6af0710eda8c456decff7832`, but that is the
# ETag (which may include compression metadata), not the sha256 of
# the bytes; the actual file sha256 is below.
WIKITEXT_SHA256 = "242ba0f20b329cfdf1ccc61e9e9e5b59becf189db7f7a81cd2a0e2fc31539590"


def download(url: str, dest: Path) -> None:
    """Stream-download `url` to `dest`, printing progress every 10 MB."""
    print(f"[fetch] downloading {url} -> {dest}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "qfm_text/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as fh:
        total = int(resp.headers.get("Content-Length", "0"))
        done = 0
        chunk = 1 << 20  # 1 MiB
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if total > 0 and done % (10 * chunk) < chunk:
                pct = 100.0 * done / total
                print(f"  {done / 1e6:6.1f} MB / {total / 1e6:6.1f} MB ({pct:5.1f}%)",
                      file=sys.stderr)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    if WIKITEXT_SHA256 and h != WIKITEXT_SHA256:
        raise RuntimeError(
            f"WikiText-103 sha256 mismatch: expected {WIKITEXT_SHA256}, got {h}"
        )
    print(f"[fetch] sha256 = {h}", file=sys.stderr)


def extract_wikitext_split(zip_path: Path, work: Path, split: str) -> str:
    """Extract one WikiText-103 split (`train` / `valid` / `test`) to a
    single .wiki text file. The default for the QFM-Text plan was the
    test split; the production target is the train split (~100M tokens)."""
    work.mkdir(parents=True, exist_ok=True)
    target = work / f"wiki.{split}.tokens"
    if target.exists() and target.stat().st_size > 0:
        print(f"[extract] reusing {target}", file=sys.stderr)
        return target.read_text(encoding="utf-8")
    print(f"[extract] opening {zip_path}", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as zf:
        needle = f"wiki.{split}.tokens"
        for name in zf.namelist():
            if name.endswith(needle):
                with zf.open(name) as src, target.open("wb") as dst:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf:
                            break
                        dst.write(buf)
                break
        else:
            raise RuntimeError(f"{needle} not found inside the zip")
    print(f"[extract] wrote {target} ({target.stat().st_size / 1e6:.1f} MB)",
          file=sys.stderr)
    return target.read_text(encoding="utf-8")


_WHITESPACE = re.compile(r"\s+")


def normalize_wikitext(text: str) -> str:
    """WikiText-103 pre-tokenization: lowercased, headings kept, articles
    separated by blank lines (the standard convention from Merity et al.).

    We do not re-implement the original Moses-style pre-tokenizer because
    the BPE handles whitespace; we only collapse runs of whitespace to a
    single space and strip the leading/trailing whitespace on each line.
    """
    out_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            out_lines.append("")
            continue
        line = _WHITESPACE.sub(" ", line)
        out_lines.append(line)
    return "\n".join(out_lines)


def train_bpe(text: str, vocab_size: int):
    """Train a 16k-vocab byte-level BPE on `text`. Returns the trained
    tokenizer instance and its serialized JSON (for hashing / reload)."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDec
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDec()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=False,
    )
    # WikiText-103 fits in RAM; pass as a single iterator.
    print(f"[tokenizer] training BPE vocab_size={vocab_size} ...", file=sys.stderr)
    tok.train_from_iterator([text], trainer=trainer)
    print(f"[tokenizer] done. vocab = {tok.get_vocab_size()}", file=sys.stderr)
    return tok


def tokenize(tok, text: str) -> list[int]:
    """Encode a string into a flat list of token ids. For very long
    texts the tokenizers library's `encode` builds the full
    offsets/ids list in one shot, which can OOM on 500 MB inputs.
    Stream the text in line-batches and concatenate."""
    out: list[int] = []
    # Use `encode_batch` with chunks of lines for memory efficiency.
    # A chunk size of ~5000 lines keeps the peak per-batch ids list
    # under ~5 MB on WikiText-103 (avg ~30 tokens / line).
    BATCH_LINES = 5000
    lines = text.splitlines(keepends=True)
    for i in range(0, len(lines), BATCH_LINES):
        chunk = lines[i : i + BATCH_LINES]
        joined = "".join(chunk)
        enc = tok.encode(joined)
        out.extend(enc.ids)
    return out


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_shards(
    token_ids: list[int],
    out_dir: Path,
    tokens_per_shard: int,
    manifest_extra: dict,
) -> dict:
    """Write `token_ids` into shards of `tokens_per_shard` u32s each, and
    return the parsed manifest dict (also written as manifest.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tokens = len(token_ids)
    n_shards = (n_tokens + tokens_per_shard - 1) // tokens_per_shard
    shards: list[dict] = []
    for i in range(n_shards):
        a = i * tokens_per_shard
        b = min(n_tokens, a + tokens_per_shard)
        chunk = token_ids[a:b]
        path = out_dir / f"shard_{i:05d}.bin"
        blob = struct.pack(f"<{len(chunk)}I", *chunk)
        path.write_bytes(blob)
        shards.append({
            "path": path.name,
            "n_tokens": len(chunk),
            "sha256": sha256_bytes(blob),
        })
        print(f"[shard] {path.name}: {len(chunk)} tokens", file=sys.stderr)
    manifest = {
        "schema": "qfm_text.shard_manifest/v1",
        "vocab_size": manifest_extra["vocab_size"],
        "tokens_per_shard": tokens_per_shard,
        "n_shards": n_shards,
        "n_tokens": n_tokens,
        "shards": shards,
        "attribution": "WikiText-103 test split (CC BY-SA 4.0)",
    }
    manifest.update(manifest_extra)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="Output shard directory")
    p.add_argument("--split", choices=["train", "valid", "test"], default="test",
                   help="WikiText-103 split (default: test)")
    p.add_argument("--shard-tokens", type=int, default=524_288,
                   help="Tokens per shard (default: 524288 = 2 MiB u32)")
    p.add_argument("--vocab-size", type=int, default=16_384,
                   help="BPE vocabulary size (default: 16384) — only used "
                        "when training a new BPE (i.e. --tokenizer-in is "
                        "not given)")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="Cap total tokens (0 = use all of the test split)")
    p.add_argument("--bpe-train-chars", type=int, default=8_000_000,
                   help="Use the first N chars of text for BPE training "
                        "(default: 8M, ~16x more than the BPE merge budget; "
                        "full-text training is unnecessary and slow for the "
                        "~500 MB train split). Set to 0 to use the full text.")
    p.add_argument("--tokenizer-in", type=Path, default=None,
                   help="Reuse a pre-trained tokenizer (tokenizer.json) from "
                        "this path. Skips BPE training entirely. The BPE "
                        "merges are corpus-agnostic (they depend on the "
                        "character-level statistics, which saturate at "
                        "a few MB), so a tokenizer trained on the test "
                        "split works fine for the train split.")
    p.add_argument("--work", type=Path, default=None,
                   help="Scratch directory (default: ${out}/_work)")
    p.add_argument("--zip", type=Path, default=None,
                   help="Path to the WikiText-103 zip (skips download)")
    p.add_argument("--seed", type=int, default=0,
                   help="Reserved for future use (default: 0)")
    args = p.parse_args()

    # Ensure output directory exists up front (so ATTRIBUTION.txt
    # and the shards can be written without a "directory not found"
    # error if BPE training is skipped).
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.work or (args.out / "_work")
    work.mkdir(parents=True, exist_ok=True)
    zip_path = args.zip or (work / "wikitext-103-v1.zip")
    if not zip_path.exists():
        download(WIKITEXT_URL, zip_path)
    raw_text = extract_wikitext_split(zip_path, work, args.split)
    text = normalize_wikitext(raw_text)

    if args.tokenizer_in is not None:
        # Reuse an existing tokenizer — skip BPE training entirely.
        from tokenizers import Tokenizer
        print(f"[tokenizer] reusing {args.tokenizer_in}", file=sys.stderr)
        tok = Tokenizer.from_file(str(args.tokenizer_in))
        tok_json = args.tokenizer_in.read_text(encoding="utf-8")
        tok_sha = sha256_bytes(tok_json.encode("utf-8"))
    else:
        # Train BPE on a manageable subset. The HuggingFace tokenizers
        # BPE trainer is O(N · V) in characters × vocab_size; for the
        # 515 MB train split with a 16k-vocab BPE that takes
        # 5-10 minutes and uses ~2 GB RAM. The BPE merge statistics
        # saturate long before 500 MB of text — 8 MB is more than
        # enough for a 16k vocab. If you don't have an existing
        # tokenizer, use --bpe-train-chars 0 to train on the full
        # text (slow, only required for the very first run).
        bpe_train_text = text
        if args.bpe_train_chars > 0 and len(text) > args.bpe_train_chars:
            bpe_train_text = text[: args.bpe_train_chars]
            print(
                f"[tokenizer] using first {args.bpe_train_chars / 1e6:.1f} MB "
                f"of {len(text) / 1e6:.1f} MB for BPE training "
                f"(pass --bpe-train-chars 0 to use all)",
                file=sys.stderr,
            )

        tok = train_bpe(bpe_train_text, args.vocab_size)
        tok_json = tok.to_str()
        tok_sha = sha256_bytes(tok_json.encode("utf-8"))
        (work / "tokenizer.json").write_text(tok_json, encoding="utf-8")

    ids = tokenize(tok, text)
    if args.max_tokens > 0 and len(ids) > args.max_tokens:
        ids = ids[: args.max_tokens]

    # CC-BY-SA attribution
    (args.out / "ATTRIBUTION.txt").write_text(WIKITEXT_ATTRIBUTION, encoding="utf-8")

    manifest = write_shards(
        ids,
        args.out,
        args.shard_tokens,
        manifest_extra={
            "vocab_size": tok.get_vocab_size(),
            "tokenizer_sha256": tok_sha,
            "tokenizer_path": "tokenizer.json",
            "corpus": f"wikitext-103-{args.split}",
            "license": "CC-BY-SA-4.0",
        },
    )
    print(
        f"[done] {manifest['n_tokens']} tokens, "
        f"{manifest['n_shards']} shards, "
        f"vocab={manifest['vocab_size']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
