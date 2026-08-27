#!/usr/bin/env bash
# Detached resumable downloader for the Lean 4.31.0 toolchain.
# Run with: setsid nohup bash /home/leo/Projects/unfer/.toolchain_dl.sh > /dev/null 2>&1 &
set -u
URL="https://github.com/leanprover/lean4/releases/download/v4.31.0/lean-4.31.0-linux.tar.zst"
OUT="/home/leo/Projects/.toolchain/lean31.tar.zst"
TARGET=558112470
for i in $(seq 1 60); do
  CUR=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$CUR" -ge "$TARGET" ]; then
    echo "COMPLETE size=$CUR" >> /home/leo/Projects/.toolchain/lean31_chunks.log
    break
  fi
  timeout 280 curl -sL --retry 3 --retry-all-errors -C - -o "$OUT" "$URL" 2>/dev/null
  RC=$?
  CUR=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  echo "iter=$i rc=$RC size=$CUR" >> /home/leo/Projects/.toolchain/lean31_chunks.log
  if [ "$CUR" -ge "$TARGET" ]; then
    echo "COMPLETE size=$CUR" >> /home/leo/Projects/.toolchain/lean31_chunks.log
    break
  fi
  sleep 2
done
echo "FINAL size=$(stat -c %s "$OUT" 2>/dev/null || echo 0)" >> /home/leo/Projects/.toolchain/lean31_chunks.log
echo DONE > /home/leo/Projects/.toolchain/lean31.status
