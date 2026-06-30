#!/usr/bin/env python3
"""Generate qfm/testdata/cifar10_16x16_*.json from the CIFAR-10 binary dataset.

Downloads cifar-10-binary.tar.gz from the University of Toronto mirror, extracts
the binary batch files, converts each RGB 32x32 image to grayscale 16x16 by
(a) computing the ITU-R 601 luminance, then (b) averaging non-overlapping 2x2
blocks, and saves two JSON fixtures in the same format as mnist_8x8_*.json:

    [{"label": int, "pixels": [float, ...]}, ...]

Output files (written relative to this script's parent = qfm/):
    testdata/cifar10_16x16_256training.json   -- 8 classes x 32 images
    testdata/cifar10_16x16_8heldout.json      -- 8 classes x 1 image

Usage:
    cd unfer/qfm
    python3 scripts/make_cifar10_fixture.py

The script is idempotent: if the tar.gz is already in /tmp it is reused.
"""

import json
import struct
import tarfile
import urllib.request
import io
import os
import sys
import numpy as np

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
TARBALL_CACHE = "/tmp/cifar-10-binary.tar.gz"

# We use 8 of the 10 CIFAR-10 classes (0-7) so that M = 8*32 = 256 = d = 16*16.
# The 8 classes are: airplane(0), automobile(1), bird(2), cat(3),
#                    deer(4), dog(5), frog(6), horse(7).
N_CLASSES = 8
N_TRAIN_PER_CLASS = 32   # 8*32 = 256 training images
N_HELDOUT_PER_CLASS = 1  # 8*1  = 8 held-out images
IMG_D = 16 * 16           # 256 pixels after downsampling

# CIFAR-10 binary format: each record is (label_byte, 3072 bytes of RGB)
RECORD_BYTES = 1 + 3 * 32 * 32

# Training source batch (data_batch_1.bin is always present early in the tarball).
TRAIN_BATCH = "data_batch_1.bin"
# Held-out source: use test_batch.bin if available (the canonical split), else
# fall back to data_batch_3.bin (a different training split — valid for our tests
# since we only need out-of-distribution images, not the official evaluation split).
HELDOUT_BATCH_PREFERRED = "test_batch.bin"
HELDOUT_BATCH_FALLBACK = "data_batch_3.bin"


def download_tarball():
    if os.path.exists(TARBALL_CACHE):
        print(f"Reusing cached {TARBALL_CACHE}", flush=True)
        return
    print(f"Downloading CIFAR-10 binary from {CIFAR_URL} ...", flush=True)
    print("  (this is ~170 MB; may take a few minutes)", flush=True)
    urllib.request.urlretrieve(CIFAR_URL, TARBALL_CACHE)
    print(f"  Saved to {TARBALL_CACHE}", flush=True)


def read_batch(tarball_path: str, batch_name: str):
    """Return list of (label, gray_16x16) from a CIFAR-10 binary batch.

    Uses streaming mode so partial archives (in-progress downloads) work as
    long as the requested batch file is fully present in the downloaded bytes.
    """
    member_path = f"cifar-10-batches-bin/{batch_name}"
    raw = None
    # Streaming mode: iterates members without seeking (works on partial downloads).
    with tarfile.open(tarball_path, "r|gz") as tar:
        for member in tar:
            if member.name == member_path:
                f = tar.extractfile(member)
                if f is None:
                    raise FileNotFoundError(f"{member_path} found but not extractable")
                raw = f.read()
                break
    if raw is None:
        raise FileNotFoundError(f"{member_path} not found in tarball")

    n_images = len(raw) // RECORD_BYTES
    records = []
    offset = 0
    for _ in range(n_images):
        label = raw[offset]
        pixels = np.frombuffer(raw, dtype=np.uint8,
                                count=3 * 32 * 32, offset=offset + 1)
        offset += RECORD_BYTES

        # Reshape: (3, 32, 32) in R-G-B order.
        rgb = pixels.reshape(3, 32, 32).astype(np.float64) / 255.0
        r, g, b = rgb[0], rgb[1], rgb[2]

        # ITU-R 601 grayscale.
        gray32 = 0.299 * r + 0.587 * g + 0.114 * b  # (32, 32)

        # Downsample 32x32 -> 16x16 by averaging 2x2 blocks.
        gray16 = gray32.reshape(16, 2, 16, 2).mean(axis=(1, 3))  # (16, 16)

        records.append((label, gray16.flatten().tolist()))
    return records


def select_subset(records, n_classes, n_per_class):
    """Pick up to n_per_class records from each of the first n_classes."""
    buckets = {c: [] for c in range(n_classes)}
    for label, pixels in records:
        if label < n_classes and len(buckets[label]) < n_per_class:
            buckets[label].append(pixels)
    result = []
    for c in range(n_classes):
        for pixels in buckets[c]:
            result.append({"label": c, "pixels": pixels})
    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Script lives in qfm/scripts/; output goes in qfm/testdata/.
    testdata_dir = os.path.join(script_dir, "..", "testdata")
    os.makedirs(testdata_dir, exist_ok=True)

    download_tarball()

    # --- Training fixture: data_batch_1.bin ---
    print(f"Reading {TRAIN_BATCH} ...", flush=True)
    train_records = read_batch(TARBALL_CACHE, TRAIN_BATCH)
    print(f"  {len(train_records)} images read", flush=True)

    training = select_subset(train_records, N_CLASSES, N_TRAIN_PER_CLASS)
    assert len(training) == N_CLASSES * N_TRAIN_PER_CLASS, \
        f"Expected {N_CLASSES * N_TRAIN_PER_CLASS} training images, got {len(training)}"
    assert len(training[0]["pixels"]) == IMG_D, \
        f"Expected {IMG_D} pixels per image, got {len(training[0]['pixels'])}"

    train_path = os.path.join(testdata_dir, "cifar10_16x16_256training.json")
    with open(train_path, "w") as f:
        json.dump(training, f, separators=(",", ":"))
    print(f"Wrote {len(training)} training images to {train_path}", flush=True)

    # --- Held-out fixture: prefer test_batch.bin; fall back to data_batch_3.bin ---
    # test_batch.bin is the canonical held-out split, but data_batch_3.bin is also
    # acceptable for the purpose of testing out-of-training-batch generalization.
    for heldout_batch in (HELDOUT_BATCH_PREFERRED, HELDOUT_BATCH_FALLBACK):
        try:
            print(f"Reading {heldout_batch} ...", flush=True)
            test_records = read_batch(TARBALL_CACHE, heldout_batch)
            print(f"  {len(test_records)} images read", flush=True)
            break
        except Exception as e:
            print(f"  {heldout_batch} not available ({e}); trying fallback ...", flush=True)
    else:
        raise RuntimeError("Neither test_batch.bin nor data_batch_3.bin is available in the archive")

    heldout = select_subset(test_records, N_CLASSES, N_HELDOUT_PER_CLASS)
    assert len(heldout) == N_CLASSES * N_HELDOUT_PER_CLASS, \
        f"Expected {N_CLASSES} held-out images, got {len(heldout)}"

    heldout_path = os.path.join(testdata_dir, "cifar10_16x16_8heldout.json")
    with open(heldout_path, "w") as f:
        json.dump(heldout, f, separators=(",", ":"))
    print(f"Wrote {len(heldout)} held-out images to {heldout_path}", flush=True)

    # --- Verify ---
    labels_train = sorted(set(s["label"] for s in training))
    labels_heldout = [s["label"] for s in heldout]
    print(f"\nTraining labels: {labels_train} ({len(training)} total)")
    print(f"Held-out labels: {labels_heldout} ({len(heldout)} total)")
    print(f"Pixel range (train[0]): [{min(training[0]['pixels']):.3f}, {max(training[0]['pixels']):.3f}]")
    print("\nDone.")


if __name__ == "__main__":
    main()
