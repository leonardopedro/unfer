#!/usr/bin/env python3
"""Shared UltraBERT encoder settings for pCNL fixtures and live parity checks.

The fixture generators and the live/fixture parity check import this module so
that a change to the pinned revision, truncation length, pooling or
normalization cannot silently affect only one side.
"""

MODEL = "dmis-lab/UltraBERT-Embed-150M"
REVISION = "b06b36724c5417a1ec1b7863dd4fa542b74e4899"
MAX_LENGTH = 96
BATCH_SIZE = 24


def encode_texts(texts, batch_size=BATCH_SIZE):
    """Return f32 CLS embeddings, L2-normalized, in input order."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True, dtype=torch.float32
    ).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            # The model card declares CLS pooling and L2-normalised output.
            h = model(**enc).last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(h, dim=-1))
    if not chunks:
        raise ValueError("encode_texts needs at least one text")
    return torch.cat(chunks).to(torch.float32)
