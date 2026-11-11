#!/usr/bin/env python3
"""CPU-side Breeze text-encoder embedding lookup -- numpy only (board-friendly).

Mirrors `T5Gemma2TextScaledWordEmbedding.forward` from breeze-tts-src/models/t5gemma2_compat.py:

    emb = weight[token] * sqrt(hidden_size)          # embed_scale
    if token == eoi_token_index: emb = eoi_embedding  # unscaled

The embedding table is kept as raw bf16 bytes on disk and mmap'ed, so a board process
does not need to hold a 1.2 GB fp32 copy in RAM: only the rows actually looked up
(<= seq_len rows) are converted. numpy has no bfloat16 dtype, hence the uint16 ->
uint32 << 16 -> float32 bit trick.

Usage (board / conversion host):
    lut = BreezeEmbeddingLUT("/home/m5stack/work/breeze-tts/textenc/weights")
    embeds = lut.lookup(token_ids)          # [T, 1152] float32
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class BreezeEmbeddingLUT:
    def __init__(self, weights_dir: str | Path, mode: str = "r"):
        self.dir = Path(weights_dir)
        meta = json.loads((self.dir / "embedding_meta.json").read_text(encoding="utf-8"))
        self.vocab_size = int(meta["vocab_size"])
        self.hidden_size = int(meta["hidden_size"])
        self.embed_scale = float(meta["embed_scale"])
        self.eoi_token_index = int(meta["eoi_token_index"])
        self.pad_token_id = int(meta["pad_token_id"])
        raw = np.memmap(self.dir / meta["embedding_file"], dtype=np.uint16,
                        mode=mode, shape=(self.vocab_size, self.hidden_size))
        self._raw = raw
        self._eoi = np.load(self.dir / meta["eoi_embedding_file"]).astype(np.float32)

    @staticmethod
    def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
        """uint16 bf16 bit patterns -> float32 (bf16 keeps float32's top 16 bits)."""
        u32 = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
        return u32.view(np.float32)

    def lookup(self, token_ids) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ValueError(f"token id out of range [0,{self.vocab_size}): "
                             f"{ids.min()}..{ids.max()}")
        rows = np.asarray(self._raw[ids], dtype=np.uint16)  # [T, hidden], gather only needed rows
        embeds = self.bf16_bits_to_float32(rows) * np.float32(self.embed_scale)
        eoi_mask = ids == self.eoi_token_index
        if eoi_mask.any():
            embeds = embeds.copy()
            embeds[eoi_mask] = self._eoi
        return embeds

    def build_graph_inputs(self, token_ids, seq_len: int):
        """Token ids -> (inputs_embeds [1,L,1152] f32, attention_mask [1,L] f32)."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size > seq_len:
            raise ValueError(f"sequence of {ids.size} tokens does not fit bucket {seq_len}")
        embeds = np.zeros((seq_len, self.hidden_size), dtype=np.float32)
        mask = np.zeros((seq_len,), dtype=np.float32)
        real = ids.size
        embeds[:real] = self.lookup(ids)
        mask[:real] = 1.0
        return embeds[None], mask[None]
