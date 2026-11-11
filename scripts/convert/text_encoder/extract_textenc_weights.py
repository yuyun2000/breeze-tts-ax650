#!/usr/bin/env python3
"""Extract the Breeze TTS 2 text-encoder weights from the official HF safetensors.

Breeze checkpoint (BreezeBlue/Breeze-TTS-2) is ~7 GB bf16 split into 2 shards.
The text encoder + projection live in `text_encoder.*` / `text_encoder_proj.*`
(about 2.0 GB bf16, 1.0 B params incl. the 262158x1152 embedding table) and all
of it happens to sit in one shard, so we only read the keys we need with
`safetensors.safe_open` (never materialising the 5 GB backbone/depth/codec).

Outputs (under --out):
  textenc_state.pt                      {"encoder": {key: bf16 tensor}, "proj": bf16 tensor}
  text_embed_262158x1152.bfloat16.bin   raw bf16 embedding table (CPU-side lookup, 604 MB)
  eoi_embedding.npy                     fp32 eoi embedding (1152,) -- NOT scaled by sqrt(hidden)
  embedding_meta.json                   shapes / scale / eoi id for the CPU lookup helper

CPU lookup semantics (mirror of `T5Gemma2TextScaledWordEmbedding.forward`):
  emb = table[token] * sqrt(hidden_size);  if token == eoi_token_index: emb = eoi_embedding
  (padding_idx=0 row is used as-is, and is masked out at the attention level.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path,
                    default=Path("/home/m5stack/work/breeze-tts/data/breeze-tts-2"),
                    help="HF snapshot dir of BreezeBlue/Breeze-TTS-2")
    ap.add_argument("--out", type=Path,
                    default=Path("/home/m5stack/work/breeze-tts/textenc/weights"))
    args = ap.parse_args()

    repo, out = args.repo, args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((repo / "config.json").read_text(encoding="utf-8"))
    te = cfg["text_encoder_config"]

    index = json.loads((repo / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    want = [k for k in weight_map
            if k.startswith("text_encoder.") or k.startswith("text_encoder_proj.")]
    if not want:
        raise SystemExit("no text_encoder keys found in safetensors index")

    by_shard: dict[str, list[str]] = {}
    for key in want:
        by_shard.setdefault(weight_map[key], []).append(key)

    sd: dict[str, torch.Tensor] = {}
    for shard, keys in sorted(by_shard.items()):
        print(f"[extract] reading {len(keys)} tensors from {shard}", flush=True)
        with safe_open(str(repo / shard), framework="pt", device="cpu") as f:
            for key in keys:
                sd[key] = f.get_tensor(key)

    encoder = {k[len("text_encoder."):]: v for k, v in sd.items()
               if k.startswith("text_encoder.")}
    proj = sd["text_encoder_proj.weight"]
    assert proj.shape == (cfg["hidden_size"], te["hidden_size"]), proj.shape
    assert not any(k.startswith("text_encoder_proj.bias") for k in sd), \
        "unexpected text_encoder_proj.bias in checkpoint (Breeze uses bias=False)"

    # bf16 -> raw bf16 bytes (torch has no numpy bf16; store uint16 bit patterns)
    emb = encoder["embed_tokens.weight"]
    assert emb.dtype == torch.bfloat16, emb.dtype
    assert emb.shape == (cfg["text_vocab_size"], te["hidden_size"]), emb.shape
    emb_raw = emb.contiguous().view(torch.uint16).numpy()
    emb_raw.tofile(str(out / f"text_embed_{emb.shape[0]}x{emb.shape[1]}.bfloat16.bin"))
    np.save(out / "eoi_embedding.npy", encoder["embed_tokens.eoi_embedding"].float().numpy())

    # keep the rest as bf16 tensors for the export / parity scripts
    state = {k: v.contiguous() for k, v in encoder.items()}
    torch.save({"encoder": state, "proj": proj.contiguous()},
               str(out / "textenc_state.pt"))

    meta = {
        "source_repo": str(repo),
        "vocab_size": int(cfg["text_vocab_size"]),
        "hidden_size": int(te["hidden_size"]),
        "num_hidden_layers": int(te["num_hidden_layers"]),
        "embed_scale": float(te["hidden_size"]) ** 0.5,
        "eoi_token_index": int(te["eoi_token_index"]),
        "pad_token_id": int(te["pad_token_id"]),
        "embedding_file": f"text_embed_{emb.shape[0]}x{emb.shape[1]}.bfloat16.bin",
        "embedding_dtype": "bfloat16",
        "eoi_embedding_file": "eoi_embedding.npy",
        "out_proj": {"in_features": int(te["hidden_size"]),
                     "out_features": int(cfg["hidden_size"]), "bias": False},
    }
    (out / "embedding_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[extract] OK -> {out}  ({sum(v.numel() for v in state.values()) / 1e6:.1f} M enc params)")


if __name__ == "__main__":
    main()
