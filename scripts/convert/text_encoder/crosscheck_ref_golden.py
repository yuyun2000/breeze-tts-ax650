#!/usr/bin/env python3
"""Cross-check the exported Breeze text-encoder ONNX against ref-2038's Mac golden data.

ref-2038 ran the original Breeze model on macOS (torch 2.15 / transformers 4.57.3) and
saved, for a real prompt, `golden_zh/probe.npz`:
    text_embeds [T, 2048]  = text_encoder(prompt_ids) -> text_encoder_proj   (authoritative)
and `golden_zh/meta.json` with the prompt ids (starting with BOS=2, [S0]=262146, ...).

We rebuild the inputs purely from the *CPU* path (bf16 embedding lookup + sqrt(1152) scale
+ eoi replacement), feed the ONNX chunk chain, and compare the first T rows of the padded
[1,128,2048] output with the golden [T,2048].

Usage:
    python crosscheck_ref_golden.py \
        --golden-dir /home/m5stack/work/breeze-tts/ref/golden_zh \
        --onnx ../onnx/textenc_L128_c1of2.onnx ../onnx/textenc_L128_c2of2.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from text_embed_cpu import BreezeEmbeddingLUT

TEXTENC = Path(__file__).resolve().parent.parent


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-dir", type=Path,
                    default=Path("/home/m5stack/work/breeze-tts/ref/golden_zh"))
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--weights", type=Path, default=TEXTENC / "weights")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--out", type=Path, default=TEXTENC / "verify" / "crosscheck_ref.json")
    args = ap.parse_args()

    import onnxruntime as ort  # noqa: PLC0415

    meta = json.loads((args.golden_dir / "meta.json").read_text(encoding="utf-8"))
    probe = np.load(args.golden_dir / "probe.npz")
    golden = probe["text_embeds"].astype(np.float32)

    prompt_ids = None
    for path in (("prompt", "ids"), ("prompt_ids",), ("ids",), ("token_ids",)):
        node = meta
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, list) and node and isinstance(node[0], int):
            prompt_ids = node
            print(f"[crosscheck] prompt ids from meta{list(path)}")
            break
    if prompt_ids is None:
        # fall back: look for a list of ints that has the same length as golden rows
        for key, value in meta.items():
            if isinstance(value, list) and value and isinstance(value[0], int) \
                    and len(value) == golden.shape[0]:
                prompt_ids = value
                print(f"[crosscheck] prompt ids from meta key '{key}'")
                break
    if prompt_ids is None:
        raise SystemExit("no prompt_ids found in golden meta.json")

    prompt_ids = [int(x) for x in prompt_ids]
    T = len(prompt_ids)
    lut = BreezeEmbeddingLUT(args.weights)
    embeds, mask = lut.build_graph_inputs(prompt_ids, args.seq_len)

    sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for p in args.onnx]
    x = embeds
    for sess in sessions:
        names = [i.name for i in sess.get_inputs()]
        x = sess.run(None, {names[0]: x, names[1]: mask})[0]

    got = x[0, :T]
    if got.shape != golden.shape:
        raise SystemExit(f"shape mismatch: got {got.shape}, golden {golden.shape}")
    cos = cosine(got, golden)
    abs_diff = np.abs(got - golden)
    result = {
        "golden_dir": str(args.golden_dir),
        "onnx": [str(p) for p in args.onnx],
        "n_tokens": T,
        "cosine": cos,
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "golden_abs_mean": float(np.abs(golden).mean()),
        "prompt_prefix_ids": prompt_ids[:6],
        "seq_len_bucket": args.seq_len,
    }
    print(f"[crosscheck] T={T} cosine={cos:.8f} max_abs_diff={abs_diff.max():.3e} "
          f"(golden |x| mean {result['golden_abs_mean']:.4f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[crosscheck] -> {args.out}")
    if cos < 0.999:
        raise SystemExit("crosscheck cosine < 0.999")


if __name__ == "__main__":
    main()
