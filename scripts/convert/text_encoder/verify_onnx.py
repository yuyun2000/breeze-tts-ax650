#!/usr/bin/env python3
"""Verify exported Breeze text-encoder ONNX against the PyTorch reference.

Reference = breeze-tts-src/models/t5gemma2_compat.py (the vendored T5Gemma2 encoder the
Breeze checkpoint was trained with), fp32, eager attention, real -inf padding mask.
ONNX = the graph produced by export_textenc_onnx.py (bounded PAD_BIAS=-60 mask).

Checks
  * single ONNX:  ONNX(inputs_embeds, attention_mask) ~= proj(encoder(embeds, mask))
  * chunked ONNX: intermediate hidden after chunk1 ~= reference hidden_states[k], and the
                  chained output ~= proj(encoder(...))
  * cosine over all positions and over real tokens only (padding outputs are discarded
    on the board, so the latter is the acceptance-relevant number)

Usage:
    python verify_onnx.py --onnx onnx/textenc_L128.onnx --calib calib/val/val.npz
    python verify_onnx.py --onnx onnx/half1.onnx onnx/half2.onnx --calib ...
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent          # .../textenc/scripts
TEXTENC = HERE.parent                           # .../textenc
BREEZE_ROOT = TEXTENC.parent                    # .../breeze-tts
DEFAULT_SRC = BREEZE_ROOT / "data" / "breeze-tts-src"
DEFAULT_REPO = BREEZE_ROOT / "data" / "breeze-tts-2"
DEFAULT_WEIGHTS = TEXTENC / "weights" / "textenc_state.pt"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def load_reference(src: Path, repo: Path, weights: Path, num_layers: int | None = None):
    src = src.resolve()
    sys.path.insert(0, str(src))
    from models.t5gemma2_compat import T5Gemma2TextConfig, T5Gemma2TextEncoder

    full_cfg = json.loads((repo / "config.json").read_text(encoding="utf-8"))
    te = full_cfg["text_encoder_config"]
    sig = inspect.signature(T5Gemma2TextConfig.__init__)
    allowed = {p for p in sig.parameters if p != "self"}
    kwargs = {k: v for k, v in te.items() if k in allowed}
    if num_layers is not None:
        kwargs["num_hidden_layers"] = num_layers
        kwargs["layer_types"] = list(te["layer_types"][:num_layers])
    cfg = T5Gemma2TextConfig(**kwargs)
    cfg._attn_implementation = "eager"  # deterministic torch matmul/softmax path
    encoder = T5Gemma2TextEncoder(cfg).to(torch.float32).eval()

    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    enc_state = state["encoder"]
    if num_layers is not None:
        enc_state = {k: v for k, v in enc_state.items()
                     if not (k.startswith("layers.")
                             and int(k.split(".")[1]) >= num_layers)}
    encoder.load_state_dict(enc_state, strict=True)
    proj = nn.Linear(int(te["hidden_size"]), int(full_cfg["hidden_size"]), bias=False)
    proj.weight.data.copy_(state["proj"].to(torch.float32))
    proj.eval()
    return encoder, proj, int(te["hidden_size"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", required=True,
                    help="one full-model ONNX or the chunk ONNX files in execution order")
    ap.add_argument("--calib", type=Path, default=TEXTENC / "calib" / "val" / "val.npz")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--chunk-layers", default=None,
                    help="comma separated layer ranges of the chunks, e.g. 0:13,13:26")
    ap.add_argument("--out", type=Path, default=None, help="metrics json path")
    args = ap.parse_args()

    import onnxruntime as ort  # noqa: PLC0415

    val = np.load(args.calib)
    embeds_all = val["inputs_embeds"]
    mask_all = val["attention_mask"]
    lengths = val["lengths"]

    sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for p in args.onnx]
    input_names = [s.get_inputs()[0].name for s in sessions]
    mask_name = sessions[0].get_inputs()[1].name

    encoder, proj, hidden_size = load_reference(args.src, args.repo, args.weights)

    chunks = None
    if args.chunk_layers:
        chunks = [tuple(int(x) for x in part.split(":"))
                  for part in args.chunk_layers.split(",")]
        if len(chunks) != len(sessions):
            raise SystemExit(f"{len(chunks)} chunk ranges vs {len(sessions)} ONNX files")

    rows = []
    for i in range(embeds_all.shape[0]):
        embeds = torch.from_numpy(embeds_all[i:i + 1].copy())
        mask = torch.from_numpy(mask_all[i:i + 1].copy())
        n_real = int(lengths[i])
        with torch.inference_mode():
            t0 = time.time()
            ref = encoder(inputs_embeds=embeds, attention_mask=mask,
                          position_ids=torch.arange(embeds.shape[1]).unsqueeze(0),
                          output_hidden_states=bool(chunks)).last_hidden_state
            ref_full = proj(ref)
            torch_time = time.time() - t0

        # run the ONNX chain
        t0 = time.time()
        feed = {input_names[0]: embeds_all[i:i + 1].copy(), mask_name: mask_all[i:i + 1].copy()}
        chunk_errors = []
        if chunks is None or len(sessions) == 1:
            x = sessions[0].run(None, feed)[0]
        else:
            with torch.inference_mode():
                hs = encoder(inputs_embeds=embeds, attention_mask=mask,
                             position_ids=torch.arange(embeds.shape[1]).unsqueeze(0),
                             output_hidden_states=True).hidden_states
            x = feed[input_names[0]]
            for j, sess in enumerate(sessions):
                names = [sess.get_inputs()[0].name, sess.get_inputs()[1].name]
                x = sess.run(None, {names[0]: x, names[1]: feed[mask_name]})[0]
                if j < len(sessions) - 1:
                    start, end = chunks[j]
                    # all_hidden_states[0] = input embeds, [end] = after layer end-1
                    chunk_errors.append({"chunk": f"{start}:{end}",
                                         "cosine": cosine(x, hs[end].numpy())})
                    print(f"[verify]   chunk {start}:{end} cosine={chunk_errors[-1]['cosine']:.6f}")
        onnx_time = time.time() - t0

        out = x
        ref_np = ref_full.numpy()
        row = {
            "sample": i,
            "n_tokens": n_real,
            "cosine_all_positions": cosine(out, ref_np),
            "cosine_real_tokens_only": cosine(out[:, :n_real], ref_np[:, :n_real]),
            "max_abs_diff_real_tokens": float(np.max(np.abs(out[:, :n_real] - ref_np[:, :n_real]))),
            "torch_forward_s": round(torch_time, 2),
            "onnx_forward_s": round(onnx_time, 2),
        }
        if chunk_errors:
            row["chunk_cosines"] = chunk_errors
            row["min_chunk_cosine"] = min(c["cosine"] for c in chunk_errors)
        rows.append(row)
        print(f"[verify] sample {i} tokens={n_real} "
              f"cos_all={row['cosine_all_positions']:.6f} "
              f"cos_real={row['cosine_real_tokens_only']:.6f}")

    summary = {
        "onnx": [str(p) for p in args.onnx],
        "calib": str(args.calib),
        "chunks": args.chunk_layers,
        "mean_cosine_real_tokens": float(np.mean([r["cosine_real_tokens_only"] for r in rows])),
        "min_cosine_real_tokens": float(np.min([r["cosine_real_tokens_only"] for r in rows])),
        "mean_cosine_all_positions": float(np.mean([r["cosine_all_positions"] for r in rows])),
        "rows": rows,
    }
    print(f"[verify] mean cosine (real tokens) = {summary['mean_cosine_real_tokens']:.6f}  "
          f"min = {summary['min_cosine_real_tokens']:.6f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[verify] metrics -> {args.out}")
    if summary["mean_cosine_real_tokens"] < 0.999:
        raise SystemExit("ONNX vs PyTorch cosine below 0.999")


if __name__ == "__main__":
    main()
