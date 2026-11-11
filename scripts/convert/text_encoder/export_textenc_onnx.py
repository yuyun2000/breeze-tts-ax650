#!/usr/bin/env python3
"""Breeze TTS 2 text encoder -> static ONNX (input for Pulsar2 / AX650).

Graph
-----
    inputs_embeds   float32 [1, L, 1152]   CPU-side embedding lookup (see extract_textenc_weights.py)
    attention_mask  float32 [1, L]         1 = real token, 0 = padding
  -> T5Gemma2TextEncoder (26 bidirectional layers, per-layer-type RoPE, RMSNorm (1+w), GELU-tanh MLP)
  -> text_encoder_proj (Linear 1152 -> 2048, no bias)
    text_embeds     float32 [1, L, 2048]

Faithfulness notes (see breeze-tts-src/models/t5gemma2_compat.py)
---------------------------------------------------------------
* embed scale sqrt(1152) and eoi_token_index=256000 -> eoi_embedding are handled in the
  CPU embedding lookup, i.e. outside of this graph (same as Breeze, which replaces the
  embedding *after* scaling).
* Attn scaling is `query_pre_attn_scalar ** -0.5` = 1/16 (not head_dim**-0.5); q_norm/k_norm
  are RMSNorm over head_dim=256 with (1+w) and eps=1e-6; no attention softcap.
* RoPE: sliding layers use theta=1e4/default, full layers use theta=1e6 + linear factor 8
  (layer_types: every 6th layer = full_attention). Both variants are kept per layer.
* Bidirectional attention, no causal mask. sliding_window=512 does not bind for L<=256, so
  the graph only carries the padding mask (asserted below); the equivalence is exact for
  L<=256 ((512+1)//2 = 256 left, 512//2+1 = 257 right).
* PAD_BIAS = -60 replaces -inf/finfo.min so INT8 calibration scales stay sane; after softmax
  exp(-60)~8.8e-27, which is numerically identical to a hard mask.

Modes
-----
  --layers 0:26           export those transformer layers (default: all)
  --layers 13:26 --proj   second half of a split export (with output projection)
  --no-final-norm         drop the trailing RMSNorm (for non-final chunks)
  --export-params false   expose weights as graph inputs instead of initializers
                          (needed when the ONNX would exceed the 2 GB protobuf limit;
                          used by split/half workflows)

The ONNX is exported with the TorchScript exporter, opset 17, fp32 throughout.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent          # .../textenc/scripts
TEXTENC = HERE.parent                           # .../textenc
BREEZE_ROOT = TEXTENC.parent                    # .../breeze-tts
DEFAULT_SRC = BREEZE_ROOT / "data" / "breeze-tts-src"
DEFAULT_REPO = BREEZE_ROOT / "data" / "breeze-tts-2"
DEFAULT_WEIGHTS = TEXTENC / "weights" / "textenc_state.pt"
DEFAULT_OUT = TEXTENC / "onnx"

# pad-position attention bias (see module docstring)
PAD_BIAS = -60.0


def load_text_config(repo: Path) -> dict:
    cfg = json.loads((repo / "config.json").read_text(encoding="utf-8"))
    return cfg["text_encoder_config"]


def patch_padding_only_mask(src_root: Path) -> None:
    """Replace the shim's additive-mask builder with a padding-only, bounded bias.

    * sliding window 512 >= L is asserted by the caller, so no local-window mask is needed;
    * finfo(dtype).min -> PAD_BIAS keeps the graph quantization friendly.
    """
    sys.path.insert(0, str(src_root))
    from models import t5gemma2_compat as compat

    def build(attention_mask, inputs_embeds, layer_type, sliding_window):
        if attention_mask is None:
            return None
        bias = (1.0 - attention_mask.to(inputs_embeds.dtype)) * PAD_BIAS
        return bias[:, None, None, :]  # [B,1,1,L] broadcasts over heads & query positions

    compat.T5Gemma2TextEncoder._build_additive_attention_mask = staticmethod(build)


class EncoderChunkGraph(nn.Module):
    """Runs the encoder (optionally without final norm) and optionally the output proj."""

    def __init__(self, encoder, proj: nn.Linear | None, seq_len: int):
        super().__init__()
        self.encoder = encoder
        self.proj = proj
        self.register_buffer(
            "position_ids",
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=self.position_ids,
            output_hidden_states=False,
        ).last_hidden_state
        if self.proj is not None:
            hidden = self.proj(hidden)
        return hidden


def build_chunk(te_cfg: dict, state: dict, start: int, end: int,
                with_proj: bool, with_final_norm: bool, src_root: Path,
                out_hidden: int):
    sys.path.insert(0, str(src_root))
    from models.t5gemma2_compat import T5Gemma2TextConfig, T5Gemma2TextEncoder

    sig = inspect.signature(T5Gemma2TextConfig.__init__)
    allowed = {p for p in sig.parameters if p != "self"}
    kwargs = {k: v for k, v in te_cfg.items() if k in allowed}
    kwargs["num_hidden_layers"] = end - start
    kwargs["layer_types"] = list(te_cfg["layer_types"][start:end])

    config = T5Gemma2TextConfig(**kwargs)
    config._attn_implementation = "eager"  # torch matmul/softmax path, export friendly
    encoder = T5Gemma2TextEncoder(config).to(torch.float32)
    encoder.eval()

    # remap absolute layer indices -> chunk-local indices
    enc_state = {}
    for key, value in state["encoder"].items():
        if key.startswith("layers."):
            _, idx, *rest = key.split(".")
            idx = int(idx)
            if not (start <= idx < end):
                continue
            key = ".".join(["layers", str(idx - start), *rest])
        enc_state[key] = value
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        raise SystemExit(f"missing weights for chunk {start}:{end}: {missing[:5]}")
    if unexpected:
        raise SystemExit(f"unexpected weights for chunk {start}:{end}: {unexpected[:5]}")

    if not with_final_norm:
        encoder.norm = nn.Identity()
    # The graph consumes inputs_embeds, so the 262158x1152 embedding table must not be
    # dragged into the ONNX (it would add ~1.2 GB fp32 for nothing).
    encoder.embed_tokens = nn.Identity()

    proj = None
    if with_proj:
        proj = nn.Linear(int(te_cfg["hidden_size"]), int(out_hidden), bias=False)
        proj.weight.data.copy_(state["proj"].to(torch.float32))
        proj.eval()

    return encoder, proj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--layers", default="0:26", help="layer range START:END (exclusive)")
    ap.add_argument("--proj", action="store_true", help="append text_encoder_proj (1152->2048)")
    ap.add_argument("--no-final-norm", action="store_true")
    ap.add_argument("--export-params", dest="export_params", default="true",
                    choices=["true", "false"],
                    help="false => weights become graph inputs (ONNX stays small)")
    ap.add_argument("--tag", default=None, help="output file tag, default derived from args")
    args = ap.parse_args()

    start, end = (int(x) for x in args.layers.split(":"))
    if not (0 <= start < end <= 26):
        raise SystemExit(f"bad --layers {args.layers}")
    if args.seq_len > 256:
        raise SystemExit("seq-len > 256: sliding_window=512 would bind, padding-only mask is invalid")

    te_cfg = load_text_config(args.repo)
    full_cfg = json.loads((args.repo / "config.json").read_text(encoding="utf-8"))
    out_hidden = int(full_cfg["hidden_size"])  # 2048 (Breeze talker hidden size)
    state = torch.load(str(args.weights), map_location="cpu", weights_only=True)
    patch_padding_only_mask(args.src)
    encoder, proj = build_chunk(te_cfg, state, start, end,
                                with_proj=args.proj,
                                with_final_norm=not args.no_final_norm,
                                src_root=args.src,
                                out_hidden=out_hidden)

    graph = EncoderChunkGraph(encoder, proj, args.seq_len).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (f"textenc_L{args.seq_len}"
                       + (f"_layers{start}-{end}" if (start, end) != (0, 26) else "")
                       + ("" if args.proj else "_noproj"))
    onnx_path = args.out / f"{tag}.onnx"

    embeds = torch.zeros(1, args.seq_len, int(te_cfg["hidden_size"]), dtype=torch.float32)
    mask = torch.ones(1, args.seq_len, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            graph,
            (embeds, mask),
            str(onnx_path),
            input_names=["inputs_embeds", "attention_mask"],
            output_names=["text_embeds"],
            opset_version=17,
            do_constant_folding=True,
            export_params=(args.export_params == "true"),
            dynamo=False,
        )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"[export] {onnx_path}  ({size_mb:.1f} MB)  layers={start}:{end} proj={args.proj} "
          f"final_norm={not args.no_final_norm} export_params={args.export_params}")

    meta = {
        "onnx": str(onnx_path),
        "size_mb": round(size_mb, 1),
        "seq_len": args.seq_len,
        "layers": [start, end],
        "with_proj": bool(args.proj),
        "with_final_norm": not args.no_final_norm,
        "export_params": args.export_params,
        "inputs": {"inputs_embeds": [1, args.seq_len, int(te_cfg["hidden_size"])],
                   "attention_mask": [1, args.seq_len]},
        "outputs": {"text_embeds": [1, args.seq_len,
                                    out_hidden if args.proj else int(te_cfg["hidden_size"])]},
        "pad_bias": PAD_BIAS,
    }
    (args.out / f"{tag}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
