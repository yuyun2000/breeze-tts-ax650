#!/usr/bin/env python3
"""Repack Breeze TTS 2 weights into standard HF-format directories that
`pulsar2 llm_build` can compile with the generic model types.

Mapping (verified against breeze-tts-src/models/breeze.py +
breeze_backbone_factory.py, 2026-11-11):

  talker  : Breeze backbone is a Qwen3Adapter over `backbone_config`
            (Qwen3, hidden 2048, 28 layers, heads 16/8, head_dim 128,
             inter 6144, rope_theta 1e6, NO rope scaling, eps 1e-6,
             per-head q/k RMSNorm) -> Qwen3ForCausalLM layout, compiled with
            `pulsar2 llm_build --model_type qwen3`.
  cp      : Breeze depth decoder is a Llama-style 12-layer model
            (hidden 1024, heads 8/2, head_dim 128, inter 8192, eps 1e-5,
             rope_theta 5e5 + llama3 scaling factor 32 / orig 16)
            -> LlamaForCausalLM layout (no q_norm/k_norm), compiled with
            `pulsar2 llm_build --model_type llama`.

Runtime side artefacts (board python runtime, no torch):
  talker/audio_embed_16x2051x2048.bf16.bin     16 audio codebook embedding groups
  cp/codec_embed_proj_15x2051x1024.bf16.bin    E'_i = E_i @ proj.T  (i = 0..14)
  cp/inputs_embeds_projector_1024x2048.bf16.bin  proj matrix (backbone hidden -> cp hidden)
  cp/codebooks_head_15x1024x2051.bf16.bin      raw position-dependent heads (reference)
  lm_head/code_predictor_lm_head_{0..14}.onnx  weight = codebooks_head[i].T ([2051,1024])

Usage:
  repack_breeze.py --model-dir MODEL_DIR --out-dir OUT_DIR [--layers 28] [--skip-embeds] [--skip-onnx]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Breeze constants (from ref/model-breeze-tts-2/config.json)
# ---------------------------------------------------------------------------
BACKBONE = dict(
    hidden_size=2048,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=6144,
    rope_theta=1000000.0,       # backbone_config.rope_theta
    rms_norm_eps=1e-6,          # backbone_config.rms_norm_eps
    num_codebooks=16,
    audio_vocab=2051,
    lm_vocab=2052,              # 2051 audio codes + 1 backbone EOS
)
DEPTH = dict(
    hidden_size=1024,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=128,
    intermediate_size=8192,
    rope_theta=500000.0,
    rms_norm_eps=1e-5,
    max_position_embeddings=33,
    audio_embed_size=2048,
)
DEPTH_ROPE_SCALING = {
    "factor": 32.0,
    "high_freq_factor": 0.0078125,
    "low_freq_factor": 0.001953125,
    "original_max_position_embeddings": 16,
    "rope_type": "llama3",
    "type": "llama3",
}

TALKER_LAYER_KEYS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]
CP_LAYER_KEYS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def log(msg):
    print(f"[repack] {msg}", flush=True)


class Shards:
    """Lazy per-tensor reader over a sharded safetensors checkpoint."""

    def __init__(self, model_dir: Path):
        self.root = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            self.weight_map = json.loads(index_path.read_text())["weight_map"]
            self.single = None
        else:
            single = model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no model.safetensors[.index.json] in {model_dir}")
            self.weight_map = None
            self.single = single
        self._handles = {}

    def _handle(self, filename):
        if filename not in self._handles:
            self._handles[filename] = safe_open(str(self.root / filename), "pt", device="cpu")
        return self._handles[filename]

    def has(self, key):
        return self.single is not None or key in self.weight_map

    def get(self, key):
        if self.single is not None:
            return self._handle(self.single.name).get_tensor(key)
        return self._handle(self.weight_map[key]).get_tensor(key)

    def close(self):
        for h in self._handles.values():
            del h
        self._handles.clear()


def _contig(state):
    return {k: v.contiguous() for k, v in state.items()}


def write_model_dir(out_dir: Path, config: dict, state: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"writing {out_dir}/model.safetensors ({len(state)} tensors)")
    save_file(_contig(state), str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def talker_config(num_layers: int, max_position_embeddings: int) -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": BACKBONE["lm_vocab"] - 1,
        "eos_token_id": BACKBONE["lm_vocab"] - 1,
        "head_dim": BACKBONE["head_dim"],
        "hidden_act": "silu",
        "hidden_size": BACKBONE["hidden_size"],
        "initializer_range": 0.02,
        "intermediate_size": BACKBONE["intermediate_size"],
        "max_position_embeddings": max_position_embeddings,
        "mlp_bias": False,
        "model_type": "qwen3",
        "num_attention_heads": BACKBONE["num_attention_heads"],
        "num_hidden_layers": num_layers,
        "num_key_value_heads": BACKBONE["num_key_value_heads"],
        "rms_norm_eps": BACKBONE["rms_norm_eps"],
        "rope_scaling": None,
        "rope_theta": BACKBONE["rope_theta"],
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": BACKBONE["lm_vocab"],
        # pulsar2 `qwen3` CfgLoader copies fields that exist in llm_utils.Config:
        "use_qk_norm": True,
        "qk_norm_after_rope": False,
        "layer_types": ["full_attention"] * num_layers,
    }


def cp_config(num_layers: int, max_position_embeddings: int, vocab_size: int) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "head_dim": DEPTH["head_dim"],
        "hidden_act": "silu",
        "hidden_size": DEPTH["hidden_size"],
        "initializer_range": 0.02,
        "intermediate_size": DEPTH["intermediate_size"],
        "max_position_embeddings": max_position_embeddings,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": DEPTH["num_attention_heads"],
        "num_hidden_layers": num_layers,
        "num_key_value_heads": DEPTH["num_key_value_heads"],
        "rms_norm_eps": DEPTH["rms_norm_eps"],
        "rope_scaling": dict(DEPTH_ROPE_SCALING),
        "rope_theta": DEPTH["rope_theta"],
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": vocab_size,
        # generic Config fields pulsar2 maps from config.json
        "rs_rope_type": "llama3",
        "rs_factor": DEPTH_ROPE_SCALING["factor"],
        "rs_high_freq_factor": DEPTH_ROPE_SCALING["high_freq_factor"],
        "rs_low_freq_factor": DEPTH_ROPE_SCALING["low_freq_factor"],
        "rs_original_max_position_embeddings": DEPTH_ROPE_SCALING["original_max_position_embeddings"],
    }


def repack_talker(shards: Shards, out_dir: Path, num_layers: int, max_position_embeddings: int):
    state = {}
    for i in range(num_layers):
        src = f"backbone_model.layers.{i}."
        dst = f"model.layers.{i}."
        for k in TALKER_LAYER_KEYS:
            state[dst + k] = shards.get(src + k)
    state["model.norm.weight"] = shards.get("backbone_model.norm.weight")
    state["lm_head.weight"] = shards.get("lm_head.weight")  # [2052, 2048]
    # dummy, never used by llm_build (per-layer graphs take embeddings)
    state["model.embed_tokens.weight"] = torch.zeros(
        (BACKBONE["lm_vocab"], BACKBONE["hidden_size"]), dtype=torch.bfloat16
    )
    write_model_dir(out_dir, talker_config(num_layers, max_position_embeddings), state)
    log(f"talker done: {sum(v.numel() for v in state.values()) / 1e6:.1f}M params")


def repack_cp(shards, out_dir: Path, num_layers: int, max_position_embeddings: int, vocab_size: int):
    state = {}
    for i in range(num_layers):
        src = f"depth_decoder.model.layers.{i}."
        dst = f"model.layers.{i}."
        for k in CP_LAYER_KEYS:
            if not shards.has(src + k):
                raise KeyError(f"missing depth key {src + k}")
            state[dst + k] = shards.get(src + k)
    state["model.norm.weight"] = shards.get("depth_decoder.model.norm.weight")
    # dummy head / embedding: post layer builds them but runtime uses the 15
    # position-dependent lm_head axmodels + projected embedding bins instead.
    state["lm_head.weight"] = torch.zeros((vocab_size, DEPTH["hidden_size"]), dtype=torch.bfloat16)
    state["model.embed_tokens.weight"] = torch.zeros((vocab_size, DEPTH["hidden_size"]), dtype=torch.bfloat16)
    write_model_dir(out_dir, cp_config(num_layers, max_position_embeddings, vocab_size), state)
    log(f"cp done: {sum(v.numel() for v in state.values()) / 1e6:.1f}M params")


def _bf16_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bf16 bytes (numpy has no bf16 dtype)."""
    return t.contiguous().view(torch.uint16).numpy().tobytes()


def export_runtime_artifacts(shards: Shards, embeds_dir: Path, onnx_dir: Path | None):
    embeds_dir.mkdir(parents=True, exist_ok=True)
    n_cb, vocab, aemb = BACKBONE["num_codebooks"], BACKBONE["audio_vocab"], DEPTH["audio_embed_size"]
    table = shards.get("depth_decoder.model.embed_tokens.weight")  # [16*2051, 2048]
    assert table.shape == (n_cb * vocab, aemb), table.shape
    groups = table.reshape(n_cb, vocab, aemb).contiguous()
    (embeds_dir / "audio_embed_16x2051x2048.bf16.bin").write_bytes(_bf16_bytes(groups))
    log("wrote talker/audio_embed_16x2051x2048.bf16.bin")

    proj = shards.get("depth_decoder.model.inputs_embeds_projector.weight")  # [1024, 2048]
    assert proj.shape == (DEPTH["hidden_size"], aemb), proj.shape
    proj_bf16 = proj.to(torch.bfloat16).contiguous()
    (embeds_dir / "inputs_embeds_projector_1024x2048.bf16.bin").write_bytes(_bf16_bytes(proj_bf16))
    log("wrote cp/inputs_embeds_projector_1024x2048.bf16.bin")

    proj_f32 = proj.float()
    e_proj = torch.stack([groups[i].float() @ proj_f32.t() for i in range(n_cb - 1)])  # [15,2051,1024]
    e_proj = e_proj.to(torch.bfloat16).contiguous()
    (embeds_dir / "codec_embed_proj_15x2051x1024.bf16.bin").write_bytes(_bf16_bytes(e_proj))
    del e_proj
    log("wrote cp/codec_embed_proj_15x2051x1024.bf16.bin")

    heads = shards.get("depth_decoder.codebooks_head.weight")  # [15, 1024, 2051]
    assert heads.shape == (n_cb - 1, DEPTH["hidden_size"], vocab), heads.shape
    heads_bf16 = heads.to(torch.bfloat16).contiguous()
    (embeds_dir / "codebooks_head_15x1024x2051.bf16.bin").write_bytes(_bf16_bytes(heads_bf16))
    log("wrote cp/codebooks_head_15x1024x2051.bf16.bin")

    if onnx_dir is not None:
        export_lm_head_onnx(heads, onnx_dir)


def export_lm_head_onnx(heads, onnx_dir: Path):
    onnx_dir.mkdir(parents=True, exist_ok=True)
    n_heads, hidden, vocab = heads.shape
    with torch.no_grad():
        for i in range(n_heads):
            w = heads[i].t().contiguous().float()  # Breeze: linear(x, weight[i].T) => [vocab, hidden]
            lin = torch.nn.Linear(hidden, vocab, bias=False)
            lin.weight.copy_(w)
            lin.eval()
            dummy = torch.randn(1, 1, hidden, dtype=torch.float32)
            out = onnx_dir / f"code_predictor_lm_head_{i}.onnx"
            torch.onnx.export(
                lin,
                (dummy,),
                str(out),
                input_names=["input"],
                output_names=["output"],
                opset_version=16,
                dynamo=False,
            )
        log(f"wrote {n_heads} lm_head onnx -> {onnx_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Breeze HF model dir (config.json + safetensors)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", type=int, default=0, help="limit layers (tiny build smoke test)")
    ap.add_argument("--talker-max-pos", type=int, default=2048)
    ap.add_argument("--cp-max-pos", type=int, default=128)
    ap.add_argument("--cp-vocab", type=int, default=2051)
    ap.add_argument("--skip-embeds", action="store_true")
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--only-embeds", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    shards = Shards(model_dir)
    n_talker = args.layers or BACKBONE["num_hidden_layers"]
    n_cp = args.layers or DEPTH["num_hidden_layers"]
    if not args.only_embeds:
        repack_talker(shards, out_dir / "breeze-tts-talker", n_talker, args.talker_max_pos)
        repack_cp(shards, out_dir / "breeze-tts-code-predictor", n_cp, args.cp_max_pos, args.cp_vocab)
    if not args.skip_embeds:
        onnx_dir = None if args.skip_onnx else out_dir / "lm_head-onnx"
        export_runtime_artifacts(shards, out_dir / "embeds", onnx_dir)
    shards.close()
    log("ALL DONE")


if __name__ == "__main__":
    main()
