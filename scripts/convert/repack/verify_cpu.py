#!/usr/bin/env python3
"""CPU numerical equivalence check: original Breeze TTS 2 modules vs the
repacked HF-format directories exported by repack_breeze.py.

Checks
  [0] emitted runtime bf16 bins vs f32 reference (fidelity only)
  [1] talker: original BreezeBackboneAdapter vs Qwen3ForCausalLM(repack)
  [2] depth : BreezeDepthDecoderForCausalLM(2048-dim unprojected inputs) vs
              LlamaForCausalLM(repack, 1024-dim projected inputs)
  [3] end-to-end greedy 16-codebook x N frames: original pipeline vs repacked
      pipeline (both float32, so this measures structural/rope/norm equivalence)

Breeze depth decoder ALWAYS applies `inputs_embeds_projector` (2048->1024) to
its inputs, which is why the reference path receives 2048-dim embeddings and
the repacked Llama path receives the pre-projected 1024-dim ones.

Run with the ref venv (torch 2.15 + transformers 4.57.3):
  ./venv-ref/bin/python repack/verify_cpu.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import LlamaForCausalLM, Qwen3ForCausalLM

torch.set_num_threads(12)
torch.manual_seed(0)


def load_shard_state(model_dir: Path, prefixes):
    out = {}
    index_path = model_dir / "model.safetensors.index.json"
    handles = {}
    if index_path.exists():
        wm = json.loads(index_path.read_text())["weight_map"]
        for k, f in wm.items():
            if not any(k.startswith(p) for p in prefixes):
                continue
            if f not in handles:
                handles[f] = safe_open(str(model_dir / f), "pt", device="cpu")
            out[k] = handles[f].get_tensor(k)
    else:
        f = "model.safetensors"
        handles[f] = safe_open(str(model_dir / f), "pt", device="cpu")
        for k in handles[f].keys():
            if any(k.startswith(p) for p in prefixes):
                out[k] = handles[f].get_tensor(k)
    return out


def bf16_bin(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(shape)


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0))


def stats(name, a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    print(f"  {name:44s} cos={cos(a, b):.8f} maxabs={d.max().item():.3e} "
          f"meanabs={d.mean().item():.3e}")
    return cos(a, b), float(d.max())


def causal_mask(n, dtype):
    m = torch.zeros(1, 1, n, n, dtype=dtype)
    tri = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    return m.masked_fill(tri, float("-inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breeze-dir", default="ref/model-breeze-tts-2")
    ap.add_argument("--repack-dir", default="repack/out")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--prompt-len", type=int, default=8)
    args = ap.parse_args()

    breeze_dir = Path(args.breeze_dir)
    repack_dir = Path(args.repack_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "breeze-tts-src"))

    from models.breeze import BreezeDepthDecoderForCausalLM
    from models.breeze_backbone_factory import BreezeBackboneFactory
    from models.breeze_config import BreezeConfig

    cfg_dict = json.loads((breeze_dir / "config.json").read_text())
    cfg_dict.pop("transformers_version", None)
    bcfg = BreezeConfig(**cfg_dict)
    dtype = torch.float32
    ok = True

    # ------------------------------------------------------------------
    print("== load original Breeze modules ==")
    backbone = BreezeBackboneFactory.create_backbone(bcfg)
    bb_state = {k[len("backbone_model."):]: v
                for k, v in load_shard_state(breeze_dir, ["backbone_model."]).items()}
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    # embedding table may be tied/absent; that's fine (backbone only gets embeds)
    backbone = backbone.to(dtype).eval()
    depth = BreezeDepthDecoderForCausalLM._from_config(bcfg.depth_decoder_config)
    dd_state = {k[len("depth_decoder."):]: v
                for k, v in load_shard_state(breeze_dir, ["depth_decoder."]).items()}
    missing, unexpected = depth.load_state_dict(dd_state, strict=False)
    print(f"  depth missing={len(missing)} unexpected={len(unexpected)}")
    depth = depth.to(dtype).eval()
    lm_head = load_shard_state(breeze_dir, ["lm_head.weight"])["lm_head.weight"].to(dtype)
    groups = load_shard_state(breeze_dir, ["depth_decoder.model.embed_tokens"])[
        "depth_decoder.model.embed_tokens.weight"].to(dtype).reshape(16, 2051, 2048)
    proj_w = load_shard_state(breeze_dir, ["depth_decoder.model.inputs_embeds_projector"])[
        "depth_decoder.model.inputs_embeds_projector.weight"].to(dtype)  # [1024,2048]
    heads = load_shard_state(breeze_dir, ["depth_decoder.codebooks_head"])[
        "depth_decoder.codebooks_head.weight"].to(dtype)  # [15,1024,2051]
    e_proj_f32 = torch.stack([groups[i] @ proj_w.t() for i in range(15)])  # [15,2051,1024]
    print(f"  backbone {sum(p.numel() for p in backbone.parameters())/1e6:.0f}M params, "
          f"depth {sum(p.numel() for p in depth.parameters())/1e6:.0f}M params")

    print("== load repacked HF models ==")
    talker_rep = Qwen3ForCausalLM.from_pretrained(
        str(repack_dir / "breeze-tts-talker"), torch_dtype=dtype).eval()
    cp_rep = LlamaForCausalLM.from_pretrained(
        str(repack_dir / "breeze-tts-code-predictor"), torch_dtype=dtype).eval()

    # ------------------------------------------------------------------
    print("\n== [0] emitted bf16 runtime artefacts vs f32 reference ==")
    emb_bin = bf16_bin(repack_dir / "embeds/audio_embed_16x2051x2048.bf16.bin", (16, 2051, 2048))
    proj_bin = bf16_bin(repack_dir / "embeds/inputs_embeds_projector_1024x2048.bf16.bin", (1024, 2048))
    e_proj_bin = bf16_bin(repack_dir / "embeds/codec_embed_proj_15x2051x1024.bf16.bin", (15, 2051, 1024))
    heads_bin = bf16_bin(repack_dir / "embeds/codebooks_head_15x1024x2051.bf16.bin", (15, 1024, 2051))
    stats("audio_embed bin", groups, emb_bin)
    stats("inputs_embeds_projector bin", proj_w, proj_bin)
    stats("codec_embed_proj bin", e_proj_f32, e_proj_bin)
    stats("codebooks_head bin", heads, heads_bin)

    # ------------------------------------------------------------------
    print("\n== [1] talker: BreezeBackboneAdapter vs repacked Qwen3 ==")
    T = 24
    x = torch.randn(1, T, 2048) * 0.3
    pos = torch.arange(T).unsqueeze(0)
    mask = causal_mask(T, dtype)
    ref_layers, rep_layers = [], []

    def hook(store):
        def fn(_m, _i, out):
            store.append(out[0].detach() if isinstance(out, tuple) else out.detach())
        return fn

    h_ref, h_rep = [], []
    for i, layer in enumerate(backbone.layers):
        layer.register_forward_hook(hook(h_ref))
    for i, layer in enumerate(talker_rep.model.layers):
        layer.register_forward_hook(hook(h_rep))
    with torch.no_grad():
        ref_out = backbone(inputs_embeds=x, attention_mask=mask)
        ref_normed = ref_out.last_hidden_state  # already post-final-norm
        ref_logits = torch.nn.functional.linear(ref_normed, lm_head)
        rep_out = talker_rep.model(inputs_embeds=x, attention_mask=mask, position_ids=pos)
        rep_normed = rep_out.last_hidden_state  # already post-final-norm
        rep_logits = talker_rep.lm_head(rep_normed)
    c, m = stats("talker final normed hidden", ref_normed, rep_normed)
    ok &= c > 0.9999 and m < 1e-4
    c, m = stats("talker lm_head logits", ref_logits, rep_logits)
    ok &= c > 0.9999 and m < 1e-3
    layer_cos = min(cos(a, b) for a, b in zip(h_ref, h_rep))
    print(f"  min per-layer hidden cos over {len(h_ref)} layers = {layer_cos:.8f}")
    ok &= layer_cos > 0.9999

    # ------------------------------------------------------------------
    print("\n== [2] depth: Breeze depth(2048-dim) vs repacked Llama(1024-dim) ==")
    S = 16
    raw_in = torch.randn(1, S, 2048) * 0.3          # what the talker hidden side feeds
    proj_in = torch.nn.functional.linear(raw_in, proj_w)  # 1024-dim
    posS = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        ref_h = depth.model(inputs_embeds=raw_in, position_ids=posS).last_hidden_state
        ref_lg = torch.stack(
            [torch.nn.functional.linear(ref_h[:, i + 1, :], heads[i].t()) for i in range(S - 1)],
            dim=1)
        rep_h = cp_rep.model(inputs_embeds=proj_in, position_ids=posS).last_hidden_state
        rep_lg = torch.stack(
            [torch.nn.functional.linear(rep_h[:, i + 1, :], heads[i].t()) for i in range(S - 1)],
            dim=1)
    c, m = stats("depth 16-step normed hidden", ref_h, rep_h)
    ok &= c > 0.9999 and m < 1e-4
    c, m = stats("depth 15 position-head logits", ref_lg, rep_lg)
    ok &= c > 0.9999 and m < 1e-3

    # ------------------------------------------------------------------
    print(f"\n== [3] greedy {args.frames} frames x 16 codebooks ==")

    RESERVED = torch.tensor([2048, 2049, 2050])

    def masked_argmax(logits):
        logits = logits.clone().float()
        logits[..., RESERVED] = float("-inf")
        return int(logits[0].argmax())

    def run_pipeline(use_repack):
        prompt = torch.randn(1, args.prompt_len, 2048) * 0.3
        frames = []
        all_codes = []
        for _ in range(args.frames):
            seq = torch.cat([prompt] + frames, dim=1) if frames else prompt
            n = seq.shape[1]
            posn = torch.arange(n).unsqueeze(0)
            maskn = causal_mask(n, dtype)
            with torch.no_grad():
                if use_repack:
                    out = talker_rep.model(inputs_embeds=seq, attention_mask=maskn,
                                           position_ids=posn)
                    hidden = talker_rep.model.norm(out.last_hidden_state)
                    c0 = masked_argmax(talker_rep.lm_head(hidden[:, -1]))
                    depth_inputs = [torch.nn.functional.linear(hidden[:, -1], proj_w).unsqueeze(1)]
                else:
                    rope = backbone.rotary_emb(seq, posn)
                    h = seq
                    for layer in backbone.layers:
                        _o = layer(h, attention_mask=maskn, position_ids=posn,
                                   position_embeddings=rope)
                        h = _o[0] if isinstance(_o, tuple) else _o
                    hidden = backbone.norm(h)
                    c0 = masked_argmax(torch.nn.functional.linear(hidden[:, -1], lm_head))
                    depth_inputs = [hidden[:, -1].unsqueeze(1)]  # depth model projects internally
                codes = [c0]
                depth_inputs = [torch.nn.functional.linear(hidden[:, -1], proj_w).unsqueeze(1)] \
                    if use_repack else [hidden[:, -1].unsqueeze(1)]
                for step in range(1, 16):
                    if use_repack:
                        e = e_proj_f32[step - 1][codes[step - 1]].reshape(1, 1, 1024)
                    else:
                        e = groups[step - 1][codes[step - 1]].reshape(1, 1, 2048)
                    depth_inputs.append(e)
                    seqd = torch.cat(depth_inputs, dim=1)
                    m_ = seqd.shape[1]
                    maskd = causal_mask(m_, dtype)
                    if use_repack:
                        od = cp_rep.model(inputs_embeds=seqd, attention_mask=maskd,
                                          position_ids=torch.arange(m_).unsqueeze(0))
                        hn = od.last_hidden_state[:, -1]
                    else:
                        od = depth.model(inputs_embeds=seqd, attention_mask=maskd)
                        hn = depth.model.norm(od.last_hidden_state[:, -1:])[:, -1]
                    lg = torch.nn.functional.linear(hn, heads[step - 1].t())
                    c = masked_argmax(lg)
                    codes.append(c)
                all_codes.append(codes)
                if c0 == 2051:
                    break
                frame = torch.zeros(1, 2048, dtype=torch.float32)
                for i, c in enumerate(codes):
                    frame = frame + groups[i][c].reshape(1, 2048)
                frames.append(frame.unsqueeze(1))
        return all_codes

    codes_ref = run_pipeline(False)
    codes_rep = run_pipeline(True)
    flat_ref = [c for fr in codes_ref for c in fr]
    flat_rep = [c for fr in codes_rep for c in fr]
    match = sum(a == b for a, b in zip(flat_ref, flat_rep)) / len(flat_ref)
    print(f"  greedy token agreement: {match*100:.2f}% ({len(flat_ref)} tokens)")
    for fi, (a, b) in enumerate(zip(codes_ref, codes_rep)):
        if a != b:
            print(f"  frame {fi}:\n    ref={a}\n    rep={b}")
    ok &= match == 1.0

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
