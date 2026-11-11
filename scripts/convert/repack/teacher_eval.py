#!/usr/bin/env python3
"""Teacher-forced golden verification (definitive repack correctness evidence).

For every frame f, the repacked talker is fed the GOLDEN prompt + GOLDEN previous
frame embeddings (no free-running feedback), so any mismatch is the model's, not
compounding drift:
  - talker: hidden cos vs golden post-norm hidden, raw top-5 logits agreement,
    penalized-argmax cb0 agreement vs golden tokens[:,0]
  - depth : teacher-forced steps with the golden cb0 + golden previous codes,
    per-step argmax agreement vs golden tokens[f,1:]
Also the same teacher-forced pass through the ORIGINAL full Breeze model to give
the bf16-vs-bf16 ceiling (should be ~100%).

Run:  ./venv-ref/bin/python repack/teacher_eval.py --dtype bfloat16 --frames 64
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
RESERVED = torch.tensor([2048, 2049, 2050])


def orig_tensor(model_dir: Path, key: str) -> torch.Tensor:
    wm = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(model_dir / wm[key]), "pt", device="cpu") as sf:
        return sf.get_tensor(key)


def bf16_bin(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(shape)


def causal_mask(n, dtype):
    m = torch.zeros(1, 1, n, n, dtype=torch.float32)
    m = m.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    return m.to(dtype)


def apply_penalty(scores, hist, penalty=1.1):
    sc = scores.clone().float().reshape(-1)
    for tok in set(hist):
        if 0 <= tok < sc.shape[-1]:
            sc[tok] = sc[tok] * penalty if sc[tok] < 0 else sc[tok] / penalty
    return sc


def select(logits, hist):
    sc = apply_penalty(logits, hist)
    sc[RESERVED] = float("-inf")
    return int(sc.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--frames", type=int, default=64)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    root = Path(__file__).resolve().parent.parent
    breeze_dir = root / "ref/model-breeze-tts-2"
    repack_dir = root / "repack/out"
    gold_dir = root / "ref/golden_zh_short_greedy"
    sys.path.insert(0, str(root / "breeze-tts-src"))

    probe = np.load(gold_dir / "probe.npz")
    gold = np.load(gold_dir / "tokens.npz")
    text_embeds = torch.from_numpy(probe["text_embeds"]).to(dtype).unsqueeze(0)
    tokens = torch.from_numpy(gold["tokens"])          # [F,16]
    hidden = torch.from_numpy(probe["hidden_norm"]).float()  # [20,2048]
    depth_top1 = torch.from_numpy(probe["depth_top1"])       # [20,15]
    F = min(args.frames, tokens.shape[0])
    print(f"dtype={dtype} frames={F} prompt={tuple(text_embeds.shape)}")

    W = orig_tensor(breeze_dir, "depth_decoder.model.inputs_embeds_projector.weight").to(dtype)
    groups = orig_tensor(breeze_dir, "depth_decoder.model.embed_tokens.weight").to(dtype).reshape(16, 2051, 2048)
    heads = orig_tensor(breeze_dir, "depth_decoder.codebooks_head.weight").to(dtype)
    e_proj = torch.stack([groups[i].float() @ W.float().t() for i in range(15)]).to(dtype)
    emb_bin = bf16_bin(repack_dir / "embeds/audio_embed_16x2051x2048.bf16.bin", (16, 2051, 2048)).to(dtype)
    proj_bin = bf16_bin(repack_dir / "embeds/inputs_embeds_projector_1024x2048.bf16.bin", (1024, 2048)).to(dtype)
    e_proj_bin = bf16_bin(repack_dir / "embeds/codec_embed_proj_15x2051x1024.bf16.bin", (15, 2051, 1024)).to(dtype)
    heads_bin = bf16_bin(repack_dir / "embeds/codebooks_head_15x1024x2051.bf16.bin", (15, 1024, 2051)).to(dtype)

    talker = Qwen3ForCausalLM.from_pretrained(str(repack_dir / "breeze-tts-talker"), dtype=dtype).eval()
    cp = LlamaForCausalLM.from_pretrained(str(repack_dir / "breeze-tts-code-predictor"), dtype=dtype).eval()

    def frame_emb(codes):
        e = torch.zeros(1, 1, 2048, dtype=dtype)
        for i, c in enumerate(codes):
            e = e + groups[i][c].reshape(1, 1, 2048)
        return e

    # ---------------- talker teacher-forced ----------------
    frames_emb = []
    cb0_hits = top5_hits = 0
    cos_list, mae_list = [], []
    print("\n== talker teacher-forced over golden history ==")
    for f in range(F):
        seq = torch.cat([text_embeds] + frames_emb, dim=1) if frames_emb else text_embeds
        posn = torch.arange(seq.shape[1]).unsqueeze(0)
        maskn = causal_mask(seq.shape[1], dtype)
        with torch.no_grad():
            out = talker.model(inputs_embeds=seq, attention_mask=maskn, position_ids=posn)
            hn = out.last_hidden_state[0, -1].float()
            logits = talker.lm_head(out.last_hidden_state[:, -1:].float())[0, -1].float()
        if f < hidden.shape[0]:
            c = torch.nn.functional.cosine_similarity(hn, hidden[f], dim=0).item()
            cos_list.append(c)
            # golden raw top5
            vals, idx = torch.topk(logits, 5)
            gidx = torch.from_numpy(probe["backbone_topk_idx"][f]).long()
            gval = torch.from_numpy(probe["backbone_topk_val"][f]).float()
            top5_hits += int(len(set(idx.tolist()) & set(gidx.tolist())) >= 4)
            mae_list.append((vals - gval).abs().mean().item())
        c0 = select(logits, [int(c) for c in tokens[:f, 0]])
        cb0_hits += int(c0 == int(tokens[f, 0]))
        frames_emb.append(frame_emb(tokens[f].tolist()))
    print(f"  cb0 agreement vs golden : {cb0_hits}/{F} = {cb0_hits/F*100:.2f}%")
    n20 = len(cos_list)
    if n20:
        print(f"  hidden cos (first {n20}) : mean={np.mean(cos_list):.8f} min={np.min(cos_list):.8f}")
        print(f"  raw top5 >=4/5 agreement: {top5_hits}/{n20} = {top5_hits/n20*100:.2f}%")
        print(f"  raw top5 logit MAE      : {np.mean(mae_list):.4e}")

    # ---------------- depth teacher-forced ----------------
    print("\n== depth teacher-forced over golden hidden/codes ==")
    step_hits = 0
    frame_exact = 0
    nfr = min(F, hidden.shape[0])
    for f in range(nfr):
        h_in = hidden[f].to(dtype).reshape(1, 1, 2048)
        dinputs = [torch.nn.functional.linear(h_in, proj_bin)]
        hits = 0
        for step in range(1, 16):
            # input embedding of the GOLDEN previous codebook token (group = step-1)
            e = e_proj_bin[step - 1][int(tokens[f, step - 1])].reshape(1, 1, 1024)
            dinputs.append(e)
            seqd = torch.cat(dinputs, dim=1)
            m = seqd.shape[1]
            maskd = causal_mask(m, dtype)
            with torch.no_grad():
                od = cp.model(inputs_embeds=seqd, attention_mask=maskd,
                              position_ids=torch.arange(m).unsqueeze(0))
                logits = torch.nn.functional.linear(od.last_hidden_state[:, -1].float(), heads_bin[step - 1].t())[0]
            lg = logits.clone()
            lg[RESERVED] = float("-inf")
            c = int(lg.argmax())
            hits += int(c == int(tokens[f, step]))
            step_hits += int(c == int(tokens[f, step]))
        frame_exact += int(hits == 15)
    total = nfr * 15
    print(f"  depth step agreement : {step_hits}/{total} = {step_hits/total*100:.2f}%")
    print(f"  depth frame exact    : {frame_exact}/{nfr} = {frame_exact/nfr*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
