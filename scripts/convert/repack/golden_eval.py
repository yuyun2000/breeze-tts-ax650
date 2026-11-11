#!/usr/bin/env python3
"""End-to-end golden check on the real prompt:

  golden data (ref-2038, original Breeze full model, greedy):
    ref/golden_zh_short_greedy/probe.npz   text_embeds [34,2048] (text encoder output)
    ref/golden_zh_short_greedy/tokens.npz  tokens [64,16] (ground-truth greedy frames)
  pipelines:
    A) original Breeze modules (BreezeBackboneAdapter + BreezeDepthDecoder) -- sanity vs golden
    B) repacked Qwen3ForCausalLM + LlamaForCausalLM + emitted embeds bins
  metrics:
    token exact-frame rate / codebook acc / first divergence frame / cb0 acc,
    backbone top-5 logit agreement vs golden first20,
    post-norm hidden cosine vs golden hidden_norm.

Run:  ./venv-ref/bin/python repack/golden_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import LlamaForCausalLM, Qwen3ForCausalLM

torch.set_num_threads(12)
RESERVED = torch.tensor([2048, 2049, 2050])


def load_shard_state(model_dir: Path, prefixes):
    out, handles = {}, {}
    index_path = model_dir / "model.safetensors.index.json"
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


def causal_mask(n, dtype):
    m = torch.zeros(1, 1, n, n, dtype=dtype)
    return m.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))


def masked_argmax(logits):
    lg = logits.clone().float()
    lg[..., RESERVED] = float("-inf")
    return int(lg[0].argmax())


def main():
    root = Path(__file__).resolve().parent.parent
    breeze_dir = root / "ref/model-breeze-tts-2"
    repack_dir = root / "repack/out"
    gold_dir = root / "ref/golden_zh_short_greedy"
    sys.path.insert(0, str(root / "breeze-tts-src"))

    from models.breeze import BreezeDepthDecoderForCausalLM
    from models.breeze_backbone_factory import BreezeBackboneFactory
    from models.breeze_config import BreezeConfig

    cfg_dict = json.loads((breeze_dir / "config.json").read_text())
    cfg_dict.pop("transformers_version", None)
    bcfg = BreezeConfig(**cfg_dict)
    dtype = torch.float32

    probe = np.load(gold_dir / "probe.npz")
    gold = np.load(gold_dir / "tokens.npz")
    text_embeds = torch.from_numpy(probe["text_embeds"]).float()          # [T,2048]
    gold_tokens = torch.from_numpy(gold["tokens"]).long()                # [F,16]
    gold_top5_idx = torch.from_numpy(probe["backbone_topk_idx"]).long()  # [20,5]
    gold_top5_val = torch.from_numpy(probe["backbone_topk_val"]).float() # [20,5]
    gold_hidden = torch.from_numpy(probe["hidden_norm"]).float()         # [20,2048]
    F = gold_tokens.shape[0]
    print(f"prompt text_embeds {tuple(text_embeds.shape)}, golden frames {F}")

    # ---- original modules ----
    backbone = BreezeBackboneFactory.create_backbone(bcfg)
    bb_state = {k[len("backbone_model."):]: v
                for k, v in load_shard_state(breeze_dir, ["backbone_model."]).items()}
    backbone.load_state_dict(bb_state, strict=False)
    backbone = backbone.to(dtype).eval()
    depth = BreezeDepthDecoderForCausalLM._from_config(bcfg.depth_decoder_config)
    dd_state = {k[len("depth_decoder."):]: v
                for k, v in load_shard_state(breeze_dir, ["depth_decoder."]).items()}
    depth.load_state_dict(dd_state, strict=False)
    depth = depth.to(dtype).eval()
    lm_head = load_shard_state(breeze_dir, ["lm_head.weight"])["lm_head.weight"].to(dtype)
    groups = load_shard_state(breeze_dir, ["depth_decoder.model.embed_tokens"])[
        "depth_decoder.model.embed_tokens.weight"].to(dtype).reshape(16, 2051, 2048)
    proj_w = load_shard_state(breeze_dir, ["depth_decoder.model.inputs_embeds_projector"])[
        "depth_decoder.model.inputs_embeds_projector.weight"].to(dtype)
    heads = load_shard_state(breeze_dir, ["depth_decoder.codebooks_head"])[
        "depth_decoder.codebooks_head.weight"].to(dtype)

    # ---- repacked models + runtime artefacts ----
    talker_rep = Qwen3ForCausalLM.from_pretrained(
        str(repack_dir / "breeze-tts-talker"), dtype=dtype).eval()
    cp_rep = LlamaForCausalLM.from_pretrained(
        str(repack_dir / "breeze-tts-code-predictor"), dtype=dtype).eval()
    emb_bin = bf16_bin(repack_dir / "embeds/audio_embed_16x2051x2048.bf16.bin", (16, 2051, 2048)).to(dtype)
    proj_bin = bf16_bin(repack_dir / "embeds/inputs_embeds_projector_1024x2048.bf16.bin", (1024, 2048)).to(dtype)
    e_proj_bin = bf16_bin(repack_dir / "embeds/codec_embed_proj_15x2051x1024.bf16.bin", (15, 2051, 1024)).to(dtype)
    heads_bin = bf16_bin(repack_dir / "embeds/codebooks_head_15x1024x2051.bf16.bin", (15, 1024, 2051)).to(dtype)

    def apply_rep_penalty(scores, hist, penalty=1.1):
        if not hist:
            return scores
        sc = scores.clone().float()
        for tok in set(hist):
            if 0 <= tok < sc.shape[-1]:
                sc[0, tok] = sc[0, tok] * penalty if sc[0, tok] < 0 else sc[0, tok] / penalty
        return sc

    def run_pipeline(use_repack, max_frames=F):
        frames_emb, all_codes = [], []
        top5_idx, top5_val, hidden_norms = [], [], []
        hist = []  # generated cb0 history only (repetition penalty), prompt not included
        prompt = text_embeds.unsqueeze(0)
        for f in range(max_frames):
            seq = torch.cat([prompt] + frames_emb, dim=1) if frames_emb else prompt
            n = seq.shape[1]
            posn = torch.arange(n).unsqueeze(0)
            maskn = causal_mask(n, dtype)
            with torch.no_grad():
                if use_repack:
                    out = talker_rep.model(inputs_embeds=seq, attention_mask=maskn, position_ids=posn)
                    hidden = out.last_hidden_state[:, -1:]
                    lg = talker_rep.lm_head(hidden)[:, -1]
                else:
                    rope = backbone.rotary_emb(seq, posn)
                    h = seq
                    for layer in backbone.layers:
                        o = layer(h, attention_mask=maskn, position_ids=posn, position_embeddings=rope)
                        h = o[0] if isinstance(o, tuple) else o
                    hidden = backbone.norm(h[:, -1:])
                    lg = torch.nn.functional.linear(hidden[:, -1], lm_head)
            vals, idx = torch.topk(lg.float(), 5)
            top5_idx.append(idx[0].tolist())
            top5_val.append(vals[0].tolist())
            c0 = masked_argmax(apply_rep_penalty(lg, hist))
            codes = [c0]
            has_eos = (c0 == 2051)
            dh = torch.nn.functional.linear(hidden[:, -1], proj_w)  # [1,1024]
            dinputs = [dh.unsqueeze(1)]
            for step in range(1, 16):
                if not has_eos:
                    e = groups[step - 1][codes[step - 1]].reshape(1, 1, 2048)
                    dinputs.append(e)
                seqd = torch.cat(dinputs, dim=1)
                m_ = seqd.shape[1]
                maskd = causal_mask(m_, dtype)
                with torch.no_grad():
                    if use_repack:
                        od = cp_rep.model(inputs_embeds=seqd, attention_mask=maskd,
                                          position_ids=torch.arange(m_).unsqueeze(0))
                        hn = od.last_hidden_state[:, -1]
                        lg2 = torch.nn.functional.linear(hn, heads_bin[step - 1].t())
                    else:
                        od = depth.model(inputs_embeds=seqd, attention_mask=maskd)
                        hn = depth.model.norm(od.last_hidden_state[:, -1:])[:, -1]
                        lg2 = torch.nn.functional.linear(hn, heads[step - 1].t())
                if has_eos:
                    codes.append(2050)
                else:
                    c = masked_argmax(lg2)
                    codes.append(c)
            all_codes.append(codes)
            hist.append(c0)
            # post-norm hidden is the talker post-norm output (== backbone_last_hidden_state)
            hidden_norms.append(hidden[:, -1].clone())
            frame = torch.zeros(1, 2048, dtype=dtype)
            for i, c in enumerate(codes):
                frame = frame + groups[i][c].reshape(1, 2048)
            frames_emb.append(frame.unsqueeze(1))
            if has_eos:
                break
        return all_codes, top5_idx, top5_val, torch.cat(hidden_norms, dim=0)

    def report(name, codes, top5_idx, top5_val, hidden_norms):
        tok = torch.tensor(codes, dtype=torch.long)
        n = min(tok.shape[0], gold_tokens.shape[0])
        eq = (tok[:n] == gold_tokens[:n])
        exact = eq.all(dim=1).float().mean().item()
        cb_acc = eq.float().mean(dim=0).tolist()
        gold = gold_tokens[:n]
        first_div = -1
        for i in range(n):
            if not bool(eq[i].all()):
                first_div = i
                break
        cb0_eq = (tok[:n, 0] == gold[:n, 0])
        cb0_first_div = -1
        for i in range(n):
            if not bool(cb0_eq[i]):
                cb0_first_div = i
                break
        # top5 logit comparison
        k = min(len(top5_idx), gold_top5_idx.shape[0])
        top5_match = np.mean([top5_idx[i][0] == int(gold_top5_idx[i][0]) for i in range(k)])
        top5_val_mae = float((torch.tensor(top5_val[:k]) - gold_top5_val[:k]).abs().mean())
        hn_cos = float(torch.nn.functional.cosine_similarity(
            hidden_norms[:k].flatten(), gold_hidden[:k].flatten(), dim=0))
        print(f"\n[{name}] frames={len(codes)} vs golden {gold_tokens.shape[0]}")
        print(f"   exact_frame_rate = {exact*100:.2f}%   first_divergence_frame = {first_div}")
        print(f"   cb0_acc = {cb0_eq.float().mean().item()*100:.2f}%  cb0_first_div = {cb0_first_div}")
        print(f"   codebook_acc = {[round(a*100,2) for a in cb_acc]}")
        print(f"   backbone top1 match(first20) = {top5_match*100:.2f}%  top5 logit MAE = {top5_val_mae:.4e}")
        print(f"   hidden_norm cos(first20) = {hn_cos:.8f}")
        return dict(exact_frame_rate=exact, first_divergence_frame=first_div,
                    cb0_acc=float(cb0_eq.float().mean()), cb0_first_div=cb0_first_div,
                    codebook_acc=cb_acc, top1_match=float(top5_match),
                    top5_val_mae=top5_val_mae, hidden_norm_cos=hn_cos, n_frames=len(codes))

    print("\n=== pipeline A: original Breeze modules (sanity vs golden) ===")
    codes_a, i_a, v_a, h_a = run_pipeline(False)
    res_a = report("original", codes_a, i_a, v_a, h_a)
    print("\n=== pipeline B: repacked models + emitted bins ===")
    codes_b, i_b, v_b, h_b = run_pipeline(True)
    res_b = report("repacked", codes_b, i_b, v_b, h_b)

    n = min(len(codes_a), len(codes_b))
    ab = sum(codes_a[i] == codes_b[i] for i in range(n))
    print(f"\noriginal vs repacked frame agreement: {ab}/{n} = {ab/n*100:.2f}%")
    out = Path(__file__).resolve().parent / "golden_eval_result.json"
    out.write_text(json.dumps(dict(original=res_a, repacked=res_b,
                                   ab_frames=ab, ab_total=n), indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
