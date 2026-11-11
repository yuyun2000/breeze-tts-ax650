#!/usr/bin/env python3
"""对比 repacked Qwen3/Llama 目录与原始 Breeze 模块的输出是否一致（单测 hidden 与 logits）。"""
import argparse, os, sys
import torch

p = argparse.ArgumentParser()
p.add_argument("--src", default=os.path.expanduser("~/work/breeze-tts/data/breeze-tts-2"))
p.add_argument("--out", default=os.path.expanduser("~/work/breeze-tts/repack-lead/output"))
p.add_argument("--breeze-src", default=os.path.expanduser("~/work/breeze-tts/data/breeze-tts-src"))
a = p.parse_args()
sys.path.insert(0, a.breeze_src)
from models.breeze import BreezeForConditionalGeneration
from transformers import AutoModelForCausalLM

def cos(x, y):
    x = x.float().flatten(); y = y.float().flatten()
    return torch.nn.functional.cosine_similarity(x, y, dim=0).item(), (x - y).abs().max().item()

print("[load] original Breeze bf16 ...")
torch.manual_seed(0)
br = BreezeForConditionalGeneration.from_pretrained(a.src, dtype=torch.bfloat16).eval()
with torch.no_grad():
    x = torch.randn(1, 8, 2048, dtype=torch.bfloat16)
    tm = AutoModelForCausalLM.from_pretrained(os.path.join(a.out, "qwen3-talker"), dtype=torch.bfloat16).eval()
    hq = tm.model(inputs_embeds=x).last_hidden_state
    hb = br.backbone_model(inputs_embeds=x).last_hidden_state
    c, m = cos(hq, hb)
    print(f"[talker] hidden cosine={c:.8f} max_abs={m:.3e}")
    c2, m2 = cos(tm.lm_head(hq), br.lm_head(hb))
    print(f"[talker] logits cosine={c2:.8f} max_abs={m2:.3e}")
    del tm

    # depth: 通过同一组 token ids 比较（Breeze 侧含 embed+proj 路径；Llama 侧喂 proj 后 embeds）
    ids = torch.tensor([[5, 7, 9, 11]])
    cache_pos = torch.arange(4)
    offs = torch.clamp(cache_pos - 1, min=0) * 2051
    emb = br.depth_decoder.model.embed_tokens(ids + offs)
    emb = br.depth_decoder.model.inputs_embeds_projector(emb)
    dm = AutoModelForCausalLM.from_pretrained(os.path.join(a.out, "llama-depth"), dtype=torch.bfloat16).eval()
    dq = dm.model(inputs_embeds=emb).last_hidden_state
    db = br.depth_decoder.model(input_ids=ids, cache_position=cache_pos).last_hidden_state
    c3, m3 = cos(dq, db)
    print(f"[depth] hidden cosine={c3:.8f} max_abs={m3:.3e}")
    del dm
print("done")
