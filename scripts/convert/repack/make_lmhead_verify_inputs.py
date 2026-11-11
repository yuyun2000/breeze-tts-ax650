"""Emit lm_head verification inputs (CP post-norm hidden) + CPU fp32 reference logits."""
import json, sys
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from transformers import LlamaForCausalLM
torch.set_num_threads(8)
root = Path(__file__).resolve().parent.parent
breeze = root / "ref/model-breeze-tts-2"
repack = root / "repack/out"
gold = root / "ref/golden_zh_short_greedy"
wm = json.loads((breeze / "model.safetensors.index.json").read_text())["weight_map"]
def orig(k):
    with safe_open(str(breeze / wm[k]), "pt") as sf:
        return sf.get_tensor(k).float()
groups = orig("depth_decoder.model.embed_tokens.weight").reshape(16, 2051, 2048)
W = orig("depth_decoder.model.inputs_embeds_projector.weight")
heads = orig("depth_decoder.codebooks_head.weight")  # [15,1024,2051]
e_proj = torch.stack([groups[i] @ W.t() for i in range(15)])
tokens = torch.from_numpy(np.load(gold / "tokens.npz")["tokens"])  # [64,16]
probe = np.load(gold / "probe.npz")
hidden = torch.from_numpy(probe["hidden_norm"]).float()  # [20,2048]
cp = LlamaForCausalLM.from_pretrained(str(repack / "breeze-tts-code-predictor"), dtype=torch.float32).eval()

def causal_mask(n):
    return torch.zeros(1,1,n,n).masked_fill(torch.triu(torch.ones(n,n,dtype=torch.bool),1), float("-inf"))

samples = []   # list of [1024] CP post-norm hidden
refs = []      # list of [15,2051] fp32 logits
for f in range(20):
    hin = hidden[f].reshape(1,1,2048)
    dins = [torch.nn.functional.linear(hin, W)]
    per_step = []
    for step in range(1, 16):
        e = e_proj[step-1][int(tokens[f, step-1])].reshape(1,1,1024)
        dins.append(e)
        seq = torch.cat(dins, dim=1)
        with torch.no_grad():
            od = cp.model(inputs_embeds=seq, attention_mask=causal_mask(seq.shape[1]),
                          position_ids=torch.arange(seq.shape[1]).unsqueeze(0))
            hn = od.last_hidden_state[:, -1]                 # [1,1024] post-norm hidden
            lg = torch.nn.functional.linear(hn, heads[step-1].t())[0]  # [2051]
        samples.append(hn[0].numpy().astype(np.float32))
        per_step.append(lg.numpy().astype(np.float32))
    refs.append(np.stack(per_step))  # [15,2051]
refs = np.concatenate(refs, axis=0)  # [60,2051] (4 frames x 15 steps)
samples_arr = np.stack(samples)      # [60,1024]
out = {"hidden": samples_arr, "ref_logits": refs}
np.savez("/tmp/lmhead_verify_inputs.npz", **out)
print("saved /tmp/lmhead_verify_inputs.npz", samples_arr.shape, refs.shape)
# per-head calibration tars: head i calibration inputs are hidden at step i of every frame
import tarfile, io, os
calib_dir = "/tmp/lmhead_calib"
os.makedirs(calib_dir, exist_ok=True)
for i in range(15):
    idxs = [f * 15 + i for f in range(15)]   # calibration frames 0..14
    tar_path = f"{calib_dir}/head_{i:02d}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for n, k in enumerate(idxs):
            a = samples_arr[k].reshape(1, 1, 1024).astype(np.float32)
            buf = io.BytesIO(); np.save(buf, a); buf.seek(0)
            info = tarfile.TarInfo(name=f"input_{n:03d}.npy"); info.size = buf.getbuffer().nbytes
            tar.addfile(info, buf)
    print("wrote", tar_path)

