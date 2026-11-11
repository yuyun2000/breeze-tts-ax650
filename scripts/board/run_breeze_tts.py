#!/usr/bin/env python3
"""Breeze TTS 2 board runtime — numpy + pyaxengine only (no torch / no transformers).

Target: AX650 (AX8850) dev board, AX Engine 2.12+, python3.10.
Two presets:
  * `qwen3tts06` — AXERA Qwen3-TTS-12Hz-0.6B-Base-AX650 package. Used to validate the glue
    (talker prefill/decode loop, 15-step depth decoder, 8-layer codec transformer, vocoder)
    before the real Breeze artifacts land. Run with `--random-text-embeds 8` for a dry run.
  * `breeze` — Breeze TTS 2 (Voice Design). Needs talker/code-predictor axmodels (repacked+built),
    16 unprojected audio-embedding tables (bf16), 15 projected depth tables + proj matrix,
    text-encoder axmodel + embedding table + tokenizer.json.

All axmodel I/O shapes/protocol are discovered from the compiled models at runtime
(`get_inputs(g)` / `get_outputs(g)`), never hardcoded — see `_work/board/BOARD_REPORT.md` §4.

Usage examples:
  # dry-run glue test with AXERA 0.6B package (random text embeddings)
  LD_LIBRARY_PATH=/soc/lib python3 run_breeze_tts.py --preset qwen3tts06 \
      --random-text-embeds 8 --max-frames 8 --out-wav /root/breeze-tts/glue_random.wav

  # Breeze Voice Design (once artifacts are in place)
  LD_LIBRARY_PATH=/soc/lib python3 run_breeze_tts.py --preset breeze \
      --text "今天天气不错，我们出去走走吧。" --instruction "用温柔亲切的女声，语速稍慢。" \
      --max-frames 120 --out-wav /root/breeze-tts/out_zh.wav --dump-tokens /root/breeze-tts/tokens.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import ml_dtypes
from axengine import InferenceSession

BF16 = ml_dtypes.bfloat16
MASK_MIN = -65536.0
SAMPLE_RATE = 24000
UPSAMPLE = 1920          # samples per audio frame @ 24 kHz, 12.5 fps
CODEC_PRE_BUCKET = 325   # fixed bucket of speech_tokenizer_decode_pre_inproj
CODEC_DECODE_POST_CONTEXT = 25


# --------------------------------------------------------------------------------------- utils
def cmm() -> str:
    try:
        for line in open("/proc/ax_proc/mem_cmm_info"):
            if "total size" in line:
                return line.strip()
    except Exception:
        pass
    return ""


def meminfo() -> str:
    d = {}
    try:
        for line in open("/proc/meminfo"):
            p = line.split()
            d[p[0].rstrip(":")] = int(p[1]) // 1024
    except Exception:
        pass
    return "MemTotal=%dMB MemAvail=%dMB" % (d.get("MemTotal", 0), d.get("MemAvailable", 0))


def rss_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def stats() -> str:
    return "rss=%.0fMB %s CMM[%s]" % (rss_mb(), meminfo(), cmm())


def np_dtype_for(name: str, dt: np.dtype) -> np.dtype:
    """axmodel reports real numpy dtypes already (bf16 via ml_dtypes). Keep as-is."""
    return dt


def iolog(tag: str, sess: InferenceSession, groups: int) -> None:
    for g in range(groups):
        print("  [%s] g%d in : %s" % (tag, g, [(n.name, str(n.dtype), tuple(n.shape)) for n in sess.get_inputs(g)]))
        print("  [%s] g%d out: %s" % (tag, g, [(n.name, str(n.dtype), tuple(n.shape)) for n in sess.get_outputs(g)]))


def discover_layers(model_dir: str, prefixes: List[str], suffix: str = "_together.axmodel") -> Dict[int, str]:
    """Find `..._l<N>_together.axmodel` files. prefixes: list of accepted basename prefixes."""
    found: Dict[int, str] = {}
    for path in glob.glob(os.path.join(model_dir, "*" + suffix)):
        base = os.path.basename(path)
        if prefixes and not any(base.startswith(p) for p in prefixes):
            continue
        m = re.search(r"_l(\d+)" + re.escape(suffix), base)
        if not m:
            continue
        found[int(m.group(1))] = path
    return dict(sorted(found.items()))


def run_sync(sess: InferenceSession, feed: Dict[str, np.ndarray], group: int) -> List[np.ndarray]:
    out = sess.run(None, feed, shape_group=group)
    return list(out)


def group_io(sess: InferenceSession, group: int) -> Tuple[Dict[str, Tuple[np.dtype, tuple]], List[str]]:
    ins = {n.name: (n.dtype, tuple(n.shape)) for n in sess.get_inputs(group)}
    outs = [n.name for n in sess.get_outputs(group)]
    return ins, outs


def zeros_for(shape: tuple, dtype: np.dtype) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def bf16_mask(shape: tuple) -> np.ndarray:
    return np.full(shape, MASK_MIN, dtype=BF16)


# --------------------------------------------------------------------------------------- sampling
def apply_repetition_penalty(logits: np.ndarray, ids: List[int], penalty: float) -> np.ndarray:
    """transformers GeneratedTokenRepetitionPenaltyLogitsProcessor semantics."""
    if penalty is None or penalty == 1.0 or not ids:
        return logits
    out = logits.copy()
    uniq = set(int(i) for i in ids)
    for i in uniq:
        if 0 <= i < out.shape[0]:
            v = out[i]
            out[i] = v * penalty if v < 0 else v / penalty
    return out


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 0 or k >= logits.shape[-1]:
        return logits
    out = logits.copy()
    kth = np.partition(out, -k)[-k]
    out[out < kth] = -np.inf
    return out


def apply_suppress(logits: np.ndarray, suppress: np.ndarray) -> np.ndarray:
    out = logits.copy()
    out[suppress] = -np.inf
    return out


def softmax_f32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = np.max(x)
    e = np.exp(x - m)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(e) / e.size
    return e / s


def sample_from(probs: np.ndarray, rng: np.random.Generator, greedy: bool) -> int:
    if greedy:
        return int(np.argmax(probs))
    r = rng.random()
    c = np.cumsum(probs)
    idx = int(np.searchsorted(c, r))
    return min(idx, probs.shape[0] - 1)


@dataclass
class SampleCfg:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    greedy: bool = False


def backbone_sample(logits: np.ndarray, history_cb0: List[int], suppress: np.ndarray,
                    cfg: SampleCfg, rng: np.random.Generator) -> int:
    """Breeze backbone order: repetition_penalty -> temperature -> top_k -> suppress -> softmax -> multinomial."""
    s = logits.astype(np.float32)
    s = apply_repetition_penalty(s, history_cb0, cfg.repetition_penalty)
    if cfg.greedy:
        s = apply_suppress(s, suppress)
        return int(np.argmax(s))
    if cfg.temperature and cfg.temperature != 1.0:
        s = s / cfg.temperature
    s = top_k_filter(s, cfg.top_k)
    s = apply_suppress(s, suppress)
    probs = softmax_f32(s)
    return sample_from(probs, rng, False)


def depth_sample(logits: np.ndarray, suppress: np.ndarray, cfg: SampleCfg, rng: np.random.Generator) -> int:
    """Breeze depth order: suppress -> temperature -> top_k -> softmax -> multinomial."""
    s = logits.astype(np.float32)
    s = apply_suppress(s, suppress)
    if cfg.greedy:
        return int(np.argmax(s))
    if cfg.temperature and cfg.temperature != 1.0:
        s = s / cfg.temperature
    s = top_k_filter(s, cfg.top_k)
    probs = softmax_f32(s)
    return sample_from(probs, rng, False)


# --------------------------------------------------------------------------------------- embedding tables
class EmbedTable:
    """A [rows, dim] bf16/fp32 table mmapped from disk (raw flat bin) or loaded from npy."""

    def __init__(self, path: str, rows: int, dim: int, dtype: str = "bf16", mmap: bool = True):
        self.path = path
        self.rows = int(rows)
        self.dim = int(dim)
        nbytes = os.path.getsize(path)
        need_bf16 = rows * dim * 2
        need_fp32 = rows * dim * 4
        if nbytes == need_bf16:
            self.src_dtype = np.dtype(BF16)
        elif nbytes == need_fp32:
            self.src_dtype = np.dtype(np.float32)
        else:
            # allow only a prefix (e.g. extra padding); else error
            if nbytes >= need_bf16:
                self.src_dtype = np.dtype(BF16)
            elif nbytes >= need_fp32:
                self.src_dtype = np.dtype(np.float32)
            else:
                raise ValueError("embed table %s size %d != %d (bf16) / %d (fp32)" %
                                 (path, nbytes, need_bf16, need_fp32))
        self.mmap = mmap
        if mmap:
            self._arr = np.memmap(path, dtype=self.src_dtype, mode="r")
        else:
            self._arr = np.fromfile(path, dtype=self.src_dtype)
        self._arr = self._arr[: rows * dim]

    def row(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.rows:
            idx = 0
        if self.src_dtype == np.dtype(np.float32):
            return self._arr[idx * self.dim:(idx + 1) * self.dim].astype(np.float32).astype(BF16)
        return np.asarray(self._arr[idx * self.dim:(idx + 1) * self.dim], dtype=BF16)

    def prewarm(self) -> float:
        """Sequentially read the file to warm the page cache (random row lookups otherwise hit eMMC
        at ~1ms/page, which costs ~18ms per generated frame for the 16x2051x2048 talker table)."""
        t0 = time.time()
        try:
            with open(self.path, "rb") as f:
                while True:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        break
        except OSError:
            pass
        return time.time() - t0


class MultiEmbed:
    """N groups of [rows, dim] embedding tables; either one concatenated flat bin (groups*rows*dim)
    or one file per group."""

    def __init__(self, paths: List[str], rows: int, dim: int, dtype: str = "bf16", n_groups: int = 0):
        self.rows = int(rows)
        self.dim = int(dim)
        self.n_groups = int(n_groups)
        if len(paths) == 1 and n_groups and n_groups > 1:
            size = os.path.getsize(paths[0])
            if size == n_groups * rows * dim * 2 or size == n_groups * rows * dim * 4:
                self._concat = n_groups
                self.tables = [EmbedTable(paths[0], n_groups * rows, dim, dtype)]
                self.n_groups = n_groups
                return
        self._concat = 0
        self.tables = [EmbedTable(p, rows, dim, dtype) for p in paths]
        self.n_groups = len(self.tables)

    def __len__(self):
        return self.n_groups

    def row(self, group: int, idx: int) -> np.ndarray:
        if self._concat:
            return self.tables[0].row(group * self.rows + idx)
        return self.tables[group].row(idx)

    def prewarm(self) -> float:
        return float(sum(t.prewarm() for t in self.tables))


class Linear:
    """bf16 weight matmul on CPU.

    layout="out_in": weight stored [out,in] (PyTorch nn.Linear), y = x @ W.T
    layout="in_out": weight stored [in,out] (pre-transposed export), y = x @ W
    """

    def __init__(self, weight_path: str, dim_a: int, dim_b: int, layout: str = "out_in"):
        nbytes = os.path.getsize(weight_path)
        n = dim_a * dim_b
        if nbytes == n * 2:
            dt = np.dtype(BF16)
        elif nbytes == n * 4:
            dt = np.dtype(np.float32)
        else:
            raise ValueError("linear weight %s size %d != %d (bf16) / %d (fp32)"
                             % (weight_path, nbytes, n * 2, n * 4))
        self.layout = layout
        if layout == "out_in":
            self.out_dim, self.in_dim = dim_a, dim_b
            self.w = np.fromfile(weight_path, dtype=dt).reshape(self.out_dim, self.in_dim)
        elif layout == "in_out":
            self.in_dim, self.out_dim = dim_a, dim_b
            self.w = np.fromfile(weight_path, dtype=dt).reshape(self.in_dim, self.out_dim)
        else:
            raise ValueError("bad linear layout %r" % layout)
        self._wf = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._wf is None:
            self._wf = self.w.astype(np.float32)
        x = x.astype(np.float32)
        return x @ self._wf.T if self.layout == "out_in" else x @ self._wf


def infer_linear_layout(path: str, dim_a: int, dim_b: int) -> str:
    """Infer storage layout from the filename (e.g. depth_proj_2048x1024 -> in_out)."""
    base = os.path.basename(path)
    m = re.search(r"_(\d+)x(\d+)[._]", base)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a * b == dim_a * dim_b:
            return "in_out" if (a, b) == (dim_a, dim_b) else "out_in"
    return "out_in"


def autodetect_depth_proj(talker_emb: "MultiEmbed", depth_emb: "MultiEmbed", proj_path: str,
                          in_dim: int, out_dim: int, n_probe: int = 4) -> Optional["Linear"]:
    """Breeze `inputs_embeds_projector` weight layout differs between exports
    ([out,in] row-major vs [in,out] row-major). Detect it numerically by checking which candidate
    reproduces the exported projected embedding table: E' = E @ W.

    Returns a Linear callable (x[in_dim] -> y[out_dim]) or None if detection is not possible.
    """
    if not (proj_path and os.path.exists(proj_path)):
        return None
    try:
        c1 = Linear(proj_path, out_dim, in_dim, layout="out_in")    # y = x @ W.T
        c2 = Linear(proj_path, in_dim, out_dim, layout="in_out")    # y = x @ W
    except Exception:
        return None
    if talker_emb is None or depth_emb is None or len(depth_emb) < 1:
        return c1
    step = max(1, talker_emb.rows // (n_probe + 1))
    cbs = [i * step for i in range(1, n_probe + 1)]
    A = np.stack([talker_emb.row(0, cb).astype(np.float32) for cb in cbs])   # [N, in]
    E = np.stack([depth_emb.row(0, cb).astype(np.float32) for cb in cbs])    # [N, out]
    best = None
    for name, fn, lin in (("out_in", lambda x: x @ lin_w(c1), c1), ("in_out", lambda x: x @ lin_w(c2), c2)):
        try:
            y = np.stack([fn(A[i]) for i in range(len(cbs))])
        except Exception:
            continue
        cosv = float(np.sum(E * y) / (np.linalg.norm(E) * np.linalg.norm(y) + 1e-30))
        if best is None or cosv > best[1]:
            best = (lin, cosv, name)
    if best is None:
        return c1
    print("  depth proj layout auto-detect: %s (cos vs exported E' = %.5f)" % (best[2], best[1]))
    return best[0]


def lin_w(lin):
    """Return effective [in, out] matmul matrix for a Linear."""
    return lin.w.T if lin.layout == "out_in" else lin.w


def find_embed_file(embeds_dir: str, patterns: List[str], expect_bytes: int,
                    allow_size_fallback: bool = True) -> Optional[str]:
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(embeds_dir, pat))):
            if os.path.getsize(p) == expect_bytes:
                return p
    if allow_size_fallback:
        for p in sorted(glob.glob(os.path.join(embeds_dir, "*.bin"))):
            if os.path.getsize(p) == expect_bytes:
                return p
    return None


# --------------------------------------------------------------------------------------- talker
class TalkerState:
    __slots__ = ("kcs", "vcs", "cur")

    def __init__(self, kcs, vcs, cur):
        self.kcs = kcs
        self.vcs = vcs
        self.cur = cur


class TalkerRunner:
    """Qwen3-TTS-style talker: N per-layer axmodels (`*_p{P}_l{L}_together.axmodel`) + post axmodel."""

    def __init__(self, model_dir: str, verbose: bool = False):
        layer_map = discover_layers(model_dir, ["qwen3_tts_talker_", "qwen3_"], "_together.axmodel")
        if not layer_map:
            raise FileNotFoundError("no talker layer axmodels in %s" % model_dir)
        self.layer_paths = list(layer_map.values())
        self.n_layers = len(self.layer_paths)
        post_candidates = [p for p in glob.glob(os.path.join(model_dir, "*post.axmodel"))
                           if not re.search(r"_l\d+_", os.path.basename(p))]
        if not post_candidates:
            raise FileNotFoundError("talker post axmodel not found in %s" % model_dir)
        t0 = time.time()
        self.layers = [InferenceSession(p) for p in self.layer_paths]
        self.post = InferenceSession(post_candidates[0])
        self.load_s = time.time() - t0
        self.n_groups = int(getattr(self.layers[0]._sess, "_shape_count", 1))
        self.g_io = [group_io(self.layers[0], g) for g in range(self.n_groups)]
        g1 = self.g_io[1][0] if self.n_groups > 1 else self.g_io[0][0]
        self.prefill_len = int(g1["input"][1][1])
        self.hidden = int(g1["input"][1][2])
        self.capacity = self.prefill_len * (self.n_groups - 1)
        g0 = self.g_io[0][0]
        self.kv_len = int(g0["K_cache"][1][1])
        self.kv_dim = int(g0["K_cache"][1][2])
        self.decode_mask_len = int(g0["mask"][1][2])
        post_outs = [n.name for n in self.post.get_outputs(0)]
        self.post_has_norm = any("norm" in n for n in post_outs)
        if verbose:
            iolog("talker", self.layers[0], self.n_groups)
            print("  [talker] post: %s" %
                  [(n.name, str(n.dtype), tuple(n.shape)) for n in self.post.get_inputs(0) + self.post.get_outputs(0)])
        # pre-allocated decode mask rows (only last column + history visible)
        self._dmask = np.full((self.kv_len + 1, self.decode_mask_len), MASK_MIN, dtype=BF16)
        for r in range(self.kv_len + 1):
            self._dmask[r, :r] = 0
            self._dmask[r, -1] = 0

    # -- state
    def new_state(self) -> TalkerState:
        kcs = [np.zeros((1, self.kv_len, self.kv_dim), dtype=BF16) for _ in range(self.n_layers)]
        vcs = [np.zeros((1, self.kv_len, self.kv_dim), dtype=BF16) for _ in range(self.n_layers)]
        return TalkerState(kcs, vcs, 0)

    def clone_state(self, st: TalkerState) -> TalkerState:
        return TalkerState([k.copy() for k in st.kcs], [v.copy() for v in st.vcs], st.cur)

    # -- prefill
    def _chunk_feed(self, st: TalkerState, gid: int, start: int, chunk: np.ndarray,
                    valid_in_chunk: int, layer_idx: int) -> Dict[str, np.ndarray]:
        ins, _ = self.g_io[gid]
        P = self.prefill_len
        data = np.zeros(ins["input"][1], dtype=ins["input"][0])
        data[0, :valid_in_chunk, :] = chunk[:valid_in_chunk]
        idx = np.ones(ins["indices"][1], dtype=ins["indices"][0])
        r = np.arange(start, start + P, dtype=np.uint32)
        r[valid_in_chunk:] = 1
        for row in range(idx.shape[0]):
            idx[row] = r
        pcol = ins["mask"][1][2]
        hist = pcol - P
        mask = bf16_mask(ins["mask"][1])
        for row in range(P):
            abs_row = hist + row
            mask[0, row, : min(abs_row, pcol - 1) + 1] = 0
        if gid == 1:
            k_in = np.zeros(ins["K_cache"][1], dtype=ins["K_cache"][0])
            v_in = np.zeros(ins["V_cache"][1], dtype=ins["V_cache"][0])
        else:
            k_in = st.kcs[layer_idx][:, :start, :]
            v_in = st.vcs[layer_idx][:, :start, :]
        return {"input": data, "indices": idx, "mask": mask, "K_cache": k_in, "V_cache": v_in}

    def prefill(self, embeds: np.ndarray, st: Optional[TalkerState] = None) -> Tuple[TalkerState, np.ndarray]:
        """embeds: [T, hidden] (bf16). Returns (state, raw_hidden_last_row[1,1,H])."""
        T = int(embeds.shape[0])
        if T <= 0:
            raise ValueError("empty prompt")
        if T > self.capacity:
            raise ValueError("prompt length %d exceeds talker prefill capacity %d (prefill groups)" % (T, self.capacity))
        st = st or self.new_state()
        nchunk = (T + self.prefill_len - 1) // self.prefill_len
        last = None
        for ci in range(nchunk):
            gid = ci + 1
            start = ci * self.prefill_len
            valid = min(self.prefill_len, T - start)
            chunk = embeds[start:start + self.prefill_len]
            for li, sess in enumerate(self.layers):
                feed = self._chunk_feed(st, gid, start, chunk, valid, li)
                outs = run_sync(sess, feed, gid)
                name_idx = {n: i for i, n in enumerate(self.g_io[gid][1])}
                hidden = outs[name_idx["output"]].astype(BF16)
                k_out = outs[name_idx["K_cache_out"]].reshape(self.prefill_len, self.kv_dim)
                v_out = outs[name_idx["V_cache_out"]].reshape(self.prefill_len, self.kv_dim)
                chunk = hidden[0]
                n = min(self.prefill_len, self.kv_len - start)
                if n > 0:
                    st.kcs[li][0, start:start + n, :] = k_out[:n]
                    st.vcs[li][0, start:start + n, :] = v_out[:n]
                if li == self.n_layers - 1:
                    last = hidden[:, valid - 1:valid, :] if valid > 0 else hidden[:, -1:, :]
            st.cur = start + valid
        return st, np.ascontiguousarray(last.astype(BF16))

    # -- decode one token
    def decode(self, embed: np.ndarray, st: TalkerState) -> np.ndarray:
        if st.cur >= self.kv_len:
            raise RuntimeError("talker KV cache full (%d >= %d): increase --max-frames or rebuild with larger kv_cache_len"
                               % (st.cur, self.kv_len))
        ins, _ = self.g_io[0]
        d = np.ascontiguousarray(embed.reshape(1, 1, self.hidden).astype(BF16))
        indices = np.array([[st.cur]], dtype=ins["indices"][0])
        mask = np.ascontiguousarray(self._dmask[st.cur].reshape(1, 1, self.decode_mask_len))
        for li, sess in enumerate(self.layers):
            outs = run_sync(sess, {"input": d, "indices": indices, "mask": mask,
                                   "K_cache": st.kcs[li], "V_cache": st.vcs[li]}, 0)
            d = outs[2].astype(BF16)
            st.kcs[li][0, st.cur, :] = outs[0].reshape(self.kv_dim)
            st.vcs[li][0, st.cur, :] = outs[1].reshape(self.kv_dim)
        st.cur += 1
        return np.ascontiguousarray(d)

    # -- post (final norm + head)
    def post_hidden(self, raw_hidden: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        outs = list(self.post.run(None, {"input": np.ascontiguousarray(raw_hidden.astype(BF16))}))
        names = [n.name for n in self.post.get_outputs(0)]
        logits = None
        hnorm = None
        for i, name in enumerate(names):
            if "norm" in name:
                hnorm = outs[i].astype(BF16).reshape(self.hidden)
            else:
                logits = outs[i].astype(np.float32).reshape(-1)
        if logits is None:
            logits = outs[0].astype(np.float32).reshape(-1)
        return logits, hnorm


# --------------------------------------------------------------------------------------- code predictor
class CodePredictorRunner:
    """Breeze depth decoder / Qwen3-TTS code_predictor: per-layer axmodels + post + 15 lm_heads."""

    def __init__(self, cp_dir: str, verbose: bool = False, lm_head_bin: Optional[str] = None,
                 head_dir: Optional[str] = None):
        layer_map = discover_layers(cp_dir, ["qwen3_tts_talker_code_predictor_", "llama_"], "_together.axmodel")
        if not layer_map:
            raise FileNotFoundError("no code-predictor layer axmodels in %s" % cp_dir)
        self.layer_paths = list(layer_map.values())
        self.n_layers = len(self.layer_paths)
        post_candidates = [p for p in glob.glob(os.path.join(cp_dir, "*post.axmodel"))
                           if not re.search(r"_l\d+_", os.path.basename(p))]
        if not post_candidates:
            raise FileNotFoundError("code-predictor post not found in %s" % cp_dir)
        head_paths = {}
        head_search_dirs = [d for d in (head_dir, cp_dir) if d and os.path.isdir(d)]
        for p in [x for d in head_search_dirs for x in glob.glob(os.path.join(d, "*.axmodel"))]:
            base = os.path.basename(p)
            if "post" in base or "_together" in base:
                continue
            m = re.search(r"lm_?head_(\d+)\.axmodel$", base) or re.search(r"lm_head_(\d+)\.axmodel$", base)
            if m:
                head_paths[int(m.group(1))] = p
        if not head_paths and not (lm_head_bin and os.path.exists(lm_head_bin)):
            raise FileNotFoundError("no lm_head_{i}.axmodel / code_predictor_lm_head_{i}.axmodel in %s" % cp_dir)
        self.n_heads = len(head_paths) if head_paths else 15
        self.cpu_heads = None
        self.heads = []
        t0 = time.time()
        self.layers = [InferenceSession(p) for p in self.layer_paths]
        self.post = InferenceSession(post_candidates[0])
        if lm_head_bin and os.path.exists(lm_head_bin):
            # CPU lm_head fallback from codebooks_head bin [15, in=1024, out=2051] bf16.
            # Needed when the compiled lm_head axmodels quantize/saturate their output
            # (observed: ~150 distinct logit values, argmax flips vs golden).
            raw = np.memmap(lm_head_bin, dtype=np.uint16, mode="r")
            n = raw.size
            if n == 15 * 1024 * 2051:
                shape = (15, 1024, 2051)
            elif n == 15 * 2051 * 1024:
                shape = (15, 2051, 1024)
            else:
                raise ValueError("unexpected codebooks_head bin size %d" % n)
            tc = time.time()
            u32 = np.ascontiguousarray(raw, dtype=np.uint16).astype(np.uint32) << 16
            self.cpu_heads = u32.view(np.float32).reshape(shape)
            self.cpu_head_path = lm_head_bin
            self.cpu_head_mb = self.cpu_heads.nbytes / 1e6
            print("  [cp] CPU lm_heads preloaded %s %.1f MB in %.1fs"
                  % (shape, self.cpu_head_mb, time.time() - tc))
            self.n_heads = 15
        else:
            self.heads = [InferenceSession(head_paths[k]) for k in sorted(head_paths)]
        self.load_s = time.time() - t0
        self.n_groups = int(getattr(self.layers[0]._sess, "_shape_count", 1))
        self.g_io = [group_io(self.layers[0], g) for g in range(self.n_groups)]
        g1 = self.g_io[1][0] if self.n_groups > 1 else self.g_io[0][0]
        self.prefill_len = int(g1["input"][1][1])
        self.hidden = int(g1["input"][1][2])
        g0 = self.g_io[0][0]
        self.kv_len = int(g0["K_cache"][1][1])
        self.kv_dim = int(g0["K_cache"][1][2])
        self.decode_mask_len = int(g0["mask"][1][2])
        self._dmask = np.full((self.kv_len + 1, self.decode_mask_len), MASK_MIN, dtype=BF16)
        for r in range(self.kv_len + 1):
            self._dmask[r, :r] = 0
            self._dmask[r, -1] = 0
        post_out_names = [n.name for n in self.post.get_outputs(0)]
        self.post_has_norm = any("norm" in n for n in post_out_names)
        if self.cpu_heads is not None:
            self.head_in_dtype = np.float32
            self.head_in_len = 1024
            self.head_vocab = 2051
        else:
            head_in = self.heads[0].get_inputs(0)[0]
            head_out = self.heads[0].get_outputs(0)[0]
            self.head_in_dtype = head_in.dtype
            self.head_in_len = int(head_in.shape[-1])   # hidden dim (e.g. 1024), not the token dim
            self.head_vocab = int(head_out.shape[-1])
        if verbose:
            iolog("cp", self.layers[0], self.n_groups)
            print("  [cp] post in/out: %s / %s" %
                  ([(n.name, str(n.dtype), tuple(n.shape)) for n in self.post.get_inputs(0)],
                   [(n.name, str(n.dtype), tuple(n.shape)) for n in self.post.get_outputs(0)]))
            if self.heads:
                print("  [cp] head0 in/out: %s / %s" %
                      ([(n.name, str(n.dtype), tuple(n.shape)) for n in self.heads[0].get_inputs(0)],
                       [(n.name, str(n.dtype), tuple(n.shape)) for n in self.heads[0].get_outputs(0)]))

    def _prefill(self, pos0: np.ndarray, pos1: np.ndarray):
        """2-position prefill through all layers; returns (kcs, vcs, h_last[1,1,H])."""
        g1 = 1
        ins, outs_names = self.g_io[g1]
        P = self.prefill_len
        data = np.zeros(ins["input"][1], dtype=ins["input"][0])
        data[0, 0, :] = pos0.astype(BF16)
        data[0, 1, :] = pos1.astype(BF16)
        idx = np.zeros(ins["indices"][1], dtype=ins["indices"][0])
        idx[0, 0] = 0
        idx[0, 1] = 1
        mask = bf16_mask(ins["mask"][1])
        mask[0, 0, 0] = 0
        mask[0, 1, :2] = 0
        k_ph = np.zeros(ins["K_cache"][1], dtype=ins["K_cache"][0])
        v_ph = np.zeros(ins["V_cache"][1], dtype=ins["V_cache"][0])
        kcs = [np.zeros((1, self.kv_len, self.kv_dim), dtype=BF16) for _ in range(self.n_layers)]
        vcs = [np.zeros((1, self.kv_len, self.kv_dim), dtype=BF16) for _ in range(self.n_layers)]
        for li, sess in enumerate(self.layers):
            outs = run_sync(sess, {"input": data, "indices": idx, "mask": mask,
                                   "K_cache": k_ph, "V_cache": v_ph}, g1)
            name_idx = {n: i for i, n in enumerate(outs_names)}
            data = outs[name_idx["output"]].astype(BF16)
            ko = outs[name_idx["K_cache_out"]].reshape(P, self.kv_dim)
            vo = outs[name_idx["V_cache_out"]].reshape(P, self.kv_dim)
            kcs[li][0, :min(P, self.kv_len), :] = ko[:min(P, self.kv_len)]
            vcs[li][0, :min(P, self.kv_len), :] = vo[:min(P, self.kv_len)]
        return kcs, vcs, np.ascontiguousarray(data[:, 1:2, :])

    def _decode(self, embed: np.ndarray, kcs, vcs, cur: int) -> np.ndarray:
        d = np.ascontiguousarray(embed.reshape(1, 1, self.hidden).astype(BF16))
        ins, _ = self.g_io[0]
        indices = np.array([[cur]], dtype=ins["indices"][0])
        mask = np.ascontiguousarray(self._dmask[cur].reshape(1, 1, self.decode_mask_len))
        for li, sess in enumerate(self.layers):
            outs = run_sync(sess, {"input": d, "indices": indices, "mask": mask,
                                   "K_cache": kcs[li], "V_cache": vcs[li]}, 0)
            d = outs[2].astype(BF16)
            kcs[li][0, cur, :] = outs[0].reshape(self.kv_dim)
            vcs[li][0, cur, :] = outs[1].reshape(self.kv_dim)
        return np.ascontiguousarray(d)

    def _norm(self, h: np.ndarray) -> np.ndarray:
        outs = list(self.post.run(None, {"input": np.ascontiguousarray(h.astype(BF16))}))
        names = [n.name for n in self.post.get_outputs(0)]
        for i, name in enumerate(names):
            if "norm" in name:
                return outs[i].astype(BF16).reshape(self.hidden)
        return outs[-1].astype(BF16).reshape(self.hidden)

    def _head_logits(self, step: int, hnorm: np.ndarray) -> np.ndarray:
        if self.cpu_heads is not None:
            w = self.cpu_heads[step]
            x = np.asarray(hnorm, dtype=np.float32).reshape(-1)
            return x @ w
        ins = self.heads[step].get_inputs(0)[0]
        x = np.zeros(tuple(ins.shape), dtype=ins.dtype)
        n = min(x.shape[-1], hnorm.shape[0])
        x[0, 0, :n] = hnorm[:n].astype(ins.dtype)
        outs = list(self.heads[step].run(None, {"input": x}))
        return outs[0].astype(np.float32).reshape(-1)

    def generate(self, hidden_norm: np.ndarray, cb0: int, emb_tables: "MultiEmbed",
                 suppress: np.ndarray, cfg: SampleCfg, rng: np.random.Generator,
                 proj: Optional[Linear] = None, debug: Optional[Dict[str, np.ndarray]] = None,
                 force_codes: Optional[List[int]] = None) -> List[int]:
        """hidden_norm: talker post-norm hidden [H_talker]. Returns cb1..cb15 codes."""
        pos0 = proj(hidden_norm) if proj is not None else hidden_norm
        pos1 = emb_tables.row(0, cb0)
        kcs, vcs, h = self._prefill(pos0, pos1)
        cur = 2
        out: List[int] = []
        for step in range(self.n_heads):
            hnorm = self._norm(h)
            logits = self._head_logits(step, hnorm)
            if debug is not None:
                debug.setdefault("depth_logits", []).append(logits.astype(np.float32).copy())
                if step == 0:
                    debug["depth0_logits"] = logits.astype(np.float32).copy()
                    debug["depth0_norm"] = np.asarray(hnorm, dtype=np.float32).reshape(-1).copy()
            code = depth_sample(logits, suppress, cfg, rng)
            out.append(int(code))
            if step < self.n_heads - 1:
                feed_code = int(force_codes[step]) if force_codes is not None else int(code)
                emb = emb_tables.row(step + 1, feed_code)
                h = self._decode(emb, kcs, vcs, cur)
                cur += 1
        return out


# --------------------------------------------------------------------------------------- codec
class SpeechTokenizerDecoder:
    """AXERA Qwen3-TTS speech_tokenizer decode: decode_pre_inproj -> 8-layer transformer -> post -> vocoder."""

    def __init__(self, codec_dir: str, verbose: bool = False):
        self.dir = codec_dir
        pre_path = os.path.join(codec_dir, "speech_tokenizer_decode_pre_inproj.axmodel")
        if not os.path.exists(pre_path):
            raise FileNotFoundError(pre_path)
        dec_dir = os.path.join(codec_dir, "decoder")
        layer_map = discover_layers(dec_dir, ["qwen3_tts_tokenizer_12hz_"], "_together.axmodel")
        if not layer_map:
            raise FileNotFoundError("no codec decoder layers in %s" % dec_dir)
        self.layer_paths = list(layer_map.values())
        self.n_layers = len(self.layer_paths)
        post_candidates = glob.glob(os.path.join(dec_dir, "*_post.axmodel"))
        voc_candidates = glob.glob(os.path.join(codec_dir, "*decode_post_t*.axmodel"))
        if not post_candidates or not voc_candidates:
            raise FileNotFoundError("codec post / decode_post axmodel missing in %s" % codec_dir)
        t0 = time.time()
        self.pre = InferenceSession(pre_path)
        self.layers = [InferenceSession(p) for p in self.layer_paths]
        self.post = InferenceSession(post_candidates[0])
        self.voc = InferenceSession(voc_candidates[0])
        self.load_s = time.time() - t0
        pre_in = self.pre.get_inputs(0)[0]
        pre_out = self.pre.get_outputs(0)[0]
        self.pre_len = int(pre_in.shape[2])          # 325
        self.n_codebooks = int(pre_in.shape[1])      # 16
        self.dec_hidden = int(pre_out.shape[2])      # 512
        self.n_groups = int(getattr(self.layers[0]._sess, "_shape_count", 1))
        self.g_io = [group_io(self.layers[0], g) for g in range(self.n_groups)]
        g1 = self.g_io[1][0] if self.n_groups > 1 else self.g_io[0][0]
        self.prefill_len = int(g1["input"][1][1])    # 64
        g0 = self.g_io[0][0]
        self.kv_len = int(g0["K_cache"][1][1])       # 127
        self.kv_dim = int(g0["K_cache"][1][2])       # 1024
        self.decode_mask_len = int(g0["mask"][1][2])  # 128
        self._dmask_cache = None
        voc_in = self.voc.get_inputs(0)[0]
        self.voc_len = int(voc_in.shape[1])          # 64
        self.voc_hidden = int(voc_in.shape[2])       # 1024
        self.voc_samples = int(np.prod(self.voc.get_outputs(0)[0].shape))
        post_out = self.post.get_outputs(0)[0]
        self.post_hidden = int(post_out.shape[-1])
        if verbose:
            print("  [codec] pre in/out: %s / %s" %
                  ([(n.name, str(n.dtype), tuple(n.shape)) for n in self.pre.get_inputs(0)],
                   [(n.name, str(n.dtype), tuple(n.shape)) for n in self.pre.get_outputs(0)]))
            iolog("codec", self.layers[0], self.n_groups)
            print("  [codec] post out: %s" % [(n.name, str(n.dtype), tuple(n.shape)) for n in self.post.get_outputs(0)])
            print("  [codec] voc in/out: %s / %s" %
                  ([(n.name, str(n.dtype), tuple(n.shape)) for n in self.voc.get_inputs(0)],
                   [(n.name, str(n.dtype), tuple(n.shape)) for n in self.voc.get_outputs(0)]))

    def _mask_for(self, pos: int) -> np.ndarray:
        """codec transformer is sliding-window-72; decode mask: visible cols [pos-71, pos-1] + last col."""
        m = bf16_mask((1, 1, self.decode_mask_len))
        left = max(0, pos - 72 + 1)
        for i in range(left, pos):
            if i + 1 < self.decode_mask_len:
                m[0, 0, i] = 0
        m[0, 0, -1] = 0
        return m

    def decode(self, codes: np.ndarray, dump=None) -> np.ndarray:
        """codes: [T,16] int32; returns float32 wav at 24kHz."""
        if codes.ndim != 2 or codes.shape[1] != self.n_codebooks:
            raise ValueError("codes must be [T,16], got %s" % (codes.shape,))
        T = int(codes.shape[0])
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if T > self.pre_len:
            raise ValueError("T=%d exceeds codec decode_pre bucket %d" % (T, self.pre_len))
        if T > self.kv_len:
            raise ValueError("T=%d exceeds codec transformer kv_cache_len %d (~%.1fs audio)" %
                             (T, self.kv_len, self.kv_len / 12.5))
        # 1) decode_pre: q-major flat [1,16,325]
        feed = np.zeros((1, self.n_codebooks, self.pre_len), dtype=np.int32)
        for q in range(self.n_codebooks):
            feed[0, q, :T] = codes[:, q]
        hidden = self.pre.run(None, {"codes": feed})[0]
        hidden = hidden[0, :T, :].astype(np.float32)     # [T,512]
        # 2) 8-layer transformer
        P = self.prefill_len
        valid = min(T, P)
        ins1 = self.g_io[1][0]
        data = np.zeros((1, P, self.dec_hidden), dtype=BF16)
        data[0, :valid, :] = hidden[:valid].astype(BF16)
        idx = np.arange(P, dtype=ins1["indices"][0]).reshape(1, P)
        pcol = int(ins1["mask"][1][2])
        hist = pcol - P
        mask = bf16_mask((1, P, pcol))
        for r in range(P):
            abs_row = hist + r
            left = max(0, abs_row - 72 + 1)
            mask[0, r, left:min(abs_row, pcol - 1) + 1] = 0
        k_zero = np.zeros(ins1["K_cache"][1], dtype=ins1["K_cache"][0])
        v_zero = np.zeros(ins1["V_cache"][1], dtype=ins1["V_cache"][0])
        kcs = [np.zeros((1, self.kv_len, self.kv_dim), dtype=BF16) for _ in range(self.n_layers)]
        vcs = [np.zeros((1, self.kv_len, self.kv_dim), dtype=BF16) for _ in range(self.n_layers)]
        outs_names = self.g_io[1][1]
        for li, sess in enumerate(self.layers):
            outs = run_sync(sess, {"input": data, "indices": idx, "mask": mask,
                                   "K_cache": k_zero, "V_cache": v_zero}, 1)
            ni = {n: i for i, n in enumerate(outs_names)}
            data = outs[ni["output"]].astype(BF16)
            ko = outs[ni["K_cache_out"]].reshape(P, self.kv_dim)
            vo = outs[ni["V_cache_out"]].reshape(P, self.kv_dim)
            n = min(P, self.kv_len)
            kcs[li][0, :n, :] = ko[:n]
            vcs[li][0, :n, :] = vo[:n]
        frames = [data[0, i, :].copy() for i in range(valid)]
        # decode the remaining frames one by one
        for pos in range(P, T):
            d = hidden[pos:pos + 1, :].astype(BF16).reshape(1, 1, self.dec_hidden)
            indices = np.array([[pos]], dtype=int)
            indices = indices.astype(self.g_io[0][0]["indices"][0])
            dm = np.ascontiguousarray(self._mask_for(pos))
            for li, sess in enumerate(self.layers):
                outs = run_sync(sess, {"input": d, "indices": indices, "mask": dm,
                                       "K_cache": kcs[li], "V_cache": vcs[li]}, 0)
                d = outs[2].astype(BF16)
                kcs[li][0, pos, :] = outs[0].reshape(self.kv_dim)
                vcs[li][0, pos, :] = outs[1].reshape(self.kv_dim)
            frames.append(d[0, 0, :].copy())
        # 3) post per frame: [1,1,512] -> [1,1,1024]
        hid1024 = np.empty((T, self.post_hidden), dtype=np.float32)
        for i in range(T):
            h = np.ascontiguousarray(np.asarray(frames[i], dtype=BF16).reshape(1, 1, self.dec_hidden))
            o = self.post.run(None, {"input": h})[0]
            hid1024[i] = o.reshape(-1)[: self.post_hidden].astype(np.float32)
        # 4) vocoder chunks with left context 25
        wav_parts = []
        start = 0
        while start < T:
            context = min(CODEC_DECODE_POST_CONTEXT, start)
            capacity = self.voc_len - context
            end = min(T, start + capacity)
            feed = np.zeros((1, self.voc_len, self.voc_hidden), dtype=np.float32)
            seg = hid1024[start - context:end]
            feed[0, : seg.shape[0], :] = seg
            w = self.voc.run(None, {"hidden": feed})[0].reshape(-1).astype(np.float32)
            skip = context * UPSAMPLE
            take = (end - start) * UPSAMPLE
            if w.size > skip:
                wav_parts.append(w[skip:skip + min(take, w.size - skip)])
            start = end
        wav = np.concatenate(wav_parts) if wav_parts else np.zeros(0, dtype=np.float32)
        out = np.zeros(T * UPSAMPLE, dtype=np.float32)
        n = min(out.size, wav.size)
        out[:n] = wav[:n]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------------------- text encoder
class TextEncoderRunner:
    """Breeze T5Gemma2 text encoder axmodel(s), with CPU-side Breeze embedding lookup.

    Breeze semantics (mirrors T5Gemma2TextScaledWordEmbedding.forward, see
    `textenc/scripts/text_embed_cpu.py` and `extract_textenc_weights.py`):
        emb = table[token] * sqrt(hidden=1152)      # embed_scale
        if token == eoi_token_index: emb = eoi_embedding   # unscaled
    The ONNX/axmodel takes `inputs_embeds [1,P,1152] float32` + `attention_mask [1,P] float32`
    and outputs `text_embeds [1,P,2048]` (second split stage contains proj + final norm).
    """

    def __init__(self, textenc_dir: str, embed_bin: str, embed_vocab: int = 262158,
                 embed_dim: int = 1152, verbose: bool = False):
        self.dir = textenc_dir
        self.embed_vocab = int(embed_vocab)
        self.embed_dim = int(embed_dim)
        self.embed_scale = float(embed_dim) ** 0.5
        self.eoi_index = -1
        self.eoi_embedding = None
        meta_path = os.path.join(textenc_dir, "embedding_meta.json")
        if os.path.exists(meta_path):
            m = json.load(open(meta_path))
            self.embed_scale = float(m.get("embed_scale", self.embed_scale))
            self.eoi_index = int(m.get("eoi_token_index", -1))
            self.embed_vocab = int(m.get("vocab_size", self.embed_vocab))
            self.embed_dim = int(m.get("hidden_size", self.embed_dim))
            eoi_file = os.path.join(textenc_dir, m.get("eoi_embedding_file", "eoi_embedding.npy"))
            if self.eoi_index >= 0 and os.path.exists(eoi_file):
                self.eoi_embedding = np.load(eoi_file).astype(np.float32).reshape(-1)
        need = self.embed_vocab * self.embed_dim * 2
        got = os.path.getsize(embed_bin)
        if got != need:
            raise ValueError("text embedding bin %s size %d != %d (vocab=%d dim=%d bf16)"
                             % (embed_bin, got, need, self.embed_vocab, self.embed_dim))
        self._raw = np.memmap(embed_bin, dtype=np.uint16, mode="r",
                              shape=(self.embed_vocab, self.embed_dim))
        model_paths = sorted(glob.glob(os.path.join(textenc_dir, "*.axmodel")))
        if not model_paths:
            raise FileNotFoundError("no text encoder axmodel in %s" % textenc_dir)
        self.model_paths = model_paths
        t0 = time.time()
        self.models = [InferenceSession(p) for p in model_paths]
        self.load_s = time.time() - t0
        self.g_io = [group_io(m, 0) for m in self.models]
        self.bucket = None
        for ins, _ in self.g_io:
            for name, (dt, shape) in ins.items():
                if len(shape) == 3 and "mask" not in name.lower():
                    self.bucket = int(shape[1])
        if verbose:
            for pth, m in zip(model_paths, self.models):
                print("  [textenc] %s" % os.path.basename(pth))
                iolog("textenc", m, 1)

    @staticmethod
    def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
        u32 = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
        return u32.view(np.float32)

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.embed_vocab):
            raise ValueError("token id out of range [0,%d): %d..%d"
                             % (self.embed_vocab, ids.min(), ids.max()))
        rows = np.asarray(self._raw[ids], dtype=np.uint16)
        emb = self._bf16_bits_to_f32(rows) * np.float32(self.embed_scale)
        if self.eoi_index >= 0 and self.eoi_embedding is not None:
            m = ids == self.eoi_index
            if m.any():
                emb = emb.copy()
                emb[m] = self.eoi_embedding
        return emb

    def encode(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        L = int(ids.shape[0])
        bucket = self.bucket or 128
        if L > bucket:
            raise ValueError("prompt length %d exceeds text encoder bucket %d" % (L, bucket))
        embeds = np.zeros((1, bucket, self.embed_dim), dtype=np.float32)
        mask = np.zeros((1, bucket), dtype=np.float32)
        embeds[0, :L, :] = self.lookup(ids)
        mask[0, :L] = 1.0
        cur = embeds
        for m, (ins, outs) in zip(self.models, self.g_io):
            feed = {}
            for name, (dt, shape) in ins.items():
                low = name.lower()
                if "mask" in low:
                    feed[name] = mask.astype(dt)
                elif len(shape) == 3:
                    feed[name] = np.ascontiguousarray(cur.astype(dt))
                elif len(shape) == 2:
                    x = np.zeros(shape, dtype=dt)
                    x[0, :] = np.arange(shape[1], dtype=np.float32).astype(dt)
                    feed[name] = x
                elif len(shape) == 1:
                    feed[name] = np.zeros(shape, dtype=dt)
                else:
                    feed[name] = np.zeros(shape, dtype=dt)
            o = list(m.run(None, feed))
            cur = o[0]
            if cur.ndim == 3:
                cur = cur[0]
            cur = cur.astype(np.float32)
        if cur.shape[0] < L:
            raise RuntimeError("text encoder output has %d rows < prompt %d" % (cur.shape[0], L))
        return cur[:L].astype(BF16)


# --------------------------------------------------------------------------------------- presets
@dataclass
class Preset:
    name: str
    talker_dir: str
    cp_dir: str
    cp_hp_dir: str
    lmhead_hp_dir: str
    codec_dir: str
    talker_hidden: int
    n_codebooks: int = 16
    codebook_size: int = 2048
    talker_vocab: int = 2052
    audio_vocab: int = 2051           # rows per embedding group
    eos_id: int = 2051
    suppress_ids: List[int] = field(default_factory=lambda: [2048, 2049, 2050])
    talker_embed_mode: str = "sum16"  # "sum16" (breeze) | "single" (qwen3tts)
    talker_embed_paths: List[str] = field(default_factory=list)
    depth_embed_paths: List[str] = field(default_factory=list)
    depth_embed_groups: int = 0
    depth_proj_path: Optional[str] = None
    depth_proj_shape: Tuple[int, int] = (1024, 2048)
    textenc_dir: Optional[str] = None
    textenc_embed_bin: Optional[str] = None
    textenc_embed_vocab: int = 262158
    textenc_embed_dim: int = 1152
    tokenizer_json: Optional[str] = None
    embeds_dir: str = ""
    cp_vocab: int = 2051
    cp_suppress: List[int] = field(default_factory=lambda: [2048, 2049, 2050])


def preset_qwen3tts06() -> Preset:
    root = "/root/breeze-tts/qwen3tts-ref"
    cp_dir = os.path.join(root, "code-predictor")
    talker_dir = os.path.join(root, "talker")
    depth_paths = sorted(glob.glob(os.path.join(cp_dir, "talker.code_predictor.model.codec_embedding.*.weight.bfloat16.bin")),
                         key=lambda p: int(re.search(r"embedding\.(\d+)\.", p).group(1)))
    return Preset(
        name="qwen3tts06",
        talker_dir=talker_dir,
        cp_dir=cp_dir,
        cp_hp_dir="",
        lmhead_hp_dir="",
        codec_dir=os.path.join(root, "speech_tokenizer"),
        talker_hidden=1024,
        talker_vocab=3072,
        audio_vocab=3072,
        eos_id=2150,
        suppress_ids=None,   # filled below (ids outside 0..2047 except EOS)
        talker_embed_mode="single",
        talker_embed_paths=[os.path.join(talker_dir, "talker.model.codec_embedding.weight.bfloat16.bin")],
        depth_embed_paths=depth_paths,
        cp_vocab=2048,
        cp_suppress=[],
    )


def _first_existing(*paths: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0] if paths else ""


def looks_like_breeze_package(base: str) -> bool:
    """HF package layout (<base>/talker, <base>/code-predictor, <base>/embeds) or board layout (<base>/builds/...)."""
    if not base:
        return False
    for rel in ("talker", "code-predictor", "embeds",
                "builds/talker", "builds/code-predictor"):
        if os.path.isdir(os.path.join(base, rel)):
            return True
    return False


def preset_breeze(base: str = "/root/breeze-tts/breeze") -> Preset:
    # support both layouts: HF package (<base>/talker, <base>/code-predictor, ...) and
    # board workspace (<base>/builds/talker, ...)
    talker_dir = _first_existing(os.path.join(base, "builds/talker"), os.path.join(base, "talker"))
    cp_dir = _first_existing(os.path.join(base, "builds/code-predictor"), os.path.join(base, "code-predictor"))
    cp_hp_dir = _first_existing(os.path.join(base, "builds/code-predictor-hp"),
                                os.path.join(base, "code-predictor-hp"))
    lmhead_hp_dir = _first_existing(os.path.join(base, "lm-head-hp"),
                                    os.path.join(base, "builds/lm-head-hp"))
    embeds_dir = os.path.join(base, "embeds")
    # talker audio embedding tables: 16 x 2051 x 2048 bf16 (one flat bin is fine)
    talker_embed = find_embed_file(embeds_dir, [
        "audio_embed_16x2051x2048*.bin", "audio_embeds_16x2051x2048*.bin",
        "talker_audio_embeds*.bin"], 16 * 2051 * 2048 * 2, allow_size_fallback=False) or \
        find_embed_file(embeds_dir, ["*.bin"], 16 * 2051 * 2048 * 2) or \
        os.path.join(embeds_dir, "audio_embed_16x2051x2048.bf16.bin")
    # depth input tables: projected E' (1024-dim), 15 groups (repack) or 16 groups (lead export)
    depth_embed = None
    depth_embed_groups = 0
    for pat, ng in (("codec_embed_proj_15x2051x1024*.bin", 15),
                    ("audio_embeds_proj_15x2051x1024*.bin", 15)):
        p15 = find_embed_file(embeds_dir, [pat], ng * 2051 * 1024 * 2, allow_size_fallback=False)
        if p15:
            depth_embed, depth_embed_groups = p15, ng
            break
    if depth_embed is None:
        p16 = find_embed_file(embeds_dir, ["proj_audio_embed_16x2051x1024*.bin"],
                              16 * 2051 * 1024 * 2, allow_size_fallback=False)
        if p16:
            depth_embed, depth_embed_groups = p16, 16
    if depth_embed is None:
        depth_embed = os.path.join(embeds_dir, "codec_embed_proj_15x2051x1024.bf16.bin")
    # depth proj: 2048->1024, stored [out,in] or [in,out]
    proj = find_embed_file(embeds_dir, [
        "inputs_embeds_projector_1024x2048*.bin", "depth_proj_2048x1024*.bin",
        "depth_proj_1024x2048*.bin"], 1024 * 2048 * 2) or \
        os.path.join(embeds_dir, "inputs_embeds_projector_1024x2048.bf16.bin")
    # speech tokenizer: HF package (<base>/speech-tokenizer) or the old AXERA package path
    codec_dir = _first_existing(os.path.join(base, "speech-tokenizer"),
                                "/root/breeze-tts/qwen3tts-ref/speech_tokenizer")
    # text encoder: HF package (<base>/text-encoder) or board workspace (<base>/textenc)
    textenc_dir = _first_existing(os.path.join(base, "text-encoder"),
                                  os.path.join(base, "textenc"))
    textenc_embed_bin = _first_existing(
        os.path.join(textenc_dir, "text_embed_262158x1152.bfloat16.bin"),
        os.path.join(base, "textenc/text_embed_262158x1152.bfloat16.bin"))
    tokenizer_json = _first_existing(
        os.path.join(textenc_dir, "tokenizer/tokenizer.json"),
        os.path.join(base, "tokenizer/tokenizer.json"),
        os.path.join(base, "text-encoder/tokenizer/tokenizer.json"))
    return Preset(
        name="breeze",
        talker_dir=talker_dir,
        cp_dir=cp_dir,
        cp_hp_dir=cp_hp_dir,
        lmhead_hp_dir=lmhead_hp_dir,
        codec_dir=codec_dir,
        embeds_dir=embeds_dir,
        talker_hidden=2048,
        talker_vocab=2052,
        audio_vocab=2051,
        eos_id=2051,
        suppress_ids=[2048, 2049, 2050],
        talker_embed_mode="sum16",
        talker_embed_paths=[talker_embed],
        depth_embed_paths=[depth_embed],
        depth_embed_groups=depth_embed_groups,
        depth_proj_path=proj,
        depth_proj_shape=(1024, 2048),
        textenc_dir=textenc_dir,
        textenc_embed_bin=textenc_embed_bin,
        tokenizer_json=tokenizer_json,
    )


def suppress_array(vocab: int, ids: List[int]) -> np.ndarray:
    mask = np.zeros(vocab, dtype=bool)
    for i in ids:
        if 0 <= i < vocab:
            mask[i] = True
    return mask


def qwen3tts_suppress(vocab: int = 3072, keep_max: int = 2047, eos: int = 2150) -> np.ndarray:
    mask = np.ones(vocab, dtype=bool)
    mask[: keep_max + 1] = False
    mask[eos] = False
    return mask


# --------------------------------------------------------------------------------------- tokenizer
class PromptTokenizer:
    def __init__(self, tokenizer_json: str):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(tokenizer_json)
        self.bos_id = 2

    def encode(self, text: str, instruction: str = "", speaker: str = "S0") -> np.ndarray:
        prefix = ("[%s]" % speaker) if speaker else ""
        s = prefix + "<ins_bos>" + instruction + "<ins_eos>" + text
        ids = self.tok.encode(s, add_special_tokens=False).ids
        return np.asarray([self.bos_id] + ids, dtype=np.int64)


# --------------------------------------------------------------------------------------- wav io
def save_wav(path: str, wav: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    import struct
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------------------- main loop
def synthesize(args) -> dict:
    if args.preset == "qwen3tts06":
        preset = preset_qwen3tts06()
    else:
        base = args.breeze_base or ""
        if not base:
            base = "." if looks_like_breeze_package(".") else "/root/breeze-tts/breeze"
        preset = preset_breeze(base)
        print("breeze base: %s" % os.path.abspath(base))
        if args.lmhead_hp and preset.lmhead_hp_dir and os.path.isdir(preset.lmhead_hp_dir):
            print("using U16 lm_heads from %s" % preset.lmhead_hp_dir)
    if args.talker_dir:
        preset.talker_dir = args.talker_dir
    if args.cp_dir:
        preset.cp_dir = args.cp_dir
    if args.depth_mode == "hp" and preset.cp_hp_dir:
        preset.cp_dir = preset.cp_hp_dir
        print("depth mode: quality (hp) -> %s" % preset.cp_hp_dir)
    if args.codec_dir:
        preset.codec_dir = args.codec_dir
    if args.dry_run:
        print("=== dry-run preset resolution ===")
        print("preset              :", preset.name)
        print("breeze_base         :", getattr(args, "breeze_base", "") or "(auto)")
        print("talker_dir          :", preset.talker_dir, "exists=", os.path.isdir(preset.talker_dir))
        print("cp_dir (depth)      :", preset.cp_dir, "exists=", os.path.isdir(preset.cp_dir))
        print("cp_hp_dir           :", preset.cp_hp_dir, "exists=", os.path.isdir(preset.cp_hp_dir))
        print("lmhead_hp_dir       :", preset.lmhead_hp_dir, "exists=", os.path.isdir(preset.lmhead_hp_dir))
        print("codec_dir           :", preset.codec_dir, "exists=", os.path.isdir(preset.codec_dir))
        print("embeds_dir          :", preset.embeds_dir, "exists=", os.path.isdir(preset.embeds_dir))
        print("talker embeds       :", preset.talker_embed_paths)
        print("depth embeds        :", preset.depth_embed_paths, "groups=", preset.depth_embed_groups)
        print("depth proj          :", preset.depth_proj_path)
        print("textenc_dir         :", preset.textenc_dir, "exists=", os.path.isdir(preset.textenc_dir))
        print("textenc embed bin   :", preset.textenc_embed_bin,
              "exists=", os.path.exists(preset.textenc_embed_bin or ""))
        print("tokenizer.json      :", preset.tokenizer_json,
              "exists=", os.path.exists(preset.tokenizer_json or ""))
        for name in ("talker_embed_paths", "depth_embed_paths"):
            for p in getattr(preset, name):
                print("  %s: %s exists=%s size=%s"
                      % (name, p, os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else -1))
        return {"ok": True}

    greedy = bool(args.greedy)
    cfg = SampleCfg(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty, greedy=greedy)
    rng = np.random.default_rng(args.seed)
    timings: Dict[str, float] = {}
    t_start = time.time()
    print("preset=%s greedy=%s max_frames=%d stats: %s" % (preset.name, greedy, args.max_frames, stats()))

    # ---- codec-only mode: decode given codes (no talker/depth) ----
    if args.codes:
        z = np.load(args.codes, allow_pickle=False)
        codes = None
        for k in ("codes", "tokens", "frames", "audio_codes"):
            if k in z.files:
                codes = np.asarray(z[k])
                break
        if codes is None:
            raise SystemExit("%s has no codes/tokens array (keys=%s)" % (args.codes, list(z.files)))
        if codes.ndim == 3:
            codes = codes[0]
        if codes.ndim == 2 and codes.shape[0] == 16 and codes.shape[1] != 16:
            codes = codes.T
        codes = np.asarray(codes, dtype=np.int32)
        if codes.ndim != 2 or codes.shape[1] != 16:
            raise SystemExit("codes shape %s != [T,16]" % (codes.shape,))
        print("codec-only mode: codes %s (from %s)" % (codes.shape, args.codes))
        t0 = time.time()
        codec = SpeechTokenizerDecoder(preset.codec_dir, verbose=args.verbose)
        timings["load_codec_s"] = time.time() - t0
        t0 = time.time()
        wav = codec.decode(codes)
        timings["codec_decode_s"] = time.time() - t0
        out_wav = args.out_wav or "/root/breeze-tts/out/codec_decoded.wav"
        save_wav(out_wav, wav)
        print("codec-only: %d frames -> %d samples (%.2fs) in %.0f ms; RTF %.3f"
              % (codes.shape[0], wav.size, wav.size / SAMPLE_RATE, timings["codec_decode_s"] * 1000,
                 timings["codec_decode_s"] / (codes.shape[0] / 12.5)))
        print("saved wav:", out_wav)
        if args.dump_tokens:
            np.savez(args.dump_tokens, codes=codes,
                     meta=np.asarray(json.dumps({"mode": "codec_only", "codes_file": args.codes,
                                                 "timings": {k: float(v) for k, v in timings.items()}},
                                                ensure_ascii=False)))
        print("wav stats: min=%.3f max=%.3f rms=%.4f" %
              (float(wav.min()), float(wav.max()), float(np.sqrt((wav.astype(np.float64) ** 2).mean()))))
        print("final stats:", stats())
        return timings

    # ---- prompt -> text embeddings FIRST (textenc axmodels can be huge; load, encode, release)
    tokens = None
    embeds = None
    t0 = time.time()
    if args.random_text_embeds:
        L = int(args.random_text_embeds)
        embeds = rng.standard_normal((L, preset.talker_hidden)).astype(np.float32).astype(BF16)
        timings["textenc_s"] = time.time() - t0
        print("random text embeds: L=%d H=%d (glue test)" % (L, preset.talker_hidden))
    elif args.text_embeds:
        z = np.load(args.text_embeds, allow_pickle=False)
        arr = None
        for k in ("text_embeds", "par_text_embeds", "embeddings", "embeds", "text_encoder_out"):
            if k in z.files:
                arr = np.asarray(z[k])
                break
        if arr is None:
            raise SystemExit("%s has no text_embeds-like array (keys=%s)" % (args.text_embeds, list(z.files)))
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape[-1] != preset.talker_hidden:
            raise SystemExit("text embeds dim %d != talker hidden %d" % (arr.shape[-1], preset.talker_hidden))
        embeds = arr.astype(np.float32).astype(BF16)
        if "prompt_ids" in z.files:
            tokens = np.asarray(z["prompt_ids"]).reshape(-1).astype(np.int64)
        timings["textenc_s"] = time.time() - t0
        print("loaded text embeds from %s: %s (skip text encoder axmodel)" %
              (args.text_embeds, tuple(embeds.shape)))
    else:
        if not preset.tokenizer_json:
            raise SystemExit("preset has no tokenizer; use --random-text-embeds/--text-embeds for glue test")
        if args.prompt_ids:
            if os.path.exists(args.prompt_ids):
                if args.prompt_ids.endswith(".npy"):
                    tokens = np.load(args.prompt_ids).astype(np.int64).reshape(-1)
                elif args.prompt_ids.endswith(".npz"):
                    zz = np.load(args.prompt_ids)
                    tokens = np.asarray(zz["prompt_ids"], dtype=np.int64).reshape(-1)
                else:
                    jd = json.load(open(args.prompt_ids))
                    if isinstance(jd, dict) and "prompt_ids" in jd:
                        ids = jd["prompt_ids"]
                    elif isinstance(jd, dict) and isinstance(jd.get("prompt"), dict) and "ids" in jd["prompt"]:
                        ids = jd["prompt"]["ids"]
                    elif isinstance(jd, dict) and "ids" in jd:
                        ids = jd["ids"]
                    else:
                        raise SystemExit("no prompt_ids/prompt.ids/ids in %s" % args.prompt_ids)
                    tokens = np.asarray(ids, dtype=np.int64).reshape(-1)
            else:
                tokens = np.asarray([int(x) for x in args.prompt_ids.replace(" ", "").split(",") if x],
                                    dtype=np.int64)
            print("using explicit prompt ids: L=%d" % len(tokens))
        else:
            tok = PromptTokenizer(preset.tokenizer_json)
            tokens = tok.encode(args.text, args.instruction, args.speaker)
        print("prompt ids (L=%d): %s" % (len(tokens), tokens.tolist()[:40]))
        print("  before textenc:", stats())
        te = TextEncoderRunner(preset.textenc_dir, preset.textenc_embed_bin,
                               preset.textenc_embed_vocab, preset.textenc_embed_dim, verbose=args.verbose)
        embeds = te.encode(tokens)
        timings["textenc_s"] = time.time() - t0
        print("text encoder: L=%d -> %s %.1fms  %s"
              % (len(tokens), tuple(embeds.shape), timings["textenc_s"] * 1000, stats()))
        if not args.keep_textenc:
            del te
            gc.collect()
            print("  released text encoder sessions -> %s" % stats())

    # ---- load models
    t0 = time.time()
    talker = TalkerRunner(preset.talker_dir, verbose=args.verbose)
    timings["load_talker_s"] = time.time() - t0
    t0 = time.time()
    lmhead_bin = None
    if args.lmhead_cpu:
        cands = sorted(glob.glob(os.path.join(preset.embeds_dir, "codebooks_head_15x*.bin")))
        cands += sorted(glob.glob("/root/breeze-tts/breeze/embeds/codebooks_head_15x*.bin"))
        cands += sorted(glob.glob("/root/breeze-tts/breeze/embeds/codebooks_head*.bin"))
        for c in cands:
            if c and os.path.exists(c):
                lmhead_bin = c
                break
        if not lmhead_bin:
            print("WARNING: --lmhead-cpu but codebooks_head bin not found; using axmodel heads")
    head_dir = preset.lmhead_hp_dir if (args.lmhead_hp and os.path.isdir(preset.lmhead_hp_dir)) else None
    cp = CodePredictorRunner(preset.cp_dir, verbose=args.verbose, lm_head_bin=lmhead_bin, head_dir=head_dir)
    timings["load_cp_s"] = time.time() - t0
    t0 = time.time()
    codec = SpeechTokenizerDecoder(preset.codec_dir, verbose=args.verbose)
    timings["load_codec_s"] = time.time() - t0
    print("loaded talker(%dL,g=%d,H=%d,P=%d,cap=%d,kv=%d) cp(%dL,g=%d,P=%d,kv=%d,heads=%d,vocab=%d) codec(%dL,P=%d,kv=%d,T<=%d)"
          % (talker.n_layers, talker.n_groups, talker.hidden, talker.prefill_len, talker.capacity, talker.kv_len,
             cp.n_layers, cp.n_groups, cp.prefill_len, cp.kv_len, cp.n_heads, cp.head_vocab,
             codec.n_layers, codec.prefill_len, codec.kv_len, min(codec.kv_len, codec.pre_len)))
    print("load: talker %.1fs cp %.1fs codec %.1fs  stats: %s"
          % (timings["load_talker_s"], timings["load_cp_s"], timings["load_codec_s"], stats()))

    # ---- embedding tables
    t0 = time.time()
    if preset.talker_embed_mode == "sum16":
        talker_emb = MultiEmbed(preset.talker_embed_paths, preset.audio_vocab, preset.talker_hidden,
                                n_groups=preset.n_codebooks)
        depth_emb = MultiEmbed(preset.depth_embed_paths, preset.audio_vocab, preset.depth_proj_shape[0],
                               n_groups=preset.depth_embed_groups)
        proj = None
        if preset.depth_proj_path and os.path.exists(preset.depth_proj_path):
            out_d, in_d = preset.depth_proj_shape  # (1024, 2048)
            proj = autodetect_depth_proj(talker_emb, depth_emb, preset.depth_proj_path, in_d, out_d)
            if proj is None:
                proj = Linear(preset.depth_proj_path, out_d, in_d, layout="out_in")
            print("depth proj: %s shape(in=%d,out=%d) effective_layout=%s"
                  % (os.path.basename(preset.depth_proj_path), in_d, out_d, proj.layout))
    else:
        talker_emb = MultiEmbed(preset.talker_embed_paths, preset.talker_vocab, preset.talker_hidden)
        depth_emb = MultiEmbed(preset.depth_embed_paths, preset.codebook_size, preset.talker_hidden)
        proj = None
    print("embedding tables: talker groups=%d  depth groups=%d" % (len(talker_emb), len(depth_emb)))
    timings["load_embeds_s"] = time.time() - t0
    if not args.no_prewarm:
        t0 = time.time()
        p_t = talker_emb.prewarm()
        p_d = depth_emb.prewarm()
        timings["prewarm_embeds_s"] = time.time() - t0
        print("prewarmed embed tables: talker %.0fms depth %.0fms (total %.1fs)" %
              (p_t * 1000, p_d * 1000, timings["prewarm_embeds_s"]))


    # ---- suppression masks
    if preset.name == "qwen3tts06":
        bb_suppress = qwen3tts_suppress(preset.talker_vocab, 2047, preset.eos_id)
        dp_suppress = np.zeros(max(cp.head_vocab, 1), dtype=bool)
    else:
        bb_suppress = suppress_array(preset.talker_vocab, preset.suppress_ids)
        dp_suppress = suppress_array(cp.head_vocab, preset.cp_suppress)

    # ---- talker prefill
    t0 = time.time()
    state, raw = talker.prefill(embeds)
    timings["talker_prefill_s"] = time.time() - t0
    logits, hnorm = talker.post_hidden(raw)
    par = {}
    probe_n = int(args.probe_frames)
    probe_hidden: List[np.ndarray] = []
    probe_btopk_idx: List[np.ndarray] = []
    probe_btopk_val: List[np.ndarray] = []
    probe_depth: List[np.ndarray] = []
    probe_depth_top1: List[np.ndarray] = []
    if args.dump_tokens:
        par["prefill_logits"] = logits.astype(np.float32).copy()
        if hnorm is not None:
            par["prefill_hnorm"] = np.asarray(hnorm, dtype=np.float32).reshape(-1).copy()
        if probe_n > 0:
            if not args.random_text_embeds:
                par["text_embeds"] = np.asarray(embeds, dtype=np.float32).copy()
            par["prompt_ids"] = np.asarray([] if tokens is None else tokens, dtype=np.int64)
    print("talker prefill L=%d: %.1fms  logits=%s hnorm=%s  %s"
          % (len(embeds), timings["talker_prefill_s"] * 1000, tuple(logits.shape),
             None if hnorm is None else tuple(hnorm.shape), stats()))

    # ---- autoregressive frames
    max_frames = int(args.max_frames)
    codec_limit = min(codec.kv_len, codec.pre_len)
    if max_frames > codec_limit:
        print("WARNING: --max-frames %d > codec limit %d frames (%.1fs audio); capping to %d"
              % (max_frames, codec_limit, codec_limit / 12.5, codec_limit))
        max_frames = codec_limit
    talker_limit = talker.kv_len - len(embeds)
    if max_frames > talker_limit:
        print("WARNING: talker kv_cache_len=%d leaves room for %d frames; capping"
              % (talker.kv_len, talker_limit))
        max_frames = max(0, talker_limit)
    frames: List[List[int]] = []
    history_cb0: List[int] = []
    t_talker_dec = t_depth = 0.0
    t0 = time.time()
    eos_hit = False
    for step in range(max_frames):
        cb0 = backbone_sample(logits, history_cb0, bb_suppress, cfg, rng)
        if cb0 == preset.eos_id:
            eos_hit = True
            print("  frame %d: EOS" % step)
            break
        if probe_n > 0 and step < probe_n:
            probe_hidden.append(np.asarray(hnorm, dtype=np.float32).reshape(-1).copy() if hnorm is not None
                                else np.zeros(0, dtype=np.float32))
            topk = min(5, logits.shape[0])
            idx = np.argsort(logits)[-topk:][::-1]
            probe_btopk_idx.append(idx.astype(np.int32))
            probe_btopk_val.append(logits[idx].astype(np.float32))
            _dbg: Optional[Dict[str, np.ndarray]] = {}
        else:
            _dbg = None
        td0 = time.time()
        if preset.name == "qwen3tts06":
            pos0 = hnorm if hnorm is not None else None
            if pos0 is None:
                raise RuntimeError("talker post has no output_norm; cannot feed depth decoder")
            codes = cp.generate(np.asarray(pos0).reshape(-1), cb0, depth_emb, dp_suppress, cfg, rng, proj=None,
                                debug=_dbg)
        else:
            if hnorm is None:
                raise RuntimeError("talker post has no output_norm (was it built with --ret_postnorm?)")
            codes = cp.generate(np.asarray(hnorm).reshape(-1), int(cb0), depth_emb, dp_suppress, cfg, rng,
                                proj=proj, debug=_dbg)
        if _dbg is not None:
            dl = _dbg.get("depth_logits", [])
            if dl:
                arr = np.stack(dl).astype(np.float16)
                probe_depth.append(arr)
                probe_depth_top1.append(np.argmax(np.stack(dl), axis=-1).astype(np.int32))
            if step == 0:
                for k, v in _dbg.items():
                    if k.startswith("depth0"):
                        par[k] = v
        t_depth += time.time() - td0
        frame = [int(cb0)] + codes
        frames.append(frame)
        history_cb0.append(int(cb0))
        # next frame embedding
        if preset.talker_embed_mode == "sum16":
            emb = np.zeros(talker.hidden, dtype=np.float32)
            for g in range(preset.n_codebooks):
                emb += talker_emb.row(g, int(frame[g])).astype(np.float32)
            emb = emb.astype(BF16).reshape(1, 1, talker.hidden)
        else:
            emb = talker_emb.row(0, int(frame[0]))
        t1 = time.time()
        raw = talker.decode(emb, state)
        t_talker_dec += time.time() - t1
        logits, hnorm = talker.post_hidden(raw)
        if args.log_every and (step % args.log_every == 0):
            print("  frame %d: cb0=%d talker=%.0fms depth=%.0fms  %s"
                  % (step, cb0, (time.time() - t1) * 1000, (time.time() - td0) * 1000, stats()))
    timings["frames_total_s"] = time.time() - t0
    if probe_n > 0 and probe_hidden:
        par["probe_hidden_norm"] = np.stack(probe_hidden)
        par["probe_backbone_topk_idx"] = np.stack(probe_btopk_idx)
        par["probe_backbone_topk_val"] = np.stack(probe_btopk_val)
        if probe_depth:
            par["probe_depth_logits"] = np.stack(probe_depth)
            par["probe_depth_top1"] = np.stack(probe_depth_top1)
    timings["talker_decode_s"] = t_talker_dec
    timings["depth_s"] = t_depth
    n_gen = len(frames)
    print("generated %d frames (%.2fs audio), eos_hit=%s" % (n_gen, n_gen / 12.5, eos_hit))

    # ---- codec decode
    codes_arr = np.asarray(frames, dtype=np.int32) if frames else np.zeros((0, preset.n_codebooks), dtype=np.int32)
    t0 = time.time()
    if n_gen and not args.no_codec:
        wav = codec.decode(codes_arr)
    else:
        wav = np.zeros(0, dtype=np.float32)
    timings["codec_decode_s"] = time.time() - t0
    timings["total_s"] = time.time() - t_start
    timings["rtf"] = (timings["total_s"] / (wav.size / SAMPLE_RATE)) if wav.size else float("nan")

    # ---- save
    if args.out_wav and wav.size:
        save_wav(args.out_wav, wav)
        print("saved wav:", args.out_wav, "%.3fs" % (wav.size / SAMPLE_RATE))
    if args.dump_tokens:
        meta = {
            "preset": preset.name, "greedy": greedy, "seed": args.seed,
            "text": args.text, "instruction": args.instruction, "speaker": args.speaker,
            "prompt_ids": None if tokens is None else np.asarray(tokens).tolist(),
            "n_frames": int(n_gen), "eos_hit": bool(eos_hit),
            "timings": {k: float(v) for k, v in timings.items()},
            "talker": {"layers": talker.n_layers, "hidden": talker.hidden, "kv_len": talker.kv_len},
            "cp": {"layers": cp.n_layers, "heads": cp.n_heads, "vocab": cp.head_vocab},
            "codec": {"layers": codec.n_layers, "kv_len": codec.kv_len},
        }
        save_kw = dict(codes=codes_arr,
                       prompt_ids=np.asarray([] if tokens is None else tokens, dtype=np.int64),
                       meta=np.asarray(json.dumps(meta, ensure_ascii=False)))
        for k, v in par.items():
            save_kw["par_" + k] = v
        np.savez(args.dump_tokens, **save_kw)
        print("saved tokens:", args.dump_tokens)

    # ---- summary
    print("\n=== TIMING ===")
    for k in ["load_talker_s", "load_cp_s", "load_codec_s", "load_embeds_s", "prewarm_embeds_s",
              "textenc_s", "talker_prefill_s"]:
        if k in timings:
            print("  %-18s %8.1f ms" % (k, timings[k] * 1000))
    if n_gen:
        print("  %-18s %8.1f ms (%.1f ms/frame)" % ("talker_decode", timings["talker_decode_s"] * 1000,
                                                    timings["talker_decode_s"] / n_gen * 1000))
        print("  %-18s %8.1f ms (%.1f ms/frame)" % ("depth_15steps", timings["depth_s"] * 1000,
                                                    timings["depth_s"] / n_gen * 1000))
        print("  %-18s %8.1f ms" % ("codec_decode", timings["codec_decode_s"] * 1000))
        print("  %-18s %8.1f ms" % ("total", timings["total_s"] * 1000))
        print("  %-18s %8.2f x realtime (audio %.2fs)" % ("RTF", timings["rtf"], n_gen / 12.5))
    print("final stats:", stats())
    return timings


def main():
    ap = argparse.ArgumentParser(description="Breeze TTS 2 board runtime (numpy + pyaxengine)")
    ap.add_argument("--preset", choices=["qwen3tts06", "breeze"], default="breeze")
    ap.add_argument("--breeze-base", default="",
                    help="Breeze package/workspace root (default: '.' if it looks like a package, "
                         "else /root/breeze-tts/breeze)")
    ap.add_argument("--talker-dir", default="")
    ap.add_argument("--cp-dir", "--depth-dir", dest="cp_dir", default="",
                    help="override depth-decoder axmodel dir (fast INT8: builds/code-predictor, "
                         "quality FP32: builds/code-predictor-hp)")
    ap.add_argument("--codec-dir", default="")
    ap.add_argument("--text", default="今天天气不错，我们出去走走吧。")
    ap.add_argument("--instruction", default="用温柔亲切的女声，语速稍慢。")
    ap.add_argument("--speaker", default="S0", help="speaker prefix [S0]; empty string to disable")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--random-text-embeds", type=int, default=0, help="glue test: use N random text-embedding rows")
    ap.add_argument("--prompt-ids", default="", help="explicit prompt token ids: comma list, or .npy/.npz/.json file")
    ap.add_argument("--text-embeds", default="",
                    help="npz with precomputed text_embeds [T,2048] (skip text encoder axmodel; fallback path)")
    ap.add_argument("--codes", default="",
                    help="codec-only mode: npz/npy with codes [T,16] to decode directly (no talker/depth)")
    ap.add_argument("--no-codec", action="store_true", help="skip codec decode (fast token-only parity runs)")
    ap.add_argument("--depth-mode", choices=["fast", "hp"], default="fast",
                    help="fast = INT8 depth (default), hp = high-precision depth dir (code-predictor-hp)")
    ap.add_argument("--lmhead-hp", action="store_true",
                    help="use U16 lm_head axmodels from <base>/lm-head-hp if present")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve presets/paths and exit (no model loading)")
    ap.add_argument("--lmhead-cpu", action="store_true",
                    help="compute depth lm_heads on CPU from codebooks_head bin (avoids quantized axmodel heads)")
    ap.add_argument("--keep-textenc", action="store_true",
                    help="keep text encoder sessions loaded (default: release before loading talker/cp/codec)")
    ap.add_argument("--no-prewarm", action="store_true",
                    help="skip sequential page-cache warmup of the embedding tables")
    ap.add_argument("--probe-frames", type=int, default=0,
                    help="capture per-frame probe arrays (hidden_norm/backbone top5/depth logits) for parity with ref probe.npz")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-wav", default="/root/breeze-tts/out.wav")
    ap.add_argument("--dump-tokens", default="")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not args.speaker:
        args.speaker = ""
    synthesize(args)


if __name__ == "__main__":
    main()
