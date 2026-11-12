#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breeze TTS 2 · AX650 常驻（warm）推理引擎

思路：把 run_breeze_tts.py 里「加载模型」的部分提到进程启动时做一次，之后每个请求
只跑 textenc → talker prefill/decode → depth 15 步 → codec decode，省掉每次 ~24s 的
模型加载（textenc 12s + talker/cp/codec 12s）。

注意：board 上系统内存 1.9GB、CMM 6GB。常驻占用约 3.3GB CMM（textenc 0.77 + talker 1.67
+ cp 0.40 + codec 0.16 + embedding 表 0.27），加其它服务约 1GB，仍在 6GB 预算内；
若其它服务后续增长导致加载失败，服务会自动回退到「每请求子进程」模式。
"""
import gc, os, sys, time, glob
import numpy as np

RUNTIME_DIR = "/root/breeze-tts"
if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)

import run_breeze_tts as rt  # noqa: E402

BF16 = rt.BF16
SAMPLE_RATE = rt.SAMPLE_RATE


class Cancelled(Exception):
    pass


class WarmEngine:
    def __init__(self, base="/root/breeze-tts/breeze", log=print):
        self.base = base
        self.log = log
        self.ready = False
        self.preset = rt.preset_breeze(base)
        self._load()

    # ---------------- load once ----------------
    def _load(self):
        p = self.preset
        log = self.log
        t_all = time.time()
        t0 = time.time()
        self.tok = rt.PromptTokenizer(p.tokenizer_json)
        self.te = rt.TextEncoderRunner(p.textenc_dir, p.textenc_embed_bin,
                                       p.textenc_embed_vocab, p.textenc_embed_dim, verbose=False)
        self.load_textenc_s = time.time() - t0
        log("warm: text encoder loaded in %.1fs" % self.load_textenc_s)

        t0 = time.time()
        self.talker = rt.TalkerRunner(p.talker_dir, verbose=False)
        head_dir = p.lmhead_hp_dir if (p.lmhead_hp_dir and os.path.isdir(p.lmhead_hp_dir)) else None
        self.cp = rt.CodePredictorRunner(p.cp_dir, verbose=False, lm_head_bin=None, head_dir=head_dir)
        self.codec = rt.SpeechTokenizerDecoder(p.codec_dir, verbose=False)
        self.load_models_s = time.time() - t0
        self.head_dir = head_dir
        self.cp_mode = "fast"
        log("warm: talker/cp/codec loaded in %.1fs (talker %dL cp %dL heads=%s codec %dL)"
            % (self.load_models_s, self.talker.n_layers, self.cp.n_layers,
               "u16" if head_dir else "int8", self.codec.n_layers))

        t0 = time.time()
        self.talker_emb = rt.MultiEmbed(p.talker_embed_paths, p.audio_vocab, p.talker_hidden,
                                        n_groups=p.n_codebooks)
        self.depth_emb = rt.MultiEmbed(p.depth_embed_paths, p.audio_vocab, p.depth_proj_shape[0],
                                       n_groups=p.depth_embed_groups)
        self.proj = None
        if p.depth_proj_path and os.path.exists(p.depth_proj_path):
            out_d, in_d = p.depth_proj_shape
            self.proj = rt.autodetect_depth_proj(self.talker_emb, self.depth_emb,
                                                 p.depth_proj_path, in_d, out_d)
            if self.proj is None:
                self.proj = rt.Linear(p.depth_proj_path, out_d, in_d, layout="out_in")
        self.talker_emb.prewarm()
        self.depth_emb.prewarm()
        self.load_embeds_s = time.time() - t0

        if p.name == "qwen3tts06":
            self.bb_suppress = rt.qwen3tts_suppress(p.talker_vocab, 2047, p.eos_id)
            self.dp_suppress = np.zeros(max(self.cp.head_vocab, 1), dtype=bool)
        else:
            self.bb_suppress = rt.suppress_array(p.talker_vocab, p.suppress_ids)
            self.dp_suppress = rt.suppress_array(self.cp.head_vocab, p.cp_suppress)

        self.load_total_s = time.time() - t_all
        self.ready = True
        self.log("warm: ALL models resident in %.1fs; CMM: %s" % (self.load_total_s, rt.stats()))

    # ---------------- depth 模式切换（fast <-> quality） ----------------
    def ensure_cp(self, quality=False, log=print):
        want = "hp" if quality else "fast"
        if getattr(self, "cp_mode", None) == want:
            return
        p = self.preset
        t0 = time.time()
        try:
            del self.cp
        except Exception:
            pass
        self.cp = None
        gc.collect()
        cp_dir = p.cp_hp_dir if (want == "hp" and p.cp_hp_dir) else p.cp_dir
        self.cp = rt.CodePredictorRunner(cp_dir, verbose=False, lm_head_bin=None,
                                         head_dir=getattr(self, "head_dir", None))
        self.cp_mode = want
        log("warm: depth mode -> %s (%s) %.1fs; CMM: %s"
            % (want, os.path.basename(cp_dir), time.time() - t0, rt.stats()))

    # ---------------- one synthesis ----------------
    def synth(self, text, instruction="", speaker="S0", out_wav="/tmp/out.wav",
              max_frames=120, seed=42, greedy=False, temperature=0.9, top_k=50,
              top_p=1.0, repetition_penalty=1.1, dump_tokens="", should_cancel=None,
              on_progress=None, quality=False, log=print):
        p = self.preset
        self.ensure_cp(quality=quality, log=log)
        t_start = time.time()
        timings = {}

        cfg = rt.SampleCfg(temperature=temperature, top_k=top_k, top_p=top_p,
                           repetition_penalty=repetition_penalty, greedy=greedy)
        rng = np.random.default_rng(seed)

        # ---- text encoder (resident)
        t0 = time.time()
        tokens = self.tok.encode(text, instruction, speaker)
        log("  prompt ids (L=%d): %s" % (len(tokens), tokens.tolist()[:40]))
        embeds = self.te.encode(tokens)
        timings["textenc_s"] = time.time() - t0
        log("  text encoder: L=%d -> %s %.0fms" % (len(tokens), tuple(embeds.shape),
                                                   timings["textenc_s"] * 1000))
        if should_cancel and should_cancel():
            raise Cancelled()

        # ---- talker prefill
        t0 = time.time()
        state, raw = self.talker.prefill(embeds)
        timings["talker_prefill_s"] = time.time() - t0
        logits, hnorm = self.talker.post_hidden(raw)
        log("  talker prefill L=%d: %.1fms" % (len(embeds), timings["talker_prefill_s"] * 1000))

        # ---- frames
        max_frames = int(max_frames)
        codec_limit = min(self.codec.kv_len, self.codec.pre_len)
        if max_frames > codec_limit:
            log("  WARNING: max_frames %d > codec limit %d, capping" % (max_frames, codec_limit))
            max_frames = codec_limit
        talker_limit = self.talker.kv_len - len(embeds)
        if max_frames > talker_limit:
            log("  WARNING: talker kv limit, capping to %d" % max(0, talker_limit))
            max_frames = max(0, talker_limit)

        frames = []
        history_cb0 = []
        t_talker_dec = t_depth = 0.0
        eos_hit = False
        t_frames0 = time.time()
        for step in range(max_frames):
            if should_cancel and should_cancel():
                raise Cancelled()
            cb0 = rt.backbone_sample(logits, history_cb0, self.bb_suppress, cfg, rng)
            if cb0 == p.eos_id:
                eos_hit = True
                log("  frame %d: EOS" % step)
                break
            td0 = time.time()
            codes = self.cp.generate(np.asarray(hnorm).reshape(-1), int(cb0), self.depth_emb,
                                     self.dp_suppress, cfg, rng, proj=self.proj, debug=None)
            t_depth += time.time() - td0
            frame = [int(cb0)] + codes
            frames.append(frame)
            history_cb0.append(int(cb0))
            emb = np.zeros(self.talker.hidden, dtype=np.float32)
            for g in range(p.n_codebooks):
                emb += self.talker_emb.row(g, int(frame[g])).astype(np.float32)
            emb = emb.astype(BF16).reshape(1, 1, self.talker.hidden)
            t1 = time.time()
            raw = self.talker.decode(emb, state)
            t_talker_dec += time.time() - t1
            logits, hnorm = self.talker.post_hidden(raw)
            if on_progress and (step % 5 == 0):
                on_progress(step + 1, (time.time() - t1) * 1000, (time.time() - td0) * 1000)
            if step % 10 == 0:
                log("  frame %d: cb0=%d talker=%.0fms depth=%.0fms"
                    % (step, cb0, (time.time() - t1) * 1000, (time.time() - td0) * 1000))
        timings["frames_total_s"] = time.time() - t_frames0
        timings["talker_decode_s"] = t_talker_dec
        timings["depth_s"] = t_depth
        n_gen = len(frames)
        log("generated %d frames (%.2fs audio), eos_hit=%s" % (n_gen, n_gen / 12.5, eos_hit))

        # ---- codec
        codes_arr = (np.asarray(frames, dtype=np.int32) if frames
                     else np.zeros((0, p.n_codebooks), dtype=np.int32))
        t0 = time.time()
        wav = self.codec.decode(codes_arr) if n_gen else np.zeros(0, dtype=np.float32)
        timings["codec_decode_s"] = time.time() - t0
        timings["total_s"] = time.time() - t_start
        timings["rtf"] = (timings["total_s"] / (wav.size / SAMPLE_RATE)) if wav.size else float("nan")

        if out_wav and wav.size:
            rt.save_wav(out_wav, wav)
            log("saved wav: %s %.3fs" % (out_wav, wav.size / SAMPLE_RATE))
        if dump_tokens:
            meta = {"preset": p.name, "greedy": greedy, "seed": seed,
                    "text": text, "instruction": instruction, "speaker": speaker,
                    "warm_engine": True,
                    "prompt_ids": np.asarray(tokens).tolist(),
                    "n_frames": int(n_gen), "eos_hit": bool(eos_hit),
                    "timings": {k: float(v) for k, v in timings.items()}}
            np.savez(dump_tokens, codes=codes_arr,
                     prompt_ids=np.asarray(tokens, dtype=np.int64),
                     meta=np.asarray(__import__("json").dumps(meta, ensure_ascii=False)))
            log("saved tokens: %s" % dump_tokens)

        log("=== TIMING (warm) ===")
        for k in ("textenc_s", "talker_prefill_s", "talker_decode_s", "depth_s",
                  "codec_decode_s", "total_s"):
            if k in timings:
                log("  %-18s %8.1f ms" % (k, timings[k] * 1000))
        if n_gen:
            log("  %-18s %8.1f ms (%.1f ms/frame)" % ("talker_decode", timings["talker_decode_s"] * 1000,
                                                      timings["talker_decode_s"] / n_gen * 1000))
            log("  %-18s %8.1f ms (%.1f ms/frame)" % ("depth_15steps", timings["depth_s"] * 1000,
                                                      timings["depth_s"] / n_gen * 1000))
            log("  %-18s %8.2f x realtime" % ("RTF", timings["rtf"]))
        log("final stats: %s" % rt.stats())
        return {"timings": timings, "n_frames": n_gen, "eos_hit": eos_hit,
                "audio_sec": round(wav.size / SAMPLE_RATE, 3) if wav.size else 0.0,
                "prompt_ids": tokens.tolist(), "codes": codes_arr}


if __name__ == "__main__":
    # 自测：加载一次，连续合成两次
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="你好，欢迎体验 AX650 上的 Breeze TTS 语音合成。")
    ap.add_argument("--instruction", default="一位温柔自信的年轻女性，声音清晰，语气亲切。")
    ap.add_argument("--out", default="/tmp/warm_test.wav")
    ap.add_argument("--times", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    eng = WarmEngine(log=print)
    print("LOAD %.1fs" % (time.time() - t0))
    for i in range(a.times):
        t = time.time()
        r = eng.synth(a.text + ((" 第%d次。" % (i + 1)) if i else ""), a.instruction,
                      out_wav=a.out, seed=42 + i, log=print)
        print("RUN %d: %.1fs  audio %.2fs  frames %d  timings %s"
              % (i + 1, time.time() - t, r["audio_sec"], r["n_frames"],
                 {k: round(v, 1) for k, v in r["timings"].items()}))
        if i == 0:
            t = time.time()
            r = eng.synth(a.text, a.instruction, out_wav=a.out, seed=99, log=print)
            print("RUN warm: %.1fs audio %.2fs frames %d" % (time.time() - t, r["audio_sec"], r["n_frames"]))
