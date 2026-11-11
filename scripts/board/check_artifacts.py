#!/usr/bin/env python3
"""Validate Breeze artifacts on the board before running run_breeze_tts.py --preset breeze.

Checks (as they become available — missing pieces are reported, not fatal):
  1) talker: 28 per-layer axmodels + post; post output logits dim = 2052 and output_norm = 2048
     (i.e. built with --ret_postnorm); kv_cache_len; prefill shape group count; hidden 2048, kv_dim 1024.
  2) code-predictor: 12 layers + post + 15 lm_heads; hidden 1024; kv_dim (256 expected for Breeze);
     head output vocab 2051.
  3) embedding bins: 16x2051x2048 bf16 (talker), 15x2051x1024 bf16 (depth projected),
     depth proj matrix 1024x2048 bf16.
  4) text encoder axmodel(s) I/O + tokenizer.json presence; embed bin 262158x1152.

Usage:
  LD_LIBRARY_PATH=/soc/lib python3 check_artifacts.py --base /root/breeze-tts/breeze
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    from axengine import InferenceSession
except Exception as e:  # pragma: no cover
    print("FATAL: cannot import axengine:", e)
    sys.exit(2)


def io_of(sess, g):
    ins = [(n.name, str(n.dtype), tuple(n.shape)) for n in sess.get_inputs(g)]
    outs = [(n.name, str(n.dtype), tuple(n.shape)) for n in sess.get_outputs(g)]
    return ins, outs


def report(name, ok, detail=""):
    print(("[ OK ] " if ok else "[FAIL] ") + name + ((": " + detail) if detail else ""))
    return ok


def filesize(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/breeze-tts/breeze")
    ap.add_argument("--talker-dir", default="")
    ap.add_argument("--cp-dir", default="")
    args = ap.parse_args()
    base = args.base
    talker_dir = args.talker_dir or os.path.join(base, "builds/talker")
    cp_dir = args.cp_dir or os.path.join(base, "builds/code-predictor")
    embeds_dir = os.path.join(base, "embeds")
    textenc_dir = os.path.join(base, "textenc")
    tok_dir = os.path.join(base, "tokenizer")
    all_ok = True

    print("== base:", base)
    print("== talker_dir:", talker_dir)
    print("== cp_dir:", cp_dir)

    # ---------- talker
    print("\n--- talker ---")
    layers = {}
    for p in glob.glob(os.path.join(talker_dir, "*_l*_together.axmodel")):
        m = re.search(r"_l(\d+)_together\.axmodel$", os.path.basename(p))
        if m:
            layers[int(m.group(1))] = p
    all_ok &= report("talker layers found: %d (expect 28)" % len(layers), len(layers) == 28,
                     "indices=%s" % (sorted(layers)[:5] + ["..."] if len(layers) > 5 else sorted(layers)))
    if layers:
        sess = InferenceSession(sorted(layers.items())[0][1])
        ng = int(getattr(sess._sess, "_shape_count", 1))
        print("      shape groups:", ng, "(expect 6: 1 decode + 5 prefill)")
        if ng > 1:
            ins0, outs0 = io_of(sess, 0)
            ins1, outs1 = io_of(sess, 1)
            print("      g0 in:", ins0)
            print("      g0 out:", outs0)
            print("      g1 in:", ins1)
            hidden = ins0[3][2][2] if len(ins0) > 3 else -1
            kv_dim = ins0[0][2][2]
            kv_len = ins0[0][2][1]
            mask_len = ins0[4][2][2] if len(ins0) > 4 else -1
            all_ok &= report("talker hidden == 2048", hidden == 2048, "got %s" % hidden)
            all_ok &= report("talker kv_dim == 1024", kv_dim == 1024, "got %s" % kv_dim)
            all_ok &= report("talker decode mask len == kv_len+1", mask_len == kv_len + 1,
                             "kv_len=%s mask_len=%s" % (kv_len, mask_len))
            print("      kv_cache_len =", kv_len, " prefill chunk =", ins1[3][2][1],
                  " capacity =", ins1[3][2][1] * (ng - 1))
    posts = glob.glob(os.path.join(talker_dir, "*post.axmodel"))
    if posts:
        ps = InferenceSession(posts[0])
        ins, outs = io_of(ps, 0)
        print("      post in:", ins)
        print("      post out:", outs)
        logits_dim = None
        norm_dim = None
        for n, d, s in outs:
            if "norm" in n:
                norm_dim = s[-1]
            else:
                logits_dim = s[-1]
        all_ok &= report("talker post logits dim == 2052", logits_dim == 2052, "got %s" % logits_dim)
        all_ok &= report("talker post has output_norm (built with --ret_postnorm)", norm_dim == 2048,
                         "got %s" % norm_dim)
    else:
        all_ok &= report("talker post found", False, os.path.join(talker_dir, "*post.axmodel"))

    # ---------- code predictor
    print("\n--- code-predictor ---")
    cplayers = {}
    for p in glob.glob(os.path.join(cp_dir, "*_l*_together.axmodel")):
        m = re.search(r"_l(\d+)_together\.axmodel$", os.path.basename(p))
        if m:
            cplayers[int(m.group(1))] = p
    all_ok &= report("cp layers found: %d (expect 12)" % len(cplayers), len(cplayers) == 12)
    heads = sorted(glob.glob(os.path.join(cp_dir, "code_predictor_lm_head_*.axmodel")))
    all_ok &= report("cp lm_heads found: %d (expect 15)" % len(heads), len(heads) == 15)
    if heads:
        hs = InferenceSession(heads[0])
        ins, outs = io_of(hs, 0)
        print("      head0 in:", ins, " out:", outs)
        vocab = outs[0][2][-1]
        all_ok &= report("cp head vocab == 2051", vocab == 2051, "got %s" % vocab)
    if cplayers:
        sess = InferenceSession(sorted(cplayers.items())[0][1])
        ng = int(getattr(sess._sess, "_shape_count", 1))
        ins0, _ = io_of(sess, 0)
        print("      shape groups:", ng)
        print("      g0 in:", ins0)
        hidden = ins0[3][2][2] if len(ins0) > 3 else -1
        kv_dim = ins0[0][2][2]
        kv_len = ins0[0][2][1]
        all_ok &= report("cp hidden == 1024", hidden == 1024, "got %s" % hidden)
        print("      cp kv_cache_len =", kv_len, " kv_dim =", kv_dim,
              "(Breeze expected kv_dim=256 = 2 kv heads x128; 1024 => llm_build used Qwen3-TTS cp config / duplicated KV)")
    cposts = glob.glob(os.path.join(cp_dir, "*post.axmodel"))
    if cposts:
        ps = InferenceSession(cposts[0])
        print("      cp post in/out:", io_of(ps, 0))

    # ---------- embedding bins
    print("\n--- embedding bins ---")
    checks = [
        ("talker audio embeds 16x2051x2048 bf16",
         ["audio_embed_16x2051x2048.bf16.bin", "audio_embeds_16x2051x2048.bin"],
         16 * 2051 * 2048 * 2),
        ("depth projected embeds 15x2051x1024 bf16",
         ["codec_embed_proj_15x2051x1024.bf16.bin", "audio_embeds_proj_15x2051x1024.bin"],
         15 * 2051 * 1024 * 2),
        ("depth proj 1024x2048 bf16",
         ["inputs_embeds_projector_1024x2048.bf16.bin", "depth_proj_1024x2048.bin"], 1024 * 2048 * 2),
        ("depth codebooks_head 15x1024x2051 bf16 (CPU lm_head fallback)",
         ["codebooks_head_15x1024x2051.bf16.bin"], 15 * 1024 * 2051 * 2),
        ("text embedding 262158x1152 bf16",
         ["text_embed_262158x1152.bfloat16.bin", "textenc_embed_262158x1152.bin"], 262158 * 1152 * 2),
    ]
    for label, names, expect in checks:
        found = None
        for n in names:
            for d in (embeds_dir, textenc_dir, base):
                p = os.path.join(d, n)
                if os.path.exists(p):
                    found = p
                    break
            if found:
                break
        if not found:
            # fall back to any bin of the right size in the two dirs
            for d in (embeds_dir, textenc_dir):
                for p in glob.glob(os.path.join(d, "*.bin")):
                    if filesize(p) == expect:
                        found = p
                        break
                if found:
                    break
        if found:
            all_ok &= report(label, filesize(found) == expect,
                             "%s size=%d expect=%d" % (found, filesize(found), expect))
        else:
            all_ok &= report(label, False, "file not found (dirs: %s, %s)" % (embeds_dir, textenc_dir))

    # ---------- text encoder
    print("\n--- text encoder ---")
    tes = sorted(glob.glob(os.path.join(textenc_dir, "*.axmodel")))
    if tes:
        for p in tes:
            s = InferenceSession(p)
            print("      %s" % os.path.basename(p))
            for g in range(int(getattr(s._sess, "_shape_count", 1))):
                print("        g%d in : %s" % (g, io_of(s, g)[0]))
                print("        g%d out: %s" % (g, io_of(s, g)[1]))
        all_ok &= report("text encoder axmodels found: %d" % len(tes), True)
    else:
        all_ok &= report("text encoder axmodels", False, textenc_dir)

    print("\n--- tokenizer ---")
    tj = os.path.join(tok_dir, "tokenizer.json")
    all_ok &= report("tokenizer.json", os.path.exists(tj), tj)
    if os.path.exists(tj):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tj)
            ids = [2] + tok.encode("[S0]<ins_bos>用温柔亲切的女声，语速稍慢。<ins_eos>今天天气不错，我们出去走走吧。",
                                   add_special_tokens=False).ids
            expect = [2, 262146, 262156, 237105, 171109, 239148, 238201, 146440, 238389, 236900,
                      238384, 238121, 241066, 239364, 236924, 262157, 22601, 122362, 56033,
                      236900, 6932, 67038, 238302, 238302, 238702, 236924]
            all_ok &= report("tokenizer prompt ids match GEN_LOGIC §2.1", ids == expect, "len=%d" % len(ids))
        except Exception as e:
            all_ok &= report("tokenizer load", False, repr(e))

    print("\n=== RESULT:", "ALL GOOD" if all_ok else "SOME CHECKS FAILED", "===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
