#!/usr/bin/env python3
"""Staged parity report: board runtime probe vs ref-2038 golden probe.

Compares (as available):
  * text encoder:  board par_text_embeds            vs golden text_embeds          (cos / max-abs)
  * talker hidden: board par_probe_hidden_norm       vs golden hidden_norm          (per-frame cos)
  * backbone top5: board par_probe_backbone_topk_idx vs golden backbone_topk_idx   (top-5 overlap)
  * depth logits:  board par_probe_depth_logits      vs golden depth_logits         (per-frame/step cos, top1 match)
  * tokens:        board codes                       vs golden tokens               (delegated to compare_tokens)

Usage:
  python3 compare_probe.py board_dump.npz golden_probe.npz [--tokens golden_tokens.npz]
"""
import argparse
import sys

import numpy as np


def load_npz(path):
    try:
        return np.load(path, allow_pickle=False)
    except Exception as e:
        raise SystemExit("cannot load %s: %s" % (path, e))


def get(z, *names):
    for n in names:
        if n in z.files:
            return np.asarray(z[n])
    return None


def cos_rows(a, b):
    """per-row cosine between a [N,D] and b [N,D]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    den = np.maximum(na * nb, 1e-30)
    return (a * b).sum(axis=-1) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board")
    ap.add_argument("golden")
    ap.add_argument("--tokens", default="", help="golden tokens.npz (optional; if present reports token parity)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    import json

    b = load_npz(args.board)
    g = load_npz(args.golden)
    out = {}
    print("board keys:", list(b.files))
    print("golden keys:", list(g.files))

    # 1) text encoder
    be = get(b, "par_text_embeds")
    ge = get(g, "text_embeds")
    if be is not None and ge is not None:
        be = be.astype(np.float32)
        ge = ge.astype(np.float32)
        c = cos_rows(ge, be)
        out["text_embeds_cos_mean"] = float(c.mean())
        out["text_embeds_cos_min"] = float(c.min())
        out["text_embeds_rows"] = int(min(be.shape[0], ge.shape[0]))
        mad = float(np.abs(be[: ge.shape[0]] - ge[: be.shape[0]]).max())
        out["text_embeds_max_abs_diff"] = mad
        print("\n[1] text encoder: rows=%d  cos(mean/min)=%.6f / %.6f  max|diff|=%.4f  %s"
              % (out["text_embeds_rows"], out["text_embeds_cos_mean"], out["text_embeds_cos_min"], mad,
                 "PASS" if out["text_embeds_cos_min"] >= 0.99 else "CHECK"))
    else:
        print("\n[1] text encoder: SKIP (par_text_embeds or golden text_embeds missing)")

    # 2) talker post-norm hidden per frame
    bh = get(b, "par_probe_hidden_norm")
    gh = get(g, "hidden_norm")
    if bh is not None and gh is not None and bh.ndim == 2 and bh.shape[-1] == gh.shape[-1]:
        bh = bh.astype(np.float32)
        gh = gh.astype(np.float32)
        n = min(bh.shape[0], gh.shape[0])
        c = cos_rows(gh[:n], bh[:n])
        out["hidden_norm_cos_mean"] = float(c.mean())
        out["hidden_norm_cos_min"] = float(c.min())
        out["hidden_norm_frames"] = int(n)
        print("\n[2] talker hidden_norm: frames=%d  cos(mean/min)=%.6f / %.6f  %s"
              % (n, c.mean(), c.min(), "PASS" if c.min() >= 0.99 else "CHECK"))
        print("    per-frame cos:", " ".join("%.4f" % x for x in c[:12]) + (" ..." if n > 12 else ""))
    else:
        print("\n[2] talker hidden_norm: SKIP")

    # 3) backbone top-5 overlap
    bi = get(b, "par_probe_backbone_topk_idx")
    gi = get(g, "backbone_topk_idx")
    if bi is not None and gi is not None:
        n = min(len(bi), len(gi))
        overlaps = []
        for f in range(n):
            overlaps.append(len(set(int(x) for x in bi[f]) & set(int(x) for x in gi[f])))
        out["backbone_top5_mean_overlap"] = float(np.mean(overlaps))
        out["backbone_top5_exact_frames"] = float(np.mean([1.0 if o == 5 else 0.0 for o in overlaps]))
        print("\n[3] backbone top5: frames=%d  mean overlap=%.2f/5  exact frames=%.2f  %s"
              % (n, out["backbone_top5_mean_overlap"], out["backbone_top5_exact_frames"],
                 "PASS" if out["backbone_top5_mean_overlap"] >= 4.5 else "CHECK"))
        print("    frame0 board top5:", bi[0].tolist(), " golden:", gi[0].tolist())
    else:
        print("\n[3] backbone top5: SKIP")

    # 4) depth logits
    bd = get(b, "par_probe_depth_logits")
    gd = get(g, "depth_logits")
    if bd is not None and gd is not None and bd.shape[-1] == gd.shape[-1]:
        bd = bd.astype(np.float32)
        gd = gd.astype(np.float32)
        n = min(bd.shape[0], gd.shape[0])
        steps = min(bd.shape[1], gd.shape[1])
        cs = []
        top1 = []
        for f in range(n):
            for s in range(steps):
                a = bd[f, s]
                b2 = gd[f, s]
                den = (np.linalg.norm(a) * np.linalg.norm(b2))
                cs.append(float(np.dot(a, b2) / den) if den else float("nan"))
                top1.append(int(np.argmax(a) == np.argmax(b2)))
        cs = np.asarray(cs)
        out["depth_logits_cos_mean"] = float(cs.mean())
        out["depth_logits_cos_min"] = float(cs.min())
        out["depth_top1_match_rate"] = float(np.mean(top1))
        out["depth_frames"] = int(n)
        print("\n[4] depth logits: frames=%d steps=%d  cos(mean/min)=%.5f / %.5f  top1 match=%.3f  %s"
              % (n, steps, cs.mean(), cs.min(), np.mean(top1),
                 "PASS" if cs.min() >= 0.99 else "CHECK"))
    else:
        print("\n[4] depth logits: SKIP")

    # 5) tokens (optional)
    if args.tokens:
        bt = get(b, "codes")
        gt = get(load_npz(args.tokens), "tokens", "codes")
        if bt is not None and gt is not None:
            n = min(len(bt), len(gt))
            eq = (np.asarray(bt[:n]) == np.asarray(gt[:n]))
            out["token_match_rate"] = float(eq.mean())
            out["frames_compared"] = int(n)
            bad = np.where(~eq.all(axis=1))[0]
            out["first_token_divergence_frame"] = int(bad[0]) if bad.size else None
            print("\n[5] tokens: frames=%d  token match=%.4f  first divergence=%s"
                  % (n, out["token_match_rate"], out["first_token_divergence_frame"]))
        else:
            print("\n[5] tokens: SKIP (no codes in board dump or tokens in golden)")

    ok = True
    if "text_embeds_cos_min" in out:
        ok &= out["text_embeds_cos_min"] >= 0.99
    if "hidden_norm_cos_min" in out:
        ok &= out["hidden_norm_cos_min"] >= 0.99
    if "backbone_top5_mean_overlap" in out:
        ok &= out["backbone_top5_mean_overlap"] >= 4.0
    if "depth_logits_cos_min" in out:
        ok &= out["depth_logits_cos_min"] >= 0.98
    print("\n=== RESULT:", "STAGES PASS" if ok else "STAGE CHECKS NEED REVIEW", "===")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print("saved", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
