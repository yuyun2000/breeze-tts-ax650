#!/usr/bin/env python3
"""Compare two token dumps (golden from ref-2038 vs board runtime) and print parity metrics.

Handles npz files produced by:
  * board:  run_breeze_tts.py --dump-tokens out.npz   (keys: codes, prompt_ids, meta, par_*)
  * ref:    ref-2038 golden tokens.npz                (keys may be: codes / tokens / frames, prompt_ids, meta)

Usage:
  python3 compare_tokens.py golden.npz board.npz [--max-frames N] [--json out.json]
"""
import argparse
import json
import sys

import numpy as np


def load_codes(path):
    z = np.load(path, allow_pickle=False)
    codes = None
    for k in ("codes", "tokens", "frames", "audio_codes"):
        if k in z.files:
            codes = np.asarray(z[k])
            break
    if codes is None:
        raise SystemExit("%s: no codes array (keys=%s)" % (path, list(z.files)))
    if codes.ndim == 3:          # [1, T, 16] or [1, 16, T]
        codes = codes[0]
        if codes.shape[0] == 16 and codes.shape[1] != 16:
            codes = codes.T
    elif codes.ndim == 1:
        codes = codes.reshape(-1, 16)
    if codes.ndim != 2 or codes.shape[1] != 16:
        raise SystemExit("%s: unexpected codes shape %s" % (path, codes.shape))
    out = {"codes": codes.astype(np.int64), "meta": None, "prompt_ids": None}
    if "meta" in z.files:
        try:
            m = z["meta"]
            out["meta"] = m.item() if m.shape == () else str(m)
            if isinstance(out["meta"], str):
                try:
                    out["meta"] = json.loads(out["meta"])
                except Exception:
                    pass
        except Exception:
            pass
    if "prompt_ids" in z.files:
        out["prompt_ids"] = np.asarray(z["prompt_ids"]).reshape(-1)
    for k in ("prefill_logits", "par_prefill_logits"):
        if k in z.files:
            out["prefill_logits"] = np.asarray(z[k], dtype=np.float32).reshape(-1)
    return out


def cos(a, b):
    n = min(a.size, b.size)
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return float("nan")
    return float(np.dot(a, b) / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("board")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    g = load_codes(args.golden)
    b = load_codes(args.board)
    T = min(len(g["codes"]), len(b["codes"]))
    if args.max_frames:
        T = min(T, args.max_frames)
    report = {
        "golden_frames": int(len(g["codes"])),
        "board_frames": int(len(b["codes"])),
        "frames_compared": int(T),
        "golden_meta": g["meta"] if isinstance(g["meta"], dict) else None,
        "board_meta": b["meta"] if isinstance(b["meta"], dict) else None,
    }
    if g["prompt_ids"] is not None and b["prompt_ids"] is not None:
        pg, pb = g["prompt_ids"], b["prompt_ids"]
        report["prompt_ids_equal"] = bool(pg.shape == pb.shape and np.array_equal(pg, pb))
    if "prefill_logits" in g and "prefill_logits" in b:
        report["prefill_logits_cos"] = cos(g["prefill_logits"], b["prefill_logits"])

    if T == 0:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 1

    gc = g["codes"][:T]
    bc = b["codes"][:T]
    eq = (gc == bc)
    report["token_match_rate"] = float(eq.mean())
    report["frame_exact_match_rate"] = float(eq.all(axis=1).mean())
    per_cb = eq.mean(axis=0)
    report["per_codebook_match_rate"] = [float(x) for x in per_cb]
    report["per_codebook_match_rate_cb0"] = float(per_cb[0])
    frame_bad = np.where(~eq.all(axis=1))[0]
    report["first_divergence_frame"] = int(frame_bad[0]) if frame_bad.size else None
    if frame_bad.size:
        f = int(frame_bad[0])
        report["first_divergence"] = {
            "frame": f,
            "golden": gc[f].tolist(),
            "board": bc[f].tolist(),
            "golden_cb0_hist": gc[: f + 1, 0].tolist(),
            "board_cb0_hist": bc[: f + 1, 0].tolist(),
        }
    # how many frames match exactly after the first divergence (informational)
    if frame_bad.size:
        rest = eq[frame_bad[0]:].all(axis=1)
        report["exact_match_after_first_divergence"] = float(rest.mean())

    print("=== token parity ===")
    print("golden frames: %d   board frames: %d   compared: %d"
          % (report["golden_frames"], report["board_frames"], T))
    print("token match rate        : %.4f (%.1f%%)" % (report["token_match_rate"], 100 * report["token_match_rate"]))
    print("frame exact match rate  : %.4f" % report["frame_exact_match_rate"])
    print("cb0 match rate          : %.4f" % per_cb[0])
    print("per-codebook match rate : %s" % " ".join("%.2f" % x for x in per_cb))
    if report["first_divergence_frame"] is not None:
        f = report["first_divergence_frame"]
        print("first divergence frame  : %d (golden cb0=%d board cb0=%d)"
              % (f, gc[f, 0], bc[f, 0]))
        print("  golden:", gc[f].tolist())
        print("  board :", bc[f].tolist())
    elif T > 0:
        print("first divergence frame  : none (compare window fully identical)")
    for k in ("prompt_ids_equal", "prefill_logits_cos"):
        if k in report:
            print("%-24s: %s" % (k, report[k]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        print("saved", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
