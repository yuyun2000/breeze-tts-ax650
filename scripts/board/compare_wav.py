#!/usr/bin/env python3
"""Compare two 24kHz mono wav files: length, sample correlation, RMS envelope, last-N-second energy.

Used to verify that decoding the SAME golden codes through the board codec reproduces the reference
waveform (i.e. the AXERA speech_tokenizer decode axmodels are faithful, including the tail).

Run: python3 compare_wav.py a.wav b.wav [--last-seconds 0.5]
"""
import argparse
import sys
import wave

import numpy as np


def load(p):
    w = wave.open(p, "rb")
    n = w.getnframes()
    sr = w.getframerate()
    ch = w.getnchannels()
    d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    return d, sr


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--last-seconds", type=float, default=0.5)
    ap.add_argument("--frame-ms", type=float, default=20.0)
    args = ap.parse_args()
    x, sr = load(args.a)
    y, sr2 = load(args.b)
    print("A: %s  %d samples (%.3fs) sr=%d" % (args.a, x.size, x.size / sr, sr))
    print("B: %s  %d samples (%.3fs) sr=%d" % (args.b, y.size, y.size / sr2, sr2))
    n = min(x.size, y.size)
    if n == 0:
        print("empty"); return 1
    xc, yc = x[:n].astype(np.float64), y[:n].astype(np.float64)
    den = np.linalg.norm(xc) * np.linalg.norm(yc)
    corr = float(np.dot(xc, yc) / den) if den else float("nan")
    print("length equal: %s   global waveform corr: %.6f" % (x.size == y.size, corr))
    # per-20ms frame correlation / energy
    hop = max(1, int(sr * args.frame_ms / 1000))
    nf = n // hop
    cs, ea, eb = [], [], []
    for i in range(nf):
        a = xc[i * hop:(i + 1) * hop]
        b = yc[i * hop:(i + 1) * hop]
        d = np.linalg.norm(a) * np.linalg.norm(b)
        cs.append(float(np.dot(a, b) / d) if d else float("nan"))
        ea.append(rms(a.astype(np.float32)))
        eb.append(rms(b.astype(np.float32)))
    cs = np.asarray(cs, dtype=np.float64)
    ea = np.asarray(ea)
    eb = np.asarray(eb)
    print("per-frame corr (20ms): mean=%.4f median=%.4f min=%.3f  frames=%d"
          % (np.nanmean(cs), np.nanmedian(cs), np.nanmin(cs), nf))
    print("RMS: A mean=%.4f  B mean=%.4f  envelope corr=%.4f"
          % (ea.mean(), eb.mean(), float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else float("nan")))
    k = int(sr * args.last_seconds)
    print("last %.1fs RMS: A=%.5f  B=%.5f  (board/golden=%.3f)"
          % (args.last_seconds, rms(x[-k:]), rms(y[-k:]), (rms(y[-k:]) / rms(x[-k:])) if rms(x[-k:]) else float("nan")))
    # where does per-frame corr drop?
    bad = np.where(np.nan_to_num(cs, nan=0.0) < 0.5)[0]
    if bad.size:
        print("frames with corr<0.5: %d, first=%.2fs, last=%.2fs"
              % (bad.size, bad[0] * args.frame_ms / 1000, bad[-1] * args.frame_ms / 1000))
    else:
        print("frames with corr<0.5: 0")
    # spectral magnitude corr (overall)
    X = np.abs(np.fft.rfft(xc))
    Y = np.abs(np.fft.rfft(yc))
    d = np.linalg.norm(X) * np.linalg.norm(Y)
    print("spectral magnitude corr: %.6f" % (float(np.dot(X, Y) / d) if d else float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
