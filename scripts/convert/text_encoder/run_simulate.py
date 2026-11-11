#!/usr/bin/env python3
"""Run Pulsar2 simulation (`pulsar2 run`) on the compiled text-encoder axmodel and
compare its output with the ONNX reference -> acceptance cosine (target >= 0.99).

Pulsar2 simulation I/O convention (Magnetar/docs/input-format-cheatsheet.md):
  input dir:  <tensor_name>.bin   float32 raw bytes, shape == model input shape
  output dir: <tensor_name>.bin   float32 raw bytes

For a chunked (split) model pass --axmodel a.axmodel b.axmodel; the output of chunk j
is written as its float32 .bin and fed to chunk j+1.

Usage:
  python run_simulate.py --axmodel build/textenc_L128.axmodel --onnx onnx/textenc_L128.onnx \
      --calib calib/val/val.npz --n 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # .../textenc/scripts
TEXTENC = HERE.parent                           # .../textenc
PULSAR2_BIN = Path(os.environ.get("PULSAR2_HOME", os.path.expanduser(
    "~/.cache/magnetar/pulsar2/8.0"))) / "bin"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_pulsar2(model: Path, input_dir: Path, output_dir: Path, log: Path) -> None:
    env = dict(os.environ)
    env["PATH"] = f"{PULSAR2_BIN}:{env.get('PATH', '')}"
    cmd = [str(PULSAR2_BIN / "pulsar2"), "run", "--model", str(model),
           "--input_dir", str(input_dir), "--output_dir", str(output_dir)]
    with log.open("w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=3600)
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        raise RuntimeError(f"pulsar2 run failed (rc={proc.returncode}):\n" + "\n".join(tail))


def read_output(output_dir: Path, expected_shape) -> np.ndarray:
    bins = sorted(output_dir.glob("*.bin"))
    if not bins:
        raise RuntimeError(f"pulsar2 run produced no *.bin in {output_dir}")
    # prefer the file named like our single output, else the only/first one
    preferred = [p for p in bins if p.stem == "text_embeds"]
    path = preferred[0] if preferred else bins[0]
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != int(np.prod(expected_shape)):
        raise RuntimeError(f"{path.name}: {raw.size} floats, expected {np.prod(expected_shape)}")
    return raw.reshape(expected_shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axmodel", nargs="+", required=True)
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--calib", type=Path, default=TEXTENC / "calib" / "val" / "val.npz")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--work", type=Path, default=TEXTENC / "simulate")
    ap.add_argument("--chunk-layers", default=None, help="e.g. 0:13,13:26")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import onnxruntime as ort  # noqa: PLC0415

    val = np.load(args.calib)
    embeds_all, mask_all, lengths = val["inputs_embeds"], val["attention_mask"], val["lengths"]
    sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for p in args.onnx]

    work = args.work.resolve()
    rows = []
    for i in range(min(args.n, embeds_all.shape[0])):
        sample_dir = work / f"sample{i}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True)
        embeds = embeds_all[i:i + 1].astype(np.float32)
        mask = mask_all[i:i + 1].astype(np.float32)
        n_real = int(lengths[i])

        # ONNX reference
        names = [s.get_inputs()[0].name for s in sessions]
        mask_name = sessions[0].get_inputs()[1].name
        ref = sessions[0].run(None, {names[0]: embeds, mask_name: mask})[0] if len(sessions) == 1 else None
        if ref is None:
            x = embeds
            for sess in sessions:
                nm = [sess.get_inputs()[0].name, sess.get_inputs()[1].name]
                x = sess.run(None, {nm[0]: x, nm[1]: mask})[0]
            ref = x

        ax_out = None
        for j, (axmodel) in enumerate(args.axmodel):
            in_dir = sample_dir / f"chunk{j}_input"
            out_dir = sample_dir / f"chunk{j}_output"
            in_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            if j == 0:
                in_arr = embeds
            else:
                in_arr = ax_out
            in_arr.astype(np.float32).tofile(in_dir / "inputs_embeds.bin")
            mask.tofile(in_dir / "attention_mask.bin")
            t0 = time.time()
            run_pulsar2(Path(axmodel), in_dir, out_dir, sample_dir / f"chunk{j}_pulsar2.log")
            sim_time = time.time() - t0
            # expected shape of this chunk output
            if j == len(args.axmodel) - 1:
                expected = ref.shape
            else:
                # intermediate hidden: (1, L, 1152)
                expected = (1, embeds.shape[1], embeds.shape[2])
            ax_out = read_output(out_dir, expected)
            rows.append({"sample": i, "chunk": j, "axmodel": str(axmodel),
                         "sim_time_s": round(sim_time, 2)})

        row = {
            "sample": i,
            "n_tokens": n_real,
            "cosine_all_positions": cosine(ax_out, ref),
            "cosine_real_tokens_only": cosine(ax_out[:, :n_real], ref[:, :n_real]),
            "mae_real_tokens": float(np.mean(np.abs(ax_out[:, :n_real] - ref[:, :n_real]))),
            "max_abs_diff_real_tokens": float(np.max(np.abs(ax_out[:, :n_real] - ref[:, :n_real]))),
            "sim_time_s": rows[-1]["sim_time_s"],
        }
        print(f"[simulate] sample {i} tokens={n_real} "
              f"cos_all={row['cosine_all_positions']:.6f} "
              f"cos_real={row['cosine_real_tokens_only']:.6f} "
              f"({row['sim_time_s']:.1f}s)")
        rows.append(row)

    cos_rows = [r for r in rows if "cosine_real_tokens_only" in r]
    summary = {
        "axmodel": [str(p) for p in args.axmodel],
        "onnx": [str(p) for p in args.onnx],
        "chunks": args.chunk_layers,
        "mean_cosine_real_tokens": float(np.mean([r["cosine_real_tokens_only"] for r in cos_rows])),
        "min_cosine_real_tokens": float(np.min([r["cosine_real_tokens_only"] for r in cos_rows])),
        "mean_cosine_all_positions": float(np.mean([r["cosine_all_positions"] for r in cos_rows])),
        "mean_sim_time_s": float(np.mean([r["sim_time_s"] for r in cos_rows])),
        "rows": rows,
    }
    print(f"[simulate] mean cosine (real tokens) = {summary['mean_cosine_real_tokens']:.6f}  "
          f"min = {summary['min_cosine_real_tokens']:.6f}  sim {summary['mean_sim_time_s']:.1f}s/run")
    out = args.out or (work / "simulate_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[simulate] metrics -> {out}")


if __name__ == "__main__":
    main()
