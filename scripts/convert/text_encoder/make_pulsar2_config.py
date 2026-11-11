#!/usr/bin/env python3
"""Generate a Pulsar2 build config for the Breeze text-encoder ONNX chunk.

Follows the Magnetar/Pulsar2 8.0 schema (see Magnetar/magnetar/stages/compile.py
`_build_config`) and the calibration format documented in
Magnetar/docs/input-format-cheatsheet.md:
  * one tar.gz per input, float32 .npy with the batch dim, shape == model input shape
  * INT8 (MinMax), highest_mix_precision=false, real business calibration data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=None,
                    help="export .meta.json (defaults to <onnx stem>.meta.json)")
    ap.add_argument("--calib-dir", type=Path, required=True,
                    help="dir containing inputs_embeds.tar.gz and attention_mask.tar.gz")
    ap.add_argument("--build-root", type=Path, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-calib", type=int, default=40)
    ap.add_argument("--target", default="AX650")
    ap.add_argument("--npu-mode", default="NPU1")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mixed", default="none", choices=["none", "u16", "auto"],
                    help="none=INT8; u16=U16 compute for all float op types (explicit layer_configs); "
                         "auto=highest_mix_precision auto-upgrade")
    args = ap.parse_args()

    onnx = args.onnx.resolve()
    meta_path = args.meta or onnx.with_name(onnx.stem + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tag = args.tag or onnx.stem
    build_root = args.build_root.resolve()
    (build_root / "work").mkdir(parents=True, exist_ok=True)

    inputs = meta["inputs"]  # {"inputs_embeds": [1,L,1152], "attention_mask": [1,L]}
    input_shapes = ";".join(
        f"{name}:{'x'.join(str(d) for d in shape)}" for name, shape in inputs.items())

    input_configs, input_processors = [], []
    for name, shape in inputs.items():
        tar = (args.calib_dir / f"{name}.tar.gz").resolve()
        if not tar.is_file():
            raise SystemExit(f"missing calibration archive: {tar}")
        input_configs.append({
            "tensor_name": name,
            "calibration_dataset": str(tar),
            "calibration_format": "Numpy",
            "calibration_size": args.n_calib,
            "calibration_mean": [],
            "calibration_std": [],
        })
        input_processors.append({
            "tensor_name": name,
            "tensor_layout": "NCHW",
            "src_dtype": "FP32",
            "src_layout": "NCHW",
        })

    config = {
        "input": str(onnx),
        "output_dir": str(build_root),
        "output_name": f"{tag}.axmodel",
        "work_dir": str(build_root / "work" / tag),
        "model_type": "ONNX",
        "target_hardware": args.target,
        "npu_mode": args.npu_mode,
        "input_shapes": input_shapes,
        "onnx_opt": {
            "disable_onnx_optimization": False,
            "enable_onnxsim": False,
            "model_check": True,
        },
        "quant": {
            "input_configs": input_configs,
            "calibration_method": "MinMax",
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
            "highest_mix_precision": False,
        },
        "input_processors": input_processors,
    }
    if args.mixed == "u16":
        # mixed precision: keep INT8 weights for most ops, compute the numerically sensitive
        # float ops in U16 (RMSNorm decomposition + residual adds + attention/softmax + MLP)
        config["quant"]["layer_configs"] = [
            {"op_types": ["MatMul", "Add", "Mul", "Sub", "Div", "Pow", "ReduceMean",
                          "Sqrt", "Tanh", "Softmax", "Neg",
                          "RMSNormalization", "LayerNormalization"],
             "data_type": "U16", "output_data_type": "U16"},
            {"op_type": "MatMul", "data_type": "U16",
             "weight_data_type": "U16", "output_data_type": "U16"},
        ]
    elif args.mixed == "auto":
        config["quant"]["highest_mix_precision"] = True

    out = args.out or (build_root / f"{tag}.pulsar2_config.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[config] {out}  input_shapes={input_shapes}")


if __name__ == "__main__":
    main()
