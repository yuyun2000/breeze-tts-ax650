#!/bin/bash
# Compile the 15 position-dependent code-predictor lm_head ONNX models for AX650.
set -euo pipefail
BASE=$HOME/work/breeze-tts
SRC=$BASE/repack-lead/output/lm_head_onnx
OUT=$BASE/builds/code-predictor
CALIB=$BASE/repack/calib/input.tar.gz
CFGDIR=$BASE/repack/lm_head_cfg
mkdir -p "$OUT" "$CFGDIR"
export PATH=$HOME/.cache/magnetar/pulsar2/8.0/bin:$PATH
run_one() {
  i=$1
  cfg=$CFGDIR/lm_head_$i.json
  cat > "$cfg" <<JSON
{
  "input": "$SRC/code_predictor_lm_head_$i.onnx",
  "output_dir": "$OUT/lmhead_build_$i",
  "output_name": "code_predictor_lm_head_$i.axmodel",
  "model_type": "ONNX",
  "target_hardware": "AX650",
  "npu_mode": "NPU1",
  "input_shapes": "input:1x1x1024",
  "onnx_opt": {"model_check": true},
  "output_processors": [{"tensor_name": "output", "output_dtype": "FP32", "tensor_layout": "DefaultLayout"}],
  "quant": {
    "input_configs": [{"tensor_name": "input", "calibration_dataset": "$CALIB", "calibration_format": "Numpy", "calibration_size": 64}],
    "calibration_method": "MinMax",
    "precision_analysis": false,
    "highest_mix_precision": false
  },
  "compiler": {"check": 2}
}
JSON
  pulsar2 build --config "$cfg" > "$OUT/lmhead_$i.log" 2>&1
}
export -f run_one
export BASE SRC OUT CALIB CFGDIR
ls "$SRC"/code_predictor_lm_head_*.onnx | wc -l
for i in $(seq 0 14); do echo "$i"; done | xargs -P 2 -I{} bash -c "run_one {}"
echo "ALL_DONE"
