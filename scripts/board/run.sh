#!/bin/bash
# Breeze TTS 2 on AX650 — inference/demo entry point.
#
# Works in both layouts:
#   * HF package:   <pkg>/runtime/run.sh     (axmodels in <pkg>/talker, <pkg>/code-predictor, ...)
#   * board tree:   /root/breeze-tts/run.sh  (axmodels in <base>/builds/...)
#
# Usage:
#   ./run.sh demo             # replay all demos available (sampled mode, fast depth)
#   ./run.sh demo-zh          # short Chinese demo (needs demos/text_embeds_zh.npz or golden)
#   ./run.sh demo-zh-long
#   ./run.sh demo-en
#   ./run.sh demo-zh2
#   ./run.sh demo-hp          # quality mode (FP32 depth), slower
#   ./run.sh text "<句子>" "<instruction>"   # requires text encoder axmodels (not shipped yet)
#   ./run.sh codes <tokens.npz>
#   ./run.sh check            # artifacts self-check
#   ./run.sh dry-run          # print resolved paths
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH}

# detect package root + runtime dir
if [ -d "$HERE/../talker" ] || [ -d "$HERE/../embeds" ] || [ -d "$HERE/../builds" ]; then
  PKG="$(cd "$HERE/.." && pwd)"; RT="$HERE"
elif [ -d "$HERE/builds" ] || [ -d "$HERE/embeds" ]; then
  PKG="$HERE"; RT="$HERE"
else
  PKG="/root/breeze-tts/breeze"; RT="$HERE"
fi
OUT="${OUT_DIR:-$PKG/outputs}"; mkdir -p "$OUT"
COMMON="--preset breeze --breeze-base $PKG --max-frames 120 --temperature 0.9 --top-k 50 --seed 42"
[ -d "$PKG/lm-head-hp" ] && COMMON="$COMMON --lmhead-hp"
TEXTEMB="$PKG/demos/text_embeds"
[ -d "$TEXTEMB" ] || TEXTEMB="$PKG/text_embeds"

run_emb() { # name npz
  python3 "$RT/run_breeze_tts.py" $COMMON --text-embeds "$2" \
    --out-wav "$OUT/demo_$1.wav" --dump-tokens "$OUT/demo_$1.npz" | tee "$OUT/demo_$1.log"
}
pick() { # name: prefer package demos/text_embeds then board golden dirs
  for p in "$TEXTEMB/$1.npz" "$PKG/ref_$1/text_embeds.npz" "$PKG/ref_golden_short/probe.npz"; do
    [ -f "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}
case "${1:-demo}" in
  demo) for n in zh zh_long en zh2; do "$0" demo-$n || true; done ;;
  demo-zh|demo-zh-long|demo-en|demo-zh2)
    n="${1#demo-}"; f="$(pick "$n")" || { echo "no text_embeds for $n"; exit 1; }; run_emb "$n" "$f" ;;
  demo-hp)
    python3 "$RT/run_breeze_tts.py" $COMMON --depth-mode hp --lmhead-hp \
      --text-embeds "$(pick zh)" --out-wav "$OUT/demo_zh_hp.wav" --dump-tokens "$OUT/demo_zh_hp.npz" \
      | tee "$OUT/demo_zh_hp.log" ;;
  text)
    python3 "$RT/run_breeze_tts.py" $COMMON --text "${2:?text}" \
      --instruction "${3:-一位温柔自信的年轻女性，声音清晰，语气亲切。}" \
      --out-wav "$OUT/demo_text.wav" --dump-tokens "$OUT/demo_text.npz" ;;
  codes) python3 "$RT/run_breeze_tts.py" --preset breeze --breeze-base "$PKG" --codes "${2:?npz}" \
           --out-wav "${3:-$OUT/codec_decoded.wav}" ;;
  check) python3 "$RT/check_artifacts.py" --base "$PKG" ;;
  dry-run) python3 "$RT/run_breeze_tts.py" --preset breeze --breeze-base "$PKG" --dry-run ;;
  *) echo "unknown mode: $1"; exit 2 ;;
esac
