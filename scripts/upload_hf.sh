#!/usr/bin/env bash
# 把本地模型包目录上传到 HuggingFace（逐文件 hf cp，走 xet；hf upload 的 folder 模式会被拒）
set -euo pipefail
REPO=${1:?usage: upload_hf.sh <local_pkg_dir> [repo_id]}
REPO=${2:-yunyu1258/breeze-tts-2-ax650}
cd "$(dirname "$REPO")" 2>/dev/null || true
ROOT=$(cd "$1" && pwd)
HF=${HF:-hf}
cd "$ROOT"
find . -type f ! -name '*.log' ! -name '.git*' | sort | while read -r f; do
  rel=${f#./}
  case "$rel" in
    *.pyc|*/__pycache__/*) continue ;;
  esac
  echo "[upload] $rel"
  $HF cp "$rel" "hf://$REPO/$rel"
done
echo "[done] https://huggingface.co/$REPO"
