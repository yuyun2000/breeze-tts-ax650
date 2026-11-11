#!/bin/bash
# Integrate Breeze artifacts onto the AX650 board (192.168.10.105) and validate them.
#
# Usage:
#   ./integrate_artifacts.sh local <local_layout_dir>      # push a local dir (layout: builds/, embeds/, textenc/, tokenizer/)
#   ./integrate_artifacts.sh 2038 <remote_dir_on_20.38>    # stream 20.38 -> Mac -> board directly (tar over ssh)
#   ./integrate_artifacts.sh check                          # only run check_artifacts.py on the board
#
# The board target layout is /root/breeze-tts/breeze/ .
set -euo pipefail

BOARD=root@192.168.10.105
REMOTE=m5stack@192.168.20.38
DEST=/root/breeze-tts/breeze

check_only() {
  ssh "$BOARD" 'cd /root/breeze-tts && LD_LIBRARY_PATH=/soc/lib python3 check_artifacts.py --base /root/breeze-tts/breeze' 2>&1 | grep -vE '^\[INFO\]'
}

case "${1:-}" in
  local)
    SRC=${2:?usage: integrate_artifacts.sh local <dir>}
    echo "== push $SRC -> $BOARD:$DEST"
    tar cf - -C "$SRC" . | ssh "$BOARD" "mkdir -p $DEST && tar xf - -C $DEST"
    check_only
    ;;
  2038)
    SRC=${2:?usage: integrate_artifacts.sh 2038 <remote_dir>}
    echo "== stream $REMOTE:$SRC -> $BOARD:$DEST"
    ssh "$REMOTE" "tar cf - -C '$SRC' ." | ssh "$BOARD" "mkdir -p $DEST && tar xf - -C $DEST"
    check_only
    ;;
  check)
    check_only
    ;;
  *)
    echo "usage: $0 {local <dir>|2038 <remote_dir>|check}" >&2
    exit 2
    ;;
esac
