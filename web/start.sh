#!/usr/bin/env bash
# Breeze TTS 2 · AX650 Web 服务启动脚本（在板子上运行）
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
mkdir -p jobs static
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "[start] already running pid=$(cat server.pid)"
  exit 0
fi
setsid nohup python3 breeze_web.py > server.log 2>&1 < /dev/null &
echo $! > server.pid
sleep 1.5
IP=$(ip -4 addr show eth1 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)
[ -z "$IP" ] && IP=192.168.10.105
echo "[start] pid=$(cat server.pid)  local: http://127.0.0.1:8010/  lan: http://$IP:8010/"
