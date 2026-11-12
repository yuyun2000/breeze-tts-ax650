# 局域网网页体验服务（板端自托管）

纯板端服务：Python HTTP/API + 静态网页都在 AX650 上，局域网内其它设备只需浏览器访问
`http://<板卡IP>:8010/`，客户端不需要任何本地服务。

## 部署（板端）

```bash
# 假设模型包在 /root/breeze-tts/breeze，runtime 在 /root/breeze-tts/run_breeze_tts.py
mkdir -p /root/breeze-tts/web
cp -r breeze_web.py start.sh static /root/breeze-tts/web/
cd /root/breeze-tts/web && bash start.sh
# 输出: lan: http://192.168.10.105:8010/
```

依赖：系统 python3（numpy / ml_dtypes / tokenizers / axengine / Flask）、`LD_LIBRARY_PATH=/soc/lib`。
`start.sh` 用 `setsid nohup` 常驻（SSH 断开不影响），并写入 `server.pid` / `server.log`。

### 常驻（warm）引擎：模型只加载一次

`breeze_warm_engine.py` 把 talker/depth/codec/text-encoder 与 embedding 表**一次性加载**并常驻
（`BREEZE_WARM=1` 默认开启），之后每次请求只做 textenc → prefill → 逐帧生成 → codec，省掉约 22–24 s 的模型加载。

| 模式 | 单句 5 s 音频耗时 | 说明 |
|---|---|---|
| 常驻 fast | **~42 s（RTF ≈ 8.2×）** | 模型常驻；CMM 约 4.3 GB |
| 常驻 fast（重启后首个请求） | ~68 s | 含 warm 加载 25 s |
| quality | ~71 s（3.5 s 音频，含切 hp 加载 ~15 s） | FP32 depth 常驻切换；任务结束自动切回 fast 释放 1.3 GB |

常驻内存预算（实测 CMM 6 GB 中）：textenc 0.77 GB + talker 1.67 GB + cp 0.4 GB + codec 0.16 GB +
embedding 表 0.27 GB ≈ 3.3 GB，加其它服务约 1 GB，remain ~1.8 GB。
若加载失败（CMM 不足），服务自动回退为「每请求子进程」模式，功能不受影响。

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网页 UI（文本 + 音色风格 + 快速/高质量模式） |
| GET | `/health` | 服务状态：CMM 使用量、内存、队列、任务统计 |
| GET | `/api/presets` | 音色风格预设、模式说明、示例音频 |
| POST | `/api/tts` | 提交任务 `{text, instruction, mode: fast|quality, max_frames}` → `{id}` |
| GET | `/api/jobs/<id>` | 任务状态/进度/耗时分解（`state/stage/elapsed/frames/timings`） |
| GET | `/api/jobs` | 最近 20 个任务 |
| POST | `/api/jobs/<id>/cancel` | 取消（先 SIGINT，保证 axengine 会话清理、不泄漏 CMM） |
| GET | `/api/jobs/<id>/audio.wav` | 下载合成音频（`?download=1` 附件） |
| GET | `/api/jobs/<id>/tokens.npz` | 音频 codes / prompt ids / meta |
| GET | `/api/jobs/<id>/log.txt` | 完整 runtime 日志 |

## 说明

- 单 worker 串行队列（NPU 单实例）；常驻模式下每句约 **42–55 s**（5–6 s 音频，RTF ≈ 8–9×），进度页显示已生成帧数。
- `/health` 的 `warm.state`（loading/ready/error/off）与 `warm.load_s` 可看常驻引擎状态；CMM 不足时自动回退子进程。
- 每次请求由独立子进程运行 `run_breeze_tts.py`（全 NPU：textenc → talker → depth → codec），
  进程退出即回收 CMM；取消/超时用 SIGINT→SIGTERM→SIGKILL 顺序，避免 CMM 泄漏（板端踩过坑）。
- 已配置 `@reboot` crontab 自启；手动重启：`bash /root/breeze-tts/web/start.sh`。
