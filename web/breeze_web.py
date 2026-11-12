#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breeze TTS 2 · AX650 局域网体验服务
- Flask Web/API 全部跑在板端；浏览器只需访问 http://<板IP>:8010/
- 推理调用板端 run_breeze_tts.py（全 NPU：textenc + talker + depth + codec）
- 单 worker 队列串行（CMM 有限），任务可轮询进度
"""
import json, os, random, re, signal, subprocess, sys, threading, time, uuid, wave
from collections import deque

from flask import Flask, jsonify, request, send_file, send_from_directory

APP_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(APP_DIR, "jobs")
STATIC_DIR = os.path.join(APP_DIR, "static")
os.makedirs(JOBS_DIR, exist_ok=True)

RUNTIME = "/root/breeze-tts/run_breeze_tts.py"
BREEZE_BASE = "/root/breeze-tts/breeze"
PYTHON = sys.executable or "python3"
PORT = int(os.environ.get("BREEZE_WEB_PORT", "8010"))
MAX_FRAMES = 300
JOB_TIMEOUT = 600.0
KEEP_JOBS = 40

INSTRUCTION_PRESETS = [
    {"id": "gentle_female", "label": "温柔女声", "text": "一位温柔自信的年轻女性，声音清晰，语气亲切。"},
    {"id": "news", "label": "新闻播报", "text": "一位专业新闻主播，发音标准，语速平稳，语气客观。"},
    {"id": "lively_girl", "label": "活泼少女", "text": "一位活泼可爱的少女，声音清脆，语气轻快。"},
    {"id": "calm_male", "label": "沉稳男声", "text": "一位沉稳的中年男性，声音低沉，语速稍慢，语气稳重。"},
    {"id": "story", "label": "讲故事", "text": "一位擅长讲童话故事的女声，娓娓道来，富有感情。"},
    {"id": "slow", "label": "放慢语速", "text": "一位温柔的女声，语速较慢，吐字清晰。"},
]

app = Flask(__name__, static_folder=None)
jobs = {}
jobs_lock = threading.Lock()
job_queue = deque()
queue_cv = threading.Condition()
_recent = deque(maxlen=30)


def now():
    return time.time()


def _cmm_info():
    try:
        with open("/proc/ax_proc/mem_cmm_info", "r", errors="ignore") as f:
            txt = f.read()
        used = remain = total = None
        m = re.search(r"total size=(\d+)KB", txt)
        if m:
            total = int(m.group(1))
        m = re.search(r"used=(\d+)KB", txt)
        if m:
            used = int(m.group(1))
        m = re.search(r"remain=(\d+)KB", txt)
        if m:
            remain = int(m.group(1))
        return {"total_mb": round(total / 1024, 1) if total else None,
                "used_mb": round(used / 1024, 1) if used else None,
                "remain_mb": round(remain / 1024, 1) if remain else None}
    except Exception as e:
        return {"error": str(e)}


def _meminfo():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0]) // 1024
        return {"total_mb": info.get("MemTotal"), "available_mb": info.get("MemAvailable")}
    except Exception:
        return {}


def _new_job(text, instruction, mode, max_frames, seed):
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = {
        "id": jid, "text": text, "instruction": instruction, "mode": mode,
        "max_frames": max_frames, "seed": seed,
        "state": "queued", "created": now(), "started": None, "finished": None,
        "elapsed": 0.0, "stage": "排队中", "frames": 0, "error": "",
        "timings": {}, "audio_sec": None, "n_frames": None, "rtf": None,
        "wav": os.path.join(JOBS_DIR, jid, "out.wav"),
        "tokens": os.path.join(JOBS_DIR, jid, "tokens.npz"),
        "log": os.path.join(JOBS_DIR, jid, "log.txt"),
        "proc": None, "cancel": False,
    }
    os.makedirs(os.path.join(JOBS_DIR, jid), exist_ok=True)
    with jobs_lock:
        jobs[jid] = job
        _recent.appendleft(jid)
        stale = list(_recent)[KEEP_JOBS:]
    for old in stale:
        with jobs_lock:
            j = jobs.pop(old, None)
        if j:
            try:
                import shutil
                shutil.rmtree(os.path.dirname(j["wav"]), ignore_errors=True)
            except Exception:
                pass
    return job


def _parse_log_progress(job):
    """从 runtime 日志解析阶段/帧进度/最终 TIMING。"""
    try:
        with open(job["log"], "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return
    job["log_tail"] = "".join(lines[-4:]).strip()
    frames = 0
    for ln in lines:
        m = re.search(r"frame (\d+):", ln)
        if m:
            frames = max(frames, int(m.group(1)) + 1)
        for pat, stage in (
            (r"load: talker", "加载 talker/depth/codec"),
            (r"breeze base:", "解析模型布局"),
            (r"text encoder:", "文本编码器"),
            (r"released text encoder", "释放文本编码器显存"),
            (r"talker prefill", "talker prefill"),
            (r"=+ TIMING =+", "汇总"),
        ):
            if re.search(pat, ln):
                job["stage"] = stage
    if frames:
        job["frames"] = frames
        job["stage"] = "生成音频帧 %d" % frames
    timings = {}
    for ln in lines:
        m = re.match(r"\s+(\w+)\s+([\d.]+) ms", ln)
        if m:
            timings[m.group(1)] = round(float(m.group(2)), 1)
        m = re.match(r"\s+RTF\s+([\d.]+) x", ln)
        if m:
            timings["RTF_x"] = float(m.group(1))
    if timings:
        job["timings"] = timings
    m = re.search(r"generated (\d+) frames \(([\d.]+)s audio\), eos_hit=(\w+)", "".join(lines))
    if m:
        job["n_frames"] = int(m.group(1))
        job["audio_sec"] = float(m.group(2))
        job["eos_hit"] = m.group(3) == "true"


def _worker():
    while True:
        with queue_cv:
            while not job_queue:
                queue_cv.wait(timeout=2.0)
            if not job_queue:
                continue
            jid = job_queue.popleft()
        job = jobs.get(jid)
        if not job or job["cancel"]:
            continue
        try:
            _run_job(job)
        except Exception as e:
            job["state"] = "error"
            job["error"] = "server error: %s" % e
            job["finished"] = now()


def _run_job(job):
    logd = os.path.dirname(job["log"])
    cmd = [PYTHON, RUNTIME,
           "--preset", "breeze",
           "--breeze-base", BREEZE_BASE,
           "--text", job["text"],
           "--instruction", job["instruction"],
           "--speaker", "S0",
           "--max-frames", str(job["max_frames"]),
           "--seed", str(job["seed"]),
           "--out-wav", job["wav"],
           "--dump-tokens", job["tokens"],
           "--lmhead-hp",
           "--log-every", "5"]
    if job["mode"] == "quality":
        cmd += ["--depth-mode", "hp"]
    env = dict(os.environ)
    env.setdefault("LD_LIBRARY_PATH", "/soc/lib")
    env["PYTHONUNBUFFERED"] = "1"
    job["state"] = "running"
    job["started"] = now()
    job["stage"] = "启动推理进程"
    with open(job["log"], "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=BREEZE_BASE, env=env,
                                preexec_fn=os.setsid)
        job["proc"] = proc
        t0 = now()
        while True:
            rc = proc.poll()
            job["elapsed"] = now() - t0
            _parse_log_progress(job)
            if rc is not None:
                break
            if job["elapsed"] > JOB_TIMEOUT:
                job["error"] = "推理超时(%ds)" % int(JOB_TIMEOUT)
                _graceful_stop(proc)
                rc = proc.poll()
                break
            if job["cancel"]:
                job["error"] = "已取消"
                _graceful_stop(proc)
                rc = proc.poll()
                break
            time.sleep(0.5)
    job["elapsed"] = round(now() - t0, 1)
    job["proc"] = None
    job["finished"] = now()
    if job.get("cancel"):
        job["state"] = "cancelled"
        return
    if rc != 0 or not os.path.exists(job["wav"]):
        job["state"] = "error"
        if not job["error"]:
            try:
                with open(job["log"], "r", errors="ignore") as f:
                    tail = f.read()[-1200:]
                job["error"] = "推理失败(rc=%s): %s" % (rc, tail.strip()[-500:])
            except Exception:
                job["error"] = "推理失败(rc=%s)" % rc
        return
    try:
        with wave.open(job["wav"], "rb") as w:
            job["audio_sec"] = round(w.getnframes() / float(w.getframerate()), 3)
            job["sample_rate"] = w.getframerate()
    except Exception:
        pass
    _parse_log_progress(job)
    if job["audio_sec"]:
        job["rtf"] = round(job["elapsed"] / job["audio_sec"], 2)
    job["state"] = "done"
    job["stage"] = "完成"


def _graceful_stop(proc):
    """先 SIGINT（让 Python 清理 axengine 会话，避免 CMM 泄漏），再 SIGTERM/SIGKILL。"""
    for sig, wait in ((signal.SIGINT, 8.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 3.0)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            try:
                proc.send_signal(sig)
            except Exception:
                return
        t = time.time()
        while time.time() - t < wait:
            if proc.poll() is not None:
                return
            time.sleep(0.2)


def _job_view(job, full=False):
    d = {k: job[k] for k in ("id", "text", "instruction", "mode", "state", "stage",
                             "elapsed", "frames", "error", "timings", "audio_sec",
                             "n_frames", "rtf", "seed", "max_frames")}
    d["has_audio"] = os.path.exists(job["wav"])
    d["has_tokens"] = os.path.exists(job["tokens"])
    d["audio_url"] = "/api/jobs/%s/audio.wav" % job["id"]
    d["tokens_url"] = "/api/jobs/%s/tokens.npz" % job["id"]
    d["download_url"] = "/api/jobs/%s/audio.wav?download=1" % job["id"]
    if full:
        d["log_tail"] = job.get("log_tail", "")
        d["log_url"] = "/api/jobs/%s/log.txt" % job["id"]
    return d


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:fn>")
def static_files(fn):
    return send_from_directory(STATIC_DIR, fn)


@app.route("/health")
def health():
    with jobs_lock:
        states = {}
        for j in jobs.values():
            states[j["state"]] = states.get(j["state"], 0) + 1
    return jsonify({
        "ok": True, "service": "breeze-tts-ax650-web", "version": "1.0",
        "model": "Breeze TTS 2 (3.45B) · AX650 全 NPU",
        "cmm": _cmm_info(), "mem": _meminfo(),
        "queue": len(job_queue), "jobs": states, "uptime_s": round(now() - START_TIME, 1),
    })


@app.route("/api/presets")
def api_presets():
    return jsonify({
        "instructions": INSTRUCTION_PRESETS,
        "modes": [
            {"id": "fast", "label": "快速", "desc": "depth INT8 + U16 lm_head，约 12-16× 实时"},
            {"id": "quality", "label": "高质量", "desc": "depth FP32 版，约 2.5× 慢"},
        ],
        "default_text": "你好，欢迎体验 AX650 上的 Breeze TTS 语音合成。",
        "samples": [
            {"label": "中文示例（此前生成）", "url": "/static/sample_zh.wav"},
            {"label": "英文示例（此前生成）", "url": "/static/sample_en.wav"},
        ],
    })


@app.route("/api/tts", methods=["POST"])
def api_tts():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    instruction = str(data.get("instruction", "")).strip()
    mode = "quality" if data.get("mode") == "quality" else "fast"
    try:
        max_frames = int(data.get("max_frames") or 120)
    except Exception:
        max_frames = 120
    max_frames = max(20, min(MAX_FRAMES, max_frames))
    if not text:
        return jsonify({"error": "text 不能为空"}), 400
    if len(text) > 300:
        return jsonify({"error": "文本过长（最多 300 字）"}), 400
    if len(instruction) > 200:
        return jsonify({"error": "风格描述过长（最多 200 字）"}), 400
    if not instruction:
        instruction = INSTRUCTION_PRESETS[0]["text"]
    seed = random.randint(1, 2**31 - 1)
    job = _new_job(text, instruction, mode, max_frames, seed)
    with queue_cv:
        job_queue.append(job["id"])
        queue_cv.notify()
    pos = len(job_queue)
    return jsonify({"id": job["id"], "state": "queued", "queue_position": pos})


@app.route("/api/jobs")
def api_jobs():
    with jobs_lock:
        ids = list(_recent)[:20]
    out = []
    for jid in ids:
        job = jobs.get(jid)
        if job:
            out.append(_job_view(job))
    return jsonify({"jobs": out})


@app.route("/api/jobs/<jid>")
def api_job(jid):
    job = jobs.get(jid)
    if not job:
        return jsonify({"error": "not found"}), 404
    with jobs_lock:
        pos = list(job_queue).index(jid) + 1 if jid in job_queue else 0
    d = _job_view(job, full=True)
    d["queue_position"] = pos
    return jsonify(d)


@app.route("/api/jobs/<jid>/cancel", methods=["POST"])
def api_cancel(jid):
    job = jobs.get(jid)
    if not job:
        return jsonify({"error": "not found"}), 404
    job["cancel"] = True
    return jsonify({"ok": True, "state": job["state"]})


@app.route("/api/jobs/<jid>/audio.wav")
def api_audio(jid):
    job = jobs.get(jid)
    if not job or not os.path.exists(job["wav"]):
        return jsonify({"error": "not found"}), 404
    dl = request.args.get("download") == "1"
    return send_file(job["wav"], mimetype="audio/wav", as_attachment=dl,
                     download_name="breeze_tts_%s.wav" % jid)


@app.route("/api/jobs/<jid>/tokens.npz")
def api_tokens(jid):
    job = jobs.get(jid)
    if not job or not os.path.exists(job["tokens"]):
        return jsonify({"error": "not found"}), 404
    return send_file(job["tokens"], mimetype="application/octet-stream", as_attachment=True,
                     download_name="tokens_%s.npz" % jid)


@app.route("/api/jobs/<jid>/log.txt")
def api_log(jid):
    job = jobs.get(jid)
    if not job or not os.path.exists(job["log"]):
        return jsonify({"error": "not found"}), 404
    return send_file(job["log"], mimetype="text/plain")


START_TIME = time.time()
threading.Thread(target=_worker, daemon=True).start()

if __name__ == "__main__":
    print("[breeze-web] listening on 0.0.0.0:%d" % PORT)
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False, use_reloader=False)
