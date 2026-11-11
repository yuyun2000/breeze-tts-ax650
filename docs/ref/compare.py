#!/usr/bin/env python3
"""Breeze TTS 2 端到端对拍工具：golden vs 板端/重打包产物。

支持两类输入，可只做一类：
  1) codebook tokens：[T,16]（npz/npy/json/txt）→ 逐帧/逐 codebook 一致率、首分叉帧、全局对齐偏移
  2) 音频 wav：golden wav vs candidate wav → 采样率统一、时长、RMS/包络、log-MEL L1/相关/cosine（含最优对齐 lag）

用法：
  # 只对 token
  python compare.py --golden ~/work/breeze-tts/ref/golden_zh --cand board_tokens.npz --out-dir ~/work/breeze-tts/ref/cmp_board
  # token + wav（golden 目录里有 out.wav，candidate 传 wav）
  python compare.py --golden golden_zh/ --cand-codes cand.npz --cand-wav cand.wav --out-dir cmp/
  # 只对 wav
  python compare.py --golden-wav golden.wav --cand-wav cand.wav --out-dir cmp/

输出：cmp/report.md + cmp/report.json（终端打印摘要）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- utils


def _log(msg: str) -> None:
    print(f"[compare] {msg}", flush=True)


# ---------------------------------------------------------------- token IO

_TOKEN_KEYS = ("tokens", "audio_codes", "codes", "token_ids", "codebook_ids", "frames", "cb")


def _as_token_matrix(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"tokens must be 2D [T,C], got shape {arr.shape}")
    if arr.dtype.kind == "f":
        arr = np.rint(arr)
    return arr.astype(np.int64)


def load_tokens(path: Path) -> tuple[np.ndarray, dict]:
    """从 npz/npy/json/txt 读 tokens；返回 [T,C] int64 + 附带字段 dict。"""
    extra: dict = {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            chosen = None
            for k in _TOKEN_KEYS:
                if k in z.files:
                    chosen = k
                    break
            if chosen is None:
                # 兜底：选第一个 2D/1D 数组
                for k in z.files:
                    a = z[k]
                    if a.ndim in (1, 2) and a.dtype.kind in "iu":
                        chosen = k
                        break
            if chosen is None:
                raise ValueError(f"no token-like array in {path}: keys={z.files}")
            tokens = _as_token_matrix(z[chosen])
            for k in z.files:
                if k == chosen:
                    continue
                if k == "prompt_ids":
                    extra["prompt_ids"] = z[k]
                elif k in ("tokens_with_eos",):
                    extra["tokens_with_eos"] = _as_token_matrix(z[k])
            extra["key"] = chosen
    elif path.suffix == ".npy":
        tokens = _as_token_matrix(np.load(path, allow_pickle=False))
        extra["key"] = "npy"
    elif path.suffix in (".json", ".js"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for k in _TOKEN_KEYS:
                if k in obj:
                    tokens = _as_token_matrix(obj[k])
                    break
            else:
                raise ValueError(f"no token-like key in {path}: {list(obj)[:10]}")
            if "prompt_ids" in obj:
                extra["prompt_ids"] = np.asarray(obj["prompt_ids"])
        else:
            tokens = _as_token_matrix(obj)
        extra["key"] = "json"
    else:  # txt/csv：每行一帧
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().strip(",")
            if not line:
                continue
            rows.append([int(float(x)) for x in line.replace(",", " ").split()])
        tokens = _as_token_matrix(rows)
        extra["key"] = "txt"
    return tokens, extra


# ---------------------------------------------------------------- audio IO

def load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype="float32"), int(sr)


def resample_linear(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or audio.size == 0:
        return audio.astype("float32")
    n_out = int(round(audio.shape[0] * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype="float32")
    x_in = np.arange(audio.shape[0], dtype=np.float64)
    x_out = np.linspace(0.0, audio.shape[0] - 1.0, n_out, dtype=np.float64)
    return np.interp(x_out, x_in, audio.astype(np.float64)).astype("float32")


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype("float64") ** 2)))


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def best_lag(a: np.ndarray, b: np.ndarray, sr: int, max_lag_s: float = 0.5) -> int:
    """在枚举 lag（样本）上最大化归一化互相关；返回 b 相对 a 的偏移（b 后移为正）。"""
    if a.size == 0 or b.size == 0:
        return 0
    max_lag = int(max_lag_s * sr)
    n = min(a.size, b.size)
    if n <= 8:
        return 0
    a0 = a[:n].astype(np.float64)
    a0 = a0 - a0.mean()
    na = np.linalg.norm(a0)
    if na <= 0:
        return 0
    best, best_r = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a0[: n - lag], b[lag:n].astype(np.float64)
        else:
            x, y = a0[-lag:n], b[: n + lag].astype(np.float64)
        if x.size < 16:
            continue
        y = y - y.mean()
        ny = np.linalg.norm(y)
        if ny <= 0:
            continue
        r = float(np.dot(x, y) / (na * ny))
        if r > best_r:
            best_r, best = r, lag
    return best


# ---------------------------------------------------------------- mel

def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    hz = mel_to_hz(mels)
    fb = np.zeros((n_mels, freqs.size), dtype=np.float64)
    for i in range(n_mels):
        lo, ctr, hi = hz[i], hz[i + 1], hz[i + 2]
        left = (freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(left, right))
    return fb


def log_mel(audio: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256, n_mels: int = 80) -> np.ndarray:
    """numpy 实现 log-mel（[n_mels, frames]，dB）。优先 librosa，失败则 numpy 兜底。"""
    try:
        import librosa

        m = librosa.feature.melspectrogram(
            y=audio.astype("float32"), sr=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, fmin=0.0, fmax=sr / 2.0, power=2.0,
        )
        return 10.0 * np.log10(np.maximum(m, 1e-10))
    except Exception:
        if audio.size < n_fft:
            audio = np.pad(audio, (0, n_fft - audio.size))
        win = np.hanning(n_fft)
        n_frames = 1 + (audio.size - n_fft) // hop
        fb = _mel_filterbank(sr, n_fft, n_mels, 0.0, sr / 2.0)
        out = np.empty((n_mels, n_frames), dtype=np.float64)
        for i in range(n_frames):
            seg = audio[i * hop : i * hop + n_fft] * win
            spec = np.abs(np.fft.rfft(seg)) ** 2
            out[:, i] = fb @ spec
        return 10.0 * np.log10(np.maximum(out, 1e-10))


def compare_mel(a: np.ndarray, b: np.ndarray, sr: int) -> dict:
    ma, mb = log_mel(a, sr), log_mel(b, sr)
    n = min(ma.shape[1], mb.shape[1])
    if n <= 0:
        return {"error": "empty mel"}
    ma, mb = ma[:, :n], mb[:, :n]
    # 去掉整体静音偏移（幅度域），只比较结构
    a_c = ma - ma.mean()
    b_c = mb - mb.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    cosine = float(np.dot(a_c.ravel(), b_c.ravel()) / denom) if denom > 0 else None
    corr = None
    if a_c.std() > 0 and b_c.std() > 0:
        corr = float(np.corrcoef(a_c.ravel(), b_c.ravel())[0, 1])
    l1 = float(np.mean(np.abs(ma - mb)))
    # 仅比较有声帧（golden 帧能量 > 全集 50 分位）
    energy = ma.mean(axis=0)
    thr = np.quantile(energy, 0.5)
    voiced = energy > thr
    l1_voiced = float(np.mean(np.abs(ma[:, voiced] - mb[:, voiced]))) if voiced.any() else None
    return {
        "n_mels": int(ma.shape[0]),
        "frames": int(n),
        "logmel_cosine": cosine,
        "logmel_pearson": corr,
        "logmel_l1_db": l1,
        "logmel_l1_db_voiced": l1_voiced,
    }


# ---------------------------------------------------------------- token compare


def compare_tokens(g: np.ndarray, c: np.ndarray, n_codebooks: int) -> dict:
    """逐帧/逐 codebook 对拍 + 全局偏移搜索。返回 dict."""
    # 若是整包 roundtrip 的 golden（两侧可能都含 EOS pad 帧），调用方已裁过
    gt, ct = g.shape[0], c.shape[0]
    T = min(gt, ct)
    out: dict = {"golden_frames": int(gt), "cand_frames": int(ct), "overlap_frames": int(T)}

    if T > 0:
        gg, cc = g[:T], c[:T]
        e = gg == cc
        out["exact_frame_rate"] = float(e.all(axis=1).mean())
        out["codebook_acc"] = [float(x) for x in e.mean(axis=0)]
        out["codebook_acc_mean"] = float(e.mean())
        # 首分叉帧（任一 codebook 不一致）
        first_bad = int(np.flatnonzero(~e.all(axis=1))[0]) if (~e.all(axis=1)).any() else -1
        out["first_divergence_frame"] = first_bad
        out["identical_prefix_frames"] = first_bad if first_bad >= 0 else int(T)
        # cb0 单独（对板端最重要）
        out["cb0_acc"] = float(e[:, 0].mean())
        out["cb0_first_divergence_frame"] = (
            int(np.flatnonzero(g[:T, 0] != c[:T, 0])[0]) if (g[:T, 0] != c[:T, 0]).any() else -1
        )
        # 不匹配统计
        if not e.all():
            diff = np.abs(gg - cc)[~e]
            out["mismatch_abs_diff_mean"] = float(diff.mean())
            out["mismatch_count"] = int((~e).sum())
            per_cb_bad = (~e).sum(axis=0).tolist()
            out["mismatch_per_codebook"] = per_cb_bad
            out["worst_codebook"] = int(np.argmax(per_cb_bad))

    # 全局对齐偏移搜索：candidate 相对 golden 的帧偏移
    best = {"offset": 0, "acc": None, "frames": 0}
    max_off = min(32, max(0, min(gt, ct) - 1))
    for off in range(-max_off, max_off + 1):
        if off >= 0:
            n = min(gt - off, ct)
            a, b = g[off : off + n], c[:n]
        else:
            n = min(gt, ct + off)
            a, b = g[:n], c[-off : -off + n]
        if n <= 0:
            continue
        acc = float((a == b).mean())
        if best["acc"] is None or acc > best["acc"]:
            best = {"offset": int(off), "acc": acc, "frames": int(n)}
    out["best_offset"] = best
    return out


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Breeze TTS 2 golden vs candidate compare")
    p.add_argument("--golden", type=Path, default=None, help="golden 目录（含 tokens.npz/out.wav/meta.json）或 tokens 文件")
    p.add_argument("--golden-codes", type=Path, default=None)
    p.add_argument("--golden-wav", type=Path, default=None)
    p.add_argument("--cand", type=Path, default=None, help="candidate tokens 文件，或候选目录")
    p.add_argument("--cand-codes", type=Path, default=None)
    p.add_argument("--cand-wav", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-codebooks", type=int, default=0, help="默认从 golden 推断（16）")
    p.add_argument("--max-lag-ms", type=float, default=500.0)
    return p


def _resolve_golden(args):
    gdir = args.golden
    codes, wav, meta = args.golden_codes, args.golden_wav, None
    if gdir is not None:
        if gdir.is_dir():
            if codes is None and (gdir / "tokens.npz").is_file():
                codes = gdir / "tokens.npz"
            if wav is None:
                for name in ("out.wav", "golden.wav", "audio.wav"):
                    if (gdir / name).is_file():
                        wav = gdir / name
                        break
            if (gdir / "meta.json").is_file():
                meta = json.loads((gdir / "meta.json").read_text(encoding="utf-8"))
        elif codes is None:
            codes = gdir
    return codes, wav, meta


def _resolve_cand(args):
    c = args.cand
    codes, wav = args.cand_codes, args.cand_wav
    if c is not None:
        if c.is_dir():
            if codes is None:
                for name in ("tokens.npz", "cand.npz", "out_tokens.npz", "codes.npz"):
                    if (c / name).is_file():
                        codes = c / name
                        break
            if wav is None:
                for name in ("out.wav", "cand.wav", "audio.wav"):
                    if (c / name).is_file():
                        wav = c / name
                        break
        elif codes is None:
            codes = c
    return codes, wav


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    golden_codes_path, golden_wav_path, golden_meta = _resolve_golden(args)
    cand_codes_path, cand_wav_path = _resolve_cand(args)

    report: dict = {
        "golden": {"codes": str(golden_codes_path) if golden_codes_path else None,
                   "wav": str(golden_wav_path) if golden_wav_path else None},
        "candidate": {"codes": str(cand_codes_path) if cand_codes_path else None,
                      "wav": str(cand_wav_path) if cand_wav_path else None},
        "token_compare": None,
        "audio_compare": None,
    }
    md: list[str] = ["# Breeze TTS 2 对拍报告", ""]
    md.append(f"- golden codes: `{golden_codes_path}`")
    md.append(f"- golden wav  : `{golden_wav_path}`")
    md.append(f"- cand codes  : `{cand_codes_path}`")
    md.append(f"- cand wav    : `{cand_wav_path}`")
    md.append("")

    n_cb = args.n_codebooks or 16
    if golden_meta and golden_meta.get("config", {}).get("num_codebooks"):
        n_cb = int(golden_meta["config"]["num_codebooks"])
        report["n_codebooks"] = n_cb

    # --- token ---
    if golden_codes_path is not None and cand_codes_path is not None:
        g, g_extra = load_tokens(golden_codes_path)
        c, c_extra = load_tokens(cand_codes_path)
        n_cb = max(n_cb, g.shape[1], c.shape[1])
        # 若两侧不同宽，报错（不能对拍）
        if g.shape[1] != c.shape[1]:
            report["token_compare"] = {
                "error": f"width mismatch golden={g.shape[1]} cand={c.shape[1]}"
            }
        else:
            tc = compare_tokens(g, c, n_cb)
            # prompt 对拍
            gp = g_extra.get("prompt_ids")
            cp = c_extra.get("prompt_ids")
            if gp is not None and cp is not None:
                gp, cp = np.asarray(gp).ravel(), np.asarray(cp).ravel()
                tc["prompt"] = {
                    "golden_len": int(gp.size),
                    "cand_len": int(cp.size),
                    "same": bool(gp.size == cp.size and np.array_equal(gp, cp)),
                    "first_diff": int(np.flatnonzero(gp != cp)[0]) if gp.size == cp.size and not np.array_equal(gp, cp) else -1,
                }
            if "tokens_with_eos" in g_extra or "tokens_with_eos" in c_extra:
                tc["raw_frames"] = {
                    "golden": int(g_extra.get("tokens_with_eos", g).shape[0]),
                    "cand": int(c_extra.get("tokens_with_eos", c).shape[0]),
                }
            report["token_compare"] = tc

            md += ["## 1. Codebook tokens 对拍", ""]
            if "error" in tc:
                md.append(f"- **错误**: {tc['error']}")
            else:
                md.append(f"- golden 帧数 / cand 帧数 / 重叠: **{tc['golden_frames']} / {tc['cand_frames']} / {tc['overlap_frames']}**")
                md.append(f"- 整帧完全一致率: **{tc['exact_frame_rate']:.4f}**；codebook 全量一致率: **{tc['codebook_acc_mean']:.4f}**")
                md.append(f"- cb0 一致率: **{tc['cb0_acc']:.4f}**；第一处不一致帧（任一 cb）: **{tc['first_divergence_frame']}**")
                md.append(f"- 完全一致前缀帧数: **{tc['identical_prefix_frames']}**")
                md.append(f"- 最优全局帧偏移: {tc['best_offset']['offset']}（重叠 {tc['best_offset']['frames']} 帧，一致率 {tc['best_offset']['acc']:.4f}）")
                if tc.get("mismatch_per_codebook"):
                    md.append(f"- 各 codebook 不匹配次数: `{tc['mismatch_per_codebook']}`（最差 cb#{tc['worst_codebook']}，平均 |Δtoken|={tc['mismatch_abs_diff_mean']:.2f}）")
                if tc.get("prompt"):
                    p_ = tc["prompt"]
                    md.append(f"- prompt ids 一致: **{p_['same']}**（golden {p_['golden_len']} / cand {p_['cand_len']}，首差 {p_['first_diff']}）")
            md.append("")
            _log(f"token compare: exact_frame_rate={tc.get('exact_frame_rate')} first_div={tc.get('first_divergence_frame')}")
    else:
        _log("skip token compare (missing golden or cand codes)")

    # --- audio ---
    if golden_wav_path is not None and cand_wav_path is not None:
        ga, gsr = load_wav(golden_wav_path)
        ca, csr = load_wav(cand_wav_path)
        target_sr = int(golden_meta["audio"]["sample_rate"]) if golden_meta and golden_meta.get("audio") else gsr
        ga2 = resample_linear(ga, gsr, target_sr)
        ca2 = resample_linear(ca, csr, target_sr)
        ac: dict = {
            "golden_sr": gsr, "cand_sr": csr, "target_sr": target_sr,
            "golden_samples": int(ga.size), "cand_samples": int(ca.size),
            "golden_duration_s": round(ga.size / gsr, 4),
            "cand_duration_s": round(ca.size / csr, 4),
            "duration_diff_s": round(ca.size / csr - ga.size / gsr, 4),
            "golden_rms": rms(ga), "cand_rms": rms(ca),
            "golden_peak": peak(ga), "cand_peak": peak(ca),
        }
        lag = best_lag(ga2, ca2, target_sr, max_lag_s=args.max_lag_ms / 1000.0)
        ac["best_lag_samples"] = int(lag)
        ac["best_lag_ms"] = round(lag * 1000.0 / target_sr, 2)
        n = min(ga2.size, ca2.size)
        if lag >= 0:
            a, b = ga2[: n - lag], ca2[lag:n]
        else:
            a, b = ga2[-lag:n], ca2[: n + lag]
        ac["aligned_overlap_samples"] = int(a.size)
        if a.size > 8:
            aa = a.astype(np.float64) - a.mean()
            bb = b.astype(np.float64) - b.mean()
            den = np.linalg.norm(aa) * np.linalg.norm(bb)
            ac["waveform_pearson"] = float(np.dot(aa, bb) / den) if den > 0 else None
            ac["aligned_rms_ratio"] = (rms(b) / rms(a)) if rms(a) > 0 else None
        ac.update(compare_mel(a, b, target_sr) if a.size > 1024 else {"mel_error": "too short"})
        report["audio_compare"] = ac

        md += ["## 2. 音频对拍", ""]
        md.append(f"- 采样率: golden {gsr} / cand {csr} → 统一 {target_sr}")
        md.append(f"- 时长: **{ac['golden_duration_s']} s / {ac['cand_duration_s']} s**（差 {ac['duration_diff_s']:+.3f} s）")
        md.append(f"- RMS: {ac['golden_rms']:.4f} / {ac['cand_rms']:.4f}；峰值: {ac['golden_peak']:.3f} / {ac['cand_peak']:.3f}")
        md.append(f"- 最优对齐 lag: **{ac['best_lag_ms']} ms**（{ac['best_lag_samples']} samples）")
        if "waveform_pearson" in ac:
            md.append(f"- 对齐后波形相关系数: **{ac['waveform_pearson']:.4f}**；RMS 比: {ac['aligned_rms_ratio']:.3f}")
        if "logmel_cosine" in ac and ac["logmel_cosine"] is not None:
            md.append(
                f"- log-MEL: cosine **{ac['logmel_cosine']:.4f}**，Pearson **{ac['logmel_pearson']:.4f}**，"
                f"L1 **{ac['logmel_l1_db']:.2f} dB**（有声帧 {ac['logmel_l1_db_voiced']:.2f} dB，{ac['frames']} 帧）"
            )
        md.append("")
        _log(f"audio compare: logmel_cosine={ac.get('logmel_cosine')} lag_ms={ac['best_lag_ms']}")
    else:
        _log("skip audio compare (missing golden or cand wav)")

    # --- 总结 ---
    md += ["## 3. 结论", ""]
    verdict = []
    tc, ac = report.get("token_compare"), report.get("audio_compare")
    if tc and "error" not in tc:
        if tc["exact_frame_rate"] >= 0.99:
            verdict.append("token 一致率 ≥0.99：pipeline 数值等价（可认为对拍通过）")
        elif tc["exact_frame_rate"] >= 0.8:
            verdict.append(f"token 一致率 {tc['exact_frame_rate']:.3f}：主流程正确，存在数值抖动导致的后续分叉（首分叉帧 {tc['first_divergence_frame']}）")
        else:
            verdict.append(f"token 一致率 {tc['exact_frame_rate']:.3f} 偏低：需排查接口/掩码/采样参数（首分叉帧 {tc['first_divergence_frame']}）")
    if ac and ac.get("logmel_cosine") is not None:
        verdict.append(f"音频 log-MEL cosine {ac['logmel_cosine']:.3f}，对齐 lag {ac['best_lag_ms']} ms")
    if not verdict:
        verdict.append("输入不完整，未产生有效对比")
    md += [f"- {v}" for v in verdict] + [""]
    report["verdict"] = verdict

    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _log(f"report saved: {args.out_dir/'report.md'} / report.json")
    print("\n".join(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
