#!/usr/bin/env python3
"""Breeze TTS 2 原始模型参考推理（CPU/bf16 eager，Voice Design / voice clone 可选）。

产出（--outdir）：
  - tokens.npz   : tokens [T,16] int32（有效帧，已截掉 EOS pad 帧）、tokens_with_eos、prompt_ids、audio(float32)
  - meta.json    : prompt/采样参数/耗时/token 统计/环境版本
  - out.wav      : 24 kHz PCM_16（与官方 infer.py 一致；--no-wav 可关）

用法示例：
  source ~/work/breeze-tts/env/activate.sh
  python ref_generate.py --outdir golden_zh \
      --text "今天天气不错，我们出去走走吧。" \
      --instruction "用温柔、亲切的女声，语速稍慢。" \
      --seed 42 --max-frames 100            # 采样模式（默认 do_sample）
  python ref_generate.py --outdir golden_greedy --greedy --max-frames 60 ...   # 贪心（对拍用）

注意：Breeze 官方 generate 的结论（详见 GEN_LOGIC.md）：
  - EOS 只认 backbone id 2051；depth decoder 无 EOS，固定 15 token；
  - 默认 cfg_scale=1.0 → 不做 CFG；cfg_scale != 1.0 时 voice design 走单 CFG（negative=去掉 instruction）；
  - 结果序列末尾 EOS 帧是整帧 codebook_pad_token_id(2050)，解码前截断。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

# 默认路径（20.38 上由 exp-2038 准备）
DEFAULT_SRC = Path.home() / "work/breeze-tts/data/breeze-tts-src"
DEFAULT_MODEL = Path.home() / "work/breeze-tts/data/breeze-tts-2"


def _log(msg: str) -> None:
    print(f"[ref_generate] {msg}", flush=True)


def _extract_audio(decoded) -> "object":
    """把 qwen_tts decode 输出规整成 1D numpy float32."""
    import numpy as np
    import torch

    if hasattr(decoded, "audio_values"):
        decoded = decoded.audio_values
    while isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    if isinstance(decoded, np.ndarray):
        arr = decoded
    else:
        if isinstance(decoded, torch.Tensor):
            arr = decoded.detach().float().cpu().numpy()
        else:
            raise TypeError(f"unsupported decoded audio type: {type(decoded)!r}")
    while arr.ndim > 1:
        arr = arr[0]
    return np.asarray(arr, dtype="float32")


def _peak_normalize(audio, target_peak: float = 0.95):
    import numpy as np

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        return audio * (target_peak / peak)
    return audio


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Breeze TTS 2 CPU reference generation")
    # 输入
    p.add_argument("--text", default="今天天气不错，我们一起出去走走吧。")
    p.add_argument("--instruction", default=None, help="Voice Design 指令；给了就走 tts_instruction")
    p.add_argument("--speaker", default="S0", help='speaker 前缀，默认 S0（空串则不加 [Sx]）')
    p.add_argument("--ref-audio", type=Path, default=None, help="可选：参考音频（voice clone）")
    p.add_argument("--ref-text", default=None, help="参考音频对应文本")
    # 生成
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-frames", type=int, default=100, help="最多生成音频帧数（12.5fps，100 帧 = 8s）")
    p.add_argument("--greedy", action="store_true", help="backbone+depth 全部贪心（对拍用；默认采样）")
    p.add_argument("--cfg-scale", type=float, default=1.0, help="1.0 = 无 CFG（默认）")
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--depth-temperature", type=float, default=0.9)
    p.add_argument("--depth-top-k", type=int, default=50)
    p.add_argument("--depth-top-p", type=float, default=1.0)
    # 运行
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--threads", type=int, default=0, help="torch CPU 线程数；0 = 默认")
    p.add_argument("--skip-decode", action="store_true", help="只出 token，不加载 codec 解码 wav（省时间/内存）")
    p.add_argument("--no-wav", action="store_true", help="解出音频但不写 wav")
    p.add_argument("--peak-normalize", action="store_true", help="wav 写盘前峰值归一化到 0.95（默认不归一化）")
    p.add_argument(
        "--save-scores",
        action="store_true",
        help="保存探针 probe.npz：前 N 帧 backbone cb0 logits top-k / post-norm hidden / text_embeds / depth 15 步 logits",
    )
    p.add_argument("--probe-frames", type=int, default=5, help="探针保存前多少帧（默认 5）")
    p.add_argument("--probe-topk", type=int, default=5, help="backbone logits 每帧保存 top-k（默认 5）")
    return p


def main() -> int:
    args = build_parser().parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.threads and args.threads > 0:
        import torch

        torch.set_num_threads(args.threads)
        _log(f"torch threads = {args.threads}")

    if not args.src_dir.is_dir():
        _log(f"ERROR: src dir not found: {args.src_dir}")
        return 2
    if not args.model.is_dir():
        _log(f"ERROR: model dir not found: {args.model}")
        return 2

    sys.path.insert(0, str(args.src_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import soundfile as sf
    import torch
    import transformers

    from models.breeze import BreezeForConditionalGeneration
    from breeze_infer.runtime import (
        set_all_seeds,
        update_generation_config_for_breeze,
    )
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name

    meta = {
        "tool": "ref_generate.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": args.device,
        "dtype": "torch.bfloat16" if args.dtype == "bf16" else "torch.float32",
        "model_dir": str(args.model.resolve()),
        "src_dir": str(args.src_dir.resolve()),
    }

    # ---------------- 1) 加载 tokenizer + model ----------------
    t0 = time.perf_counter()
    _log(f"loading tokenizer/model from {args.model} ({args.dtype}, eager) ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), fix_mistral_regex=False)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = BreezeForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(args.device).eval()
    load_time = time.perf_counter() - t0
    meta["load_time_s"] = round(load_time, 3)
    _log(f"model loaded in {load_time:.1f}s")

    cfg = model.config
    meta["config"] = {
        "num_codebooks": int(cfg.num_codebooks),
        "audio_vocab_size": int(cfg.vocab_size),
        "codebook_size": int(cfg.codec_config.codebook_size),
        "codebook_pad_token_id": int(cfg.codebook_pad_token_id),
        "codebook_eos_token_id": int(cfg.codebook_eos_token_id),
        "backbone_eos_token_id": int(cfg.vocab_size),
        "hidden_size": int(cfg.hidden_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "codec_sampling_rate": int(cfg.codec_config.sampling_rate),
        "codec_frame_rate": float(cfg.codec_config._frame_rate),
    }

    # ---------------- 2) 生成配置（与 infer.py / runtime.py 保持一致） ----------------
    update_generation_config_for_breeze(model)
    model.generation_config.max_new_tokens = int(args.max_frames)
    model.generation_config.repetition_penalty = float(args.repetition_penalty)
    model.generation_config.use_cache = True
    if args.greedy:
        model.generation_config.do_sample = False
        vars(model.depth_decoder.generation_config)["do_sample"] = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_k = 0
        model.generation_config.top_p = 1.0
    else:
        model.generation_config.do_sample = True
        model.generation_config.temperature = float(args.temperature)
        model.generation_config.top_k = int(args.top_k)
        model.generation_config.top_p = float(args.top_p)
        dcfg = model.depth_decoder.generation_config
        vars(dcfg)["do_sample"] = True
        dcfg.temperature = float(args.depth_temperature)
        dcfg.top_k = int(args.depth_top_k)
        dcfg.top_p = float(args.depth_top_p)
    depth_cfg = model.depth_decoder.generation_config
    meta["sampling"] = {
        "greedy": bool(args.greedy),
        "seed": int(args.seed),
        "max_frames": int(args.max_frames),
        "cfg_scale": float(args.cfg_scale),
        "backbone": {
            "do_sample": bool(model.generation_config.do_sample),
            "temperature": float(model.generation_config.temperature),
            "top_k": int(model.generation_config.top_k),
            "top_p": float(model.generation_config.top_p),
            "repetition_penalty": float(model.generation_config.repetition_penalty),
        },
        "depth_decoder": {
            "do_sample": bool(depth_cfg.do_sample),
            "temperature": float(depth_cfg.temperature),
            "top_k": int(depth_cfg.top_k),
            "top_p": float(depth_cfg.top_p),
            "min_new_tokens": int(depth_cfg.min_new_tokens) if depth_cfg.min_new_tokens else None,
            "max_new_tokens": int(depth_cfg.max_new_tokens) if depth_cfg.max_new_tokens else None,
        },
    }

    # ---------------- 2b) 探针（对拍定位用） ----------------
    # 记录：backbone 每帧 cb0 logits top-k（lm_head 输出）、post-norm hidden（depth 输入）、
    #       text_embeds（文本编码器输出=backbone prefill 输入）、depth 15 步 logits。
    probe_state = {
        "lm_head_calls": 0,
        "norm_calls": 0,
        "depth_head_calls": 0,
        "text_proj_calls": 0,
    }
    probe_hooks: list = []
    backbone_topk_val: list = []
    backbone_topk_idx: list = []
    hidden_norm_frames: list = []
    depth_logits_frames: list = []
    text_embeds_captured: list = []
    if args.save_scores:
        import torch as _torch

        n_frames_want = int(args.probe_frames)
        k = int(args.probe_topk)

        def _lm_head_hook(_module, _inputs, output):
            probe_state["lm_head_calls"] += 1
            if len(backbone_topk_val) >= n_frames_want or not _torch.is_tensor(output):
                return
            logits = output[0, -1].detach().float()
            vals, idx = _torch.topk(logits, min(k, logits.shape[-1]))
            backbone_topk_val.append(vals.cpu().numpy().astype("float32"))
            backbone_topk_idx.append(idx.cpu().numpy().astype("int32"))

        def _norm_hook(_module, _inputs, output):
            probe_state["norm_calls"] += 1
            if len(hidden_norm_frames) >= n_frames_want or not _torch.is_tensor(output):
                return
            hidden_norm_frames.append(
                output[0, -1].detach().float().cpu().numpy().astype("float32")
            )

        def _text_proj_hook(_module, _inputs, output):
            probe_state["text_proj_calls"] += 1
            if text_embeds_captured or not _torch.is_tensor(output):
                return
            out = output[0] if output.ndim == 3 else output
            text_embeds_captured.append(
                out.detach().float().cpu().numpy().astype("float32")
            )

        def _depth_head_hook(_module, _inputs, output):
            probe_state["depth_head_calls"] += 1
            if len(depth_logits_frames) >= n_frames_want * 15 or not _torch.is_tensor(output):
                return
            depth_logits_frames.append(
                output[0, -1].detach().float().cpu().numpy().astype("float16")
            )

        probe_hooks = [
            model.lm_head.register_forward_hook(_lm_head_hook),
            model.backbone_model.norm.register_forward_hook(_norm_hook),
            model.depth_decoder.codebooks_head.register_forward_hook(_depth_head_hook),
        ]
        if getattr(model, "text_encoder_proj", None) is not None:
            probe_hooks.append(
                model.text_encoder_proj.register_forward_hook(_text_proj_hook)
            )
        _log(
            f"probes enabled: first {n_frames_want} frames, backbone top-{k} logits + "
            "post-norm hidden + text_embeds + depth logits"
        )

    # ---------------- 3) prompt 构造 ----------------
    request = {"id": "ref", "text": args.text, "speaker": args.speaker}
    if args.instruction:
        request["instruction"] = args.instruction
    if args.ref_audio is not None:
        if not args.ref_text:
            _log("ERROR: --ref-audio requires --ref-text")
            return 2
        request["ref_audio_path"] = str(args.ref_audio)
        request["ref_text"] = args.ref_text
    template_name = select_template_name(request)
    meta["request"] = request
    meta["template"] = template_name
    _log(f"template={template_name} text_len={len(args.text)}")

    audio_tokenizer = None
    if args.ref_audio is not None:
        from qwen_tts import Qwen3TTSTokenizer

        _log("loading codec for reference audio encoding ...")
        audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(args.model / "audio_tokenizer"), device_map=args.device
        )

    inputs = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [request],
        get_template(template_name),
        guidance_scale=args.cfg_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    prompt_ids = inputs["input_ids"][0].tolist()
    meta["prompt"] = {
        "ids": prompt_ids,
        "len": len(prompt_ids),
        "rendered": tokenizer.decode(prompt_ids, skip_special_tokens=False),
        "has_input_values": inputs.get("input_values") is not None,
    }
    _log(f"prompt tokens = {len(prompt_ids)}")

    # ---------------- 4) 推理 ----------------
    # 与 infer.py 一致：prepare_inputs 之后 set seed，再 generate
    set_all_seeds(args.seed)
    gen_start = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, output_audio=False)
    except Exception:
        traceback.print_exc()
        return 3
    gen_time = time.perf_counter() - gen_start

    sequences = generated
    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences
    if not isinstance(sequences, torch.Tensor):
        sequences = torch.as_tensor(sequences)
    tokens_all = sequences[0].cpu().numpy().astype(np.int32)
    # 兼容 generate 返回 (frames,16) 的情况
    if tokens_all.ndim == 1:
        tokens_all = tokens_all.reshape(-1, int(cfg.num_codebooks))
    pad = int(cfg.codebook_pad_token_id)
    is_pad_frame = (tokens_all == pad).all(axis=-1)
    pad_idx = np.flatnonzero(is_pad_frame)
    eos_frame = int(pad_idx[0]) if pad_idx.size else -1
    tokens = tokens_all[:eos_frame] if eos_frame >= 0 else tokens_all
    n_frames = int(tokens.shape[0])

    meta["generation"] = {
        "gen_time_s": round(gen_time, 3),
        "frames": n_frames,
        "frames_with_eos_frame": int(tokens_all.shape[0]),
        "eos_frame_index": eos_frame,
        "eos_reached": bool(eos_frame >= 0),
        "frames_per_s": round(n_frames / gen_time, 2) if gen_time > 0 else None,
        "rtf": round(gen_time / (n_frames / float(cfg.codec_config._frame_rate)), 2) if n_frames else None,
        "audio_duration_s": round(n_frames / float(cfg.codec_config._frame_rate), 3),
        "tokens_min": int(tokens.min()) if n_frames else None,
        "tokens_max": int(tokens.max()) if n_frames else None,
        "frames_per_codebook_first16": [
            [int(x) for x in tokens[i].tolist()] for i in range(min(n_frames, 16))
        ],
        "last_frames": [[int(x) for x in tokens[i].tolist()] for i in range(max(0, n_frames - 4), n_frames)],
    }
    _log(
        f"generated {n_frames} frames ({meta['generation']['audio_duration_s']}s) in {gen_time:.1f}s "
        f"(eos_reached={eos_frame >= 0})"
    )

    # ---------------- 4b) 探针整理 ----------------
    probe_arrays: dict = {}
    if args.save_scores:
        for h in probe_hooks:
            h.remove()
        n_want = int(args.probe_frames)
        if backbone_topk_val:
            probe_arrays["backbone_topk_val"] = np.stack(backbone_topk_val)
            probe_arrays["backbone_topk_idx"] = np.stack(backbone_topk_idx)
        if hidden_norm_frames:
            hn = np.stack(hidden_norm_frames)
            probe_arrays["hidden_norm"] = hn
            meta["probe_hidden_norm_summary"] = [
                {
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "l2": float(np.linalg.norm(x)),
                    "first4": [float(v) for v in x[:4]],
                }
                for x in hn
            ]
        if text_embeds_captured:
            probe_arrays["text_embeds"] = text_embeds_captured[0]
        if depth_logits_frames:
            n_frames_avail = len(depth_logits_frames) // 15
            n_use = min(n_want, n_frames_avail)
            if n_use > 0:
                dl = np.stack(depth_logits_frames[: n_use * 15]).reshape(n_use, 15, -1)
                probe_arrays["depth_logits"] = dl
                probe_arrays["depth_top1"] = dl.argmax(axis=-1).astype("int32")
        meta["probe"] = {
            "enabled": True,
            "probe_frames": n_want,
            "probe_topk": int(args.probe_topk),
            "counts": probe_state,
            "expected_steps": int(tokens_all.shape[0]),
            "backbone_frames_captured": len(backbone_topk_val),
            "hidden_norm_frames_captured": len(hidden_norm_frames),
            "depth_frames_captured": len(depth_logits_frames) // 15,
            "text_embeds_shape": (
                list(probe_arrays["text_embeds"].shape) if "text_embeds" in probe_arrays else None
            ),
            "note": (
                "backbone_topk_* 第 i 行 = 生成第 i 帧时 backbone cb0 logits 的 top-k（值/索引，原始 logits 未经温度/掩码）；"
                "hidden_norm 第 i 行 = 第 i 帧 depth decoder 的输入（backbone 最终 RMSNorm 后 hidden，2048 维）；"
                "text_embeds = 文本编码器+proj 输出 [L,2048]（= backbone prefill 输入）；"
                "depth_logits [F,15,2051] 第 i 帧 15 步采样前 logits（已含 suppress 2048..2050）"
            ),
        }
        _log(
            "probe captured: lm_head={} norm={} depth_head={} text_proj={}".format(
                probe_state["lm_head_calls"],
                probe_state["norm_calls"],
                probe_state["depth_head_calls"],
                probe_state["text_proj_calls"],
            )
        )
    else:
        meta["probe"] = {"enabled": False}

    # ---------------- 5) codec 解码 ----------------
    audio = None
    decode_time = 0.0
    if not args.skip_decode:
        t1 = time.perf_counter()
        from qwen_tts import Qwen3TTSTokenizer

        _log("loading codec for decoding ...")
        if audio_tokenizer is None:
            audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
                str(args.model / "audio_tokenizer"), device_map=args.device
            )
        if n_frames > 0:
            codes_t = torch.as_tensor(tokens, dtype=torch.long)
            with torch.inference_mode():
                decoded = audio_tokenizer.decode({"audio_codes": [codes_t]})
            audio = _extract_audio(decoded)
        else:
            audio = np.zeros(0, dtype="float32")
        decode_time = time.perf_counter() - t1
        sr = int(cfg.codec_config.sampling_rate)
        meta["audio"] = {
            "sample_rate": sr,
            "samples": int(audio.shape[0]),
            "duration_s": round(audio.shape[0] / sr, 4),
            "decode_time_s": round(decode_time, 3),
            "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
            "rms": float(np.sqrt(np.mean(audio.astype("float64") ** 2))) if audio.size else 0.0,
        }
        if not args.no_wav and audio.size:
            wav_audio = _peak_normalize(audio) if args.peak_normalize else audio
            wav_path = outdir / "out.wav"
            sf.write(wav_path, np.clip(wav_audio, -1.0, 1.0), sr, subtype="PCM_16")
            _log(f"wav saved: {wav_path}")
    else:
        meta["audio"] = None

    # ---------------- 6) 落盘 ----------------
    save = {
        "tokens": tokens.astype(np.int32),
        "tokens_with_eos": tokens_all.astype(np.int32),
        "prompt_ids": np.asarray(prompt_ids, dtype=np.int32),
    }
    if audio is not None:
        save["audio"] = audio.astype(np.float32)
    np.savez_compressed(outdir / "tokens.npz", **save)
    if probe_arrays:
        np.savez_compressed(outdir / "probe.npz", **probe_arrays)
        _log(f"probe.npz saved ({outdir / 'probe.npz'}): keys={sorted(probe_arrays)}")

    meta["total_time_s"] = round(load_time + gen_time + decode_time, 3)
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    _log(f"tokens.npz + meta.json saved to {outdir}")
    print(json.dumps({k: meta[k] for k in ("template", "prompt", "generation", "audio")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
