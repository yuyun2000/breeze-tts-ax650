---
license: other
license_name: breezeblue-research-and-non-commercial-license
license_link: https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE
language:
- en
- zh
library_name: axera
tags:
- text-to-speech
- breeze-tts
- ax650
- axmodel
- npu
- axera
---

# Breeze TTS 2 — AX650 (AXMODEL)

[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)（约 3.45 B 参数、中英双语、Voice Design / Voice Clone）
在 **AX650/AX8850 NPU** 上的转换产物 + 板端纯 Python 运行时（numpy + pyaxengine）。

> ⚠️ **许可证**：本模型包是 Breeze TTS 2 权重的**衍生模型**，受
> **BreezeBlue Research and Non-Commercial License 1.1** 约束：**仅限研究与非商业用途，不可商用**。
> 转换/运行代码为 Apache-2.0。Qwen3-TTS tokenizer 组件为 Apache-2.0（QwenLM/Qwen3-TTS）。
> 转换工具链与部分现成二进制来自 AXERA（见 NOTICE）。

## 包内容

| 路径 | 说明 |
|---|---|
| `text-encoder/` | T5Gemma2（26L, hidden 1152）+ proj 的 2 段静态 axmodel（INT8）+ CPU embedding LUT（262158×1152 bf16）+ tokenizer |
| `talker/` | Qwen3 式 28 层（hidden 2048）per-layer axmodel + post（final norm + lm_head 2048→2052）；`audio_embed_16x2051x2048.bfloat16.bin`（16 组音频 embedding 表） |
| `code-predictor/` | Llama 式 12 层（hidden 1024, kv 256）per-layer axmodel + post(norm) + 15 个位置相关 lm_head axmodel；投影后音频表 15 组 + depth 投影矩阵 |
| `speech-tokenizer/` | Qwen3-TTS Tokenizer 12Hz decoder（decode_pre_inproj + 8 层 transformer + decode_post），**直接复用 AXERA 编译产物**（与 Breeze codec 权重字节一致） |
| `runtime/` | 板端 Python runtime（`run_breeze_tts.py`，numpy + pyaxengine + tokenizers） |
| `config/configuration.json` | 维度 / 文件名模板 / 采样默认值 |

## 板端使用

```bash
# 依赖：python3.10 + numpy + ml-dtypes + pyaxengine（AX650）
pip install axengine-0.1.3-py3-none-any.whl ml-dtypes tokenizers
export LD_LIBRARY_PATH=/soc/lib   # libax_engine.so

python3 runtime/run_breeze_tts.py \
  --text "你好，欢迎体验 AX650 上的 Breeze TTS 语音合成。" \
  --instruction "一位温柔自信的年轻女性，声音清晰，语气亲切。" \
  --out out.wav --dump-tokens tokens.npz
```

Voice Design 模式（无参考音频）。CFG 默认 1.0（不做 CFG）；贪心/采样均支持（采样参数与官方默认一致：
backbone temperature 0.9 / top_k 50；depth 同）。

## 实测（AX650C）

| 项 | 结果 |
|---|---|
| talker decode | 待填（约 0.2 s/帧，INT8） |
| depth decoder（15 步 × 12 层） | 待填（约 0.55–0.9 s/帧） |
| codec decode | 待填 |
| 整句 RTF | 待填（约 10× 实时） |
| 音质 | 与 PyTorch 参考 token 一致率 / ASR 转写见 GitHub 仓库报告 |

## 来源与引用

- 原始模型与权重：BreezeBlue/Breeze-TTS-2（BreezeBlue Research & Non-Commercial License）
- 转换工程：https://github.com/yuyun2000/breeze-tts-ax650
- AXERA Qwen3-TTS 转换脚本：AXERA-TECH/Qwen3-TTS.AXERA；AXERA Qwen3-TTS 编译产物：AXERA-TECH/Qwen3-TTS-12Hz-0.6B-Base-AX650
