---
license: other
license_name: breezeblue-research-and-non-commercial-license
license_link: https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE
language:
- en
- zh
library_name: axera
base_model: BreezeBlue/Breeze-TTS-2
tags:
- text-to-speech
- breeze-tts
- ax650
- axmodel
- npu
- axera
- int8
---

# Breeze TTS 2 — AX650 / AX8850 (AXMODEL)

[Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)（约 3.45 B 参数、中英双语、开源 TTS 榜单第一）
在 **AX650/AX8850 NPU** 上的转换产物 + 板端纯 Python 运行时（numpy + pyaxengine，无 PyTorch）。

> ⚠️ **许可证**：本模型包是 Breeze TTS 2 权重的**衍生模型**，受
> **BreezeBlue Research and Non-Commercial License 1.1** 约束：**仅限研究与非商业用途，不可商用**（见 `BREEZEBLUE_LICENSE.txt`）。
> 转换/运行代码 Apache-2.0。Qwen3-TTS tokenizer 组件版权归 QwenLM（Apache-2.0）。
> `speech-tokenizer/` 中的 axmodel 复用自 AXERA 公开仓库（见 `NOTICE`）。

## 包内容（约 4.5 GB）

| 路径 | 说明 | 体积 |
|---|---|---|
| `talker/` | Qwen3 式 28 层（hidden 2048）per-layer axmodel + post（final norm + lm_head 2048→2052） | 1.63 GB |
| `code-predictor/` | Llama 式 12 层（hidden 1024, kv 256）per-layer axmodel + post(norm) + 15 个位置相关 lmmodel（INT8）；U16 版在 `lm-head-hp/` | 0.40 GB |
| `embeds/` | 16 组未投影音频 embedding 表（talker 帧求和）+ 15 组投影表（depth 输入）+ depth 投影矩阵 | 0.21 GB |
| `text-encoder/` | T5Gemma2 文本编码器：两段 NPU axmodel（U16 激活 + S8 权重，链式 cos 0.99968）+ CPU embedding LUT（bf16）+ tokenizer | 1.38 GB |
| `speech-tokenizer/` | Qwen3-TTS Tokenizer 12Hz decoder（decode_pre + 8 层 transformer + decode_post），**复用 AXERA 已编译产物** | 0.16 GB |
| `runtime/` | 板端 Python runtime（`run_breeze_tts.py`） | <1 MB |
| `demos/` | 板端合成示例（24 kHz wav） | ~1 MB |

## 板端使用

```bash
# 依赖（AX650 板端，无需 PyTorch）：pyaxengine 0.1.3 + ml-dtypes + tokenizers + numpy
pip3 install axengine-0.1.3-py3-none-any.whl --no-deps
pip3 install "ml-dtypes>=0.1.0" tokenizers numpy
export LD_LIBRARY_PATH=/soc/lib

python3 runtime/run_breeze_tts.py \
  --text "今天天气不错，我们一起出去走走吧。你觉得去哪里比较好呢？" \
  --instruction "用温柔亲切的女声，语速稍慢。" \
  --out out.wav --dump-tokens tokens.npz
```

采样模式默认（官方参数：backbone temperature 0.9 / top_k 50 / repetition_penalty 1.1；depth 同）。
`--greedy` 可切贪心（token 对拍用）；`--depth-dir` 切 quality 模式（fp32 权重版 depth，约 2.5× 慢）。

## 实测（AX650C 真板）

| 项 | 结果 |
|---|---|
| 文本编码器 | 两段串行：加载 1.58 s、编码 196 ms/次（34 token）、CMM +742 MB 后释放 | 
| talker decode | 133–146 ms/帧（28 层 INT8，kv 512） |
| depth decoder（15 步 × 12 层） | 564–758 ms/帧（INT8 fast） / 1433 ms/帧（hp quality） |
| codec decode | 0.5–1.25 s/句 |
| 整句端到端 | 5.12 s 音频 **72.2 s（RTF 14.1×）**；6.72 s 音频 **73.4 s（RTF 10.9×）**（demo 生成时 lm_head 在 CPU；U16 NPU head 后 depth 省 ~50 ms/帧，预期 RTF ~11×） |
| 资源 | CMM 3.47 GB（fast）/ 4.39 GB（hp）；python RSS ~1.1 GB |
| 精度 | talker frame0 logits 与 PyTorch 参考 top5 逐位一致；codec 同 codes 幅度谱相关 0.99958 |
| ASR 验收 | 采样模式：中文短句/长句经 paraformer-zh + SenseVoice 转写**与输入文本一致** |

## 已知限制

1. 文本编码器为 **U16 激活 + S8 权重** 混合精度（两段串行，链式 cos 0.99968）；若需要 bit 级一致，
   可用 `--text-embeds <npz>` 注入参考实现算出的 embedding（`demos/text_embeds/` 有 4 条示例）。
2. depth INT8 量化噪声在 greedy free-run 时会于 near-tie 翻转（音频仍可懂，采样模式 ASR 通过）；
   `--depth-mode hp` 可切 quality 模式。
3. Voice Clone / Voice Direction（codec encoder + speaker encoder）、CFG 未实现。

## 来源

- 原始权重：[BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)（非商业研究许可）
- 转换工程：https://github.com/yuyun2000/breeze-tts-ax650
- 参考：AXERA-TECH/Qwen3-TTS.AXERA、AXERA-TECH/Qwen3-TTS-12Hz-0.6B-Base-AX650（codec 编译产物来源）
