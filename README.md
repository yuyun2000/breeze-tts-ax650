# Breeze TTS 2 on AX650

把 [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)（约 3.45 B 参数的中英双语 TTS，
榜单第一的开源权重 TTS）转换并部署到 **AX650/AX8850** NPU 的可复现工程。

> ⚠️ **状态：进行中（2026-11-11 夜）**。本 README 会随转换/上板结果更新；数字以 `reports/` 与 `docs/` 里的实测报告为准。

> ⚠️ **许可证**：Breeze TTS 2 的**模型权重、衍生模型与自托管输出仅限研究与非商业用途**
> （BreezeBlue Research and Non-Commercial License 1.1，见 [licenses/BREEZEBLUE_LICENSE.txt](licenses/BREEZEBLUE_LICENSE.txt)）。
> 本仓库的**代码**为 Apache-2.0（见 [licenses/APACHE-2.0.txt](licenses/APACHE-2.0.txt)）；
> 本仓库发布的 axmodel 转换产物属于 Breeze 权重的**衍生模型**，同样受 BreezeBlue 非商业许可证约束，**不可商用**。
> Qwen3-TTS tokenizer 相关组件来自 [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（Apache-2.0）。
> AXERA 的工具链/现成编译产物版权归 AXERA，引用时注明来源。

---

## 1. 这是什么

Breeze TTS 2 由四个部分组成，本工程把其中三部分搬上 AX650 NPU，并保留一个纯 Python 运行时：

| 组件 | 原始结构 | 本工程做法 | 状态 |
|---|---|---|---|
| 文本编码器 | T5Gemma2 encoder（26 层，hidden 1152，双向） | 静态 ONNX → `pulsar2 build` INT8 axmodel | 待填 |
| Backbone（talker） | Qwen3 式 28 层，hidden 2048，预测 codebook0 | 重打包成 Qwen3-TTS talker → `pulsar2 llm_build` 逐层 axmodel | 待填 |
| Depth decoder | Llama 式 12 层，hidden 1024，预测 codebook1..15 | 重打包 → `pulsar2 llm_build` 逐层 axmodel + 15 个位置相关 lm_head | 待填 |
| 音频 codec（vocoder） | Qwen3-TTS Tokenizer 12Hz（Mimi 系），24 kHz | **直接复用** AXERA 已编译的 Qwen3-TTS speech_tokenizer axmodel（权重与 Breeze 字节一致） | 待填 |

板端运行时不依赖 PyTorch：`numpy + pyaxengine + tokenizers`，在 AX650（1.9 GB 系统内存）上跑通
Voice Design（自然语言描述生成音色，无需参考音频）模式。

```
text: "[S0]<ins_bos>{voice description}<ins_eos>{target text}"
  → T5Gemma2 text encoder (NPU) → 2048-d embeddings
  → talker prefill (NPU, 28 layers, chunked)
  → autoregressive loop:
       cb0 = sample(talker logits[2052])          # NPU post layer + CPU sample
       cb1..cb15 = depth decoder 15 steps          # NPU, 12 layers + 15 lm_head
       frame_emb = Σ_{g=0..15} E_g[cb_g]           # CPU embedding lookup
       talker decode one token (NPU)
  → 16×T codebook tokens
  → speech_tokenizer decoder (NPU: decode_pre → 8 transformer layers → decode_post)
  → 24 kHz wav
```

## 2. 目录

```
scripts/convert/   转换脚本（在 Linux x86 转换机上跑）
  text_encoder/    文本编码器 ONNX 导出 + Pulsar2 编译 + 校准
  repack/          Breeze → Qwen3-TTS 结构重打包（权重映射 + 数值对拍）
  build/           talker / code-predictor 的 pulsar2 llm_build 脚本
scripts/board/     板端运行时（Python + pyaxengine）
docs/              架构、转换、部署、性能、经验文档
reports/           实测报告与日志摘要
models/            模型包目录结构说明（权重本身在 HF 上）
```

## 3. 快速开始

> 模型文件较大（约 2.5–3 GB），存放在 HuggingFace：**[yunyu1258/breeze-tts-2-ax650](https://huggingface.co/yunyu1258/breeze-tts-2-ax650)**（待上传）。

```bash
# 1) 下载模型包
export HF_ENDPOINT=https://hf-mirror.com
hf download yunyu1258/breeze-tts-2-ax650 --local-dir ./models/breeze-tts-2-ax650

# 2) 板端跑（AX650，需先装 pyaxengine）
cd models/breeze-tts-2-ax650
python3 runtime/run_breeze_tts.py \
  --text "欢迎来到今晚的故事时间，让我们一起开始吧。" \
  --instruction "一位温柔自信的年轻女性，声音清晰，语气亲切。" \
  --out output.wav
```

## 4. 实测（AX650 真板，待补）

| 项 | 结果 |
|---|---|
| talker 单帧 decode | 待填 |
| depth decoder 一帧 15 步 | 待填 |
| speech tokenizer 解码 | 待填 |
| 整句 RTF / 首帧延迟 | 待填 |
| 峰值 CMM / 系统内存 | 待填 |
| 音质对拍（vs PyTorch 参考） | 待填 |

## 5. 文档

- [docs/architecture.md](docs/architecture.md)：Breeze TTS 2 架构与 AX650 映射（各张量形状/接口）
- [docs/conversion.md](docs/conversion.md)：转换流水线（重打包、pulsar2、复用 codec）
- [docs/board-deploy.md](docs/board-deploy.md)：板端环境、部署、运行
- [docs/performance.md](docs/performance.md)：性能与精度对拍
- [docs/experience.md](docs/experience.md)：踩坑与经验（llm_build model_type 适配、rope scaling、weight-only 量化等）

## 6. 已知限制

- 第一版只支持 **Voice Design**（无参考音频）；Voice Clone / Voice Direction 需要 codec encoder + speaker encoder，暂未上板。
- 默认 `cfg_scale=1.0`（无 CFG）。CFG=4 需要双分支前向（2× 计算量），后续可加。
- **非商业**：见上方许可证说明。
