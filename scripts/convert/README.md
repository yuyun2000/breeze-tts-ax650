# 转换流水线（在 Linux x86_64 转换机上执行）

前置：Pulsar2 8.0、Python 环境（torch CPU、transformers==4.57.3、qwen-tts==0.1.1、onnx、onnxsim）。

顺序：

| 步骤 | 脚本 | 产物 |
|---|---|---|
| 1. 下载模型 | `hf download BreezeBlue/Breeze-TTS-2` | 6.97 GB 权重 + audio_tokenizer |
| 2. 重打包 | `repack/repack_talker.py`、`repack/repack_codepredictor.py` | Qwen3-TTS 风格 checkpoint（talker/code predictor）+ 板端 embedding bin |
| 3. 数值对拍 | `repack/verify_repack.py` | 逐层 cosine / 前 N 帧 token 一致率 |
| 4. 编译 talker | `build/build_talker.sh` | 28 层 axmodel + post（norm+lm_head）+ output_norm |
| 5. 编译 depth decoder | `build/build_codepredictor.sh` | 12 层 axmodel + post(norm) + 15 lm_head axmodel |
| 6. 文本编码器 | `text_encoder/export_text_encoder_onnx.py` + `text_encoder/build_text_encoder.sh` | 1 个静态 axmodel |
| 7. codec | 直接复用 AXERA `Qwen3-TTS-12Hz-0.6B-Base-AX650` HF 包里的 `speech_tokenizer/*`（Breeze audio_tokenizer 与其字节一致） | decoder axmodel ×10 |
| 8. 组装 | `assemble_package.py` | 可上传 HF 的模型包目录 |

为什么重打包：Pulsar2 没有 Breeze 这个自定义架构的 `model_type`，但它有官方 Qwen3-TTS 支持
（`qwen3_tts_talker` / `qwen3_tts_talker_code_predictor` / `qwen3_tts_tokenizer_12hz`）。
Breeze 的 talker 是 Qwen3 式 28 层（与 Qwen3-1.7B 完全同形），depth decoder 是 Llama 式 12 层，
codec 与 Qwen3-TTS tokenizer 字节相同 —— 因此把权重按 Qwen3-TTS 的命名/配置重排即可复用官方流水线。
细节见 [docs/architecture.md](../../docs/architecture.md) 与 [docs/conversion.md](../../docs/conversion.md)。
