# 转换报告（实测）

> 2026-11-11 夜。转换机：Linux x86_64 16C/62G（QEMU CPU 无 AVX），Pulsar2 8.0；对拍在 Apple M4 Pro CPU。

## 1. 重打包（权重搬运，44 s）

| 产物 | 内容 | 体积 |
|---|---|---|
| qwen3-talker/model.safetensors | Qwen3ForCausalLM：28 层 + norm + lm_head[2052,2048] | 2.84 GB bf16 |
| llama-depth/model.safetensors | LlamaForCausalLM：12 层 + norm + dummy embed/lm_head | 0.68 GB bf16 |
| embeds/audio_embed_16x2051x2048.bfloat16.bin | 16 组未投影音频 embedding 表 | 134.4 MB |
| embeds/proj_audio_embed_16x2051x1024.bfloat16.bin | E' = E @ projᵀ（depth 输入用，E'_0..E'_14） | 67.1 MB |
| embeds/depth_proj_2048x1024.bfloat16.bin | inputs_embeds_projector.weight | 4.2 MB |
| lm_head_onnx/code_predictor_lm_head_{0..14}.onnx | 15 个位置相关 head | 15 × 8.4 MB fp32 |

数值对拍（原始 Breeze 模块 vs 重打包 HF 模型，同一输入）：
talker 逐层 hidden cos **1.00000000**（f32）、bf16 复跑 1.0000036；depth 16 步 hidden cos **1.0000010**；
lm_head logits cos 0.9999998。**重打包是精确等价映射，不是近似。**

## 2. Pulsar2 编译

| 产物 | 参数 | 结果 |
|---|---|---|
| talker（28 层 + post） | `llm_build --model_type qwen3`，bf16 hidden，kv 512，prefill 128×5 组，`--ret_postnorm`，`--parallel 10` | 24 分钟；28×~58 MB + post 4.6 MB = **1.633 GB**；post 输出 `output` BF16[1,1,2052] + `output_norm` BF16[1,1,2048] |
| depth（12 层 + post） | `llm_build --model_type llama`，kv 32，prefill 16，`--ret_postnorm`，`--parallel 6` | 5 分钟；12×30.7 MB + post 74 KB = 369 MB |
| 15 lm_head（INT8 初版） | `pulsar2 build`（INT8，随机 hidden 校准；见下"踩坑"） | 15×2.2 MB = 33 MB（**输出 clamp 不可用**） |
| 15 lm_head（U16 终版） | U16 激活 + 真实 per-head 校准（golden teacher-forced hidden） | 15×2.28 MB = 33 MB；unique 1662–1917、cos≥0.99996、top1 与 CPU fp32 一致 |
| depth quality 档 | `llm_build --model_type llama -w bf16 --post_weight_type bf16`（FP32-heavy） | 12×111 MB + post 163 KB = 1.3 GB |
| 文本编码器 | 两段静态 ONNX（T5Gemma2 26 层，L=128），INT8 → **不达标**（cos 0.45）→ U16/混合精度重建 | 见 textenc 报告 |

## 3. 踩坑

1. **文本编码器 INT8 不达标**：Pulsar2 precision_analysis 显示 MLP 中间激活 (1,128,6912) cos 0.07–0.19、
   多处 RMSNorm/残差 Add cos 0.6–0.88（动态范围过大）→ 最终 cosine 0.41–0.55。ONNX 本身与 PyTorch
   cosine 1.000000、与 Mac golden 0.99993，排除导出/掩码问题。改用 U16/混合精度重建（结果见 `reports/board.md`）。
2. **15 个 lm_head INT8 输出 clamp**：板端实测 axmodel 输出 2051 个 logits 只有 ~150 个不同值，
   argmax 直接错（CPU fp32 matmul 则 cos 0.9998）→ 重编为 U16 输出/不量化输出。
3. talker/depth 的模型结构判断：见 [experience.md](../docs/experience.md)（顶层 rope 配置是死配置、
   通用 qwen3/llama builder 才是精确路径）。

## 4. 产物清单（交付）

`talker/` 29 文件 1.633 GB；`code-predictor/` 28 文件 401 MB（12 层 + post + 15 head）；
`embeds/` 3 文件 205.7 MB；`speech-tokenizer/` 11 文件 156 MB（复用 AXERA 编译产物）；
`text-encoder/`（U16 版本，见 textenc 报告）。全部 sha256 清单见 `models/SHA256SUMS`。
