# 转换流水线（Breeze TTS 2 → AX650 axmodel）

> 状态：进行中（2026-11-11 夜）。本文件记录可复现步骤与关键决策；最终数字以 `reports/conversion.md` 为准。

## 0. 机器

- 转换机：Linux x86_64（16 核/62 GB），Pulsar2 8.0 独立包，Python 3.10 venv（torch 2.9.1+cpu、transformers 4.57.3、qwen-tts 0.1.1、onnx）。
- 板卡：AX650C（`AX650/AX8850` 同包兼容），pyaxengine 0.1.3 + ml-dtypes（bf16 必须）。

## 1. 关键结论（决定流水线）

1. **codec 直接复用**：Breeze 的 `audio_tokenizer/model.safetensors` 与 `Qwen/Qwen3-TTS-Tokenizer-12Hz` **字节一致**
   （sha256 `836b7b35…571258`）→ 直接使用 AXERA 已编译的 Qwen3-TTS speech_tokenizer axmodel
   （`AXERA-TECH/Qwen3-TTS-12Hz-0.6B-Base-AX650`），**不做任何 codec 转换**。
2. **talker 用 pulsar2 `llm_build --model_type qwen3`**：Breeze backbone 的 `backbone_model_type="qwen3"`，
   走 `BreezeBackboneAdapter`，实际用嵌套 `backbone_config`：Qwen3 / hidden 2048 / 28L / 16 heads / 8 KV /
   head_dim 128 / inter 6144 / rms eps 1e-6 / **rope_theta=1e6, rope_scaling=null（普通 RoPE）**。
   （顶层 config 的 llama3 rope 是死配置，实测不生效。）
3. **depth decoder 用 pulsar2 `llm_build --model_type llama`**：Breeze depth 是 Llama 结构（无 qk-norm、
   SwiGLU、8 heads / 2 KV、head_dim 128、inter 8192），且 **llama3 rope 真实生效**
   （theta 5e5、orig 16、factor 32、high 0.0078125、low 0.001953125）。llama builder 的 Config 打印实测
   `rs_rope_type='llama3'` 等字段与 config 一致 → rope 精确（不是近似）。
4. **通用 LLM build 的运行时契约**：per-layer axmodel 的 prefill/decode `input` 是 **bf16 embeddings（不是 token id）**
   （ax-llm 的 `LLaMaEmbedSelector` 在 CPU 查表后灌入）→ 与 Breeze 的"文本编码器输出 2048-d embedding /
   16 组音频表求和"完全契合，所有 embedding 逻辑放在板端 CPU（numpy）。
5. **文本编码器**：T5Gemma2（26L, hidden 1152, 双向）导出**两段静态 ONNX**（各 ~1.4 GB，规避 protobuf 2GB 上限）
   → pulsar2 build INT8：段1 输入 inputs_embeds[1,128,1152]+attention_mask[1,128] → hidden[1,128,1152]；
   段2 接 final norm + text_encoder_proj(1152→2048) → text_embeds[1,128,2048]。

## 2. 重打包（scripts/convert/repack/repack_breeze.py）

从 Breeze 原始 safetensors 直接重排权重（不加载模型、纯 bf16 搬运，44 秒完成）：

| 产物 | 内容 |
|---|---|
| `qwen3-talker/` | Qwen3ForCausalLM 目录：`model.layers.{0..27}.*` ← `backbone_model.layers.*`；`model.norm` ← `backbone_model.norm`；`lm_head` ← Breeze `lm_head`（[2052,2048]，2052=2051 audio+1 EOS）；embed_tokens 为 dummy |
| `llama-depth/` | LlamaForCausalLM 目录：`model.layers.{0..11}.*` ← `depth_decoder.model.layers.*`；`model.norm` ← depth norm；embed/lm_head 为 dummy（运行时不用） |
| `embeds/audio_embed_16x2051x2048.bfloat16.bin` | 16 组未投影音频 embedding（talker 帧 embedding = Σ 求和，134 MB） |
| `embeds/proj_audio_embed_16x2051x1024.bfloat16.bin` | `E' = E @ proj.T`（depth 输入用：E'_0 → first_code_embed；E'_1..E'_14 → 后续 14 步） |
| `embeds/depth_proj_2048x1024.bfloat16.bin` | `inputs_embeds_projector.weight` [1024,2048]（depth 位置 0 = h @ W.T） |
| `lm_head_onnx/code_predictor_lm_head_{0..14}.onnx` | 位置相关 15 个 head（输入 [1,1,1024] fp32，MatMul 权重 [1024,2051]，输出 [1,1,2051]） |

**数值验证**（同一随机输入，原始 Breeze 模块 vs 重打包目录加载的 HF 模型）：
talker hidden cosine **1.0000036**、logits cosine **1.0000031**、depth hidden cosine **1.0000010**、max_abs_diff **0**（bf16 完全一致）。

## 3. Pulsar2 编译

```bash
# talker（28 层 + post；kv 512 帧 ≈41s 音频；prefill 128×5 组，容量 640 token）
pulsar2 llm_build --model_type qwen3 --input_path repack/qwen3-talker \
  --output_path builds/talker --hidden_state_type bf16 \
  --kv_cache_len 512 --prefill_len 128 \
  --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512 \
  --ret_postnorm --chip AX650 --parallel 10

# depth decoder（12 层 + post；一帧内序列 ≤17）
pulsar2 llm_build --model_type llama --input_path repack/llama-depth \
  --output_path builds/code-predictor --hidden_state_type bf16 \
  --kv_cache_len 32 --prefill_len 16 --ret_postnorm --chip AX650 --parallel 4

# 15 个位置相关 head（静态 build，MatMul 常量权重，s8）
for i in 0..14; pulsar2 build --input_path lm_head_onnx/code_predictor_lm_head_$i.onnx ...

# 文本编码器两段（静态图 + 真实文本校准，INT8）
pulsar2 build --input_path textenc_L128_c1of2.onnx ...
pulsar2 build --input_path textenc_L128_c2of2.onnx ...
```

`--ret_postnorm` 是必须的：talker post 需要同时输出 logits（2052）与 `output_norm`（2048，给 depth 位置 0）；
depth post 只需要 `output_norm`（1024，交给 15 个 lm_head）。

## 4. 踩坑与修正（摘要）

- Pulsar2 8.0 对 **Qwen3-TTS 有官方 model_type**（`qwen3_tts_talker` / `qwen3_tts_talker_code_predictor`），
  但其 talker 走 **mRoPE**（3 轴 indices [3,P]）、code_predictor 强制 qk-norm，与 Breeze 的
  (普通 RoPE / 无 qk-norm) 不匹配 → 改用通用 `qwen3` / `llama` model_type，反而精确且配置简单。
- Breeze **顶层 rope 配置是死配置**（talker 实际 default RoPE）——详见 [experience.md](experience.md)。
- 转换机 CPU 是 QEMU 虚拟 CPU（无 AVX），PyTorch 参考推理极慢 → 参考生成/对拍在 Apple M4 Pro CPU 上做；
  Pulsar2 编译在 x86 上正常。
