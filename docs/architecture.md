# Breeze TTS 2 架构与 AX650 映射

> 来源：`models/breeze.py` / `models/generation_breeze.py`（breeze-tts @ main）、HF `BreezeBlue/Breeze-TTS-2` 的
> `config.json` / `model.safetensors.index.json` / `audio_tokenizer/config.json`、`Qwen/Qwen3-TTS-12Hz-0.6B-Base`
> 的 safetensors header。所有数字来自真实权重文件，不是估算。

## 1. 总体结构

```
                    ┌──────────────────────── 文本 ────────────────────────┐
text tokens [L] ──► T5Gemma2 encoder (26L, h=1152, 双向) ──► Linear(1152→2048) ──► text embeds [L,2048]
                                                                                        │
┌─ 音频反馈 ──────────────────────────────────────────────────────────────────────────┐ │
│ frame_emb = Σ_{g=0..15} E_g[cb_g]   (E: 16×2051×2048, 求和)                        │ │
│            ▲                                                                        │ ▼
│       talker（Qwen3 式 28L, h=2048, 16 heads/8 KV, inter 6144, qk-norm,            │
│               rope θ=5e5 + llama3 scaling）                      ◄─────────────── prefill
│            │ final RMSNorm ──► lm_head(2048→2052)  → cb0（2051 = EOS）
│            ▼ hidden_norm [2048]
│       depth decoder（Llama 式 12L, h=1024, 8 heads/2 KV, inter 8192, 无 qk-norm,
│                        rope θ=5e5 + llama3 scaling orig=16）
│        输入：[proj(hidden_norm), proj(E_0[cb0]), proj(E_1[cb1]), …]（15 步自回归）
│        输出：15 个位置相关 head（codebooks_head[15,1024,2051]）→ cb1..cb15
│            ▼
│       codes [T,16] ──► Qwen3-TTS Tokenizer 12Hz decoder ──► wav 24 kHz（12.5 帧/秒）
└──────────────────────────────────────────────────────────────────────────────────────┘
```

关键常数（`config.json`）：

| 项 | 值 |
|---|---|
| 总参数 / 权重 | ~3.45 B / 6.97 GB bf16（部署可裁剪到 ~2.9 B：`embed_text_tokens` 537 M 在有 text_encoder 时不用） |
| 音频 codebook | 16 组 × 2048（合法 id 0..2047），保留 id 2048–2050 掩码，音频词表 2051 |
| backbone | 28L / h=2048 / 16 heads / 8 KV heads / head_dim 128 / inter 6144 / rms eps **1e-6** / qk-norm |
| backbone RoPE | **θ=1,000,000，无 rope_scaling（普通 RoPE）**（走嵌套 `backbone_config`，顶层 llama3 配置不生效） |
| backbone 输出 | `lm_head: 2048 → 2052`（2051 = EOS 类别） |
| depth decoder | 12L / h=1024 / 8 heads / 2 KV / head_dim 128 / inter 8192 / 无 qk-norm |
| depth RoPE | θ=500000，llama3 scaling：factor 32，orig **16**，high 0.0078125，low 0.001953125（真实生效，必须精确实现） |
| text encoder | T5Gemma2 encoder 26L / h=1152 / 4 heads × head_dim 256 / KV 1 / inter 6912 / 双向 |
| text encoder 词表 | 262158（Gemma 262144 + 14 个 added special） |
| codec | Qwen3-TTS Tokenizer 12Hz（Mimi 系）：24 kHz、12.5 fps、每帧 1920 采样点、codebook_size 2048、16 量化器 |

## 2. 权重命名（HF safetensors）

```
text_encoder.layers.{0..25}.*             # T5Gemma2
text_encoder.embed_tokens.{weight,eoi_embedding}
text_encoder.norm.weight
text_encoder_proj.weight                  # [2048,1152]，无 bias
backbone_model.layers.{0..27}.*           # 每层 11 个：q/k/v/o, q_norm/k_norm, gate/up/down, 两个 RMSNorm
backbone_model.norm.weight
backbone_model.embed_tokens.embed_audio_tokens.weight   # [16*2051, 2048]（tied 给 depth）
lm_head.weight                            # [2052, 2048]
embed_text_tokens.weight                  # [262158, 2048]（有 text_encoder 时推理不用）
depth_decoder.model.layers.{0..11}.*      # 每层 9 个（无 qk-norm）
depth_decoder.model.norm.weight
depth_decoder.model.embed_tokens.weight   # [16*2051, 2048]（与 backbone 表 tied）
depth_decoder.model.inputs_embeds_projector.weight  # [1024,2048]
depth_decoder.codebooks_head.weight       # [15, 1024, 2051]，F.linear(hidden, W[i].T)
codec_model.*                             # Qwen3TTSTokenizerV2Model（encoder/decoder/quantizer）
```

## 3. 生成流程要点（板端必须照抄）

详见 [experience.md](experience.md) 与 `_work/ref/GEN_LOGIC.md`（开发期文档）：

1. Voice Design prompt：`"[S0]<ins_bos>{instruction}<ins_eos>{text}"`，整段文本 token 全部经 T5Gemma2 编码，
   不做 CFG（默认 `cfg_scale=1.0`），不追加音频占位。
2. 文本 embedding 注满 prefill 序列（位置 0..L-1），第 k 帧占位置 L-1+k（RoPE 连续）。
3. 每帧：backbone post-norm hidden → `lm_head` 得 2052 logits → 重复惩罚（仅历史 cb0，1.1）→
   `temperature 0.9` → `top_k 50` → **再**把 2048–2050 掩 `-inf` → softmax/multinomial 得 cb0；cb0==2051 即 EOS（追加一帧全 2050 后停止）。
4. depth decoder 固定 15 步：输入位置 0 = `proj(hidden_norm)`，位置 p = `proj(E_{p-1}[cb_{p-1}])`（`proj` = inputs_embeds_projector）；
   第 p 个位置相关 head 输出 cb_p（`W[p-1]`）；每步温度 0.9 / top_k 50 / 掩码 2048–2050。
5. 下一帧 talker 输入 = `Σ_{g=0..15} E_g[cb_g]`（16 组**求和**，2048 维；E_0..E_15 全部未投影表）。
6. 有效音频帧 = EOS 帧之前 / 全 pad(2050) 帧丢弃；codec 解码得 24 kHz wav。

## 4. AX650 映射

| 原始组件 | AX650 上跑什么 | 转换方式 |
|---|---|---|
| T5Gemma2 encoder + proj | 1 个静态 axmodel（输入 embeddings [1,128,1152] 或 token ids，输出 [1,128,2048]） | ONNX → `pulsar2 build`（INT8，真实文本校准） |
| talker 28 层 + post | 28 个 per-layer axmodel + 1 个 post（norm + lm_head） | 重打包成 Qwen3-TTS talker 目录 → `pulsar2 llm_build --model_type qwen3_tts_talker`（bf16 hidden state，KV cache 外置） |
| depth decoder 12 层 + norm | 12 个 per-layer axmodel + 1 个 post（norm） | 重打包成 code_predictor 目录 → `pulsar2 llm_build --model_type qwen3_tts_talker_code_predictor`（无 qk-norm 的差异见 experience） |
| codebooks_head 15 head | 15 个 [2051×1024] 线性层 axmodel | ONNX → `pulsar2 build` |
| 音频 embedding 表 | CPU（bf16 bin，mmap）：16×2051×2048（未投影）+ 15×2051×1024（投影后） | 重打包时导出 bin |
| speech tokenizer | 1× decode_pre_inproj + 8× decoder transformer layer + post + decode_post | **直接复用 AXERA 已编译 Qwen3-TTS axmodel**（Breeze 的 `audio_tokenizer/model.safetensors` 与 `Qwen/Qwen3-TTS-Tokenizer-12Hz` 字节一致，sha256/LFS oid 相同） |
| 采样/调度/分词 | 板端 Python（numpy + pyaxengine + tokenizers） | 自写 runtime |
